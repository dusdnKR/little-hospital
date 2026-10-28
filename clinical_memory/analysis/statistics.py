from collections import Counter
import json
from statistics import mean
from clinical_memory.environment.events import event_exams
from clinical_memory.analysis.paired import correct, saw_image
from clinical_memory.analysis.transitions import changes

BINS = (
    (1, "same complaint"),
    (0.5, "half of the complaint or more"),
    (0.01, "less than half"),
    (0, "no shared word"),
)


def point(rows):
    """Counts of one set of visits, and the two rates that the growth measure of the earlier studies reads."""
    n, reached = (
        len(rows),
        [
            r["metrics"]["requests_until_examination"]
            for r in rows
            if r["metrics"]["requests_until_examination"]
        ],
    )
    out = {
        "n": n,
        "correct": sum(map(correct, rows)),
        "image_reached": sum(map(saw_image, rows)),
        "examination_reached": sum(
            (any((event_exams(e) for e in r["trace"]["events"])) for r in rows)
        ),
        "no_exam_request": sum((r["metrics"]["exam_requests"] == 0 for r in rows)),
        "records_received": sum((r["metrics"]["records_received"] for r in rows)),
        "collection_actions": sum((r["metrics"]["collection_actions"] for r in rows)),
        "wasted_actions": sum((r["metrics"]["wasted_actions"] for r in rows)),
        "requests_until_examination_mean": round(mean(reached), 3) if reached else None,
        "valid_final": sum((r["summary"]["valid_final"] for r in rows)),
        "presentation_named": sum((r["summary"]["presentation_named"] for r in rows)),
        "presentation_settled": sum(
            (r["summary"]["presentation_settled"] for r in rows)
        ),
        "with_lessons": sum((r["summary"]["gathering_lessons"] > 0 for r in rows)),
        "with_notes": sum((r["summary"]["diagnosis_notes"] > 0 for r in rows)),
        "tokens": sum(
            (
                r["all_calls_cost"]["input_tokens"]
                + r["all_calls_cost"]["output_tokens"]
                for r in rows
            )
        ),
    }
    out.update(accuracy=out["correct"] / n, image_reach=out["image_reached"] / n)
    return out


def slim(record):
    """What the summaries read of one visit. Runs are compared after all of them are read, so only this is kept
    of each record; the journal of one run at a time is all that stays in memory."""
    return {
        "trace": {
            "id": record["trace"]["id"],
            "events": [
                {"exams": [{}]} for e in record["trace"]["events"] if event_exams(e)
            ],
        },
        "score": {"correct": record["score"]["correct"]},
        "metrics": {
            key: record["metrics"][key]
            for key in (
                "final_answer_saw_image",
                "exam_requests",
                "wasted_actions",
                "collection_actions",
            )
        },
    }


def closest(record):
    """How close the nearest lesson of the notebook came to the recall rule for this patient."""
    if not record["summary"]["presentation_named"]:
        return "presentation not named"
    top = record["trace"]["retrievals"][0]["candidates"][:1]
    return (
        next((label for low, label in BINS if top[0]["complaint_match"] >= low))
        if top
        else "empty notebook"
    )


def lesson_calls(folder):
    """How the calls that write a lesson ended in one run, read from its call log. Counts only.

    A reply that was cut off at the output limit cannot be a whole JSON object, so the number of such calls tells a
    lesson lost to that limit from a lesson lost to what the doctor wrote."""
    found = {"lesson_calls": 0, "rewrite_calls": 0, "token_cap_hits": 0}
    with (folder / "private_calls.jsonl").open(encoding="utf-8") as lines:
        for line in lines:
            if '"reflection' not in line:
                continue
            row = json.loads(line)
            if row["role"] in {"reflection", "reflection_repair"}:
                found["lesson_calls"] += 1
                found["rewrite_calls"] += row["role"] == "reflection_repair"
                found["token_cap_hits"] += bool(row["hit_token_cap"])
    return found


def reached(start, final):
    """Whether lessons reached the evaluation patients and what changed in the visits that read one: the same
    patients with an empty notebook and with the notebook of the last checkpoint."""
    before = {r["trace"]["id"]: r for r in start}
    pairs = [(before[r["trace"]["id"]], r) for r in final]
    got = [(b, a) for b, a in pairs if a["summary"]["gathering_lessons"] > 0]
    rest = [(b, a) for b, a in pairs if a["summary"]["gathering_lessons"] == 0]
    return {
        "delivered": {
            "evaluation_visits": len(pairs),
            "presentation_named": sum(
                (a["summary"]["presentation_named"] for _, a in pairs)
            ),
            "with_gathering_lesson": len(got),
            "with_diagnosis_note": sum(
                (a["summary"]["diagnosis_notes"] > 0 for _, a in pairs)
            ),
            **{
                "closest_" + k: v
                for k, v in sorted(Counter((closest(a) for _, a in pairs)).items())
            },
        },
        "used_with_lesson": changes(got),
        "used_without_lesson": changes(rest),
    }
