import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from clinical_memory.memory.store import atomic_json
from clinical_memory.analysis.paired import journal
from clinical_memory.analysis.transitions import changes, combine
from clinical_memory.analysis.statistics import closest, lesson_calls, point, slim
from clinical_memory.analysis.aggregate import summarize
from clinical_memory.evaluation.audit import unusable
from clinical_memory.data.augmented import load_set
from clinical_memory.config import ROOT, RUNS, SETS, load, sha, inputs, code_hashes


def written(training, folder):
    """What became of the lessons the doctor was asked to write after its training patients."""
    stored = [r["lesson"] for r in training if r["lesson"]]
    kept = [r["reflection"] for r in training if "settled" in r["reflection"]]
    return {
        "written": {
            "training_patients": len(training),
            "presentation_named": sum(
                (r["summary"]["presentation_named"] for r in training)
            ),
            "presentation_settled": sum(
                (r["summary"]["presentation_settled"] for r in training)
            ),
            "lessons_stored": len(stored),
            "with_gathering_part": sum((bool(x["collect"]) for x in stored)),
            "with_diagnosis_part": sum((bool(x["diagnose"]) for x in stored)),
            "gathering_parts_over_45_words": sum(
                (len(x["collect"].split()) > 45 for x in stored)
            ),
            "diagnosis_parts_over_45_words": sum(
                (len(x["diagnose"].split()) > 45 for x in stored)
            ),
            **{
                "reflection_" + k: v
                for k, v in sorted(
                    Counter((r["reflection"]["status"] for r in training)).items()
                )
            },
            "stored_after_rewrite": sum(
                (
                    r["reflection"]["status"] == "stored"
                    and r["reflection"]["repairs"] == 1
                    and ("settled" not in r["reflection"])
                    for r in training
                )
            ),
            "kept_in_usable_form": len(kept),
            "rules_broken_by_kept_lessons": dict(
                sorted(Counter((e for k in kept for e in k["errors"])).items())
            ),
            "changes_to_kept_lessons": dict(
                sorted(Counter((c for k in kept for c in k["settled"])).items())
            ),
            "format_errors": dict(
                sorted(
                    Counter(
                        (
                            e
                            for r in training
                            if r["reflection"]["status"] == "invalid"
                            for e in r["reflection"]["errors"]
                        )
                    ).items()
                )
            ),
        },
        "calls": lesson_calls(folder),
    }


def noted(record, gold):
    """What the notes of one visit were about: the right option, wrong options only, or no note at all."""
    found = (record["trace"]["note_lookup"] or {}).get("selected", [])
    if not found:
        return "none"
    return (
        "right" if any((gold["label"] in s["options"] for s in found)) else "wrong_only"
    )


def reached(start, final, golds):
    """Which visits of a test set read something of the notebook, and what changed in them against the same patient
    seen with an empty notebook."""
    before = {r["trace"]["id"]: r for r in start}
    pairs = [(before[r["trace"]["id"]], r) for r in final]
    kinds = {r["trace"]["id"]: noted(r, golds[r["trace"]["id"]]) for r in final}
    read = [
        (b, a)
        for b, a in pairs
        if a["summary"]["gathering_lessons"] > 0 or a["summary"]["diagnosis_notes"] > 0
    ]
    rest = [
        (b, a)
        for b, a in pairs
        if not (
            a["summary"]["gathering_lessons"] > 0 or a["summary"]["diagnosis_notes"] > 0
        )
    ]

    def of(kind):
        return [(b, a) for b, a in pairs if kinds[a["trace"]["id"]] == kind]

    return {
        "delivered": {
            "evaluation_visits": len(pairs),
            "presentation_named": sum(
                (a["summary"]["presentation_named"] for _, a in pairs)
            ),
            "with_gathering_lesson": sum(
                (a["summary"]["gathering_lessons"] > 0 for _, a in pairs)
            ),
            "with_diagnosis_note": sum(
                (a["summary"]["diagnosis_notes"] > 0 for _, a in pairs)
            ),
            "note_on_the_right_option": len(of("right")),
            "notes_on_wrong_options_only": len(of("wrong_only")),
            "notes_read": sum((a["summary"]["diagnosis_notes"] for _, a in pairs)),
            **{
                "closest_" + k: v
                for k, v in sorted(Counter((closest(a) for _, a in pairs)).items())
            },
        },
        "used_with_lesson": changes(read),
        "used_without_lesson": changes(rest),
        "used_with_note_on_the_right_option": changes(of("right")),
        "used_with_notes_on_wrong_options_only": changes(of("wrong_only")),
        "used_without_note": changes(of("none")),
    }


def usable(root):
    """Completed, audited runs sealed with the current code; everything else with a reason."""
    runs, excluded, hashes = ([], [], code_hashes(root))
    for folder in sorted(
        (p for p in (root / RUNS).glob("*") if (p / "seal.json").exists())
    ):
        try:
            seal, summary, audit = (
                load(folder / "seal.json"),
                load(folder / "summary.json"),
                load(folder / "audit.json"),
            )
            if (
                summary["status"] != "completed"
                or not audit["passed"]
                or audit["seal_sha256"] != sha(folder / "seal.json")
            ):
                raise ValueError("not completed or replay audit not passed")
            if seal["code"] != hashes:
                raise ValueError("code seal differs from the current code")
            runs.append(
                {"folder": folder, "seal": seal, "summary": summary, "audit": audit}
            )
        except (OSError, ValueError, KeyError) as error:
            excluded.append({"run_id": folder.name, "reason": str(error)})
    return (runs, excluded)


def evaluated(tested, empty_notebook, golds, protocol):
    """One test set: every evaluation, and the patients before and after training for each finished training run.
    A checkpoint in between joins the curve once every one of these runs has been evaluated there."""
    steps = protocol["main"]["checkpoints"]
    finals = [
        key
        for key in tested
        if key[1] == steps[-1] and (key[0], 0, None, None, None) in tested
    ]
    between = [
        s
        for s in steps[1:-1]
        if finals and all(((d, s, w, o, x) in tested for d, _, w, o, x in finals))
    ]
    shown, curves, reach = ([0, *between, steps[-1]], [], {})
    for doctor, _, way, order, seed in finals:
        chain = [
            tested[doctor, 0, None, None, None],
            *[tested[doctor, s, way, order, seed] for s in between],
            tested[doctor, steps[-1], way, order, seed],
        ]
        item = {
            "doctor": doctor,
            "learning": way,
            "curriculum": order,
            "seed": seed,
            "curve": [dict(step=s, **found["point"]) for s, found in zip(shown, chain)],
        }
        curves.append((item, chain[0]["rows"], chain[-1]["rows"]))
        reach.setdefault((doctor, way), []).append(
            reached(empty_notebook[doctor], journal(chain[-1]["folder"]), golds)
        )
    out = summarize(curves, protocol, protocol["curricula"])
    out.update(
        checkpoints=shown,
        waiting_for_the_empty_notebook=sum((key[1] == steps[-1] for key in tested))
        - len(finals),
        evaluations_between={
            str(s): sum((key[1] == s for key in tested)) for s in steps[1:-1]
        },
        lessons=[
            {
                "doctor": d,
                "learning": w,
                "runs": len(reach[d, w]),
                **combine(reach[d, w]),
            }
            for d in protocol["doctors"]
            for w in protocol["learning"]["conditions"]
            if (d, w) in reach
        ],
    )
    return out


def analyze(root=None):
    root = Path(root or ROOT)
    inp = inputs(root)
    protocol, division = (inp.protocol, inp.division)
    runs, excluded = usable(root)
    made = root / "results/augmentation.json"
    report = {
        "schema": 1,
        "version": protocol["version"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha(root / "configs/experiment.json"),
        "inference_calls": 0,
        "excluded_runs": excluded,
        "division": {
            "patients": division["patients"],
            "train": len(division["train"]),
            "test1": len(division["test1"]),
            "test1_with_a_findable_note": len(division["test1_reachable"]),
            "test2": len(division["test2_sources"]),
            "sha256": protocol["pool"]["division_sha256"],
        },
        "augmentation": load(made) if made.exists() else None,
        "smoke": [],
        "training": {"runs": []},
        **{which: {"runs": []} for which in SETS},
    }
    tested, empty_notebook = (
        {which: {} for which in SETS},
        {which: {} for which in SETS},
    )
    for run in runs:
        seal, name, records = (run["seal"], run["folder"].name, journal(run["folder"]))
        base = {
            "run_id": name,
            "doctor": seal["doctor"],
            "learning": seal["learning"],
            "curriculum": seal["curriculum"],
            "seed": seal["seed"],
            "episodes": len(records),
            "calls_replayed": run["audit"]["calls_replayed"],
            "model_seconds": round(run["summary"]["all_call_cost"]["total_seconds"], 1),
            "source_commit": run["summary"]["source_commit"],
        }
        if seal["mode"] == "test":
            which, step = (seal["set"], seal["memory_step"])
            found = {
                "point": point(records),
                "rows": [slim(r) for r in records],
                "folder": run["folder"],
            }
            report[which]["runs"].append(
                {
                    **base,
                    "step": step,
                    "training_run": seal["training_run"],
                    **found["point"],
                }
            )
            if step == 0:
                (
                    tested[which][seal["doctor"], 0, None, None, None],
                    empty_notebook[which][seal["doctor"]],
                ) = (found, records)
            else:
                tested[which][
                    seal["doctor"],
                    step,
                    seal["learning"],
                    seal["curriculum"],
                    seal["seed"],
                ] = found
            continue
        training = [r for r in records if r["phase"] == "evolution"]
        item = {
            **base,
            "training": point(training),
            "memory_items": run["summary"]["memory_items"],
            "unusable_outputs": unusable(records),
            "lessons": written(training, run["folder"]),
            "main_basis": load(run["folder"] / "provenance.json").get("main_basis"),
        }
        if seal["mode"] == "smoke":
            last = [r for r in records if r["phase"] not in {"evolution", "test1_000"}]
            report["smoke"].append(
                {
                    **base,
                    "lessons_stored": item["lessons"]["written"]["lessons_stored"],
                    "kept_in_usable_form": item["lessons"]["written"][
                        "kept_in_usable_form"
                    ],
                    "presentations_unusable": item["unusable_outputs"][
                        "presentations_unusable"
                    ],
                    "presentations_settled": item["unusable_outputs"][
                        "presentations_settled"
                    ],
                    "test_visits_with_a_note": sum(
                        (r["summary"]["diagnosis_notes"] > 0 for r in last)
                    ),
                    "test_visits": len(last),
                }
            )
        else:
            report["training"]["runs"].append(item)
    done = {(x["doctor"], x["learning"]) for x in report["training"]["runs"]}
    report["training"]["lessons"] = [
        {
            "doctor": d,
            "learning": w,
            "runs": len(group),
            **combine([x["lessons"] for x in group]),
        }
        for d, w, group in (
            (
                d,
                w,
                [
                    x
                    for x in report["training"]["runs"]
                    if (x["doctor"], x["learning"]) == (d, w)
                ],
            )
            for d in protocol["doctors"]
            for w in protocol["learning"]["conditions"]
            if (d, w) in done
        )
    ]
    golds = {
        "test1": inp.golds,
        "test2": load_set(root, inp)[1] if tested["test2"] else {},
    }
    for which in SETS:
        report[which].update(
            evaluated(tested[which], empty_notebook[which], golds[which], protocol)
        )
    atomic_json(root / "results/summary.json", report)
    return report


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    from clinical_memory.preflight import compute_only

    compute_only(ROOT)
    result = analyze()
    print(
        json.dumps(
            {
                "smoke": len(result["smoke"]),
                "training": len(result["training"]["runs"]),
                **{which: len(result[which]["runs"]) for which in SETS},
                "excluded": len(result["excluded_runs"]),
            }
        )
    )
