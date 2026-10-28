from clinical_memory.prompts.reasoning import REASONING
from collections import Counter
from copy import deepcopy
import time
from clinical_memory.models.actors import released_images
from clinical_memory.analysis.metrics import episode_metrics
from clinical_memory.memory.evidence import evidence_view
from clinical_memory.environment.finish import FinishTolerantEpisode
from clinical_memory.environment.engine import evaluate
from clinical_memory.environment.events import event_exams
from clinical_memory.analysis.statuses import WASTED
from clinical_memory.memory.presentation import describe
from clinical_memory.memory.matching import match
from clinical_memory.models.clinical import ClinicalActors
from clinical_memory.memory.diagnosis import lookup, shown


def received(trace):
    """The history items and examination records that one visit obtained."""
    return {
        ("history", f["id"]) for e in trace["events"] for f in e.get("facts", [])
    } | {("examination", x["id"]) for e in trace["events"] for x in event_exams(e)}


def requests_until(events, found):
    """How many collection actions it took until the first one that returned what `found` looks for, or None."""
    return next((e["collection_actions"] for e in events if found(e)), None)


def execute(backend, case, gold, memory, settings, max_actions=8, notes_on=False):
    start, first, before = (time.monotonic(), len(backend.calls), memory.digest())
    entries, actors = (memory.entries(), ClinicalActors(backend))
    ep = FinishTolerantEpisode(case, gold["options"], max_actions)
    initial = ep.doctor_view()
    presentation = describe(
        actors.backend,
        {"public_context": initial["public_context"], "task": initial["task"]},
    )
    lessons, recalled = match(
        entries,
        presentation["named"],
        settings["retrieval_top_k"],
        settings["retrieval_threshold"],
        settings["retrieval_min_terms"],
    )
    actors.deliver(lessons)
    while ep.state == "collecting":
        raw, repair = actors.request("doctor", ep.doctor_view())
        ep.collect(raw, actors.patient, actors.measurement, repair)
    actors.deliver([])
    full = ep.doctor_view()
    view = evidence_view(full)
    noted, looked = (
        lookup(
            entries,
            view["options"],
            settings["notes_top_k"],
            settings["notes_threshold"],
        )
        if notes_on
        else ([], None)
    )
    notes = shown(noted)
    read = {**view, "notes": notes} if notes else view
    reasoning = actors.backend.generate(
        "reasoning", REASONING, deepcopy(read), released_images(view)
    ).strip()
    raw, repair = actors.request("final", {**read, "reasoning": reasoning})
    ep.submit_final(raw, repair)
    result, calls = (ep.result(), backend.calls[first:])
    if before != memory.digest():
        raise RuntimeError("Diagnosis mutated memory")
    events = full["history"]
    trace = {
        "id": case["id"],
        "mode": "clinical_memory",
        "initial_doctor_view": initial,
        "presentation": presentation,
        "events": events,
        "final_view": view,
        "notes": notes,
        "note_lesson_ids": [m["lesson_id"] for m in noted],
        "note_lookup": looked,
        "reasoning": reasoning,
        "final_choices": full["options"],
        "result": result,
        "finish_normalizations": ep.normalizations,
        "router_repairs": actors.router_repairs,
        "retrieved_lesson_ids": [m["lesson_id"] for m in lessons],
        "retrievals": [{"action_index": 1, **recalled}],
        "memory_before_sha256": before,
        "memory_after_diagnosis_sha256": memory.digest(),
        "memory_lesson_count": len(entries),
        "first_call_index": first + 1,
        "last_diagnosis_call_index": len(backend.calls),
    }
    summary = {
        "id": case["id"],
        "actions": ep.actions,
        "stop_reason": ep.stop_reason,
        "valid_final": result["submission"]["valid"],
        "model_calls": len(calls),
        "statuses": dict(Counter((e["status"] for e in events))),
        "input_tokens": sum((c["input_tokens"] for c in calls)),
        "output_tokens": sum((c["output_tokens"] for c in calls)),
        "model_call_seconds": round(sum((c["total_seconds"] for c in calls)), 3),
        "token_cap_hits": sum((c["hit_token_cap"] for c in calls)),
        "retrieved_lessons": len(lessons),
        "gathering_lessons": sum((bool(m["collect"]) for m in lessons)),
        "diagnosis_notes": len(notes),
        "finish_normalizations": len(ep.normalizations),
        "presentation_named": bool(presentation["named"]),
        "presentation_settled": bool(presentation.get("settled")),
        "reasoning_words": len(reasoning.split()),
    }
    metrics = episode_metrics(
        events, calls, result, time.monotonic() - start, max_actions
    )
    metrics["wasted_actions"] = sum((e["status"] in WASTED for e in events))
    metrics["case_images_attached"] = sum(
        (len(e.get("case_image_attached", [])) for e in events)
    )
    metrics["blocked_actions"] = sum(
        (e["status"].startswith("blocked") for e in events)
    )
    metrics["records_received"] = len(received(trace))
    metrics["examination_records_received"] = sum((len(event_exams(e)) for e in events))
    metrics["requests_until_examination"] = requests_until(
        events, lambda e: bool(event_exams(e))
    )
    metrics["requests_until_image"] = requests_until(
        events, lambda e: any((x.get("images") for x in event_exams(e)))
    )
    return {
        "trace": trace,
        "score": evaluate(result, gold),
        "summary": summary,
        "metrics": metrics,
    }
