from collections import Counter
from clinical_memory.environment.events import event_exams


def episode_metrics(events, calls, result, wall_seconds, max_actions):
    requests = [
        e
        for e in events
        if e["type"] in {"ask_patient", "request_exam", "invalid_action"}
    ]
    image_requests = [
        e
        for e in requests
        if e.get("requested_image_exam")
        or any((x.get("images") for x in event_exams(e)))
    ]
    images = list(
        dict.fromkeys(
            (n for e in events for x in event_exams(e) for n in x.get("images", []))
        )
    )
    by_role = {}
    for c in calls:
        d = by_role.setdefault(
            c["role"], {"calls": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0}
        )
        d["calls"] += 1
        for key in ("input_tokens", "output_tokens"):
            d[key] += c[key]
        d["seconds"] = round(d["seconds"] + c["total_seconds"], 3)
    return {
        "metric_schema": 1,
        "max_collection_actions": max_actions,
        "collection_actions": len(requests),
        "final_dialogue_turn": len(requests) + 1,
        "doctor_decision_calls": sum((c["role"] == "doctor" for c in calls)),
        "patient_exchanges": sum((e["type"] == "ask_patient" for e in requests)),
        "exam_requests": sum((e["type"] == "request_exam" for e in requests)),
        "image_requested": bool(image_requests),
        "matched_image_requests": len(image_requests),
        "first_image_request_turn": image_requests[0]["collection_actions"]
        if image_requests
        else None,
        "image_names_released": images,
        "unique_images_released": len(images),
        "image_model_calls": sum((bool(c["images"]) for c in calls)),
        "final_answer_saw_image": any(
            (c["images"] and c["role"].startswith("final") for c in calls)
        ),
        "statuses": dict(Counter((e["status"] for e in requests))),
        "new_history_items": sum((len(e.get("facts", [])) for e in requests)),
        "model_calls": len(calls),
        "input_tokens": sum((c["input_tokens"] for c in calls)),
        "output_tokens": sum((c["output_tokens"] for c in calls)),
        "total_tokens": sum((c["input_tokens"] + c["output_tokens"] for c in calls)),
        "model_seconds": round(sum((c["total_seconds"] for c in calls)), 3),
        "episode_wall_seconds": round(wall_seconds, 3),
        "by_role": by_role,
        "stop_reason": result["stop_reason"],
        "valid_final": result["submission"]["valid"],
        "token_cap_hits": sum((c["hit_token_cap"] for c in calls)),
    }
