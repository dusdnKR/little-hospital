from copy import deepcopy
from clinical_memory.environment.events import event_exams
from clinical_memory.environment.encounter import PLACEHOLDER, UNLABELED, UNLABELED_TEXT

TOLD = {True: "confirmed", False: "not confirmed"}


def answer_key(case, gold, visit):
    """The answer sheet of one patient for the visit just finished: the confirmed diagnosis, whether the submitted
    diagnosis matched it, and what the case file holds that this visit did not obtain. Returns (sheet, image names)."""
    trace = visit["trace"]
    heard = {f["id"] for e in trace["events"] for f in e.get("facts", [])} | set(
        case["public_context"].get("initially_disclosed_fact_ids", [])
    )
    seen = {x["id"] for e in trace["events"] for x in event_exams(e)}
    speech = (case.get("dialogue_profile") or {}).get("utterances", {})
    missed = [e for e in case["exams"] if e["id"] not in seen]
    sheet = {
        "confirmed_diagnosis": gold["options"][gold["label"]],
        "your_diagnosis_was_confirmed": bool(visit["score"]["correct"]),
        "history_you_did_not_hear": [
            speech.get(f["id"], f["text"])
            for f in case["patient_facts"]
            if f["id"] not in heard
        ],
        "examinations_you_did_not_receive": [
            {
                "name": e["id"],
                "findings": UNLABELED_TEXT
                if UNLABELED.match(e["id"]) and e["text"] == PLACEHOLDER
                else e["text"],
                "images": list(e["images"]),
            }
            for e in missed
        ],
    }
    return (sheet, [name for e in missed for name in e["images"]])


def own_record(trace):
    """What the reflecting doctor reads of its own visit, one reference per step."""
    start, answer = (
        trace["initial_doctor_view"],
        trace["result"]["submission"]["answer"],
    )
    return {
        "opening": start["public_context"],
        "task": start["task"],
        "presentation": deepcopy(trace["presentation"]["named"]),
        "encounter": [
            {"ref": f"step:{i}", **event} for i, event in enumerate(trace["events"], 1)
        ],
        "reasoning": trace["reasoning"],
        "submitted_diagnosis": trace["final_choices"][answer],
    }
