from copy import deepcopy
from clinical_memory.memory.store import tokens
from clinical_memory.environment.events import event_exams


def evidence_view(view):
    history = []
    for event in view.get("history", []):
        facts = event.get("facts", [])
        exams = event_exams(event)
        if not facts and (not exams):
            continue
        row = {}
        if facts:
            row.update(
                type="ask_patient",
                facts=deepcopy(facts),
                speaker=event.get("speaker", "patient"),
            )
        if exams:
            row.update(type="request_exam", exams=deepcopy(exams))
        if event.get("case_image_note"):
            row["case_image_note"] = event["case_image_note"]
        history.append(row)
    return {
        "public_context": deepcopy(view["public_context"]),
        "task": view["task"],
        "history": history,
        **({"options": deepcopy(view["options"])} if "options" in view else {}),
    }


GENERIC = set(
    "ask asks asked request requested exam examination examinations test tests result results finding findings obtain history symptom symptoms consider assess evaluate check review information relevant clinical condition presentation when action boundary".split()
)


def content_tokens(text):
    return set(tokens(text)) - GENERIC
