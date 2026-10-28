from copy import deepcopy
import json
import re
from clinical_memory.environment.engine import ProtocolError, parse_object
from clinical_memory.environment.formatting import unfence, clean_repair
from clinical_memory.environment.guards import GuardedEpisode, exact_key

UNLABELED = re.compile("^(?:source_image|clinical_image|clinical_photograph)")
PLACEHOLDER = "Source image for the requested examination."
UNLABELED_TEXT = "Unlabeled image from the case file. The source does not state its modality or body site."
ATTACH_NOTE = "An unlabeled image from the case file is attached to this examination request. It is not the result of the examination that was named."
EXHAUSTED = "I have already told you everything I know. I have nothing more to add."
BLOCKED = {
    "blocked_exhausted": "The patient already said there is nothing more to add. Request an examination or finish.",
    "blocked_repeat": "You already asked this exact question; the answer will not change. Ask something different, request an examination, or finish.",
}
RULE = "Examinations in unavailable_exams cannot be requested again in this encounter; their findings stay unknown, not normal or negative. Rephrasing a blocked request does not make it available. An identical question is not answered twice. Once the patient says there is nothing more to add, further history questions return nothing. A blocked attempt still spends one action. For a compound request, omit its blocked part and request the rest separately."


def valid_action(obj):
    return (
        obj.get("type") == "finish"
        and set(obj) == {"type"}
        or (
            obj.get("type") in {"ask_patient", "request_exam"}
            and set(obj) == {"type", "query"}
            and isinstance(obj["query"], str)
            and (0 < len(obj["query"].strip()) <= 2000)
        )
    )


class ClinicalEpisode(GuardedEpisode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._asked = set()
        self._history_closed = False

    def history_exhausted(self):
        return all((("patient", x) in self._seen for x in self._facts))

    def unlabeled_pending(self):
        return [
            x
            for x, exam in self._exams.items()
            if exam["images"] and UNLABELED.match(x) and (("exam", x) not in self._seen)
        ]

    def doctor_view(self):
        view = super().doctor_view()
        view["request_rule"] = RULE
        if self._history_closed:
            view["patient_history"] = "The patient has nothing more to add."
        return view

    def _history_block(self, query):
        if self._history_closed:
            return ("blocked_exhausted", BLOCKED["blocked_exhausted"])
        if self.history_exhausted():
            self._history_closed = True
            return ("exhausted", EXHAUSTED)
        if exact_key(query) in self._asked:
            return ("blocked_repeat", BLOCKED["blocked_repeat"])
        return None

    def _spend(self, query, repairs, status, message):
        self.actions += 1
        if self.actions == self.max_actions:
            self._finish("action_limit")
        event = {
            "type": "ask_patient",
            "query": query,
            "status": status,
            "collection_actions": self.actions,
            "format_repairs": repairs,
            "patient_called": False,
            "message": message,
            "state_after": self.state,
        }
        if status == "exhausted":
            event["utterance"] = message
        self._events.append(deepcopy(event))
        return deepcopy(event)

    def collect(self, raw, patient_router, measurement_router, repair=None):
        if self.state != "collecting":
            raise ProtocolError("Information collection is closed")
        action, repairs = parse_object(unfence(raw), valid_action, clean_repair(repair))
        if action and action["type"] == "ask_patient":
            blocked = self._history_block(action["query"])
            if blocked:
                return self._spend(action["query"], repairs, *blocked)
            self._asked.add(exact_key(action["query"]))
        attached = []

        def with_case_image(view):
            choice = measurement_router(view)
            if isinstance(choice, dict) and isinstance(choice.get("exam_ids"), list):
                attached.extend(
                    (x for x in self.unlabeled_pending() if x not in choice["exam_ids"])
                )
                choice = {**choice, "exam_ids": choice["exam_ids"] + attached}
            return choice

        event = super().collect(
            json.dumps(action) if action else "{}", patient_router, with_case_image
        )
        delivered = {x["id"] for x in event.get("exams", [])}
        attached = [x for x in attached if x in delivered]
        for target in (event, self._events[-1]):
            target["format_repairs"] = repairs
            if attached:
                target["case_image_attached"] = list(attached)
                target["case_image_note"] = ATTACH_NOTE
                for exam in target["exams"]:
                    if exam["id"] in attached and exam["text"] == PLACEHOLDER:
                        exam["text"] = UNLABELED_TEXT
        return event
