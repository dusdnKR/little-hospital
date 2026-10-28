from clinical_memory.prompts.roles import PROMPTS
from copy import deepcopy
from clinical_memory.environment.engine import parse_object
from clinical_memory.environment.events import event_exams
from clinical_memory.prompts.measurement import PROMPT as MEASUREMENT_PROMPT
from clinical_memory.environment.measurement import (
    parse_selection,
    final_json_text,
    modality_guard,
    safe_unavailable_text,
)


def released_images(view):
    """Read images only from already released exam events, never a raw case."""
    return list(
        dict.fromkeys(
            (
                name
                for event in view.get("history", [])
                for exam in event_exams(event)
                for name in exam.get("images", [])
            )
        )
    )


class QwenActors:
    def __init__(self, backend, lessons=None):
        self.backend = backend
        self.router_repairs = []
        self.lessons = [{"lesson_text": item["lesson_text"]} for item in lessons or []]

    def request(self, role, view):
        payload = deepcopy(view)
        system = PROMPTS[role]
        if getattr(self.backend, "measurement_protocol", None) == "multi_exam_v1":
            if role == "measurement":
                system = MEASUREMENT_PROMPT
            if role == "measurement" and any(
                ("scope" in e for e in view["available_exams"])
            ):
                from clinical_memory.prompts.scope import SCOPE_PROMPT

                system += SCOPE_PROMPT
            if role == "doctor":
                system = system.replace(
                    "request one relevant exam", "request relevant examinations"
                )
                system = system.replace(
                    '"one exam request"', '"an examination request"'
                )
        if role in {"doctor", "final", "final_open"} and self.lessons:
            payload["memory"] = deepcopy(self.lessons)
            system += " Memory contains lessons from earlier training cases, not facts about this patient. Use them as fallible suggestions; verify relevance against the current evidence."
        images = (
            released_images(payload)
            if role in {"doctor", "final", "final_open"}
            else []
        )
        raw = self.backend.generate(role, system, payload, images)

        def repair(error):
            return self.backend.generate(
                role + "_repair",
                system,
                {"record": payload, "previous_output": raw, "format_error": error},
                images,
            )

        return (raw, repair)

    def patient(self, view):
        allowed = {f["id"] for f in view["facts"]}
        raw, repair = self.request("patient", view)
        obj, count = parse_object(
            raw,
            lambda x: (
                set(x) == {"fact_ids"}
                and isinstance(x["fact_ids"], list)
                and all((isinstance(i, str) and i in allowed for i in x["fact_ids"]))
            ),
            repair,
        )
        self.router_repairs.append(
            {"role": "patient", "repairs": count, "valid": obj is not None}
        )
        return obj or {}

    def measurement(self, view):
        allowed = {exam["id"] for exam in view["available_exams"]}
        raw, repair = self.request("measurement", view)
        if getattr(self.backend, "measurement_protocol", None) == "multi_exam_v1":
            obj = parse_selection(final_json_text(raw), allowed)
            count = 0
            if obj is None:
                count = 1
                obj = parse_selection(
                    final_json_text(
                        repair(
                            "Return only JSON with exam_ids and unavailable_requests arrays."
                        )
                    ),
                    allowed,
                )
            rejected = []
            if obj is not None:
                obj, rejected = modality_guard(view["query"], obj, allowed)
                if any(("scope" in e for e in view["available_exams"])):
                    from clinical_memory.environment.scope import scope_guard

                    obj, scope_rejected = scope_guard(
                        view["query"],
                        obj,
                        view["available_exams"],
                        getattr(
                            self.backend, "measurement_scope_guard", "source_topics_v1"
                        ),
                    )
                    rejected += scope_rejected
                obj = safe_unavailable_text(view["query"], obj)
            self.router_repairs.append(
                {
                    "role": "measurement",
                    "repairs": count,
                    "valid": obj is not None,
                    "modality_guard_rejected_ids": rejected,
                }
            )
            return obj or {}
        obj, count = parse_object(
            raw,
            lambda x: (
                set(x) == {"exam_id"}
                and (
                    x["exam_id"] is None
                    or (isinstance(x["exam_id"], str) and x["exam_id"] in allowed)
                )
            ),
            repair,
        )
        self.router_repairs.append(
            {"role": "measurement", "repairs": count, "valid": obj is not None}
        )
        return obj or {}
