import re
from clinical_memory.models.actors import QwenActors


def procedure_guard(query, choice):
    if not isinstance(choice, dict) or "exam_ids" not in choice:
        return (choice, [])
    match = re.search("\\banoscop(?:y|ic examination)\\b", query, re.I)
    if not match or "perianal_exam" not in choice["exam_ids"]:
        return (choice, [])
    separate_external_request = re.search(
        "\\b(?:inspection|physical exam(?:ination)?|digital rectal exam(?:ination)?)\\b",
        query,
        re.I,
    )
    rejected = [] if separate_external_request else ["perianal_exam"]
    return (
        {
            "exam_ids": [x for x in choice["exam_ids"] if x not in rejected],
            "unavailable_requests": list(
                dict.fromkeys(choice["unavailable_requests"] + [match.group(0)])
            ),
        },
        rejected,
    )


class GuardedActors(QwenActors):
    def measurement(self, view):
        choice = super().measurement(view)
        if any(("anoscop" in e["id"].casefold() for e in view["available_exams"])):
            return choice
        choice, rejected = procedure_guard(view["query"], choice)
        self.router_repairs[-1]["procedure_guard_rejected_ids"] = rejected
        return choice
