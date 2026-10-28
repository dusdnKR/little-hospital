import json
from clinical_memory.environment.formatting import unfence
from clinical_memory.environment.encounter import ClinicalEpisode


def normalize_finish(raw):
    try:
        obj = json.loads(unfence(raw)) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return (raw, False)
    if (
        isinstance(obj, dict)
        and set(obj) == {"type", "query"}
        and (obj["type"] == "finish")
        and isinstance(obj["query"], str)
    ):
        return ('{"type":"finish"}', True)
    return (raw, False)


class FinishTolerantEpisode(ClinicalEpisode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalizations = []

    def collect(self, raw, patient_router, measurement_router, repair=None):

        def normalize(value, stage):
            normalized, changed = normalize_finish(value)
            if changed:
                self.normalizations.append(
                    {
                        "event_index": len(self._events) + 1,
                        "stage": stage,
                        "discarded_field": "query",
                    }
                )
            return normalized

        repaired = (
            (lambda error: normalize(repair(error), "repair")) if repair else None
        )
        return super().collect(
            normalize(raw, "initial"), patient_router, measurement_router, repaired
        )
