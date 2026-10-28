from copy import deepcopy
import json


class RoleBackend:
    measurement_protocol = "multi_exam_v1"

    def __init__(self, primary, measurement, output, doctor=None):
        self.primary, self.measurement, self.output = (primary, measurement, output)
        self.doctor = doctor or primary
        self.config = primary.config
        self.calls = []
        self.case_id = None

    def generate(self, role, system, payload, images):
        base_role = role.removesuffix("_repair")
        selected = (
            self.measurement
            if base_role == "measurement"
            else self.doctor
            if base_role in {"doctor", "final", "final_open", "direct", "image_probe"}
            else self.primary
        )
        if base_role == "measurement" and images:
            raise ValueError(
                "Measurement only selects IDs; it must not interpret images"
            )
        selected.case_id = self.case_id
        raw = selected.generate(role, system, payload, images)
        entry = deepcopy(selected.calls[-1])
        entry.update(
            backend_call_index=entry["call_index"],
            call_index=len(self.calls) + 1,
            model_id=selected.config["model_id"],
            model_revision=selected.config["model_revision"],
        )
        self.calls.append(entry)
        with (self.output / "private_calls.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return raw
