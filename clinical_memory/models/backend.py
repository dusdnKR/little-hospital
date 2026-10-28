import hashlib
import json


def call_seed(seed, cid, role, index):
    return (
        int(hashlib.sha256(f"{seed}:{cid}:{role}:{index}".encode()).hexdigest()[:8], 16)
        % 2147483647
    )


from clinical_memory.models.routing import RoleBackend

OWN = frozenset({"doctor", "final", "reasoning", "presentation", "reflection"})
SAMPLED = frozenset({"doctor", "final", "reasoning"})
SAMPLING = {"temperature": 0.3, "top_p": 0.9, "top_k": 20}


class ClinicalBackend(RoleBackend):
    def __init__(self, *args, seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = seed
        self.phase = None
        self.role_counts = {}
        path = self.output / "private_calls.jsonl"
        if path.exists():
            self.calls = [
                json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()
            ]

    def start_episode(self, phase, cid):
        self.phase = phase
        self.case_id = phase + ":" + cid
        self.role_counts = {}
        self.cid = cid

    def restart_roles(self):
        """A second visit of the same patient starts its role counters again, so it draws the same seeds as the first."""
        self.role_counts = {}

    def generate(self, role, system, payload, images):
        base = role.removesuffix("_repair")
        selected = (
            self.measurement
            if base == "measurement"
            else self.doctor
            if base in OWN
            else self.primary
        )
        previous, primary = (selected.config.copy(), self.primary)
        self.role_counts[role] = self.role_counts.get(role, 0) + 1
        try:
            sampling = self.phase == "evolution" and base in SAMPLED
            selected.config = {k: v for k, v in previous.items() if k not in SAMPLING}
            selected.config["do_sample"] = sampling
            if sampling:
                selected.config.update(SAMPLING)
            selected.set_seed(
                call_seed(self.seed, self.cid, role, self.role_counts[role])
            )
            if base in OWN:
                self.primary = self.doctor
            return RoleBackend.generate(self, role, system, payload, images)
        finally:
            selected.config, self.primary = (previous, primary)
