from clinical_memory.prompts.instructions import (
    EXAMINATION_NOTE,
    REASONING_NOTE,
    NOTES_NOTE,
)
from clinical_memory.models.guarded import GuardedActors
from clinical_memory.models.backend import ClinicalBackend as ClinicalBackend


class Instructed:
    """What the actors see of the backend: the same calls with the added instructions for the doctor's roles."""

    def __init__(self, backend):
        self._backend = backend

    def __getattr__(self, name):
        return getattr(self._backend, name)

    def generate(self, role, system, payload, images):
        record, base = (
            payload["record"] if role.endswith("_repair") else payload,
            role.removesuffix("_repair"),
        )
        if base == "doctor":
            system += EXAMINATION_NOTE
        if base in {"reasoning", "final"} and "notes" in record:
            system += NOTES_NOTE
        if base == "final" and "reasoning" in record:
            system += REASONING_NOTE
        return self._backend.generate(role, system, payload, images)


def gathering_text(lesson):
    """How the gathering part of a lesson is shown to the collecting doctor: its own words under its own name."""
    return "When: " + lesson["when"] + " Action: " + lesson["collect"]


class ClinicalActors(GuardedActors):
    def __init__(self, backend):
        super().__init__(Instructed(backend), [])

    def deliver(self, lessons):
        """The gathering parts of the lessons recalled for this patient, passed on unchanged."""
        self.lessons = [
            {"lesson_text": gathering_text(m)} for m in lessons if m["collect"]
        ]

    def request(self, role, view):
        if role == "doctor":
            return super().request(role, view)
        lessons, self.lessons = (self.lessons, [])
        try:
            return super().request(role, view)
        finally:
            self.lessons = lessons
