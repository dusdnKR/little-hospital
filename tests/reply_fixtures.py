import tests.base_fixtures as base_fixtures

NAMED = {"complaint": "knee pain", "context": ["adult", "injury"]}
WHEN = "knee pain, adult, injury"
LOOSE_NAME = {"complaint": "knee pain", "context": ["40 years old", "injury"]}
LOOSE_WHEN = "knee pain, years old, injury"
UNUSABLE_NAME = {"complaint": "40", "context": ["7 days"]}
COLLECT = "Request a knee radiograph before finishing."
COLLECT_KEY = "Request a knee radiograph first; the case file held one with an image."
DIAGNOSE = (
    "A fracture line on the radiograph after a fall separates a fracture from a sprain."
)
REASONED = "The radiograph settles it."
REASONED_WITHOUT_IMAGE = "No examination record was received."


class Scripted(base_fixtures.ScriptedBackend):
    unusable_action_for = loose_for = unnamed_for = rewrite_for = frozenset()

    def reply(self, base, payload, images):
        record = payload.get("record", payload)
        if base == "presentation":
            return (
                UNUSABLE_NAME
                if self.case_id in Scripted.unnamed_for
                else LOOSE_NAME
                if self.case_id in Scripted.loose_for
                else NAMED
            )
        if base == "reasoning":
            return REASONED if images else REASONED_WITHOUT_IMAGE
        if (
            base == "doctor"
            and self.case_id in Scripted.unusable_action_for
            and (not record["history"])
        ):
            return {
                "type": "ask_patient",
                "query": "What symptoms do you have?",
                "reason": "history first",
            }
        if base != "reflection":
            return super().reply(base, payload, images)
        if self.reflection is not None:
            return self.reflection
        sheet = "answer_key" in record
        if self.case_id in Scripted.rewrite_for and "previous_output" not in payload:
            return {
                "collect": "Choose option B after the radiograph.",
                **({"diagnose": DIAGNOSE} if sheet else {}),
            }
        if self.case_id in base_fixtures.ScriptedBackend.abstain_for:
            return {"collect": "", "diagnose": ""} if sheet else {"collect": ""}
        return (
            {"collect": COLLECT_KEY, "diagnose": DIAGNOSE}
            if sheet
            else {"collect": COLLECT}
        )


def reset():
    base_fixtures.ScriptedBackend.count, base_fixtures.ScriptedBackend.fail_at = (
        0,
        None,
    )
    base_fixtures.ScriptedBackend.abstain_for = (
        base_fixtures.ScriptedBackend.keep_guidance_for
    ) = frozenset()
    Scripted.unusable_action_for = Scripted.loose_for = Scripted.unnamed_for = (
        Scripted.rewrite_for
    ) = frozenset()
    Scripted.reflection = None
