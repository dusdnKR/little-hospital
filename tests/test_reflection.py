import json
import unittest
from clinical_memory.prompts.reflection import COLLECT_WORDS, DIAGNOSE_WORDS, PROMPTS
from clinical_memory.memory.reflection import draft, usable

TRACE = {
    "id": "E1",
    "result": {"submission": {"valid": True, "answer": "B"}},
    "presentation": {
        "named": {"complaint": "knee pain", "context": ["adult", "injury"]}
    },
    "initial_doctor_view": {
        "public_context": "I am a man with knee pain.",
        "task": "Determine the most likely diagnosis.",
    },
    "events": [],
    "reasoning": "No examination record was received.",
    "final_choices": {"A": "Tibial plateau fracture", "B": "Sprain"},
}
SHEET = (
    {
        "confirmed_diagnosis": "Tibial plateau fracture",
        "your_diagnosis_was_confirmed": False,
        "history_you_did_not_hear": [],
        "examinations_you_did_not_receive": [],
    },
    [],
)
GOOD = {
    "collect": "Request a knee radiograph first.",
    "diagnose": "A fracture line separates a fracture from a sprain.",
}
LONG = " ".join((f"word{i}" for i in range(60)))


class Replies:
    """A doctor that gives the listed replies in turn."""

    def __init__(self, *replies):
        self.replies, self.calls = (list(replies), [])

    def generate(self, role, system, payload, images):
        self.calls.append(
            {
                "role": role,
                "system": system,
                "payload": payload,
                "model_revision": "rev",
            }
        )
        reply = self.replies.pop(0)
        return reply if isinstance(reply, str) else json.dumps(reply)


class LessonTests(unittest.TestCase):
    def test_a_lesson_written_with_the_answer_sheet_is_filed_under_the_confirmed_diagnosis(
        self,
    ):
        backend = Replies(GOOD)
        lesson, info = draft(backend, TRACE, "evolution", 3, SHEET)
        self.assertEqual(info, {"status": "stored", "repairs": 0})
        self.assertEqual(
            (
                lesson["lesson_id"],
                lesson["disease"],
                lesson["outcome"],
                lesson["collect"],
                lesson["diagnose"],
                lesson["when"],
                lesson["evidence_refs"],
                lesson["source_case_id"],
                lesson["created_after_step"],
            ),
            (
                "L3",
                "Tibial plateau fracture",
                "not confirmed",
                GOOD["collect"],
                GOOD["diagnose"],
                "knee pain, adult, injury",
                ["encounter", "answer_key"],
                "E1",
                3,
            ),
        )
        self.assertEqual(
            lesson["lesson_text"],
            "When: knee pain, adult, injury Action: "
            + GOOD["collect"]
            + " Diagnosis: "
            + GOOD["diagnose"],
        )
        call = backend.calls[0]
        self.assertEqual(
            (call["role"], call["system"], call["payload"]["answer_key"]),
            ("reflection", PROMPTS["key"], SHEET[0]),
        )
        own, info = draft(Replies({"collect": GOOD["collect"]}), TRACE, "evolution", 1)
        self.assertEqual(
            (
                info["status"],
                own["diagnose"],
                "disease" in own,
                "outcome" in own,
                own["evidence_refs"],
            ),
            ("stored", "", False, False, ["encounter"]),
        )

    def test_the_instruction_says_for_which_later_patient_each_part_is_written(self):
        self.assertIn("Longer lessons are allowed when useful.", PROMPTS["self"])
        for words in (
            "collect, for a later patient with the same main complaint",
            "diagnose, for a later patient for whom the confirmed diagnosis of this patient is one of the candidate diagnoses",
            f"aiming for about {COLLECT_WORDS} words",
            f"aiming for about {DIAGNOSE_WORDS} words",
            "do not write answer letters",
            'Prefer {"collect":"...","diagnose":"..."}',
        ):
            self.assertIn(words, PROMPTS["key"])

    def test_long_lessons_are_stored_whole_without_a_rewrite(self):
        for policy in ("settle", "discard"):
            backend = Replies({"collect": LONG, "diagnose": LONG + " diagnosis ending"})
            lesson, info = draft(backend, TRACE, "evolution", 1, SHEET, policy)
            self.assertEqual(info, {"status": "stored", "repairs": 0})
            self.assertEqual(
                (lesson["collect"], lesson["diagnose"]),
                (LONG, LONG + " diagnosis ending"),
            )
            self.assertEqual(len(backend.calls), 1)
        lesson, info = draft(Replies({"collect": LONG}), TRACE, "evolution", 1)
        self.assertEqual(lesson["collect"], LONG)

    def test_usable_format_variations_are_accepted_on_the_first_reply(self):
        variants = [
            ("Here is my lesson: " + json.dumps(GOOD) + " End.", GOOD),
            ({**GOOD, "comment": "extra"}, GOOD),
            (
                {"collect": GOOD["collect"]},
                {"collect": GOOD["collect"], "diagnose": ""},
            ),
            (
                {"diagnose": GOOD["diagnose"]},
                {"collect": "", "diagnose": GOOD["diagnose"]},
            ),
            (
                {
                    "collect": ["Ask about pain.", "Request a radiograph."],
                    "diagnose": GOOD["diagnose"],
                },
                {
                    "collect": "Ask about pain. Request a radiograph.",
                    "diagnose": GOOD["diagnose"],
                },
            ),
            (
                {
                    "collect": "Choose option B after the radiograph of MM-12.",
                    "diagnose": GOOD["diagnose"],
                },
                {
                    "collect": "Choose after the radiograph of",
                    "diagnose": GOOD["diagnose"],
                },
            ),
        ]
        for reply, expected in variants:
            backend = Replies(reply)
            lesson, info = draft(backend, TRACE, "evolution", 1, SHEET)
            self.assertEqual({k: lesson[k] for k in expected}, expected)
            self.assertEqual(info["repairs"], 0)
            self.assertTrue(info["settled"])
            self.assertEqual(len(backend.calls), 1)
        lesson, info = draft(Replies(GOOD), TRACE, "evolution", 1)
        self.assertEqual((lesson["collect"], lesson["diagnose"]), (GOOD["collect"], ""))

    def test_unreadable_parts_get_one_rewrite_and_no_invented_lesson(self):
        backend = Replies("not json", GOOD)
        lesson, info = draft(backend, TRACE, "evolution", 1, SHEET)
        self.assertEqual(info, {"status": "stored", "repairs": 1})
        self.assertEqual(lesson["diagnose"], GOOD["diagnose"])
        for reply in (
            "not json",
            "[1, 2]",
            {"collect": 7, "diagnose": None},
            {"collect": "option B", "diagnose": 3},
            json.dumps(GOOD) + " " + json.dumps(GOOD),
        ):
            lesson, info = draft(Replies(reply, reply), TRACE, "evolution", 1, SHEET)
            self.assertEqual(
                (lesson, info["status"], info["repairs"]), (None, "invalid", 1)
            )
        with self.assertRaisesRegex(ValueError, "Unknown lesson format policy"):
            draft(Replies(GOOD), TRACE, "evolution", 1, SHEET, "keep")
        self.assertEqual(
            draft(
                Replies({"collect": "", "diagnose": ""}), TRACE, "evolution", 1, SHEET
            ),
            (None, {"status": "abstained", "repairs": 0}),
        )
        with self.assertRaises(PermissionError):
            draft(Replies(GOOD), TRACE, "test1", 1, SHEET)
        invalid = {**TRACE, "result": {"submission": {"valid": False}}}
        unnamed = {**TRACE, "presentation": {"named": None}}
        self.assertEqual(
            [
                draft(Replies(), t, "evolution", 1, SHEET)[1]["status"]
                for t in (invalid, unnamed)
            ],
            ["skipped_invalid_final", "no_presentation"],
        )

    def test_usable_preserves_words_and_empty_abstention(self):
        parts, done = usable(
            json.dumps(
                {"collect": "  Request   a radiograph. ", "diagnose": "answer: C fits."}
            ),
            ("collect", "diagnose"),
        )
        self.assertEqual(
            (parts, done),
            (
                {"collect": "Request a radiograph.", "diagnose": "fits."},
                ["diagnose lost a case identifier or an answer letter"],
            ),
        )
        self.assertEqual(
            usable('```json\n{"collect": "Ask about the fall."}\n```', ("collect",)),
            ({"collect": "Ask about the fall."}, []),
        )
        for nothing in ("", "text", '"a string"', json.dumps({"other": "x"})):
            self.assertEqual(
                usable(nothing, ("collect", "diagnose")), (None, []), nothing
            )
        self.assertEqual(
            usable('{"collect":"","diagnose":""}', ("collect", "diagnose")),
            ({"collect": "", "diagnose": ""}, []),
        )


if __name__ == "__main__":
    unittest.main()
