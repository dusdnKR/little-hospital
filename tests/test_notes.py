import unittest
from clinical_memory.memory.diagnosis import likeness, lookup, name_terms, shown


def lesson(index, disease, diagnose="Findings that point to it."):
    return {
        "lesson_id": f"L{index}",
        "disease": disease,
        "diagnose": diagnose,
        "collect": "Request an examination.",
    }


OPTIONS = {
    "A": "Acute aortic dissection",
    "B": "Hemothorax",
    "C": "Glycogen storage disease type I",
    "D": "Reiter’s syndrome",
    "E": "Renal cell carcinoma",
}


class NameTests(unittest.TestCase):
    def test_a_name_is_compared_by_its_words(self):
        self.assertEqual(
            name_terms("Carcinoma of the Bladder"), {"carcinoma", "bladder"}
        )
        self.assertEqual(
            name_terms("Reiter’s syndrome"), name_terms("Reiter's syndrome")
        )
        self.assertEqual(name_terms("Reiter’s syndrome"), {"reiter", "syndrome"})
        self.assertEqual(name_terms("Hydatid cysts"), {"hydatid", "cyst"})
        self.assertEqual(
            name_terms("Glycogen storage disease type II")
            - name_terms("Glycogen storage disease type I"),
            {"ii"},
        )
        self.assertEqual(
            (name_terms("A"), likeness("A", "A"), likeness("", "Hemothorax")),
            (frozenset(), 0.0, 0.0),
        )

    def test_the_same_disease_is_found_through_a_qualifier_or_another_word_order_but_not_through_a_synonym(
        self,
    ):
        for first, second in (
            ("Aortic dissection", "Acute aortic dissection"),
            ("Renal cell carcinoma", "Metastatic renal cell carcinoma"),
            ("Retinal artery occlusion", "Central retinal-artery occlusion"),
            ("Type B aortic dissection", "Aortic dissection, type B"),
            (
                "Stanford type B dissection of the aorta",
                "Aortic dissection, Stanford type B",
            ),
            ("Right middle lobe pneumonia", "Pneumonia of the right middle lobe"),
        ):
            self.assertGreaterEqual(likeness(first, second), 0.6, (first, second))
        for first, second in (
            ("Hemochromatosis", "Cardiac hemochromatosis"),
            ("Reiter’s syndrome", "Turner’s syndrome"),
            ("Basal cell carcinoma", "Renal-cell carcinoma"),
            ("Hepatic abscess", "Splenic abscess"),
            ("Ascariasis", "Ascaris lumbricoides"),
            ("Complete bowel obstruction", "High grade (complete) SBO obstruction"),
        ):
            self.assertLess(likeness(first, second), 0.6, (first, second))
        self.assertEqual(
            likeness("Renal cell carcinoma", "renal  CELL carcinoma."), 1.0
        )


class LookupTests(unittest.TestCase):
    def test_the_notes_about_the_candidate_diagnoses_are_found_best_match_first(self):
        entries = [
            lesson(1, "Hemothorax"),
            lesson(2, "Aortic dissection"),
            lesson(3, "Myeloma"),
            lesson(4, "Acute aortic dissection"),
            lesson(5, "Hemothorax"),
            lesson(6, "Turner’s syndrome"),
        ]
        found, audit = lookup(entries, OPTIONS, top_k=5, threshold=0.6)
        self.assertEqual([m["lesson_id"] for m in found], ["L1", "L4", "L5", "L2"])
        self.assertEqual(
            audit,
            {
                "threshold": 0.6,
                "top_k": 5,
                "memory_count": 6,
                "with_diagnosis_part": 6,
                "matched": 4,
                "selected": [
                    {"lesson_id": "L1", "score": 1.0, "options": ["B"]},
                    {"lesson_id": "L4", "score": 1.0, "options": ["A"]},
                    {"lesson_id": "L5", "score": 1.0, "options": ["B"]},
                    {"lesson_id": "L2", "score": 2 / 3, "options": ["A"]},
                ],
            },
        )
        self.assertEqual(
            shown(found)[0],
            {"diagnosis": "Hemothorax", "note": "Findings that point to it."},
        )
        found[0]["diagnose"] = "changed"
        self.assertEqual(entries[0]["diagnose"], "Findings that point to it.")

    def test_at_most_top_k_notes_and_only_lessons_with_a_diagnosis_part_filed_under_a_diagnosis(
        self,
    ):
        entries = [lesson(i, "Hemothorax") for i in range(1, 8)]
        self.assertEqual(
            [m["lesson_id"] for m in lookup(entries, OPTIONS, top_k=3)[0]],
            ["L1", "L2", "L3"],
        )
        without = [
            {"lesson_id": "L1", "collect": "Request an examination.", "diagnose": ""},
            {
                "lesson_id": "L2",
                "disease": "Hemothorax",
                "diagnose": "",
                "collect": "x",
            },
            {
                "lesson_id": "L3",
                "diagnose": "A note that is filed under nothing.",
                "collect": "x",
            },
        ]
        found, audit = lookup(without, OPTIONS)
        self.assertEqual(
            (found, audit["with_diagnosis_part"], audit["matched"], audit["selected"]),
            ([], 0, 0, []),
        )
        self.assertEqual(lookup([], OPTIONS)[1]["memory_count"], 0)
        for bad in ({"threshold": 0}, {"threshold": 1.1}, {"top_k": 0}, {"top_k": 2.0}):
            with self.assertRaises(ValueError):
                lookup(entries, OPTIONS, **bad)

    def test_a_note_can_be_about_a_wrong_option(self):
        found, audit = lookup(
            [lesson(1, "Renal cell carcinoma"), lesson(2, "Reiter syndrome")], OPTIONS
        )
        self.assertEqual([s["options"] for s in audit["selected"]], [["E"], ["D"]])
        self.assertEqual(len(found), 2)


if __name__ == "__main__":
    unittest.main()
