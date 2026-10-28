from contextlib import ExitStack
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from clinical_memory.memory.store import atomic_json
from clinical_memory.environment.finish import FinishTolerantEpisode
from clinical_memory.data import augment as make_augmented
from clinical_memory.data.augmented import (
    IMAGES,
    aug_id,
    aug_image,
    build_case,
    load_set,
    numbers,
    opening_errors,
    similarity,
)
from clinical_memory.config import inputs, load
from tests.fixtures import REWRITTEN, build_project, write_augmented

REAL = Path(__file__).resolve().parents[1]
RULE = load(REAL / "configs/experiment.json")["test2"]
SOURCE = "I am a 64-year-old woman. I came to the clinic because of a 3-week history of dry cough and a weight loss of 2 kg."
GOOD = "For 3 weeks I have had a dry cough and I have lost 2 kg, so I came to the clinic. I am a woman, 64 years old."


class RuleTests(unittest.TestCase):
    def test_the_augmented_patient_takes_the_number_of_its_source_moved_by_the_offset(
        self,
    ):
        self.assertEqual(
            [aug_id("MM-101", 5000), aug_id("MM-0", 5000), aug_id("E1", 5000)],
            ["MM-5101", "MM-5000", "E5001"],
        )
        self.assertEqual(
            [
                aug_image("MM-101-a.jpeg", "MM-101", "MM-5101"),
                aug_image("E1.jpeg", "E1", "E5001"),
            ],
            ["MM-5101-a.jpeg", "E5001.jpeg"],
        )
        with self.assertRaises(ValueError):
            aug_id("patient", 5000)
        with self.assertRaises(ValueError):
            aug_image("MM-7-a.png", "MM-101", "MM-5101")

    def test_a_rewritten_opening_statement_keeps_every_number_and_is_not_the_source_text(
        self,
    ):
        self.assertEqual(opening_errors(SOURCE, GOOD, RULE), [])
        self.assertLess(similarity(SOURCE, GOOD), 0.9)
        self.assertEqual(
            numbers("I weigh 12.5 kg, for 3 days, 64-year-old."), ["12.5", "3", "64"]
        )
        for rewritten, problem in (
            (SOURCE.upper(), "It is the text of the source patient."),
            (GOOD.replace("64", "65"), "A number was changed, added or lost."),
            (
                GOOD.replace("3 weeks", "three weeks"),
                "A number was changed, added or lost.",
            ),
            (GOOD + " This is case MM-101.", "A number was changed, added or lost."),
            (
                GOOD + " I think option B.",
                "It holds a case identifier or an answer letter.",
            ),
            ("I cough: 64, 3, 2.", "Its length is too far from that of the source."),
            (
                SOURCE.replace("dry", "mild"),
                "It is almost the text of the source patient.",
            ),
            ("  ", "It is empty."),
            (None, "It is empty."),
        ):
            self.assertIn(problem, opening_errors(SOURCE, rewritten, RULE), rewritten)
        self.assertEqual(
            opening_errors("I have a cough.", "I keep coughing lately.", RULE), []
        )
        self.assertEqual(
            opening_errors("I have a cough.", "I have a cough.", RULE),
            ["It is the text of the source patient."],
        )
        self.assertIn(
            "Its length is too far from that of the source.",
            opening_errors("I have a cough.", "Cough.", RULE),
        )

    def test_the_rewritten_statement_replaces_the_source_wording_wherever_the_patient_speaks_it(
        self,
    ):
        from tests.base_fixtures import OPENING, make_case, with_profile

        source = with_profile(make_case("MM-7"))
        source["dialogue_profile"]["utterances"]["P1"] = OPENING
        case = build_case(source, "MM-5007", GOOD)
        self.assertEqual(
            (
                case["dialogue_profile"]["utterances"],
                case["public_context"]["text"],
                case["augmented_from"],
                case["exams"][0]["images"],
            ),
            (
                {"OPENING": GOOD, "P1": GOOD, "P2": "It began after I fell."},
                GOOD,
                "MM-7",
                ["MM-5007.jpeg"],
            ),
        )
        episode = FinishTolerantEpisode(case, {k: "Diagnosis " + k for k in "ABCDE"}, 8)
        seen = json.dumps(
            [episode.doctor_view(), episode.patient_view("What brings you in?")]
        )
        self.assertTrue(OPENING not in seen and GOOD in seen)
        self.assertEqual(
            (
                source["id"],
                source["public_context"]["text"],
                source["dialogue_profile"]["utterances"]["P1"],
            ),
            ("MM-7", OPENING, OPENING),
        )

    def test_the_committed_recipe_changes_an_image_mildly_and_never_flips_it(self):
        recipe = RULE["image"]
        self.assertEqual(
            (
                recipe["flip"],
                recipe["rotation_degrees"] <= 10,
                recipe["crop_min"] >= 0.9,
                recipe["brightness"] <= 0.2,
                recipe["contrast"] <= 0.2,
            ),
            (False, True, True, True, True),
        )
        drawn = [
            make_augmented.transform_for(f"MM-{i}-a.jpeg", 42, recipe)
            for i in range(200)
        ]
        self.assertEqual(
            drawn[0], make_augmented.transform_for("MM-0-a.jpeg", 42, recipe)
        )
        self.assertNotEqual(
            drawn[0], make_augmented.transform_for("MM-0-a.jpeg", 43, recipe)
        )
        self.assertTrue(
            all(
                (
                    abs(t["rotation_degrees"]) <= recipe["rotation_degrees"]
                    and recipe["crop_min"] <= t["keep"] <= 1
                    and (abs(t["brightness"] - 1) <= recipe["brightness"])
                    and (abs(t["contrast"] - 1) <= recipe["contrast"])
                    and (0 <= t["shift_x"] <= 1)
                    and (0 <= t["shift_y"] <= 1)
                    for t in drawn
                )
            )
        )
        self.assertGreater(len({t["rotation_degrees"] for t in drawn}), 190)


class AugmentedSetTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        build_project(self.root)
        self.stack.enter_context(patch.object(make_augmented, "ROOT", self.root))
        self.addCleanup(self.stack.close)
        self.inp = inputs(self.root)
        self.folder = self.root / self.inp.protocol["test2"]["folder"]

    def built(self, openings=None):
        write_augmented(self.root, self.inp, openings)
        return make_augmented.build(self.root, self.inp)

    def test_the_work_list_holds_the_opening_statement_and_nothing_else_of_a_case_file(
        self,
    ):
        done = make_augmented.todo(self.root, self.inp)
        rows = [
            json.loads(s)
            for s in (self.folder / "openings_todo.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(
            (
                done["source_patients"],
                [r["source_id"] for r in rows],
                {tuple(sorted(r)) for r in rows},
            ),
            (6, self.inp.division["test2_sources"], {("opening", "source_id")}),
        )
        self.assertTrue(
            all(
                (
                    r["opening"]
                    == self.inp.cases[r["source_id"]]["public_context"]["text"]
                    for r in rows
                )
            )
        )

    def test_an_augmented_patient_is_its_source_with_a_rewritten_opening_and_transformed_images(
        self,
    ):
        public = self.built()
        cases, golds, sealed = load_set(self.root, self.inp)
        sources = self.inp.division["test2_sources"]
        self.assertEqual(
            (
                list(cases),
                public["patients"],
                public["images"],
                public["manifest_sha256"],
                make_augmented.check(self.root, self.inp)["answers_carried_over"],
            ),
            ([aug_id(c, 5000) for c in sources], 6, 6, sealed, 6),
        )
        for source_id, (new_id, case) in zip(sources, cases.items()):
            source = self.inp.cases[source_id]
            self.assertEqual(
                (
                    case["id"],
                    case["augmented_from"],
                    case["public_context"]["text"],
                    case["dialogue_profile"]["utterances"]["OPENING"],
                ),
                (new_id, source_id, REWRITTEN, REWRITTEN),
            )
            self.assertEqual(
                (
                    case["patient_facts"],
                    [(e["id"], e["text"]) for e in case["exams"]],
                    golds[new_id]["options"],
                    golds[new_id]["label"],
                    golds[new_id]["id"],
                ),
                (
                    source["patient_facts"],
                    [(e["id"], e["text"]) for e in source["exams"]],
                    self.inp.golds[source_id]["options"],
                    self.inp.golds[source_id]["label"],
                    new_id,
                ),
            )
            self.assertEqual(
                [n for e in case["exams"] for n in e["images"]],
                [
                    aug_image(n, source_id, new_id)
                    for e in source["exams"]
                    for n in e["images"]
                ],
            )
            episode = FinishTolerantEpisode(case, golds[new_id]["options"], 8)
            self.assertEqual(episode.doctor_view()["public_context"], REWRITTEN)
            self.assertNotIn(
                source["public_context"]["text"],
                json.dumps(episode.patient_view("How are you?")),
            )
            self.assertEqual(
                self.inp.cases[source_id]["public_context"]["text"],
                source["public_context"]["text"],
            )
        text = (self.root / "results/augmentation.json").read_text(encoding="utf-8")
        self.assertTrue(
            "knee" not in text
            and REWRITTEN not in text
            and (public["opening_statements"]["numbers_changed"] == 0)
        )

    def test_a_statement_that_breaks_a_rule_stops_the_build_without_printing_it(self):
        first = self.inp.division["test2_sources"][0]
        write_augmented(
            self.root,
            self.inp,
            {first: "My knee has been hurting for a while; I am a man, 41 years old."},
        )
        with self.assertRaises(ValueError) as caught:
            make_augmented.build(self.root, self.inp)
        self.assertEqual(
            str(caught.exception),
            "1 rewritten opening statements break a rule:\n"
            + first
            + ": A number was changed, added or lost.",
        )
        self.assertFalse((self.folder / "manifest.json").exists())
        rows = [
            json.loads(s)
            for s in (self.folder / "openings.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        for broken in (rows[1:], rows + [rows[0]], [{**r, "note": "x"} for r in rows]):
            (self.folder / "openings.jsonl").write_text(
                "".join((json.dumps(r) + "\n" for r in broken)), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "exactly one row"):
                make_augmented.build(self.root, self.inp)

    def test_a_run_refuses_augmented_patients_that_are_not_the_ones_that_were_built(
        self,
    ):
        with self.assertRaisesRegex(ValueError, "have not been built yet"):
            load_set(self.root, self.inp)
        self.built()
        manifest = load(self.folder / "manifest.json")
        name = manifest["patients"][0]["images"][0]["name"]
        (self.root / IMAGES / name).write_bytes(b"another image")
        with self.assertRaisesRegex(
            ValueError, "missing, changed or the source image itself: " + name
        ):
            load_set(self.root, self.inp)
        source = manifest["patients"][0]["images"][0]["source"]
        (self.root / IMAGES / name).write_bytes(
            (self.root / IMAGES / source).read_bytes()
        )
        changed = json.loads(json.dumps(manifest))
        changed["patients"][0]["images"][0]["sha256"] = make_augmented.file_sha(
            self.root / IMAGES / name
        )
        atomic_json(self.folder / "manifest.json", changed)
        with self.assertRaisesRegex(ValueError, "the source image itself"):
            load_set(self.root, self.inp)
        self.built()
        for change, message in (
            ({"division_sha256": "0" * 64}, "do not belong to this division"),
            ({"version": "other"}, "do not belong to this division"),
            (
                {"image": {**manifest["image"], "flip": True}},
                "do not belong to this division",
            ),
            (
                {"patients": manifest["patients"][::-1]},
                "do not belong to this division",
            ),
            (
                {
                    "patients": [
                        {
                            **manifest["patients"][0],
                            "opening": self.inp.cases[
                                manifest["patients"][0]["source_id"]
                            ]["public_context"]["text"],
                        },
                        *manifest["patients"][1:],
                    ]
                },
                "It is the text of the source patient",
            ),
            (
                {
                    "patients": [
                        {**manifest["patients"][0], "id": "E9999"},
                        *manifest["patients"][1:],
                    ]
                },
                "Invalid augmented patient",
            ),
            (
                {
                    "patients": [
                        {**manifest["patients"][0], "images": []},
                        *manifest["patients"][1:],
                    ]
                },
                "not those of its source",
            ),
        ):
            atomic_json(self.folder / "manifest.json", {**manifest, **change})
            with self.assertRaisesRegex(ValueError, message):
                load_set(self.root, self.inp)

    def test_real_images_are_transformed_once_under_the_name_of_the_augmented_patient(
        self,
    ):
        from PIL import Image

        sources = self.inp.division["test2_sources"]
        for cid in sources:
            picture = Image.new("RGB", (120, 80), (200, 30, 30))
            for x in range(60):
                for y in range(80):
                    picture.putpixel((x, y), (20, 20, 220))
            picture.save(
                self.root / IMAGES / (cid + ".jpeg"), format="JPEG", quality=95
            )
        done = make_augmented.images(self.root, self.inp)
        record = load(self.folder / "images.json")
        self.assertEqual(
            (done["images"], sorted(record["images"]), record["image"]),
            (
                6,
                sorted((c + ".jpeg" for c in sources)),
                self.inp.protocol["test2"]["image"],
            ),
        )
        for cid in sources:
            made, source = (
                self.root / IMAGES / (aug_id(cid, 5000) + ".jpeg"),
                self.root / IMAGES / (cid + ".jpeg"),
            )
            with Image.open(made) as im:
                width, height = im.size
                left, right = (
                    im.getpixel((width // 6, height // 2)),
                    im.getpixel((5 * width // 6, height // 2)),
                )
            self.assertTrue(
                made.read_bytes() != source.read_bytes()
                and 0.94 * 120 - 1 <= width <= 120
                and (0.94 * 80 - 1 <= height <= 80)
            )
            self.assertTrue(
                left[2] > left[0] and right[0] > right[2],
                "the left half stays on the left",
            )
        again = {p.name: p.read_bytes() for p in (self.root / IMAGES).glob("E5*.jpeg")}
        make_augmented.images(self.root, self.inp)
        self.assertEqual(
            again,
            {p.name: p.read_bytes() for p in (self.root / IMAGES).glob("E5*.jpeg")},
        )
        (self.root / IMAGES / (aug_id(sources[0], 5000) + ".jpeg")).write_bytes(
            b"something else"
        )
        with self.assertRaisesRegex(ValueError, "give --force to replace it"):
            make_augmented.images(self.root, self.inp)
        make_augmented.images(self.root, self.inp, force=True)
        self.assertEqual(
            again,
            {p.name: p.read_bytes() for p in (self.root / IMAGES).glob("E5*.jpeg")},
        )
        rows = [{"source_id": c, "opening": REWRITTEN} for c in sources]
        (self.folder / "openings.jsonl").write_text(
            "".join((json.dumps(r) + "\n" for r in rows)), encoding="utf-8"
        )
        self.assertEqual(make_augmented.build(self.root, self.inp)["images"], 6)
        self.assertEqual(len(load_set(self.root, self.inp)[0]), 6)


if __name__ == "__main__":
    unittest.main()
