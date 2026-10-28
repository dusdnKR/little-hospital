from contextlib import ExitStack
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from clinical_memory.memory.store import MemoryStore, atomic_json
from tests.base_fixtures import ScriptedBackend
from clinical_memory.data import augment as make_augmented
from clinical_memory.training import run as driver
from clinical_memory.analysis.report import analyze
from clinical_memory.evaluation.audit import Replay, audit
from clinical_memory.data.augmented import IMAGES, aug_id
from clinical_memory.config import (
    LEARNING,
    RUNS,
    inputs,
    load,
    smoke_run_id,
    test_run_id,
    train_run_id,
)
from clinical_memory.data.split import sizes
from clinical_memory.analysis.notebook import show
from tests.fixtures import (
    COLLECT,
    COLLECT_KEY,
    DIAGNOSE,
    PERMIT,
    REASON,
    WHEN,
    Scripted,
    build_project,
    reset,
    write_augmented,
)

CLEAN = {
    "invalid_actions": 0,
    "invalid_selections": 0,
    "collection_actions": 13,
    "invalid_share": 0.0,
    "visits": 8,
    "invalid_final_answers": 0,
    "presentations_asked": 8,
    "presentations_unusable": 0,
    "presentations_settled": 0,
    "max_share_in_smoke": 0.1,
    "max_unnamed_share_in_smoke": 0.5,
}
ALL_RIGHT = {
    "delta_pp": 100.0,
    "improved": 3,
    "worsened": 0,
    "paired_bootstrap_95_pp": [100.0, 100.0],
}
MOVED = {
    "encounters": 3,
    "collection_changed": 3,
    "image_gained": 3,
    "image_lost": 0,
    "answer_changed": 3,
    "wrong_to_right": 3,
    "right_to_wrong": 0,
}
NAMED = ["--curriculum", "random", "--seed", "42"]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        build_project(self.root)
        reset()
        for item in (
            patch.object(driver, "ROOT", self.root),
            patch.object(driver, "TransformersBackend", Scripted),
            patch.object(make_augmented, "ROOT", self.root),
            patch.dict(os.environ, {"SLURM_JOB_ID": "synthetic"}),
            patch.object(driver.subprocess, "check_output", return_value="0" * 40),
            patch("clinical_memory.preflight.preflight", return_value={"passed": True}),
        ):
            self.stack.enter_context(item)
        self.addCleanup(self.stack.close)
        self.inp = inputs(self.root)
        self.division = self.inp.division

    def calls(self, run_id):
        return [
            json.loads(s)
            for s in (self.root / RUNS / run_id / "private_calls.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]

    def smoke(self, way="key_dx", doctor="4B", *extra):
        run_id = smoke_run_id(doctor, way)
        driver.main(
            [
                "--run-id",
                run_id,
                "--mode",
                "smoke",
                "--doctor",
                doctor,
                "--learning",
                way,
                *extra,
            ]
        )
        return audit(self.root / RUNS / run_id, self.root)

    def smokes(self):
        return {way: self.smoke(way) for way in LEARNING}

    def permit(self, **change):
        atomic_json(self.root / "configs/conditions.json", {**PERMIT, **change})

    def train(self, way, *extra, run_id=None):
        run_id = run_id or train_run_id("4B", way, "random", 42)
        driver.main(
            [
                "--run-id",
                run_id,
                "--mode",
                "train",
                "--doctor",
                "4B",
                "--learning",
                way,
                *NAMED,
                *extra,
            ]
        )
        return audit(self.root / RUNS / run_id, self.root)

    def seen(self, which, step, way=None, *extra):
        """One test set with the notebook of the training run of `way` after `step` training patients; step 0 is the empty notebook."""
        run_id = test_run_id(which, "4B", step, way, way and "random", way and 42)
        named = ["--learning", way, *NAMED] if step else []
        driver.main(
            [
                "--run-id",
                run_id,
                "--mode",
                "test",
                "--doctor",
                "4B",
                "--set",
                which,
                "--step",
                str(step),
                *named,
                *extra,
            ]
        )
        return audit(self.root / RUNS / run_id, self.root)

    def augment(self):
        write_augmented(self.root, self.inp)
        return make_augmented.build(self.root, self.inp)

    def test_the_stages_run_in_order_for_the_three_ways_of_learning_and_both_test_sets(
        self,
    ):
        partners = [
            p
            for c in self.division["smoke"]["test1"]
            for p in self.division["test1_partners"][c]
        ]
        self.assertTrue(set(partners) <= set(self.division["smoke"]["train"]))
        smokes = self.smokes()
        for smoke in smokes.values():
            self.assertEqual(
                (
                    smoke["mode"],
                    smoke["episodes"],
                    smoke["calls_replayed"],
                    smoke["orphan_or_retried_calls"],
                    smoke["reflection_status"],
                    smoke["memory_items"],
                    smoke["episodes_with_lessons"],
                    smoke["lessons_kept_in_usable_form"],
                    smoke["unusable_outputs"],
                ),
                ("smoke", 8, 62, 0, {"stored": 4}, 4, 5, 0, CLEAN),
            )
        self.assertEqual(
            {
                way: (
                    s["answer_sheet_calls"],
                    s["lessons_with_diagnosis_part"],
                    s["episodes_with_notes"],
                )
                for way, s in smokes.items()
            },
            {"self": (0, 0, 0), "key": (4, 4, 0), "key_dx": (4, 4, 3)},
        )
        for call in self.calls(smoke_run_id("4B", "self")):
            text = json.dumps(call["payload"])
            self.assertTrue(
                "answer_key" not in text
                and "confirmed" not in text
                and ("notes" not in call["payload"]),
                call["role"],
            )
        for way in ("key", "key_dx"):
            noted = []
            for call in self.calls(smoke_run_id("4B", way)):
                text = json.dumps(
                    {k: v for k, v in call["payload"].items() if k != "options"}
                )
                self.assertEqual(
                    "answer_key" in call["payload"],
                    call["role"] == "reflection",
                    call["role"],
                )
                if call["role"] != "reflection":
                    self.assertTrue(
                        "your_diagnosis_was_confirmed" not in text
                        and "confirmed_diagnosis" not in text
                        and ("you_did_not" not in text),
                        call["role"],
                    )
                self.assertEqual(DIAGNOSE in text, "notes" in call["payload"])
                if "notes" in call["payload"]:
                    noted.append((call["case_id"], call["role"]))
                    self.assertTrue(
                        all(
                            (
                                n["note"] == DIAGNOSE
                                and any(
                                    (
                                        set(n["diagnosis"].lower().split())
                                        <= set(o.lower().split())
                                        for o in call["payload"]["options"].values()
                                    )
                                )
                                for n in call["payload"]["notes"]
                            )
                        )
                    )
                if call["role"] == "doctor" and "memory" in call["payload"]:
                    self.assertEqual(
                        call["payload"]["memory"],
                        [{"lesson_text": "When: " + WHEN + " Action: " + COLLECT_KEY}]
                        * len(call["payload"]["memory"]),
                    )
            last = "test1_004:"
            expected = sorted(
                (
                    (last + c, role)
                    for c in self.division["smoke"]["test1"]
                    for role in ("reasoning", "final")
                )
            )
            self.assertEqual(
                sorted((n for n in noted if n[0].startswith(last))),
                expected if way == "key_dx" else [],
            )
            self.assertEqual(len(noted), 6 if way == "key_dx" else 0)
        for call in self.calls(smoke_run_id("4B", "key_dx")):
            sampled = call["case_id"].startswith("evolution:") and call[
                "role"
            ].removesuffix("_repair") in {"doctor", "reasoning", "final"}
            self.assertEqual(
                call["generation_settings"]["do_sample"],
                sampled,
                (call["case_id"], call["role"]),
            )
        second = self.division["smoke"]["train"][1]
        self.assertEqual(
            [
                c["role"]
                for c in self.calls(smoke_run_id("4B", "key_dx"))
                if c["case_id"] == "evolution:" + second
            ],
            [
                "presentation",
                "doctor",
                "patient",
                "doctor",
                "measurement",
                "doctor",
                "reasoning",
                "final",
                "reflection",
            ],
        )
        with self.assertRaises(PermissionError):
            self.train("key")
        self.permit(conditions=["4B/key", "4B/key_dx"])
        with self.assertRaises(PermissionError):
            self.train("self")
        with self.assertRaisesRegex(
            ValueError, "must be named " + train_run_id("4B", "key", "random", 42)
        ):
            self.train("key", run_id="training")
        for args, message in (
            (
                ["--mode", "train", "--learning", "key"],
                "takes a way of learning, an order and a seed",
            ),
            (
                ["--mode", "train", "--learning", "key", *NAMED, "--step", "0"],
                "takes a way of learning, an order and a seed",
            ),
            (
                ["--mode", "smoke", "--learning", "key", "--seed", "42"],
                "and nothing else",
            ),
            (
                ["--mode", "smoke", "--learning", "key"],
                "must be named " + smoke_run_id("4B", "key"),
            ),
        ):
            with self.assertRaisesRegex(ValueError, message):
                driver.main(["--run-id", "x", "--doctor", "4B", *args])
        self.permit()
        with self.assertRaisesRegex(ValueError, "completed and audited first"):
            self.seen("test1", 9, "key")
        trained = {way: self.train(way) for way in LEARNING}
        for way, run in trained.items():
            self.assertEqual(
                (
                    run["mode"],
                    run["set"],
                    run["episodes"],
                    run["calls_replayed"],
                    run["reflection_status"],
                    run["memory_items"],
                    run["episodes_with_lessons"],
                    run["answer_sheet_calls"],
                    run["lessons_with_diagnosis_part"],
                ),
                (
                    "train",
                    None,
                    9,
                    7 + 8 * 9,
                    {"stored": 9},
                    9,
                    8,
                    0 if way == "self" else 9,
                    0 if way == "self" else 9,
                ),
            )
        folder = self.root / RUNS / train_run_id("4B", "key_dx", "random", 42)
        seal = load(folder / "seal.json")
        self.assertEqual(
            (
                seal["schedule"],
                seal["initial_memory"],
                seal["training_run"],
                seal["augmentation_sha256"],
                seal["division_sha256"],
                {c["case_id"].split(":")[0] for c in self.calls(folder.name)},
            ),
            (
                [["evolution", c] for c in self.division["orders"]["random"]],
                [],
                None,
                None,
                self.inp.protocol["pool"]["division_sha256"],
                {"evolution"},
            ),
        )
        self.assertEqual(
            load(folder / "provenance.json"),
            {
                "source_commit": "0" * 40,
                "main_basis": {"basis": "configured_conditions", "reason": REASON},
            },
        )
        memory = load(folder / "memory_snapshot.json")["entries"]
        self.assertEqual(
            {
                (e["when"], e["collect"], e["diagnose"], e["model_revision"])
                for e in memory
            },
            {(WHEN, COLLECT_KEY, DIAGNOSE, "rev-4B")},
        )
        self.assertEqual(
            [e["disease"] for e in memory],
            [
                self.inp.golds[c]["options"]["A"]
                for c in self.division["orders"]["random"]
            ],
        )
        self.assertEqual(
            [e["outcome"] for e in memory], ["not confirmed"] + ["confirmed"] * 8
        )
        lesson = "When: " + WHEN + " Action: " + COLLECT_KEY + " Diagnosis: " + DIAGNOSE
        first = self.inp.golds[self.division["orders"]["random"][0]]["options"]["A"]
        self.assertEqual(
            show(folder.name, 1, self.root).splitlines(),
            [
                f"=== {folder.name}: doctor 4B, learning key_dx, 9 lessons, 9 with a diagnosis part",
                "--- first 1 lessons [after training patient N] (own diagnosis for that patient; filed under)",
                f"[1] (not confirmed; {first}) " + lesson,
            ],
        )
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.train("key")
        self.permit(test_authorized=False)
        with self.assertRaisesRegex(PermissionError, "tests wait"):
            self.seen("test1", 0)
        self.permit()
        for message, args in (
            (
                "must be named cm_test1_4b_t000",
                ["--run-id", "x", "--set", "test1", "--step", "0"],
            ),
            (
                "must be a checkpoint of the training: 0, 4, 9",
                [
                    "--run-id",
                    "x",
                    "--set",
                    "test1",
                    "--step",
                    "5",
                    "--learning",
                    "key",
                    *NAMED,
                ],
            ),
            (
                "give no way of learning",
                [
                    "--run-id",
                    test_run_id("test1", "4B", 0),
                    "--set",
                    "test1",
                    "--step",
                    "0",
                    "--learning",
                    "key",
                ],
            ),
            (
                "name the training run",
                ["--run-id", "x", "--set", "test1", "--step", "4"],
            ),
            ("takes --set and --step", ["--run-id", "x", "--step", "0"]),
            (
                "have not been built yet",
                [
                    "--run-id",
                    test_run_id("test2", "4B", 0),
                    "--set",
                    "test2",
                    "--step",
                    "0",
                ],
            ),
        ):
            with self.assertRaisesRegex(ValueError, message):
                driver.main(["--mode", "test", "--doctor", "4B", *args])
        with self.assertRaises(PermissionError):
            driver.main(
                [
                    "--run-id",
                    test_run_id("test1", "9B", 0),
                    "--mode",
                    "test",
                    "--doctor",
                    "9B",
                    "--set",
                    "test1",
                    "--step",
                    "0",
                ]
            )
        before = self.seen("test1", 0)
        after = {way: self.seen("test1", 9, way) for way in LEARNING}
        self.assertEqual(
            (
                before["mode"],
                before["set"],
                before["learning"],
                before["episodes"],
                before["memory_items"],
                before["episodes_with_lessons"],
                before["episodes_with_notes"],
                before["reflection_status"],
                before["answer_sheet_calls"],
            ),
            ("test", "test1", "self", 3, 0, 0, 0, {}, 0),
        )
        for way, run in after.items():
            self.assertEqual(
                (
                    run["set"],
                    run["learning"],
                    run["episodes"],
                    run["memory_items"],
                    run["episodes_with_lessons"],
                    run["episodes_with_notes"],
                    run["reflection_status"],
                    run["answer_sheet_calls"],
                ),
                ("test1", way, 3, 9, 3, 2 if way == "key_dx" else 0, {}, 0),
            )
        last = self.root / RUNS / test_run_id("test1", "4B", 9, "key_dx", "random", 42)
        seal = load(last / "seal.json")
        self.assertEqual(
            (
                load(last / "memory_snapshot.json"),
                seal["training_run"],
                seal["memory_step"],
                seal["schedule"],
                seal["code"],
            ),
            (
                load(folder / "memory_snapshot.json"),
                folder.name,
                9,
                [["test1", c] for c in self.division["test1"]],
                load(folder / "seal.json")["code"],
            ),
        )
        notes = {
            load(p)["trace"]["id"]: load(p)["trace"]
            for p in (last / "episodes").glob("*.json")
        }
        for cid, trace in notes.items():
            partner = self.division["test1_partners"][cid][0]
            found = cid in self.division["test1_reachable"]
            self.assertEqual(
                (
                    trace["notes"],
                    [s["options"] for s in trace["note_lookup"]["selected"]],
                ),
                (
                    [
                        {
                            "diagnosis": self.inp.golds[partner]["options"]["A"],
                            "note": DIAGNOSE,
                        }
                    ],
                    [["A"]],
                )
                if found
                else ([], []),
            )
        self.assertEqual(self.augment()["patients"], 6)
        early, later = (
            self.seen("test2", 0),
            {way: self.seen("test2", 9, way) for way in LEARNING},
        )
        self.assertEqual(
            (
                early["set"],
                early["episodes"],
                early["memory_items"],
                early["episodes_with_lessons"],
            ),
            ("test2", 6, 0, 0),
        )
        for way, run in later.items():
            self.assertEqual(
                (
                    run["set"],
                    run["episodes"],
                    run["memory_items"],
                    run["episodes_with_lessons"],
                    run["episodes_with_notes"],
                    run["answer_sheet_calls"],
                ),
                ("test2", 6, 9, 6, 6 if way == "key_dx" else 0, 0),
            )
        second = (
            self.root / RUNS / test_run_id("test2", "4B", 9, "key_dx", "random", 42)
        )
        seal, ids = (
            load(second / "seal.json"),
            [aug_id(c, 5000) for c in self.division["test2_sources"]],
        )
        self.assertEqual(
            (seal["schedule"], seal["augmentation_sha256"]),
            (
                [["test2", c] for c in ids],
                load(self.root / "results/augmentation.json")["manifest_sha256"],
            ),
        )
        for call in self.calls(second.name):
            text = json.dumps(call["payload"])
            self.assertTrue(
                "I am a 40-year-old man with knee pain." not in text
                and all(
                    (
                        im["name"].startswith("E5") or im["name"].startswith("R5")
                        for im in call["images"]
                    )
                )
            )
        for which in ("test1", "test2"):
            for run_id in (
                test_run_id(which, "4B", 0),
                test_run_id(which, "4B", 9, "key_dx", "random", 42),
            ):
                self.assertFalse(
                    any(
                        (
                            "answer_key" in json.dumps(c["payload"])
                            or c["role"].startswith("reflection")
                            for c in self.calls(run_id)
                        )
                    )
                )
        calls = ScriptedBackend.count
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.seen("test2", 9, "key")
        self.seen("test2", 9, "key", "--resume")
        self.assertEqual(ScriptedBackend.count, calls)
        report = analyze(self.root)
        self.assertEqual(
            (
                report["excluded_runs"],
                len(report["smoke"]),
                len(report["training"]["runs"]),
                len(report["test1"]["runs"]),
                len(report["test2"]["runs"]),
                report["division"],
                report["augmentation"]["patients"],
            ),
            (
                [],
                3,
                3,
                4,
                4,
                {
                    "patients": 12,
                    "train": 9,
                    "test1": 3,
                    "test1_with_a_findable_note": 2,
                    "test2": 6,
                    "sha256": self.inp.protocol["pool"]["division_sha256"],
                },
                6,
            ),
        )
        for which, n in (("test1", 3), ("test2", 6)):
            test = report[which]
            right = {**ALL_RIGHT, "improved": n}
            self.assertEqual(
                [
                    (
                        c["doctor"],
                        c["learning"],
                        c["runs"],
                        c["runs_expected"],
                        c["complete"],
                        [
                            (
                                p["step"],
                                p["accuracy"],
                                p["examination_reach"],
                                p["image_reach"],
                            )
                            for p in c["curve"]
                        ],
                        c["metrics"]["accuracy"],
                    )
                    for c in test["cells"]
                ],
                [
                    ("4B", way, 1, 1, True, [(0, 0, 0, 0), (9, 100, 100, 100)], right)
                    for way in LEARNING
                ],
            )
            self.assertEqual(
                [
                    (
                        c["kind"],
                        c["first"],
                        c["second"],
                        c["pairs"],
                        c["pairs_expected"],
                        c["metrics"]["accuracy"]["delta_pp"],
                    )
                    for c in test["contrasts"]
                ],
                [
                    ("learning", "key_dx", "key", 1, 1, 0),
                    ("learning", "key", "self", 1, 1, 0),
                ],
            )
            self.assertEqual(
                (
                    test["checkpoints"],
                    test["waiting_for_the_empty_notebook"],
                    test["evaluations_between"],
                    [x["learning"] for x in test["lessons"]],
                ),
                ([0, 9], 0, {"4": 0}, list(LEARNING)),
            )
            self.assertEqual(
                sorted(
                    (
                        (
                            t["step"],
                            t["learning"],
                            t["correct"],
                            t["n"],
                            t["with_lessons"],
                            t["with_notes"],
                        )
                        for t in test["runs"]
                    )
                ),
                sorted(
                    [(0, "self", 0, n, 0, 0)]
                    + [
                        (
                            9,
                            way,
                            n,
                            n,
                            n,
                            (2 if which == "test1" else 6) if way == "key_dx" else 0,
                        )
                        for way in LEARNING
                    ]
                ),
            )
        moved = {**MOVED}
        reach = {x["learning"]: x for x in report["test1"]["lessons"]}
        self.assertEqual(
            reach["key_dx"]["delivered"],
            {
                "evaluation_visits": 3,
                "presentation_named": 3,
                "with_gathering_lesson": 3,
                "with_diagnosis_note": 2,
                "note_on_the_right_option": 2,
                "notes_on_wrong_options_only": 0,
                "notes_read": 2,
                "closest_same complaint": 3,
            },
        )
        self.assertEqual(
            (
                reach["key_dx"]["used_with_lesson"],
                reach["key_dx"]["used_with_note_on_the_right_option"]["wrong_to_right"],
                reach["key_dx"]["used_without_note"]["encounters"],
                reach["key"]["delivered"]["with_diagnosis_note"],
                reach["self"]["used_without_lesson"]["encounters"],
            ),
            (moved, 2, 1, 0, 0),
        )
        wide = {x["learning"]: x["delivered"] for x in report["test2"]["lessons"]}[
            "key_dx"
        ]
        self.assertEqual(
            (
                wide["with_diagnosis_note"],
                wide["note_on_the_right_option"],
                wide["notes_on_wrong_options_only"],
                wide["notes_read"],
            ),
            (6, 6, 0, 7),
        )
        written = {x["learning"]: x for x in report["training"]["lessons"]}
        self.assertEqual(
            {
                way: (
                    x["runs"],
                    x["written"]["lessons_stored"],
                    x["written"]["with_diagnosis_part"],
                    x["written"]["kept_in_usable_form"],
                    x["written"]["format_errors"],
                    x["calls"],
                )
                for way, x in written.items()
            },
            {
                way: (
                    1,
                    9,
                    0 if way == "self" else 9,
                    0,
                    {},
                    {"lesson_calls": 9, "rewrite_calls": 0, "token_cap_hits": 0},
                )
                for way in LEARNING
            },
        )
        self.assertEqual(
            [
                (
                    x["doctor"],
                    x["learning"],
                    x["training"]["n"],
                    x["memory_items"],
                    x["main_basis"]["basis"],
                )
                for x in report["training"]["runs"]
            ],
            [
                ("4B", way, 9, 9, "configured_conditions")
                for way in ("key_dx", "key", "self")
            ],
        )
        self.assertEqual(
            {
                x["learning"]: (
                    x["episodes"],
                    x["lessons_stored"],
                    x["test_visits_with_a_note"],
                    x["test_visits"],
                )
                for x in report["smoke"]
            },
            {"self": (8, 4, 0, 2), "key": (8, 4, 0, 2), "key_dx": (8, 4, 2, 2)},
        )
        for word in (
            "knee",
            "radiograph",
            "fracture",
            "arthritis",
            "Condition of patient",
            "hurting",
        ):
            self.assertNotIn(word, json.dumps(report))
        fifth = load(sorted((folder / "episodes").glob("*_evolution_*.json"))[4])
        self.assertEqual(
            (
                MemoryStore(entries=driver.notebook_at(folder, 4, 9)).digest(),
                len(driver.notebook_at(folder, 4, 9)),
            ),
            (fifth["trace"]["memory_before_sha256"], 4),
        )
        with self.assertRaisesRegex(
            ValueError, "not the one the training run held there"
        ):
            driver.notebook_at(folder, 10, 9)
        middle = self.seen("test1", 4, "key_dx")
        self.assertEqual((middle["memory_items"], middle["episodes"]), (4, 3))
        report = analyze(self.root)
        self.assertEqual(
            (report["test1"]["checkpoints"], report["test1"]["evaluations_between"]),
            ([0, 9], {"4": 1}),
        )
        for way in ("self", "key"):
            self.seen("test1", 4, way)
        report = analyze(self.root)
        self.assertEqual(
            (
                report["test1"]["checkpoints"],
                [p["step"] for p in report["test1"]["cells"][0]["curve"]],
            ),
            ([0, 4, 9], [0, 4, 9]),
        )
        halfway = (
            self.root / RUNS / test_run_id("test1", "4B", 4, "key_dx", "random", 42)
        )
        sealed = load(halfway / "seal.json")
        atomic_json(
            halfway / "seal.json",
            {
                **sealed,
                "initial_memory": load(folder / "memory_snapshot.json")["entries"],
            },
        )
        with self.assertRaisesRegex(
            ValueError, "not that of its training run at this checkpoint"
        ):
            audit(halfway, self.root)
        atomic_json(halfway / "seal.json", sealed)
        self.assertTrue(audit(halfway, self.root)["passed"])

    def test_the_audit_refuses_information_outside_the_call_that_may_read_it(self):
        self.smokes()
        run_id = smoke_run_id("4B", "key_dx")
        folder, second = (self.root / RUNS / run_id, self.division["smoke"]["train"][1])
        rows, seal = (self.calls(run_id), load(folder / "seal.json"))

        def row(role, case):
            return next((r for r in rows if r["role"] == role and r["case_id"] == case))

        def names(call):
            return [im["name"] for im in call["images"]]

        case = "evolution:" + second
        naming, doctor, reasoning, final, reflection = (
            row(r, case)
            for r in ("presentation", "doctor", "reasoning", "final", "reflection")
        )
        noted = [
            row(r, "test1_004:" + self.division["smoke"]["test1"][0])
            for r in ("reasoning", "final")
        ]
        images, sheet = (names(reflection), reflection["payload"]["answer_key"])
        self.assertEqual(
            (images, sheet),
            (
                [second + ".jpeg"],
                {
                    "confirmed_diagnosis": self.inp.golds[second]["options"]["A"],
                    "your_diagnosis_was_confirmed": True,
                    "history_you_did_not_hear": [],
                    "examinations_you_did_not_receive": [],
                },
            ),
        )
        note = noted[0]["payload"]["notes"]
        self.assertEqual(
            (len(note), set(note[0]), "notes" in reasoning["payload"]),
            (1, {"diagnosis", "note"}, False),
        )

        def check(
            learning,
            role,
            record,
            pictures,
            response="{}",
            decided=True,
            source=(reasoning, final),
            phase="evolution",
        ):
            replay = Replay(rows, {**seal, "learning": learning}, self.root)
            replay.start_episode(phase, second, 0)
            if decided:
                replay.reasoned = (
                    source[0]["payload"],
                    names(source[0]),
                    source[0]["response"].strip(),
                )
                if role != "final":
                    replay.final = (source[1]["payload"], names(source[1]))
            replay.boundaries(
                role, role.removesuffix("_repair"), record, pictures, response
            )
            return replay

        self.assertEqual(
            check("key_dx", "reflection", reflection["payload"], images).shown, [sheet]
        )
        for role, recorded, pictures in (
            ("presentation", naming, []),
            ("doctor", doctor, []),
            ("reasoning", reasoning, images),
            ("final", final, images),
        ):
            check("key_dx", role, recorded["payload"], pictures, recorded["response"])
        check(
            "key_dx",
            "reasoning",
            noted[0]["payload"],
            names(noted[0]),
            noted[0]["response"],
            phase="test1_004",
        )
        check(
            "key_dx",
            "final",
            noted[1]["payload"],
            names(noted[1]),
            noted[1]["response"],
            source=noted,
            phase="test1_004",
        )
        own = {k: v for k, v in reflection["payload"].items() if k != "answer_key"}
        for learning, role, record, pictures, message in (
            (
                "key_dx",
                "doctor",
                {**doctor["payload"], "answer_key": sheet},
                [],
                "answer sheet reached a role other than reflection",
            ),
            (
                "key_dx",
                "doctor",
                {**doctor["payload"], "confirmed_diagnosis": "x"},
                [],
                "answer sheet reached a role other than reflection",
            ),
            (
                "key_dx",
                "doctor",
                {**doctor["payload"], "options": {}},
                [],
                "Options leaked",
            ),
            (
                "key_dx",
                "doctor",
                {**doctor["payload"], "notes": note},
                [],
                "diagnosis note reached a call",
            ),
            (
                "key_dx",
                "doctor",
                {**doctor["payload"], "guidance": "x"},
                [],
                "no global guidance",
            ),
            (
                "key_dx",
                "presentation",
                {**naming["payload"], "history": []},
                [],
                "named from the opening statement only",
            ),
            ("key_dx", "presentation", naming["payload"], images, "reads no image"),
            (
                "key_dx",
                "reasoning",
                {**reasoning["payload"], "memory": []},
                images,
                "other than the collecting doctor",
            ),
            (
                "key_dx",
                "reasoning",
                {**reasoning["payload"], "state": "awaiting_final"},
                images,
                "evidence view and at most the notes",
            ),
            (
                "key_dx",
                "reasoning",
                {**reasoning["payload"], "answer_key": sheet},
                images,
                "answer sheet reached a role other than reflection",
            ),
            (
                "key_dx",
                "reasoning",
                {**reasoning["payload"], "notes": [DIAGNOSE]},
                images,
                "nothing else",
            ),
            (
                "key_dx",
                "reasoning",
                {**reasoning["payload"], "notes": []},
                images,
                "nothing else",
            ),
            (
                "key_dx",
                "reasoning",
                {**reasoning["payload"], "notes": [{**note[0], "lesson_id": "L1"}]},
                images,
                "nothing else",
            ),
            (
                "key",
                "reasoning",
                noted[0]["payload"],
                names(noted[0]),
                "diagnosis note reached a call",
            ),
            (
                "self",
                "final",
                noted[1]["payload"],
                names(noted[1]),
                "diagnosis note reached a call",
            ),
            (
                "key_dx",
                "final",
                {**final["payload"], "answer_key": sheet},
                images,
                "answer sheet reached a role other than reflection",
            ),
            (
                "key_dx",
                "final",
                {**final["payload"], "reasoning": "Something the doctor never wrote."},
                images,
                "what the reasoning read",
            ),
            (
                "key_dx",
                "final",
                {**final["payload"], "history": []},
                images,
                "what the reasoning read",
            ),
            (
                "key_dx",
                "final",
                {**final["payload"], "notes": note},
                images,
                "what the reasoning read",
            ),
            ("key_dx", "final", final["payload"], [], "what the reasoning read"),
            (
                "self",
                "reflection",
                reflection["payload"],
                images,
                "Unexpected reflection input",
            ),
            ("key", "reflection", own, images, "Unexpected reflection input"),
            (
                "key_dx",
                "reflection",
                {**reflection["payload"], "options": {}},
                images,
                "Options leaked",
            ),
            (
                "key_dx",
                "reflection",
                {**reflection["payload"], "notes": note},
                images,
                "diagnosis note reached a call",
            ),
            (
                "key_dx",
                "reflection",
                reflection["payload"],
                [],
                "differs from what the doctor saw",
            ),
            (
                "key_dx",
                "reflection",
                {**reflection["payload"], "reasoning": "Another reasoning."},
                images,
                "differs from what the doctor saw",
            ),
            (
                "key_dx",
                "reflection",
                {**reflection["payload"], "answer_key": {**sheet, "options": {}}},
                images,
                "Unexpected answer sheet",
            ),
            (
                "key_dx",
                "reflection",
                reflection["payload"],
                images + ["E1.jpeg"],
                "images of the answer sheet",
            ),
            (
                "self",
                "reflection",
                own,
                images + ["E1.jpeg"],
                "sees only the images the doctor received",
            ),
        ):
            with self.assertRaisesRegex(
                ValueError, message, msg=(learning, role, message)
            ):
                check(learning, role, record, pictures)
        self.assertEqual(check("self", "reflection", own, images).shown, [])
        with self.assertRaisesRegex(ValueError, "outside a training visit"):
            check(
                "key_dx", "reflection", reflection["payload"], images, phase="test1_004"
            )
        changed = [
            dict(
                r,
                payload={
                    **r["payload"],
                    "answer_key": {
                        **r["payload"]["answer_key"],
                        "confirmed_diagnosis": "Another disease",
                    },
                },
            )
            if r is reflection
            else r
            for r in rows
        ]
        (folder / "private_calls.jsonl").write_text(
            "".join((json.dumps(r) + "\n" for r in changed)), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "Replay mismatch"):
            audit(folder, self.root)
        other = self.root / RUNS / smoke_run_id("4B", "key")
        atomic_json(
            other / "seal.json", {**load(other / "seal.json"), "learning": "self"}
        )
        with self.assertRaises(ValueError):
            audit(other, self.root)

    def test_an_interrupted_run_resumes_from_the_journal_and_refuses_changed_code(self):
        ScriptedBackend.fail_at = 22
        with self.assertRaisesRegex(RuntimeError, "simulated crash"):
            self.smoke()
        folder = self.root / RUNS / smoke_run_id("4B", "key_dx")
        failed = load(folder / "summary.json")
        self.assertEqual(
            (failed["status"], failed["completed_episodes"], failed["memory_items"]),
            ("failed", 3, 1),
        )
        with self.assertRaises(ValueError):
            audit(folder, self.root)
        ScriptedBackend.fail_at = None
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.smoke()
        result = self.smoke("key_dx", "4B", "--resume")
        self.assertEqual(
            (
                result["episodes"],
                result["memory_items"],
                result["orphan_or_retried_calls"],
                result["answer_sheet_calls"],
            ),
            (8, 4, 2, 4),
        )
        memory = (folder / "memory.jsonl").read_bytes()
        (folder / "memory.jsonl").unlink()
        (folder / "summary.json").unlink()
        calls = ScriptedBackend.count
        self.smoke("key_dx", "4B", "--resume")
        self.assertEqual(
            (ScriptedBackend.count, (folder / "memory.jsonl").read_bytes()),
            (calls, memory),
        )
        (self.root / "clinical_memory/memory/diagnosis.py").write_text(
            "changed after the run", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "configuration/data/code changed"):
            self.smoke("key_dx", "4B", "--resume")
        with self.assertRaisesRegex(ValueError, "differs from the sealed run"):
            audit(folder, self.root)

    def test_a_training_run_longer_than_the_old_notebook_is_interrupted_and_resumed(
        self,
    ):
        self.smoke("key")
        self.permit()
        ScriptedBackend.fail_at = ScriptedBackend.count + 30
        with self.assertRaisesRegex(RuntimeError, "simulated crash"):
            self.train("key")
        ScriptedBackend.fail_at = None
        run = self.train("key", "--resume")
        self.assertEqual(
            (run["episodes"], run["memory_items"], run["orphan_or_retried_calls"] > 0),
            (9, 9, True),
        )
        folder = self.root / RUNS / train_run_id("4B", "key", "random", 42)
        full = MemoryStore(
            entries=load(folder / "memory_snapshot.json")["entries"], max_lessons=9
        )
        with self.assertRaisesRegex(ValueError, "capacity exceeded"):
            full.add({**full.entries()[0], "source_case_id": "X1", "lesson_id": "L10"})
        snapshot = load(folder / "memory_snapshot.json")
        atomic_json(
            folder / "memory_snapshot.json",
            {**snapshot, "entries": snapshot["entries"][:8]},
        )
        with self.assertRaisesRegex(ValueError, "Training memory changed"):
            self.seen("test1", 9, "key")
        with self.assertRaisesRegex(ValueError, "Final memory mismatch"):
            audit(folder, self.root)

    def test_rare_unusable_outputs_do_not_fail_a_readiness_check_but_frequent_ones_do(
        self,
    ):
        test1, taught = (
            self.division["smoke"]["test1"],
            self.division["smoke"]["train"],
        )
        Scripted.unusable_action_for = frozenset({"test1_000:" + test1[0]})
        Scripted.unnamed_for = frozenset({"evolution:" + taught[2]})
        Scripted.loose_for = frozenset({"evolution:" + taught[3]})
        passed = self.smoke()
        self.assertEqual(
            passed["unusable_outputs"],
            {
                **CLEAN,
                "invalid_actions": 1,
                "collection_actions": 12,
                "invalid_share": 1 / 12,
                "presentations_unusable": 1,
                "presentations_settled": 1,
            },
        )
        self.assertEqual(
            (passed["reflection_status"], passed["memory_items"]),
            ({"stored": 3, "no_presentation": 1}, 3),
        )
        reset()
        Scripted.unnamed_for = frozenset(
            ("evolution:" + c for c in taught)
        ) | frozenset(
            (f"{phase}:{c}" for phase in ("test1_000", "test1_004") for c in test1)
        )
        with self.assertRaisesRegex(
            ValueError, "too many unusable outputs: presentations"
        ):
            self.smoke("key")
        self.assertFalse(
            (self.root / RUNS / smoke_run_id("4B", "key") / "audit.json").exists()
        )
        reset()
        self.permit()
        self.assertEqual(self.train("key_dx")["episodes"], 9)
        with self.assertRaisesRegex(
            ValueError, "readiness check of the same code, doctor and way of learning"
        ):
            self.train("key")
        report = analyze(self.root)
        self.assertEqual(
            [x["run_id"] for x in report["excluded_runs"]], [smoke_run_id("4B", "key")]
        )
        self.assertEqual(
            [
                (
                    x["presentations_unusable"],
                    x["presentations_settled"],
                    x["lessons_stored"],
                )
                for x in report["smoke"]
            ],
            [(1, 1, 3)],
        )

    def test_every_role_of_the_doctor_runs_on_the_doctor_model(self):
        result = self.smoke("key_dx", "9B")
        self.assertEqual(
            (result["doctor"], result["learning"], result["reflection_status"]),
            ("9B", "key_dx", {"stored": 4}),
        )
        self.assertEqual(
            {
                (c["role"], c["model_id"])
                for c in self.calls(smoke_run_id("9B", "key_dx"))
            },
            {
                ("presentation", "model-9B"),
                ("doctor", "model-9B"),
                ("reasoning", "model-9B"),
                ("final", "model-9B"),
                ("reflection", "model-9B"),
                ("patient", "model-4B"),
                ("measurement", "model-9B"),
            },
        )
        small = self.smoke("self", "2B")
        self.assertEqual(
            {
                c["model_id"]
                for c in self.calls(smoke_run_id("2B", "self"))
                if c["role"]
                in {"presentation", "doctor", "reasoning", "final", "reflection"}
            },
            {"model-2B"},
        )
        self.assertEqual((small["memory_items"], small["answer_sheet_calls"]), (4, 0))
        self.permit(conditions=["2B/self", "4B/key"])
        with self.assertRaisesRegex(ValueError, "readiness check of this doctor"):
            driver.main(
                [
                    "--run-id",
                    test_run_id("test1", "4B", 0),
                    "--mode",
                    "test",
                    "--doctor",
                    "4B",
                    "--set",
                    "test1",
                    "--step",
                    "0",
                ]
            )
        driver.main(
            [
                "--run-id",
                test_run_id("test1", "2B", 0),
                "--mode",
                "test",
                "--doctor",
                "2B",
                "--set",
                "test1",
                "--step",
                "0",
            ]
        )
        self.assertEqual(
            audit(self.root / RUNS / test_run_id("test1", "2B", 0), self.root)[
                "doctor"
            ],
            "2B",
        )

    def test_long_lessons_and_format_variations_are_kept_delivered_and_counted(self):
        long = " ".join(["word"] * 60)
        Scripted.reflection = {"collect": COLLECT_KEY + " " + long, "diagnose": long}
        kept = self.smoke("key_dx")
        self.assertEqual(
            (
                kept["reflection_status"],
                kept["memory_items"],
                kept["lessons_kept_in_usable_form"],
                kept["answer_sheet_calls"],
            ),
            ({"stored": 4}, 4, 0, 4),
        )
        entries = load(
            self.root / RUNS / smoke_run_id("4B", "key_dx") / "memory_snapshot.json"
        )["entries"]
        self.assertEqual({e["diagnose"] for e in entries}, {long})
        delivered = [
            m["lesson_text"]
            for c in self.calls(smoke_run_id("4B", "key_dx"))
            for m in c["payload"].get("memory", [])
        ]
        self.assertTrue(delivered)
        self.assertTrue(all((m.endswith(COLLECT_KEY + " " + long) for m in delivered)))
        notes = [
            n
            for c in self.calls(smoke_run_id("4B", "key_dx"))
            for n in c["payload"].get("notes", [])
        ]
        self.assertTrue(notes)
        self.assertTrue(all((n["note"] == long for n in notes)))
        self.assertTrue(
            all(
                (
                    c["generation_settings"]["max_new_tokens"] == 1024
                    for c in self.calls(smoke_run_id("4B", "key_dx"))
                    if c["role"] == "reflection"
                )
            )
        )
        Scripted.reflection = {"collect": COLLECT, "diagnose": DIAGNOSE}
        own = self.smoke("self")
        self.assertEqual(
            (
                own["reflection_status"],
                own["lessons_kept_in_usable_form"],
                own["lessons_with_diagnosis_part"],
            ),
            ({"stored": 4}, 4, 0),
        )
        Scripted.reflection = ["not", "a", "lesson"]
        with self.assertRaisesRegex(ValueError, "Smoke stored no lesson"):
            self.smoke("key")
        Scripted.reflection = None
        self.permit()
        Scripted.rewrite_for = frozenset(
            {"evolution:" + self.division["orders"]["random"][3]}
        )
        self.train("key_dx")
        Scripted.reflection = {
            "collect": "Choose option B after the radiograph.",
            "diagnose": DIAGNOSE,
        }
        self.train("self")
        report = analyze(self.root)
        written = {x["learning"]: x["written"] for x in report["training"]["lessons"]}
        self.assertEqual(
            {
                way: (
                    w["lessons_stored"],
                    w["stored_after_rewrite"],
                    w["kept_in_usable_form"],
                    w["rules_broken_by_kept_lessons"],
                    w["changes_to_kept_lessons"],
                    w["format_errors"],
                )
                for way, w in written.items()
            },
            {
                "key_dx": (
                    9,
                    0,
                    1,
                    {},
                    {"collect lost a case identifier or an answer letter": 1},
                    {},
                ),
                "self": (
                    9,
                    0,
                    9,
                    {},
                    {
                        "a key that was not asked for was ignored": 9,
                        "collect lost a case identifier or an answer letter": 9,
                    },
                    {},
                ),
            },
        )
        self.assertEqual(
            {x["learning"]: x["calls"] for x in report["training"]["lessons"]},
            {
                "key_dx": {"lesson_calls": 9, "rewrite_calls": 0, "token_cap_hits": 0},
                "self": {"lesson_calls": 9, "rewrite_calls": 0, "token_cap_hits": 0},
            },
        )
        protocol = load(self.root / "configs/experiment.json")
        protocol["memory"]["lesson_format_policy"] = "discard"
        atomic_json(self.root / "configs/experiment.json", protocol)
        with self.assertRaisesRegex(ValueError, "Smoke stored no lesson"):
            driver.main(
                [
                    "--run-id",
                    smoke_run_id("2B", "self"),
                    "--mode",
                    "smoke",
                    "--doctor",
                    "2B",
                    "--learning",
                    "self",
                ]
            )
            audit(self.root / RUNS / smoke_run_id("2B", "self"), self.root)

    def test_a_lesson_a_notebook_or_an_augmented_patient_changed_after_the_run_does_not_replay(
        self,
    ):
        self.smokes()
        folder, second = (
            self.root / RUNS / smoke_run_id("4B", "key_dx"),
            self.division["smoke"]["train"][1],
        )
        path = next(
            iter(sorted((folder / "episodes").glob(f"*_evolution_{second}.json")))
        )
        record = load(path)
        self.assertEqual(
            (
                record["lesson"]["diagnose"],
                record["lesson"]["outcome"],
                record["reflection"],
            ),
            (DIAGNOSE, "confirmed", {"status": "stored", "repairs": 0}),
        )
        for change in (
            {"diagnose": "A note the doctor never wrote."},
            {"disease": "Another disease"},
        ):
            atomic_json(path, {**record, "lesson": {**record["lesson"], **change}})
            with self.assertRaisesRegex(
                ValueError, "Reflection or memory write mismatch"
            ):
                audit(folder, self.root)
        atomic_json(path, record)
        self.assertTrue(audit(folder, self.root)["passed"])
        self.permit()
        self.train("key_dx")
        self.augment()
        self.seen("test2", 0)
        self.seen("test2", 9, "key_dx")
        run = self.root / RUNS / test_run_id("test2", "4B", 9, "key_dx", "random", 42)
        self.assertTrue(audit(run, self.root)["passed"])
        write_augmented(
            self.root,
            self.inp,
            {
                c: "I am 40, a man, and my knee has hurt since I fell."
                for c in self.division["test2_sources"]
            },
        )
        make_augmented.build(self.root, self.inp)
        with self.assertRaisesRegex(ValueError, "not those of the sealed run"):
            audit(run, self.root)
        self.augment()
        self.assertTrue(audit(run, self.root)["passed"])
        name = aug_id(self.division["test2_sources"][0], 5000) + ".jpeg"
        (self.root / IMAGES / name).write_bytes(b"another image")
        with self.assertRaisesRegex(
            ValueError, "missing, changed or the source image itself"
        ):
            audit(run, self.root)

    def test_the_committed_decision_file_and_job_scripts_name_what_exists(self):
        real = Path(__file__).resolve().parents[1]
        permit, protocol = (
            load(real / "configs/conditions.json"),
            load(real / "configs/experiment.json"),
        )
        self.assertEqual(protocol["training_seeds"], [42, 43, 44])
        self.assertEqual((driver.EMPTY["seed"], driver.SMOKE["seed"]), (42, 42))
        known = {
            doctor + "/" + way for doctor in protocol["doctors"] for way in LEARNING
        }
        self.assertTrue(
            type(permit["authorized"]) is bool
            and type(permit["test_authorized"]) is bool
        )
        self.assertTrue(
            set(permit["conditions"]) <= known
            and len(set(permit["conditions"])) == len(permit["conditions"]),
            permit["conditions"],
        )
        if permit["authorized"]:
            self.assertTrue(
                permit["conditions"] and permit["reason"].strip(),
                "An authorization names its conditions and gives the reason",
            )
        else:
            self.assertFalse(
                permit["test_authorized"],
                "The tests are not opened before the training",
            )
        for name in ("train.sbatch", "test.sbatch"):
            job = (real / "scripts/slurm" / name).read_text(encoding="utf-8")
            for line in (
                'curriculum="${curricula[$((SLURM_ARRAY_TASK_ID / 3))]}"',
                'seed="${seeds[$((SLURM_ARRAY_TASK_ID % 3))]}"',
                "curricula=(random easy_to_hard hard_to_easy)",
                "seeds=(42 43 44)",
                "--gres=gpu:a100:1",
            ):
                self.assertIn(line, job, name)
        last = protocol["main"]["checkpoints"][-1]
        self.assertIn(
            f"--step {last}",
            (real / "scripts/slurm/train.sbatch").read_text(encoding="utf-8"),
        )
        for name in (
            "gate.sbatch",
            "smoke.sbatch",
            "augment.sbatch",
            "report.sbatch",
            "audit.sbatch",
        ):
            self.assertTrue(
                (real / "scripts/slurm" / name)
                .read_text(encoding="utf-8")
                .startswith("#!/bin/bash\n#SBATCH"),
                name,
            )

    def test_the_plan_prints_the_size_of_every_stage(self):
        found = sizes(self.root)
        self.assertEqual(
            (
                found["patients"],
                found["training_patients"],
                found["test1_patients"],
                found["test1_with_a_same_disease_training_patient"],
                found["test1_note_on_the_right_option_can_be_found"],
                found["test2_patients"],
                found["test2_images"],
                found["test2_built"],
                found["visits"],
                found["training_runs"],
                found["evaluations_per_test_set"],
                found["ways_of_learning"],
            ),
            (
                12,
                9,
                3,
                3,
                2,
                6,
                6,
                False,
                {"smoke": 8, "train": 9, "test1": 3, "test2": 6},
                9,
                12,
                list(LEARNING),
            ),
        )


if __name__ == "__main__":
    unittest.main()
