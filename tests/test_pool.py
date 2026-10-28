from contextlib import ExitStack
import json
import re
import tempfile
import unittest
from pathlib import Path
from clinical_memory.data.io import read_rows
from clinical_memory.memory.store import atomic_json
from clinical_memory.config import inputs, load, validate_protocol
from clinical_memory.data.pool import digest, divide, key, share
from tests.fixtures import GROUPS, build_project

REAL = Path(__file__).resolve().parents[1]


class DivisionTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        build_project(self.root)
        self.addCleanup(self.stack.close)

    def parts(self):
        protocol, manifest = (
            load(self.root / "configs/experiment.json"),
            load(self.root / "data/frozen/manifest.json"),
        )
        golds = read_rows(self.root / "data/cohort/private_gold.jsonl")
        levels = {
            cid: row["correct_count"]
            for cid, row in read_rows(
                self.root / "data/cohort/difficulty.jsonl"
            ).items()
        }
        return (manifest, golds, levels, protocol)

    def test_every_patient_is_used_and_a_test_patient_keeps_a_same_disease_training_patient(
        self,
    ):
        inp = inputs(self.root)
        d, pool = (
            inp.division,
            [c for part in inp.frozen["splits"].values() for c in part],
        )
        train, test1 = (set(d["train"]), set(d["test1"]))
        self.assertEqual(
            (len(pool), len(train), len(test1), train & test1, train | test1),
            (12, 9, 3, set(), set(pool)),
        )
        for cid in d["test1"]:
            group = next((g for g in GROUPS if cid in g))
            self.assertTrue(
                d["test1_partners"][cid]
                and set(d["test1_partners"][cid]) <= train & set(group),
                cid,
            )
        self.assertEqual(sorted((len(test1 & set(g)) for g in GROUPS)), [1, 1, 1])
        self.assertEqual(
            (
                {
                    next((i for i, g in enumerate(GROUPS) if c in g))
                    for c in d["test1_reachable"]
                },
                len(d["test1_reachable"]),
            ),
            ({0, 1}, 2),
        )
        self.assertTrue(
            set(d["test2_sources"]) <= train
            and len(d["test2_sources"]) == len(set(d["test2_sources"])) == 6
        )
        for order in d["orders"].values():
            self.assertEqual((sorted(order), len(order)), (sorted(train), 9))
        self.assertEqual(
            d["orders"]["random"], sorted(train, key=lambda c: key(42, "order", c))
        )
        smoke = d["smoke"]
        self.assertEqual(
            (
                smoke["test1"],
                len(smoke["train"]),
                len(set(smoke["train"])),
                set(smoke["train"]) <= train,
            ),
            (d["test1_reachable"][:2], 4, 4, True),
        )
        self.assertTrue(
            all(
                (
                    any((p in smoke["train"] for p in d["test1_partners"][c]))
                    for c in smoke["test1"]
                )
            )
        )
        self.assertEqual(digest(d), inp.protocol["pool"]["division_sha256"])

    def test_the_division_is_the_same_wherever_it_is_computed_and_changes_with_its_inputs(
        self,
    ):
        manifest, golds, levels, protocol = self.parts()
        first, again = (
            divide(manifest, golds, levels, GROUPS, protocol),
            divide(manifest, golds, levels, [g[::-1] for g in GROUPS][::-1], protocol),
        )
        self.assertEqual(
            (digest(first), first["test1"]), (digest(again), again["test1"])
        )
        self.assertNotEqual(
            digest(
                divide(
                    manifest,
                    golds,
                    levels,
                    GROUPS,
                    {**protocol, "pool": {**protocol["pool"], "seed": 7}},
                )
            ),
            digest(first),
        )
        levels = {**levels, "E4": 0, "E5": 1}
        orders = divide(manifest, golds, levels, GROUPS, protocol)["orders"]
        self.assertEqual(
            (orders["easy_to_hard"][-2:], orders["hard_to_easy"][:2]),
            (["E5", "E4"], ["E4", "E5"]),
        )
        self.assertEqual(
            [c for c in orders["easy_to_hard"] if levels[c] == 3],
            [c for c in orders["random"] if levels[c] == 3],
        )
        atomic_json(
            self.root / "configs/disease_groups.json",
            {"groups": [["E1", "T1"], ["E2", "T2"], ["E4", "D1"]]},
        )
        with self.assertRaisesRegex(ValueError, "not the one the protocol pins"):
            inputs(self.root)

    def test_groups_and_sizes_that_do_not_fit_are_refused(self):
        manifest, golds, levels, protocol = self.parts()
        for groups in (
            [["E1", "T1"], ["E1", "T2"], ["E3", "D1"]],
            [["E1", "T1"], ["E2", "T2"], ["E3", "X9"]],
            [["E1", "T1"], ["E2", "T2"], ["E3"]],
        ):
            with self.assertRaisesRegex(ValueError, "groups of same-disease patients"):
                divide(manifest, golds, levels, groups, protocol)
        with self.assertRaisesRegex(ValueError, "stated number of test patients"):
            divide(manifest, golds, levels, GROUPS[:2], protocol)
        self.assertEqual(
            [share(0.1, 336), share(0.5, 336), share(0.25, 12)], [34, 168, 3]
        )
        for change in ({"test1": 4}, {"train": 8}, {"test2": 5}):
            with self.assertRaisesRegex(ValueError, "sizes of the division"):
                validate_protocol({**protocol, "pool": {**protocol["pool"], **change}})
        with self.assertRaisesRegex(ValueError, "sizes of the division"):
            validate_protocol(
                {**protocol, "memory": {**protocol["memory"], "max_lessons": 8}}
            )
        with self.assertRaisesRegex(ValueError, "checkpoints"):
            validate_protocol(
                {**protocol, "main": {**protocol["main"], "checkpoints": [0, 4, 8]}}
            )

    def test_the_committed_protocol_and_groups_describe_the_division_the_user_decided(
        self,
    ):
        protocol, groups = (
            load(REAL / "configs/experiment.json"),
            load(REAL / "configs/disease_groups.json")["groups"],
        )
        validate_protocol(protocol)
        spec = protocol["pool"]
        self.assertEqual(
            (
                spec["patients"],
                spec["test1"],
                spec["train"],
                spec["test2"],
                protocol["memory"]["max_lessons"],
                protocol["main"]["checkpoints"],
                protocol["memory"]["notes_threshold"],
                protocol["memory"]["lesson_format_policy"],
            ),
            (336, 34, 302, 168, 302, [0, 151, 302], 0.6, "settle"),
        )
        self.assertRegex(spec["division_sha256"], "^[0-9a-f]{64}$")
        members = [c for g in groups for c in g]
        self.assertEqual(
            (len(groups), len(members), len(set(members)), len(members) - len(groups)),
            (30, 68, 68, 38),
        )
        self.assertTrue(all((re.fullmatch("MM-\\d+", c) for c in members)))
        self.assertEqual(
            set(
                json.loads(
                    (REAL / "configs/disease_groups.json").read_text(encoding="utf-8")
                )
            ),
            {"version", "groups"},
        )


if __name__ == "__main__":
    unittest.main()
