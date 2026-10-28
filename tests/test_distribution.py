import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from clinical_memory.data.prepare import verified_write
from clinical_memory.simulation.replay import load_replay

ROOT = Path(__file__).resolve().parents[1]


class DistributionTests(unittest.TestCase):
    def test_replay_assets_match_manifest_and_complete_model_cohorts(self):
        folder = ROOT / "resources/replay"
        manifest = json.loads((folder / "manifest.json").read_text())
        for record in manifest["records"]:
            self.assertEqual(
                hashlib.sha256((ROOT / record["file"]).read_bytes()).hexdigest(),
                record["sha256"],
            )
        for name, digest in manifest["images"].items():
            self.assertEqual(
                hashlib.sha256((folder / "images" / name).read_bytes()).hexdigest(),
                digest,
            )
        replay = load_replay()
        self.assertFalse(replay.warnings)
        self.assertEqual(
            {m: len(rows) for m, rows in replay.encounters.items()},
            {"2B": 34, "4B": 34, "9B": 34},
        )
        self.assertEqual(
            len({r.case_id for rows in replay.encounters.values() for r in rows}), 34
        )
        self.assertTrue(
            all(
                (
                    a.path
                    for rows in replay.encounters.values()
                    for r in rows
                    for m in r.messages
                    for a in m.images
                )
            )
        )

    def test_curated_input_checksums_and_membership(self):
        folder = ROOT / "resources/cohort"
        checks = json.loads((folder / "checksums.json").read_text())
        for name, digest in checks["files"].items():
            self.assertEqual(
                hashlib.sha256((folder / name).read_bytes()).hexdigest(), digest
            )
        manifest = json.loads((folder / "manifest.json").read_text())
        patients = set(manifest["patients"])
        self.assertEqual(len(patients), 336)
        for name in ["cases.jsonl", "private_gold.jsonl", "difficulty.jsonl"]:
            self.assertEqual(
                {
                    json.loads(line)["id"]
                    for line in (folder / name).read_text(encoding="utf-8").splitlines()
                },
                patients,
            )
        division = json.loads((folder / "division.json").read_text())
        self.assertFalse(set(division["train"]) & set(division["test1"]))
        self.assertEqual(set(division["train"]) | set(division["test1"]), patients)
        self.assertTrue(set(division["test2_sources"]) <= set(division["train"]))

    def test_materialization_accepts_identical_bytes_and_rejects_changed_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested/input.json"
            verified_write(path, b"original")
            verified_write(path, b"original")
            with self.assertRaisesRegex(ValueError, "overwrite changed input"):
                verified_write(path, b"changed")
            self.assertEqual(path.read_bytes(), b"original")


if __name__ == "__main__":
    unittest.main()
