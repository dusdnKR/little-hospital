import json
from pathlib import Path
from clinical_memory.data.io import read_rows
from clinical_memory.memory.store import atomic_json
import tests.base_fixtures as base_fixtures
from clinical_memory.data.inputs import sha
from tests.reply_fixtures import (
    COLLECT,
    COLLECT_KEY,
    DIAGNOSE,
    LOOSE_WHEN,
    NAMED,
    WHEN,
    Scripted,
    reset,
)
from clinical_memory.data.augmented import IMAGES, aug_id, aug_image
from clinical_memory.config import RUNTIME_FILES
from clinical_memory.data.pool import digest, divide

GROUPS = [["E1", "T1"], ["E2", "T2"], ["E3", "D1"]]
DISEASES = {
    "E1": "Tibial plateau fracture",
    "T1": "Tibial plateau fracture",
    "E2": "Septic arthritis",
    "T2": "Acute septic arthritis",
    "E3": "Jumper knee",
    "D1": "Patellar tendinopathy",
}
REWRITTEN = "My knee has been hurting for a while; I am a man, 40 years old."
REASON = "Enable all learning conditions for the synthetic test fixture"
PERMIT = {
    "authorized": True,
    "conditions": ["4B/self", "4B/key", "4B/key_dx"],
    "reason": REASON,
    "test_authorized": True,
}


def gold_for(cid):
    """One confirmed diagnosis per synthetic patient; a wrong option names the disease of another patient."""
    name = DISEASES.get(cid, "Condition of patient " + cid)
    return {
        "id": cid,
        "label": "A",
        "options": {
            "A": name,
            "B": "Tibial plateau fracture" if cid == "E5" else "Sprain " + cid,
            "C": "Contusion " + cid,
            "D": "Bursitis " + cid,
            "E": "Tendon tear " + cid,
        },
    }


def build_project(root):
    """Twelve synthetic patients in one pool: nine training patients, three test 1 patients, six augmented for test 2."""
    root = Path(root)
    splits = base_fixtures.build_project(root)
    cohort = root / "data/cohort"
    ids = [c for part in splits.values() for c in part]
    (cohort / "private_gold.jsonl").write_text(
        "".join((json.dumps(gold_for(c)) + "\n" for c in ids)), encoding="utf-8"
    )
    manifest = json.loads(
        (root / "data/frozen/manifest.json").read_text(encoding="utf-8")
    )
    manifest["source_hashes"]["data/cohort/private_gold.jsonl"] = sha(
        cohort / "private_gold.jsonl"
    )
    atomic_json(root / "data/frozen/manifest.json", manifest)
    for folder in (
        "clinical_memory",
        "clinical_memory/memory",
        "clinical_memory/models",
        "clinical_memory/environment",
        "clinical_memory/memory",
    ):
        (root / folder).mkdir(parents=True, exist_ok=True)
    protocol = json.loads(
        (base_fixtures.REAL / "configs/experiment.json").read_text(encoding="utf-8")
    )
    protocol.update(
        training_seeds=[42],
        curricula=["random"],
        dataset={"manifest_sha256": sha(root / "data/frozen/manifest.json")},
    )
    protocol["pool"].update(
        patients=12, test1_share=0.25, test1=3, test2_share=0.5, test2=6, train=9
    )
    protocol["memory"]["max_lessons"] = 9
    protocol["smoke"].update(train=4, test1=2)
    protocol["main"]["checkpoints"] = [0, 4, 9]
    atomic_json(
        root / "configs/disease_groups.json", {"version": "synthetic", "groups": GROUPS}
    )
    levels = {
        cid: row["correct_count"]
        for cid, row in read_rows(cohort / "difficulty.jsonl").items()
    }
    protocol["pool"]["division_sha256"] = digest(
        divide(
            manifest, read_rows(cohort / "private_gold.jsonl"), levels, GROUPS, protocol
        )
    )
    atomic_json(root / "configs/experiment.json", protocol)
    atomic_json(
        root / "configs/conditions.json",
        {"authorized": False, "conditions": [], "reason": "", "test_authorized": False},
    )
    for name in RUNTIME_FILES:
        if not (root / name).exists():
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_text(
                "synthetic code seal for " + name, encoding="utf-8"
            )
    return splits


def write_augmented(root, inp, openings=None):
    """The augmented patients as the server makes them, without an image library: the rewritten statements, stand-in
    image files and the record of the images. Returns the folder; `make_augmented.build` writes the manifest from it."""
    root, rule = (Path(root), inp.protocol["test2"])
    folder = root / rule["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    rows, made = ([], {})
    for cid in inp.division["test2_sources"]:
        new_id = aug_id(cid, rule["id_offset"])
        rows.append({"source_id": cid, "opening": (openings or {}).get(cid, REWRITTEN)})
        for exam in inp.cases[cid]["exams"]:
            for name in exam["images"]:
                target = root / IMAGES / aug_image(name, cid, new_id)
                target.write_bytes(b"augmented image " + new_id.encode())
                made[name] = {
                    "name": target.name,
                    "source": name,
                    "sha256": sha(target),
                    "transform": {"rotation_degrees": 1.0},
                }
    (folder / "openings.jsonl").write_text(
        "".join((json.dumps(r) + "\n" for r in rows)), encoding="utf-8"
    )
    atomic_json(
        folder / "images.json",
        {"seed": inp.protocol["pool"]["seed"], "image": rule["image"], "images": made},
    )
    return folder
