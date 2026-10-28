from collections import namedtuple
import hashlib
import json
from pathlib import Path
from clinical_memory.data.io import read_rows
from clinical_memory.data.speech import load_profiles
from clinical_memory.environment.scope import with_scopes

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_FILES = (
    "clinical_memory/environment/guards.py",
    "clinical_memory/environment/formatting.py",
    "clinical_memory/models/guarded.py",
    "clinical_memory/data/inputs.py",
    "clinical_memory/environment/encounter.py",
    "clinical_memory/analysis/statuses.py",
    "configs/models.json",
    "configs/inference.json",
)
Inputs = namedtuple("Inputs", "study frozen cases golds")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_id(value):
    if not value or not value.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Invalid run ID")
    return value


def code_hashes(root=None):
    root = Path(root or ROOT)
    paths = list((root / "clinical_memory").rglob("*.py")) + [
        root / "configs" / name
        for name in (
            "experiment.json",
            "models.json",
            "inference.json",
            "disease_groups.json",
        )
        if (root / "configs" / name).is_file()
    ]
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(paths)}


def inputs(root=None):
    root = Path(root or ROOT)
    protocol = load(root / "configs/experiment.json")
    study = load(root / "configs/models.json")
    if sha(root / study["manifest"]) != protocol["dataset"]["manifest_sha256"]:
        raise ValueError("Frozen manifest changed")
    frozen = load(root / study["manifest"])
    for name, digest in frozen["source_hashes"].items():
        if sha(root / name) != digest:
            raise ValueError("Frozen source changed: " + name)
    cohort = root / frozen["cohort"]
    cases = with_scopes(
        load_profiles(
            cohort / "speech_profiles.json", read_rows(cohort / "cases.jsonl")
        ),
        study.get("measurement_scope_annotations"),
    )
    return Inputs(study, frozen, cases, read_rows(cohort / "private_gold.jsonl"))


def runtime_config(root=None):
    cfg = load(Path(root or ROOT) / "configs/inference.json")
    cfg.update(
        max_context_tokens=32768,
        max_model_calls=100000,
        max_backend_seconds=144000,
        resume_call_log=True,
    )
    cfg["max_new_tokens"].update(reflection=384, measurement=1024)
    return cfg
