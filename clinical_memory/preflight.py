import json
import os
import sys
from pathlib import Path


def compute_only(root):
    if os.environ.get("CLINICAL_MEMORY_REQUIRE_SLURM") == "1" and not os.environ.get(
        "SLURM_JOB_ID"
    ):
        raise RuntimeError("This environment requires a Slurm allocation")
    if sys.flags.optimize or os.environ.get("PYTHONOPTIMIZE"):
        raise RuntimeError("Python assertions must be enabled")


def stamp(root):
    from clinical_memory.config import code_hashes, inputs

    inp = inputs(root)
    return {
        "code": code_hashes(root),
        "data": inp.frozen["source_hashes"],
        "division": inp.protocol["pool"]["division_sha256"],
    }


def preflight(root, purpose):
    compute_only(root)
    receipt = Path(root) / "runs/release_gate.json"
    if not receipt.exists():
        raise ValueError("Run python -m clinical_memory.validation before inference")
    state = json.loads(receipt.read_text())
    if not state.get("passed") or state["stamp"] != stamp(root):
        raise ValueError("Code or inputs changed after the release gate")
    return {"passed": True, "purpose": purpose, "launches_job": False}
