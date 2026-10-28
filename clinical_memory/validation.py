"""Validate the standalone code and inputs before any model inference."""

import json
import subprocess
import sys
from pathlib import Path
from clinical_memory.preflight import compute_only, stamp

ROOT = Path(__file__).resolve().parents[1]
def main():
    compute_only(ROOT)
    subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-b"],
        cwd=ROOT,
        check=True,
    )
    current = stamp(ROOT)
    subprocess.run(
        [sys.executable, "-m", "clinical_memory.data.split"], cwd=ROOT, check=True
    )
    path = ROOT / "runs/release_gate.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        json.dumps({"passed": True, "stamp": current, "inference_calls": 0}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print("Release gate passed. No model inference performed.")


if __name__ == "__main__":
    main()
