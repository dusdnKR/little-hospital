"""Download only the model revisions pinned by the final study."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    from clinical_memory.preflight import compute_only

    compute_only(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", choices=["2B", "4B", "9B"], default=["2B", "4B", "9B"]
    )
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    models = json.loads((ROOT / "configs/models.json").read_text())["models"]
    for size in args.models:
        cfg = models[size]
        target = ROOT / cfg["model_directory"]
        snapshot_download(
            repo_id=cfg["model_id"], revision=cfg["model_revision"], local_dir=target
        )
        (target / "ksc_download_manifest.json").write_text(
            json.dumps(
                {"model_id": cfg["model_id"], "model_revision": cfg["model_revision"]},
                indent=2,
            )
            + "\n"
        )
        print(size, cfg["model_revision"])


if __name__ == "__main__":
    main()
