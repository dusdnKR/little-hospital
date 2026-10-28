import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import urllib.request
import zipfile

REVISION = "7e7c465a68eb2b866926bfa59c8c9d17a8daba65"
BASE = f"https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA/resolve/{REVISION}"
FILES = ("README.md", "MM/dev.jsonl", "MM/test.jsonl", "images.zip")
REQUIRED = {
    "id",
    "question",
    "options",
    "label",
    "images",
    "medical_task",
    "body_system",
    "question_type",
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(root):
    for name in FILES:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            continue
        partial = target.with_name(target.name + ".part")
        req = urllib.request.Request(
            f"{BASE}/{name}", headers={"User-Agent": "clinical-memory-data/1.0"}
        )
        with (
            urllib.request.urlopen(req, timeout=120) as response,
            partial.open("wb") as out,
        ):
            while block := response.read(1024 * 1024):
                out.write(block)
        partial.replace(target)


def audit(root):
    errors, rows, counts = ([], [], {})
    seen = set()
    for split in ("dev", "test"):
        records = [
            json.loads(line)
            for line in (root / "MM" / f"{split}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        counts[split] = len(records)
        for row in records:
            missing = REQUIRED - row.keys()
            if missing:
                errors.append(f"Missing fields: {sorted(missing)}")
                continue
            if row["id"] in seen:
                errors.append(f"Duplicate ID: {row['id']}")
            seen.add(row["id"])
            if not isinstance(row["options"], dict) or set(row["options"]) != set(
                "ABCDE"
            ):
                errors.append(f"Invalid options: {row['id']}")
            elif row["label"] not in row["options"]:
                errors.append(f"Invalid label: {row['id']}")
            if not isinstance(row["question"], str) or not row["question"].strip():
                errors.append(f"Invalid question: {row['id']}")
            if not isinstance(row["images"], list) or not row["images"]:
                errors.append(f"No images: {row['id']}")
            rows.append(row)
    if counts != {"dev": 5, "test": 2000}:
        errors.append(f"Unexpected row counts for pinned revision: {counts}")
    with zipfile.ZipFile(root / "images.zip") as z:
        files = [item for item in z.infolist() if not item.is_dir()]
        image_members = {}
        for item in files:
            path = PurePosixPath(item.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in item.filename
                or (":" in item.filename)
            ):
                errors.append(f"Unsafe ZIP path: {item.filename}")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                if path.name in image_members:
                    errors.append(f"Ambiguous image basename: {path.name}")
                image_members[path.name] = item.filename
        bad = z.testzip()
        if bad:
            errors.append(f"CRC failure: {bad}")
        used = Counter((name for row in rows for name in row["images"]))
        missing = sorted(set(used) - image_members.keys())
        if missing:
            errors.append(f"Missing referenced images: {missing}")
    expected = json.loads(
        (
            Path(__file__).resolve().parents[2] / "configs/dataset_source.json"
        ).read_text()
    )["files"]
    for name, digest in expected.items():
        if sha256(root / name) != digest:
            errors.append("Pinned SHA-256 mismatch: " + name)
    return {
        "dataset": "TsinghuaC3I/MedXpertQA",
        "revision": REVISION,
        "rows": counts,
        "medical_task": dict(Counter((r["medical_task"] for r in rows))),
        "unique_referenced_images": len(used),
        "zip_image_members": len(image_members),
        "image_references": sum(used.values()),
        "filenames_reused_between_rows": sum((n > 1 for n in used.values())),
        "zip_crc_checked": True,
        "images_decoded": False,
        "clinical_curation_complete": False,
        "files": {
            name: {"bytes": (root / name).stat().st_size, "sha256": sha256(root / name)}
            for name in FILES
        },
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("data/raw/medxpertqa") / REVISION
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--report", type=Path, default=Path("data/source_manifest.json")
    )
    args = parser.parse_args()
    if args.download:
        download(args.root)
    report = audit(args.root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
