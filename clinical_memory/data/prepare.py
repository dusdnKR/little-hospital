"""Validate the pinned source and materialize the curated final cohort."""

import argparse
from collections import Counter
import json
from pathlib import Path, PurePosixPath
import zipfile
from .download import REVISION, sha256
from .io import read_rows
from .speech import load_profiles
from clinical_memory.data.inputs import ROOT


def verified_write(target, content):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != content:
        raise ValueError(f"Refusing to overwrite changed input: {target}")
    if not target.exists():
        partial = target.with_suffix(target.suffix + ".part")
        partial.write_bytes(content)
        partial.replace(target)


def prepare(root=ROOT, source=None):
    root = Path(root)
    source = Path(source) if source else root / "data/raw/medxpertqa" / REVISION
    spec = json.loads((root / "configs/dataset_source.json").read_text())
    for name, digest in spec["files"].items():
        if sha256(source / name) != digest:
            raise ValueError(f"Pinned source checksum mismatch: {name}")
    annotations = root / "resources/cohort"
    checks = json.loads((annotations / "checksums.json").read_text())
    for name, digest in checks["files"].items():
        if sha256(annotations / name) != digest:
            raise ValueError(f"Curated input checksum mismatch: {name}")
    raw = {**read_rows(source / "MM/dev.jsonl"), **read_rows(source / "MM/test.jsonl")}
    cases = read_rows(annotations / "cases.jsonl")
    golds = read_rows(annotations / "private_gold.jsonl")
    manifest = json.loads((annotations / "manifest.json").read_text())
    patients = manifest["patients"]
    if set(patients) != set(cases) or set(cases) != set(golds):
        raise ValueError(
            "Cohort membership differs between manifest, cases and answer keys"
        )
    load_profiles(annotations / "speech_profiles.json", cases)
    wanted = set()
    for cid in patients:
        case, gold, original = (cases[cid], golds[cid], raw[cid])
        if gold != {
            "id": cid,
            "options": original["options"],
            "label": original["label"],
        }:
            raise ValueError(f"Answer key differs from original dataset: {cid}")
        if (
            "label" in case
            or "options" in case
            or "Answer Choices:" in json.dumps(case)
        ):
            raise ValueError(f"Answer options leaked into the collection record: {cid}")
        images = [name for exam in case["exams"] for name in exam["images"]]
        if Counter(images) != Counter(original["images"]):
            raise ValueError(f"Image assignment differs from original dataset: {cid}")
        wanted.update(images)
    with zipfile.ZipFile(source / "images.zip") as archive:
        members = {}
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in info.filename
                or (":" in info.filename)
            ):
                raise ValueError("Unsafe archive member")
            if not info.is_dir() and path.name in wanted:
                if path.name in members:
                    raise ValueError("Ambiguous image basename")
                members[path.name] = info.filename
        if set(members) != wanted:
            raise ValueError("An image required by the cohort is absent")
        for name, member in sorted(members.items()):
            verified_write(root / "data/images" / name, archive.read(member))
    for name in [
        "cases.jsonl",
        "private_gold.jsonl",
        "difficulty.jsonl",
        "speech_profiles.json",
    ]:
        verified_write(
            root / "data/processed/cohort" / name, (annotations / name).read_bytes()
        )
    verified_write(
        root / "data/processed/manifest.json",
        (annotations / "manifest.json").read_bytes(),
    )
    verified_write(
        root / "data/augmented/openings.jsonl",
        (annotations / "openings.jsonl").read_bytes(),
    )
    for name in ["runs", "results", "tmp"]:
        (root / name).mkdir(exist_ok=True)
    from clinical_memory.config import inputs

    division = inputs(root).division
    return {
        "patients": len(patients),
        "images": len(wanted),
        "train": len(division["train"]),
        "test1": len(division["test1"]),
        "test2_sources": len(division["test2_sources"]),
        "new_model_calls": 0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, help="Existing pinned MedXpertQA snapshot"
    )
    args = parser.parse_args(argv)
    print(json.dumps(prepare(source=args.source), indent=2))


if __name__ == "__main__":
    main()
