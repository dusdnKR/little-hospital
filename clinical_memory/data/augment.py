import argparse
from datetime import datetime, timezone
import hashlib
import json
import random
from statistics import median
from clinical_memory.memory.store import atomic_json
from clinical_memory.data.augmented import (
    IMAGES,
    aug_id,
    aug_image,
    file_sha,
    load_set,
    opening_errors,
    similarity,
)
from clinical_memory.config import ROOT, inputs


def sources(inp):
    """(source patient, ID of its augmented version, the names of its images) for every patient of test 2."""
    offset = inp.protocol["test2"]["id_offset"]
    return [
        (
            inp.cases[cid],
            aug_id(cid, offset),
            [n for exam in inp.cases[cid]["exams"] for n in exam["images"]],
        )
        for cid in inp.division["test2_sources"]
    ]


def todo(root, inp):
    folder = root / inp.protocol["test2"]["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    rows = [
        {"source_id": case["id"], "opening": case["public_context"]["text"]}
        for case, _, _ in sources(inp)
    ]
    (folder / "openings_todo.jsonl").write_text(
        "".join((json.dumps(r, ensure_ascii=False) + "\n" for r in rows)),
        encoding="utf-8",
    )
    return {
        "source_patients": len(rows),
        "written": (folder / "openings_todo.jsonl").relative_to(root).as_posix(),
    }


def transform_for(name, seed, recipe):
    """The parameters of one image, drawn from the seed and the name of the source file."""
    rng = random.Random(
        int(hashlib.sha256(f"image-{seed}:{name}".encode()).hexdigest()[:16], 16)
    )
    turn, crop, light, depth = (
        recipe[k] for k in ("rotation_degrees", "crop_min", "brightness", "contrast")
    )
    return {
        "rotation_degrees": round(rng.uniform(-turn, turn), 3),
        "keep": round(rng.uniform(crop, 1.0), 4),
        "shift_x": round(rng.random(), 4),
        "shift_y": round(rng.random(), 4),
        "brightness": round(rng.uniform(1 - light, 1 + light), 4),
        "contrast": round(rng.uniform(1 - depth, 1 + depth), 4),
    }


def transformed(source, change, quality):
    """The bytes of the transformed image. No flip: side matters in a medical image."""
    from io import BytesIO
    from PIL import Image, ImageEnhance

    with Image.open(source) as original:
        im = original.convert("RGB")
    width, height = im.size
    corners = [
        im.getpixel(p)
        for p in ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1))
    ]
    fill = tuple((sum(sorted((c[i] for c in corners))[1:3]) // 2 for i in range(3)))
    im = im.rotate(
        change["rotation_degrees"],
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=fill,
    )
    kept_w, kept_h = (
        max(1, round(width * change["keep"])),
        max(1, round(height * change["keep"])),
    )
    left, top = (
        round((width - kept_w) * change["shift_x"]),
        round((height - kept_h) * change["shift_y"]),
    )
    im = im.crop((left, top, left + kept_w, top + kept_h))
    im = ImageEnhance.Contrast(
        ImageEnhance.Brightness(im).enhance(change["brightness"])
    ).enhance(change["contrast"])
    out = BytesIO()
    if source.suffix.lower() == ".png":
        im.save(out, format="PNG")
    else:
        im.save(out, format="JPEG", quality=quality)
    return out.getvalue()


def images(root, inp, force=False):
    rule, seed = (inp.protocol["test2"], inp.protocol["pool"]["seed"])
    folder, made = (root / rule["folder"], {})
    folder.mkdir(parents=True, exist_ok=True)
    for case, new_id, names in sources(inp):
        for name in names:
            target, change = (
                root / IMAGES / aug_image(name, case["id"], new_id),
                transform_for(name, seed, rule["image"]),
            )
            data = transformed(
                root / IMAGES / name, change, rule["image"]["jpeg_quality"]
            )
            if target.exists() and target.read_bytes() != data and (not force):
                raise ValueError(
                    "An augmented image exists with other content; give --force to replace it: "
                    + target.name
                )
            if not target.exists() or force:
                target.write_bytes(data)
            made[name] = {
                "name": target.name,
                "source": name,
                "sha256": file_sha(target),
                "transform": change,
            }
    atomic_json(
        folder / "images.json", {"seed": seed, "image": rule["image"], "images": made}
    )
    return {
        "images": len(made),
        "written": (folder / "images.json").relative_to(root).as_posix(),
    }


def build(root, inp):
    rule = inp.protocol["test2"]
    folder = root / rule["folder"]
    rows = [
        json.loads(s)
        for s in (folder / "openings.jsonl").read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]
    wanted = [case["id"] for case, _, _ in sources(inp)]
    if sorted((r.get("source_id") for r in rows)) != sorted(wanted) or any(
        (set(r) != {"source_id", "opening"} for r in rows)
    ):
        raise ValueError(
            'openings.jsonl must hold exactly one row {"source_id", "opening"} for every source patient of test 2'
        )
    rewritten, made = (
        {r["source_id"]: r["opening"] for r in rows},
        json.loads((folder / "images.json").read_text(encoding="utf-8")),
    )
    if made["image"] != rule["image"] or made["seed"] != inp.protocol["pool"]["seed"]:
        raise ValueError(
            "The images were made with another recipe; run the images step again"
        )
    patients, failed, alike, lengths = ([], [], [], [])
    for case, new_id, names in sources(inp):
        original, opening = (case["public_context"]["text"], rewritten[case["id"]])
        problems = opening_errors(original, opening, rule)
        if problems:
            failed.append(case["id"] + ": " + " ".join(problems))
            continue
        opening = " ".join(opening.split())
        alike.append(similarity(original, opening))
        lengths.append((len(original.split()), len(opening.split())))
        patients.append(
            {
                "id": new_id,
                "source_id": case["id"],
                "opening": opening,
                "images": [made["images"][n] for n in names],
            }
        )
    if failed:
        raise ValueError(
            f"{len(failed)} rewritten opening statements break a rule:\n"
            + "\n".join(failed)
        )
    manifest = {
        "version": rule["version"],
        "division_sha256": inp.protocol["pool"]["division_sha256"],
        "seed": inp.protocol["pool"]["seed"],
        "image": rule["image"],
        "patients": patients,
    }
    atomic_json(folder / "manifest.json", manifest)
    cases, _, sealed = load_set(root, inp)
    public = {
        "schema": 1,
        "version": rule["version"],
        "made_at": datetime.now(timezone.utc).isoformat(),
        "patients": len(cases),
        "images": sum((len(p["images"]) for p in patients)),
        "manifest_sha256": sealed,
        "division_sha256": manifest["division_sha256"],
        "image": rule["image"],
        "opening_statements": {
            "words_source_median": median((a for a, _ in lengths)),
            "words_rewritten_median": median((b for _, b in lengths)),
            "similarity_median": round(median(alike), 3),
            "similarity_max": round(max(alike), 3),
            "numbers_changed": 0,
            "rules": {
                k: rule[k]
                for k in (
                    "opening_min_words",
                    "opening_length_ratio",
                    "opening_max_similarity",
                    "opening_similarity_from_words",
                )
            },
        },
    }
    atomic_json(root / "results/augmentation.json", public)
    return public


def check(root, inp):
    cases, golds, sealed = load_set(root, inp)
    return {
        "patients": len(cases),
        "images": sum((len(e["images"]) for c in cases.values() for e in c["exams"])),
        "manifest_sha256": sealed,
        "answers_carried_over": sum(
            (
                golds[c]["label"] == inp.golds[cases[c]["augmented_from"]]["label"]
                for c in cases
            )
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("step", choices=["todo", "images", "build", "check"])
    parser.add_argument(
        "--force",
        action="store_true",
        help="images: replace an augmented image that exists with other content",
    )
    args = parser.parse_args(argv)
    from clinical_memory.preflight import compute_only

    compute_only(ROOT)
    inp = inputs(ROOT)
    result = (
        images(ROOT, inp, args.force)
        if args.step == "images"
        else {"todo": todo, "build": build, "check": check}[args.step](ROOT, inp)
    )
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
