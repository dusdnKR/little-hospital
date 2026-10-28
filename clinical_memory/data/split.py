import argparse
import json
from clinical_memory.config import ROOT, LEARNING, SETS, inputs, code_hashes
from clinical_memory.memory.diagnosis import lookup
from clinical_memory.training.run import smoke_plan


def sizes(root=ROOT):
    inp = inputs(root)
    division, protocol, memory = (inp.division, inp.protocol, inp.protocol["memory"])
    full = [
        {
            "lesson_id": c,
            "diagnose": "note",
            "disease": inp.golds[c]["options"][inp.golds[c]["label"]],
        }
        for c in division["train"]
    ]
    right = wrong = 0
    for cid in division["test1"]:
        gold = inp.golds[cid]
        found = lookup(
            full, gold["options"], memory["notes_top_k"], memory["notes_threshold"]
        )[1]["selected"]
        right += any((gold["label"] in s["options"] for s in found))
        wrong += any((gold["label"] not in s["options"] for s in found))
    runs = (
        len(protocol["doctors"])
        * len(LEARNING)
        * len(protocol["curricula"])
        * len(protocol["training_seeds"])
    )
    sources = division["test2_sources"]
    return {
        "patients": division["patients"],
        "training_patients": len(division["train"]),
        "test1_patients": len(division["test1"]),
        "test1_with_a_same_disease_training_patient": len(division["test1_partners"]),
        "test1_note_on_the_right_option_can_be_found": right,
        "test1_note_on_a_wrong_option_can_be_found": wrong,
        "test2_patients": len(sources),
        "test2_images": sum(
            (len(e["images"]) for c in sources for e in inp.cases[c]["exams"])
        ),
        "test2_built": (root / protocol["test2"]["folder"] / "manifest.json").exists(),
        "visits": {
            "smoke": len(smoke_plan(division)),
            "train": len(division["train"]),
            **{
                which: len(division["test1"] if which == "test1" else sources)
                for which in SETS
            },
        },
        "training_runs": runs,
        "evaluations_per_test_set": runs + len(protocol["doctors"]),
        "ways_of_learning": list(LEARNING),
        "division_sha256": protocol["pool"]["division_sha256"],
        "sealed_runtime_files": len(code_hashes(root)),
    }


def composition(root=ROOT):
    """Where the patients of each part come from and how hard the plain question was for the 4B model: counts only."""
    from collections import Counter
    from clinical_memory.data.io import read_rows

    inp = inputs(root)
    division, frozen = (inp.division, inp.frozen)
    levels = {
        cid: row["correct_count"]
        for cid, row in read_rows(root / frozen["cohort"] / "difficulty.jsonl").items()
    }

    def described(ids):
        return {
            "times_right_of_three": {
                str(k): v for k, v in sorted(Counter((levels[c] for c in ids)).items())
            }
        }

    return {
        name: described(division[key])
        for name, key in (
            ("train", "train"),
            ("test1", "test1"),
            ("test2_sources", "test2_sources"),
        )
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="also write the counts to results/split.json",
    )
    args = parser.parse_args()
    found = sizes()
    if args.write:
        from clinical_memory.memory.store import atomic_json

        atomic_json(
            ROOT / "results/split.json",
            {
                "schema": 1,
                **{
                    k: v
                    for k, v in found.items()
                    if k not in {"test2_built", "sealed_runtime_files"}
                },
                "composition": composition(),
            },
        )
    print(json.dumps(found, indent=1))
