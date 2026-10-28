import argparse
from pathlib import Path
from clinical_memory.config import ROOT, RUNS, load
from clinical_memory.data.inputs import safe_id


def show(run_id, lessons=10, root=None):
    folder = Path(root or ROOT) / RUNS / safe_id(run_id)
    seal, entries = (
        load(folder / "seal.json"),
        load(folder / "memory_snapshot.json")["entries"],
    )
    lines = [
        f"=== {run_id}: doctor {seal['doctor']}, learning {seal['learning']}, {len(entries)} lessons, {sum((bool(e['diagnose']) for e in entries))} with a diagnosis part",
        f"--- first {lessons} lessons [after training patient N] (own diagnosis for that patient; filed under)",
    ]
    for entry in entries[:lessons]:
        told = (
            f" ({entry['outcome']}; {entry['disease']})" if "outcome" in entry else ""
        )
        lines.append(f"[{entry['created_after_step']}]{told} {entry['lesson_text']}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--lessons", type=int, default=10)
    args = parser.parse_args()
    from clinical_memory.preflight import compute_only

    compute_only(ROOT)
    print(show(args.run_id, args.lessons))
