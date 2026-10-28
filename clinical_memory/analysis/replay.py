"""Summarize a recorded simulation cohort without loading model weights."""

import argparse
import json
from pathlib import Path
from clinical_memory.simulation.replay import load_replay


def analyze(config=None):
    replay = load_replay(config)
    if replay.warnings:
        raise ValueError("; ".join(replay.warnings))
    return {
        model: {
            "patients": len(rows),
            "correct": sum((row.success for row in rows)),
            "accuracy": sum((row.success for row in rows)) / len(rows)
            if rows
            else None,
        }
        for model, rows in replay.encounters.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("results/replay_summary.json")
    )
    args = parser.parse_args()
    result = analyze(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
