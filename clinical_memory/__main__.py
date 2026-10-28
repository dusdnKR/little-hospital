"""Command-line entry point for the complete clinical-memory pipeline."""

import runpy
import sys

COMMANDS = {
    "download-data": "data.download",
    "prepare-data": "data.prepare",
    "augment": "data.augment",
    "split": "data.split",
    "check": "validation",
    "download-models": "models.download",
    "run": "training.run",
    "audit": "evaluation.audit",
    "analyze": "analysis.report",
    "notebook": "analysis.notebook",
    "analyze-replay": "analysis.replay",
    "simulate": "simulation",
    "record-simulation": "simulation.record",
    "validate-simulation": "simulation.validate",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(
            "Usage: python -m clinical_memory COMMAND [arguments]\n\nCommands:\n  "
            + "\n  ".join(COMMANDS)
        )
        return
    command = sys.argv.pop(1)
    if command not in COMMANDS:
        raise SystemExit("Unknown command: " + command)
    runpy.run_module("clinical_memory." + COMMANDS[command], run_name="__main__")


if __name__ == "__main__":
    main()
