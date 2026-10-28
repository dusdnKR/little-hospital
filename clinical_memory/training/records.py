from pathlib import Path
import json


def read_journal(directory):
    records = {}
    for path in Path(directory).glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        key = row["phase"] + ":" + row["trace"]["id"]
        if key in records:
            raise ValueError("Duplicate episode journal")
        records[key] = row
    return records
