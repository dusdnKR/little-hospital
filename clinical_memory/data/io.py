import json
from pathlib import Path


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def read_rows(path):
    path = Path(path)
    result = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line, object_pairs_hook=unique_object)
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("id"), str)
            or (not row["id"].strip())
        ):
            raise ValueError(f"Invalid case ID at {path}:{line_number}")
        if row["id"] in result:
            raise ValueError(f"Duplicate case ID at {path}:{line_number}: " + row["id"])
        result[row["id"]] = row
    return result
