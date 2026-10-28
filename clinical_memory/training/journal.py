from pathlib import Path
from clinical_memory.memory.store import MemoryStore, atomic_json


def restore_memory(memory_path, records, schedule, capacity):
    """Replay only completed, contiguous journal records; repair a missed notebook append."""
    expected, ordered, gap = (MemoryStore(max_lessons=capacity), [], False)
    for phase, cid in schedule:
        key = phase + ":" + cid
        if key not in records:
            gap = True
            continue
        if gap:
            raise ValueError("Non-contiguous completed episode journal")
        record = records[key]
        if record["phase"] != phase or record["trace"]["id"] != cid:
            raise ValueError("Journal identity mismatch")
        if phase == "evolution" and record.get("lesson"):
            if record["lesson"]["source_case_id"] != cid:
                raise ValueError("Lesson source mismatch")
            expected.add(record["lesson"])
            ordered.append(record["lesson"])
        elif record.get("lesson"):
            raise ValueError("Evaluation journal contains a new lesson")
    if set(records) - {phase + ":" + cid for phase, cid in schedule}:
        raise ValueError("Unexpected journal record")
    actual = MemoryStore(memory_path, max_lessons=capacity)
    existing = actual.entries()
    if existing != ordered[: len(existing)]:
        raise ValueError("Memory differs from completed journal")
    for lesson in ordered[len(existing) :]:
        actual.add(lesson)
    if actual.digest() != expected.digest():
        raise ValueError("Memory replay mismatch")
    return actual


def commit_episode(path, record, store):
    """Journal first, then the notebook. A crash between them is recovered by replay."""
    path = Path(path)
    if path.exists():
        raise FileExistsError("Completed episode cannot be overwritten")
    if record.get("lesson"):
        if (
            record["phase"] != "evolution"
            or record["lesson"]["source_case_id"] != record["trace"]["id"]
        ):
            raise ValueError("Lesson source must match completed evolution episode")
        MemoryStore(entries=store.entries(), max_lessons=store.max_lessons).add(
            record["lesson"]
        )
    atomic_json(path, record)
    if record.get("lesson"):
        store.add(record["lesson"])
