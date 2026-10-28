from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

STOP = set(
    "a an the this that these those is are was were be been to of and or for in on at with by from as it its he she his her has have had patient patients year old years most likely diagnosis presents presented man woman male female".split()
)


def tokens(text):
    return [
        w for w in re.findall("[a-z]+", text.lower()) if len(w) > 1 and w not in STOP
    ]


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def atomic_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(canonical(value) + b"\n")
    tmp.replace(path)


class MemoryStore:
    def __init__(self, path=None, readonly=False, entries=None, max_lessons=120):
        self.path = Path(path) if path is not None else None
        self.readonly, self.max_lessons = (readonly, max_lessons)
        self._entries = deepcopy(entries or [])
        if self.path and self.path.exists():
            self._entries = [
                json.loads(s)
                for s in self.path.read_text(encoding="utf-8").splitlines()
            ]
        self._validate()

    def _validate(self):
        ids = [e["source_case_id"] for e in self._entries]
        if len(ids) != len(set(ids)) or len(ids) > self.max_lessons:
            raise ValueError("Duplicate lesson source or memory capacity exceeded")
        for e in self._entries:
            if (
                e["source_split"] != "evolution"
                or not e["lesson_text"].strip()
                or (not e["evidence_refs"])
            ):
                raise ValueError("Memory accepts validated evolution lessons only")

    def entries(self):
        return deepcopy(self._entries)

    def digest(self):
        return hashlib.sha256(
            b"".join((canonical(e) + b"\n" for e in self._entries))
        ).hexdigest()

    def snapshot(self):
        return {"sha256": self.digest(), "entries": self.entries()}

    def readonly_view(self):
        return MemoryStore(
            readonly=True, entries=self._entries, max_lessons=self.max_lessons
        )

    def add(self, lesson):
        if self.readonly:
            raise PermissionError("Evaluation memory is read-only")
        if lesson["source_split"] != "evolution":
            raise ValueError("Only evolution may update memory")
        for previous in self._entries:
            if previous["source_case_id"] == lesson["source_case_id"]:
                if previous != lesson:
                    raise ValueError("Conflicting lesson for completed case")
                return False
        candidate = self._entries + [deepcopy(lesson)]
        MemoryStore(entries=candidate, max_lessons=self.max_lessons)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_bytes(b"".join((canonical(e) + b"\n" for e in candidate)))
            tmp.replace(self.path)
        self._entries = candidate
        return True

    def retrieve(self, public_context, k=4):
        if k < 0:
            raise ValueError("Negative retrieval size")
        query = set(tokens(public_context))
        documents = [Counter(tokens(e["lesson_text"])) for e in self._entries]
        n = len(documents)
        if not n or not query or (not k):
            return []
        lengths = [sum(d.values()) for d in documents]
        average = sum(lengths) / n or 1
        df = {w: sum((w in d for d in documents)) for w in query}
        scored = []
        for index, (entry, doc, length) in enumerate(
            zip(self._entries, documents, lengths)
        ):
            score = 0.0
            for word in query:
                tf = doc[word]
                if tf:
                    idf = math.log(1 + (n - df[word] + 0.5) / (df[word] + 0.5))
                    score += (
                        idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * length / average))
                    )
            if score > 0:
                scored.append((score, index, entry))
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [deepcopy(item) for _, _, item in scored[:k]]
