from copy import deepcopy
import re

FILLER = frozenset(
    {
        "of",
        "the",
        "and",
        "with",
        "in",
        "to",
        "due",
        "a",
        "an",
        "by",
        "from",
        "on",
        "for",
        "or",
    }
)


def name_terms(text):
    """The words of a diagnosis name: lower case, without a possessive ending and filler words, a plain plural
    folded onto its singular. Numerals and single letters stay, because they can tell two types of a disease apart."""
    words = re.findall("[a-z0-9]+", re.sub("[’']s\\b", "", text.lower()))
    return frozenset(
        (
            w[:-1] if len(w) > 3 and w.endswith("s") and (not w.endswith("ss")) else w
            for w in words
            if w not in FILLER
        )
    )


def likeness(first, second):
    """Shared words over all words of two diagnosis names; 0 when either has no word."""
    a, b = (name_terms(first), name_terms(second))
    return len(a & b) / len(a | b) if a and b else 0.0


def lookup(entries, options, top_k=5, threshold=0.6):
    """The lessons whose diagnosis part is filed under a name like one of the options, best match first, then the
    older lesson; at most `top_k`. Returns (lessons, what was found) and never changes an entry."""
    if not 0 < threshold <= 1 or type(top_k) is not int or top_k < 1:
        raise ValueError("Invalid note lookup rule")
    found = []
    for index, entry in enumerate(entries):
        if not entry.get("diagnose") or not entry.get("disease"):
            continue
        scores = {
            letter: likeness(text, entry["disease"])
            for letter, text in sorted(options.items())
        }
        best = max(scores.values(), default=0.0)
        if best >= threshold:
            found.append(
                {
                    "entry": entry,
                    "index": index,
                    "score": best,
                    "options": [
                        letter for letter, score in scores.items() if score >= threshold
                    ],
                }
            )
    found.sort(key=lambda r: (-r["score"], r["index"]))
    selected = found[:top_k]
    audit = {
        "threshold": threshold,
        "top_k": top_k,
        "memory_count": len(entries),
        "with_diagnosis_part": sum(
            (bool(e.get("diagnose") and e.get("disease")) for e in entries)
        ),
        "matched": len(found),
        "selected": [
            {
                "lesson_id": r["entry"]["lesson_id"],
                "score": r["score"],
                "options": r["options"],
            }
            for r in selected
        ],
    }
    return ([deepcopy(r["entry"]) for r in selected], audit)


def shown(lessons):
    """How the notes are shown to the reasoning and the answer: the diagnosis each is filed under and the doctor's
    own words about it."""
    return [{"diagnosis": m["disease"], "note": m["diagnose"]} for m in lessons]
