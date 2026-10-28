from clinical_memory.prompts.presentation import MAX_KEYWORDS, KEYWORD_WORDS
from collections import Counter
from copy import deepcopy
import json
import math
import re
from clinical_memory.environment.formatting import unfence
from clinical_memory.memory.evidence import content_tokens

FORBIDDEN = re.compile("\\d|\\bMM-|\\b(?i:answer|option|choice)\\s*[:=]?\\s*[A-E]\\b")


def terms(text):
    """Content words of a name, with a plain plural folded onto its singular."""
    return {
        w[:-1] if len(w) > 3 and w.endswith("s") and (not w.endswith("ss")) else w
        for w in content_tokens(text)
    }


def when_text(named):
    """The doctor's own name for a presentation, as one line."""
    return ", ".join([named["complaint"], *named["context"]])


def check(raw):
    try:
        obj = json.loads(unfence(raw))
    except (TypeError, ValueError):
        obj = None
    if not isinstance(obj, dict) or set(obj) != {"complaint", "context"}:
        return (
            None,
            ["Return one JSON object with exactly the keys complaint and context."],
        )
    complaint, context = (obj["complaint"], obj["context"])
    if (
        not isinstance(complaint, str)
        or not isinstance(context, list)
        or (not all((isinstance(w, str) for w in context)))
    ):
        return (None, ["complaint must be a string and context a list of strings."])
    named, errors = (
        {"complaint": complaint.strip(), "context": [w.strip() for w in context]},
        [],
    )
    words = [named["complaint"], *named["context"]]
    if not 1 <= len(named["context"]) <= MAX_KEYWORDS - 1:
        errors.append(f"context has one to {MAX_KEYWORDS - 1} keywords.")
    if any((not 1 <= len(w.split()) <= KEYWORD_WORDS for w in words)):
        errors.append(
            f"complaint and each context keyword have one to {KEYWORD_WORDS} words."
        )
    if any((FORBIDDEN.search(w) for w in words)):
        errors.append("Do not use digits, case identifiers or answer letters.")
    if len({w.casefold() for w in words}) != len(words):
        errors.append("Do not repeat a keyword.")
    if not errors and (
        not terms(named["complaint"]) or len(terms(" ".join(words))) < 2
    ):
        errors.append(
            "complaint needs a clinical word, and the whole name at least two."
        )
    return (None, errors) if errors else (named, [])


def match(entries, named, k=3, threshold=0.5, min_terms=2):
    """Lessons from patients with the same main complaint as this one.

    A lesson is recalled when its complaint and this patient's complaint share at least `threshold` of their words
    (shared words over all words of the two), and the two names share at least `min_terms` words in all, so that a
    one-word complaint needs one more shared detail. Recalled lessons are ordered by the IDF-weighted share of
    their name that returns in this patient's name. These are lexical measures, not clinical probabilities. Words
    are added in sorted order, so a run and its replay compute the same numbers.
    """
    if not 0 <= threshold <= 1 or k < 0:
        raise ValueError("Invalid recall rule")
    complaint = terms(named["complaint"]) if named else set()
    query = terms(when_text(named)) if named else set()
    documents = [terms(e["when"]) for e in entries]
    n, df = (len(documents), Counter((w for d in documents for w in d)))
    ranked = []
    for index, (entry, document) in enumerate(zip(entries, documents)):
        weight = {w: 1 + math.log((n + 1) / (df[w] + 1)) for w in sorted(document)}
        matched, total, own = (
            sorted(query & document),
            sum(weight.values()),
            terms(entry["complaint"]),
        )
        ranked.append(
            {
                "entry": entry,
                "index": index,
                "matched_terms": matched,
                "complaint_match": len(complaint & own) / len(complaint | own)
                if complaint | own
                else 0.0,
                "score": sum((weight[w] for w in matched)) / total if total else 0.0,
            }
        )
    ranked.sort(key=lambda r: (-r["complaint_match"], -r["score"], r["index"]))
    selected = [
        r
        for r in ranked
        if r["complaint_match"] >= threshold and len(r["matched_terms"]) >= min_terms
    ][:k]
    audit = {
        "threshold": threshold,
        "min_terms": min_terms,
        "memory_count": n,
        "complaint_terms": sorted(complaint),
        "query_terms": sorted(query),
        "candidates": [
            {
                "lesson_id": r["entry"]["lesson_id"],
                "complaint_match": r["complaint_match"],
                "score": r["score"],
                "matched_terms": r["matched_terms"],
            }
            for r in ranked[:k]
        ],
        "selected_ids": [r["entry"]["lesson_id"] for r in selected],
        "abstained": not selected,
    }
    return ([deepcopy(r["entry"]) for r in selected], audit)
