import json
import re
from clinical_memory.environment.formatting import unfence
from clinical_memory.memory.matching import FORBIDDEN
from clinical_memory.prompts.presentation import MAX_KEYWORDS, KEYWORD_WORDS, PROMPT
from clinical_memory.memory.matching import check, terms, when_text

RESERVED = re.compile("\\bMM-\\S*|\\b(?i:answer|option|choice)\\s*[:=]?\\s*[A-E]\\b")


def trimmed(phrase):
    """One phrase of a name without what the format forbids: identifiers, answer letters, tokens with a digit,
    and words beyond the limit."""
    words = [w for w in RESERVED.sub(" ", phrase).split() if not re.search("\\d", w)]
    return " ".join(words[:KEYWORD_WORDS])


def settle(raw):
    """The usable form of a name that broke a format rule, or None when nothing usable remains."""
    try:
        obj = json.loads(unfence(raw))
    except (TypeError, ValueError):
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("complaint"), str):
        return None
    context = obj.get("context")
    context = (
        [context]
        if isinstance(context, str)
        else context
        if isinstance(context, list)
        else []
    )
    complaint, kept = (trimmed(obj["complaint"]), [])
    for phrase in context:
        phrase = trimmed(phrase) if isinstance(phrase, str) else ""
        if phrase and phrase.casefold() not in {
            complaint.casefold(),
            *(k.casefold() for k in kept),
        }:
            kept.append(phrase)
    named = {"complaint": complaint, "context": kept[: MAX_KEYWORDS - 1]}
    if (
        not terms(complaint)
        or len(terms(when_text(named))) < 2
        or FORBIDDEN.search(when_text(named))
    ):
        return None
    return named


def describe(backend, view):
    """The doctor's name for this opening statement. A name in the right format is taken as written; otherwise the
    doctor rewrites it once, told which rule it broke, and if the rewrite breaks a rule too its usable form is kept."""
    raw = backend.generate("presentation", PROMPT, view, [])
    named, errors = check(raw)
    if named is not None:
        return {"named": named, "repairs": 0}
    again = backend.generate(
        "presentation_repair",
        PROMPT,
        {"record": view, "previous_output": raw, "format_error": " ".join(errors)},
        [],
    )
    named, errors = check(again)
    if named is not None:
        return {"named": named, "repairs": 1}
    named = settle(again) or settle(raw)
    return {
        "named": named,
        "repairs": 1,
        "errors": errors,
        **({"settled": True} if named else {}),
    }
