from clinical_memory.prompts.reflection import PROMPTS
from copy import deepcopy
import json
from clinical_memory.models.actors import released_images
from clinical_memory.environment.formatting import unfence
from clinical_memory.memory.presentation import RESERVED
from clinical_memory.memory.matching import when_text
from clinical_memory.memory.feedback import TOLD, own_record


def object_reply(raw):
    """Read a JSON object, allowing prose around one unambiguous object with lesson fields."""
    if not isinstance(raw, str):
        return (None, [])
    text = unfence(raw)
    try:
        obj = json.loads(text)
        return (obj, []) if isinstance(obj, dict) else (None, [])
    except ValueError:
        pass
    decoder, found, pos = (json.JSONDecoder(), [], 0)
    while pos < len(text):
        start = text.find("{", pos)
        if start < 0:
            break
        try:
            obj, length = decoder.raw_decode(text[start:])
        except ValueError:
            pos = start + 1
            continue
        pos = start + length
        if isinstance(obj, dict) and set(obj) & {"collect", "diagnose"}:
            found.append(obj)
    return (
        (found[0], ["text around the JSON object was ignored"])
        if len(found) == 1
        else (None, [])
    )


def usable(raw, keys):
    """Keep all lesson words except reserved identifiers; never shorten a part to a word target."""
    obj, done = object_reply(raw)
    if obj is None:
        return (None, [])
    parts, explicit_empty = ({}, False)
    for name in keys:
        value = obj.get(name)
        if isinstance(value, list) and all((isinstance(x, str) for x in value)):
            value = " ".join(value)
            done.append(name + " text list was joined")
        if isinstance(value, str) and (not value.strip()):
            explicit_empty = True
        if not isinstance(value, str):
            done.append(name + (" was not text" if name in obj else " was missing"))
            value = ""
        kept = " ".join(RESERVED.sub(" ", value).split())
        if kept != " ".join(value.split()):
            done.append(name + " lost a case identifier or an answer letter")
        parts[name] = kept
    if set(obj) - set(keys):
        done.append("a key that was not asked for was ignored")
    return (parts, done) if any(parts.values()) or explicit_empty else (None, [])


def read_reply(raw, keys, policy):
    parts, done = usable(raw, keys)
    if parts is not None and (policy == "settle" or not done):
        return (parts, done)
    return (None, [])


def ask(backend, system, payload, images, keys, policy):
    """Accept usable text first. Rewrite once only when its parts cannot be read; length is never an error."""
    raw = backend.generate("reflection", system, payload, images)
    obj, done = read_reply(raw, keys, policy)
    if obj is not None:
        return (obj, 0, [], done)
    errors = [
        "Return readable text in these JSON fields: "
        + ", ".join(keys)
        + ". Length is not restricted."
    ]
    again = backend.generate(
        "reflection_repair",
        system,
        {"record": payload, "previous_output": raw, "format_error": " ".join(errors)},
        images,
    )
    obj, done = read_reply(again, keys, policy)
    if obj is not None:
        return (obj, 1, [], done)
    return (None, 1, errors, [])


def draft(backend, trace, phase, step, key=None, policy="settle"):
    """Return (lesson or None, what happened). `key` is (answer sheet, its image names), or None without the sheet."""
    if phase != "evolution":
        raise PermissionError("Reflection is disabled outside evolution")
    if policy not in {"settle", "discard"}:
        raise ValueError("Unknown lesson format policy")
    if not trace["result"]["submission"]["valid"]:
        return (None, {"status": "skipped_invalid_final", "repairs": 0})
    if not trace["presentation"]["named"]:
        return (None, {"status": "no_presentation", "repairs": 0})
    payload, images = (own_record(trace), released_images({"history": trace["events"]}))
    keys, system = (("collect",), PROMPTS["self"])
    if key is not None:
        sheet, extra = key
        payload["answer_key"], images = (
            deepcopy(sheet),
            images + [n for n in extra if n not in images],
        )
        keys, system = (("collect", "diagnose"), PROMPTS["key"])
    parts, repairs, errors, done = ask(backend, system, payload, images, keys, policy)
    if parts is None:
        return (None, {"status": "invalid", "repairs": repairs, "errors": errors})
    collect, diagnose = (parts["collect"], parts.get("diagnose", ""))
    if not collect and (not diagnose):
        return (None, {"status": "abstained", "repairs": repairs})
    named = payload["presentation"]
    when = when_text(named)
    text = (
        "When: "
        + when
        + (" Action: " + collect if collect else "")
        + (" Diagnosis: " + diagnose if diagnose else "")
    )
    lesson = {
        "lesson_id": "L" + str(step),
        "source_case_id": trace["id"],
        "source_split": "evolution",
        "created_after_step": step,
        "when": when,
        "complaint": named["complaint"],
        "context": named["context"],
        "collect": collect,
        "diagnose": diagnose,
        "lesson_text": text,
        "evidence_refs": ["encounter"] + (["answer_key"] if key is not None else []),
        "model_revision": backend.calls[-1]["model_revision"],
    }
    if key is not None:
        lesson.update(
            outcome=TOLD[key[0]["your_diagnosis_was_confirmed"]],
            disease=key[0]["confirmed_diagnosis"],
        )
    info = {"status": "stored", "repairs": repairs}
    if done:
        info.update(settled=done, errors=errors)
    return (lesson, info)
