import re


def unfence(raw):
    if not isinstance(raw, str):
        return raw
    match = re.fullmatch(
        "\\s*```(?:json)?\\s*\\n([\\s\\S]*?)\\n```\\s*", raw, flags=re.IGNORECASE
    )
    return match.group(1).strip() if match else raw


def clean_repair(repair):
    return (lambda error: unfence(repair(error))) if repair is not None else None
