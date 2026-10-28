import json
import re
from clinical_memory.data.io import unique_object


def final_json_text(raw):
    """Only remove explicit answer-channel/fence wrappers; never mine reasoning for JSON."""
    if not isinstance(raw, str):
        return raw
    if "<unused94>" in raw:
        if "<unused95>" not in raw:
            return ""
        raw = raw.rsplit("<unused95>", 1)[1]
    raw = raw.strip()
    if raw.startswith("```json\n") and raw.endswith("```"):
        raw = raw[8:-3].strip()
    elif raw.startswith("```\n") and raw.endswith("```"):
        raw = raw[4:-3].strip()
    return raw


def parse_selection(raw, allowed):
    try:
        obj = json.loads(raw, object_pairs_hook=unique_object)
        if not isinstance(obj, dict) or set(obj) != {
            "exam_ids",
            "unavailable_requests",
        }:
            return None
        ids, missing = (obj["exam_ids"], obj["unavailable_requests"])
        if (
            not isinstance(ids, list)
            or any((not isinstance(x, str) or x not in allowed for x in ids))
            or len(ids) != len(set(ids))
            or (not isinstance(missing, list))
            or any((not isinstance(x, str) or not x.strip() for x in missing))
        ):
            return None
        return obj
    except (ValueError, TypeError):
        return None


MODALITIES = {
    "mri": "\\bmri\\b|magnetic resonance",
    "ct": "\\bct\\b|computed tomography|computerized tomography",
    "ultrasound": "ultrasound|sonograph|echocardiogra",
    "ecg": "\\becg\\b|\\bekg\\b|electrocardiogra",
    "radiograph": "x[ -]?ray|radiograph|plain films?|chest films?",
    "biopsy": "biops|histolog|histopatholog|tissue sampl",
    "physical": "physical exam|auscultat|palpat|\\binspect|\\bexamine\\b|heart sounds|oral exam|mouth photograph|tongue photograph",
    "blood": "blood tests?|blood count|\\bcbc\\b",
    "vitals": "vital signs?|\\bvitals\\b",
    "growth": "growth (?:chart|curve)",
    "bone_age": "bone age",
    "cytology": "pap smear|cervical cytology",
    "pcr": "\\b(?:HPV\\s+)?PCR\\b|polymerase chain reaction",
}


def modality_guard(query, selection, catalog_ids=None):
    """Remove known modality substitutions and log them; unknown synonyms stay model-routed."""
    from copy import deepcopy

    result = deepcopy(selection)
    requested = {k for k, p in MODALITIES.items() if re.search(p, query, re.I)}
    rejected = []
    if requested:
        for name in result["exam_ids"]:
            words = name.replace("_", " ")
            known = {k for k, p in MODALITIES.items() if re.search(p, words, re.I)}
            if known and (not known & requested):
                rejected.append(name)
    result["exam_ids"] = [x for x in result["exam_ids"] if x not in rejected]
    if rejected and (not result["exam_ids"]) and (not result["unavailable_requests"]):
        result["unavailable_requests"] = [query]
    if catalog_ids is not None and rejected:
        catalog_types = {
            k
            for name in catalog_ids
            for k, p in MODALITIES.items()
            if re.search(p, name.replace("_", " "), re.I)
        }
        for kind in sorted(requested - catalog_types):
            phrase = re.search(MODALITIES[kind], query, re.I).group(0)
            if not any(
                (
                    re.search(MODALITIES[kind], x, re.I)
                    for x in result["unavailable_requests"]
                )
            ):
                result["unavailable_requests"].append(phrase)
    return (result, rejected)


def safe_unavailable_text(query, selection):
    """A model-written missing-test label must not become invented clinical content."""
    from copy import deepcopy

    result = deepcopy(selection)
    neutral = "An additional part of the original request could not be matched."
    result["unavailable_requests"] = list(
        dict.fromkeys(
            (
                x if x.casefold() in query.casefold() else neutral
                for x in result["unavailable_requests"]
            )
        )
    )
    return result
