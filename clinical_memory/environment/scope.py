import re
import hashlib
from copy import deepcopy

TOPICS = {
    "height": "\\bheight\\b|\\blength\\b",
    "weight": "\\bweight\\b",
    "orthostatic": "orthostatic|postural|standing.{0,50}(?:pressure|pulse)",
    "neurological": "neurolog|cranial|strength|reflex|sensation|gait|pupil|consciousness|oriented",
    "eye": "\\beye|ocular|ophthal|vision|retinal|fundus|proptosis",
    "ear": "\\bear\\b|otologic|otoscop|tympan|hearing",
    "skin": "skin|rash|lesion|scalp|hair|cutaneous|bullae",
    "hair_scalp": "scalp|hair",
    "swelling": "swelling|edema",
    "range_of_motion": "range of motion|\\bROM\\b",
    "tenderness": "tenderness|tender to",
    "cardiac": "cardiac|heart|murmur|\\bS[1234]\\b|gallop",
    "lung": "lung|breath|crackle|rales|respiratory|wheez",
    "abdomen": "abdomin|liver|bowel|hepat|spleen|rectal",
    "pelvis": "pelvic|vagina|cervi[xc]|uter|adnex",
    "wrist": "wrist|radial|ulnar",
    "knee": "knee|patell|lachman",
    "temperature": "temperature|afebrile|febrile",
    "blood_pressure": "blood pressure|mm\\s*Hg|mmHg",
    "heart_rate": "heart rate|pulse|beats per minute",
    "respiratory_rate": "respiratory rate|respirations",
    "oxygen_saturation": "oxygen saturation|SpO2|O2 saturation",
}


def with_scopes(cases, annotations=None):
    result = deepcopy(cases)
    for c in result.values():
        for e in c["exams"]:
            text = e["id"].replace("_", " ") + " " + e["text"]
            topics = [k for k, p in TOPICS.items() if re.search(p, text, re.I)]
            annotation = (annotations or {}).get(c["id"], {}).get(e["id"])
            if annotation:
                opening = c["public_context"].get(
                    "source_text", c["public_context"]["text"]
                )
                if (
                    hashlib.sha256(opening.encode()).hexdigest()
                    != annotation["opening_sha256"]
                    or hashlib.sha256(e["text"].encode()).hexdigest()
                    != annotation["exam_sha256"]
                    or any((t not in TOPICS for t in annotation["topics"]))
                ):
                    raise ValueError(
                        "Reviewed examination annotation does not match source"
                    )
                topics = list(dict.fromkeys(topics + annotation["topics"]))
            e["request_scope"] = {
                "topics": topics,
                "image_available": bool(e["images"]),
                "coverage": "Only the source examination topics are available; unlisted measurements are unknown.",
            }
    return result


def scope_guard(query, selection, catalog, version="source_topics_v1"):
    result = deepcopy(selection)
    by_id = {e["id"]: e for e in catalog}
    rejected = []
    for eid in selection["exam_ids"]:
        item = by_id[eid]
        if "scope" not in item:
            continue
        topics = set(item["scope"]["topics"])
        q = query.lower()
        words = eid.replace("_", " ")
        vitals = bool(re.search("vital", words))
        physical = bool(re.search("physical|exam", words))
        bad = (
            vitals
            and re.search(TOPICS["orthostatic"], q, re.I)
            and ("orthostatic" not in topics)
        )
        if (
            version in {"source_topics_v2", "source_topics_v3"}
            and vitals
            and re.search("\\bstanding\\b|\\bupright\\b", q)
            and ("orthostatic" not in topics)
        ):
            bad = True
        requested_size = {
            k for k in ["height", "weight"] if re.search(TOPICS[k], q, re.I)
        }
        if physical and requested_size and (not topics & requested_size):
            if not re.search("physical examination|physical exam", q):
                bad = True
        if (
            physical
            and re.search("neurological exam|neurologic exam", q)
            and ("neurological" not in topics)
        ):
            bad = True
        if (
            physical
            and re.search("pelvic exam|vaginal exam", q)
            and ("pelvis" not in topics)
        ):
            bad = True
        if bad:
            rejected.append(eid)
    result["exam_ids"] = [x for x in result["exam_ids"] if x not in rejected]
    if rejected and (not result["unavailable_requests"]):
        result["unavailable_requests"] = [query]
    return (result, rejected)
