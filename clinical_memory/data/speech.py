from copy import deepcopy
import hashlib
import json
from pathlib import Path

PERSONAS = {
    "patient": "You speak for yourself. Refer to your own symptoms and history as I/my.",
    "guardian": "You are the child's guardian. Describe the child as my child/my baby; never claim the child's symptoms or age as your own. Do not invent a family relationship.",
    "home_nurse": "You are the home nurse speaking about the patient you care for. The patient's symptoms are not your own.",
}


def source_digest(case):
    source = {
        "id": case["id"],
        "speaker": case["speaker"],
        "opening": case["public_context"]["text"],
        "facts": {f["id"]: f["text"] for f in case["patient_facts"]},
    }
    return hashlib.sha256(
        json.dumps(source, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def apply_profile(case, profile):
    reviewed = profile.get("review_status") == "reviewed"
    verified = (
        profile.get("review_status") == "source_verified"
        and profile.get("validation_method") == "text_rules_v1"
    )
    if profile.get("source_sha256") != source_digest(case) or not (
        reviewed or verified
    ):
        raise ValueError(
            "Speech overlay must be reviewed or source-verified against this exact source"
        )
    expected = {f["id"] for f in case["patient_facts"]} | {"OPENING"}
    if set(profile.get("utterances", {})) != expected:
        raise ValueError("Every history item and opening needs a reviewed utterance")
    if not all(
        (isinstance(s, str) and s.strip() for s in profile["utterances"].values())
    ):
        raise ValueError("Empty/non-text utterance")
    expected_speaker = (
        "guardian" if case["speaker"] == "patient_and_guardian" else case["speaker"]
    )
    if profile.get("speaker") != expected_speaker or expected_speaker not in PERSONAS:
        raise ValueError("Speech role conflicts with the source speaker")
    if "persona" in profile and (
        not isinstance(profile["persona"], str) or not profile["persona"].strip()
    ):
        raise ValueError("Invalid case-specific persona")
    result = deepcopy(case)
    result["dialogue_profile"] = deepcopy(profile)
    return result


def load_profiles(path, cases):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or set(data["profiles"]) != set(cases):
        raise ValueError("Profile schema/case set mismatch")
    return {
        cid: apply_profile(case, data["profiles"][cid]) for cid, case in cases.items()
    }
