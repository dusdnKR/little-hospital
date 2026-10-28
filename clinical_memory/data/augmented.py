from copy import deepcopy
import difflib
import hashlib
import json
from pathlib import Path
import re
from clinical_memory.data.speech import source_digest
from clinical_memory.memory.presentation import RESERVED

IMAGES = "data/images"


def aug_id(cid, offset):
    """The ID of the augmented version of a patient: the same form with the number moved by `offset`."""
    found = re.fullmatch("(.*?)(\\d+)", cid)
    if not found:
        raise ValueError("A patient ID without a number cannot be augmented")
    return f"{found.group(1)}{int(found.group(2)) + offset}"


def aug_image(name, cid, new_id):
    """The file name of an augmented image: that of the source with the patient ID replaced."""
    if not name.startswith(cid):
        raise ValueError("An image name must start with the ID of its patient")
    return new_id + name[len(cid) :]


def numbers(text):
    return sorted(re.findall("\\d+(?:[.,]\\d+)*", text))


def similarity(original, rewritten):
    """How much of the two statements is the same run of characters, between 0 and 1."""
    return difflib.SequenceMatcher(
        None,
        " ".join(original.split()).casefold(),
        " ".join(rewritten.split()).casefold(),
    ).ratio()


def opening_errors(original, rewritten, rule):
    """The rules a rewritten opening statement breaks. Meaning cannot be checked by code; these are the checks that can."""
    if not isinstance(rewritten, str) or not rewritten.strip():
        return ["It is empty."]
    first, second = (" ".join(original.split()), " ".join(rewritten.split()))
    errors, words, source_words = ([], len(second.split()), len(first.split()))
    low, high = rule["opening_length_ratio"]
    if first.casefold() == second.casefold():
        errors.append("It is the text of the source patient.")
    if numbers(first) != numbers(second):
        errors.append("A number was changed, added or lost.")
    if RESERVED.search(second):
        errors.append("It holds a case identifier or an answer letter.")
    if words < rule["opening_min_words"] or not low * source_words <= words <= max(
        high * source_words, rule["opening_min_words"]
    ):
        errors.append("Its length is too far from that of the source.")
    if (
        source_words >= rule["opening_similarity_from_words"]
        and similarity(first, second) > rule["opening_max_similarity"]
    ):
        errors.append("It is almost the text of the source patient.")
    return errors


def build_case(source, new_id, opening):
    """The augmented patient: the source patient under a new ID with the rewritten opening statement and the names
    of the transformed images. The patient's own answers and the examination records are those of the source."""
    case, old = (deepcopy(source), source["id"])
    case.update(
        id=new_id,
        augmented_from=old,
        public_context={**case["public_context"], "text": opening},
    )
    for exam in case["exams"]:
        exam["images"] = [aug_image(name, old, new_id) for name in exam["images"]]
    profile = case.get("dialogue_profile")
    if profile:
        spoken = profile["utterances"]["OPENING"]
        profile["utterances"] = {
            key: opening if key == "OPENING" or said == spoken else said
            for key, said in profile["utterances"].items()
        }
        profile.update(
            review_status="reviewed",
            validation_method="augmented_cm_opening_check",
            source_sha256=source_digest(case),
        )
    return case


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_set(root, inp):
    """(cases, golds, digest of the manifest) of the augmented patients, in the order of the division."""
    root, rule = (Path(root), inp.protocol["test2"])
    path = root / rule["folder"] / "manifest.json"
    if not path.exists():
        raise ValueError("The augmented patients of test 2 have not been built yet")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("version") != rule["version"]
        or manifest.get("division_sha256") != inp.protocol["pool"]["division_sha256"]
        or manifest.get("image") != rule["image"]
        or (
            [p["source_id"] for p in manifest["patients"]]
            != inp.division["test2_sources"]
        )
    ):
        raise ValueError(
            "The augmented patients do not belong to this division and recipe"
        )
    cases, golds = ({}, {})
    for item in manifest["patients"]:
        source, new_id = (
            inp.cases[item["source_id"]],
            aug_id(item["source_id"], rule["id_offset"]),
        )
        problems = opening_errors(
            source["public_context"]["text"], item["opening"], rule
        )
        names = [name for exam in source["exams"] for name in exam["images"]]
        if item["id"] != new_id or new_id in inp.cases or new_id in cases or problems:
            raise ValueError(
                "Invalid augmented patient " + new_id + ": " + " ".join(problems)
            )
        if [(im["source"], im["name"]) for im in item["images"]] != [
            (name, aug_image(name, source["id"], new_id)) for name in names
        ]:
            raise ValueError(
                "The images of an augmented patient are not those of its source: "
                + new_id
            )
        for im in item["images"]:
            made = root / IMAGES / im["name"]
            if (
                not made.is_file()
                or file_sha(made) != im["sha256"]
                or im["sha256"] == file_sha(root / IMAGES / im["source"])
            ):
                raise ValueError(
                    "An augmented image is missing, changed or the source image itself: "
                    + im["name"]
                )
        cases[new_id] = build_case(source, new_id, " ".join(item["opening"].split()))
        golds[new_id] = {**deepcopy(inp.golds[item["source_id"]]), "id": new_id}
    return (cases, golds, file_sha(path))
