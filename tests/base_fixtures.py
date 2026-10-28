from copy import deepcopy
import json
from pathlib import Path
from clinical_memory.memory.store import atomic_json
from clinical_memory.data.speech import apply_profile, source_digest
from clinical_memory.data.inputs import RUNTIME_FILES, sha

REAL = Path(__file__).resolve().parents[1]
OPENING = "I am a 40-year-old man with knee pain."
FILED = "Source image for the requested examination."
LESSON = "When knee pain follows a fall, history alone did not settle the diagnosis; request a knee radiograph."
GUIDANCE = "Request an examination before finishing; do not answer from history alone."
GUIDANCE_AFTER_IMAGE = "Request an examination before finishing; the image I received settled the question."


def make_case(cid, exam_id="radiograph", hidden=True, opening=OPENING, exam_text=FILED):
    facts = [{"id": "P1", "text": "Patient " + cid + " has knee pain."}]
    if hidden:
        facts.append({"id": "P2", "text": "The pain began after a fall."})
    return {
        "id": cid,
        "speaker": "patient",
        "task": "Determine the most likely diagnosis for this case.",
        "public_context": {"text": opening, "initially_disclosed_fact_ids": ["P1"]},
        "patient_facts": facts,
        "exams": [{"id": exam_id, "text": exam_text, "images": [cid + ".jpeg"]}],
        "unavailable_response": "This information is not provided in the case.",
    }


def profile_for(case):
    utterances = {"OPENING": case["public_context"]["text"], "P1": "I have knee pain."}
    if any((f["id"] == "P2" for f in case["patient_facts"])):
        utterances["P2"] = "It began after I fell."
    return {
        "speaker": "patient",
        "review_status": "reviewed",
        "source_sha256": source_digest(case),
        "utterances": utterances,
    }


def with_profile(case):
    return apply_profile(case, profile_for(case))


GOLD = {"options": {k: "Diagnosis " + k for k in "ABCDE"}, "label": "A"}


class ScriptedBackend:
    """Deterministic replies in the real call-log format.

    `fail_at` simulates a crash at that call. `abstain_for` lists episodes
    (phase:case) in which the scripted reflection writes nothing, and
    `keep_guidance_for` those in which it writes a lesson but leaves its guidance
    as it is. An instance may set `reflection` to return one fixed reflection.
    """

    count, fail_at, abstain_for, keep_guidance_for, reflection = (
        0,
        None,
        frozenset(),
        frozenset(),
        None,
    )

    def __init__(self, config, root, output):
        self.config, self.root, self.calls, self.case_id, self.generation_seed = (
            dict(config),
            Path(root),
            [],
            None,
            0,
        )

    def set_seed(self, seed):
        self.generation_seed = seed

    def reply(self, base, payload, images):
        record = payload.get("record", payload)
        if base == "doctor":
            history = record["history"]
            if not history:
                return {"type": "ask_patient", "query": "What symptoms do you have?"}
            if ("memory" in record or "guidance" in record) and (
                not any((e["type"] == "request_exam" for e in history))
            ):
                return {"type": "request_exam", "query": "Please obtain a radiograph."}
            return {"type": "finish"}
        if base == "patient":
            return {"fact_ids": [f["id"] for f in record["facts"] if f["id"] == "P2"]}
        if base == "measurement":
            ids = [
                e["id"] for e in record["available_exams"] if "radiograph" in e["id"]
            ]
            return {
                "exam_ids": ids,
                "unavailable_requests": [] if ids else [record["query"]],
            }
        if base == "final":
            return {"answer": "A" if images else "B"}
        if base == "reflection":
            if self.reflection is not None:
                return self.reflection
            if self.case_id in ScriptedBackend.abstain_for:
                return {"local_lesson": "", "global_guidance": "", "evidence_refs": []}
            guidance = (
                ""
                if self.case_id in ScriptedBackend.keep_guidance_for
                else GUIDANCE_AFTER_IMAGE
                if images
                else GUIDANCE
            )
            return {
                "local_lesson": LESSON,
                "global_guidance": guidance,
                "evidence_refs": ["opening", record["encounter"][0]["ref"]],
            }
        raise AssertionError("Unexpected role " + base)

    def generate(self, role, system, payload, images):
        ScriptedBackend.count += 1
        if ScriptedBackend.fail_at == ScriptedBackend.count:
            raise RuntimeError("simulated crash")
        base = role.removesuffix("_repair")
        text = json.dumps(self.reply(base, payload, images))
        settings = {"do_sample": self.config["do_sample"]}
        if self.config["do_sample"]:
            settings.update(
                {k: self.config[k] for k in ("temperature", "top_p", "top_k")}
            )
        settings.update(
            enable_thinking=False,
            max_new_tokens=self.config["max_new_tokens"][base],
            run_seed=self.generation_seed,
        )
        self.calls.append(
            {
                "call_index": len(self.calls) + 1,
                "case_id": self.case_id,
                "role": role,
                "generation_settings": settings,
                "system_prompt": system,
                "payload": deepcopy(payload),
                "response": text,
                "input_tokens": 10,
                "output_tokens": 5,
                "total_seconds": 0.01,
                "images": [
                    {"name": n, "sha256": sha(self.root / "data/images" / n)}
                    for n in images
                ],
                "image_grid_thw": [[1, 2, 2] for _ in images],
                "input_shapes": {"pixel_values": [4 * len(images), 8]},
                "hit_token_cap": False,
            }
        )
        return text


def build_project(root):
    root = Path(root)
    evolution = [f"E{i}" for i in range(1, 9)]
    splits = {
        "evolution": evolution,
        "dev": ["D1"],
        "reserve": ["R1"],
        "test": ["T1", "T2"],
    }
    cases = [
        make_case(c, "source_image" if c == "R1" else "radiograph")
        for ids in splits.values()
        for c in ids
    ]
    cohort = root / "data/cohort"
    for folder in (
        cohort,
        root / "data/frozen",
        root / "configs",
        root / "data/images",
        root / "clinical_memory/data",
        root / "clinical_memory/environment",
        root / "results",
    ):
        folder.mkdir(parents=True, exist_ok=True)

    def rows(path, items):
        path.write_text(
            "".join((json.dumps(x) + "\n" for x in items)), encoding="utf-8"
        )

    rows(cohort / "cases.jsonl", cases)
    rows(cohort / "private_gold.jsonl", [{"id": c["id"], **GOLD} for c in cases])
    rows(
        cohort / "difficulty.jsonl",
        [{"id": c["id"], "correct_count": 3} for c in cases],
    )
    atomic_json(
        cohort / "speech_profiles.json",
        {"schema_version": 1, "profiles": {c["id"]: profile_for(c) for c in cases}},
    )
    for c in cases:
        (root / "data/images" / (c["id"] + ".jpeg")).write_bytes(
            b"image " + c["id"].encode()
        )
    names = [
        "cases.jsonl",
        "private_gold.jsonl",
        "difficulty.jsonl",
        "speech_profiles.json",
    ]
    manifest = {
        "cohort": "data/cohort",
        "source_hashes": {"data/cohort/" + n: sha(cohort / n) for n in names},
        "groups": {c["id"]: c["id"] for c in cases},
        "patients": [c for part in splits.values() for c in part],
        "splits": splits,
        "orders": {
            k: evolution if k != "hard_to_easy" else evolution[::-1]
            for k in ("random", "easy_to_hard", "hard_to_easy")
        },
    }
    atomic_json(root / "data/frozen/manifest.json", manifest)
    atomic_json(
        root / "configs/models.json",
        {
            "manifest": "data/frozen/manifest.json",
            "main_authorized": True,
            "measurement_catalog": "source_topics_v3",
            "models": {
                s: {"model_id": "model-" + s, "model_revision": "rev-" + s}
                for s in ("2B", "4B", "9B")
            },
        },
    )
    (root / "configs/inference.json").write_bytes(
        (REAL / "configs/inference.json").read_bytes()
    )
    for name in RUNTIME_FILES:
        if not (root / name).exists():
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_text(
                "synthetic code seal for " + name, encoding="utf-8"
            )
    return splits
