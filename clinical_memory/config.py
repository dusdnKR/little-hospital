from collections import namedtuple
from clinical_memory.data.io import read_rows
from clinical_memory.data.inputs import ROOT, load, sha
from clinical_memory.data.inputs import (
    inputs as load_inputs,
    code_hashes as source_hashes,
    runtime_config as load_runtime,
)
from clinical_memory.data.pool import digest, divide, share

RUNS = "runs/experiments"
MODES = ("smoke", "train", "test")
SETS = ("test1", "test2")
LEARNING = ("self", "key", "key_dx")
POLICIES = ("settle", "discard")
RUNTIME_FILES = (
    "clinical_memory/config.py",
    "configs/experiment.json",
    "configs/disease_groups.json",
    "clinical_memory/data/pool.py",
    "clinical_memory/memory/diagnosis.py",
    "clinical_memory/models/clinical.py",
    "clinical_memory/evaluation/episode.py",
    "clinical_memory/memory/reflection.py",
    "clinical_memory/training/learning.py",
    "clinical_memory/training/journal.py",
    "clinical_memory/data/augmented.py",
    "clinical_memory/training/run.py",
    "clinical_memory/memory/presentation.py",
    "clinical_memory/memory/feedback.py",
    "clinical_memory/memory/matching.py",
    "clinical_memory/models/backend.py",
    "clinical_memory/environment/finish.py",
    "clinical_memory/memory/evidence.py",
)
NEW_TOKENS = {"reasoning": 256, "presentation": 64, "reflection": 1024}
Inputs = namedtuple("Inputs", "protocol study frozen cases golds division")


def validate_protocol(protocol):
    """Refuse a protocol file that describes something the code does not do."""
    from clinical_memory.models.clinical import EXAMINATION_NOTE
    from clinical_memory.prompts.reasoning import REASONING_WORDS
    from clinical_memory.prompts.presentation import (
        MIN_KEYWORDS,
        MAX_KEYWORDS,
        KEYWORD_WORDS,
    )
    from clinical_memory.prompts.reflection import COLLECT_WORDS, DIAGNOSE_WORDS

    memory, learning, decision, spec = (
        protocol["memory"],
        protocol["learning"],
        protocol["decision"],
        protocol["pool"],
    )
    if (
        protocol["version"] != "clinical_memory"
        or tuple(learning["conditions"]) != LEARNING
        or memory["writer"] != "doctor_reflection"
        or (memory["reflection_model"] != "doctor")
        or (memory["global_guidance"] is not False)
        or (memory["visits_per_training_patient"] != 1)
        or (protocol["environment"]["final_input"] != "evidence")
        or (protocol["collection"]["added_instruction"] != EXAMINATION_NOTE.strip())
        or (decision["steps"] != ["reasoning", "final"])
        or (decision["reasoning_max_words"] != REASONING_WORDS)
        or (
            (
                decision["reasoning_max_new_tokens"],
                memory["presentation_max_new_tokens"],
            )
            != (NEW_TOKENS["reasoning"], NEW_TOKENS["presentation"])
        )
        or (
            (
                memory["keywords_min"],
                memory["keywords_max"],
                memory["keyword_max_words"],
            )
            != (MIN_KEYWORDS, MAX_KEYWORDS, KEYWORD_WORDS)
        )
        or (
            (memory["collect_target_words"], memory["diagnose_target_words"])
            != (COLLECT_WORDS, DIAGNOSE_WORDS)
        )
        or (memory["lesson_word_limit_enforced"] is not False)
        or (memory["reflection_max_new_tokens"] != NEW_TOKENS["reflection"])
        or (memory["lesson_format_policy"] not in POLICIES)
    ):
        raise ValueError("Protocol does not describe the implementation")
    for key in ("retrieval_top_k", "retrieval_min_terms", "notes_top_k"):
        if type(memory[key]) is not int or memory[key] < 1:
            raise ValueError("Invalid memory setting: " + key)
    if (
        not 0 <= memory["retrieval_threshold"] <= 1
        or not 0 < memory["notes_threshold"] <= 1
    ):
        raise ValueError("Invalid memory setting: a threshold")
    patients = spec["patients"]
    if (
        spec["test1"] != share(spec["test1_share"], patients)
        or spec["test2"] != share(spec["test2_share"], patients)
        or spec["train"] != patients - spec["test1"]
        or (spec["test2"] > spec["train"])
        or (memory["max_lessons"] != spec["train"])
    ):
        raise ValueError(
            "The sizes of the division do not follow from the stated shares"
        )
    checks = protocol["main"]["checkpoints"]
    if checks[0] != 0 or checks[-1] != spec["train"] or sorted(set(checks)) != checks:
        raise ValueError(
            "The checkpoints must run from no training patient to all of them"
        )


def inputs(root=ROOT):
    base = load_inputs(root)
    protocol = load(root / "configs/experiment.json")
    validate_protocol(protocol)
    levels = {
        cid: row["correct_count"]
        for cid, row in read_rows(
            root / base.frozen["cohort"] / "difficulty.jsonl"
        ).items()
    }
    groups = load(root / protocol["pool"]["same_disease_groups"])["groups"]
    division = divide(base.frozen, base.golds, levels, groups, protocol)
    if digest(division) != protocol["pool"]["division_sha256"]:
        raise ValueError(
            "The division of the patients is not the one the protocol pins"
        )
    return Inputs(protocol, base.study, base.frozen, base.cases, base.golds, division)


def code_hashes(root=ROOT):
    return {**source_hashes(root), **{name: sha(root / name) for name in RUNTIME_FILES}}


def runtime_config(root=None):
    config = load_runtime(root)
    config["max_new_tokens"].update(NEW_TOKENS)
    return config


def smoke_run_id(doctor, learning):
    return f"cm_smoke_{doctor.lower()}_{learning}"


def train_run_id(doctor, learning, curriculum, seed):
    return f"cm_{doctor.lower()}_{learning}_{curriculum}_s{seed}"


def test_run_id(which, doctor, step, learning=None, curriculum=None, seed=None):
    """An evaluation of one test set: with an empty notebook once per doctor, else with the notebook of one training
    run after `step` training patients."""
    tag = doctor.lower()
    return (
        f"cm_{which}_{tag}_t000"
        if step == 0
        else f"cm_{which}_{tag}_{learning}_{curriculum}_s{seed}_t{step:03d}"
    )


def condition(doctor, learning):
    """Return the model/learning identifier used by the execution allowlist."""
    return doctor + "/" + learning
