import argparse
from collections import Counter
import json
import os
import subprocess
import time
from clinical_memory.models.transformers import TransformersBackend
from clinical_memory.memory.store import MemoryStore, atomic_json
from clinical_memory.models.backend import ClinicalBackend
from clinical_memory.data.augmented import load_set
from clinical_memory.config import (
    ROOT,
    RUNS,
    MODES,
    SETS,
    LEARNING,
    load,
    sha,
    inputs,
    code_hashes,
    runtime_config,
    smoke_run_id,
    train_run_id,
    test_run_id,
    condition,
)
from clinical_memory.data.inputs import safe_id
from clinical_memory.training.records import read_journal
from clinical_memory.training.journal import restore_memory, commit_episode
from clinical_memory.training.learning import learn, visit

COST = ("input_tokens", "output_tokens", "total_seconds")
EMPTY = {"learning": "self", "curriculum": "random", "seed": 42}
SMOKE = {"curriculum": "random", "seed": 42}


def phase_stats(records):
    groups = {}
    for r in records:
        groups.setdefault(r["phase"], []).append(r)
    return {
        phase: {
            "n": len(rows),
            "correct": sum((bool(r["score"]["correct"]) for r in rows)),
            "valid_final": sum((r["summary"]["valid_final"] for r in rows)),
            "image_reached_final": sum(
                (r["metrics"]["final_answer_saw_image"] for r in rows)
            ),
            "examination_reached": sum(
                (r["metrics"]["examination_records_received"] > 0 for r in rows)
            ),
            "no_exam_request": sum((r["metrics"]["exam_requests"] == 0 for r in rows)),
            "collection_actions": sum(
                (r["metrics"]["collection_actions"] for r in rows)
            ),
            "wasted_actions": sum((r["metrics"]["wasted_actions"] for r in rows)),
            "presentation_named": sum(
                (r["summary"]["presentation_named"] for r in rows)
            ),
            "with_lessons": sum((r["summary"]["gathering_lessons"] > 0 for r in rows)),
            "with_notes": sum((r["summary"]["diagnosis_notes"] > 0 for r in rows)),
            "reflection": dict(Counter((r["reflection"]["status"] for r in rows))),
            "lessons_stored": sum((r["lesson"] is not None for r in rows)),
        }
        for phase, rows in groups.items()
    }


def audited(folder, hashes, **expected):
    """A completed run of this exact code whose replay audit passed and still matches its seal."""
    try:
        seal, summary, audit = (
            load(folder / "seal.json"),
            load(folder / "summary.json"),
            load(folder / "audit.json"),
        )
    except OSError:
        return False
    return (
        summary["status"] == "completed"
        and seal["code"] == hashes
        and audit["passed"]
        and (audit["seal_sha256"] == sha(folder / "seal.json"))
        and all((seal[k] == v for k, v in expected.items()))
    )


def smoke_plan(division):
    seen, taught = (division["smoke"]["test1"], division["smoke"]["train"])
    return (
        [["test1_000", c] for c in seen]
        + [["evolution", c] for c in taught]
        + [[f"test1_{len(taught):03d}", c] for c in seen]
    )


def train_plan(division, curriculum):
    """The training patients in one order: the whole schedule of a training run."""
    return [["evolution", c] for c in division["orders"][curriculum]]


def test_plan(which, ids):
    return [[which, c] for c in ids]


def notebook_at(folder, step, capacity):
    """The notebook of a finished training run as it stood after `step` training patients, checked against what
    the run itself recorded: the digest of the notebook it held when it saw its next training patient, or, after
    the last patient, the digest of its final notebook."""
    snapshot, summary = (
        load(folder / "memory_snapshot.json"),
        load(folder / "summary.json"),
    )
    if (
        snapshot["sha256"] != summary["memory_sha256"]
        or MemoryStore(entries=snapshot["entries"], max_lessons=capacity).digest()
        != snapshot["sha256"]
    ):
        raise ValueError("Training memory changed")
    entries = [e for e in snapshot["entries"] if e["created_after_step"] <= step]
    visits = sorted((folder / "episodes").glob("*_evolution_*.json"))
    held = (
        load(visits[step])["trace"]["memory_before_sha256"]
        if step < len(visits)
        else snapshot["sha256"]
        if step == len(visits)
        else None
    )
    if held != MemoryStore(entries=entries, max_lessons=capacity).digest():
        raise ValueError(
            "The notebook rebuilt for this checkpoint is not the one the training run held there"
        )
    return entries


def source(root, chosen, hashes, step, capacity):
    """(notebook entries, training run id) for one doctor, way, order and seed at a checkpoint; ([], None) at step 0."""
    if step == 0:
        return ([], None)
    doctor, learning, curriculum, seed = (
        chosen[k] for k in ("doctor", "learning", "curriculum", "seed")
    )
    run_id = train_run_id(doctor, learning, curriculum, seed)
    if not audited(
        root / RUNS / run_id,
        hashes,
        mode="train",
        doctor=doctor,
        learning=learning,
        curriculum=curriculum,
        seed=seed,
    ):
        raise ValueError(
            "The training run of this doctor, way of learning, order and seed must be completed and audited first"
        )
    return (notebook_at(root / RUNS / run_id, step, capacity), run_id)


def gate(root, a, inp, hashes):
    """Preconditions that never call a model. Returns (the way, order and seed the run is sealed under, why a
    training run may start, the notebook an evaluation reads, the training run it comes from)."""
    from clinical_memory.preflight import preflight

    given = (a.learning, a.curriculum, a.seed)
    if a.mode == "smoke":
        preflight(root, "smoke")
        if a.learning is None or any(
            (x is not None for x in (a.curriculum, a.seed, a.which, a.step))
        ):
            raise ValueError(
                "The readiness check takes a doctor and a way of learning, and nothing else"
            )
        if a.run_id != smoke_run_id(a.doctor, a.learning):
            raise ValueError(
                "The readiness check must be named "
                + smoke_run_id(a.doctor, a.learning)
            )
        return ({"learning": a.learning, **SMOKE}, None, [], None)
    preflight(root, "main")
    decided = load(root / "configs/conditions.json")
    reason, allowed = (
        str(decided.get("reason", "")).strip(),
        decided.get("conditions", []),
    )
    if decided.get("authorized") is not True or not reason:
        raise PermissionError(
            "The full training and the tests wait for the enabled conditions configuration"
        )
    if a.mode == "train":
        if any((x is None for x in given)) or a.which is not None or a.step is not None:
            raise ValueError(
                "A training run takes a way of learning, an order and a seed, and nothing else"
            )
        if condition(a.doctor, a.learning) not in allowed:
            raise PermissionError(
                "The full training waits for the enabled conditions configuration for this doctor and way of learning"
            )
        if not audited(
            root / RUNS / smoke_run_id(a.doctor, a.learning),
            hashes,
            mode="smoke",
            doctor=a.doctor,
            learning=a.learning,
        ):
            raise ValueError(
                "The readiness check of the same code, doctor and way of learning must pass its replay audit first"
            )
        if a.run_id != train_run_id(a.doctor, *given):
            raise ValueError(
                "A training run must be named " + train_run_id(a.doctor, *given)
            )
        return (
            {"learning": a.learning, "curriculum": a.curriculum, "seed": a.seed},
            {"basis": "configured_conditions", "reason": reason},
            [],
            None,
        )
    if decided.get("test_authorized") is not True:
        raise PermissionError("The tests wait for the enabled conditions configuration")
    checkpoints = inp.protocol["main"]["checkpoints"]
    if a.which is None or a.step is None:
        raise ValueError("An evaluation takes --set and --step")
    if a.step not in checkpoints:
        raise ValueError(
            "The step must be a checkpoint of the training: "
            + ", ".join(map(str, checkpoints))
        )
    if a.step == 0:
        if any((x is not None for x in given)):
            raise ValueError(
                "The empty notebook is evaluated once per doctor: give no way of learning, order or seed with --step 0"
            )
        if not any((c.split("/")[0] == a.doctor for c in allowed)):
            raise PermissionError(
                "The tests wait for the enabled conditions configuration for this doctor"
            )
        if not any(
            (
                audited(
                    root / RUNS / smoke_run_id(a.doctor, way),
                    hashes,
                    mode="smoke",
                    doctor=a.doctor,
                )
                for way in LEARNING
            )
        ):
            raise ValueError(
                "A readiness check of this doctor with the same code must pass its replay audit first"
            )
        chosen = dict(EMPTY)
    else:
        if any((x is None for x in given)):
            raise ValueError(
                "--learning, --curriculum and --seed name the training run whose notebook is read"
            )
        if condition(a.doctor, a.learning) not in allowed:
            raise PermissionError(
                "The tests wait for the enabled conditions configuration for this doctor and way of learning"
            )
        chosen = {"learning": a.learning, "curriculum": a.curriculum, "seed": a.seed}
    if a.run_id != test_run_id(a.which, a.doctor, a.step, *given):
        raise ValueError(
            "This evaluation must be named "
            + test_run_id(a.which, a.doctor, a.step, *given)
        )
    entries, training_run = source(
        root,
        {"doctor": a.doctor, **chosen},
        hashes,
        a.step,
        inp.protocol["memory"]["max_lessons"],
    )
    return (chosen, None, entries, training_run)


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run-id", required=True)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--doctor", choices=["2B", "4B", "9B"], required=True)
    p.add_argument("--learning", choices=LEARNING)
    p.add_argument("--curriculum", choices=["random", "easy_to_hard", "hard_to_easy"])
    p.add_argument("--seed", type=int, choices=[42, 43, 44])
    p.add_argument("--set", dest="which", choices=SETS)
    p.add_argument("--step", type=int)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args(argv)
    safe_id(a.run_id)
    inp = inputs(ROOT)
    protocol, study, division = (inp.protocol, inp.study, inp.division)
    hashes = code_hashes(ROOT)
    chosen, why, entries, training_run = gate(ROOT, a, inp, hashes)
    cases, golds, augmentation = (inp.cases, inp.golds, None)
    if a.mode == "smoke":
        plan = smoke_plan(division)
    elif a.mode == "train":
        plan = train_plan(division, a.curriculum)
    elif a.which == "test1":
        plan = test_plan("test1", division["test1"])
    else:
        cases, golds, augmentation = load_set(ROOT, inp)
        plan = test_plan("test2", list(cases))
    taught = [cid for phase, cid in plan if phase == "evolution"]
    runtime, settings = (runtime_config(ROOT), protocol["memory"])
    seal = {
        "schema": 1,
        "version": protocol["version"],
        "mode": a.mode,
        "set": a.which,
        "doctor": a.doctor,
        **chosen,
        "protocol": protocol,
        "study": study,
        "schedule": plan,
        "manifest_sha256": sha(ROOT / study["manifest"]),
        "data_hashes": inp.frozen["source_hashes"],
        "division_sha256": protocol["pool"]["division_sha256"],
        "runtime_config": runtime,
        "code": hashes,
        "training_run": training_run,
        "memory_step": a.step,
        "initial_memory": entries,
        "augmentation_sha256": augmentation,
    }
    seal = json.loads(json.dumps(seal))
    out = ROOT / RUNS / a.run_id
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    if (out / "seal.json").exists():
        if not a.resume:
            raise ValueError("This run already exists; continue it with --resume")
        if load(out / "seal.json") != seal:
            raise ValueError("Resume configuration/data/code changed")
    else:
        atomic_json(out / "seal.json", seal)
    provenance = out / "provenance.json"
    if provenance.exists():
        source_commit = load(provenance)["source_commit"]
    else:
        try:
            source_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            source_commit = "source-archive"
        atomic_json(provenance, {"source_commit": source_commit, "main_basis": why})
    records = read_journal(out / "episodes")
    keys = [phase + ":" + cid for phase, cid in plan]
    if taught:
        memory = restore_memory(
            out / "memory.jsonl", records, plan, settings["max_lessons"]
        )
    else:
        if set(records) != set(keys[: len(records)]) or any(
            (r.get("lesson") for r in records.values())
        ):
            raise ValueError("Invalid evaluation journal")
        memory = MemoryStore(
            entries=seal["initial_memory"],
            readonly=True,
            max_lessons=settings["max_lessons"],
        )

    def finish(summary, calls):
        ordered = [records[key] for key in keys if key in records]
        summary.update(
            source_commit=source_commit,
            planned_episodes=len(plan),
            completed_episodes=len(ordered),
            memory_items=len(memory.entries()),
            memory_sha256=memory.digest(),
            phases=phase_stats(ordered),
        )
        if calls is not None:
            summary["model_calls"] = len(calls)
            summary["all_call_cost"] = {k: sum((c[k] for c in calls)) for k in COST}
        atomic_json(out / "summary.json", summary)

    if len(records) == len(plan):
        summary = load(out / "summary.json") if (out / "summary.json").exists() else {}
        summary.update(status="completed")
        summary.pop("error", None)
        summary.pop("error_type", None)
        calls = [
            json.loads(s)
            for s in (out / "private_calls.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        atomic_json(out / "memory_snapshot.json", memory.snapshot())
        finish(summary, calls)
        print("Already complete; summary recovered, no model loaded.", flush=True)
        return

    def backend(size):
        target = out / ("model_" + size)
        target.mkdir(exist_ok=True)
        return TransformersBackend({**runtime, **study["models"][size]}, ROOT, target)

    started, combined = (time.monotonic(), None)
    summary = {
        "status": "running",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "mode": a.mode,
        "set": a.which,
        "development_only": a.mode == "smoke",
    }
    try:
        primary, measurement = (backend("4B"), backend("9B"))
        doctor = (
            primary
            if a.doctor == "4B"
            else measurement
            if a.doctor == "9B"
            else backend(a.doctor)
        )
        combined = ClinicalBackend(
            primary, measurement, out, doctor=doctor, seed=seal["seed"]
        )
        combined.measurement_scope_guard = study.get(
            "measurement_catalog", "source_topics_v1"
        )
        for i, (phase, cid) in enumerate(plan):
            key = phase + ":" + cid
            if key in records:
                continue
            combined.start_episode(phase, cid)
            before, limit, first_call = (
                memory.digest(),
                protocol["max_collection_actions"],
                len(combined.calls),
            )
            if phase == "evolution":
                seen, lesson, info = learn(
                    combined,
                    cases[cid],
                    golds[cid],
                    memory.readonly_view(),
                    settings,
                    limit,
                    seal["learning"],
                    taught.index(cid) + 1,
                )
                rec = {**seen, "phase": phase, "lesson": lesson, "reflection": info}
            else:
                rec = visit(
                    combined,
                    cases[cid],
                    golds[cid],
                    memory.readonly_view(),
                    settings,
                    limit,
                    seal["learning"],
                )
                rec.update(
                    phase=phase, lesson=None, reflection={"status": "not_called"}
                )
            rec["last_call_index"] = len(combined.calls)
            rec["all_calls_cost"] = {
                k: sum((c[k] for c in combined.calls[first_call:])) for k in COST
            }
            if before != memory.digest():
                raise ValueError("Memory changed before commit")
            commit_episode(
                out / "episodes" / f"{i:04d}_{phase}_{cid}.json", rec, memory
            )
            records[key] = rec
            if phase != "evolution" and before != memory.digest():
                raise ValueError("Evaluation wrote memory")
            print(
                json.dumps(
                    {
                        "step": i + 1,
                        "total": len(plan),
                        "phase": phase,
                        "id": cid,
                        "correct": rec["score"]["correct"],
                        "image": rec["metrics"]["final_answer_saw_image"],
                        "actions": rec["summary"]["actions"],
                        "wasted": rec["metrics"]["wasted_actions"],
                        "lessons": rec["summary"]["gathering_lessons"],
                        "notes": rec["summary"]["diagnosis_notes"],
                        "reflection": rec["reflection"]["status"],
                        "memory": len(memory.entries()),
                    }
                ),
                flush=True,
            )
        summary["status"] = "completed"
        atomic_json(out / "memory_snapshot.json", memory.snapshot())
    except Exception as error:
        summary.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        summary["wall_seconds_this_session"] = round(time.monotonic() - started, 3)
        finish(summary, combined.calls if combined else None)


if __name__ == "__main__":
    main()
