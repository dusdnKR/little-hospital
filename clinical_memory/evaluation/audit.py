import argparse
from collections import Counter
import json
from pathlib import Path
from clinical_memory.models.backend import call_seed
from clinical_memory.memory.evidence import evidence_view
from clinical_memory.memory.store import MemoryStore, atomic_json
from clinical_memory.analysis.validity import invalid_counts
from clinical_memory.models.backend import OWN, SAMPLED, SAMPLING
from clinical_memory.data.augmented import load_set
from clinical_memory.config import ROOT, RUNS, load, sha, inputs, code_hashes
from clinical_memory.data.inputs import safe_id
from clinical_memory.training.learning import NOTES_SHOWN, SHEET_SHOWN, learn, visit
from clinical_memory.memory.feedback import answer_key
from clinical_memory.training.run import smoke_plan, source, test_plan, train_plan

SHEET_KEYS = {
    "confirmed_diagnosis",
    "your_diagnosis_was_confirmed",
    "history_you_did_not_hear",
    "examinations_you_did_not_receive",
}
EVIDENCE_KEYS = {"public_context", "task", "history", "options"}
RECORD_KEYS = {
    "opening",
    "task",
    "presentation",
    "encounter",
    "reasoning",
    "submitted_diagnosis",
}
SMOKE_MAX_UNUSABLE_SHARE = 0.1
SMOKE_MAX_UNNAMED_SHARE = 0.5


def require(value, message):
    if not value:
        raise ValueError(message)


class Replay:
    measurement_protocol = "multi_exam_v1"

    def __init__(self, rows, seal, root):
        self.rows, self.seal, self.root, self.calls = (rows, seal, root, [])
        self.measurement_scope_guard = seal["study"].get(
            "measurement_catalog", "source_topics_v1"
        )

    def start_episode(self, phase, cid, position):
        self.phase, self.cid, self.role_counts, self.shown = (phase, cid, {}, [])
        self.reasoned = self.final = None
        self.calls = self.rows[:position]

    def boundaries(self, role, base, record, images, response):
        learning = self.seal["learning"]
        if base not in {"final", "reasoning"}:
            require(
                "options" not in record,
                "Options leaked outside the reasoning and the answer",
            )
        if base != "doctor":
            require(
                "memory" not in record,
                "The gathering part of a lesson reached a role other than the collecting doctor",
            )
        require("guidance" not in record, "There is no global guidance in this study")
        if base not in {"final", "reasoning"} or not NOTES_SHOWN[learning]:
            require(
                "notes" not in record,
                "A diagnosis note reached a call that must not read it",
            )
        elif "notes" in record:
            require(
                isinstance(record["notes"], list)
                and record["notes"]
                and all(
                    (
                        isinstance(n, dict) and set(n) == {"diagnosis", "note"}
                        for n in record["notes"]
                    )
                ),
                "A note holds the diagnosis it is filed under and the doctor's words, nothing else",
            )
        if base in {"patient", "measurement", "presentation"}:
            require(not images, "A role that reads no image received one")
        if base != "reflection":
            require(
                "answer_key" not in record and (not SHEET_KEYS & set(record)),
                "The answer sheet reached a role other than reflection",
            )
        if base == "presentation":
            require(
                set(record) == {"public_context", "task"},
                "The presentation must be named from the opening statement only",
            )
        if base == "reasoning":
            require(
                set(record) - {"notes"} == EVIDENCE_KEYS,
                "The reasoning must read the evidence view and at most the notes",
            )
            self.reasoned = (record, images, response.strip())
        if base == "final":
            require(
                self.reasoned is not None
                and {k: v for k, v in record.items() if k != "reasoning"}
                == self.reasoned[0]
                and (record.get("reasoning") == self.reasoned[2])
                and (images == self.reasoned[1]),
                "The answer must read what the reasoning read and the reasoning written from it, nothing else",
            )
            if role == "final":
                self.final = (record, images)
        if base != "reflection":
            return
        sheet = SHEET_SHOWN[learning]
        require(
            self.phase == "evolution" and self.final is not None,
            "Reflection ran outside a training visit",
        )
        require(
            set(record) == RECORD_KEYS | ({"answer_key"} if sheet else set()),
            "Unexpected reflection input",
        )
        seen, seen_images = self.final
        steps = [
            {k: v for k, v in step.items() if k != "ref"}
            for step in record["encounter"]
        ]
        own = evidence_view(
            {
                "public_context": record["opening"],
                "task": record["task"],
                "history": steps,
            }
        )
        require(
            own["history"] == seen["history"]
            and (record["opening"], record["task"])
            == (seen["public_context"], seen["task"])
            and (record["reasoning"] == seen["reasoning"])
            and (record["submitted_diagnosis"] in seen["options"].values())
            and (images[: len(seen_images)] == seen_images),
            "Reflection read a record of the visit that differs from what the doctor saw",
        )
        if sheet:
            require(set(record["answer_key"]) == SHEET_KEYS, "Unexpected answer sheet")
            missed = [
                n
                for e in record["answer_key"]["examinations_you_did_not_receive"]
                for n in e["images"]
            ]
            require(
                images[len(seen_images) :]
                == [n for n in missed if n not in seen_images],
                "The images of the answer sheet must be those of the examinations that were not received",
            )
            self.shown.append(record["answer_key"])
        else:
            require(
                images == seen_images,
                "Without the answer sheet reflection sees only the images the doctor received",
            )

    def generate(self, role, system, payload, images):
        require(
            len(self.calls) < len(self.rows), "Replay requested an extra model call"
        )
        row, seal = (self.rows[len(self.calls)], self.seal)
        base = role.removesuffix("_repair")
        self.role_counts[role] = self.role_counts.get(role, 0) + 1
        sampling = self.phase == "evolution" and base in SAMPLED
        settings = {
            "do_sample": sampling,
            "enable_thinking": False,
            "max_new_tokens": seal["runtime_config"]["max_new_tokens"][base],
            "run_seed": call_seed(seal["seed"], self.cid, role, self.role_counts[role]),
        }
        if sampling:
            settings.update(SAMPLING)
        require(
            row["generation_settings"] == settings
            and row["case_id"] == self.phase + ":" + self.cid,
            "Generation seed/settings/episode mismatch",
        )
        size = (
            seal["doctor"] if base in OWN else "9B" if base == "measurement" else "4B"
        )
        model = seal["study"]["models"][size]
        require(
            all((row[k] == model[k] for k in ("model_id", "model_revision"))),
            "Wrong role model",
        )
        for key, value in (
            ("role", role),
            ("system_prompt", system),
            ("payload", payload),
        ):
            require(
                row[key] == value,
                f"Replay mismatch at call {len(self.calls) + 1}: {key}",
            )
        require(
            [im["name"] for im in row["images"]] == images, "Released image mismatch"
        )
        for im in row["images"]:
            require(
                sha(self.root / "data/images" / im["name"]) == im["sha256"],
                "Image bytes changed",
            )
        if images:
            grids = row["image_grid_thw"]
            require(
                len(grids) == len(images)
                and row["input_shapes"]["pixel_values"][0]
                == sum((t * h * w for t, h, w in grids)),
                "Image patches missing from the model input",
            )
        self.boundaries(
            role,
            base,
            payload.get("record", payload) if role.endswith("_repair") else payload,
            images,
            row["response"],
        )
        require(
            0 < row["output_tokens"] <= settings["max_new_tokens"]
            and row["input_tokens"] + settings["max_new_tokens"]
            <= seal["runtime_config"]["max_context_tokens"],
            "Token limits bypassed",
        )
        self.calls.append(row)
        return row["response"]


def same(saved, rebuilt, name):
    """A stored visit equals its replay; only the wall time of the original run may differ."""
    for key in ("trace", "score", "summary"):
        require(saved[key] == rebuilt[key], f"Visit replay mismatch {name}:{key}")
    for key, value in saved["metrics"].items():
        require(
            key == "episode_wall_seconds" or rebuilt["metrics"][key] == value,
            f"Metric mismatch {name}:{key}",
        )


def unusable(records):
    """Outputs that could not be used, and names that were kept in their usable form."""
    counts = invalid_counts(records)
    return {
        **{
            k: counts[k]
            for k in (
                "invalid_actions",
                "invalid_selections",
                "collection_actions",
                "invalid_share",
            )
        },
        "visits": len(records),
        "invalid_final_answers": sum(
            (not r["summary"]["valid_final"] for r in records)
        ),
        "presentations_asked": len(records),
        "presentations_unusable": sum(
            (not r["summary"]["presentation_named"] for r in records)
        ),
        "presentations_settled": sum(
            (r["summary"]["presentation_settled"] for r in records)
        ),
        "max_share_in_smoke": SMOKE_MAX_UNUSABLE_SHARE,
        "max_unnamed_share_in_smoke": SMOKE_MAX_UNNAMED_SHARE,
    }


def smoke_tolerates(counts):
    """A smoke run passes when each kind of unusable output stays within the tolerated share."""
    share = SMOKE_MAX_UNUSABLE_SHARE
    return {
        "actions_and_selections": counts["invalid_share"] <= share,
        "final_answers": counts["invalid_final_answers"] <= share * counts["visits"],
        "presentations": counts["presentations_unusable"]
        <= SMOKE_MAX_UNNAMED_SHARE * counts["presentations_asked"],
    }


def audit(folder, root=None):
    root = Path(root or ROOT)
    folder = Path(folder)
    seal, summary = (load(folder / "seal.json"), load(folder / "summary.json"))
    inp = inputs(root)
    require(summary["status"] == "completed", "Incomplete run")
    require(
        seal["code"] == code_hashes(root)
        and seal["protocol"] == inp.protocol
        and (seal["study"] == inp.study)
        and (seal["data_hashes"] == inp.frozen["source_hashes"])
        and (seal["division_sha256"] == inp.protocol["pool"]["division_sha256"]),
        "Audit protocol, code or data differs from the sealed run",
    )
    settings, limit, learning = (
        inp.protocol["memory"],
        inp.protocol["max_collection_actions"],
        seal["learning"],
    )
    cases, golds, mode = (inp.cases, inp.golds, seal["mode"])
    if mode == "smoke":
        plan = smoke_plan(inp.division)
    elif mode == "train":
        plan = train_plan(inp.division, seal["curriculum"])
    elif seal["set"] == "test1":
        plan = test_plan("test1", inp.division["test1"])
    else:
        cases, golds, sealed = load_set(root, inp)
        require(
            seal["augmentation_sha256"] == sealed,
            "The augmented patients are not those of the sealed run",
        )
        plan = test_plan("test2", list(cases))
    require(seal["schedule"] == plan, "Schedule mismatch")
    if mode == "test":
        require(
            [seal["initial_memory"], seal["training_run"]]
            == list(
                source(
                    root,
                    seal,
                    seal["code"],
                    seal["memory_step"],
                    settings["max_lessons"],
                )
            ),
            "The notebook of this evaluation is not that of its training run at this checkpoint",
        )
    else:
        require(
            seal["initial_memory"] == []
            and seal["training_run"] is None
            and (seal["augmentation_sha256"] is None),
            "A readiness check or a training run starts with an empty notebook and sees no augmented patient",
        )
    taught = [cid for phase, cid in plan if phase == "evolution"]
    rows = [
        json.loads(s)
        for s in (folder / "private_calls.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    require(
        len(list((folder / "episodes").glob("*.json"))) == len(plan),
        "Missing or extra episode files",
    )
    memory = MemoryStore(
        entries=seal["initial_memory"], max_lessons=settings["max_lessons"]
    )
    backend, used, records, reflections = (Replay(rows, seal, root), 0, [], Counter())
    sheet_calls = lessons_asked = settled = 0
    for i, (phase, cid) in enumerate(plan):
        record = load(folder / "episodes" / f"{i:04d}_{phase}_{cid}.json")
        require(
            record["phase"] == phase and record["trace"]["id"] == cid,
            "Episode identity/order mismatch",
        )
        first, case, gold = (
            record["trace"]["first_call_index"] - 1,
            cases[cid],
            golds[cid],
        )
        backend.start_episode(phase, cid, first)
        before, known = (memory.digest(), {e["lesson_id"] for e in memory.entries()})
        if phase == "evolution":
            rebuilt, lesson, info = learn(
                backend,
                case,
                gold,
                memory.readonly_view(),
                settings,
                limit,
                learning,
                taught.index(cid) + 1,
            )
            same(record, rebuilt, f"{phase}:{cid}")
            require(
                record["lesson"] == lesson and record["reflection"] == info,
                "Reflection or memory write mismatch",
            )
            asked = info["status"] in {"stored", "abstained", "invalid"}
            expected = (
                answer_key(case, gold, rebuilt)[0] if SHEET_SHOWN[learning] else None
            )
            require(
                backend.shown == [expected] * len(backend.shown)
                and bool(backend.shown) == (asked and expected is not None),
                "The answer sheet must be exactly that of this patient and this visit, shown only when the way of learning says so",
            )
            require(
                lesson is None
                or lesson.get("disease")
                == (expected["confirmed_diagnosis"] if expected else None),
                "A lesson is filed under the confirmed diagnosis of its own patient, and only with the answer sheet",
            )
            sheet_calls, lessons_asked, settled = (
                sheet_calls + len(backend.shown),
                lessons_asked + asked,
                settled + ("settled" in info),
            )
            if lesson:
                memory.add(lesson)
        else:
            rebuilt = visit(
                backend, case, gold, memory.readonly_view(), settings, limit, learning
            )
            same(record, rebuilt, f"{phase}:{cid}")
            require(
                record["lesson"] is None
                and record["reflection"] == {"status": "not_called"}
                and (memory.digest() == before)
                and (not backend.shown),
                "Evaluation wrote memory or saw an answer sheet",
            )
        require(
            set(rebuilt["trace"]["retrieved_lesson_ids"])
            | set(rebuilt["trace"]["note_lesson_ids"])
            <= known,
            "A lesson that is not in the notebook was recalled",
        )
        require(
            NOTES_SHOWN[learning] or not rebuilt["trace"]["notes"],
            "A diagnosis note was shown at the answer",
        )
        require(
            record["last_call_index"] == len(backend.calls), "Call boundary mismatch"
        )
        used += len(backend.calls) - first
        reflections[record["reflection"]["status"]] += 1
        records.append(record)
    snapshot = load(folder / "memory_snapshot.json")
    require(
        snapshot == memory.snapshot() and summary["memory_sha256"] == memory.digest(),
        "Final memory mismatch",
    )
    require(
        (sheet_calls > 0) == (SHEET_SHOWN[learning] and lessons_asked > 0),
        "Answer sheets do not match the way of learning of this run",
    )
    counts, entries = (unusable(records), memory.entries())
    if mode == "smoke":
        failed = [name for name, ok in smoke_tolerates(counts).items() if not ok]
        require(not failed, "Smoke has too many unusable outputs: " + ", ".join(failed))
        require(reflections["stored"] > 0, "Smoke stored no lesson")
    result = {
        "passed": True,
        "run_id": folder.name,
        "mode": mode,
        "set": seal["set"],
        "doctor": seal["doctor"],
        "learning": learning,
        "episodes": len(records),
        "calls_replayed": used,
        "orphan_or_retried_calls": len(rows) - used,
        "valid_final": sum((r["summary"]["valid_final"] for r in records)),
        "episodes_with_lessons": sum(
            (r["summary"]["gathering_lessons"] > 0 for r in records)
        ),
        "episodes_with_notes": sum(
            (r["summary"]["diagnosis_notes"] > 0 for r in records)
        ),
        "memory_items": len(entries),
        "lessons_with_diagnosis_part": sum((bool(e["diagnose"]) for e in entries)),
        "lessons_kept_in_usable_form": settled,
        "reflection_status": {
            k: v for k, v in reflections.items() if k != "not_called"
        },
        "answer_sheet_calls": sheet_calls,
        "unusable_outputs": counts,
        "seal_sha256": sha(folder / "seal.json"),
        "new_inference_calls": 0,
        "clinical_validity_proven": False,
    }
    atomic_json(folder / "audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(ROOT / RUNS / safe_id(args.run_id))))
