"""Adapt saved episode files without importing or calling any model runtime."""

from dataclasses import dataclass, field
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Attachment:
    name: str
    path: str = ""


@dataclass(frozen=True)
class Message:
    role: str
    text: str
    kind: str = "speech"
    images: tuple = ()


@dataclass(frozen=True)
class Encounter:
    key: str
    case_id: str
    model: str
    success: bool
    messages: tuple
    guardian: bool = False
    child: bool = False
    source: str = ""
    phase: str = ""
    diagnosis: str = ""
    recorded_at: str = ""
    npc: bool = False


@dataclass
class Replay:
    encounters: dict = field(
        default_factory=lambda: {m: [] for m in ("2B", "4B", "9B")}
    )
    sources: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    image_roots: tuple = ()

    @property
    def count(self):
        return sum(map(len, self.encounters.values()))


def clean_clinical_text(value):
    return re.sub("(?m)^\\s*미제공 요청:\\s*", "", str(value)).strip()


def resolve_image(name, roots):
    if isinstance(name, dict):
        name = next(
            (
                name.get(key)
                for key in ("path", "filename", "file", "name", "image_path")
                if name.get(key)
            ),
            "",
        )
    name = str(name).strip().replace("\\", "/")
    if not name or "://" in name or ".." in Path(name).parts:
        return Attachment(name)
    for root in roots:
        root = Path(root).resolve()
        for reference in (name, Path(name).name):
            candidate = (root / reference).resolve()
            if candidate.is_relative_to(root) and candidate.is_file():
                return Attachment(candidate.name, str(candidate))
    return Attachment(name)


def exam_images(exam, roots):
    references = []
    for key in ("images", "image", "image_paths", "image_path"):
        value = exam.get(key, [])
        references.extend(
            value if isinstance(value, list) else [value] if value else []
        )
    references.extend(
        re.findall(
            "(?<![\\w])((?:[\\w.\\-]+[/\\\\])*[\\w\\-]+\\.(?:jpeg|jpg|png|webp|gif))\\b",
            str(exam.get("text", "")),
            re.I,
        )
    )
    attachments = []
    for reference in references:
        attachment = resolve_image(reference, roots)
        if attachment not in attachments:
            attachments.append(attachment)
    return attachments


def adapt_episode(raw, model, source, image_roots=None):
    """Only an explicit boolean score is usable; absent scores are never deaths."""
    if model not in ("2B", "4B", "9B"):
        raise ValueError("지원하지 않는 모델 크기")
    trace = raw.get("trace", {})
    correct = raw.get("score", {}).get("correct")
    if type(correct) is not bool or not isinstance(trace.get("events"), list):
        raise ValueError("trace.events 또는 명시적인 score.correct가 없습니다")
    case = str(trace.get("id") or raw.get("score", {}).get("id") or "")
    if not case:
        raise ValueError("환자 ID가 없습니다")
    opening = trace.get("initial_doctor_view", {}).get("public_context", "")
    if not isinstance(opening, str):
        opening = json.dumps(opening, ensure_ascii=False)
    guardian = any((e.get("speaker") == "guardian" for e in trace["events"]))
    guardian |= bool(
        re.search("\\b(my|our) (son|daughter|child|baby|infant)\\b", opening, re.I)
    )
    age = re.search("\\b(\\d+)[ -]year[ -]old\\b", opening, re.I)
    child = bool(age and int(age.group(1)) < 18) or bool(
        re.search("\\b(infant|baby|toddler)\\b", opening, re.I)
    )
    role = "guardian" if guardian else "patient"
    image_roots = image_roots if image_roots is not None else [ROOT / "data/images"]
    messages = [Message(role, clean_clinical_text(opening))] if opening else []
    for event in trace["events"]:
        kind = event.get("type")
        query = event.get("query", "")
        if kind == "ask_patient":
            if query:
                messages.append(Message("doctor", clean_clinical_text(query)))
            answer = event.get("utterance") or event.get("message")
            if answer:
                messages.append(
                    Message(event.get("speaker", role), clean_clinical_text(answer))
                )
        elif kind == "request_exam":
            if query:
                messages.append(Message("doctor", clean_clinical_text(query)))
            exams = event.get("exams", [event["exam"]] if "exam" in event else [])
            parts = []
            attachments = []
            for exam in exams:
                part = clean_clinical_text(exam.get("text", ""))
                for image in exam_images(exam, image_roots):
                    if image not in attachments:
                        attachments.append(image)
                parts.append(part)
            if event.get("unavailable_requests"):
                parts.append(
                    "; ".join(
                        (clean_clinical_text(s) for s in event["unavailable_requests"])
                    )
                )
            if not exams:
                parts.append(
                    clean_clinical_text(
                        event.get("unavailable_message")
                        or event.get("message")
                        or "No examination results were provided."
                    )
                )
            messages.append(
                Message(
                    "system",
                    "\n".join(parts),
                    "exam" if exams else "notice",
                    tuple(attachments),
                )
            )
        elif kind != "finish":
            messages.append(
                Message(
                    "system",
                    clean_clinical_text(
                        event.get("message") or event.get("status") or kind
                    ),
                    "notice",
                )
            )
    if trace.get("reasoning"):
        messages.append(
            Message("doctor", clean_clinical_text(trace["reasoning"]), "reasoning")
        )
    submission = trace.get("result", {}).get("submission", {})
    answer = submission.get("answer") or submission.get("diagnosis")
    choices = (
        trace.get("final_choices") or trace.get("final_view", {}).get("options") or {}
    )
    diagnosis = (
        str(choices.get(answer, answer) if isinstance(choices, dict) else answer)
        if answer
        else "No valid final diagnosis"
    )
    messages.append(Message("doctor", "Final diagnosis · " + diagnosis, "diagnosis"))
    return Encounter(
        str(source),
        case,
        model,
        correct,
        tuple(messages),
        guardian,
        child,
        str(source),
        str(raw.get("phase", trace.get("mode", "recorded"))),
        diagnosis,
        str(raw.get("recorded_at") or raw.get("timestamp") or "Not recorded"),
    )


def load_replay(config_path=None):
    config_path = Path(config_path) if config_path else ROOT / "configs/simulation.json"
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    image_roots = [
        Path(p) if Path(p).is_absolute() else config_path.parent / p
        for p in config.get("image_roots", [str(ROOT / "data/images")])
    ]
    replay = Replay(image_roots=tuple((str(Path(p).resolve()) for p in image_roots)))
    for model in replay.encounters:
        value = config.get("sources", {}).get(model)
        if not value:
            replay.warnings.append(f"{model}: data path not configured")
            continue
        directory = Path(value)
        if not directory.is_absolute():
            directory = config_path.parent / directory
        directory = directory.resolve()
        replay.sources[model] = str(directory)
        if not directory.exists():
            replay.warnings.append(f"{model}: record directory missing · {directory}")
            continue
        run = directory.parent if directory.name == "episodes" else directory
        seal = run / "seal.json"
        if seal.exists():
            metadata = json.loads(seal.read_text(encoding="utf-8"))
            actual = str(metadata.get("doctor", ""))
            if actual and actual != model:
                replay.warnings.append(
                    f"{model}: sealed doctor {actual} does not match; not loaded"
                )
                continue
        if directory.is_file():
            files = [directory]
        else:
            folder = (
                directory / "episodes"
                if (directory / "episodes").is_dir()
                else directory
            )
            files = sorted(folder.glob("*.json"))
        for path in files:
            if path.name in (
                "seal.json",
                "summary.json",
                "audit.json",
                "memory_snapshot.json",
            ):
                continue
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                replay.encounters[model].append(
                    adapt_episode(raw, model, path, image_roots)
                )
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                replay.warnings.append(f"{model}/{path.name}: {exc}")
        if not replay.encounters[model]:
            replay.warnings.append(f"{model}: no replayable encounter records")
    return replay


def npc_encounter(index, rng):
    return Encounter(
        f"npc-{index}",
        f"NPC-{index:03}",
        "NPC",
        rng.random() < 0.9,
        (
            Message("patient", "I am here for my appointment."),
            Message("doctor", "Consultation in progress."),
            Message("doctor", "Consultation complete.", "diagnosis"),
        ),
        guardian=index % 7 == 0,
        child=index % 7 == 0,
        source="background scene",
        npc=True,
    )
