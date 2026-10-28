from copy import deepcopy
import json
import re
import unicodedata
from clinical_memory.environment.engine import Episode, ProtocolError, parse_object
from clinical_memory.environment.formatting import unfence, clean_repair

ALIASES = (
    ("\\bmagnetic resonance imaging\\b", "mri"),
    ("\\bcomputed tomography(?: scan)?\\b", "ct"),
    ("\\b(?:x[ -]?rays?|radiographs?|radiography)\\b", "xray"),
    ("\\b(?:electrocardiograms?|electrocardiography|ekg)\\b", "ecg"),
    ("\\b(?:echocardiograms?|echocardiography)\\b", "echo"),
    ("\\b(?:ultrasonography|sonography)\\b", "ultrasound"),
    ("\\bcomplete blood count\\b", "cbc"),
    ("\\b(?:lumbar puncture|spinal tap)\\b", "lumbar_puncture"),
    ("\\b(?:physical examination|physical exam)\\b", "physical_exam"),
)
FILLER = set(
    "please can could would you we i me my the a an of for on to in this that patient patients his her their obtain perform request order check get provide show review result results scan test testing exam examination imaging available current report reported".split()
)
COMPOUND = re.compile("\\b(?:and|plus)\\b|[,;/+]")
NEUTRAL = "An additional part of the original request could not be matched."


def exact_key(query):
    return " ".join(re.findall("\\w+", unicodedata.normalize("NFKC", query).casefold()))


def identity(query):
    text = unicodedata.normalize("NFKC", query).casefold()
    for pattern, replacement in ALIASES:
        text = re.sub(pattern, replacement, text)
    words = [w for w in re.findall("\\w+", text) if w not in FILLER]
    return tuple(sorted(set(words)))


class DeniedExams:
    def __init__(self):
        self.entries = []

    def add(self, text, turn, partial=False):
        if not text or text == NEUTRAL or (not identity(text)):
            return
        item = {
            "query": text,
            "first_denied_turn": turn,
            "partial_request": partial,
            "exact": exact_key(text),
            "identity": list(identity(text)),
        }
        if not any((e["exact"] == item["exact"] for e in self.entries)):
            self.entries.append(item)

    def observe(self, event):
        if event.get("type") != "request_exam" or event.get("status") not in {
            "unavailable",
            "provided",
            "repeated",
        }:
            return
        query = event.get("query", "")
        turn = event["collection_actions"]
        if event["status"] == "unavailable":
            self.add(query, turn)
        for fragment in event.get("unavailable_requests", []):
            if fragment != NEUTRAL and fragment.casefold() in query.casefold():
                self.add(fragment, turn, partial=event["status"] != "unavailable")

    def match(self, query):
        key, semantic = (exact_key(query), identity(query))
        for entry in self.entries:
            if key == entry["exact"] or (
                semantic and list(semantic) == entry["identity"]
            ):
                return {
                    "denied_query": entry["query"],
                    "first_denied_turn": entry["first_denied_turn"],
                    "rule": "exact" if key == entry["exact"] else "lexical_alias",
                }
        parts = [p.strip() for p in COMPOUND.split(query) if p.strip()]
        if len(parts) > 1:
            for part in parts:
                part_key = identity(part)
                for entry in self.entries:
                    if part_key and list(part_key) == entry["identity"]:
                        return {
                            "denied_query": entry["query"],
                            "first_denied_turn": entry["first_denied_turn"],
                            "rule": "blocked_component",
                        }
        return None

    def public(self):
        return [
            {
                k: deepcopy(e[k])
                for k in ("query", "first_denied_turn", "partial_request")
            }
            for e in self.entries
        ]


class GuardedEpisode(Episode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.denied = DeniedExams()

    def doctor_view(self):
        view = super().doctor_view()
        view["unavailable_exams"] = self.denied.public()
        view["request_rule"] = (
            "Examinations in unavailable_exams cannot be requested again in this encounter. Their findings remain unknown, not normal or negative. Choose a different useful action or finish. Rephrasing a blocked request does not make it available. A blocked attempt spends one action but never calls Measurement. For a compound request, omit its blocked component and request the other part separately."
        )
        return view

    def collect(self, raw, patient_router, measurement_router, repair=None):
        if self.state != "collecting":
            raise ProtocolError("Information collection is closed")

        def valid(obj):
            return (
                obj.get("type") == "finish"
                and set(obj) == {"type"}
                or (
                    obj.get("type") in {"ask_patient", "request_exam"}
                    and set(obj) == {"type", "query"}
                    and isinstance(obj["query"], str)
                    and (0 < len(obj["query"].strip()) <= 2000)
                )
            )

        action, repairs = parse_object(unfence(raw), valid, clean_repair(repair))
        match = (
            self.denied.match(action["query"])
            if action and action["type"] == "request_exam"
            else None
        )
        if match:
            self.actions += 1
            if self.actions == self.max_actions:
                self._finish("action_limit")
            event = {
                "type": "request_exam",
                "query": action["query"],
                "status": "blocked_unavailable",
                "collection_actions": self.actions,
                "format_repairs": repairs,
                "measurement_called": False,
                "block": match,
                "message": "Already unavailable in this encounter. Choose a different request or finish; findings remain unknown.",
                "state_after": self.state,
            }
            self._events.append(deepcopy(event))
        else:
            event = super().collect(
                json.dumps(action) if action else "{}",
                patient_router,
                measurement_router,
            )
            event["format_repairs"] = repairs
            self._events[-1]["format_repairs"] = repairs
            self.denied.observe(event)
        return deepcopy(event)

    def submit_final(self, raw, repair=None):
        return super().submit_final(unfence(raw), clean_repair(repair))
