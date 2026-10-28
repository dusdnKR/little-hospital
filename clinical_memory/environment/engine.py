from copy import deepcopy
import json
from pathlib import PurePosixPath


class ProtocolError(ValueError):
    pass


def parse_object(raw, validate, repair=None):
    """At most one format repair, without correctness feedback or hidden data."""
    for attempt in range(2 if repair is not None else 1):
        try:
            obj = json.loads(raw) if isinstance(raw, str) else deepcopy(raw)
            if not isinstance(obj, dict) or not validate(obj):
                raise ValueError("Invalid schema")
            return (obj, attempt)
        except (ValueError, TypeError):
            if attempt == 0 and repair is not None:
                raw = repair(
                    "Invalid JSON/schema. Return only the required JSON object."
                )
    return (None, int(repair is not None))


class Episode:
    def __init__(self, case, options, max_actions=8, answer_format="mcq"):
        if type(max_actions) is not int or max_actions < 1:
            raise ValueError("max_actions must be a positive integer")
        if set(options) != set("ABCDE") or not all(
            (isinstance(v, str) for v in options.values())
        ):
            raise ValueError("Expected original five options")
        self._case = deepcopy(case)
        if answer_format not in {"mcq", "open"}:
            raise ValueError("Unknown final answer format")
        self.answer_format = answer_format
        self._profile = deepcopy(case.get("dialogue_profile"))
        if self._profile:
            from clinical_memory.data.speech import apply_profile

            apply_profile(case, self._profile)
        self._options = deepcopy(options)
        self._facts = {item["id"]: item for item in case["patient_facts"]}
        self._exams = {item["id"]: item for item in case["exams"]}
        if len(self._facts) != len(case["patient_facts"]) or len(self._exams) != len(
            case["exams"]
        ):
            raise ValueError("Duplicate information ID")
        self._facts, self._exams = (deepcopy(self._facts), deepcopy(self._exams))
        if self._profile:
            if "OPENING" in self._facts:
                raise ValueError("Reserved opening ID")
            self._facts = {
                "OPENING": {"id": "OPENING", "text": case["public_context"]["text"]},
                **self._facts,
            }
        for exam in self._exams.values():
            for name in exam["images"]:
                if (
                    not isinstance(name, str)
                    or not name
                    or name in {".", ".."}
                    or (PurePosixPath(name).name != name)
                    or ("\\" in name)
                    or (":" in name)
                ):
                    raise ValueError("Image must be a local basename")
        self.max_actions = max_actions
        self.actions = 0
        self.state = "collecting"
        self.stop_reason = None
        self._seen = {("patient", "OPENING")} if self._profile else set()
        initial_ids = case["public_context"].get("initially_disclosed_fact_ids", [])
        if (
            not isinstance(initial_ids, list)
            or len(initial_ids) != len(set(initial_ids))
            or any((x not in self._facts or x == "OPENING" for x in initial_ids))
        ):
            raise ValueError("Invalid initially disclosed history IDs")
        self._seen.update((("patient", x) for x in initial_ids))
        self._events = []
        self._submission = None

    def doctor_view(self):
        view = {
            "public_context": self._case["public_context"]["text"],
            "task": self._case["task"],
            "state": self.state,
            "remaining_actions": self.max_actions - self.actions,
            "history": deepcopy(self._events),
        }
        if self.state != "collecting" and self.answer_format == "mcq":
            view["options"] = deepcopy(self._options)
        return view

    def patient_view(self, query):
        if self._profile:
            from clinical_memory.data.speech import PERSONAS

            return {
                "query": query,
                "speaker": self._profile["speaker"],
                "persona": self._profile.get(
                    "persona", PERSONAS[self._profile["speaker"]]
                ),
                "public_context": self._case["public_context"]["text"],
                "facts": [
                    {"id": x, "text": text}
                    for x, text in self._profile["utterances"].items()
                ],
                "history": [
                    {
                        "query": e["query"],
                        "utterance": e.get("utterance", ""),
                        "status": e["status"],
                    }
                    for e in self._events
                    if e["type"] == "ask_patient"
                ],
            }
        return {
            "query": query,
            "speaker": self._case["speaker"],
            "public_context": self._case["public_context"]["text"],
            "facts": [{"id": f["id"], "text": f["text"]} for f in self._facts.values()],
        }

    def measurement_view(self, query):
        return {
            "query": query,
            "available_exams": [
                dict(
                    {"id": x},
                    **{"scope": deepcopy(e["request_scope"])}
                    if "request_scope" in e
                    else {},
                )
                for x, e in self._exams.items()
            ],
        }

    def _finish(self, reason):
        self.state, self.stop_reason = ("awaiting_final", reason)

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

        action, repairs = parse_object(raw, valid, repair)
        if action and action["type"] == "finish":
            self._finish("doctor_finish")
            result = {
                "type": "finish",
                "status": "finished",
                "collection_actions": self.actions,
                "format_repairs": repairs,
            }
        else:
            self.actions += 1
            result = {
                "type": action["type"] if action else "invalid_action",
                "query": action.get("query", "") if action else "",
                "collection_actions": self.actions,
                "format_repairs": repairs,
            }
            if action is None:
                result["status"] = "invalid_action"
            elif action["type"] == "ask_patient":
                choice = patient_router(self.patient_view(action["query"]))
                if (
                    not isinstance(choice, dict)
                    or set(choice) != {"fact_ids"}
                    or (not isinstance(choice["fact_ids"], list))
                    or any(
                        (
                            not isinstance(x, str) or x not in self._facts
                            for x in choice["fact_ids"]
                        )
                    )
                ):
                    result["status"] = "invalid_selection"
                else:
                    ids = set(choice["fact_ids"])
                    available = [x for x in self._facts if x in ids]
                    fresh = [x for x in available if ("patient", x) not in self._seen]
                    result.update(
                        status="provided"
                        if fresh
                        else "repeated"
                        if available
                        else "unavailable",
                        speaker=self._case["speaker"],
                        facts=[
                            {"id": x, "text": self._facts[x]["text"]} for x in fresh
                        ],
                    )
                    self._seen.update((("patient", x) for x in fresh))
                    if self._profile:
                        speech = self._profile["utterances"]
                        result.update(
                            speaker=self._profile["speaker"],
                            selected_fact_ids=available,
                            facts=[{"id": x, "text": speech[x]} for x in fresh],
                            utterance=" ".join((speech[x] for x in available)),
                        )
            else:
                choice = measurement_router(self.measurement_view(action["query"]))
                if isinstance(choice, dict) and "exam_ids" in choice:
                    from clinical_memory.environment.measurement import parse_selection

                    checked = parse_selection(json.dumps(choice), set(self._exams))
                    if checked is None:
                        result["status"] = "invalid_selection"
                    else:
                        ids = checked["exam_ids"]
                        fresh = [x for x in ids if ("exam", x) not in self._seen]
                        missing = checked["unavailable_requests"]
                        result.update(
                            status="provided"
                            if fresh
                            else "repeated"
                            if ids
                            else "unavailable",
                            requested_exam_ids=ids,
                            requested_image_exam=any(
                                (self._exams[x]["images"] for x in ids)
                            ),
                            exams=[
                                {
                                    "id": x,
                                    "text": self._exams[x]["text"],
                                    "images": deepcopy(self._exams[x]["images"]),
                                }
                                for x in fresh
                            ],
                            repeated_exam_ids=[x for x in ids if x not in fresh],
                            unavailable_requests=missing,
                        )
                        if missing:
                            result["unavailable_message"] = (
                                "Not provided for the requested examination."
                            )
                        self._seen.update((("exam", x) for x in fresh))
                elif (
                    not isinstance(choice, dict)
                    or set(choice) != {"exam_id"}
                    or (
                        choice["exam_id"] is not None
                        and (
                            not isinstance(choice["exam_id"], str)
                            or choice["exam_id"] not in self._exams
                        )
                    )
                ):
                    result["status"] = "invalid_selection"
                elif choice["exam_id"] is None:
                    result["status"] = "unavailable"
                else:
                    x = choice["exam_id"]
                    if self._profile:
                        result.update(
                            requested_exam_id=x,
                            requested_image_exam=bool(self._exams[x]["images"]),
                        )
                    if ("exam", x) in self._seen:
                        result.update(status="repeated", exam_id=x)
                    else:
                        exam = self._exams[x]
                        result.update(
                            status="provided",
                            exam={
                                "id": x,
                                "text": exam["text"],
                                "images": deepcopy(exam["images"]),
                            },
                        )
                        self._seen.add(("exam", x))
            if result["status"] == "unavailable":
                result["message"] = self._case["unavailable_response"]
            elif result["status"] == "repeated":
                result["message"] = "Already provided; see earlier history."
            if self._profile and result["status"] == "unavailable":
                result["message"] = (
                    "I don't know about that."
                    if result["type"] == "ask_patient"
                    else "Not provided for the requested examination."
                )
                if result["type"] == "ask_patient":
                    result["utterance"] = result["message"]
            if (
                self._profile
                and result["type"] == "ask_patient"
                and (result["status"] == "invalid_selection")
            ):
                result["message"] = "Patient response could not be processed."
            if self.actions == self.max_actions:
                self._finish("action_limit")
        result["state_after"] = self.state
        self._events.append(deepcopy(result))
        return deepcopy(result)

    def submit_final(self, raw, repair=None):
        if self.state != "awaiting_final":
            raise ProtocolError(
                "Finish collection first; final answer is accepted only once"
            )
        key = "answer" if self.answer_format == "mcq" else "diagnosis"
        obj, repairs = parse_object(
            raw,
            lambda x: (
                set(x) == {key}
                and isinstance(x[key], str)
                and (
                    x[key] in self._options
                    if key == "answer"
                    else 0 < len(x[key].strip()) <= 240
                )
            ),
            repair,
        )
        self._submission = {
            key: obj[key] if obj else None,
            "valid": obj is not None,
            "format_repairs": repairs,
        }
        self.state = "done"
        return deepcopy(self._submission)

    def result(self):
        if self.state != "done":
            raise ProtocolError("No final answer yet")
        return {
            "id": self._case["id"],
            "collection_actions": self.actions,
            "stop_reason": self.stop_reason,
            "submission": deepcopy(self._submission),
        }


def evaluate(result, gold):
    """Trusted code only; return value must never be fed to a diagnosis actor."""
    if result["id"] != gold["id"] or gold["label"] not in gold["options"]:
        raise ValueError("Mismatched or invalid answer key")
    return {
        "id": result["id"],
        "correct": bool(
            result["submission"]["valid"]
            and result["submission"]["answer"] == gold["label"]
        ),
    }
