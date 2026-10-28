"""Deterministic playback clock, queues and choreography; no GUI dependencies."""

from collections import deque
from dataclasses import dataclass, field
import math
import random
from .replay import npc_encounter
from .identity import Pseudonyms, DOCTORS

COLORS = ("#dc6465", "#548bd5", "#dcaf42", "#61a986", "#ce80a4")
DEFAULT_SPEED = 1.0
MODELS = ("2B", "4B", "9B", "NPC", "NPC")
ROOMS = (
    (114, 88, 414, 274),
    (450, 88, 750, 274),
    (786, 88, 1086, 274),
    (114, 398, 414, 568),
    (450, 398, 750, 568),
)
LAB = (786, 398, 1086, 568)
ENTRY = (603.0, 660.0)
RECEPTION = ((158.0, 620.0), (266.0, 620.0))
PAYMENT = (976.0, 620.0)
BENCHES = (
    (240.0, 350.0),
    (290.0, 350.0),
    (340.0, 350.0),
    (470.0, 350.0),
    (520.0, 350.0),
    (570.0, 350.0),
    (700.0, 350.0),
    (750.0, 350.0),
    (800.0, 350.0),
)
LAB_OUT = (924.0, 376.0)
LAB_PATIENT = (924.0, 508.0)
LAB_NURSE = (978.0, 508.0)


def door(room):
    x, y, _, _ = ROOMS[room]
    return (x + 138.0, 300.0 if room < 3 else 376.0)


def seat(room):
    x, y, _, _ = ROOMS[room]
    return (x + 115.0, y + 141.0)


def nurse_home(room):
    x, y, _, _ = ROOMS[room]
    return (x + 238.0, y + 113.0)


def doctor_home(room):
    x, y, _, _ = ROOMS[room]
    return (x + 115.0, y + 85.0)


def hall_route(start, target):
    """The east passage connects reception and the central corridor, avoiding walls."""
    if start[1] > 580 and target[1] < 390:
        return [(1110.0, 620.0), (1110.0, 328.0), (target[0], 328.0), target]
    if start[1] < 390 and target[1] > 580:
        return [(start[0], 328.0), (1110.0, 328.0), (1110.0, 620.0), target]
    return [(start[0], 328.0), (target[0], 328.0), target]


@dataclass
class Action:
    kind: str
    value: object = None
    duration: float = 0


@dataclass
class Visit:
    uid: int
    encounter: object
    patient: tuple = ENTRY
    nurse: tuple = (0.0, 0.0)
    room: int | None = None
    state: str = "입장"
    actions: deque = field(default_factory=deque)
    current: Action | None = None
    elapsed: float = 0
    messages: list = field(default_factory=list)
    speaker: str = ""
    desk: int | None = None
    bench: int | None = None
    wait_since: float = 0
    started: float = 0
    finished: float | None = None
    paid: bool = False
    resolved: bool = False
    ghost: float = 0
    done: bool = False
    name: str = ""
    assigned_at: float | None = None

    @property
    def display_name(self):
        return f"{self.name} ({self.encounter.case_id})"


class Hospital:
    def __init__(self, replay, seed=42):
        self.replay = replay
        self.seed = seed
        self.rng = random.Random(seed)
        self.pseudonyms = Pseudonyms(seed)
        self.queues = {m: deque(v) for m, v in replay.encounters.items()}
        self.visits = []
        self.records = []
        self.rooms = [None] * 5
        self.desks = [None, None]
        self.benches = [None] * len(BENCHES)
        self.exam_queue = deque()
        self.exam_owner = None
        self.payment_queue = deque()
        self.payment_owner = None
        self.reg_queue = deque()
        self.time = 0.0
        self.paused = False
        self.speed = DEFAULT_SPEED
        self.coins = self.deaths = self.recovered = 0
        self.next_arrival = 0.0
        self.uid = 0
        self.admission_cursor = 0
        self.events = deque(maxlen=5)
        self.completed = False

    @property
    def clock_minutes(self):
        return 8 * 60 + self.time * 8

    @property
    def day(self):
        return int(self.clock_minutes // 1440) + 1

    @property
    def clock(self):
        minute = int(self.clock_minutes % 1440)
        return f"{minute // 60:02}:{minute % 60:02}"

    @property
    def night(self):
        return not 6 <= self.clock_minutes / 60 % 24 < 18

    @property
    def daylight(self):
        """Smooth dawn 05:00–07:00 and dusk 17:00–19:00, including midnight continuity."""
        hour = self.clock_minutes / 60 % 24

        def smooth(t):
            t = min(1.0, max(0.0, t))
            return t * t * (3 - 2 * t)

        return smooth((hour - 5) / 2) * (1 - smooth((hour - 17) / 2))

    @property
    def survival(self):
        total = self.deaths + self.recovered
        return self.recovered / total if total else None

    def stamp(self, at=None):
        minutes = 480 + (self.time if at is None else at) * 8
        return f"Day {int(minutes // 1440) + 1} · {int(minutes % 1440) // 60:02}:{int(minutes % 60):02}"

    def note(self, key, **values):
        self.events.appendleft((self.stamp(), key, values))

    def active_calls(self):
        """One current patient per reception desk for the assignment bubbles."""
        calls = {}
        for v in self.visits:
            if (
                v.assigned_at is not None
                and v.desk is not None
                and (0 <= self.time - v.assigned_at < 6)
            ):
                previous = calls.get(v.desk)
                if previous is None or (v.assigned_at, v.uid) > (
                    previous.assigned_at,
                    previous.uid,
                ):
                    calls[v.desk] = v
        return tuple(
            (
                calls[key]
                for key in sorted(calls)
                if calls[key].state == "진료실 이동" and (not calls[key].done)
            )
        )

    def enqueue(self, visit, kind, value=None, duration=0):
        visit.actions.append(Action(kind, value, duration))

    def move(self, visit, who, points):
        for point in points:
            self.enqueue(visit, "move", (who, point))

    def arrive(self, encounter):
        self.uid += 1
        v = Visit(self.uid, encounter, started=self.time)
        v.name = self.pseudonyms.get(encounter.case_id)
        self.visits.append(v)
        v.state = "접수 대기"
        self.reg_queue.append(v)
        return v

    def _admit(self):
        if self.time < self.next_arrival:
            return
        self.next_arrival = self.time + 4.5
        if len([v for v in self.visits if not v.done]) >= 14:
            return
        pending = any(self.queues.values())
        active = any((not v.done and (not v.encounter.npc) for v in self.visits))
        if not pending and (not active):
            return
        pattern = ("2B", "4B", "9B", "NPC", "NPC")
        for _ in pattern:
            m = pattern[self.admission_cursor % 5]
            self.admission_cursor += 1
            if m == "NPC":
                if sum((not v.done and v.encounter.npc for v in self.visits)) < 3:
                    self.arrive(npc_encounter(self.uid + 1, self.rng))
                    break
            elif (
                self.queues[m]
                and sum((not v.done and v.encounter.model == m for v in self.visits))
                < 3
            ):
                self.arrive(self.queues[m].popleft())
                break

    def _dispatch(self):
        for i, owner in enumerate(self.desks):
            if owner is None and self.reg_queue:
                v = self.reg_queue.popleft()
                v.desk = i
                self.desks[i] = v
                v.state = "접수"
                self.move(v, "patient", [RECEPTION[i]])
                self.enqueue(v, "wait", duration=1.8)
                self.enqueue(v, "registered")
        for v in self.visits:
            if v.state == "벤치 대기" and v.bench is None:
                slot = next((i for i, x in enumerate(self.benches) if x is None), None)
                if slot is not None:
                    v.bench = slot
                    self.benches[slot] = v
                    self.move(v, "patient", hall_route(v.patient, BENCHES[slot]))
                    self.enqueue(v, "wait", duration=2.5)
                    self.enqueue(v, "ready")
            if v.state == "배정 대기" and v.current is None and (not v.actions):
                choices = (
                    range(3, 5)
                    if v.encounter.npc
                    else [MODELS.index(v.encounter.model)]
                )
                free = next((r for r in choices if self.rooms[r] is None), None)
                if free is not None:
                    self._assign(v, free)
        if self.exam_owner is None and self.exam_queue:
            v = self.exam_queue.popleft()
            self.exam_owner = v
            self._exam_trip(v)
        if self.payment_owner is None and self.payment_queue:
            v = self.payment_queue.popleft()
            self.payment_owner = v
            v.state = "결제"
            self.move(v, "patient", [PAYMENT])
            self.enqueue(v, "wait", duration=2.0)
            self.enqueue(v, "paid")

    def _assign(self, v, room):
        self.rooms[room] = v
        v.room = room
        v.nurse = nurse_home(room)
        v.state = "진료실 이동"
        v.assigned_at = self.time
        self.benches[v.bench] = None
        v.bench = None
        if not v.encounter.npc:
            self.note("assigned", name=v.display_name, doctor=DOCTORS[room])
        self.enqueue(v, "wait", duration=1.7)
        self.move(v, "patient", hall_route(v.patient, door(room)) + [seat(room)])
        self.enqueue(v, "consult")

    def _next_message(self, v):
        index = len(v.messages)
        if index >= len(v.encounter.messages):
            self.enqueue(v, "resolve")
            return
        msg = v.encounter.messages[index]
        if msg.kind == "exam":
            v.state = "검사실 대기"
            self.exam_queue.append(v)
        else:
            v.state = "진단 중" if msg.kind == "diagnosis" else "문진 중"
            v.speaker = msg.role
            self.enqueue(v, "say", msg, min(6.0, max(2.0, len(msg.text) / 65)))

    def _exam_trip(self, v):
        v.state = "검사 이동"
        out = door(v.room)
        self.move(v, "nurse", [out])
        self.move(v, "patient", [out])
        self.move(v, "pair", hall_route(out, LAB_OUT))
        self.move(v, "patient", [LAB_PATIENT])
        self.move(v, "nurse", [LAB_NURSE])
        self.enqueue(v, "exam_start")
        self.enqueue(v, "wait", duration=4.0)
        self.enqueue(v, "exam_result")
        self.move(v, "nurse", [LAB_OUT])
        self.move(v, "patient", [LAB_OUT])
        self.enqueue(v, "exam_release")
        self.move(v, "pair", hall_route(LAB_OUT, out))
        self.move(v, "patient", [seat(v.room)])
        self.move(v, "nurse", [nurse_home(v.room)])
        self.enqueue(v, "consult")

    def _resolve(self, v):
        if v.resolved:
            return
        v.resolved = True
        v.finished = self.time
        self.records.append(v)
        v.speaker = ""
        if v.encounter.success:
            if not v.encounter.npc:
                self.recovered += 1
            v.state = "회복"
            self.enqueue(v, "wait", duration=2.0)
            self.move(v, "patient", [door(v.room)])
            self.enqueue(v, "room_release")
            self.move(v, "patient", hall_route(door(v.room), (1050.0, 620.0)))
            self.enqueue(v, "checkout")
        else:
            if not v.encounter.npc:
                self.deaths += 1
            v.state = "영혼"
            self.enqueue(v, "ghost", duration=3.0)
            self.enqueue(v, "room_release")
            self.enqueue(v, "done")
        if not v.encounter.npc:
            self.note(
                "outcome",
                doctor=DOCTORS[v.room],
                name=v.display_name,
                result="recovered" if v.encounter.success else "dead",
            )

    def _finish_action(self, v, a):
        k = a.kind
        if k == "registered":
            self.desks[v.desk] = None
            v.state = "벤치 대기"
        elif k == "ready":
            v.state = "배정 대기"
        elif k == "consult":
            v.speaker = ""
            self._next_message(v)
        elif k == "say":
            v.messages.append(a.value)
            v.speaker = ""
            self._next_message(v)
        elif k == "exam_start":
            v.state = "검사 중"
        elif k == "exam_result":
            v.messages.append(v.encounter.messages[len(v.messages)])
            v.state = "진료실 복귀"
            if not v.encounter.npc:
                self.note("exam_received", doctor=DOCTORS[v.room])
        elif k == "exam_release":
            self.exam_owner = None
        elif k == "resolve":
            self._resolve(v)
        elif k == "room_release":
            self.rooms[v.room] = None
        elif k == "checkout":
            v.state = "결제 대기"
            self.payment_queue.append(v)
        elif k == "paid":
            if not v.paid:
                v.paid = True
                if not v.encounter.npc:
                    self.coins += 1
                    self.note("payment_event", name=v.display_name)
            self.payment_owner = None
            v.state = "퇴장"
            self.move(v, "patient", [ENTRY, (603.0, 700.0)])
            self.enqueue(v, "done")
        elif k == "done":
            v.done = True

    def _step_visit(self, v, dt):
        if v.done:
            return
        if v.current is None:
            if not v.actions:
                return
            v.current = v.actions.popleft()
            v.elapsed = 0.0
        a = v.current
        v.elapsed += dt
        finished = True
        if a.kind == "move":
            who, target = a.value
            origin = v.nurse if who in ("nurse", "pair") else v.patient
            dx, dy = (target[0] - origin[0], target[1] - origin[1])
            dist = math.hypot(dx, dy)
            travel = 135 * dt
            pos = (
                target
                if dist <= travel
                else (origin[0] + dx / dist * travel, origin[1] + dy / dist * travel)
            )
            if who == "pair":
                v.nurse = pos
                v.patient = (
                    pos[0] - dx / max(dist, 1) * 24,
                    pos[1] - dy / max(dist, 1) * 24,
                )
            else:
                setattr(v, who, pos)
            finished = dist <= travel
        elif a.kind in ("wait", "say", "ghost"):
            finished = v.elapsed >= a.duration
            if a.kind == "ghost":
                v.ghost = min(1.0, v.elapsed / a.duration)
        if finished:
            v.current = None
            self._finish_action(v, a)

    def tick(self, seconds):
        if self.paused or self.completed:
            return
        remaining = max(0.0, seconds) * self.speed
        while remaining > 1e-09:
            dt = min(0.05, remaining)
            remaining -= dt
            self.time += dt
            self._admit()
            self._dispatch()
            for v in self.visits:
                self._step_visit(v, dt)
            for i, v in enumerate(self.reg_queue):
                v.patient = (485.0 + i % 5 * 28, 625.0 + i // 5 * 27)
            for i, v in enumerate(self.payment_queue):
                v.patient = (1020.0 + i % 3 * 30, 620.0 + i // 3 * 28)
            if not any(self.queues.values()) and all((v.done for v in self.visits)):
                self.completed = True
                self.note("complete")
                break

    def export_records(self):
        return [
            {
                "visit": v.uid,
                "patient_name": v.name,
                "case_id": v.encounter.case_id,
                "model": v.encounter.model,
                "room": v.room + 1,
                "npc": v.encounter.npc,
                "source": v.encounter.source,
                "phase": v.encounter.phase,
                "recorded_at": v.encounter.recorded_at,
                "simulation_started": self.stamp(v.started),
                "simulation_finished": self.stamp(v.finished),
                "success": v.encounter.success,
                "paid": v.paid,
                "messages": [
                    {
                        "role": m.role,
                        "kind": m.kind,
                        "text": m.text,
                        "images": [{"name": a.name, "path": a.path} for a in m.images],
                    }
                    for m in v.messages
                ],
            }
            for v in self.records
            if not v.encounter.npc
        ]
