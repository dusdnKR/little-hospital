"""Behavioral checks for playback, accounting, isolation and UI navigation."""

import json
from pathlib import Path
import random
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch
from PIL import Image, ImageChops, ImageFilter
from clinical_memory.simulation.engine import (
    Hospital,
    LAB_PATIENT,
    LAB_NURSE,
    LAB_OUT,
    door,
    seat,
    nurse_home,
)
from clinical_memory.simulation.replay import (
    Attachment,
    Encounter,
    Message,
    Replay,
    adapt_episode,
    load_replay,
    npc_encounter,
    clean_clinical_text,
    resolve_image,
)
from clinical_memory.simulation.ui import Screen, clinical_image
from clinical_memory.simulation.identity import Pseudonyms
from clinical_memory.simulation.graphics import Draw, Surface


def encounter(model="2B", success=True, exam=False, guardian=False):
    messages = [
        Message("guardian" if guardian else "patient", "A test-only opening."),
        Message("doctor", "A test-only question."),
    ]
    if exam:
        messages.append(Message("system", "A test-only result.", "exam"))
    messages.append(Message("doctor", "A test-only diagnosis.", "diagnosis"))
    return Encounter(
        model,
        model,
        model,
        success,
        tuple(messages),
        guardian,
        guardian,
        source="test fixture",
    )


def run_until(h, predicate, limit=1800):
    for _ in range(limit * 20):
        if predicate():
            return
        h.tick(0.05)
    raise AssertionError("Simulation failed to reach the expected state")


class HospitalTests(unittest.TestCase):
    def test_complete_outcomes_and_payment(self):
        r = Replay(
            {
                "2B": [encounter()],
                "4B": [encounter("4B", False)],
                "9B": [encounter("9B")],
            }
        )
        h = Hospital(r)
        run_until(h, lambda: h.recovered > 0)
        self.assertEqual(h.coins, 0, "Recovery alone must not award a coin")
        run_until(h, lambda: h.completed)
        self.assertEqual((h.recovered, h.deaths, h.coins), (2, 1, 2))
        self.assertAlmostEqual(h.survival, 2 / 3)
        real = [v for v in h.records if not v.encounter.npc]
        self.assertEqual(len(real), 3)
        self.assertEqual(sum((v.paid for v in real)), 2)
        stopped = h.time
        h.tick(100)
        self.assertEqual(h.coins, 2)
        self.assertEqual(h.time, stopped)

    def test_npc_does_not_change_metrics(self):
        h = Hospital(Replay())
        for i in range(12):
            h.arrive(npc_encounter(i, h.rng))
        run_until(h, lambda: h.completed)
        self.assertEqual((h.coins, h.deaths, h.recovered), (0, 0, 0))
        self.assertIsNone(h.survival)
        self.assertEqual(len(h.records), 12)

    def test_npc_probability_and_seed(self):
        a, b = (random.Random(42), random.Random(42))
        left = [npc_encounter(i, a).success for i in range(10000)]
        self.assertEqual(left, [npc_encounter(i, b).success for i in range(10000)])
        self.assertTrue(0.88 < sum(left) / len(left) < 0.92)

    def test_pause_freezes_everything(self):
        h = Hospital(Replay({"2B": [encounter(exam=True)], "4B": [], "9B": []}))
        h.tick(45)
        before = (
            h.time,
            h.clock,
            h.coins,
            [
                (v.patient, v.nurse, len(v.messages), v.state, v.elapsed)
                for v in h.visits
            ],
        )
        h.paused = True
        h.tick(1000)
        after = (
            h.time,
            h.clock,
            h.coins,
            [
                (v.patient, v.nurse, len(v.messages), v.state, v.elapsed)
                for v in h.visits
            ],
        )
        self.assertEqual(before, after)
        h.paused = False
        h.tick(1)
        self.assertGreater(h.time, before[0])

    def test_exam_exclusive_and_ordered(self):
        r = Replay(
            {m: [encounter(m, exam=True, guardian=True)] for m in ("2B", "4B", "9B")}
        )
        h = Hospital(r)
        states = {}
        previous = {}
        for _ in range(20000):
            h.tick(0.05)
            testing = [v for v in h.visits if v.state == "검사 중"]
            self.assertLessEqual(len(testing), 1)
            if testing:
                v = testing[0]
                self.assertIs(h.exam_owner, v)
                self.assertEqual(v.patient, LAB_PATIENT)
                self.assertEqual(v.nurse, LAB_NURSE)
            for v in h.visits:
                if v.encounter.npc or v.current is None:
                    continue
                if v.current.kind == "move":
                    mark = v.current.value
                    if previous.get(v.uid) != mark:
                        states.setdefault(v.uid, []).append(mark)
                        previous[v.uid] = mark
            if h.completed:
                break
        self.assertTrue(h.completed)
        for v in h.records:
            if v.encounter.npc:
                continue
            moves = states[v.uid]
            self.assertLess(
                moves.index(("nurse", door(v.room))),
                moves.index(
                    ("patient", door(v.room)), moves.index(("nurse", door(v.room)))
                ),
            )
            self.assertLess(
                moves.index(("patient", LAB_PATIENT)), moves.index(("nurse", LAB_NURSE))
            )
            self.assertLess(
                moves.index(("nurse", LAB_OUT)), moves.index(("patient", LAB_OUT))
            )
            returning = moves[moves.index(("patient", LAB_OUT)) + 1 :]
            self.assertLess(
                returning.index(("patient", seat(v.room))),
                returning.index(("nurse", nurse_home(v.room))),
            )
            self.assertEqual([m.kind for m in v.messages].count("exam"), 1)

    def test_five_rooms_and_queues_do_not_deadlock(self):
        r = Replay(
            {
                m: [encounter(m, success=i % 2 == 0, exam=True) for i in range(8)]
                for m in ("2B", "4B", "9B")
            }
        )
        h = Hospital(r)
        run_until(h, lambda: h.completed, limit=3000)
        self.assertEqual(len([v for v in h.records if not v.encounter.npc]), 24)
        self.assertEqual(h.coins, 12)
        self.assertEqual(h.deaths, 12)
        self.assertFalse(h.exam_queue)
        self.assertFalse(h.payment_queue)
        self.assertTrue(all((v is None for v in h.rooms)))

    def test_speed_preserves_results(self):
        r = Replay({"2B": [encounter(exam=True)], "4B": [], "9B": []})
        outcomes = []
        for speed in (0.5, 1, 2, 4):
            h = Hospital(r)
            h.speed = speed
            run_until(h, lambda: h.completed)
            outcomes.append(
                (
                    h.coins,
                    h.deaths,
                    [(v.encounter.key, v.encounter.success) for v in h.records],
                )
            )
        self.assertTrue(all((result == outcomes[0] for result in outcomes)))

    def test_export_preserves_full_conversation_and_source(self):
        h = Hospital(
            Replay({"2B": [encounter(exam=True, guardian=True)], "4B": [], "9B": []})
        )
        run_until(h, lambda: h.completed)
        row = next((v for v in h.export_records() if not v["npc"]))
        self.assertEqual(row["messages"][0]["role"], "guardian")
        self.assertEqual(len(row["messages"]), 4)
        self.assertEqual(row["source"], "test fixture")
        self.assertTrue(row["paid"])

    def test_clock_rollover_and_night(self):
        h = Hospital(Replay())
        h.time = 80
        self.assertTrue(h.night)
        h.time = 121
        self.assertEqual(h.day, 2)
        h.time = 165
        self.assertFalse(h.night)

    def test_empty_replay_is_not_synthetic_agent_data(self):
        h = Hospital(Replay())
        h.tick(1)
        self.assertTrue(h.completed)
        self.assertEqual(h.visits, [])

    def test_ui_pause_blocks_room_and_live_has_no_action(self):
        h = Hospital(Replay())
        s = Screen()
        s.render(h)
        self.assertIsNone(s.hit((221, 63)))
        s.activate("menu", h)
        s.render(h)
        self.assertEqual(s.hit((1100, 180)), "room:0")
        s.activate("room:0", h)
        self.assertEqual(s.zoom, 0)
        s.activate("pause", h)
        s.activate("room:2", h)
        self.assertEqual(s.zoom, 0)
        s.render(h)
        self.assertEqual(s.hit((720, 430)), "resume")
        self.assertIsNone(s.hit((1180, 130)))
        s.activate("resume", h)
        self.assertFalse(h.paused)

    def test_missing_score_rejected(self):
        raw = {"trace": {"id": "fixture", "events": []}}
        with self.assertRaises(ValueError):
            adapt_episode(raw, "2B", "test")
        raw["score"] = {"correct": "false"}
        with self.assertRaises(ValueError):
            adapt_episode(raw, "2B", "test")

    def test_unavailable_exam_does_not_make_lab_trip(self):
        raw = {
            "score": {"correct": False},
            "trace": {
                "id": "fixture",
                "events": [
                    {
                        "type": "request_exam",
                        "query": "X",
                        "status": "unavailable",
                        "message": "Not provided",
                    }
                ],
                "initial_doctor_view": {"public_context": "Opening"},
            },
        }
        ep = adapt_episode(raw, "4B", "test")
        self.assertFalse(any((m.kind == "exam" for m in ep.messages)))
        self.assertIn("Not provided", ep.messages[2].text)

    def test_guardian_and_legacy_exam_and_image(self):
        raw = {
            "score": {"correct": True},
            "trace": {
                "id": "fixture",
                "events": [
                    {
                        "type": "ask_patient",
                        "speaker": "guardian",
                        "utterance": "Test reply",
                        "query": "Q",
                    },
                    {
                        "type": "request_exam",
                        "query": "Exam",
                        "exam": {"text": "Saved finding", "images": ["test.png"]},
                    },
                ],
                "initial_doctor_view": {
                    "public_context": "My 9-year-old daughter has a test symptom."
                },
                "result": {"submission": {"answer": "A"}},
                "final_choices": {"A": "Saved diagnosis"},
            },
        }
        ep = adapt_episode(raw, "9B", "test")
        self.assertTrue(ep.child and ep.guardian)
        self.assertEqual(ep.diagnosis, "Saved diagnosis")
        self.assertEqual(
            next((m for m in ep.messages if m.kind == "exam")).images[0].name,
            "test.png",
        )

    def test_names_are_stable_and_do_not_consume_clinical_rng(self):
        h = Hospital(Replay())
        before = h.rng.getstate()
        a = h.arrive(encounter("2B"))
        b = h.arrive(encounter("2B"))
        self.assertEqual(a.name, b.name)
        self.assertIn("(2B)", a.display_name)
        self.assertEqual(before, h.rng.getstate())
        first, again = (Pseudonyms(42), Pseudonyms(42))
        names = [first.get(str(i)) for i in range(100)]
        self.assertEqual(len(set(names)), 100)
        self.assertEqual(names, [again.get(str(i)) for i in range(100)])

    def test_npc_rooms_are_inert_and_not_exported(self):
        h = Hospital(Replay())
        h.arrive(npc_encounter(0, h.rng))
        run_until(h, lambda: h.completed)
        s = Screen()
        s.render(h)
        self.assertFalse(any((action in ("room:3", "room:4") for _, action in s.hits)))
        for r in (3, 4):
            s.activate(f"room:{r}", h)
            self.assertIsNone(s.zoom)
        s.overlay = "history"
        s.render(h)
        self.assertIsNone(s.selected)
        self.assertFalse(any((action.startswith("record:") for _, action in s.hits)))
        self.assertEqual(h.export_records(), [])

    def test_daylight_changes_continuously_and_inside_stays_lit(self):
        from clinical_memory.simulation.art import render_world

        h = Hospital(Replay())
        samples = []
        for hour in (17, 17.5, 18, 18.5, 19):
            h.time = (hour * 60 - 480) / 8
            samples.append(h.daylight)
        self.assertEqual(samples[0], 1)
        self.assertEqual(samples[-1], 0)
        self.assertTrue(all((a > b for a, b in zip(samples, samples[1:]))))
        for hour in (5, 6, 7, 17, 18, 19, 24):
            h.time = (hour * 60 - 480 - 0.001) / 8
            before = h.daylight
            h.time = (hour * 60 - 480 + 0.001) / 8
            after = h.daylight
            self.assertLess(abs(before - after), 0.0001)
        h.time = 0
        day = render_world(h)
        h.time = 90
        night = render_world(h)
        self.assertEqual(day.getpixel((350, 240)), night.getpixel((350, 240)))
        self.assertNotEqual(day.getpixel((10, 10)), night.getpixel((10, 10)))

    def test_bilingual_ui_keeps_clinical_text_and_native_resolution(self):
        h = Hospital(Replay({"2B": [encounter()], "4B": [], "9B": []}))
        run_until(h, lambda: any((v.messages for v in h.visits)))
        s = Screen()
        s.zoom = 0
        text_before = [m.text for v in h.visits for m in v.messages]
        labels = []
        original = Draw.draw_text

        def capture(target, xy, value, *args, **kwargs):
            labels.append(value)
            return original(target, xy, value, *args, **kwargs)

        s.activate("language", h)
        with patch.object(Draw, "draw_text", capture):
            for scale in (1, 1.5, 2):
                image = s.render(h, scale)
                self.assertEqual(image.size, (int(1440 * scale), int(960 * scale)))
                self.assertEqual(s.hit((60, 130)), "pause")
            s.overlay = "help"
            s.render(h)
        self.assertTrue(
            all((not any(("가" <= c <= "힣" for c in label)) for label in labels))
        )
        self.assertEqual(text_before, [m.text for v in h.visits for m in v.messages])
        s.activate("language", h)
        self.assertEqual(s.language, "ko")

    def test_clinical_prefix_removed_without_losing_english(self):
        self.assertEqual(
            clean_clinical_text("미제공 요청: No information available."),
            "No information available.",
        )
        self.assertEqual(
            clean_clinical_text("First line.\n미제공 요청: Second line."),
            "First line.\nSecond line.",
        )

    def test_local_images_render_enlarge_close_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "exam.png"
            Image.new("RGB", (120, 80), (220, 40, 60)).save(source)
            attachment = resolve_image("exam.png", [root])
            self.assertEqual(Path(attachment.path), source)
            self.assertFalse(resolve_image("../outside.png", [root]).path)
            self.assertIsNone(clinical_image(""))
            raw = {
                "score": {"correct": True},
                "trace": {
                    "id": "fixture",
                    "events": [
                        {
                            "type": "request_exam",
                            "exam": {"text": "A finding.", "images": ["exam.png"]},
                        }
                    ],
                },
            }
            episode = adapt_episode(raw, "2B", "fixture", [root])
            h = Hospital(Replay({"2B": [episode], "4B": [], "9B": []}))
            run_until(h, lambda: h.completed)
            exported = h.export_records()[0]
            self.assertEqual(exported["messages"][0]["images"][0]["path"], str(source))
            self.assertTrue(exported["patient_name"])
            s = Screen()
            s.zoom = 0
            s.render(h)
            image_hit = next(
                ((box, action) for box, action in s.hits if action.startswith("image:"))
            )
            box, action = image_hit
            self.assertTrue(306 <= box[1] < box[3] <= 744)
            s.activate(action, h)
            self.assertEqual(s.selected_image, attachment)
            enlarged = s.render(h, 1.5)
            self.assertEqual(enlarged.getpixel((1080, 720))[:3], (220, 40, 60))
            s.activate("close", h)
            self.assertIsNone(s.selected_image)

    def test_model_mismatch_does_not_relabel(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "seal.json").write_text(json.dumps({"doctor": "9B"}))
            (root / "config.json").write_text(json.dumps({"sources": {"2B": "."}}))
            replay = load_replay(root / "config.json")
            self.assertEqual(replay.count, 0)
            self.assertTrue(any(("does not match" in w for w in replay.warnings)))

    def test_assignment_bubbles_and_wall_tv_show_current_patients_and_expire(self):
        from clinical_memory.simulation.art import draw_calls, call_tile, wall_tv

        h = Hospital(Replay())
        old = h.arrive(encounter())
        current = h.arrive(encounter())
        other = h.arrive(encounter())
        h.time = 10
        for v, desk, at, room in ((old, 0, 6, 0), (current, 0, 8, 1), (other, 1, 9, 2)):
            v.desk = desk
            v.room = room
            v.assigned_at = at
            v.state = "진료실 이동"
        self.assertEqual(h.active_calls(), (current, other))
        canvas = Image.new("RGBA", (1200, 700))
        with patch("clinical_memory.simulation.art.call_tile", wraps=call_tile) as tile:
            draw_calls(canvas, h)
            on = wall_tv(h)
            self.assertEqual(
                [call.args[1].uid for call in tile.call_args_list],
                [current.uid, other.uid, current.uid, other.uid],
            )
        h.paused = True
        h.tick(20)
        self.assertEqual(h.active_calls(), (current, other))
        self.assertEqual(wall_tv(h).tobytes(), on.tobytes())
        current.state = "문진 중"
        self.assertEqual(h.active_calls(), (other,), "An older call must not reappear")
        h.time = 16
        self.assertFalse(h.active_calls())
        with patch("clinical_memory.simulation.art.call_tile", wraps=call_tile) as tile:
            draw_calls(canvas, h)
            off = wall_tv(h)
            tile.assert_not_called()
        self.assertNotEqual(on.tobytes(), off.tobytes())

    def test_fading_ghost_is_composited_in_front_of_doctor(self):
        from clinical_memory.simulation.art import render_world

        h = Hospital(Replay())
        background = render_world(h)
        v = h.arrive(encounter(success=False))
        v.room = 0
        v.patient = seat(0)
        v.nurse = nurse_home(0)
        v.state = "영혼"
        v.resolved = True
        v.ghost = 0.5
        h.rooms[0] = v
        point = (229, 171)
        underneath = background.getpixel(point)
        expected = Image.alpha_composite(
            Image.new("RGBA", (1, 1), underneath),
            Image.new("RGBA", (1, 1), (239, 247, 255, 127)),
        ).getpixel((0, 0))
        self.assertNotEqual(expected, underneath)
        self.assertEqual(render_world(h).getpixel(point), expected)
        v.ghost = 1
        self.assertEqual(render_world(h).getpixel(point), underneath)

    def test_people_pass_in_front_of_and_behind_coin_stack(self):
        from clinical_memory.simulation.art import render_world

        for count in (1, 30, 150):
            with self.subTest(coins=count):
                h = Hospital(Replay())
                h.coins = count
                coins_only = render_world(h)
                v = h.arrive(encounter())
                v.patient = (946, 620)
                front = render_world(h)
                h.coins = 0
                person_only = render_world(h)
                point = (946, 592)
                self.assertNotEqual(
                    person_only.getpixel(point), coins_only.getpixel(point)
                )
                self.assertEqual(front.getpixel(point), person_only.getpixel(point))
                h.coins = count
                v.patient = (946, 610)
                behind = render_world(h)
                self.assertEqual(behind.getpixel(point), coins_only.getpixel(point))

    def test_doctors_are_behind_desks_and_patients_are_in_front(self):
        from clinical_memory.simulation.art import (
            render_world,
            character,
            PATIENT_COLORS,
        )
        from clinical_memory.simulation.engine import ROOMS

        h = Hospital(Replay())
        empty = render_world(h)
        for room, (x, y, _, _) in enumerate(ROOMS):
            with self.subTest(room=room):
                table_y = y + (85 if room < 3 else 76)
                point = (x + 115, table_y + 10)
                self.assertEqual(
                    empty.getpixel(point),
                    (199, 166, 118, 255),
                    "The desk must cover the doctor's lower body",
                )
                v = h.arrive(encounter())
                v.patient = seat(room)
                v.room = room
                self.assertGreater(v.patient[1], table_y + 35)
                visible = render_world(h)
                actor = Image.new("RGBA", empty.size)
                seated_pos = (v.patient[0], v.patient[1] + 26)
                character(
                    actor,
                    seated_pos,
                    "patient",
                    v.uid,
                    PATIENT_COLORS[v.uid % 6],
                    shadow=False,
                )
                point = (x + 115, int(seated_pos[1]) - 28)
                self.assertEqual(visible.getpixel(point), actor.getpixel(point))
                v.done = True

    def test_world_text_is_occluded_at_native_display_resolution(self):
        from clinical_memory.simulation.art import (
            world_layers,
            draw_coin_stack,
            character,
            PATIENT_COLORS,
        )

        h = Hospital(Replay())
        h.coins = 150
        v = h.arrive(encounter())
        v.patient = (976, 630)
        layers = world_layers(h)
        no_text = [replace(layer, labels=()) for layer in layers]
        mask = Image.new("RGBA", (1200, 700))
        draw_coin_stack(mask, h.coins)
        character(mask, v.patient, "patient", v.uid, PATIENT_COLORS[v.uid % 6])
        for scale in (1, 1.5, 2):
            with self.subTest(scale=scale):
                screen = Screen()
                screen.scale = scale
                labeled = Surface((1200, 700), (0, 0, 0, 0), scale)
                plain = Surface((1200, 700), (0, 0, 0, 0), scale)
                screen._world_region(labeled, h, layers, (0, 0, 1200, 700), (0, 0), 1)
                screen._world_region(plain, h, no_text, (0, 0, 1200, 700), (0, 0), 1)
                difference = ImageChops.difference(
                    labeled.image.convert("RGB"), plain.image.convert("RGB")
                )
                self.assertIsNotNone(
                    difference.getbbox(), "Visible wall text must still be drawn"
                )
                opaque = mask.getchannel("A").transform(
                    difference.size,
                    Image.Transform.AFFINE,
                    (1 / scale, 0, 0, 0, 1 / scale, 0),
                    Image.Resampling.NEAREST,
                )
                opaque = opaque.filter(ImageFilter.MinFilter(3))
                covered = Image.composite(
                    difference, Image.new("RGB", difference.size), opaque
                )
                self.assertIsNone(
                    covered.getbbox(), "Text must not bleed through people or coins"
                )

    def test_curtain_hides_exam_team_but_not_people_in_front(self):
        from clinical_memory.simulation.art import (
            render_world,
            character,
            PATIENT_COLORS,
        )

        h = Hospital(Replay())
        v = h.arrive(encounter(exam=True))
        v.room = 0
        v.patient = LAB_PATIENT
        v.nurse = LAB_NURSE
        v.state = "검사 중"
        h.rooms[0] = v
        h.exam_owner = v
        inside = render_world(h)
        v.done = True
        hidden = render_world(h)
        for point in ((924, 490), (978, 490)):
            self.assertEqual(inside.getpixel(point), hidden.getpixel(point))
        passer = h.arrive(encounter())
        passer.patient = (924, 560)
        front = render_world(h)
        actor = Image.new("RGBA", front.size)
        character(
            actor, passer.patient, "patient", passer.uid, PATIENT_COLORS[passer.uid % 6]
        )
        point = (924, 520)
        self.assertNotEqual(front.getpixel(point), hidden.getpixel(point))
        self.assertEqual(front.getpixel(point), actor.getpixel(point))

    def test_legacy_dataset_image_references_and_persistent_thumbnails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (100, 60), (50, 180, 110)).save(root / "exam.png")
            for data in (
                {"text": "Image: images/exam.png"},
                {"text": "Finding", "images": "images/exam.png"},
                {"text": "Finding", "image": {"path": "images/exam.png"}},
            ):
                raw = {
                    "score": {"correct": True},
                    "trace": {
                        "id": "fixture",
                        "events": [{"type": "request_exam", "exam": data}],
                    },
                }
                ep = adapt_episode(raw, "2B", "fixture", [root])
                exam = next((m for m in ep.messages if m.kind == "exam"))
                self.assertEqual(len(exam.images), 1)
                self.assertEqual(Path(exam.images[0].path), root / "exam.png")
            self.assertFalse(resolve_image("../exam.png", [root]).path)
            self.assertFalse(resolve_image("https://example.com/exam.png", [root]).path)
            h = Hospital(Replay())
            v = h.arrive(ep)
            v.room = 0
            h.rooms[0] = v
            v.messages = [exam, Message("doctor", "A later discussion.\n" * 40)]
            s = Screen()
            s.zoom = 0
            rendered = s.render(h)
            thumbnails = [
                box
                for box, action in s.hits
                if action.startswith("image:") and box[1] < 380
            ]
            self.assertEqual(len(thumbnails), 1)
            x1, y1, x2, y2 = thumbnails[0]
            self.assertEqual(
                rendered.getpixel((int((x1 + x2) / 2), int((y1 + y2) / 2)))[:3],
                (50, 180, 110),
            )

    def test_default_speed_and_grouped_help_without_sources_button(self):
        h = Hospital(Replay())
        self.assertEqual(h.speed, 1)
        s = Screen()
        s.render(h)
        self.assertNotIn("sources", [action for _, action in s.hits])
        s.activate("help", h)
        labels = []
        original = Draw.draw_text

        def capture(target, xy, value, *args, **kwargs):
            labels.append(value)
            return original(target, xy, value, *args, **kwargs)

        with patch.object(Draw, "draw_text", capture):
            s.render(h)
        for label in ("프로그램 안내", "조작 방법", "데이터 출처"):
            self.assertIn(label, labels)


if __name__ == "__main__":
    unittest.main()
