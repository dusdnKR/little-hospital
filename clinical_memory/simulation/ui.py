"""Desktop screen composition. The same renderer is used for GUI and QA images."""

import math
import re
from functools import lru_cache
from PIL import Image, ImageOps
from .art import BG, INK, MUTED, font, text, sprite, world_layers
from .engine import ROOMS, COLORS
from .graphics import Surface, draw
from .identity import DOCTORS, FULL_MODELS, doctor_label
from .i18n import tr, state_text, HELP

WIDTH, HEIGHT = (1440, 960)


@lru_cache(maxsize=64)
def clinical_image(path):
    if not path:
        return None
    try:
        with Image.open(path) as source:
            return ImageOps.exif_transpose(source).convert("RGBA")
    except (OSError, ValueError, Image.DecompressionBombError):
        return None


@lru_cache(maxsize=2048)
def wrap(value, width, size=14):
    f = font(size)
    lines = []
    for paragraph in str(value).split("\n"):
        line = ""
        for token in re.findall("\\S+\\s*|\\s+", paragraph):
            if line and f.getlength(line + token) > width:
                lines.append(line)
                line = ""
            for char in token:
                if line and f.getlength(line + char) > width:
                    lines.append(line)
                    line = ""
                line += char
        lines.append(line)
    return lines


class Screen:
    def __init__(self, language="ko"):
        self.language = language
        self.scale = 1.0
        self.image_targets = []
        self.selected_image = None
        self.zoom = None
        self.menu = False
        self.overlay = None
        self.filter = "ALL"
        self.selected = None
        self.record_offset = 0
        self.chat_scroll = 0
        self.detail_scroll = 0
        self.help_scroll = 0
        self.hits = []
        self.hover = None
        self.toast = ""
        self.last_chat = None
        self.scroll_limits = {"chat": 0, "detail": 0, "records": 0, "help": 0}

    def t(self, key, **values):
        return tr(self.language, key, **values)

    def hit(self, point):
        x, y = point
        for rect, action in reversed(self.hits):
            if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                return action
        return None

    def button(self, d, box, label, action, accent=None, active=False, size=14):
        fill = "#2c4051" if active or self.hover == action else "#1b2c3a"
        d.rounded_rectangle(
            box, radius=8, fill=fill, outline=accent or "#3b4f60", width=1
        )
        text(
            d,
            ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
            label,
            size,
            accent or INK,
            active,
            "mm",
        )
        self.hits.append((box, action))

    def portrait(self, im, xy, kind, index, color, happy=False, size=36):
        s = sprite(kind, index, color, happy).crop((3, 1, 21, 19))
        im.alpha_composite(s, xy, size=(size, size))

    def _header(self, im, h):
        d = draw(im)
        d.rectangle((0, 0, WIDTH, 100), fill="#152431")
        d.line((24, 99, 1416, 99), fill="#314453")
        cx, cy, radius = (59, 49, 28)
        d.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            fill="#283b55",
            outline="#829caf",
            width=2,
        )
        d.pieslice(
            (cx - radius + 3, cy - radius + 3, cx + radius - 3, cy + radius - 3),
            180,
            360,
            fill="#d9b568",
        )
        for i in range(12):
            a = i * math.pi / 6
            d.rectangle(
                (
                    cx + math.sin(a) * 21 - 1,
                    cy - math.cos(a) * 21 - 1,
                    cx + math.sin(a) * 21 + 1,
                    cy - math.cos(a) * 21 + 1,
                ),
                fill="#eee9d0",
            )
        a = h.clock_minutes % 1440 / 1440 * 2 * math.pi - math.pi / 2
        d.line(
            (cx, cy, cx + math.cos(a) * 19, cy + math.sin(a) * 19),
            fill="#fff8df",
            width=3,
        )
        d.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill="#fff8df")
        text(d, (102, 20), f"DAY {h.day:02}  /  {h.clock}", 20, bold=True)
        text(d, (103, 53), self.t("night_shift" if h.night else "day_shift"), 12, MUTED)
        d.rounded_rectangle((206, 54, 238, 70), radius=8, fill="#354551")
        d.ellipse((208, 56, 220, 68), fill="#788994")
        text(d, (245, 54), "LIVE OFF", 11, "#7b909f", True)
        text(d, (378, 20), "Little Hospital", 28, "#f3f0df", True)
        text(d, (380, 58), self.t("subtitle"), 12, MUTED)
        stats = [
            (916, "coins", str(h.coins), "#e7bf64"),
            (1084, "deaths", str(h.deaths), "#dc8587"),
            (
                1243,
                "survival",
                "—" if h.survival is None else f"{h.survival:.0%}",
                "#79c5a3"
                if h.survival is None or h.survival >= 0.7
                else "#e7bf64"
                if h.survival >= 0.4
                else "#e88585",
            ),
        ]
        for x, label, value, color in stats:
            if label == "coins":
                d.ellipse((x, 32, x + 28, 60), fill="#a77c39", outline=color, width=3)
                text(d, (x + 14, 46), "C", 17, color, True, "mm")
            elif label == "deaths":
                d.ellipse((x, 29, x + 27, 53), fill="#d2dce1")
                d.rectangle((x + 5, 48, x + 22, 59), fill="#d2dce1")
                d.rectangle((x + 5, 38, x + 10, 44), fill=BG)
                d.rectangle((x + 17, 38, x + 22, 44), fill=BG)
                for tx in (9, 14, 19):
                    d.line((x + tx, 54, x + tx, 60), fill=BG, width=2)
            else:
                d.line(
                    (x, 46, x + 6, 46, x + 10, 35, x + 16, 58, x + 21, 44, x + 30, 44),
                    fill=color,
                    width=3,
                )
            text(d, (x + 42, 21), self.t(label), 11, MUTED, True)
            text(d, (x + 42, 41), value, 25, color, True)
            if label == "survival":
                text(d, (x + 42, 80), self.t("counted_rooms"), 10, MUTED)

    def _toolbar(self, d, h):
        self.button(d, (24, 115, 134, 150), "", "pause")
        for x in (38, 45):
            d.rectangle((x, 126, x + 3, 138), fill=INK)
        text(d, (88, 132), self.t("pause"), 13, INK, anchor="mm")
        for i, speed in enumerate((0.5, 1, 2, 4)):
            self.button(
                d,
                (146 + i * 50, 115, 190 + i * 50, 150),
                f"{speed:g}×",
                f"speed:{speed}",
                active=h.speed == speed,
                size=13,
            )
        self.button(d, (356, 115, 442, 150), "RESET", "restart")
        completed = sum((not v.encounter.npc for v in h.records))
        text(
            d,
            (462, 124),
            self.t("count", done=completed, total=h.replay.count),
            13,
            "#9fc7b5",
        )
        if self.zoom is not None:
            self.button(d, (828, 115, 962, 150), self.t("home"), "home", size=13)
        self.button(d, (974, 115, 1125, 150), "", "menu", active=self.menu)
        text(d, (1040, 132), self.t("chat"), 13, INK, anchor="mm")
        d.line(
            (
                (1105, 135 if self.menu else 129),
                (1110, 130 if self.menu else 134),
                (1115, 135 if self.menu else 129),
            ),
            fill=INK,
            width=2,
        )
        self.button(d, (1137, 115, 1276, 150), self.t("history"), "history", size=13)
        self.button(d, (1288, 115, 1364, 150), "KO / EN", "language", size=12)
        self.button(d, (1376, 115, 1416, 150), "?", "help", size=18)

    def _world_region(self, im, h, world, crop, destination, factor):
        left, top, right, bottom = crop
        region = Surface(
            ((right - left) * factor, (bottom - top) * factor), (0, 0, 0, 0), self.scale
        )
        d = draw(region)
        sx, sy = (
            region.image.width / (right - left),
            region.image.height / (bottom - top),
        )
        for layer in world:
            x, y = layer.position
            l, t = (
                max(0, math.floor((x - left) * sx)),
                max(0, math.floor((y - top) * sy)),
            )
            r = min(region.image.width, math.ceil((x + layer.image.width - left) * sx))
            b = min(region.image.height, math.ceil((y + layer.image.height - top) * sy))
            if l < r and t < b:
                pixels = layer.image.transform(
                    (r - l, b - t),
                    Image.Transform.AFFINE,
                    (1 / sx, 0, l / sx + left - x, 0, 1 / sy, t / sy + top - y),
                    Image.Resampling.NEAREST,
                )
                region.image.alpha_composite(pixels, (l, t))
            for (x, y), value, size, fill, bold, anchor in layer.labels:
                text(
                    d,
                    ((x - left) * factor, (y - top) * factor),
                    value,
                    size * factor,
                    fill,
                    bold,
                    anchor,
                )
        im.alpha_composite(region, destination)

    def _world(self, im, h):
        d = draw(im)
        world = world_layers(h, self.language)
        if self.zoom is None:
            box = (24, 164, 1416, 796)
            d.rounded_rectangle(
                box, radius=12, fill="#21333c", outline="#3b525b", width=1
            )
            factor = 628 / 700
            x, y = ((WIDTH - 1200 * factor) / 2, 166)
            self._world_region(im, h, world, (0, 0, 1200, 700), (x, y), factor)
            for r, room in enumerate(ROOMS[:3]):
                self.hits.append(
                    (
                        (
                            x + room[0] * factor,
                            y + room[1] * factor,
                            x + room[2] * factor,
                            y + room[3] * factor,
                        ),
                        f"room:{r}",
                    )
                )
            text(d, (45, 188), self.t("overview"), 11, "#a4c3bf", True)
            text(d, (45, 212), "24H", 25, "#dce9de", True)
            text(d, (43, 260), self.t("controls"), 11, "#a4c3bf", True)
            for i, (key, label) in enumerate(
                (
                    ("SPACE", "key_pause"),
                    ("1–3", "key_rooms"),
                    ("ESC", "key_back"),
                    (self.t("wheel"), "key_scroll"),
                )
            ):
                y = 293 + i * 64
                text(d, (43, y), key, 12, INK, True)
                text(d, (43, y + 23), self.t(label), 10, MUTED)
            text(d, (1287, 189), "AGENTS", 11, "#a4c3bf", True)
            for i, color in enumerate(COLORS[:3]):
                d.rectangle((1287, 220 + i * 34, 1293, 227 + i * 34), fill=color)
                text(d, (1300, 216 + i * 34), FULL_MODELS[i], 11, color, True)
        else:
            r = self.zoom
            x, y, right, bottom = ROOMS[r]
            crop = (x - 30, y - 40, right + 30, bottom + 70)
            dest = (24, 164, 948, 796)
            d.rounded_rectangle(
                dest, radius=12, fill="#23383e", outline=COLORS[r], width=2
            )
            cw, ch = (crop[2] - crop[0], crop[3] - crop[1])
            factor = min(900 / cw, 536 / ch)
            self._world_region(
                im,
                h,
                world,
                crop,
                (486 - cw * factor / 2, 480 - ch * factor / 2),
                factor,
            )
            text(
                d,
                (44, 181),
                f"ROOM 0{r + 1}  /  {doctor_label(r)}",
                16,
                COLORS[r],
                True,
            )
            self.button(d, (44, 742, 192, 778), self.t("home"), "home")
            self._chat_panel(im, h, r)

    def _cards(self, im, h):
        d = draw(im)
        for r in range(5):
            x = 24 + r * 280
            color = COLORS[r]
            box = (x, 812, x + 272, 910)
            d.rounded_rectangle(
                box,
                radius=10,
                fill="#1b2b38",
                outline=color if self.zoom == r else "#334858",
                width=2,
            )
            d.rectangle((x + 1, 830, x + 4, 890), fill=color)
            self.portrait(im, (x + 14, 830), "doctor", r, color, size=42)
            text(d, (x + 66, 824), DOCTORS[r], 15, color, True)
            if r < 3:
                text(d, (x + 66, 846), f"({FULL_MODELS[r]})", 12, color)
            v = h.rooms[r]
            text(
                d,
                (x + 66, 867),
                state_text(self.language, v.state) if v else self.t("waiting"),
                11,
                INK,
            )
            visits = [v for v in h.records if v.room == r]
            wins = sum((v.encounter.success for v in visits))
            if r < 3:
                text(
                    d,
                    (x + 16, 891),
                    self.t("card_count", wins=wins, visits=len(visits)),
                    10,
                    MUTED,
                )
                self.hits.append((box, f"room:{r}"))
        event = self.t("welcome")
        if h.events:
            _, key, values = h.events[0]
            values = dict(values)
            if "result" in values:
                values["result"] = self.t(values["result"])
            event = self.t(key, **values)
        text(d, (1416, 930), self.toast or event, 11, "#a4c9b8", anchor="ra")

    def _messages(self, im, messages, v, region, scroll=0, limit_key="chat"):
        x, y, w, hh = region
        entries = []
        total = 12
        for msg in messages:
            system = msg.role == "system"
            lines = wrap(msg.text, w - (56 if system else 148), 14)
            height = 46 + len(lines) * 21
            pictures = []
            for attachment in msg.images:
                picture = clinical_image(attachment.path)
                if picture is not None:
                    factor = min((w - 64) / picture.width, 180 / picture.height)
                    size = (
                        round(picture.width * factor),
                        round(picture.height * factor),
                    )
                    pictures.append((attachment, picture, size, height))
                    height += size[1] + 32
                else:
                    pictures.append((attachment, None, (0, 0), height))
                    height += 28
            entries.append((total, height, msg, lines, pictures))
            total += height + 14
        maximum = max(0, total - hh)
        self.scroll_limits[limit_key] = maximum
        scroll = min(maximum, max(0, scroll))
        offset = max(0, total - hh) - scroll
        panel = Surface((w, hh), "#152430", self.scale)
        d = draw(panel)
        for top, height, msg, lines, pictures in entries:
            top -= offset
            if top + height < 0 or top > hh:
                continue
            system = msg.role == "system"
            right = msg.role in ("patient", "guardian")
            left = 16 if system else 54 if not right else 70
            end = w - 16 if system else w - 70 if not right else w - 54
            d.rounded_rectangle(
                (left, top, end, top + height),
                radius=10,
                fill="#2a3048" if system else "#28483f" if right else "#293d4c",
            )
            label = {
                "doctor": DOCTORS[v.room or 0],
                "patient": v.name,
                "guardian": "Guardian",
                "system": "Examination → Doctor's computer"
                if msg.kind == "exam"
                else "Examination notice",
            }.get(msg.role, msg.role)
            text(
                d,
                (left + 14, top + 10),
                label,
                11,
                "#b6b0dd" if system else "#a6d3b7" if right else "#9dbedc",
                True,
            )
            for i, line in enumerate(lines):
                text(d, (left + 14, top + 33 + i * 21), line, 14, INK)
            if not system:
                px = 14 if not right else w - 46
                r = v.room or 0
                patient_index = v.uid + 4 if msg.role == "guardian" else v.uid
                palette = (
                    "#897dbc",
                    "#d68965",
                    "#759ba0",
                    "#6f94c5",
                    "#c184a3",
                    "#d3b362",
                )
                patient_color = (
                    "#79939b" if msg.role == "guardian" else palette[v.uid % 6]
                )
                portrait = sprite(
                    "patient" if right else "doctor",
                    patient_index if right else r,
                    patient_color if right else COLORS[r],
                    v.resolved and v.encounter.success,
                ).crop((3, 1, 21, 19))
                panel.alpha_composite(portrait, (px, top + 5), size=(32, 32))
            for attachment, picture, size, relative in pictures:
                ix, iy = (left + 14, top + relative)
                if picture is not None:
                    panel.alpha_composite(picture, (ix, iy), size=size, smooth=True)
                    text(d, (ix, iy + size[1] + 5), attachment.name, 10, MUTED)
                    clipped = (
                        x + ix,
                        y + max(0, iy),
                        x + ix + size[0],
                        y + min(hh, iy + size[1]),
                    )
                    if clipped[1] < clipped[3]:
                        target = len(self.image_targets)
                        self.image_targets.append(attachment)
                        self.hits.append((clipped, f"image:{target}"))
                else:
                    text(
                        d, (ix, iy), f"Image unavailable: {attachment.name}", 11, MUTED
                    )
        if maximum:
            d.rounded_rectangle((w - 5, 0, w - 2, hh), radius=2, fill="#354956")
            thumb = max(28, hh * hh / max(total, 1))
            ty = (hh - thumb) * offset / maximum
            d.rounded_rectangle(
                (w - 5, ty, w - 2, ty + thumb), radius=2, fill="#9aafbb"
            )
        im.alpha_composite(panel, (x, y))

    def _chat_panel(self, im, h, room):
        d = draw(im)
        d.rounded_rectangle(
            (964, 164, 1416, 796), radius=12, fill="#152430", outline="#3b5262"
        )
        text(d, (986, 182), self.t("transcript"), 20, INK, True)
        self.button(d, (1287, 179, 1397, 211), self.t("previous"), "history", size=12)
        v = h.rooms[room]
        if v is None:
            v = next((v for v in reversed(h.records) if v.room == room), None)
        key = (room, v.uid if v else None)
        if key != self.last_chat:
            self.chat_scroll = 0
            self.last_chat = key
        if v is None:
            text(d, (986, 243), self.t("waiting_patient"), 16, MUTED)
            text(d, (986, 280), self.t("arrival_route"), 13, MUTED)
            return
        text(d, (986, 226), v.display_name, 14, COLORS[room], True)
        status = (
            state_text(self.language, v.state) if not v.done else self.t("finished")
        )
        text(d, (986, 252), f"{status} · {h.stamp(v.started)}", 11, MUTED)
        text(d, (986, 273), self.t("record_phase", phase=v.encounter.phase), 11, MUTED)
        received = []
        for msg in v.messages:
            for attachment in msg.images:
                if (
                    attachment not in received
                    and clinical_image(attachment.path) is not None
                ):
                    received.append(attachment)
        if received:
            text(d, (986, 294), "EXAMINATION IMAGES", 10, "#b6b0dd", True)
            for i, attachment in enumerate(received[-5:]):
                picture = clinical_image(attachment.path)
                x, y = (986 + i * 82, 314)
                box = (x, y, x + 74, y + 51)
                d.rounded_rectangle(box, radius=4, fill="#2a3848", outline="#758995")
                factor = min(70 / picture.width, 47 / picture.height)
                size = (picture.width * factor, picture.height * factor)
                im.alpha_composite(
                    picture,
                    (x + (74 - size[0]) / 2, y + (51 - size[1]) / 2),
                    size=size,
                    smooth=True,
                )
                target = len(self.image_targets)
                self.image_targets.append(attachment)
                self.hits.append((box, f"image:{target}"))
        top = 380 if received else 306
        self._messages(im, v.messages, v, (978, top, 424, 744 - top), self.chat_scroll)
        if v.speaker:
            label = (
                DOCTORS[room]
                if v.speaker == "doctor"
                else "Guardian"
                if v.speaker == "guardian"
                else v.name
            )
            text(
                d,
                (985, 764),
                self.t("speaking", name=label, dots="." * (int(h.time * 3) % 4)),
                11,
                "#a1c8b6",
            )
        else:
            text(d, (985, 764), self.t("scroll_chat"), 11, MUTED)
        self.button(d, (1290, 755, 1397, 785), self.t("latest"), "latest", size=11)

    def _veil(self, im, alpha=190):
        im.alpha_composite(Surface(im.size, (6, 14, 22, alpha), self.scale))

    def _history(self, im, h):
        self._veil(im)
        self.hits = []
        d = draw(im)
        d.rounded_rectangle(
            (70, 58, 1370, 905), radius=16, fill="#152430", outline="#597383", width=1
        )
        text(d, (100, 86), self.t("history"), 27, INK, True)
        text(d, (100, 128), self.t("history_hint"), 13, MUTED)
        self.button(d, (1116, 83, 1242, 120), self.t("json"), "export", size=12)
        self.button(d, (1254, 83, 1338, 120), self.t("close"), "close")
        for i, (key, label) in enumerate(
            (
                ("ALL", self.t("all")),
                ("2B", "2B"),
                ("4B", "4B"),
                ("9B", "9B"),
                ("SUCCESS", self.t("recovered")),
                ("FAILURE", self.t("dead")),
            )
        ):
            x = 100 + i * 93
            self.button(
                d,
                (x, 163, x + 85, 199),
                label,
                f"filter:{key}",
                active=self.filter == key,
                size=12,
            )
        self.button(d, (1152, 163, 1338, 199), self.t("csv"), "export_csv", size=12)
        records = [
            v
            for v in reversed(h.records)
            if not v.encounter.npc
            and (
                self.filter == "ALL"
                or self.filter == v.encounter.model
                or self.filter == ("SUCCESS" if v.encounter.success else "FAILURE")
            )
        ]
        self.scroll_limits["records"] = max(0, len(records) - 7)
        self.record_offset = min(self.record_offset, self.scroll_limits["records"])
        if self.selected not in [v.uid for v in records]:
            self.selected = records[0].uid if records else None
            self.detail_scroll = 0
        d.line((524, 218, 524, 871), fill="#354b5a")
        text(d, (100, 217), self.t("record_count", count=len(records)), 11, MUTED)
        for i, v in enumerate(records[self.record_offset : self.record_offset + 7]):
            x, y = (100, 247 + i * 84)
            active = v.uid == self.selected
            d.rounded_rectangle(
                (x, y, 505, y + 74),
                radius=8,
                fill="#2d4453" if active else "#1e303e",
                outline="#718a98" if active else "#354b5a",
            )
            text(d, (x + 14, y + 8), v.display_name, 13, INK, True)
            text(
                d,
                (x + 14, y + 31),
                f"{DOCTORS[v.room]} · {h.stamp(v.finished)}",
                10,
                MUTED,
            )
            text(
                d,
                (x + 14, y + 51),
                self.t("recovered" if v.encounter.success else "dead"),
                11,
                "#91cfaa" if v.encounter.success else "#e29191",
                True,
            )
            text(
                d,
                (490, y + 51),
                self.t("paid" if v.paid else "pending_payment")
                if v.encounter.success
                else "",
                10,
                MUTED,
                anchor="ra",
            )
            self.hits.append(((x, y, 505, y + 74), f"record:{v.uid}"))
        v = next((v for v in records if v.uid == self.selected), None)
        if v:
            text(
                d,
                (550, 220),
                f"{v.display_name}  /  {doctor_label(v.room)}",
                16,
                INK,
                True,
            )
            diagnosis = wrap(v.encounter.diagnosis, 748, 14)
            for i, line in enumerate(diagnosis[:2]):
                text(d, (550, 255 + i * 20), line, 14, "#cbbfa3")
            source = v.encounter.source.replace("\\", "/")
            parts = source.split("/")
            text(
                d,
                (550, 305),
                self.t("source_file", path="/".join(parts[-3:])),
                11,
                MUTED,
            )
            text(
                d,
                (550, 327),
                self.t(
                    "source_time",
                    phase=v.encounter.phase,
                    time=v.encounter.recorded_at or self.t("unrecorded"),
                ),
                11,
                MUTED,
            )
            self._messages(
                im, v.messages, v, (542, 362, 800, 495), self.detail_scroll, "detail"
            )
        else:
            text(d, (560, 300), self.t("empty_history"), 20, MUTED)
            text(d, (560, 339), self.t("history_later"), 14, MUTED)

    def _info(self, im, h):
        self._veil(im)
        self.hits = []
        d = draw(im)
        d.rounded_rectangle(
            (222, 108, 1218, 853), radius=16, fill="#182a38", outline="#657d8a"
        )
        text(d, (260, 143), self.t("help_title"), 28, INK, True)
        self.button(d, (1082, 140, 1178, 177), self.t("close"), "close")
        entries = []
        for section in ("program", "controls"):
            entries.append((self.t(section), 18, "#a6d3b7", True, 16))
            entries.extend(
                ((p, 13, MUTED, False, 10) for p in HELP[self.language][section])
            )
        entries.append((self.t("sources"), 18, "#a6d3b7", True, 16))
        entries.append((self.t("source_intro"), 13, MUTED, False, 10))
        for i, (model, files) in enumerate(h.replay.encounters.items()):
            entries.append(
                (
                    self.t("model_records", model="Qwen3.5-" + model, count=len(files)),
                    13,
                    COLORS[i],
                    True,
                    1,
                )
            )
            entries.append(
                (
                    h.replay.sources.get(model, "—").replace("\\", "/"),
                    11,
                    MUTED,
                    False,
                    10,
                )
            )
        entries.append((self.t("image_sources"), 13, "#b6b0dd", True, 1))
        entries.extend(
            (
                (path.replace("\\", "/"), 11, MUTED, False, 5)
                for path in h.replay.image_roots
            )
        )
        entries.append((self.t("source_config"), 12, MUTED, False, 8))
        entries.extend(
            ((warning, 12, "#e29191", False, 8) for warning in h.replay.warnings)
        )
        layout = []
        total = 0
        for value, size, color, bold, gap in entries:
            lines = wrap(value, 890, size)
            layout.append((total, lines, size, color, bold))
            total += len(lines) * (size + 7) + gap
        height = 624
        self.scroll_limits["help"] = max(0, total - height)
        self.help_scroll = min(self.help_scroll, self.scroll_limits["help"])
        panel = Surface((920, height), "#182a38", self.scale)
        pd = draw(panel)
        for y, lines, size, color, bold in layout:
            for line in lines:
                text(pd, (0, y - self.help_scroll), line, size, color, bold)
                y += size + 7
        if self.scroll_limits["help"]:
            thumb = max(28, height * height / total)
            sy = (height - thumb) * self.help_scroll / self.scroll_limits["help"]
            pd.rectangle((915, 0, 918, height), fill="#354956")
            pd.rectangle((915, sy, 918, sy + thumb), fill="#9aafbb")
        im.alpha_composite(panel, (260, 207))

    def _image_viewer(self, im):
        picture = clinical_image(self.selected_image.path)
        if picture is None:
            self.selected_image = None
            return
        self._veil(im, 235)
        self.hits = []
        d = draw(im)
        text(d, (72, 43), self.t("image_view"), 23, INK, True)
        text(d, (72, 78), self.selected_image.name, 13, MUTED)
        self.button(
            d, (1218, 45, 1368, 85), self.t("image_close"), "image_close", size=13
        )
        factor = min(1296 / picture.width, 790 / picture.height)
        size = (picture.width * factor, picture.height * factor)
        im.alpha_composite(
            picture,
            ((WIDTH - size[0]) / 2, 126 + (790 - size[1]) / 2),
            size=size,
            smooth=True,
        )

    def render(self, h, scale=1.0):
        self.scale = max(0.05, scale)
        self.hits = []
        self.image_targets = []
        im = Surface((WIDTH, HEIGHT), BG, self.scale)
        self._header(im, h)
        self._toolbar(draw(im), h)
        self._world(im, h)
        self._cards(im, h)
        d = draw(im)
        if self.menu:
            d.rounded_rectangle(
                (974, 155, 1258, 302), radius=10, fill="#152737", outline="#567183"
            )
            for r in range(3):
                self.button(
                    d,
                    (985, 166 + r * 43, 1247, 202 + r * 43),
                    doctor_label(r),
                    f"room:{r}",
                    COLORS[r],
                    size=13,
                )
        if self.overlay == "history":
            self._history(im, h)
        elif self.overlay:
            self._info(im, h)
        if self.selected_image:
            self._image_viewer(im)
        if h.paused:
            self._veil(im, 185)
            self.hits = []
            d = draw(im)
            hot = self.hover == "resume"
            radius = 74 if hot else 67
            d.ellipse(
                (720 - radius, 430 - radius, 720 + radius, 430 + radius),
                fill="#ddede1",
                outline="#90bdab",
                width=3,
            )
            if hot:
                d.polygon(((704, 400), (704, 460), (748, 430)), fill="#284d43")
            else:
                d.rectangle((697, 402, 710, 457), fill="#284d43")
                d.rectangle((730, 402, 743, 457), fill="#284d43")
            text(d, (720, 529), self.t("paused"), 29, INK, True, "ma")
            text(d, (720, 578), self.t("resume_hint"), 14, MUTED, anchor="ma")
            self.hits.append(((620, 330, 820, 515), "resume"))
        return im.image

    def scroll(self, delta, x):
        if self.selected_image:
            return
        if self.overlay == "help":
            self.help_scroll = max(
                0, min(self.scroll_limits["help"], self.help_scroll - delta * 65)
            )
        elif self.overlay == "history":
            if x < 525:
                self.record_offset = max(
                    0, min(self.scroll_limits["records"], self.record_offset - delta)
                )
            else:
                self.detail_scroll = max(
                    0,
                    min(self.scroll_limits["detail"], self.detail_scroll + delta * 65),
                )
        elif self.zoom is not None:
            self.chat_scroll = max(
                0, min(self.scroll_limits["chat"], self.chat_scroll + delta * 65)
            )

    def activate(self, action, h):
        if not action:
            return
        if h.paused and action not in ("pause", "resume"):
            return
        if action in ("pause", "resume"):
            h.paused = not h.paused
        elif action == "menu":
            self.menu = not self.menu
        elif action.startswith("room:"):
            room = int(action.split(":")[1])
            if 0 <= room < 3:
                self.zoom = room
                self.menu = False
        elif action.startswith("speed:"):
            speed = float(action.split(":")[1])
            if speed in (0.5, 1, 2, 4):
                h.speed = speed
        elif action == "language":
            self.language = "en" if self.language == "ko" else "ko"
            self.toast = ""
        elif action == "home":
            self.zoom = None
            self.menu = False
        elif action in ("history", "help"):
            self.overlay = action
            self.menu = False
        elif action == "close":
            if self.selected_image:
                self.selected_image = None
            else:
                self.overlay = None
        elif action == "image_close":
            self.selected_image = None
        elif action.startswith("image:"):
            index = int(action.split(":")[1])
            if 0 <= index < len(self.image_targets):
                self.selected_image = self.image_targets[index]
        elif action.startswith("filter:"):
            self.filter = action.split(":")[1]
            self.record_offset = 0
            self.detail_scroll = 0
        elif action.startswith("record:"):
            uid = int(action.split(":")[1])
            if any((v.uid == uid and (not v.encounter.npc) for v in h.records)):
                self.selected = uid
                self.detail_scroll = 0
        elif action == "latest":
            self.chat_scroll = 0
