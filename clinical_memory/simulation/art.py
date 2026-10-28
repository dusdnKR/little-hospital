"""Code-drawn pixel furniture and sprites; no external image assets required."""

from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
import math
from PIL import Image, ImageDraw, ImageFont
from .engine import ROOMS, LAB, BENCHES, COLORS, doctor_home, nurse_home, seat
from .identity import doctor_label
from .i18n import tr

BG = "#111c28"
INK = "#e5edf4"
MUTED = "#9aafc1"
PATIENT_COLORS = ("#897dbc", "#d68965", "#759ba0", "#6f94c5", "#c184a3", "#d3b362")


@dataclass(frozen=True)
class WorldLayer:
    depth: float
    image: Image.Image
    position: tuple = (0, 0)
    labels: tuple = ()


def painted_layer(depth, painter, labels=()):
    """Keep only occupied pixels; labels retain world coordinates and depth."""
    canvas = Image.new("RGBA", (1200, 700))
    painter(canvas)
    box = canvas.getbbox()
    if box is None:
        return WorldLayer(depth, Image.new("RGBA", (1, 1)), labels=tuple(labels))
    return WorldLayer(depth, canvas.crop(box), box[:2], tuple(labels))


@lru_cache(maxsize=64)
def font(size=14, bold=False):
    choices = (
        ["C:/Windows/Fonts/malgunbd.ttf", "C:/Windows/Fonts/malgun.ttf"]
        if bold
        else ["C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/segoeui.ttf"]
    )
    choices += [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in choices:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def text(draw, xy, value, size=14, fill=INK, bold=False, anchor=None):
    if hasattr(draw, "draw_text"):
        draw.draw_text(xy, str(value), size, fill, bold, anchor)
        return
    draw.text(xy, str(value), font=font(size, bold), fill=fill, anchor=anchor)


@lru_cache(maxsize=512)
def sprite(kind, index=0, color="#548bd5", happy=False, child=False):
    """24×34 original sprite. Faces, hair, glasses and clothing vary by identity."""
    im = Image.new("RGBA", (24, 36))
    d = ImageDraw.Draw(im)
    skin = ("#edc4a0", "#d5a17d", "#a76f52", "#f1cbae")[index % 4]
    hair = ("#3b2c2a", "#584235", "#a46d40", "#252a38")[index % 4]
    if kind == "nurse":
        hair, skin = ("#302621", "#edc4a0")
    d.rectangle((6, 31, 10, 34), fill="#242b35")
    d.rectangle((14, 31, 18, 34), fill="#242b35")
    d.rectangle((7, 24, 10, 31), fill=skin if kind == "nurse" else "#394a62")
    d.rectangle((14, 24, 17, 31), fill=skin if kind == "nurse" else "#394a62")
    cloth = color if kind in ("nurse", "patient") else "#f4f6ef"
    d.rectangle((5, 15, 19, 26), fill=cloth)
    d.rectangle((3, 17, 5, 26), fill=cloth)
    d.rectangle((19, 17, 21, 26), fill=cloth)
    d.rectangle((3, 25, 5, 28), fill=skin)
    d.rectangle((19, 25, 21, 28), fill=skin)
    if kind == "nurse":
        d.polygon([(6, 18), (18, 18), (21, 29), (3, 29)], fill=color)
        for by in (18, 22, 26):
            d.rectangle(
                (11, by, 12, by + 1),
                fill="#4b6269" if color == "#f6f5eb" else "#fff8e8",
            )
    if kind == "doctor":
        d.rectangle((11, 16, 14, 27), fill=color)
        d.rectangle((6, 21, 9, 24), fill=color)
        d.line((8, 16, 8, 21, 16, 21, 16, 16), fill="#4b6774", width=1)
    d.rectangle((6, 5, 18, 14), fill=skin)
    d.rectangle((5, 4, 19, 8), fill=hair)
    d.rectangle((5, 7, 7, 11), fill=hair)
    if kind == "nurse":
        d.rectangle((17, 5, 19, 10), fill=hair)
        d.rectangle((17, 1, 21, 5), fill=hair)
        d.rectangle((19, 2, 22, 4), fill=hair)
        d.rectangle((7, 3, 16, 7), fill="#fffdf3")
        d.rectangle((6, 6, 17, 7), fill="#c4dfd9")
        d.rectangle((11, 3, 12, 6), fill="#568f89")
        d.rectangle((10, 4, 13, 5), fill="#568f89")
    elif index in (2, 4):
        d.rectangle((17, 6, 20, 16), fill=hair)
    elif index % 3 == 0:
        d.rectangle((7, 2, 13, 5), fill=hair)
    elif index % 3 == 1:
        d.rectangle((13, 7, 18, 9), fill=hair)
    d.point((9, 10), fill="#282d36")
    d.point((15, 10), fill="#282d36")
    if kind != "nurse" and index % 3 == 0:
        d.rectangle((7, 9, 11, 11), outline="#30404b")
        d.rectangle((13, 9, 17, 11), outline="#30404b")
        d.line((11, 10, 13, 10), fill="#30404b")
    if happy or kind in ("doctor", "nurse"):
        d.line((10, 12, 11, 13, 14, 13, 15, 12), fill="#8f5545")
    else:
        d.line((10, 14, 11, 13, 14, 13, 15, 14), fill="#8f5545")
        d.rectangle((18, 9, 19, 11), fill="#82c6dc")
    if child:
        im = im.resize((18, 27), Image.Resampling.NEAREST)
    return im


def character(
    image,
    pos,
    kind,
    index=0,
    color="#548bd5",
    happy=False,
    child=False,
    scale=1.5,
    walking=False,
    time=0,
    shadow=True,
):
    s = sprite(kind, index, color, happy, child)
    s = s.resize(
        (int(s.width * scale), int(s.height * scale)), Image.Resampling.NEAREST
    )
    x, y = (int(pos[0]), int(pos[1]))
    bob = int(time * 7) % 2 * 2 if walking else 0
    if shadow:
        ImageDraw.Draw(image).ellipse((x - 17, y - 5, x + 17, y + 3), fill="#637e7b")
    image.alpha_composite(s, (x - s.width // 2, y - s.height - bob))


def plant(d, x, y):
    d.rectangle((x - 8, y - 10, x + 9, y + 6), fill="#b97b58")
    d.rectangle((x - 11, y - 13, x + 12, y - 7), fill="#deaa7d")
    d.rectangle((x - 3, y - 36, x + 3, y - 12), fill="#487b5c")
    d.rectangle((x - 16, y - 31, x - 1, y - 17), fill="#4e946c")
    d.rectangle((x + 1, y - 38, x + 15, y - 22), fill="#72af79")


def chair(d, x, y, color="#719a9a"):
    d.rectangle((x - 19, y - 12, x + 19, y + 8), fill="#476d73")
    d.rectangle((x - 19, y - 19, x + 19, y - 8), fill=color)
    d.rectangle((x - 14, y + 8, x - 10, y + 17), fill="#4d6265")
    d.rectangle((x + 10, y + 8, x + 14, y + 17), fill="#4d6265")


@lru_cache(maxsize=2)
def background(night=False):
    im = Image.new("RGBA", (1200, 700), "#253d49" if night else "#91b8a2")
    d = ImageDraw.Draw(im)
    for i in range(65):
        x, y = ((i * 83 + 19) % 1200, (i * 109 + 25) % 700)
        if x < 82 or x > 1138 or y < 30:
            d.rectangle((x, y, x + 5, y + 3), fill="#456852" if night else "#6e9f7f")
    for x, y in [(38, 150), (1155, 180), (44, 470), (1162, 530)]:
        d.rectangle((x - 5, y, x + 5, y + 32), fill="#6b5943")
        d.rectangle(
            (x - 28, y - 37, x + 28, y + 7), fill="#2c5950" if night else "#54896a"
        )
        d.rectangle(
            (x - 20, y - 48, x + 20, y - 30), fill="#376a57" if night else "#76a77a"
        )
    d.rectangle((91, 49, 1137, 664), fill="#577675")
    d.rectangle((82, 40, 1126, 652), fill="#8ba7a1")
    d.rectangle((88, 46, 1120, 646), fill="#e0e7d9")
    for x in range(90, 1120, 28):
        d.line((x, 48, x, 646), fill="#d4dece")
    for y in range(48, 648, 28):
        d.line((90, y, 1120, y), fill="#d4dece")
    d.rectangle((96, 285, 1097, 387), fill="#cfddd4")
    d.line((98, 320, 1098, 320), fill="#bed1c7", width=2)
    for i, box in enumerate((*ROOMS, LAB)):
        x, y, right, bottom = box
        color = COLORS[i] if i < 5 else "#8b82b6"
        d.rectangle(box, fill="#6c898a")
        d.rectangle((x + 6, y + 6, right - 6, bottom - 6), fill="#edf0e4")
        for tx in range(x + 8, right - 6, 24):
            for ty in range(y + 8, bottom - 6, 24):
                d.rectangle((tx, ty, tx + 21, ty + 21), fill="#e3e8da")
        d.rectangle((x, y, right, y + 28), fill=color)
        if i < 3:
            d.rectangle((x + 111, bottom - 7, x + 165, bottom + 1), fill="#cfddd4")
        else:
            d.rectangle((x + 111, y - 1, x + 165, y + 7), fill="#cfddd4")
        if i < 5:
            d.rectangle(
                (x + 23, y + 39, x + 66, y + 71),
                fill="#9ec2cc" if not night else "#3b5c78",
            )
            d.line((x + 44, y + 40, x + 44, y + 70), fill="#f7f5e7", width=3)
    d.rectangle((550, 641, 654, 653), fill="#bfd7d2")
    d.rectangle((562, 650, 642, 697), fill="#c2c6b1" if not night else "#536c70")
    exterior_sign(im, night)
    return im


def desk_top(room):
    return ROOMS[room][1] + (85 if room < 3 else 76)


def seated_doctor(room):
    return (doctor_home(room)[0], desk_top(room) + 18)


@lru_cache(maxsize=10)
def clinic_desk(room, exam_ready=False):
    x = ROOMS[room][0]
    y = desk_top(room)

    def paint(im):
        d = ImageDraw.Draw(im)
        d.rectangle((x + 72, y + 7, x + 172, y + 35), fill="#8e7357")
        d.rectangle((x + 68, y, x + 176, y + 27), fill="#c7a676")
        d.rectangle((x + 145, y - 17, x + 174, y + 3), fill="#405867")
        d.rectangle(
            (x + 149, y - 14, x + 170, y - 2),
            fill="#c0edb3" if exam_ready else "#8fd6c8",
        )
        d.rectangle((x + 152, y + 5, x + 171, y + 8), fill="#d5dad3")
        d.rectangle((x + 80, y + 4, x + 100, y + 17), fill="#f1efdf")

    return painted_layer(y + 35, paint)


@lru_cache(maxsize=1)
def fixed_furniture():
    layers = []
    for r, (x, y, right, bottom) in enumerate(ROOMS):
        layers.append(
            painted_layer(
                y + 62, lambda im: chair(ImageDraw.Draw(im), x + 115, y + 62, COLORS[r])
            )
        )
        layers.append(
            painted_layer(
                seat(r)[1] - 1,
                lambda im: chair(ImageDraw.Draw(im), x + 115, y + 151, COLORS[r]),
            )
        )

        def cabinet(im):
            d = ImageDraw.Draw(im)
            d.rectangle(
                (right - 59, bottom - 58, right - 21, bottom - 14), fill="#b0c6b9"
            )
            d.rectangle(
                (right - 55, bottom - 54, right - 25, bottom - 35),
                outline="#77958c",
                width=2,
            )

        layers.append(painted_layer(bottom - 14, cabinet))
        layers.append(
            painted_layer(
                y + 77, lambda im: plant(ImageDraw.Draw(im), right - 39, y + 71)
            )
        )
    for x in (218, 448, 678):

        def bench(im):
            d = ImageDraw.Draw(im)
            d.rectangle((x - 4, 327, x + 146, 342), fill="#a7ae8d")
            for sx in range(x, x + 150, 50):
                d.rectangle((sx, 345, sx + 42, 357), fill="#b5bda0")

        layers.append(painted_layer(349, bench))
        layers.append(
            painted_layer(
                365,
                lambda im: ImageDraw.Draw(im).rectangle(
                    (x - 4, 358, x + 146, 365), fill="#7d8570"
                ),
            )
        )
    for x in (400, 836):
        layers.append(painted_layer(626, lambda im: plant(ImageDraw.Draw(im), x, 620)))

    def exam_bed(im):
        x, y, _, _ = LAB
        d = ImageDraw.Draw(im)
        d.rectangle((x + 54, y + 52, x + 230, y + 135), fill="#bac3d0")
        d.rectangle((x + 64, y + 61, x + 219, y + 128), fill="#f1efe5")
        d.rectangle((x + 73, y + 69, x + 112, y + 119), fill="#dae3e6")

    layers.append(painted_layer(507, exam_bed))
    return tuple(layers)


@lru_cache(maxsize=6)
def reception_counter(x, key, language):

    def paint(im):
        d = ImageDraw.Draw(im)
        d.rectangle((x - 44, 599, x + 44, 615), fill="#9a795c")
        d.rectangle((x - 44, 590, x + 44, 603), fill="#c5a783")

    label = ((x, 592), tr(language, key), 9, "#584a3a", True, "ma")
    return painted_layer(615, paint, (label,))


@lru_cache(maxsize=4)
def exam_curtain(closed, language):
    x, y, _, _ = LAB

    def paint(im):
        d = ImageDraw.Draw(im)
        d.line((x + 38, y + 43, x + 252, y + 43), fill="#77808f", width=4)
        for cx in range(x + 40, x + 255, 14):
            if closed or cx < x + 82 or cx > x + 211:
                d.rectangle(
                    (cx, y + 46, cx + 13, y + 139),
                    fill="#8c91ae" if cx % 28 else "#a5a9c0",
                )

    labels = (
        (((936, 477), tr(language, "examining"), 15, "#f6f5ff", True, "ma"),)
        if closed
        else ()
    )
    return painted_layer(y + 139, paint, labels)


def exterior_sign(im, night):
    """Metal brackets mount the sign across the north exterior wall."""
    if night:
        glow = Image.new("RGBA", im.size)
        gd = ImageDraw.Draw(glow)
        for pad, alpha in ((25, 10), (17, 17), (9, 30)):
            gd.rectangle(
                (442 - pad, 10 - pad, 758 + pad, 65 + pad), fill=(255, 224, 139, alpha)
            )
        im.alpha_composite(glow)
    d = ImageDraw.Draw(im)
    for x in (467, 725):
        d.rectangle((x, 39, x + 7, 77), fill="#405d5d")
        d.rectangle((x - 3, 72, x + 11, 77), fill="#718c86")
    d.rectangle((449, 17, 765, 70), fill="#3c5759")
    d.rectangle((442, 9, 758, 64), fill="#587879")
    d.rectangle(
        (446, 13, 754, 60),
        fill="#284c51",
        outline="#f3dba5" if night else "#9bbdb0",
        width=3,
    )
    d.line((450, 60, 752, 60), fill="#fff3bd" if night else "#789b8e", width=3)
    cross = "#fff3bc" if night else "#c0dcd1"
    d.rectangle((462, 31, 486, 38), fill=cross)
    d.rectangle((470, 23, 478, 46), fill=cross)
    for x in (451, 747):
        d.rectangle((x, 18, x + 2, 20), fill="#c4c7a4")


def speech(d, pos, time, label=None):
    x, y = (int(pos[0]), int(pos[1] - 90))
    d.rounded_rectangle(
        (x - 23, y - 4, x + 23, y + 21),
        radius=5,
        fill="#fffef5",
        outline="#8c9a96",
        width=2,
    )
    d.polygon(((x - 3, y + 21), (x + 3, y + 21), (x, y + 27)), fill="#fffef5")
    text(
        d,
        (x, y - 4),
        label if label else "." * (int(time * 2.8) % 4),
        18,
        "#3e565d",
        True,
        "ma",
    )


def patient_bust(visit):
    return sprite(
        "patient",
        visit.uid,
        PATIENT_COLORS[visit.uid % len(PATIENT_COLORS)],
        visit.resolved and visit.encounter.success,
    ).crop((2, 0, 23, 26))


def call_tile(im, visit, xy, color="#34545a", compact=False):
    """Show the patient's upper-body sprite beside the assigned room number."""
    x, y = xy
    bust = patient_bust(visit).resize(
        (18, 23) if compact else (28, 35), Image.Resampling.NEAREST
    )
    im.alpha_composite(bust, (int(x), int(y)))
    text(
        ImageDraw.Draw(im),
        (x + 48, y + (11 if compact else 16)),
        f"{visit.room + 1:02}",
        20 if compact else 23,
        color,
        True,
        "mm",
    )


def wall_tv(hospital):
    """A compact landscape television with upright, readable assignment cards."""
    calls = hospital.active_calls()
    tv = Image.new("RGBA", (96, 62))
    d = ImageDraw.Draw(tv)
    d.rectangle((0, 0, 95, 61), fill="#243138")
    d.rectangle((1, 1, 94, 60), outline="#5b6b70", width=1)
    d.rectangle((4, 4, 91, 55), fill="#18383c" if calls else "#142126")
    if calls:
        d.line((5, 5, 90, 5), fill="#496567")
    compact = len(calls) > 1
    if compact:
        d.line((11, 30, 84, 30), fill="#365154")
    for index, visit in enumerate(calls):
        card = Image.new("RGBA", (74, 24 if compact else 39))
        call_tile(card, visit, (0, 0), color="#e8f5df", compact=compact)
        pos = (11, 6 + index * 25) if compact else (11, 11)
        tv.alpha_composite(card, pos)
    d.rectangle((86, 58, 88, 59), fill="#b7dbb0" if calls else "#5d6a6e")
    d.line((41, 58, 54, 58), fill="#65777b")
    return tv


def draw_wall_tv(im, hospital):
    d = ImageDraw.Draw(im)
    d.rectangle((88, 322, 95, 350), fill="#5b7273")
    d.rectangle((92, 329, 102, 343), fill="#344b50")
    d.rectangle((99, 309, 194, 370), fill="#9ab1aa")
    im.alpha_composite(wall_tv(hospital), (95, 305))


def draw_calls(im, hospital):
    calls = hospital.active_calls()
    d = ImageDraw.Draw(im)
    for v in calls:
        x, y = (158 + v.desk * 108, 489)
        d.rounded_rectangle(
            (x - 45, y, x + 45, y + 46),
            radius=5,
            fill="#fffef5",
            outline="#819994",
            width=2,
        )
        d.polygon(((x - 4, y + 46), (x + 4, y + 46), (x, y + 53)), fill="#fffef5")
        call_tile(im, v, (x - 33, y + 5))


def draw_coin_stack(im, count):
    """Stack one coin per payment on the left end of the cashier's counter."""
    d = ImageDraw.Draw(im)
    x, base = (946, 594)
    for level in range(count):
        y = base - level * 6
        offset = (-2, 1, -1, 2, 0)[level % 5]
        d.rectangle(
            (x - 12 + offset, y - 5, x + 11 + offset, y),
            fill="#dca23a",
            outline="#9c6525",
        )
        d.line((x - 11 + offset, y - 4, x + 10 + offset, y - 4), fill="#ffe28a")


def actor_layer(
    pos,
    kind,
    index=0,
    color="#548bd5",
    happy=False,
    child=False,
    walking=False,
    time=0,
    depth=None,
    seated=False,
):
    canvas = Image.new("RGBA", (40, 60))
    character(
        canvas,
        (20, 56),
        kind,
        index,
        color,
        happy,
        child,
        walking=walking,
        time=time,
        shadow=not seated,
    )
    return WorldLayer(
        pos[1] if depth is None else depth, canvas, (int(pos[0]) - 20, int(pos[1]) - 56)
    )


def patient_pose(visit):
    x, y = visit.patient
    if visit.room is not None and visit.patient == seat(visit.room):
        return ((x, y + 26), True)
    if visit.patient in BENCHES:
        return ((x, y + 12), True)
    return (visit.patient, False)


def world_layers(hospital, language="ko"):
    """Order surfaces and people by ground depth, not by their visual height."""
    floor = Image.blend(background(True), background(False), hospital.daylight)
    layers = [
        WorldLayer(-math.inf, floor, labels=tuple(world_labels(hospital, language)))
    ]
    layers.extend(fixed_furniture())
    for r, v in enumerate(hospital.rooms):
        layers.append(
            clinic_desk(r, bool(v and v.messages and (v.messages[-1].kind == "exam")))
        )
    for x, key in ((158, "reception_1"), (266, "reception_2"), (976, "payment")):
        layers.append(reception_counter(x, key, language))
        layers.append(actor_layer((x, 604), "nurse", color="#f6f5eb"))
    layers.append(painted_layer(370, lambda im: draw_wall_tv(im, hospital)))
    layers.append(painted_layer(615, lambda im: draw_coin_stack(im, hospital.coins)))
    closed = bool(hospital.exam_owner and hospital.exam_owner.state == "검사 중")
    layers.append(exam_curtain(closed, language))
    moving_nurses = {
        v.room
        for v in hospital.visits
        if v.room is not None and (not v.done) and (hospital.rooms[v.room] is v)
    }
    for r in range(5):
        layers.append(
            actor_layer(
                seated_doctor(r),
                "doctor",
                r,
                COLORS[r],
                True,
                depth=desk_top(r) - 1,
                seated=True,
            )
        )
        if r not in moving_nurses:
            layers.append(
                actor_layer(nurse_home(r), "nurse", color=COLORS[r], happy=True)
            )
    for v in hospital.visits:
        if v.done:
            continue
        walk = v.current is not None and v.current.kind == "move"
        if v.room is not None and hospital.rooms[v.room] is v:
            layers.append(
                actor_layer(
                    v.nurse,
                    "nurse",
                    color=COLORS[v.room],
                    happy=True,
                    walking=walk,
                    time=hospital.time,
                )
            )
        if v.state == "영혼":

            def ghost(im):
                (x, y), _ = patient_pose(v)
                y -= 30 + v.ghost * 60
                gd = ImageDraw.Draw(im)
                alpha = int(255 * (1 - v.ghost))
                gd.ellipse((x - 17, y - 33, x + 17, y + 1), fill=(239, 247, 255, alpha))
                gd.polygon(
                    (
                        (x - 17, y - 14),
                        (x - 17, y + 12),
                        (x - 8, y + 5),
                        (x, y + 12),
                        (x + 8, y + 5),
                        (x + 17, y + 12),
                        (x + 17, y - 14),
                    ),
                    fill=(239, 247, 255, alpha),
                )
                gd.rectangle((x - 8, y - 20, x - 5, y - 14), fill=(67, 89, 115, alpha))
                gd.rectangle((x + 5, y - 20, x + 8, y - 14), fill=(67, 89, 115, alpha))

            layers.append(painted_layer(1000001, ghost))
        else:
            pos, seated = patient_pose(v)
            layers.append(
                actor_layer(
                    pos,
                    "patient",
                    v.uid,
                    PATIENT_COLORS[v.uid % 6],
                    v.resolved,
                    v.encounter.child,
                    walk,
                    hospital.time,
                    depth=v.patient[1],
                    seated=seated,
                )
            )
        if v.encounter.guardian:
            layers.append(
                actor_layer(
                    (v.patient[0] + 31, v.patient[1] + 7),
                    "patient",
                    v.uid + 4,
                    "#79939b",
                    v.resolved and v.encounter.success,
                    False,
                    walk,
                    hospital.time,
                )
            )

    def bubbles(im):
        d = ImageDraw.Draw(im)
        for r, v in enumerate(hospital.rooms):
            if v is None or not v.speaker:
                continue
            pos = seated_doctor(r) if v.speaker == "doctor" else patient_pose(v)[0]
            if v.speaker == "guardian":
                pos = (pos[0] + 31, pos[1] + 7)
            speech(d, pos, hospital.time)
        draw_calls(im, hospital)

    layers.append(painted_layer(1000000, bubbles))
    return sorted(layers, key=lambda layer: layer.depth)


def render_world(hospital, language="ko"):
    """World-resolution rendering for image checks; UI draws the same scene at native DPI."""
    im = Image.new("RGBA", (1200, 700))
    for layer in world_layers(hospital, language):
        im.alpha_composite(layer.image, layer.position)
        for xy, value, size, fill, bold, anchor in layer.labels:
            text(ImageDraw.Draw(im), xy, value, size, fill, bold, anchor)
    return im


def world_labels(hospital, language):
    """Signs and floor markings belong behind people and raised furniture."""
    night = 1 - hospital.daylight
    sign_color = tuple(
        (
            round(day * (1 - night) + dark * night)
            for day, dark in zip((215, 235, 225), (255, 244, 190))
        )
    )
    labels = [((616, 35), "LITTLE HOSPITAL", 20, sign_color, True, "mm")]
    for i, (x, y, _, _) in enumerate((*ROOMS, LAB)):
        label = (
            f"0{i + 1}  {doctor_label(i)}"
            if i < 5
            else "06  " + tr(language, "exam_room")
        )
        labels.append(((x + 13, y + 5), label, 12, "#ffffff", True, None))
    labels.append(((936, 541), tr(language, "exam_hint"), 11, "#5d657f", False, "ma"))
    return labels
