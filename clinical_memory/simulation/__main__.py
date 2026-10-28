"""Run with python -m clinical_memory.simulation. Tk owns the window; Pillow paints pixels."""

import argparse
import csv
import json
from pathlib import Path
import time
from .engine import Hospital, DEFAULT_SPEED
from .replay import load_replay, ROOT
from .ui import Screen, WIDTH, HEIGHT


def main():
    parser = argparse.ArgumentParser(description="Clinical Memory encounter playback")
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON configuration with recorded encounters per model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        choices=(42, 43, 44),
        default=42,
        help="Visual simulation seed; recorded outcomes stay unchanged",
    )
    parser.add_argument(
        "--speed", type=float, choices=(0.5, 1, 2, 4), default=DEFAULT_SPEED
    )
    parser.add_argument("--language", choices=("ko", "en"), default="en")
    parser.add_argument(
        "--render-scale", type=float, default=1, help="PNG resolution scale"
    )
    parser.add_argument(
        "--screenshot", type=Path, help="Save a PNG without opening a desktop window"
    )
    parser.add_argument(
        "--advance", type=float, default=0, help="Seconds of playback before rendering"
    )
    parser.add_argument("--room", type=int, choices=range(1, 4))
    parser.add_argument(
        "--smoke-seconds",
        type=float,
        help="Close the desktop window after this many seconds",
    )
    args = parser.parse_args()
    try:
        replay = load_replay(args.config)
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        parser.error(f"Cannot read replay configuration: {exc}")
    hospital = Hospital(replay, args.seed)
    hospital.speed = args.speed
    screen = Screen(args.language)
    if args.room:
        screen.zoom = args.room - 1
    if args.advance:
        hospital.tick(args.advance)
    if args.screenshot:
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        screen.render(hospital, args.render_scale).convert("RGB").save(args.screenshot)
        print(
            json.dumps(
                {
                    "records_loaded": replay.count,
                    "warnings": replay.warnings,
                    "screenshot": str(args.screenshot),
                },
                ensure_ascii=False,
            )
        )
        return
    import tkinter as tk
    from tkinter import filedialog, messagebox
    from PIL import ImageTk
    import sys

    if sys.platform == "win32":
        import ctypes

        try:
            ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()
    root = tk.Tk()
    root.title("Little Hospital")
    root.configure(bg="#111c28")
    scale = min(
        1.0,
        (root.winfo_screenwidth() - 100) / WIDTH,
        (root.winfo_screenheight() - 100) / HEIGHT,
    )
    root.geometry(f"{int(WIDTH * scale)}x{int(HEIGHT * scale)}")
    root.minsize(960, 640)
    canvas = tk.Canvas(root, bg="#111c28", highlightthickness=0)
    canvas.pack(fill="both", expand=True)
    canvas_image = canvas.create_image(0, 0, anchor="nw")
    state = {
        "scale": 1.0,
        "offset": (0, 0),
        "last": time.perf_counter(),
        "image": None,
        "closing": False,
    }

    def point(event):
        ox, oy = state["offset"]
        return ((event.x - ox) / state["scale"], (event.y - oy) / state["scale"])

    def export(kind):
        suffix = ".csv" if kind == "export_csv" else ".json"
        output = ROOT / "outputs/hospital"
        output.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            parent=root,
            initialdir=output,
            initialfile="hospital_records" + suffix,
            defaultextension=suffix,
            filetypes=[(screen.t("file_type"), "*" + suffix)],
        )
        state["last"] = time.perf_counter()
        if not path:
            return
        try:
            records = hospital.export_records()
            if suffix == ".json":
                Path(path).write_text(
                    json.dumps(
                        {
                            "schema": 1,
                            "simulation_seed": hospital.seed,
                            "metric": "exact-match mapped to game recovery/death",
                            "records": records,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            else:
                keys = [
                    "visit",
                    "patient_name",
                    "case_id",
                    "model",
                    "room",
                    "simulation_started",
                    "simulation_finished",
                    "success",
                    "paid",
                    "phase",
                    "source",
                ]
                with open(path, "w", encoding="utf-8-sig", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                    writer.writeheader()
                    for record in records:
                        writer.writerow(
                            {
                                k: "'" + v
                                if isinstance(v, str)
                                and v.startswith(("=", "+", "-", "@"))
                                else v
                                for k, v in record.items()
                            }
                        )
            screen.toast = screen.t("saved", count=len(records))
        except OSError as exc:
            messagebox.showerror(screen.t("export_error"), str(exc), parent=root)

    def activate(action):
        nonlocal hospital, screen
        if hospital.paused and action not in ("pause", "resume"):
            return
        if action == "restart":
            language = screen.language
            hospital = Hospital(replay, args.seed)
            screen = Screen(language)
        elif action in ("export", "export_csv"):
            export(action)
        else:
            screen.activate(action, hospital)

    def click(event):
        activate(screen.hit(point(event)))

    def motion(event):
        screen.hover = screen.hit(point(event))
        canvas.configure(cursor="hand2" if screen.hover else "")

    def wheel(event):
        if hospital.paused:
            return
        delta = (
            (1 if event.delta > 0 else -1)
            if getattr(event, "delta", 0)
            else 1
            if event.num == 4
            else -1
        )
        screen.scroll(delta, point(event)[0])

    def key(event):
        if event.keysym == "space":
            activate("pause")
        elif event.keysym == "Escape":
            activate("close" if screen.overlay or screen.selected_image else "home")
        elif (
            event.char in "123"
            and event.char
            and (not screen.overlay)
            and (not screen.selected_image)
        ):
            activate("room:" + str(int(event.char) - 1))

    def close():
        state["closing"] = True
        root.destroy()

    def frame():
        if state["closing"]:
            return
        now = time.perf_counter()
        hospital.tick(min(0.15, now - state["last"]))
        state["last"] = now
        w, hh = (max(1, canvas.winfo_width()), max(1, canvas.winfo_height()))
        factor = min(w / WIDTH, hh / HEIGHT)
        image = screen.render(hospital, factor)
        size = image.size
        state["scale"] = factor
        state["offset"] = ((w - size[0]) // 2, (hh - size[1]) // 2)
        state["image"] = ImageTk.PhotoImage(image)
        canvas.itemconfigure(canvas_image, image=state["image"])
        canvas.coords(canvas_image, *state["offset"])
        root.after(40, frame)

    canvas.bind("<Button-1>", click)
    canvas.bind("<Motion>", motion)
    canvas.bind("<MouseWheel>", wheel)
    canvas.bind("<Button-4>", wheel)
    canvas.bind("<Button-5>", wheel)
    root.bind("<Key>", key)
    root.protocol("WM_DELETE_WINDOW", close)
    if replay.warnings or not replay.count:
        screen.overlay = "help"
    if args.smoke_seconds:
        root.after(int(args.smoke_seconds * 1000), close)
    root.after(20, frame)
    root.mainloop()


if __name__ == "__main__":
    main()
