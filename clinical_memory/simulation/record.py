"""Export the simulation renderer to a looping GIF without a desktop window."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import tempfile

from .engine import Hospital
from .replay import load_replay
from .ui import Screen, WIDTH, HEIGHT


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Expected a finite positive number")
    return number


def record(output, *, config=None, seconds=30, fps=10, width=1080,
           speed=4, advance=0, seed=42, language="en", room=None):
    import imageio_ffmpeg

    output = Path(output)
    if output.suffix.lower() != ".gif":
        raise ValueError("Output must have a .gif extension")
    replay = load_replay(config)
    if replay.warnings or not replay.count:
        raise ValueError("A complete replay is required: " + "; ".join(replay.warnings))
    hospital = Hospital(replay, seed)
    hospital.speed = speed
    hospital.tick(advance)
    screen = Screen(language)
    if room is not None:
        screen.zoom = room - 1
    frames = math.ceil(seconds * fps)
    height = round(HEIGHT * width / WIDTH)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="little-hospital-") as directory:
        folder = Path(directory)
        video, palette = folder / "frames.mkv", folder / "palette.png"
        with subprocess.Popen(
            command + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                       f"{width}x{height}", "-r", str(fps), "-i", "-",
                       "-an", "-c:v", "ffv1", str(video)],
            stdin=subprocess.PIPE,
        ) as encoder:
            try:
                for index in range(frames):
                    if index:
                        hospital.tick(1 / fps)
                    frame = screen.render(hospital, width / WIDTH).convert("RGB")
                    encoder.stdin.write(frame.tobytes())
            finally:
                encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError("FFmpeg could not encode the simulation frames")
        subprocess.run(
            command + ["-i", str(video), "-vf",
                       "palettegen=stats_mode=diff:max_colors=192",
                       "-update", "1", "-frames:v", "1", str(palette)],
            check=True,
        )
        subprocess.run(
            command + ["-i", str(video), "-i", str(palette), "-lavfi",
                       "[0:v][1:v]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle",
                       "-loop", "0", str(output)],
            check=True,
        )
    return {"output": str(output), "frames": frames, "fps": fps,
            "duration_seconds": frames / fps, "speed": speed,
            "size_bytes": output.stat().st_size, "records_loaded": replay.count}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/simulation.gif"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seconds", type=positive, default=30)
    parser.add_argument("--fps", type=int, choices=(5, 10, 20, 25), default=10)
    parser.add_argument("--width", type=int, choices=(720, 960, 1080, 1440), default=1080)
    parser.add_argument("--speed", type=float, choices=(0.5, 1, 2, 4), default=4)
    parser.add_argument("--advance", type=float, default=0,
                        help="Playback seconds to skip at the selected speed")
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--language", choices=("en", "ko"), default="en")
    parser.add_argument("--room", type=int, choices=range(1, 4))
    args = parser.parse_args(argv)
    if not math.isfinite(args.advance) or args.advance < 0:
        parser.error("--advance must be finite and nonnegative")
    try:
        result = record(**vars(args))
    except ImportError:
        parser.error("Install the GIF exporter: python -m pip install -r requirements-demo.txt")
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
