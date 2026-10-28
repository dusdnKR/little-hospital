# Simulation and encounter playback

```bash
python -m clinical_memory simulate --speed 4 --language en
```

The desktop simulation replays the final recorded encounters specified in `configs/simulation.json`. It does not load model weights or generate new diagnoses. The default bundle includes the complete Test 1 cohort for 2B, 4B and 9B with `key_dx`, random order and seed 42. Patient display names are fictional. Background NPCs are decorative and excluded from model outcome counts.

Click a room to inspect dialogue and returned examination images. Space pauses/resumes; Escape returns from a room view. Use the interface speed/language controls and the mouse wheel to scroll long transcripts. Export buttons save playback records as JSON or CSV. `--seed` changes the visual simulation, not the stored clinical outcomes.

The game maps a correct benchmark answer to recovery and an incorrect answer to death. These animations are a visualization of exact-match accuracy, **not observed patient outcomes**.

## Headless rendering and validation

```bash
python -m clinical_memory simulate --screenshot results/hospital.png --advance 60
python -m clinical_memory simulate --screenshot results/room.png --advance 60 --room 1
python -m clinical_memory validate-simulation --output results/simulation_validation.json
python -m clinical_memory analyze-replay --output results/replay_summary.json
```

Headless commands need Pillow but no desktop display or Tk. `--advance` skips playback seconds at the selected speed (for example, `--advance 15 --speed 4` advances 60 simulation seconds). Validation drains every queue, checks that each encounter is replayed once with its full transcript, rejects missing or unreadable images, verifies outcome accounting, and confirms source files are unchanged.

## Export a GIF

```bash
python -m pip install -r requirements-demo.txt
python -m clinical_memory record-simulation --output results/simulation.gif --speed 4 --seconds 30 --fps 10 --width 1080
```

The exporter uses the same renderer as the desktop window and a bundled FFmpeg binary. It records 30 seconds of playback at 4× speed (120 simulation seconds), generates a shared palette, and writes a looping GIF. Temporary frames are removed automatically. Add `--room 1` for a conversation view, `--advance 15` to skip ahead, or `--config configs/my_simulation.json` to record another evaluation. The README demo uses these defaults.

## Use another completed evaluation

Create a JSON config with model labels and paths relative to that config:

```json
{
  "image_roots": ["../data/images"],
  "sources": {
    "2B": "../runs/experiments/cm_test1_2b_key_dx_random_s42_t302",
    "4B": "../runs/experiments/cm_test1_4b_key_dx_random_s42_t302",
    "9B": "../runs/experiments/cm_test1_9b_key_dx_random_s42_t302"
  }
}
```

```bash
python -m clinical_memory simulate --config configs/my_simulation.json
python -m clinical_memory validate-simulation --config configs/my_simulation.json
```

The loader accepts a run directory or its `episodes/` directory, checks the doctor label against the seal when present, and resolves only local images within configured roots. It never parses raw private-call logs. A missing image is shown as missing rather than replaced with an invented clinical image. Use the default licensed bundle for a self-contained demonstration.
