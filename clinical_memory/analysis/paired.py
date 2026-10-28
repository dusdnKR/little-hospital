from clinical_memory.data.inputs import load
import random
from statistics import mean


def journal(folder):
    return [load(p) for p in sorted((folder / "episodes").glob("*.json"))]


def correct(r):
    return bool(r["score"]["correct"])


def saw_image(r):
    return bool(r["metrics"]["final_answer_saw_image"])


def gain_auc(points, key):
    base, span = (points[0][key], points[-1]["step"])
    return (
        sum(
            (
                (b["step"] - a["step"]) * (a[key] - base + (b[key] - base)) / 2
                for a, b in zip(points, points[1:])
            )
        )
        / span
    )


def paired(before, after, test):
    """Per-case change of one yes/no outcome between two evaluations of the same cases."""
    first, last = (
        {r["trace"]["id"]: test(r) for r in before},
        {r["trace"]["id"]: test(r) for r in after},
    )
    if first.keys() != last.keys():
        raise ValueError("Paired evaluations cover different cases")
    return {cid: int(last[cid]) - int(first[cid]) for cid in sorted(first)}


def interval(deltas, draws=2000, seed=20261011):
    """Case-resampling 95% interval of the mean change in percentage points."""
    values, rng = (list(deltas), random.Random(seed))
    means = sorted(
        (100 * mean(rng.choices(values, k=len(values))) for _ in range(draws))
    )
    return [
        round(means[int(0.025 * draws)], 2),
        round(means[int(0.975 * draws) - 1], 2),
    ]


def change(deltas):
    return {
        "delta_pp": round(100 * mean(deltas.values()), 2),
        "improved": sum((v > 0 for v in deltas.values())),
        "worsened": sum((v < 0 for v in deltas.values())),
        "paired_bootstrap_95_pp": interval(deltas.values()),
    }
