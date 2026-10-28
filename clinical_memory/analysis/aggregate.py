from itertools import combinations
from statistics import mean
from clinical_memory.environment.events import event_exams
from clinical_memory.analysis.paired import change, correct, interval, paired, saw_image

OUTCOMES = {
    "accuracy": correct,
    "image_reach": saw_image,
    "examination_reach": lambda r: any((event_exams(e) for e in r["trace"]["events"])),
    "no_exam_request": lambda r: r["metrics"]["exam_requests"] == 0,
}
COUNTS = {
    "wasted_actions": lambda r: r["metrics"]["wasted_actions"],
    "collection_actions": lambda r: r["metrics"]["collection_actions"],
}
POINT = {
    "accuracy": ("correct", 100),
    "image_reach": ("image_reached", 100),
    "examination_reach": ("examination_reached", 100),
    "no_exam_request": ("no_exam_request", 100),
    "wasted_actions": ("wasted_actions", 1),
    "collection_actions": ("collection_actions", 1),
}


def gains(before, after):
    """Per-case change of every outcome and count between two evaluations of the same cases."""
    out = {name: paired(before, after, test) for name, test in OUTCOMES.items()}
    first, last = (
        {r["trace"]["id"]: r for r in before},
        {r["trace"]["id"]: r for r in after},
    )
    if first.keys() != last.keys():
        raise ValueError("Paired evaluations cover different cases")
    out.update(
        {
            name: {cid: count(last[cid]) - count(first[cid]) for cid in sorted(first)}
            for name, count in COUNTS.items()
        }
    )
    return out


def pooled(vectors, name):
    cases = set(vectors[0])
    if any((set(v) != cases for v in vectors)):
        raise ValueError("Runs of one comparison cover different cases")
    per_case = {cid: mean((v[cid] for v in vectors)) for cid in sorted(cases)}
    if name in OUTCOMES:
        return change(per_case)
    return {
        "delta": round(mean(per_case.values()), 3),
        "bootstrap_95": [round(x / 100, 3) for x in interval(per_case.values())],
    }


def summarize(members, protocol, orders=None):
    """Members are (public run summary, records before training, records at the last checkpoint) of runs of one stage.
    `orders` are the orders that the stage runs: every curriculum for the main experiment, one for the development runs."""
    orders, seeds, ways = (
        orders or protocol["curricula"],
        sorted(protocol["training_seeds"]),
        protocol["learning"]["conditions"],
    )
    moved, curves = ({}, {})
    for item, before, after in members:
        key = (item["doctor"], item["learning"], item["curriculum"], item["seed"])
        if key in moved:
            raise ValueError("Duplicate run")
        moved[key], curves[key] = (gains(before, after), item["curve"])
    doctors, used = (
        [d for d in protocol["doctors"] if any((key[0] == d for key in moved))],
        orders,
    )
    names, cells, contrasts = ([*OUTCOMES, *COUNTS], [], [])
    for doctor in doctors:
        for way in ways:
            keys = [
                (doctor, way, c, s)
                for c in orders
                for s in seeds
                if (doctor, way, c, s) in moved
            ]
            if not keys:
                continue
            steps = [[p["step"] for p in curves[k]] for k in keys]
            if any((s != steps[0] for s in steps)):
                raise ValueError("Runs were evaluated at different checkpoints")
            cells.append(
                {
                    "doctor": doctor,
                    "learning": way,
                    "runs": len(keys),
                    "runs_expected": len(used) * len(seeds),
                    "complete": len(keys) == len(used) * len(seeds),
                    "curve": [
                        {
                            "step": step,
                            **{
                                name: mean(
                                    (
                                        scale * curves[k][i][field] / curves[k][i]["n"]
                                        for k in keys
                                    )
                                )
                                for name, (field, scale) in POINT.items()
                            },
                        }
                        for i, step in enumerate(steps[0])
                    ],
                    "metrics": {
                        name: pooled([moved[k][name] for k in keys], name)
                        for name in names
                    },
                    "run_gains_pp": [
                        100 * mean(moved[k]["accuracy"].values()) for k in keys
                    ],
                    "by_curriculum": {
                        c: {
                            "seeds": [k[3] for k in keys if k[2] == c],
                            **{
                                name: pooled(
                                    [moved[k][name] for k in keys if k[2] == c], name
                                )
                                for name in names
                            },
                        }
                        for c in orders
                        if any((k[2] == c for k in keys))
                    },
                }
            )

    def compare(kind, name, pairs):
        """Gain of the first run minus gain of the second, over the pairs of which both runs are finished."""
        done = [(a, b) for a, b in pairs if a in moved and b in moved]
        if not done:
            return
        metrics = {}
        for outcome in names:
            if any(
                (moved[a][outcome].keys() != moved[b][outcome].keys() for a, b in done)
            ):
                raise ValueError("Runs of one comparison cover different cases")
            metrics[outcome] = pooled(
                [
                    {
                        cid: moved[a][outcome][cid] - moved[b][outcome][cid]
                        for cid in moved[a][outcome]
                    }
                    for a, b in done
                ],
                outcome,
            )
        contrasts.append(
            {
                "kind": kind,
                **name,
                "pairs": len(done),
                "pairs_expected": len(pairs),
                "complete": len(done) == len(pairs),
                "metrics": metrics,
            }
        )

    for first, second in (("key_dx", "key"), ("key", "self")):
        if first in ways and second in ways:
            for group in [[d] for d in doctors] + (
                [doctors] if len(doctors) > 1 else []
            ):
                compare(
                    "learning",
                    {"first": first, "second": second, "doctors": group},
                    [
                        ((d, first, c, s), (d, second, c, s))
                        for d in group
                        for c in used
                        for s in seeds
                    ],
                )
    for first, second in combinations(used, 2):
        compare(
            "curriculum",
            {"first": second, "second": first, "doctors": doctors},
            [
                ((d, w, second, s), (d, w, first, s))
                for d in doctors
                for w in ways
                for s in seeds
            ],
        )
    for small, large in combinations(doctors, 2):
        compare(
            "size",
            {"first": large, "second": small, "doctors": [small, large]},
            [
                ((large, w, c, s), (small, w, c, s))
                for w in ways
                for c in used
                for s in seeds
            ],
        )
    return {
        "cells": cells,
        "contrasts": contrasts,
        "interval_scope": "Finished runs averaged within each case, then cases resampled; descriptive, conditional on these runs and seeds, not corrected for the number of comparisons.",
    }
