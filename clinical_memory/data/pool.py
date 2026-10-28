import hashlib
import json
from clinical_memory.memory.diagnosis import likeness


def key(seed, purpose, cid):
    """A fixed pseudo-random position for one patient and one purpose; the same on every machine."""
    return hashlib.sha256(f"{purpose}-{seed}:{cid}".encode()).hexdigest()


def digest(division):
    return hashlib.sha256(
        json.dumps(division, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def share(part, whole):
    return int(part * whole + 0.5)


def divide(frozen, golds, levels, groups, protocol):
    """The division of the pool. `levels` is the prior difficulty (times right of three) of every patient and
    `groups` the lists of patients with the same disease."""
    spec, threshold = (protocol["pool"], protocol["memory"]["notes_threshold"])
    seed = spec["seed"]
    pool = list(frozen["patients"])
    members = [c for group in groups for c in group]
    if (
        len(set(pool)) != len(pool)
        or len(pool) != spec["patients"]
        or len(set(members)) != len(members)
        or (not set(members) <= set(pool))
        or any((len(group) < 2 for group in groups))
    ):
        raise ValueError(
            "The pool or the groups of same-disease patients are not what the protocol states"
        )
    name = {c: golds[c]["options"][golds[c]["label"]] for c in pool}

    def connects(a, b):
        return likeness(name[a], name[b]) >= threshold

    candidates = []
    for group in groups:
        ordered = sorted(
            group,
            key=lambda c: (
                -sum((connects(c, o) for o in group if o != c)),
                key(seed, "anchor", c),
            ),
        )
        candidates += [
            (not connects(c, ordered[0]), key(seed, "test1", c), c) for c in ordered[1:]
        ]
    wanted = share(spec["test1_share"], len(pool))
    if wanted != spec["test1"] or len(candidates) < wanted:
        raise ValueError(
            "The same-disease groups do not give the stated number of test patients"
        )
    chosen = {c for _, _, c in sorted(candidates)[:wanted]}
    test1 = sorted(chosen, key=lambda c: key(seed, "test1-order", c))
    train = [c for c in pool if c not in chosen]
    partners = {
        c: sorted(
            (o for group in groups if c in group for o in group if o not in chosen)
        )
        for c in test1
    }
    if any((not found for found in partners.values())) or len(train) != spec["train"]:
        raise ValueError("A test patient has no training patient with the same disease")
    wanted = share(spec["test2_share"], len(pool))
    if wanted != spec["test2"] or wanted > len(train):
        raise ValueError(
            "The training patients do not give the stated number of augmented patients"
        )
    sources = sorted(train, key=lambda c: key(seed, "test2", c))[:wanted]

    def tie(c):
        return key(seed, "order", c)

    orders = {
        "random": sorted(train, key=tie),
        "easy_to_hard": sorted(train, key=lambda c: (-levels[c], tie(c))),
        "hard_to_easy": sorted(train, key=lambda c: (levels[c], tie(c))),
    }
    reachable = [c for c in test1 if any((connects(c, o) for o in partners[c]))]
    smoke_test, smoke_train = (reachable[: protocol["smoke"]["test1"]], [])
    for c in smoke_test:
        partner = next((o for o in partners[c] if connects(c, o)))
        if partner not in smoke_train:
            smoke_train.append(partner)
    smoke_train += [c for c in orders["random"] if c not in smoke_train][
        : protocol["smoke"]["train"] - len(smoke_train)
    ]
    if (
        len(smoke_test) != protocol["smoke"]["test1"]
        or len(set(smoke_train)) != protocol["smoke"]["train"]
    ):
        raise ValueError("The pool cannot fill the readiness check")
    return {
        "version": spec["version"],
        "seed": seed,
        "patients": len(pool),
        "train": train,
        "test1": test1,
        "test1_partners": partners,
        "test1_reachable": reachable,
        "test2_sources": sources,
        "orders": orders,
        "smoke": {"train": smoke_train, "test1": smoke_test},
    }
