from clinical_memory.analysis.paired import correct, saw_image


def actions(record):
    return [(e["type"], e.get("query", "")) for e in record["trace"]["events"]]


def answer(record):
    return record["trace"]["result"]["submission"].get("answer")


def changes(pairs):
    return {
        "encounters": len(pairs),
        "collection_changed": sum((actions(b) != actions(a) for b, a in pairs)),
        "image_gained": sum((saw_image(a) and (not saw_image(b)) for b, a in pairs)),
        "image_lost": sum((saw_image(b) and (not saw_image(a)) for b, a in pairs)),
        "answer_changed": sum((answer(b) != answer(a) for b, a in pairs)),
        "wrong_to_right": sum((correct(a) and (not correct(b)) for b, a in pairs)),
        "right_to_wrong": sum((correct(b) and (not correct(a)) for b, a in pairs)),
    }


def combine(items):
    """Add the funnels of several runs of one condition (seeds); repeated evaluations are not new patients."""

    def add(a, b):
        if isinstance(a, dict):
            return {k: add(a[k], b[k]) if k in b else a[k] for k in a} | {
                k: v for k, v in b.items() if k not in a
            }
        return a + b

    total = None
    for item in items:
        counted = {
            k: {x: y for x, y in v.items() if not x.startswith("rough_")}
            for k, v in item.items()
        }
        total = counted if total is None else add(total, counted)
    return total
