from clinical_memory.evaluation.episode import execute
from clinical_memory.memory.feedback import answer_key
from clinical_memory.memory.reflection import draft

SHEET_SHOWN = {"self": False, "key": True, "key_dx": True}
NOTES_SHOWN = {"self": False, "key": False, "key_dx": True}


def visit(backend, case, gold, memory, settings, max_actions, learning):
    """A visit of a patient under one way of learning; evaluation patients are seen the same way."""
    return execute(
        backend,
        case,
        gold,
        memory,
        settings,
        max_actions,
        notes_on=NOTES_SHOWN[learning],
    )


def learn(backend, case, gold, memory, settings, max_actions, learning, step):
    """Return (the visit, notebook entry or None, what happened)."""
    seen = visit(backend, case, gold, memory, settings, max_actions, learning)
    key = answer_key(case, gold, seen) if SHEET_SHOWN[learning] else None
    lesson, info = draft(
        backend, seen["trace"], "evolution", step, key, settings["lesson_format_policy"]
    )
    return (seen, lesson, info)
