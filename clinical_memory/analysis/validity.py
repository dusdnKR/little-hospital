from collections import Counter

INVALID = ("invalid_action", "invalid_selection")
SMOKE_MAX_INVALID_SHARE = 0.1


def invalid_counts(records):
    """Unusable doctor actions and unusable Patient/Measurement selections among all collection actions."""
    statuses = Counter((e["status"] for r in records for e in r["trace"]["events"]))
    actions = sum((r["metrics"]["collection_actions"] for r in records))
    invalid = {name + "s": statuses[name] for name in INVALID}
    return {
        **invalid,
        "collection_actions": actions,
        "invalid_share": sum(invalid.values()) / actions if actions else 0.0,
        "max_share_in_smoke": SMOKE_MAX_INVALID_SHARE,
    }
