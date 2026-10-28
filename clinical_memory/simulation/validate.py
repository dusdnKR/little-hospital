"""Replay the actual configured files to completion and check read-only invariants."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from PIL import Image
from .engine import Hospital
from .replay import load_replay


def validate(config=None):
    replay = load_replay(config)
    if replay.warnings or not replay.count:
        raise ValueError(
            "Complete validation requires readable episode files: "
            + "; ".join(replay.warnings)
        )
    sources = [ep for episodes in replay.encounters.values() for ep in episodes]
    attachments = [a for ep in sources for m in ep.messages for a in m.images]
    images_read = 0
    for attachment in attachments:
        if not attachment.path:
            raise ValueError(f"Replay image is missing: {attachment.name}")
        with Image.open(attachment.path) as image:
            image.verify()
        images_read += 1
    korean_dialogue = sum(
        (bool(re.search("[가-힣]", m.text)) for ep in sources for m in ep.messages)
    )

    def hashes():
        return {
            ep.source: hashlib.sha256(Path(ep.source).read_bytes()).hexdigest()
            for ep in sources
        }

    before = hashes()
    h = Hospital(replay)
    for _ in range(max(1800, replay.count * 180)):
        h.tick(1)
        if h.completed:
            break
    if not h.completed:
        raise AssertionError("Playback did not drain every queue")
    actual = [v for v in h.records if not v.encounter.npc]
    assert Counter((v.encounter.key for v in actual)) == Counter(
        (ep.key for ep in sources)
    )
    assert all((tuple(v.messages) == v.encounter.messages for v in actual))
    expected_success = sum((ep.success for ep in sources))
    assert (h.coins, h.recovered, h.deaths) == (
        expected_success,
        expected_success,
        replay.count - expected_success,
    )
    assert before == hashes(), "Original episode files changed during replay"
    assert not h.exam_queue and (not h.payment_queue) and (not h.reg_queue)
    assert h.exam_owner is None and h.payment_owner is None
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "input_records": {
            m: len(episodes) for m, episodes in replay.encounters.items()
        },
        "agent_completed": len(actual),
        "npc_completed": len(h.records) - len(actual),
        "coins": h.coins,
        "recovered": h.recovered,
        "deaths": h.deaths,
        "full_transcripts_preserved": True,
        "source_files_unchanged": True,
        "image_references": len(attachments),
        "images_read": images_read,
        "korean_dialogue_messages": korean_dialogue,
        "queues_drained": True,
        "new_model_calls": 0,
        "simulation_seed": 42,
        "note": "Historical recorded episodes mapped to game outcomes; not new clinical results.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.config)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
