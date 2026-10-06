"""Progress of the v13 human-style review: which packets are complete.

A packet is DONE when its verdicts file covers every rid exactly once with a
valid verdict. Prints one line per packet and a verdict tally.

Usage: venv/bin/python scripts/review_v13/review_status.py [--todo]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKETS = ROOT / "data" / "v13_verify" / "review_packets"
REVIEW = ROOT / "data" / "v13_verify" / "review"
VALID = {"CORRECT", "WRONG", "AMBIGUOUS", "MALFORMED", "NOT_FOUND"}


def packet_state(p: Path) -> tuple[int, int, Counter, list[str]]:
    rids = [json.loads(l)["rid"] for l in open(p, encoding="utf-8") if l.strip()]
    v = REVIEW / f"{p.stem}.verdicts.jsonl"
    got, tally, problems = {}, Counter(), []
    if v.exists():
        for line in open(v, encoding="utf-8"):
            if not line.strip():
                continue
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                problems.append("bad json line")
                continue
            if j.get("verdict") not in VALID:
                problems.append(f"bad verdict {j.get('verdict')!r}")
                continue
            if j["rid"] in got:
                problems.append(f"duplicate {j['rid']}")
            got[j["rid"]] = j["verdict"]
    missing = [r for r in rids if r not in got]
    extra = [r for r in got if r not in set(rids)]
    if extra:
        problems.append(f"{len(extra)} unknown rids")
    tally.update(got[r] for r in rids if r in got)
    return len(rids), len(rids) - len(missing), tally, problems


def main() -> int:
    todo_only = "--todo" in sys.argv
    total = Counter()
    for p in sorted(PACKETS.glob("*.jsonl")):
        n, done, tally, problems = packet_state(p)
        total.update(tally)
        state = "DONE" if done == n and not problems else f"{done}/{n}"
        if not todo_only or state != "DONE":
            print(f"{p.stem:16s} {state:9s} {dict(tally)} {'; '.join(problems[:2])}")
    print("TOTAL", dict(total), "reviewed", sum(total.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
