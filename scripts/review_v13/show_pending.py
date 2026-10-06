"""Print pending (unreviewed) rows of a review packet, compactly.

Usage: python3 show_pending.py <slice_NN> [start] [count]
Index shown is the 1-based position within the packet.
"""
import json
import sys
from pathlib import Path

ROOT = Path("/home/g2/Mohamed/temporal-news-reasoning")
PACKETS = ROOT / "data" / "v13_verify" / "review_packets"
REVIEW = ROOT / "data" / "v13_verify" / "review"

packet = sys.argv[1]
start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
count = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9

rows = [json.loads(l) for l in open(PACKETS / f"{packet}.jsonl", encoding="utf-8") if l.strip()]
done = set()
v = REVIEW / f"{packet}.verdicts.jsonl"
if v.exists():
    for l in open(v, encoding="utf-8"):
        if l.strip():
            try:
                done.add(json.loads(l)["rid"])
            except json.JSONDecodeError:
                pass
pending = [(i, r) for i, r in enumerate(rows, 1) if r["rid"] not in done]
print(f"# packet={packet} total={len(rows)} done={len(done)} pending={len(pending)}")
skipped = 0
shown = 0
for i, r in pending:
    if shown >= count:
        break
    if skipped < start:
        skipped += 1
        continue
    ctx = r.get("context") or r.get("evidence") or ""
    print(f"\n[#{i}] {r['rid']} {r.get('category','')} ({r.get('source_dataset','')})")
    print(f"Q: {r['question']}")
    print(f"GOLD: {r['gold']}")
    if ctx:
        print(f"CTX: {ctx}")
    shown += 1
print(f"\n# shown {shown} of {len(pending)} pending (from pending-index {start})")
