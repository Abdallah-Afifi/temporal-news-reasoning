"""Display pending rows grouped by shared context (prints each context once).

Usage: python3 show_grouped.py <slice_NN> [start] [count]
start/count are in PENDING-ROW order (same as show_pending).
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
print(f"# packet={packet} pending={len(pending)} showing [{start},{start+count})")

shown = 0
skipped = 0
last_ctx = None
for i, r in pending:
    if shown >= count:
        break
    if skipped < start:
        skipped += 1
        last_ctx = r.get("context") or r.get("evidence") or ""
        continue
    ctx = r.get("context") or r.get("evidence") or ""
    if ctx != last_ctx:
        print(f"\n########## NEW CONTEXT ({len(ctx)} chars)")
        print(ctx)
        print("########## END CONTEXT")
    last_ctx = ctx
    print(f"\n[#{i}] {r['rid']} {r.get('category','')}")
    print(f"Q: {r['question']}")
    print(f"GOLD: {r['gold']}")
    shown += 1
print(f"\n# done showing through pending-index {start + shown}")
