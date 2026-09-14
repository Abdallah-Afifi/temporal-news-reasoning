#!/usr/bin/env python3
"""Sample the next N pool records that have no CoT trace yet (across all
data/cot/*.jsonl caches) and print them compactly for manual tracing."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v2_training_data import norm

def done_questions() -> set[str]:
    done = set()
    for p in Path("data/cot").glob("*.jsonl"):
        for line in open(p, encoding="utf-8"):
            try:
                c = json.loads(line)
                if c.get("question"):
                    done.add(norm(str(c["question"])))
            except json.JSONDecodeError:
                continue
    return done

def main() -> int:
    n_timeqa = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    n_tlqa = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    rng = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 11)
    done = done_questions()
    timeqa, tlqa = [], []
    for i, line in enumerate(open("data/combined_80_20_v2/train.jsonl", encoding="utf-8")):
        r = json.loads(line)
        if norm(str(r.get("question") or "")) in done:
            continue
        if r.get("source_dataset") == "TimeQA" and r.get("context"):
            t = (r.get("targets") or [""])[0]
            if t and len(timeqa) < 400:
                timeqa.append((i, r, str(t)))
        elif r.get("source_dataset") == "TLQA":
            t = (r.get("final_answers") or r.get("answers") or [""])[0]
            if t and len(tlqa) < 200:
                tlqa.append((i, r, str(t)))
    picks = rng.sample(timeqa, min(n_timeqa, len(timeqa))) + rng.sample(
        tlqa, min(n_tlqa, len(tlqa)))
    print(f"# pool remaining: timeqa={len(timeqa)} tlqa={len(tlqa)}", file=sys.stderr)
    for i, r, gold in picks:
        ctx = str(r.get("context", ""))[:300].replace("\n", " ")
        print(f"IDX={i} | SRC={r['source_dataset']}")
        print(f"  Q: {r['question']}")
        if ctx:
            print(f"  CTX: {ctx}")
        print(f"  GOLD: {gold}")
        print()
    return 0

if __name__ == "__main__":
    sys.exit(main())
