"""Dry-run validator for data/glm_raw_v12 replies.

Runs the real ingest_glm_batch.check() gates + the batch-level L7 gold-position
skew check + duplicate-question detection WITHOUT writing to
data/manual_aug_glm or data/glm_raw/_rejected.jsonl.
"""
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/home/g2/Mohamed/temporal-news-reasoning")
sys.path.insert(0, str(ROOT / "scripts"))
from glm_client import extract_jsonl  # noqa: E402
import ingest_glm_batch as ig  # noqa: E402

seen: set[str] = set()
for p in sorted(Path(sys.argv[1]).glob("*.txt")):
    raw = p.read_text(encoding="utf-8")
    rows = extract_jsonl(raw)
    ok, bad, mcq = 0, [], []
    for r in rows:
        errs = ig.check(r)
        key = re.sub(r"\s+", " ", str(r.get("question", ""))).strip().lower()
        if not errs and key in seen:
            errs = ["duplicate question (within run)"]
        if errs:
            bad.append((errs, r.get("category")))
        else:
            seen.add(key)
            ok += 1
            if "Choices:" in r.get("question", ""):
                mcq.append(r)
    pos = Counter()
    for r in mcq:
        ch = ig.options(r["question"])
        g = str(r["targets"][0]).strip()
        if g in ch:
            pos["ABCDEFG"[ch.index(g)]] += 1
    skew = ""
    if pos:
        top, n = pos.most_common(1)[0]
        if n / sum(pos.values()) > 0.60:
            skew = f"  !! GOLD POSITION SKEW {dict(pos)}"
    wc = [len(str(r.get("context", "")).split())
          for r in rows if r.get("category") == "storytelling"]
    extra = f"  ctx_words={min(wc)}-{max(wc)}" if wc else ""
    golds = Counter(str(r["targets"][0]) for r in rows if not bad or r in rows)
    print(f"{p.name}: parsed={len(rows)} ok={ok} rejected={len(bad)}"
          f"{extra}  gold_pos={dict(pos) or '-'}{skew}")
    for errs, c in bad:
        print(f"    [{c}] {errs}")
if not list(Path(sys.argv[1]).glob("*.txt")):
    print("no .txt files found")
