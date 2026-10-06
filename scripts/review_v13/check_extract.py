"""Cross-check extract rows: do the gold letters equal the options whose
dates appear literally in the context?

Usage: python3 check_extract.py <packet> [--all]
Prints rows where the mechanical check disagrees with gold (for manual
review) plus a tally. Without --all, checks only pending (unverdicted) rows.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path("/home/g2/Mohamed/temporal-news-reasoning")
PACKETS = ROOT / "data" / "v13_verify" / "review_packets"
REVIEW = ROOT / "data" / "v13_verify" / "review"

MONTHS = {m: i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"])}
DATE_RE = re.compile(r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|"
                     r"August|September|October|November|December)\s*,?\s+(\d{4})")
OPT_RE = re.compile(r"^([A-Z])\.\s+(\d{1,2})\s+(January|February|March|April|May|June|"
                    r"July|August|September|October|November|December)\s*,?\s+(\d{4})$")


def dates_in(text):
    return {(int(d), MONTHS[m], int(y)) for d, m, y in DATE_RE.findall(text)}


def gold_letters(gold):
    return [p for p in gold.split() if p]


def main():
    packet = sys.argv[1]
    check_all = "--all" in sys.argv
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
    n_ok = n_bad = 0
    for r in rows:
        if r.get("category") != "extract":
            continue
        if not check_all and r["rid"] in done:
            continue
        ctx = r.get("context") or r.get("evidence") or ""
        present = dates_in(ctx)
        opts = {}
        for line in r["question"].splitlines():
            m = OPT_RE.match(line.strip())
            if m:
                opts[m.group(1)] = (int(m.group(2)), MONTHS[m.group(3)], int(m.group(4)))
        if not opts:
            continue
        expect = sorted(k for k, d in opts.items() if d in present)
        got = sorted(gold_letters(r["gold"]))
        if expect == got:
            n_ok += 1
        else:
            n_bad += 1
            print(f"DISAGREE {r['rid']}: checker={expect} gold={got}")
            print("  ctx dates:", sorted(present))
            print("  options:", opts)
    print(f"# {packet}: {n_ok} agree, {n_bad} disagree")


if __name__ == "__main__":
    main()
