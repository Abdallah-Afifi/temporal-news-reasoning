"""Generate the v9 synthetic arm (AUG_GLM2) per docs/synthetic_data_ruleset.md.

Usage:
    venv/bin/python scripts/generate_v9_aug.py [--limit N] [--seed S] \
        [--out data/manual_aug_v9]

Produces <category>_batchNN.jsonl files of ~50 rows each (§10), 6,000 rows
total across Blocks A/B/C, plus a hidden .generation_summary.json.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from v9.block_a import BlockA, CATS  # noqa: E402
from v9.block_b import BlockB  # noqa: E402
from v9.block_c import BlockC  # noqa: E402
from v9.corpus import TopicIndex, has_full_pair, load_articles  # noqa: E402


def build_pools(arts):
    full2, series3, series4, series5, month4, month5, any2 = [], [], [], [], [], [], []
    for a in arts:
        keys3 = {e.date.key() for e in a.events if e.date.precision == 3}
        months = {(e.date.y, e.date.m) for e in a.events if e.date.precision >= 2}
        keys_all = {e.date.key() for e in a.events}
        if len(keys3) >= 2:
            full2.append(a)
        if len(keys3) >= 3:
            series3.append(a)
        if len(keys3) >= 4:
            series4.append(a)
        if len(keys3) >= 5:
            series5.append(a)
        if len(months) >= 4:
            month4.append(a)
        if len(months) >= 5:
            month5.append(a)
        if len(keys_all) >= 2:
            any2.append(a)
    return full2, series3, series4, series5, month4, month5, any2


def write_batches(rows: list[dict], out_dir: Path) -> None:
    by_cat: Counter[str] = Counter()
    per_cat: dict[str, list[dict]] = {}
    for r in rows:
        per_cat.setdefault(r["category"], []).append(r)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cat, cat_rows in per_cat.items():
        random.Random(123).shuffle(cat_rows)
        for i in range(0, len(cat_rows), 50):
            batch = cat_rows[i:i + 50]
            n = i // 50 + 1
            path = out_dir / f"{cat}_batch{n:02d}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for r in batch:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        by_cat[cat] = len(cat_rows)
    print(json.dumps(dict(sorted(by_cat.items())), indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/manual_aug_v9")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None,
                    help="scan only the first N usable articles (smoke tests)")
    ap.add_argument("--ban", default=None,
                    help="file of banned source_ids (contamination purge)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    ban = set()
    if args.ban and Path(args.ban).exists():
        ban = {l.strip() for l in open(args.ban) if l.strip()}
        print(f"ban list: {len(ban)} source ids")

    cache = Path("data/.v9_cache")
    cache.mkdir(parents=True, exist_ok=True)
    ckey = cache / f"arts_{args.limit or 'all'}_{len(ban)}.pkl"
    if ckey.exists():
        print(f"loading cached article index {ckey}")
        arts = pickle.loads(ckey.read_bytes())
    else:
        print("scanning corpus...")
        arts = load_articles(PROJECT_ROOT / "data" / "corpus" / "ccnews",
                             ban=ban, max_articles=args.limit)
        pickle.dump(arts, ckey.open("wb"))
    print(f"usable articles: {len(arts)}")
    full2, series3, series4, series5, month4, month5, any2 = build_pools(arts)
    print(f"pools: full2={len(full2)} series3={len(series3)} "
          f"series4={len(series4)} series5={len(series5)} "
          f"month4={len(month4)} month5={len(month5)} any2={len(any2)}")
    topics = TopicIndex(arts)

    def topup(fn, want, *args, passes: int = 6):
        out: list[dict] = []
        for _ in range(passes):
            got = fn(*args, want - len(out)) if args else fn(want - len(out))
            out.extend(got)
            if len(out) >= want:
                break
        return out[:want]

    rows: list[dict] = []
    a = BlockA(arts, topics, rng)
    rows += topup(a.gen_computation, CATS["Computation"]["total"], full2)
    rows += topup(a.gen_timeline, CATS["Timeline"]["total"],
                  {3: any2, 4: month4 or (series4 or series3),
                   5: month5 or (series5 or (series4 or series3))})
    rows += a.gen_localization(any2, CATS["Localization"]["total"])
    rows += a.gen_duration_compare(month4, CATS["Duration_Compare"]["total"])
    rows += a.gen_order_compare(full2, CATS["Order_Compare"]["total"])
    rows += topup(a.gen_relative,
                  CATS["Relative_Reasoning"]["total"] - 2 * CATS["Relative_Reasoning"]["pairs"],
                  any2)
    rows += topup(a.gen_order_reasoning,
                  CATS["Order_Reasoning"]["total"] - 2 * CATS["Order_Reasoning"]["pairs"],
                  any2)
    rows += topup(a.gen_co_temporality,
                  CATS["Co_temporality"]["total"] - 2 * CATS["Co_temporality"]["pairs"],
                  any2)
    rows += topup(a.gen_explicit,
                  CATS["Explicit_Reasoning"]["total"] - 2 * CATS["Explicit_Reasoning"]["pairs"],
                  any2)
    rows += topup(a.gen_counterfactual, CATS["Counterfactual"]["total"] - 2 * CATS["Counterfactual"]["pairs"], any2)
    rows += a.gen_pairs(any2)
    if a.shortfalls:
        print(f"SHORTFALLS: {a.shortfalls}")

    b = BlockB(full2, rng)
    rows += b.gen_nli(400, mcq=False)
    rows += b.gen_nli(250, mcq=True)
    rows += b.gen_relation(150)
    rows += b.gen_ordering(100)

    c = BlockC(any2, topics, rng)
    rows += c.gen_dialogue(250, pool=full2)
    rows += c.gen_duration(150)
    rows += c.gen_storytelling(150)
    rows += c.gen_longform(150)
    aug_rows = [r for r in rows if r["slice"] == "A"]
    want_a = {k: v["total"] for k, v in CATS.items()}
    have_a = Counter(r["category"] for r in aug_rows)
    gap = {k: want_a[k] - have_a.get(k, 0) for k in want_a
           if want_a[k] - have_a.get(k, 0) > 0}
    if gap:
        print(f"Block A short after top-up: {gap}")

    out_dir = PROJECT_ROOT / args.out
    if out_dir.exists():
        for old in out_dir.glob("*.jsonl"):
            old.unlink()
    write_batches(rows, out_dir)
    summary = {
        "total": len(rows),
        "by_category": dict(Counter(r["category"] for r in rows)),
        "by_provenance": dict(Counter(r["provenance"] for r in rows)),
        "by_slice": dict(Counter(r["slice"] for r in rows)),
        "shortfalls": a.shortfalls,
        "seed": args.seed,
        "limit": args.limit,
    }
    (out_dir / ".generation_summary.json").write_text(
        json.dumps(summary, indent=1))
    print(f"TOTAL {len(rows)} rows -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
