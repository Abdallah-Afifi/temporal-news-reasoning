"""Build the v4 training mixture — one variable vs v3-corrected, per
docs/v4_plan.md Stage 4 and its mistake-ledger precondition
(docs/mistake_ledger.md).

v4 = v3-corrected's core + template augs, held constant
     (TimeQA, TLQA with full answer lists, AUG_DIALOG, AUG_REASON,
     AUG_MCQ, AUG_ARITH — none reformatted or rebuilt here; doing so
     would be a second variable, which is exactly what v4_plan.md's
     Stage 4 rules out)
   + AUG_NLI, with its degenerate Neutral rows dropped (label-leakage
     fix decided in docs/queue_status_and_timeline.md, "applied to v4,
     not v3-corrected": Entailment/Contradiction rows share identical
     wording and are not leaky; only Neutral's "another subject"
     phrasing is a 67.1%-guessable tell)
   + AUG_GLM curated manual records (502, all of them)
   + REHEARSAL resampled to 3,344 from an equal-part HotpotQA/DROP/CoQA
     sample (context-grounded QA, matching TIME/TimeBench's own shape)
     instead of dolly instruction-following -- the one variable
     (decision 6)
   + optional STaR/CoT overlay (--star-file, applied when the pilot
     lands)

Explicitly excluded (docs/v4_plan.md decisions 1, 7-9):
- AUG_LETTER: built only to patch Timeline's harness-bug 0%; the
  mistake ledger confirms the harness fix alone already recovers
  Timeline (zero-shot 16.5%, v3-corrected 13.3%), so its purpose is
  void. Its probe file is still used below as a defensive leak guard,
  even though nothing should train on those questions now.
- cotcoll, slimorca, squad: deferred to v5 / general instruction
  volume, not this cycle's hypothesis.

Guards: benchmark-question collision, letter-probe exclusion (checked
even with AUG_LETTER absent), stratified 80/20 split, manifest with
per-source counts.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REHEARSAL_POOLS = {
    "HOTPOT": "data/prepared_v4/hotpot.jsonl",
    "DROP": "data/prepared_v4/drop.jsonl",
    "COQA": "data/prepared_v4/coqa.jsonl",
}


def norm(q: str) -> str:
    return " ".join(q.lower().split())


def benchmark_questions() -> set[str]:
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.data.data_loader import BenchmarkLoader
    qs: set[str] = set()
    loader = BenchmarkLoader(str(PROJECT_ROOT / "data" / "benchmarks"))
    for bench in ("time", "timebench"):
        for ex in loader.load(bench):
            qs.add(norm(ex.question))
    return qs


def filter_nli_neutral(rows: list[dict]) -> tuple[list[dict], int]:
    """Drop AUG_NLI's degenerate Neutral class (label predictable from the
    literal word 'another' in the template, 67.1% vs 33.3% chance rate --
    docs/session_state.md's known-defects table, docs/mistake_ledger.md).
    Entailment/Contradiction rows are unaffected: both are phrased "the
    subject is associated with X" and require reading the premise."""
    kept, dropped = [], 0
    for r in rows:
        if r.get("source_dataset") == "AUG_NLI" and list(r.get("targets") or []) == ["Neutral"]:
            dropped += 1
            continue
        kept.append(r)
    return kept, dropped


def rehearsal_slice(pools: dict[str, str], n: int, rng: random.Random,
                    bench_qs: set[str]) -> list[dict]:
    """Equal-part sample across context-grounded QA pools (decision 6:
    dolly instruction-following -> context-grounded QA, since TIME and
    TimeBench are themselves context-grounded QA over passages)."""
    names = list(pools)
    per_pool = n // len(names)
    remainder = n - per_pool * len(names)
    out: list[dict] = []
    for idx, name in enumerate(names):
        rows = [json.loads(l) for l in open(pools[name], encoding="utf-8")]
        ok = []
        for r in rows:
            q = str(r.get("question", "")).strip()
            targets = [str(t) for t in (r.get("targets") or []) if str(t).strip()]
            if not q or not targets:
                continue
            if norm(q) in bench_qs:
                continue
            ok.append({
                "source_dataset": "REHEARSAL",
                "question": q,
                "context": str(r.get("context", "") or ""),
                "targets": targets,
                "source": f"prepared_v4/{name.lower()}",
            })
        rng.shuffle(ok)
        take = per_pool + (1 if idx < remainder else 0)
        out.extend(ok[:take])
        print(f"  rehearsal/{name}: sampled {min(take, len(ok))} of {len(ok)} eligible "
              f"({len(rows)} raw)")
    rng.shuffle(out)
    return out


def overlay_star(records: list[dict], star_path: str) -> tuple[list[dict], int]:
    by_q = {}
    for line in open(star_path, encoding="utf-8"):
        c = json.loads(line)
        if c.get("verified") and c.get("cot"):
            by_q[norm(str(c.get("question") or ""))] = c["cot"]
    n_overlaid = 0
    for r in records:
        cot = by_q.get(norm(str(r.get("question") or "")))
        if cot:
            r["targets"] = [cot.rstrip()]
            r["cot"] = True
            n_overlaid += 1
    return records, n_overlaid


def purge_near_duplicates(records: list[dict]) -> tuple[list[dict], int]:
    """Drop training records that near-duplicate (J>0.8, token-set Jaccard)
    TimeBench timeqa evaluation items. Pass-4 audit: 30/1,000 zip-timeqa
    items near-dup our train TimeQA (0.00pp measured impact; purge anyway
    so v4 is defensibly clean)."""
    import re
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.data.data_loader import BenchmarkLoader

    def tk(s: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", str(s).lower()))

    def norm_ctx(s: str) -> str:
        # DEDUPED token-set normalization: catches truncated/expanded copies
        # of eval passages (same unique vocabulary, different token counts).
        return " ".join(sorted(set(re.findall(r"[a-z0-9]+", str(s).lower()))))

    loader = BenchmarkLoader(str(PROJECT_ROOT / "data" / "benchmarks"))
    eval_timeqa = loader.load("timebench", task="timeqa")
    bank = [tk(ex.question) for ex in eval_timeqa]
    ctx_bank = {norm_ctx(ex.context) for ex in eval_timeqa if getattr(ex, "context", "")}
    kept, dropped = [], 0
    for r in records:
        q = tk(r.get("question") or "")
        # passage-level guard: same passage (different question) still leaks facts
        if r.get("context") and norm_ctx(r["context"]) in ctx_bank:
            dropped += 1
            continue
        best = 0.0
        for bq in bank:
            if bq and q:
                j = len(q & bq) / len(q | bq)
                if j > best:
                    best = j
        if best > 0.8:
            dropped += 1
        else:
            kept.append(r)
    return kept, dropped


def stratified_split(records: list[dict], val_frac: float,
                     rng: random.Random) -> tuple[list[list[dict]]]:
    by_src: dict[str, list[dict]] = {}
    for r in records:
        by_src.setdefault(r["source_dataset"], []).append(r)
    train, val = [], []
    for src, rows in by_src.items():
        rng.shuffle(rows)
        cut = int(len(rows) * val_frac)
        val.extend(rows[:cut])
        train.extend(rows[cut:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v3-dir", default="data/combined_80_20_v3_fixed",
                    help="corrected v3 mixture (full TLQA answer lists, "
                         "collapsed AUG_REASON) -- v4's held-constant core")
    ap.add_argument("--manual-aug", default="data/manual_aug")
    ap.add_argument("--letter-probe", default="data/letter_format/probe.jsonl",
                    help="held-out letter-format probe questions -- checked "
                         "as a leak guard even though AUG_LETTER is not trained on")
    ap.add_argument("--out", default="data/combined_80_20_v4")
    ap.add_argument("--rehearsal", type=int, default=3344)
    ap.add_argument("--star-file", default="",
                    help="optional verified STaR/CoT jsonl to overlay")
    ap.add_argument("--star-sources", default="TimeQA,TLQA,AUG_LETTER")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    bench_qs = benchmark_questions()

    v3_rows = [json.loads(l) for l in open(Path(args.v3_dir) / "train.jsonl", encoding="utf-8")]
    v3_val = [json.loads(l) for l in open(Path(args.v3_dir) / "val.jsonl", encoding="utf-8")]
    keep = [r for r in v3_rows + v3_val if r["source_dataset"] != "REHEARSAL"]
    keep, n_nli_dropped = filter_nli_neutral(keep)
    print(f"v3-corrected core+template (REHEARSAL excluded, AUG_NLI Neutral "
          f"dropped): {len(keep)} (dropped {n_nli_dropped} Neutral rows)")

    manual = []
    for p in sorted(Path(args.manual_aug).glob("glm_batch*.jsonl")):
        manual += [json.loads(l) for l in open(p, encoding="utf-8")]
    print(f"AUG_GLM manual: {len(manual)}")

    probe_q = {norm(json.loads(l)["question"])
               for l in open(args.letter_probe, encoding="utf-8")}
    print(f"letter probe held out (leak guard only, AUG_LETTER not trained on): {len(probe_q)}")

    print(f"rehearsal slice (target {args.rehearsal}, equal parts of "
          f"{', '.join(REHEARSAL_POOLS)}):")
    reh = rehearsal_slice(REHEARSAL_POOLS, args.rehearsal, rng, bench_qs)
    print(f"  total rehearsal: {len(reh)}")

    all_records = keep + manual + reh

    # Loader compat fix: the finetuning data_loader's TLQA branch reads
    # final_answers/answers (never targets); emit both fields.
    for r in all_records:
        if r["source_dataset"] == "TLQA" and r.get("targets"):
            r["final_answers"] = list(r["targets"])

    all_q = [norm(str(r.get("question") or "")) for r in all_records]
    n_probe_leak = sum(q in probe_q for q in all_q)
    assert n_probe_leak == 0, f"FATAL: {n_probe_leak} probe questions leaked into training"
    collisions = sum(q in bench_qs for q in all_q)
    assert collisions == 0, f"FATAL: {collisions} benchmark collisions"
    seen_q: set[str] = set()
    deduped = []
    for r, q in zip(all_records, all_q):
        if q in seen_q:
            continue
        seen_q.add(q)
        deduped.append(r)
    n_dupes = len(all_records) - len(deduped)
    print(f"guards passed: 0 probe leaks, 0 benchmark collisions; deduped {n_dupes} rows")
    all_records = deduped

    all_records, n_near = purge_near_duplicates(all_records)
    print(f"near-dup purge vs TimeBench timeqa (J>0.8): dropped {n_near} rows")

    n_overlaid = 0
    if args.star_file:
        srcs = {s.strip() for s in args.star_sources.split(",")}
        eligible = [r for r in all_records if r["source_dataset"] in srcs]
        eligible, n_overlaid = overlay_star(eligible, args.star_file)
        by_id = {id(r): r for r in eligible}
        all_records = [by_id.get(id(r), r) for r in all_records]
        print(f"STaR overlaid: {n_overlaid}/{len(eligible)} eligible")

    train, val = stratified_split(all_records, args.val_frac, rng)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train), ("val.jsonl", val)):
        with open(out / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def mix(rows):
        c = Counter(r["source_dataset"] for r in rows)
        syn = sum(v for k, v in c.items() if k.startswith("AUG_"))
        return {
            "n": len(rows),
            "synthetic_pct": round(100 * syn / len(rows), 1),
            "by_source": dict(c),
        }

    rehearsal_by_pool = Counter(r["source"] for r in reh)
    stats = {
        "train": mix(train), "val": mix(val),
        "aug_nli_neutral_dropped": n_nli_dropped,
        "rehearsal_by_pool": dict(rehearsal_by_pool),
        "star_overlaid": n_overlaid, "seed": args.seed, "args": vars(args),
    }
    (out / "manifest.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
