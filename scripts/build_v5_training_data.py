"""Build the v5 training mixture — one variable vs v4, per docs/v5_plan.md.

v5 = v4's mixture with REHEARSAL SCALED UP and nothing else changed.

Why this variable. Under the v5 scoring protocol (docs/v5_plan.md § "What v4
actually measured"), v4 lands 37.63% on TIME against zero-shot's 41.49%. That
remaining deficit is no longer concentrated the way v1/v2's was — it is spread
across Counterfactual (-12.8pp), Timeline (-6.0pp), Explicit_Reasoning
(-7.4pp), Order_Reasoning (-5.7pp) and Relative_Reasoning (-5.3pp), which is
the signature of general reasoning capability being forgotten during
fine-tuning rather than of any single missing task. Rehearsal is the standard
remedy and v4 carried only 3,344 rows of it (20.9% of the mixture), inherited
unchanged from v3. v5 scales that count and holds everything else fixed.

What is deliberately NOT changed, so the variable stays attributable:
- Pool composition stays equal-parts HotpotQA/DROP/CoQA. v4's swap already
  bought a real +10.3pp on Computation and, once the postprocessor was fixed,
  its terse-answer style turned out to be harmless (Order_Compare 62.0%, above
  zero-shot). There is no evidence left that argues for changing the mix, so
  changing it would add a second variable for nothing.
- AUG_MCQ targets stay text-first. Training letter-and-text surface forms is
  docs/v4_plan.md's format-following action; it is a *training* change and
  belongs in its own arm, not bundled with a volume change.
- Batch shape, LoRA rank, LR and epochs stay identical to v1-v4.

Optional second-arm content, OFF by default (--aug-counterfactual): a
GLM-generated Counterfactual/Timeline slice targeting the two largest
remaining category deficits. Enabling it makes the run a two-variable bundle
unless it is run as its own arm against this one — see docs/v5_plan.md.

Guards (all inherited from the v4 builder, none relaxed): benchmark-question
collision, letter-probe exclusion, passage-level and J>0.8 near-duplicate
purge against TimeBench timeqa, stratified 80/20 split, manifest with
per-source and per-pool counts.
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
# Usable (non-empty question AND targets) rows per pool, measured 2026-09-06.
# DROP ships 55,829 rows but 20,517 of them (36.7%) carry an empty target and
# are dropped by the filter below — its effective pool is 35,312, which caps
# how far an equal-parts sample can scale. Documented so a future --rehearsal
# value is chosen against the real ceiling, not the advertised one.
POOL_USABLE = {"HOTPOT": 84956, "DROP": 35312, "COQA": 108230}


def norm(q: str) -> str:
    return " ".join(q.lower().split())


def benchmark_questions() -> set[str]:
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.data.data_loader import BenchmarkLoader
    qs: set[str] = set()
    loader = BenchmarkLoader(str(PROJECT_ROOT / "data" / "benchmarks"))
    # TRAM was missing here until the 2026-09-07 audit: v6 turned out to be
    # TRAM-clean by luck, not by design, and TRAM contributes 175,929
    # questions. It matters as soon as v8's CC-News slices arrive.
    for bench in ("time", "timebench", "tram"):
        try:
            for ex in loader.load(bench):
                qs.add(norm(ex.question))
        except Exception as exc:
            raise SystemExit(
                f"FATAL: cannot load benchmark {bench!r} for the leakage guard: {exc}. "
                f"Refusing to build a mixture that has not been checked against it."
            )
    return qs


def filter_nli_neutral(rows: list[dict]) -> tuple[list[dict], int]:
    """Drop AUG_NLI's degenerate Neutral class (label predictable from the
    literal word 'another', 67.1% vs 33.3% chance) — same as v4."""
    kept, dropped = [], 0
    for r in rows:
        if r.get("source_dataset") == "AUG_NLI" and list(r.get("targets") or []) == ["Neutral"]:
            dropped += 1
            continue
        kept.append(r)
    return kept, dropped


def rehearsal_slice(pools: dict[str, str], n: int, rng: random.Random,
                    bench_qs: set[str]) -> list[dict]:
    """Equal-part sample across the context-grounded QA pools.

    Identical to the v4 builder's function except that it refuses to
    silently under-deliver: if a pool cannot supply its share, that is
    reported and raised rather than quietly shrinking the slice, because a
    short rehearsal slice would make v5 a smaller change than its manifest
    claims.
    """
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
        if take > len(ok):
            raise SystemExit(
                f"FATAL: rehearsal pool {name} can supply {len(ok)} usable rows "
                f"but {take} were requested. Lower --rehearsal (the binding "
                f"pool is DROP at ~{POOL_USABLE['DROP']} usable rows, so an "
                f"equal-parts slice caps near {3 * POOL_USABLE['DROP']}), or "
                f"pass --pools to change the composition (a second variable)."
            )
        out.extend(ok[:take])
        print(f"  rehearsal/{name}: sampled {take} of {len(ok)} eligible ({len(rows)} raw)")
    rng.shuffle(out)
    return out


def purge_near_duplicates(records: list[dict]) -> tuple[list[dict], int]:
    """Drop training records that near-duplicate (J>0.8 token-set Jaccard, or
    share a passage with) TimeBench timeqa evaluation items. Unchanged from v4."""
    import re
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.data.data_loader import BenchmarkLoader

    def tk(s: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", str(s).lower()))

    def norm_ctx(s: str) -> str:
        return " ".join(sorted(set(re.findall(r"[a-z0-9]+", str(s).lower()))))

    loader = BenchmarkLoader(str(PROJECT_ROOT / "data" / "benchmarks"))
    eval_timeqa = loader.load("timebench", task="timeqa")
    bank = [tk(ex.question) for ex in eval_timeqa]
    ctx_bank = {norm_ctx(ex.context) for ex in eval_timeqa if getattr(ex, "context", "")}
    kept, dropped = [], 0
    for r in records:
        if r.get("context") and norm_ctx(r["context"]) in ctx_bank:
            dropped += 1
            continue
        q = tk(r.get("question") or "")
        best = 0.0
        for bq in bank:
            if bq and q:
                j = len(q & bq) / len(q | bq)
                best = max(best, j)
        if best > 0.8:
            dropped += 1
        else:
            kept.append(r)
    return kept, dropped


def stratified_split(records: list[dict], val_frac: float,
                     rng: random.Random) -> tuple[list[dict], list[dict]]:
    by_src: dict[str, list[dict]] = {}
    for r in records:
        by_src.setdefault(r["source_dataset"], []).append(r)
    train, val = [], []
    for _src, rows in by_src.items():
        rng.shuffle(rows)
        cut = int(len(rows) * val_frac)
        val.extend(rows[:cut])
        train.extend(rows[cut:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def answer_style(rows: list[dict]) -> dict:
    """Answer-shape summary of the mixture.

    v4's post-mortem cost two weeks of misattribution because a shift in the
    model's ANSWER STYLE (terse "Fact 1", day-month-year dates) was read as a
    capability collapse. Style starts in the training data, so it is recorded
    here at build time and compared against the trained model's own outputs by
    scripts/audit_output_style.py before any long evaluation is spent.
    """
    import re
    words, numeric, dated, per_src = [], 0, 0, {}
    date_re = re.compile(r"\b(19|20)\d{2}\b|january|february|march|april|may|june|"
                         r"july|august|september|october|november|december", re.I)
    for r in rows:
        t = (r.get("targets") or [""])[0]
        w = len(str(t).split())
        words.append(w)
        numeric += bool(re.fullmatch(r"\W*\d+(\.\d+)?\W*", str(t)))
        dated += bool(date_re.search(str(t)))
        per_src.setdefault(r["source_dataset"], []).append(w)
    n = max(len(words), 1)
    return {
        "mean_target_words": round(sum(words) / n, 2),
        "pct_bare_numeric": round(100 * numeric / n, 1),
        "pct_date_bearing": round(100 * dated / n, 1),
        "mean_target_words_by_source": {
            k: round(sum(v) / len(v), 2) for k, v in sorted(per_src.items())
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v3-dir", default="data/combined_80_20_v3_fixed",
                    help="corrected v3 mixture — the core v4 and v5 share")
    ap.add_argument("--manual-aug", default="data/manual_aug")
    ap.add_argument("--letter-probe", default="data/letter_format/probe.jsonl")
    ap.add_argument("--out", default="data/combined_80_20_v5")
    ap.add_argument("--rehearsal", type=int, default=10032,
                    help="rehearsal rows (v4 used 3,344; default here is 3x). "
                         "Equal-parts sampling caps near 105,000 because DROP "
                         "has only ~35,312 usable rows.")
    ap.add_argument("--aug-counterfactual", default="",
                    help="optional GLM-generated Counterfactual/Timeline jsonl. "
                         "OFF by default: it is a SECOND variable and must be "
                         "run as its own arm (docs/v5_plan.md).")
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

    aug_cf = []
    if args.aug_counterfactual:
        for line in open(args.aug_counterfactual, encoding="utf-8"):
            r = json.loads(line)
            if r.get("question") and r.get("targets"):
                r.setdefault("source_dataset", "AUG_CF")
                aug_cf.append(r)
        print(f"AUG_CF (second variable — must be its own arm): {len(aug_cf)}")

    probe_q = {norm(json.loads(l)["question"])
               for l in open(args.letter_probe, encoding="utf-8")}
    print(f"letter probe held out (leak guard): {len(probe_q)}")

    print(f"rehearsal slice (target {args.rehearsal}, equal parts of "
          f"{', '.join(REHEARSAL_POOLS)}):")
    reh = rehearsal_slice(REHEARSAL_POOLS, args.rehearsal, rng, bench_qs)
    print(f"  total rehearsal: {len(reh)}")

    all_records = keep + manual + aug_cf + reh

    # Loader compat: the finetuning data_loader's TLQA branch reads
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
    print(f"near-dup purge vs TimeBench timeqa (J>0.8 or shared passage): dropped {n_near} rows")

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
        return {"n": len(rows), "synthetic_pct": round(100 * syn / len(rows), 1),
                "rehearsal_pct": round(100 * c.get("REHEARSAL", 0) / len(rows), 1),
                "by_source": dict(c)}

    stats = {
        "cycle": "v5",
        "variable_vs_v4": f"rehearsal {3344} -> {len(reh)} rows",
        "train": mix(train), "val": mix(val),
        "aug_nli_neutral_dropped": n_nli_dropped,
        "rehearsal_by_pool": dict(Counter(r["source"] for r in reh)),
        "near_dup_dropped": n_near,
        "answer_style_train": answer_style(train),
        "second_variable_enabled": bool(aug_cf),
        "seed": args.seed, "args": vars(args),
    }
    (out / "manifest.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
