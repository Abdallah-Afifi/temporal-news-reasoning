"""Build the v7 mixture — the MCQ/format arm, with the D45/D49 fixes.

v6 proved three things and this cycle acts on all of them:

1. **`AUG_SEQ` works.** Sequence emission 3.11% -> 81.87%, accuracy 17.01%
   BEATING zero-shot's 15.48% (D49). Kept unchanged.
2. **Cutting rehearsal cost the biggest bucket.** Context-grounded rehearsal
   44.2% -> 14.0% took answerable-MCQ from 64.49% to **60.99%**, below
   zero-shot — −1.14pp overall, the single biggest loss in v6 and exactly the
   risk D41/D42 pre-registered. Rehearsal is restored here.
3. **`AUG_NOANS` taught a shortcut.** v6 scored **99.88%** on the no-answer
   bucket with no temporal reasoning: the mixture never showed an
   abstain-phrased option as a WRONG answer, and TIME never offers one unless
   it IS the answer. That artifact is +1.09pp of v6's +0.73pp margin — strip
   it and v6 is 41.10% against zero-shot's 41.46%. Fixed here.

Three changes vs v6:

| change | why |
|---|---|
| rehearsal 2,600 -> 8,000, mostly context-grounded | restore the MCQ bucket (D49) |
| `AUG_MCQ2` 1,500 -> 6,000, **single target** | the untested lever (D41/D42) + D45 |
| abstain options injected as DISTRACTORS in answerable rows | kill the shortcut |

**D45 — single-target MCQ.** `AUG_MCQ`/`AUG_MCQ2` no longer carry a dual gold
`[text, LETTER]`. The finetuning loader joins a target list with `"; "`, so a
dual gold taught the model to answer `"<option text>; B"` — a shape the
benchmark does not expect, and the ONLY reason the scorer needs a
trailing-letter rule. With a single target the model answers in the
benchmark's own form. **Pre-registered check: count `_TRAILING_LETTER_RE`
firings on v7's predictions — ~0 firings plus a good score proves the result
does not depend on the scoring change.**

**Expected cost of the honesty fix, stated in advance:** v7's no-answer bucket
will fall well below v6's 99.88% — that is the point. It should be made up by
the MCQ restoration (+1.14pp) and the AUG_MCQ scale-up. Report v6 and v7 side
by side so the artifact's size is visible.

Usage:
    ./venv/bin/python scripts/build_v7_training_data.py
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_v5_training_data import (  # noqa: E402
    REHEARSAL_POOLS, answer_style, benchmark_questions, filter_nli_neutral,
    norm, purge_near_duplicates, rehearsal_slice, stratified_split,
)
from build_v6_training_data import (  # noqa: E402
    _ENTRY_RE, _NOANS_PROMPTS, build_aug_seq, diversified_rehearsal,
)
from build_v2_training_data import make_arithmetic  # noqa: E402


# --- COVERAGE SLICES (2026-09-08) -------------------------------------------
# v6 produced an exceptionless pattern (D49, coverage analysis):
#   every TIME category WITH a dedicated training slice BEAT zero-shot
#     Counterfactual +13.1 (AUG_GLM, only 502 rows), Order_Compare +7.6,
#     Localization +5.3, Timeline +1.4 (AUG_SEQ, new in v6)
#   every category WITHOUT one LOST
#     Duration_Compare -7.3, Computation -6.3, Relative_Reasoning -2.1,
#     Extract -1.9, Co_temporality -1.8
# Coverage predicts the sign. The uncovered categories are 22,822 items
# (21.7% of TIME) costing -0.91pp, plus Computation's -0.56pp -- together
# -1.47pp, twice v6's entire margin. AUG_GLM shows a 502-row slice can move a
# category 13pp, so these are cheap.
#
# Extract (3,340 items, -0.06pp) is deliberately left uncovered: lowest
# leverage in the benchmark. Co_temporality is left alone because its only
# slice, AUG_NLI, is the degenerate one (D-ledger) and fixing it is a separate
# change.


def _spans(row: dict) -> list[tuple[str, int, int]]:
    """(label, first year, last year) for each parseable TLQA entry."""
    out = []
    for tgt in row.get("targets") or []:
        m = _ENTRY_RE.match(str(tgt).strip())
        if not m:
            continue
        years = [int(y) for y in re.findall(r"\d{3,4}", m.group("years"))]
        label = m.group("label").strip()
        if years and label:
            out.append((label, min(years), max(years)))
    return out


def build_aug_duration(pool: list[dict], n: int, rng: random.Random,
                       bench_qs: set[str]) -> list[dict]:
    """'Which lasted longer?' — targets Duration_Compare (9,217 items, -7.3pp).

    Built from TLQA year spans only. The two spans must differ in length or the
    question has no answer; a 1-year gap is required so the item is decidable
    from the text rather than a coin flip.
    """
    cand = []
    for r in pool:
        if r.get("source_dataset") != "TLQA":
            continue
        sp = [s for s in _spans(r) if s[2] > s[1]]
        if len(sp) >= 2:
            cand.append(sp)
    rng.shuffle(cand)
    out: list[dict] = []
    for sp in cand:
        if len(out) >= n:
            break
        a, b = rng.sample(sp, 2)
        da, db = a[2] - a[1], b[2] - b[1]
        if da == db:
            continue
        longer = a if da > db else b
        opts = [f"{a[0]} ({a[1]}-{a[2]})", f"{b[0]} ({b[1]}-{b[2]})"]
        rng.shuffle(opts)
        correct = f"{longer[0]} ({longer[1]}-{longer[2]})"
        q = ("Which of these lasted longer?\nChoices:\n"
             + "\n".join(f"{chr(ord('A') + i)}. {o}" for i, o in enumerate(opts)))
        if norm(q) in bench_qs:
            continue
        out.append({"source_dataset": "AUG_DURATION", "question": q, "context": "",
                    "targets": [correct], "source": "augmented"})
    return out


def build_aug_relative(pool: list[dict], n: int, rng: random.Random,
                       bench_qs: set[str]) -> list[dict]:
    """'What came immediately after X?' — targets Relative_Reasoning (10,265).

    Built from TLQA timelines with >= 3 entries at distinct start years, so
    "immediately after" is well defined and the distractors are real siblings
    from the same timeline rather than unrelated strings.
    """
    cand = []
    for r in pool:
        if r.get("source_dataset") != "TLQA":
            continue
        seen: dict[int, tuple[str, int, int]] = {}
        for s in _spans(r):
            seen.setdefault(s[1], s)
        if len(seen) >= 3:
            cand.append(sorted(seen.values(), key=lambda s: s[1]))
    rng.shuffle(cand)
    out: list[dict] = []
    for sp in cand:
        if len(out) >= n:
            break
        i = rng.randrange(len(sp) - 1)          # anchor, never the last
        anchor, correct = sp[i][0], sp[i + 1][0]
        others = [s[0] for j, s in enumerate(sp) if j not in (i, i + 1)]
        if len(others) < 2:
            continue
        opts = [correct] + rng.sample(others, 2)
        if len(set(opts)) < 3:
            continue
        rng.shuffle(opts)
        q = (f"Immediately after '{anchor}', which came next?\nChoices:\n"
             + "\n".join(f"{chr(ord('A') + i2)}. {o}" for i2, o in enumerate(opts)))
        if norm(q) in bench_qs:
            continue
        out.append({"source_dataset": "AUG_RELATIVE", "question": q, "context": "",
                    "targets": [correct], "source": "augmented"})
    return out


def build_noans_and_mcq_v7(pool: list[dict], n_noans: int, n_mcq: int,
                           rng: random.Random, bench_qs: set[str],
                           distractor_frac: float,
                           single_target: bool) -> list[dict]:
    """v6's generator plus the two fixes.

    FIX 1 (the shortcut). In v6, an abstain-phrased option appeared as a WRONG
    option in the answerable rows exactly ZERO times, so the mixture was
    perfectly separable on surface form: "an option phrased as an abstention is
    always the correct one". v6 duly scored 99.88% on TIME's no-answer bucket
    without reading anything. Here a fraction of the ANSWERABLE rows carry an
    abstain-phrased option as a plain distractor, so the phrasing stops
    predicting the label and the model has to check the context.

    FIX 2 (D45). Targets are single (option text only) unless --dual-target is
    passed, so the loader cannot join them into "<text>; <LETTER>".
    """
    src = [r for r in pool
           if r.get("source_dataset") == "TimeQA"
           and str(r.get("question", "")).strip()
           and [t for t in (r.get("targets") or []) if str(t).strip()]]
    rng.shuffle(src)

    def tkey(q: str) -> str:
        return " ".join(str(q).lower().split()[:2])

    by_type: dict[str, list[str]] = {}
    for r in src:
        a = "; ".join(str(t) for t in r["targets"] if str(t).strip())
        if a:
            by_type.setdefault(tkey(r["question"]), []).append(a)
    answers = ["; ".join(str(t) for t in r["targets"] if str(t).strip()) for r in src]

    out: list[dict] = []
    idx = 0
    n_injected = 0
    for kind, count in (("noans", n_noans), ("mcq", n_mcq)):
        made = 0
        while made < count and idx < len(src):
            r = src[idx]; idx += 1
            true = "; ".join(str(t) for t in r["targets"] if str(t).strip())
            same_type = by_type.get(tkey(r["question"]), [])
            distractors: list[str] = []
            guard = 0
            while len(distractors) < 3 and guard < 400:
                guard += 1
                pool_ = same_type if (same_type and guard <= 200) else answers
                c = pool_[rng.randrange(len(pool_))]
                if c and c != true and c not in distractors:
                    distractors.append(c)
            if len(distractors) < 3:
                continue

            if kind == "noans":
                opts = distractors + [rng.choice(_NOANS_PROMPTS)]
                correct = opts[-1]
            else:
                opts = distractors[:3] + [true]
                correct = true
                # FIX 1: sometimes an answerable row also offers an abstain
                # option -- as a WRONG one. Replaces a distractor rather than
                # adding a fifth option, so every row stays 4-way.
                if rng.random() < distractor_frac:
                    opts[rng.randrange(3)] = rng.choice(_NOANS_PROMPTS)
                    n_injected += 1

            rng.shuffle(opts)
            letter = chr(ord("A") + opts.index(correct))
            q = (str(r["question"]).strip() + "\nChoices:\n"
                 + "\n".join(f"{chr(ord('A') + i)}. {o}" for i, o in enumerate(opts)))
            if norm(q) in bench_qs:
                continue
            out.append({
                "source_dataset": "AUG_NOANS" if kind == "noans" else "AUG_MCQ2",
                "question": q,
                "context": str(r.get("context", "") or ""),
                # D45: single target by default -- the loader joins a LIST with
                # "; ", which is what taught "<option text>; B".
                "targets": [correct] if single_target else [correct, letter],
                "source": "augmented",
            })
            made += 1
    print(f"  abstain options injected as DISTRACTORS into answerable rows: {n_injected}")
    return out


def strip_dual_gold(rows: list[dict]) -> int:
    """D45: collapse the inherited AUG_MCQ dual gold to a single target."""
    n = 0
    for r in rows:
        if r.get("source_dataset") != "AUG_MCQ":
            continue
        tg = [str(t) for t in (r.get("targets") or [])]
        if len(tg) == 2 and len(tg[1].strip()) == 1 and tg[1].strip().upper() in "ABCD":
            r["targets"] = [tg[0]]
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v3-dir", default="data/combined_80_20_v3_fixed")
    ap.add_argument("--manual-aug", default="data/manual_aug")
    ap.add_argument("--letter-probe", default="data/letter_format/probe.jsonl")
    ap.add_argument("--out", default="data/combined_80_20_v7")
    ap.add_argument("--rehearsal", type=int, default=8000,
                    help="v5=8,026 (MCQ 64.49%%); v6=2,600 (MCQ 60.99%%). Restored.")
    ap.add_argument("--rehearsal-long-frac", type=float, default=0.15,
                    help="long-form dolly share. v6 used 0.50 and got the best "
                         "free-text score (27.25%%); 0.15 of 8,000 keeps 1,200 "
                         "long-form while restoring ~6,800 context-grounded rows.")
    ap.add_argument("--aug-seq", type=int, default=1500)
    ap.add_argument("--aug-noans", type=int, default=500)
    ap.add_argument("--aug-mcq2", type=int, default=6000,
                    help="the untested lever: TIME is 60.7%% MCQ, v6's mixture 8.6%%")
    ap.add_argument("--noans-distractor-frac", type=float, default=0.09,
                    help="fraction of ANSWERABLE rows that also offer an "
                         "abstain option, as a WRONG one. Breaks the shortcut "
                         "that gave v6 99.88%% on the no-answer bucket. TUNED: "
                         "the goal is P(correct | option is abstain-phrased) "
                         "~= 50%%, i.e. the phrasing carries NO information and "
                         "the model must read the context. 0.25 overshot to "
                         "25.5%% -- that merely INVERTS the cue (abstain "
                         "usually wrong) and would cause under-abstention on "
                         "TIME, where an offered abstain option is always "
                         "correct. 0.09 lands near 49%%.")
    ap.add_argument("--aug-arith", type=int, default=2500,
                    help="Computation coverage. v1-v6 carried only 400 AUG_ARITH "
                         "rows and v6 scored 12.5%% vs zero-shot's 18.8%% on TIME's "
                         "9,372 Computation items — of which only 0.8%% are "
                         "extractable from the context, i.e. they MUST be "
                         "computed. Synthetic and pool-free, so no leakage risk.")
    ap.add_argument("--aug-duration", type=int, default=2000,
                    help="Duration_Compare coverage (9,217 items; v6 33.1%% vs "
                         "zero-shot 40.4%%, the worst regression in the arm).")
    ap.add_argument("--aug-relative", type=int, default=2000,
                    help="Relative_Reasoning coverage (10,265 items; v6 44.9%% vs "
                         "zero-shot 47.0%%).")
    ap.add_argument("--dual-target", action="store_true",
                    help="restore v6's [text, LETTER] gold. NOT recommended (D45).")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    single = not args.dual_target
    if args.dual_target:
        print("WARNING: --dual-target restores the '<text>; B' training shape that "
              "D45 exists to remove. The trailing-letter scoring rule will be "
              "load-bearing again.")

    rng = random.Random(args.seed)
    bench_qs = benchmark_questions()

    v3 = [json.loads(l) for l in open(Path(args.v3_dir) / "train.jsonl", encoding="utf-8")]
    v3 += [json.loads(l) for l in open(Path(args.v3_dir) / "val.jsonl", encoding="utf-8")]
    v3_rehearsal = [r for r in v3 if r["source_dataset"] == "REHEARSAL"]
    keep = [r for r in v3 if r["source_dataset"] != "REHEARSAL"]
    keep, n_nli = filter_nli_neutral(keep)
    n_stripped = strip_dual_gold(keep) if single else 0
    print(f"v3-corrected core: {len(keep)} ({n_nli} AUG_NLI Neutral dropped)")
    print(f"D45: AUG_MCQ rows collapsed to a single target: {n_stripped}")

    manual = []
    for p in sorted(Path(args.manual_aug).glob("glm_batch*.jsonl")):
        manual += [json.loads(l) for l in open(p, encoding="utf-8")]

    seq = build_aug_seq(keep, args.aug_seq, rng, bench_qs)
    print(f"AUG_SEQ: {len(seq)} (v6's slice, unchanged — it worked)")

    mcq = build_noans_and_mcq_v7(keep, args.aug_noans, args.aug_mcq2, rng,
                                 bench_qs, args.noans_distractor_frac, single)
    print(f"AUG_NOANS + AUG_MCQ2: {dict(Counter(r['source_dataset'] for r in mcq))}")

    # --- coverage slices: every uncovered category lost in v6 (see the module
    # comment). Each targets one category and is built from TLQA year spans or
    # is fully synthetic, so none of them touches a benchmark.
    dur = build_aug_duration(keep, args.aug_duration, rng, bench_qs)
    rel = build_aug_relative(keep, args.aug_relative, rng, bench_qs)
    extra_arith = make_arithmetic(args.aug_arith, rng)
    print(f"AUG_DURATION: {len(dur)} (Duration_Compare — v6 was -7.3pp)")
    print(f"AUG_RELATIVE: {len(rel)} (Relative_Reasoning — v6 was -2.1pp)")
    print(f"AUG_ARITH   : +{len(extra_arith)} on top of the core's 400 "
          f"(Computation — v6 was -6.3pp)")

    print(f"rehearsal (target {args.rehearsal}, {args.rehearsal_long_frac:.0%} long-form):")
    reh = diversified_rehearsal(v3_rehearsal, args.rehearsal,
                                args.rehearsal_long_frac, rng, bench_qs)
    ctx_grounded = sum(1 for r in reh if r["source"] != "databricks/databricks-dolly-15k")
    print(f"  total {len(reh)}  (context-grounded {ctx_grounded} — v5 had 8,026, v6 had 1,300)")

    probe_q = {norm(json.loads(l)["question"])
               for l in open(args.letter_probe, encoding="utf-8")}

    all_records = keep + manual + seq + mcq + dur + rel + extra_arith + reh
    for r in all_records:
        if r["source_dataset"] == "TLQA" and r.get("targets"):
            r["final_answers"] = list(r["targets"])

    all_q = [norm(str(r.get("question") or "")) for r in all_records]
    assert sum(q in probe_q for q in all_q) == 0, "FATAL: probe leak"
    n_coll = sum(q in bench_qs for q in all_q)
    assert n_coll == 0, f"FATAL: {n_coll} benchmark collisions"
    seen: set[str] = set()
    deduped = []
    for r, q in zip(all_records, all_q):
        if q in seen:
            continue
        seen.add(q); deduped.append(r)
    print(f"guards passed (TIME+TimeBench+TRAM): 0 leaks, 0 collisions; "
          f"deduped {len(all_records) - len(deduped)}")
    all_records, n_near = purge_near_duplicates(deduped)
    print(f"near-dup purge: dropped {n_near}")

    train, val = stratified_split(all_records, args.val_frac, rng)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
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
        "cycle": "v7",
        "variables_vs_v6": [
            f"rehearsal 2600 -> {len(reh)} ({ctx_grounded} context-grounded)",
            f"AUG_MCQ2 1500 -> {sum(1 for r in mcq if r['source_dataset']=='AUG_MCQ2')}",
            f"single-target MCQ (D45): {single}",
            f"abstain-as-distractor frac: {args.noans_distractor_frac}",
            f"COVERAGE: AUG_DURATION +{len(dur)}, AUG_RELATIVE +{len(rel)}, "
            f"AUG_ARITH +{len(extra_arith)}",
        ],
        "train": mix(train), "val": mix(val),
        "near_dup_dropped": n_near, "aug_mcq_dual_gold_stripped": n_stripped,
        "answer_style_train": answer_style(train),
        "seed": args.seed, "args": vars(args),
    }
    (out / "manifest.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
