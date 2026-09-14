"""Build the v7-corrected mixture — v6's mixture, one variable changed.

COPY OF `build_v6_training_data.py` (2026-09-09), so v6 stays reproducible
from its own untouched script. Do not edit the v6 builder to reproduce this
arm; diff the two files to see the whole variable.

THE ONE CHANGE vs v6
--------------------
`AUG_MCQ` and `AUG_MCQ2` get a SINGLE option-text target instead of the dual
gold `[text, LETTER]`. The finetuning loader joins a dual gold into the
training target `"<option text>; B"`, which teaches an output shape the
benchmark never asks for, and is the ONLY reason the scorer needs a
trailing-letter rule at all (D39). That invites the fair objection: *you
trained a nonstandard format, then widened the scorer to accept it.*

Everything else is v6 verbatim — every slice size, the seed, the recipe — so
the delta against v6 is attributable to this change and nothing else. This is
the correction to v7, which made the same change but bundled it with five
other edits (mixture 14,850 -> 26,570, synthetic share 30.7% -> 45.0%,
optimizer steps 2,787 -> 4,983) and lost 5.94pp on TIME with no way to say
which change caused it. See `docs/v7_corrected_plan.md` and D56/D57.

v7 DID achieve the methodological goal: 0 trailing-letter rewrites vs v5's
22,870. This arm re-tests that on a base that actually works (v6 is the only
arm beating zero-shot on TIME: 42.19% vs 41.46%).

---- v6's original docstring follows ----

v6 targets the ONLY three things still separating the best fine-tuned arm
from zero-shot on TIME, measured 2026-09-07 after the trailing-letter scoring
fix (docs/v6_plan.md, D39):

    bucket            share   zero-shot   v5        costs v5
    MCQ answerable    44.7%   63.56%      64.50%    +0.42pp   <- v5 already wins
    MCQ "no answer"    2.3%   52.95%       0.74%    -1.21pp   <- AUG_NOANS
    letter sequence   10.8%   19.05%       0.00%    -2.06pp   <- AUG_SEQ
    free text         40.4%   24.18%      23.71%    -0.19pp   <- rehearsal cap

Three changes vs v5, all data-only:

1. REHEARSAL capped 8,026 -> 2,600 (44.2% -> ~20% of the mixture, v4's
   measured-better point) and diversified: half long-form dolly, half short
   extractive span QA. v5 proved 44% short-span rehearsal collapses sequence
   emission; the v2 -> v3-corrected step proved relieving that pressure
   restores it (D38).
2. NEW AUG_SEQ — the comma-separated-letter ordering shape. No cycle has ever
   contained a single example of it (D38 census) and v5 emits 1 in 11,361.
   Built ONLY from TLQA training-pool timelines, never from a benchmark.
3. NEW AUG_NOANS — willingness to select an offered "no answer" option when
   the context supports none of the choices. Paired with extra answerable
   rows from the same generator (AUG_MCQ2) so the mixture cannot simply learn
   to always abstain, which would cost far more on the 44.7% answerable
   bucket than abstention can win on the 2.3% one.

METHODOLOGY. Every new row is generated from the TimeQA/TLQA training pool.
No benchmark item, answer or question is read. Instruction wording for
AUG_SEQ is paraphrased across several templates rather than copying the
benchmark's prompt string, so what is learned is the general
"order chronologically -> emit comma-separated letters" convention rather
than one memorised prompt. AUG_SEQ facts carry explicit years, so it teaches
OUTPUT FORMAT, not news chronology -- see docs/v6_plan.md, which says so in
the terms the thesis must use.

Usage:
    ./venv/bin/python scripts/build_v6_training_data.py
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
    REHEARSAL_POOLS,
    answer_style,
    benchmark_questions,
    filter_nli_neutral,
    norm,
    purge_near_duplicates,
    rehearsal_slice,
    stratified_split,
)

# TLQA target entries look like "Governor of Kansas (2011, 2012, 2013)".
_ENTRY_RE = re.compile(r"^(?P<label>.+?)\s*\((?P<years>\s*\d{3,4}(?:\s*,\s*\d{3,4})*\s*)\)\s*$")

# Paraphrases, not the benchmark's wording. The output convention
# (uppercase letters, comma separated, nothing else) is the point.
_SEQ_PROMPTS = [
    "Below are {n} facts. Put them in chronological order, earliest first. "
    "Answer with the option letters separated by commas and nothing else, for example 'A,B,C'.",
    "Sort the following {n} facts from earliest to latest. "
    "Reply with only the letters in order, comma separated, e.g. 'B,A,C'.",
    "Arrange these {n} events in the order they happened, starting with the earliest. "
    "Output nothing but the letters separated by commas, such as 'C,A,B'.",
    "Order the {n} statements below chronologically. "
    "Your entire answer must be the option letters in order, separated by commas.",
]

_NOANS_PROMPTS = [
    "There is no answer.",
    "None of the options is supported by the passage.",
    "The passage does not say.",
    "Cannot be determined from the context.",
]


def _entries(row: dict) -> list[tuple[str, int]]:
    """(label, earliest year) for each parseable TLQA timeline entry."""
    out: list[tuple[str, int]] = []
    for t in row.get("targets") or []:
        m = _ENTRY_RE.match(str(t).strip())
        if not m:
            continue
        years = [int(y) for y in re.findall(r"\d{3,4}", m.group("years"))]
        label = m.group("label").strip()
        if years and label:
            out.append((label, min(years)))
    return out


def build_aug_seq(pool: list[dict], n: int, rng: random.Random,
                  bench_qs: set[str], n_facts: int = 3) -> list[dict]:
    """Chronological-ordering rows in TIME's answer shape, from TLQA only.

    Each fact carries its own year, so the ordering is decidable from the
    text. This slice exists to teach the OUTPUT FORMAT that no cycle has ever
    contained -- not to teach event chronology, which no arm in this project
    does better than ~3pp over chance.
    """
    cand = []
    for r in pool:
        if r.get("source_dataset") != "TLQA":
            continue
        es = _entries(r)
        # distinct years, or the gold order would be ambiguous
        seen: dict[int, tuple[str, int]] = {}
        for label, y in es:
            seen.setdefault(y, (label, y))
        if len(seen) >= n_facts:
            cand.append(sorted(seen.values(), key=lambda e: e[1]))
    rng.shuffle(cand)

    out: list[dict] = []
    for es in cand:
        if len(out) >= n:
            break
        chosen = sorted(rng.sample(es, n_facts), key=lambda e: e[1])
        facts = [f"{label} began in {year}." for label, year in chosen]
        order = list(range(n_facts))
        rng.shuffle(order)                      # order[i] = which fact is option i
        letters = [chr(ord("A") + i) for i in range(n_facts)]
        shown = [facts[order[i]] for i in range(n_facts)]
        # gold: the letters of the facts in chronological (index) order
        pos_of = {order[i]: i for i in range(n_facts)}
        gold = ",".join(letters[pos_of[k]] for k in range(n_facts))
        prompt = rng.choice(_SEQ_PROMPTS).format(n=n_facts)
        q = prompt + "\n" + "\n".join(f"{letters[i]}. {shown[i]}" for i in range(n_facts))
        if norm(q) in bench_qs:
            continue
        out.append({
            "source_dataset": "AUG_SEQ",
            "question": q,
            "context": "",
            "targets": [gold],          # single element: the loader joins with "; "
            "source": "augmented",
        })
    return out


def build_noans_and_mcq(pool: list[dict], n_noans: int, n_mcq: int,
                        rng: random.Random, bench_qs: set[str]) -> list[dict]:
    """Four-option MCQ rows from TimeQA: some answerable, some not.

    The unanswerable ones offer three distractors plus an explicit no-answer
    option, so the correct response is to pick that option. The answerable
    ones come from the same generator and outnumber them, because a mixture
    that only ever teaches abstention would trade the 44.7% answerable bucket
    for the 2.3% one -- a losing trade by a factor of ~18.
    """
    src = [r for r in pool
           if r.get("source_dataset") == "TimeQA"
           and str(r.get("question", "")).strip()
           and [t for t in (r.get("targets") or []) if str(t).strip()]]
    rng.shuffle(src)

    # Distractors must be TYPE-PLAUSIBLE or the task is solvable by a shallow
    # cue. Sampling answers globally gives "Who coached Atletico Madrid?" the
    # options "Vancouver Whitecaps" and "F/A-18 Hornet", which teaches
    # "the options look unrelated -> abstain" rather than "the context does
    # not support any option -> abstain". TIME's distractors are plausible, so
    # that shortcut would not transfer. Bucket by the question's leading
    # words, which in TimeQA encode the answer type ("who coached",
    # "where did", "what organization ..."), and draw within the bucket.
    def tkey(q: str) -> str:
        return " ".join(str(q).lower().split()[:2])

    by_type: dict[str, list[str]] = {}
    for r in src:
        a = "; ".join(str(t) for t in r["targets"] if str(t).strip())
        if a:
            by_type.setdefault(tkey(r["question"]), []).append(a)
    answers = ["; ".join(str(t) for t in r["targets"] if str(t).strip()) for r in src]

    out: list[dict] = []
    want = [("noans", n_noans), ("mcq", n_mcq)]
    idx = 0
    for kind, count in want:
        made = 0
        while made < count and idx < len(src):
            r = src[idx]; idx += 1
            true = "; ".join(str(t) for t in r["targets"] if str(t).strip())
            same_type = by_type.get(tkey(r["question"]), [])
            distractors: list[str] = []
            guard = 0
            # Same-type first; fall back to the global pool only if this
            # question's type bucket is too thin to fill three slots.
            while len(distractors) < 3 and guard < 400:
                guard += 1
                src_pool = same_type if (same_type and guard <= 200) else answers
                c = src_pool[rng.randrange(len(src_pool))]
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
                # *** THE v7-CORRECTED VARIABLE (was [correct, letter] in v6) ***
                # A dual gold is joined by the loader into "<text>; <LETTER>",
                # teaching an output format the benchmark never asks for — the
                # ONLY reason the scorer needs a trailing-letter rule, and the
                # basis of the fair objection "you trained a nonstandard
                # format, then widened the scorer to accept it". A single
                # option-text target makes the model answer in the benchmark's
                # own form. Verify after the run by counting
                # _TRAILING_LETTER_RE firings (the `rewritten` column in
                # rescore_v5_protocol.py): ~0 firings plus a good score proves
                # the result does not depend on the scoring change. v7 already
                # showed this works (0 rewrites vs v5's 22,870); it just
                # bundled five other changes and lost 5.94pp.
                # `letter` is still computed above and used to build the
                # option list — only the TARGET drops it.
                "targets": [correct],
                "source": "augmented",
            })
            made += 1
    return out


def diversified_rehearsal(v3_rehearsal: list[dict], n_total: int, long_frac: float,
                          rng: random.Random, bench_qs: set[str]) -> list[dict]:
    """Half long-form (dolly, 18.47w), half short extractive span QA.

    v3-corrected -- still the reference arm -- had ONLY long-form rehearsal at
    17.2%. v4 and v5 had only short-span. The two were never separated; this
    slice holds the volume at v4's better-measured ~20% and restores the
    answer-length diversity that both later cycles lost.
    """
    n_long = int(round(n_total * long_frac))
    n_short = n_total - n_long
    longs = [dict(r) for r in v3_rehearsal]
    rng.shuffle(longs)
    if n_long > len(longs):
        raise SystemExit(f"FATAL: only {len(longs)} long-form rehearsal rows available, "
                         f"{n_long} requested.")
    out = longs[:n_long]
    out += rehearsal_slice(REHEARSAL_POOLS, n_short, rng, bench_qs)
    rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v3-dir", default="data/combined_80_20_v3_fixed")
    ap.add_argument("--manual-aug", default="data/manual_aug")
    ap.add_argument("--letter-probe", default="data/letter_format/probe.jsonl")
    ap.add_argument("--out", default="data/combined_80_20_v7_corrected")
    ap.add_argument("--rehearsal", type=int, default=2600,
                    help="total rehearsal rows (v4=2,676 ~ 20.9%%; v5=8,026 ~ 44.2%%)")
    ap.add_argument("--rehearsal-long-frac", type=float, default=0.5,
                    help="fraction of rehearsal drawn from long-form dolly")
    ap.add_argument("--aug-seq", type=int, default=1500)
    ap.add_argument("--aug-noans", type=int, default=500)
    ap.add_argument("--aug-mcq2", type=int, default=1500,
                    help="answerable counterweight to --aug-noans; keep it >= 3x")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.aug_noans and args.aug_mcq2 < 3 * args.aug_noans:
        print(f"WARNING: --aug-mcq2 ({args.aug_mcq2}) is under 3x --aug-noans "
              f"({args.aug_noans}). Over-abstention costs the 44.7% answerable "
              f"bucket ~18x what the 2.3% no-answer bucket can pay back.")

    rng = random.Random(args.seed)
    bench_qs = benchmark_questions()

    v3 = [json.loads(l) for l in open(Path(args.v3_dir) / "train.jsonl", encoding="utf-8")]
    v3 += [json.loads(l) for l in open(Path(args.v3_dir) / "val.jsonl", encoding="utf-8")]
    v3_rehearsal = [r for r in v3 if r["source_dataset"] == "REHEARSAL"]
    keep = [r for r in v3 if r["source_dataset"] != "REHEARSAL"]
    keep, n_nli = filter_nli_neutral(keep)
    print(f"v3-corrected core (REHEARSAL excluded, {n_nli} AUG_NLI Neutral dropped): {len(keep)}")
    print(f"long-form dolly rehearsal available: {len(v3_rehearsal)}")

    manual = []
    for p in sorted(Path(args.manual_aug).glob("glm_batch*.jsonl")):
        manual += [json.loads(l) for l in open(p, encoding="utf-8")]
    print(f"AUG_GLM manual: {len(manual)}")

    seq = build_aug_seq(keep, args.aug_seq, rng, bench_qs)
    print(f"AUG_SEQ built: {len(seq)} (requested {args.aug_seq})")
    if len(seq) < args.aug_seq:
        print(f"  NOTE: TLQA supplied fewer 3-distinct-year timelines than requested.")

    mcq = build_noans_and_mcq(keep, args.aug_noans, args.aug_mcq2, rng, bench_qs)
    print(f"AUG_NOANS + AUG_MCQ2 built: {Counter(r['source_dataset'] for r in mcq)}")

    print(f"rehearsal (target {args.rehearsal}, {args.rehearsal_long_frac:.0%} long-form):")
    reh = diversified_rehearsal(v3_rehearsal, args.rehearsal,
                                args.rehearsal_long_frac, rng, bench_qs)
    print(f"  total rehearsal: {len(reh)}")

    probe_q = {norm(json.loads(l)["question"])
               for l in open(args.letter_probe, encoding="utf-8")}

    all_records = keep + manual + seq + mcq + reh

    # Second half of the v7-corrected variable. The line changed above only
    # covers AUG_MCQ2, which this script BUILDS. `AUG_MCQ` (400 rows) is
    # INHERITED from the v3-corrected core and carries its own dual gold
    # ['Lazio', 'A'] — missing it would leave the "<text>; <LETTER>" shape in
    # the mixture and silently invalidate the whole arm. Applied here, after
    # every slice is assembled, so both are covered.
    n_stripped = 0
    for r in all_records:
        if r.get("source_dataset") not in ("AUG_MCQ", "AUG_MCQ2"):
            continue
        tg = list(r.get("targets") or [])
        text_only = [t for t in tg
                     if str(t).strip().upper() not in ("A", "B", "C", "D")]
        if text_only and len(text_only) != len(tg):
            r["targets"] = text_only
            n_stripped += 1
    print(f"v7-corrected variable: stripped the bare letter from {n_stripped} "
          f"inherited AUG_MCQ targets (AUG_MCQ2 was built single-target above)")

    for r in all_records:
        if r["source_dataset"] == "TLQA" and r.get("targets"):
            r["final_answers"] = list(r["targets"])

    all_q = [norm(str(r.get("question") or "")) for r in all_records]
    n_leak = sum(q in probe_q for q in all_q)
    assert n_leak == 0, f"FATAL: {n_leak} probe questions leaked into training"
    n_coll = sum(q in bench_qs for q in all_q)
    assert n_coll == 0, f"FATAL: {n_coll} benchmark collisions"
    seen: set[str] = set()
    deduped = []
    for r, q in zip(all_records, all_q):
        if q in seen:
            continue
        seen.add(q)
        deduped.append(r)
    print(f"guards passed: 0 probe leaks, 0 benchmark collisions; deduped "
          f"{len(all_records) - len(deduped)} rows")
    all_records, n_near = purge_near_duplicates(deduped)
    print(f"near-dup purge: dropped {n_near} rows")

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
        "cycle": "v6",
        "variables_vs_v5": [
            f"rehearsal 8026 -> {len(reh)} rows and {args.rehearsal_long_frac:.0%} long-form",
            f"AUG_SEQ +{len(seq)} (new: comma-separated-letter ordering shape)",
            f"AUG_NOANS +{sum(1 for r in mcq if r['source_dataset'] == 'AUG_NOANS')} "
            f"/ AUG_MCQ2 +{sum(1 for r in mcq if r['source_dataset'] == 'AUG_MCQ2')}",
        ],
        "train": mix(train), "val": mix(val),
        "rehearsal_by_source": dict(Counter(r["source"] for r in reh)),
        "near_dup_dropped": n_near,
        "answer_style_train": answer_style(train),
        "seed": args.seed, "args": vars(args),
    }
    (out / "manifest.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
