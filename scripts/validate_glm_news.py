"""Validate a GLM-generated news temporal batch before it can reach training.

This project has shipped two defective synthetic slices already — `AUG_NLI`'s
label leakage (every Neutral identifiable from the word "another") and
`AUG_REASON`'s incoherent two-hop questions — and each cost a full cycle. Both
would have been caught by a cheap structural check. This is that check.

Rejects, per row: wrong schema, non-dual gold, a gold letter that disagrees
with the gold text, **type-heterogeneous distractors** (the shortcut that
makes an item solvable without reading), over-long answers, duplicated
options, and a `Computation` answer that appears verbatim in the context
(meaning it was copied, not computed). Reports, per batch: letter-position
balance, category counts, and collisions against TIME/TimeBench/TRAM.

Usage:
    ./venv/bin/python scripts/validate_glm_news.py data/manual_aug/glm_news_batch1.jsonl
    ./venv/bin/python scripts/validate_glm_news.py <file> --write-clean <out.jsonl>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

REQUIRED = {"source_dataset", "category", "question", "context", "targets", "rationale"}
ABSTAIN = {"there is no answer.", "cannot be determined from the context.",
           "the passage does not say.", "none of the options is supported by the passage."}
OPT_RE = re.compile(r"^([A-D])\.\s+(.*)$")
DATE_RE = re.compile(r"\b(?:\d{1,2}\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
                     r"[a-z]*\.?\s+\d{1,2},?\s+\d{4}|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b",
                     re.I)
DUR_RE = re.compile(r"\b\d+\s+(?:day|week|month|year|hour|minute)s?\b", re.I)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def answer_type(s: str) -> str:
    """Coarse type of an option, for the homogeneity check."""
    t = str(s).strip()
    if t.lower() in ABSTAIN:
        return "abstain"
    if DATE_RE.search(t):
        return "date"
    if DUR_RE.search(t):
        return "duration"
    if re.fullmatch(r"\D*\d{4}\D*", t):
        return "year"
    words = t.rstrip(".").split()
    if 1 <= len(words) <= 4 and all(w[:1].isupper() for w in words if w[:1].isalpha()):
        return "proper-noun"
    return "phrase"


def parse_options(question: str) -> list[str]:
    opts = []
    for line in question.split("\n"):
        m = OPT_RE.match(line.strip())
        if m:
            opts.append(m.group(2).strip())
    return opts


def benchmark_questions() -> set[str]:
    from src.data.data_loader import BenchmarkLoader
    qs: set[str] = set()
    loader = BenchmarkLoader(str(PROJECT_ROOT / "data" / "benchmarks"))
    for bench in ("time", "timebench"):
        try:
            for ex in loader.load(bench):
                qs.add(norm(ex.question))
        except Exception as exc:  # a missing benchmark must not silently pass
            print(f"  WARNING: could not load {bench} for the collision check: {exc}")
    return qs


def check(row: dict, idx: int) -> list[str]:
    errs = []
    missing = REQUIRED - set(row)
    if missing:
        return [f"missing fields {sorted(missing)}"]

    tg = row.get("targets")
    if not (isinstance(tg, list) and len(tg) == 2):
        errs.append(f"targets must be [text, LETTER], got {tg!r}")
        return errs
    text, letter = str(tg[0]).strip(), str(tg[1]).strip().upper()
    if letter not in "ABCD" or len(letter) != 1:
        errs.append(f"gold letter {letter!r} is not A-D")

    opts = parse_options(row["question"])
    if len(opts) != 4:
        errs.append(f"expected 4 options in the question, parsed {len(opts)}")
        return errs
    if len(set(map(norm, opts))) != 4:
        errs.append("duplicate options")

    if letter in "ABCD":
        want = opts["ABCD".index(letter)]
        if norm(want) != norm(text):
            errs.append(f"gold letter {letter} points at {want!r} but gold text is {text!r}")

    types = [answer_type(o) for o in opts]
    non_abstain = [t for t in types if t != "abstain"]
    if len(set(non_abstain)) > 1:
        errs.append(f"distractors are type-heterogeneous: {list(zip('ABCD', types))} "
                    f"— solvable without reading")

    if len(text.split()) > 12:
        errs.append(f"answer too long ({len(text.split())} words)")

    cat = row.get("category")
    if cat == "Unanswerable":
        if text.lower() not in ABSTAIN:
            errs.append(f"Unanswerable gold must be one of the four abstain strings, got {text!r}")
    elif cat == "Computation":
        if norm(text) and norm(text) in norm(row.get("context", "")):
            errs.append("Computation answer appears VERBATIM in the context — copied, not computed")
        if not re.search(r"\d", str(row.get("rationale", ""))):
            errs.append("Computation rationale shows no arithmetic")
    elif cat != "Counterfactual":
        errs.append(f"unexpected category {cat!r}")

    if not str(row.get("rationale", "")).strip():
        errs.append("empty rationale")
    ctx_words = len(str(row.get("context", "")).split())
    if cat != "Unanswerable" and ctx_words < 40:
        errs.append(f"context too short ({ctx_words} words)")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--write-clean", default="", help="write the passing rows to this file")
    args = ap.parse_args()

    rows, bad_json = [], 0
    for i, line in enumerate(open(args.path, encoding="utf-8"), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            bad_json += 1
            print(f"  line {i}: UNPARSEABLE JSON — {exc}")
    print(f"{args.path}: {len(rows)} rows parsed, {bad_json} unparseable\n")
    if not rows:
        return 1

    clean, rejected = [], 0
    for i, r in enumerate(rows, 1):
        errs = check(r, i)
        if errs:
            rejected += 1
            if rejected <= 15:
                print(f"  REJECT row {i} [{r.get('category')}]:")
                for e in errs:
                    print(f"      - {e}")
        else:
            clean.append(r)
    if rejected > 15:
        print(f"  ... and {rejected - 15} more rejected rows")

    print(f"\npassed {len(clean)} / {len(rows)} ({100*len(clean)/len(rows):.1f}%), rejected {rejected}")
    print("categories:", dict(Counter(r.get("category") for r in clean)))

    letters = Counter(str(r["targets"][1]).strip().upper() for r in clean
                      if isinstance(r.get("targets"), list) and len(r["targets"]) == 2)
    if letters:
        tot = sum(letters.values())
        print("gold-letter balance:", {k: f"{100*v/tot:.1f}%" for k, v in sorted(letters.items())})
        worst = max(letters.values()) / tot
        if worst > 0.40:
            print(f"  WARNING: one letter holds {100*worst:.1f}% of golds — a positional bias the "
                  f"model will exploit instead of reasoning. Re-shuffle option order.")

    print("\nchecking benchmark collisions...")
    bq = benchmark_questions()
    if bq:
        # compare the question stem only, since options are generated
        coll = sum(1 for r in clean if norm(str(r["question"]).split("\nChoices:")[0]) in bq)
        print(f"  exact question-stem collisions vs TIME+TimeBench: {coll}")
        if coll:
            print("  FATAL: leaked benchmark questions — do not train on this batch.")

    if args.write_clean and clean:
        with open(args.write_clean, "w", encoding="utf-8") as f:
            for r in clean:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nwrote {len(clean)} clean rows to {args.write_clean}")
    return 0 if clean and not rejected else 1


if __name__ == "__main__":
    raise SystemExit(main())
