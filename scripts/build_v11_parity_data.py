"""Build the v11 mixture: v10-glm's mixture minus two verified defects.

1. CONTAMINATION (audit 2026-09-23 §D1). build_v5's purge only compares
   questions (Jaccard > 0.8) and whole-context token sets, so paraphrased
   TimeQA questions over the SAME Wikipedia document as a TimeBench test item
   pass it: 266 train/val rows shared a passage with TimeBench timeqa /
   tempreason items, and for 298 test items the training row even carries the
   identical gold. Every arm since v6 trained on them. Here ANY row whose
   context (or question) shares a run of >= 30 whitespace tokens with ANY
   TIME / TimeBench / TRAM test context is removed, from train AND val.

   Exact method: benchmark contexts are indexed as 25-token shingles at
   stride 6; every 25-token shingle of a training row is looked up. A shared
   run of >= 30 tokens (25 + 6 - 1) is therefore guaranteed to hit.

2. LABEL SPACE (audit 2026-09-23 §D2). AUG_GLM2 `relation` rows with gold
   DURING / IDENTITY and `ordering` rows with gold Undetermined teach answers
   that are outside TRAM's gold label space for those tasks (they occur there
   only as distractor options), i.e. they train the model to pick a wrong
   option. Removed. Label FREQUENCIES are deliberately not matched to TRAM's
   test distribution -- that would be tuning a prior on the test set.

Usage: venv/bin/python scripts/build_v11_parity_data.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import BenchmarkLoader  # noqa: E402

SRC = PROJECT_ROOT / "data" / "combined_80_20_v10_glm"
OUT = PROJECT_ROOT / "data" / "combined_80_20_v11"
K, STRIDE, MIN_RUN = 25, 6, 30
_TOK = re.compile(r"\w+")

BAD_LABELS = {"relation": {"during", "identity"}, "ordering": {"undetermined"}}


def toks(text: str) -> list[str]:
    return _TOK.findall((text or "").lower())


TEMPLATE_MIN_ITEMS = 20


def _norm_q(q: str) -> str:
    return " ".join(toks(q))


def bench_index() -> tuple[set[int], set[int], set[str]]:
    """Shingles of every benchmark context and question, minus TEMPLATES.

    A question shingle that recurs in >= TEMPLATE_MIN_ITEMS distinct benchmark
    questions is an instruction template (TIME's Timeline "Below are 3 facts
    ... Requirements: You must output a sequence of uppercase letters ..." is
    > 30 tokens and appears 13,159 times), not content. AUG_GLM2 questions
    reproduce those templates on purpose; counting them flagged 1,016 invented
    rows as contaminated on the first run of this script, every one a template
    hit (verified: context-vs-context matching found 0 AUG_GLM2 rows).
    """
    idx: set[int] = set()
    q_count: Counter = Counter()
    q_exact: set[str] = set()
    loader = BenchmarkLoader(data_dir=str(PROJECT_ROOT / "data" / "benchmarks"))
    def shingles(text: str) -> set[int]:
        t = toks(text)
        return {hash(tuple(t[i:i + K]))
                for i in range(0, max(0, len(t) - K + 1), STRIDE)}

    for bench in ("time", "timebench", "tram"):
        contexts: set[str] = set()
        questions: Counter = Counter()  # item multiplicity, NOT distinct texts
        for ex in loader.load(bench):
            if ex.context:
                contexts.add(ex.context)
            if ex.question:
                questions[ex.question] += 1
        for text in contexts:
            idx |= shingles(text)
        for text, mult in questions.items():
            q_exact.add(_norm_q(text))
            for h in shingles(text):
                q_count[h] += mult
        print(f"indexed {bench}: {len(idx):,} context shingles so far", flush=True)
    templates = {h for h, n in q_count.items() if n >= TEMPLATE_MIN_ITEMS}
    q_idx = {h for h in q_count if h not in templates}
    print(f"question shingles: {len(q_count):,}, templates excluded: {len(templates):,}")
    return idx, q_idx, q_exact


def _hits(text: str, *indexes: set[int]) -> bool:
    t = toks(text)
    return any(hash(tuple(t[i:i + K])) in ix
               for i in range(0, len(t) - K + 1) for ix in indexes)


def contaminated(row: dict, ctx_idx: set[int], q_idx: set[int],
                 q_exact: set[str]) -> bool:
    """The three real leakage paths, and NOT question-vs-question shingles.

    - training context  vs benchmark context OR question (TimeDial, TRAM
      relation etc. carry their passage in the question field);
    - training question vs benchmark context;
    - exact normalised question match.
    Training question vs benchmark question SHINGLES is deliberately not a
    path: AUG_GLM2 questions reproduce benchmark instructions on purpose, and
    a template variant one word longer than the template ("... season Hint:
    Please answer in the form of ...") slips under any multiplicity
    threshold -- the second run of this script flagged 17 such Computation
    rows, all false positives, before this split.
    """
    ctx, q = row.get("context") or "", row.get("question") or ""
    return (_hits(ctx, ctx_idx, q_idx) or _hits(q, ctx_idx)
            or _norm_q(q) in q_exact)


def bad_label(row: dict) -> bool:
    bad = BAD_LABELS.get(row.get("category"))
    if not bad or row.get("source_dataset") != "AUG_GLM2":
        return False
    tg = row.get("targets") or []
    return bool(tg) and str(tg[0]).strip().lower() in bad


def main() -> int:
    idx = bench_index()  # (ctx_idx, q_idx, q_exact)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"source": str(SRC.relative_to(PROJECT_ROOT)),
                      "shingle": {"k": K, "stride": STRIDE, "guaranteed_run": MIN_RUN,
                                  "template_min_items": TEMPLATE_MIN_ITEMS},
                      "splits": {}}
    for split in ("train", "val"):
        rows = [json.loads(l) for l in open(SRC / f"{split}.jsonl", encoding="utf-8")]
        kept, why = [], Counter()
        by_src = Counter()
        for r in rows:
            if contaminated(r, *idx):
                why[f"contaminated:{r.get('source_dataset')}"] += 1
                continue
            if bad_label(r):
                why[f"label_space:{r.get('category')}"] += 1
                continue
            kept.append(r)
            by_src[r.get("source_dataset")] += 1
        with open(OUT / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest["splits"][split] = {"in": len(rows), "out": len(kept),
                                     "removed": dict(why), "by_source": dict(by_src)}
        print(split, manifest["splits"][split])
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
