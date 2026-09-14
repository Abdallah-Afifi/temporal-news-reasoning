"""TRAM audit — structural sanity + leakage vs the current training pool.

Read-only: loads TRAM (980,918 rows across 11 tasks) and the frozen v4
training pool, and reports (a) schema/structural health per task and
(b) contamination between them. Does NOT run any model, does NOT queue
any eval. Mirrors the exact-match + token-set-Jaccard near-dup method
already validated for TIME/TimeBench (D27/D28, scripts/build_v4_training_data.py
:: purge_near_duplicates), scaled to TRAM's size with a rare-token blocking
index so the near-dup pass is tractable (981k x 18.7k pairwise would not be).

Usage: venv/bin/python scripts/audit_tram.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import BenchmarkLoader  # noqa: E402

TRAIN_POOL_FILES = [
    PROJECT_ROOT / "data" / "combined_80_20_v4" / "train.jsonl",
    PROJECT_ROOT / "data" / "combined_80_20_v4" / "val.jsonl",
]

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokens(s: str) -> set[str]:
    return set(TOKEN_RE.findall(str(s or "").lower()))


def norm_text(s: str) -> str:
    return " ".join(str(s or "").lower().split())


def load_train_pool() -> list[dict]:
    rows = []
    for p in TRAIN_POOL_FILES:
        if not p.exists():
            print(f"WARNING: {p} not found, skipping")
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def structural_audit(examples) -> None:
    print("=" * 70)
    print("STRUCTURAL AUDIT")
    print("=" * 70)
    per_task = Counter(ex.task for ex in examples)
    empty_q = empty_ans = empty_ctx = 0
    ids = set()
    dup_ids = 0
    qa_pairs = Counter()
    for ex in examples:
        if not str(ex.question or "").strip():
            empty_q += 1
        if not str(ex.answer or "").strip():
            empty_ans += 1
        if not str(ex.context or "").strip():
            empty_ctx += 1
        if ex.id in ids:
            dup_ids += 1
        ids.add(ex.id)
        qa_pairs[(norm_text(ex.question), norm_text(ex.answer))] += 1

    print(f"total examples: {len(examples)}")
    print("\nper-task counts:")
    for task, n in sorted(per_task.items(), key=lambda kv: -kv[1]):
        print(f"  {task:25s} {n:>8d}")
    print(f"\nempty question: {empty_q}")
    print(f"empty answer:   {empty_ans}")
    print(f"empty context:  {empty_ctx}")
    print(f"duplicate ids:  {dup_ids}")
    internal_dups = sum(c - 1 for c in qa_pairs.values() if c > 1)
    print(f"internal exact (question,answer) duplicates: {internal_dups}")


def leakage_audit(examples, train_rows: list[dict]) -> None:
    print()
    print("=" * 70)
    print("LEAKAGE AUDIT (TRAM vs current v4 training pool)")
    print("=" * 70)
    print(f"training pool size: {len(train_rows)}")

    train_q_norm = set()
    train_ctx_norm = set()
    train_q_tokens = []  # list of (row_idx, token_set)
    for i, r in enumerate(train_rows):
        q = r.get("question") or ""
        ctx = r.get("context") or ""
        train_q_norm.add(norm_text(q))
        if ctx.strip():
            train_ctx_norm.add(norm_text(ctx))
        train_q_tokens.append((i, tokens(q)))

    # ---- exact match pass (authoritative, O(n)) ----
    exact_q_hits = exact_ctx_hits = 0
    for ex in examples:
        if norm_text(ex.question) in train_q_norm:
            exact_q_hits += 1
        if ex.context and norm_text(ex.context) in train_ctx_norm:
            exact_ctx_hits += 1
    print(f"\nexact question match (TRAM item == a training question): {exact_q_hits}")
    print(f"exact context match  (TRAM item == a training passage):  {exact_ctx_hits}")

    # ---- near-dup pass, blocked on rare tokens for tractability ----
    # Build TRAM token doc-frequency to find rare tokens, then index TRAM
    # rows under their rarest tokens only, so each training question only
    # has to check a small candidate set instead of all 980,918 rows.
    df = Counter()
    tram_tokens = []
    for ex in examples:
        t = tokens(ex.question)
        tram_tokens.append(t)
        df.update(t)

    RARE_DF_CEILING = 500
    # Index under the N rarest tokens, not one (audit 2026-09-09 M13, fixed
    # 2026-09-12). With a single key the search was ASYMMETRIC: a pair was
    # only ever compared if the rarest token happened to be the SAME on both
    # sides. Two questions sharing "Mubarak" and "1981" would be missed
    # whenever the training row's rarest was "Mubarak" and the TRAM row's was
    # "1981" -- i.e. exactly the distinctive-token case the note below claims
    # to catch, so leakage could be under-counted. Probing N keys on each side
    # and unioning the postings makes the search symmetric; the df<=500
    # ceiling keeps every posting list short, so the cost is bounded.
    RARE_KEYS = 3
    index: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(tram_tokens):
        rare = [w for w in t if df[w] <= RARE_DF_CEILING]
        if not rare:
            continue
        for w in sorted(rare, key=lambda w: df[w])[:RARE_KEYS]:
            index[w].append(i)

    near_dup_hits = 0
    checked_candidates = 0
    examples_over_thresh = []
    for row_idx, qt in train_q_tokens:
        if not qt:
            continue
        rare = [w for w in qt if df.get(w, 0) <= RARE_DF_CEILING and w in index]
        if not rare:
            continue
        candidates: set[int] = set()
        for w in sorted(rare, key=lambda w: df.get(w, 0))[:RARE_KEYS]:
            candidates.update(index.get(w, ()))
        best = 0.0
        best_i = None
        for ci in candidates:
            ct = tram_tokens[ci]
            if not ct:
                continue
            checked_candidates += 1
            j = len(qt & ct) / len(qt | ct)
            if j > best:
                best, best_i = j, ci
        if best > 0.8:
            near_dup_hits += 1
            if len(examples_over_thresh) < 10:
                examples_over_thresh.append(
                    (row_idx, best, best_i, train_rows[row_idx].get("question"), examples[best_i].question)
                )

    print(f"\nnear-dup (Jaccard>0.8) training-question vs TRAM-question: {near_dup_hits}")
    print(f"(blocked search: {checked_candidates} candidate pairs actually compared, "
          f"out of {len(train_q_tokens)} training questions)")
    if examples_over_thresh:
        print("\nsample near-dup pairs:")
        for row_idx, score, ci, tq, trq in examples_over_thresh:
            print(f"  J={score:.2f} train[{row_idx}]: {tq[:70]!r}")
            print(f"           tram : {trq[:70]!r}")

    print(f"\nNote: near-dup pass is blocked on rare tokens (df<={RARE_DF_CEILING} within "
          f"TRAM's 980,918 questions) for tractability, same tradeoff style as "
          f"the project's existing purge_near_duplicates. Each side is indexed "
          f"and probed under its {RARE_KEYS} rarest tokens (symmetric since "
          f"2026-09-12; a single key missed pairs whose rarest token differed "
          f"between the two sides). It will still miss a near-dup pair where "
          f"every shared token is extremely common in TRAM (e.g. two generic "
          f"yes/no NLI templates) but will catch any pair sharing a distinctive "
          f"name/date/entity token, which is what actual passage reuse looks "
          f"like in practice.")


def main() -> int:
    loader = BenchmarkLoader(str(PROJECT_ROOT / "data" / "benchmarks"))
    print("Loading TRAM (no eval, no model)...")
    examples = loader.load("tram")
    structural_audit(examples)

    train_rows = load_train_pool()
    if not train_rows:
        print("\nNo training pool found; skipping leakage audit.")
        return 0
    leakage_audit(examples, train_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
