"""Validate and ingest GLM chat-generated CoT traces (scripts/build_cot_packets.py).

Two gates beyond a normal AUG_GLM2 ingest, because a CoT trace's whole point
is the reasoning, not just the final answer:

  1. ANSWER-MATCH: the trace's `ANSWER:` line must match the real gold under
     the project's own scorer normalization (src/evaluation/metrics + TIME
     date_equivalence) -- the SAME functions rescore_v5_protocol.py uses, so
     a trace that "verifies" here would also score correct at eval time.
  2. GROUNDING: every step must reference the passage, not just assert the
     answer. Checked two ways: (a) step 1 must not ANNOUNCE the answer -- it
     is rejected only when it contains the gold AND a giveaway phrase
     (_GIVEAWAY_RE, e.g. "the answer is"). Plain containment of the gold in
     step 1 is ALLOWED (legitimate extraction often names the entity there);
     on the 2026-10-04 bank 92% of coding-agent traces and 37% of chat-era
     traces carry the gold verbatim in step 1. This does NOT catch "answer in
     step 1, pad the rest" -- the STaR failure mode measured at 16% useful
     yield -- unless the giveaway phrase is used; (b) some step must
     share a >= 6-token run with the passage AND that run must include a
     digit or a proper noun (a very weak floor -- catches a trace that never
     touches the source text at all, not a strong faithfulness check; still
     100% automatable, unlike faithfulness itself -- see touches_passage()).

Usage:
    venv/bin/python scripts/ingest_cot_batch.py --mode grounded data/cot_raw/*.txt
    venv/bin/python scripts/ingest_cot_batch.py --mode invented data/cot_raw/*.txt
    venv/bin/python scripts/ingest_cot_batch.py --status
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.date_equivalence import matches_any  # noqa: E402
from src.evaluation.metrics import _normalize_answer  # noqa: E402

OUT_DIR = PROJECT_ROOT / "data" / "cot_verified"
ANSWER_KEY = PROJECT_ROOT / "data" / "cot_packets" / "answer_key.json"
LEDGER = OUT_DIR / ".ingested.json"

ANSWER_RE = re.compile(r"(?im)^\s*ANSWER\s*:\s*(.+?)\s*$")
_GIVEAWAY_RE = re.compile(r"(?i)\b(?:the\s+)?(?:gold|final|correct)?\s*answer\s+is\b")
STEP_RE = re.compile(r"(?im)^\s*Step\s+(\d+)\s*:")
_TOK = re.compile(r"[A-Za-z0-9]+")


def extract_answer(cot: str) -> str | None:
    m = ANSWER_RE.findall(cot or "")
    return m[-1].strip() if m else None


def count_steps(cot: str) -> int:
    nums = [int(n) for n in STEP_RE.findall(cot or "")]
    return max(nums) if nums else 0


def step_blocks(cot: str) -> list[str]:
    parts = re.split(r"(?im)^\s*Step\s+\d+\s*:\s*", cot or "")
    return [p.strip() for p in parts[1:]]  # part[0] is anything before "Step 1:"


def touches_passage(cot: str, passage: str, min_shared_run: int = 6) -> bool:
    """Weak floor, not a faithfulness check -- see module docstring.

    Found by testing this gate on a deliberately fabricated trace ("He was a
    famous Australian politician... well known for his long career", correct
    answer copied in, no real grounding): a 4-token run of generic connective
    phrasing matched somewhere in a long passage BY CHANCE and the trace was
    wrongly accepted. Raised to 6 tokens (a natural-language 6-gram
    essentially never recurs by accident) and a specific-content requirement
    was added below: SOME shared run must include a digit (a date/year/count
    -- the discriminating fact in most temporal-QA items) or a capitalized
    non-sentence-initial word (a proper noun). Generic filler passes neither.
    """
    ptoks = _TOK.findall((passage or "").lower())
    ctoks = _TOK.findall((cot or "").lower())
    craw = _TOK.findall(cot or "")
    pset = {tuple(ptoks[i:i + min_shared_run])
            for i in range(len(ptoks) - min_shared_run + 1)}
    for i in range(len(ctoks) - min_shared_run + 1):
        if tuple(ctoks[i:i + min_shared_run]) not in pset:
            continue
        run = craw[i:i + min_shared_run]
        if any(any(c.isdigit() for c in t) for t in run):
            return True
        if any(t[0].isupper() for t in run):
            return True
    return False


def check_trace(cot: str, gold: str, passage: str) -> list[str]:
    e: list[str] = []
    if not isinstance(cot, str) or not cot.strip():
        return ["cot must be a non-empty string"]
    ans = extract_answer(cot)
    if ans is None:
        return ["no ANSWER: line found (§ answer-match)"]
    norm_ans = _normalize_answer(ans)
    if not (norm_ans in {_normalize_answer(gold)} or matches_any(ans, [gold])):
        e.append(f"ANSWER {ans!r} does not match gold {gold!r} (§ answer-match)")
    n = count_steps(cot)
    if not (1 <= n <= 5):
        e.append(f"step count {n} out of [1,5] (§ step-bounds)")
    blocks = step_blocks(cot)
    # A giveaway PHRASE ("the answer is ..."), not mere containment of the
    # gold text in step 1: legitimate extraction routinely surfaces the
    # correct entity in step 1 ("In 1974 ... elected to the House of
    # Representatives") as part of real reasoning, and a plain containment
    # check flagged that as premature (found in testing 2026-09-24). What
    # this actually needs to catch is the STaR-style degenerate pattern --
    # announcing the answer as a conclusion instead of deriving it.
    if blocks and gold and _GIVEAWAY_RE.search(blocks[0]) \
            and _normalize_answer(gold) in _normalize_answer(blocks[0]):
        e.append("step 1 announces the answer instead of deriving it (§ grounding-a)")
    if passage and not touches_passage(cot, passage):
        e.append("no step shares a specific (dated/named) 6-token run with the passage (§ grounding-b)")
    return e


def parse_replies(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def ingest_grounded(paths: list[str]) -> None:
    key = json.loads(ANSWER_KEY.read_text())
    rows = parse_replies(paths)
    accepted, rejected = [], []
    for r in rows:
        rid = r.get("id")
        src = key.get(rid)
        if src is None:
            rejected.append((rid, ["id not found in answer_key.json"]))
            continue
        errs = check_trace(r.get("cot"), src["gold"], src["context"])
        if errs:
            rejected.append((rid, errs))
            continue
        accepted.append({"id": rid, "mode": "grounded", "question": src["question"],
                         "context": src["context"], "gold": src["gold"], "cot": r["cot"],
                         "source": "TimeQA", "source_dataset": "COT_GLM"})
    _write(accepted, rejected, "grounded")


def ingest_invented(paths: list[str]) -> None:
    rows = parse_replies(paths)
    accepted, rejected = [], []
    for r in rows:
        rid = r.get("id", "?")
        missing = [k for k in ("category", "passage", "question", "gold", "cot") if k not in r]
        if missing:
            rejected.append((rid, [f"missing fields {missing}"]))
            continue
        errs = check_trace(r["cot"], r["gold"], r["passage"])
        if errs:
            rejected.append((rid, errs))
            continue
        accepted.append({"id": rid, "mode": "invented", "category": r["category"],
                         "question": r["question"], "context": r["passage"], "gold": r["gold"],
                         "cot": r["cot"], "source": "augmented", "source_dataset": "COT_GLM"})
    _write(accepted, rejected, "invented")


def _write(accepted: list[dict], rejected: list[tuple], mode: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{mode}.jsonl"
    seen = set()
    if out_path.exists():
        seen = {json.loads(l)["id"] for l in open(out_path, encoding="utf-8") if l.strip()}
    new = [a for a in accepted if a["id"] not in seen]
    with open(out_path, "a", encoding="utf-8") as f:
        for a in new:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    print(f"{mode}: accepted {len(new)} new (of {len(accepted)} parsed-ok, "
         f"{len(seen)} already banked), rejected {len(rejected)}")
    if rejected:
        reasons = Counter(e for _, errs in rejected for e in errs)
        for reason, n in reasons.most_common():
            print(f"  {n:4d}  {reason}")
        rej_path = OUT_DIR / "_rejected.jsonl"
        with open(rej_path, "a", encoding="utf-8") as f:
            for rid, errs in rejected:
                f.write(json.dumps({"id": rid, "errors": errs}, ensure_ascii=False) + "\n")
        print(f"  rejected detail -> {rej_path}")


def status() -> None:
    for mode in ("grounded", "invented"):
        p = OUT_DIR / f"{mode}.jsonl"
        n = sum(1 for _ in open(p, encoding="utf-8")) if p.exists() else 0
        print(f"{mode:10s} {n}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["grounded", "invented"])
    ap.add_argument("--status", action="store_true")
    ap.add_argument("files", nargs="*")
    args = ap.parse_args()
    if args.status:
        status()
        return 0
    if not args.mode or not args.files:
        ap.error("--mode and at least one file are required (or use --status)")
    paths = [f for pat in args.files for f in (glob.glob(pat) or [pat])]
    (ingest_grounded if args.mode == "grounded" else ingest_invented)(paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
