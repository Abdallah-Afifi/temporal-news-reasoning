"""Build the v9 mixture — the synthetic-swap arm (docs/synthetic_data_ruleset.md §1 L1).

L1: new rows DISPLACE old rows; total training rows stay <= 15,000. The
6,000-row AUG_GLM2 budget (delivered 5,976 — see data/manual_aug_v9/AUDIT.md)
is swapped into v6's mixture against retired rehearsal and the spent
synthetic slices it supersedes:

| retired (v6)            | superseded by (v9)                      |
|-------------------------|-----------------------------------------|
| REHEARSAL 2,077         | L1 budget; behaviour preserved by Block C |
| AUG_NOANS 400           | §5.12 matched answerability pairs       |
| AUG_MCQ/MCQ2 1,600      | AUG_GLM2 MCQ rows (L5 fix, no shortcut) |
| AUG_DURATION 320        | Duration_Compare card                   |
| AUG_ARITH 320           | Computation card                        |
| AUG_DIALOG/NLI/REASON   | Block C dialogue / Block B NLI cards    |

Kept from v6: the real-data pools (TimeQA, TLQA), the proven AUG_GLM manual
slice (+13.1pp Counterfactual) and AUG_SEQ (sequence-emission shape),
capped so the mixture lands at exactly 15,000 rows.

Usage:
    venv/bin/python scripts/build_v9_training_data.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_v5_training_data import (  # noqa: E402
    answer_style, benchmark_questions, norm, purge_near_duplicates,
    stratified_split,
)

CAP = 15_000
KEEP = {"TimeQA", "TLQA", "AUG_GLM", "AUG_SEQ"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v6-dir", default="data/combined_80_20_v6")
    ap.add_argument("--aug-dir", default="data/manual_aug_v9")
    ap.add_argument("--letter-probe", default="data/letter_format/probe.jsonl")
    ap.add_argument("--out", default="data/combined_80_20_v9")
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    bench_qs = benchmark_questions()

    v6 = []
    for name in ("train.jsonl", "val.jsonl"):
        v6 += [json.loads(l) for l in
               open(Path(args.v6_dir) / name, encoding="utf-8")]
    pools = {k: [r for r in v6 if r["source_dataset"] == k]
             for k in KEEP}
    for k in pools:
        seen: set[str] = set()
        uniq = []
        for r in pools[k]:
            nq = norm(str(r.get("question") or ""))
            if nq in seen:
                continue
            seen.add(nq)
            uniq.append(r)
        pools[k] = uniq

    aug = []
    for p in sorted(Path(args.aug_dir).glob("*.jsonl")):
        aug += [json.loads(l) for l in open(p, encoding="utf-8")]
    n_aug = len(aug)
    print(f"AUG_GLM2 rows: {n_aug} (target 6,000; shortfall recorded in AUDIT.md)")

    budget = args.cap - n_aug
    order = ["TimeQA", "AUG_GLM", "AUG_SEQ", "TLQA"]
    picked: list[dict] = []
    for k in order:
        take = min(len(pools[k]), max(0, budget - len(picked)))
        picked += pools[k][:take]
        if len(picked) >= budget:
            break
    print("kept from v6: " + ", ".join(
        f"{k} {min(len(pools[k]), max(0, budget))}" for k in order))

    all_records = aug + picked
    for r in all_records:
        if r["source_dataset"] == "TLQA" and r.get("targets"):
            r["final_answers"] = list(r["targets"])

    probe_q = {norm(json.loads(l)["question"])
               for l in open(args.letter_probe, encoding="utf-8")}
    all_q = [norm(str(r.get("question") or "")) for r in all_records]
    assert sum(q in probe_q for q in all_q) == 0, "FATAL: probe leak"
    n_coll = sum(q in bench_qs for q in all_q)
    assert n_coll == 0, f"FATAL: {n_coll} benchmark collisions"
    seen = set()
    deduped = []
    for r, q in zip(all_records, all_q):
        if q in seen:
            continue
        seen.add(q)
        deduped.append(r)
    print(f"guards passed: 0 probe leaks, 0 collisions; deduped "
          f"{len(all_records) - len(deduped)}")
    deduped, n_near = purge_near_duplicates(deduped)
    print(f"near-dup purge: dropped {n_near}")
    assert len(deduped) <= args.cap, f"FATAL: {len(deduped)} > cap {args.cap}"

    train, val = stratified_split(deduped, args.val_frac, rng)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train), ("val.jsonl", val)):
        with open(out / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def mix(rows):
        c = Counter(r["source_dataset"] for r in rows)
        glm2_pct = round(100 * sum(v for k, v in c.items()
                                   if k == "AUG_GLM2") / max(1, len(rows)), 1)
        return {"n": len(rows), "aug_glm2_pct": glm2_pct,
                "by_source": dict(c.most_common())}

    stats = {
        "cycle": "v9",
        "variables_vs_v6": [
            f"AUG_GLM2 +{n_aug} (docs/synthetic_data_ruleset.md; audited, "
            f"all §7 gates PASS)",
            "retired: REHEARSAL, AUG_NOANS, AUG_MCQ, AUG_MCQ2, AUG_DURATION, "
            "AUG_ARITH, AUG_DIALOG, AUG_NLI, AUG_REASON (L1 swap)",
            f"total capped at {args.cap} (was 18,558 in v6)",
        ],
        "train": mix(train), "val": mix(val),
        "near_dup_dropped": n_near,
        "answer_style_train": answer_style(train),
        "seed": args.seed, "args": vars(args),
    }
    (out / "manifest.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
