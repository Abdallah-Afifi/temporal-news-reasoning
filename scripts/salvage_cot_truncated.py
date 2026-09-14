#!/usr/bin/env python
"""Salvage a CoT run that was generated with too small a --max-new-tokens.

WHY THIS IS SOUND (and not a protocol violation)
------------------------------------------------
Generation is greedy: `run_baselines.py` sets `do_sample = temperature > 0`
and temperature defaults to 0.0. A `max_new_tokens` cap can only TRUNCATE a
greedy sequence; it never changes the prefix. So any item that emitted its
`ANSWER:` line and stopped before the old cap produces a BYTE-IDENTICAL
generation under a larger cap.

Therefore: keep every row that reached `ANSWER:`, drop the truncated ones, and
re-run with the larger budget. `run_baselines.py` resumes by id, so it
regenerates exactly the dropped items. The result is equivalent to a full
re-run at the larger budget, at the cost of only the truncated fraction
(~22% of llama/TIME, ~7.5% of llama/TimeBench).

The one caveat is batch composition: the re-run batches the remaining items
differently. Under greedy decoding that is per-sequence and result-neutral by
the project's own D6 decision (the same effect already accepted for the
HF/vLLM parity legs).

USAGE
-----
    # inspect only -- writes nothing
    python scripts/salvage_cot_truncated.py --results-dir results/baseline/zero_shot_cot

    # rewrite the prediction files, keeping a .bak of each
    python scripts/salvage_cot_truncated.py --results-dir results/baseline/zero_shot_cot --apply

Then re-run the SAME leg command with --max-new-tokens 768. It will regenerate
only the dropped ids and append them.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ANCHOR = "ANSWER:"


def process(path: Path, apply: bool) -> tuple[int, int, int]:
    kept = dropped = torn = 0
    out_lines: list[str] = []
    for line in open(path, encoding="utf-8"):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            # A torn line has no recoverable id; dropping it makes the item
            # regenerate, which is what we want.
            torn += 1
            continue
        raw = rec.get("raw_prediction") or ""
        if ANCHOR in raw:
            kept += 1
            out_lines.append(line if line.endswith("\n") else line + "\n")
        else:
            dropped += 1
    if apply and (dropped or torn):
        shutil.copy2(path, path.with_suffix(".jsonl.bak"))
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(out_lines)
    return kept, dropped, torn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True,
                    help="e.g. results/baseline/zero_shot_cot")
    ap.add_argument("--apply", action="store_true",
                    help="rewrite the files (a .bak is kept); default is a dry run")
    args = ap.parse_args()

    root = Path(args.results_dir)
    files = sorted(root.glob("*/*/*/predictions.jsonl"))
    if not files:
        print(f"no predictions.jsonl under {root}")
        return 1
    print(f"{'file':58s}{'keep':>9s}{'regen':>9s}{'torn':>6s}{'regen%':>8s}")
    tot_k = tot_d = 0
    for p in files:
        k, d, t = process(p, args.apply)
        n = k + d + t
        print(f"{str(p.relative_to(root)):58s}{k:9,}{d:9,}{t:6d}"
              f"{100 * (d + t) / n if n else 0:7.1f}%")
        tot_k += k
        tot_d += d + t
    print(f"\n{'TOTAL':58s}{tot_k:9,}{tot_d:9,}")
    if args.apply:
        print(f"\nRewrote {len(files)} file(s); .bak kept beside each.")
        print("Now re-run the same leg with --max-new-tokens 768 — it will "
              "regenerate only the dropped ids.")
    else:
        print("\nDRY RUN — nothing written. Add --apply to rewrite.")
        print(f"Re-running would cost ~{100 * tot_d / max(tot_k + tot_d, 1):.0f}% "
              f"of a full re-run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
