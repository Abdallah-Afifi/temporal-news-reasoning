"""ORDER lines for the v12 data fixes (docs/v12_plan.md).

Only the five categories whose generator cards were corrected on 2026-09-25
(relation, ordering, storytelling, temporal_dialogue, duration). Replies are
ingested into a SEPARATE corpus, data/manual_aug_glm_v12/, so v11's source
data/manual_aug_glm/ stays exactly as v11 trained on it.

Usage:
    venv/bin/python scripts/make_glm_packets.py --master --out data/glm_packets_v12
    venv/bin/python scripts/make_v12_orders.py
"""
from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_glm_packets import DOMAINS, ERAS  # noqa: E402

PER_ORDER = 10
# (category, rows, shape, passages). storytelling/masked dialogue/commonsense
# duration are self-contained rows (see the brief's "shape=story/masked").
PLAN = [
    ("storytelling", 150, "story", 0),          # replaces all 150 old rows
    ("relation", 120, "none", 0),               # re-balance across 5 TRAM golds
    ("ordering", 100, "none", 0),               # adds the sequence shape
    ("temporal_dialogue", 120, "masked", 0),    # TimeDial-style, number words
    ("duration", 120, "none", 0),               # commonsense typical durations
]


def main() -> int:
    rng = random.Random("v12-orders")
    dom = itertools.cycle(rng.sample(DOMAINS, len(DOMAINS)))
    era = itertools.cycle(rng.sample(ERAS, len(ERAS)))
    lines = []
    for cat, rows, shape, passages in PLAN:
        for _ in range(rows // PER_ORDER):
            extra = ""
            if cat == "duration":
                extra = " duration_shape=commonsense"
            if cat == "ordering":
                extra = f" ordering_shape={'sequence' if len(lines) % 2 else 'truefalse'}"
            lines.append(f'ORDER: category={cat} rows={PER_ORDER} passages={passages} '
                         f'shape={shape} domain="{next(dom)}" era={next(era)}{extra}')
    out = ROOT / "data" / "glm_packets_v12" / "ORDERS_v12.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(lines)} orders ({len(lines) * PER_ORDER} rows) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
