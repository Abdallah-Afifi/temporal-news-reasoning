"""Build the v12 mixture: v11 + the corrected-card rows (docs/v12_plan.md).

  - start from data/combined_80_20_v11/{train,val}.jsonl, UNCHANGED except:
  - drop every AUG_GLM2 `storytelling` row: the old card taught "the wrong
    ending contradicts a stated date" with narrator-style endings, while TRAM
    storytelling is commonsense plausibility over plain sentences (v11 lost
    11.3pp there; results_and_methodology.md §10.3);
  - add every row banked in data/manual_aug_glm_v12/ (ingested with the
    corrected gate), re-checked here with ingest_glm_batch.check() and the
    same benchmark-contamination filter v11 used, split 80/20 by a stable hash.

v11's own files, and data/manual_aug_glm/, are only read. Writes
data/combined_80_20_v12/ and its data/training_versions/v12/ folder.

Usage: venv/bin/python scripts/build_v12_training_data.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_v11_parity_data import bench_index, contaminated  # noqa: E402
from ingest_glm_batch import check  # noqa: E402

SRC = ROOT / "data" / "combined_80_20_v11"
NEW = ROOT / "data" / "manual_aug_glm_v12"
OUT = ROOT / "data" / "combined_80_20_v12"
DROP = {("AUG_GLM2", "storytelling")}


def main() -> int:
    new_rows = [json.loads(l) for f in sorted(NEW.glob("*.jsonl"))
                for l in open(f, encoding="utf-8") if l.strip()] if NEW.exists() else []
    if not new_rows:
        raise SystemExit(f"FATAL: no rows in {NEW} -- generate and ingest the v12 orders first "
                         f"(data/glm_packets_v12/ORDERS_v12.txt).")
    bad = [r for r in new_rows if check(r)]
    if bad:
        raise SystemExit(f"FATAL: {len(bad)} banked v12 rows fail the corrected gate, e.g. "
                         f"{check(bad[0])}")
    idx = bench_index()
    clean = [r for r in new_rows if not contaminated(r, *idx)]
    print(f"new rows: {len(new_rows)}, contaminated: {len(new_rows) - len(clean)}")

    def split_of(r: dict) -> str:
        h = int(hashlib.sha1(r["question"].encode()).hexdigest(), 16)
        return "val" if h % 5 == 0 else "train"

    OUT.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"source": "data/combined_80_20_v11 + data/manual_aug_glm_v12",
                      "dropped": sorted("/".join(k) for k in DROP), "splits": {}}
    for split in ("train", "val"):
        base = [json.loads(l) for l in open(SRC / f"{split}.jsonl", encoding="utf-8")]
        kept = [r for r in base if (r.get("source_dataset"), r.get("category")) not in DROP]
        added = [r for r in clean if split_of(r) == split]
        rows = kept + added
        with open(OUT / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest["splits"][split] = {
            "from_v11": len(base), "dropped": len(base) - len(kept), "added": len(added),
            "out": len(rows),
            "added_by_category": dict(Counter(r.get("category") for r in added))}
        print(split, manifest["splits"][split])
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_version_folder.py"), "v12",
                    "data/combined_80_20_v12", "--story",
                    "v11 minus the old AUG_GLM2 storytelling rows, plus rows written to the "
                    "corrected relation / ordering / storytelling / temporal_dialogue / "
                    "duration cards (docs/v12_plan.md)."], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
