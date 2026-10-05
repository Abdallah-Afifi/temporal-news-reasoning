"""Create data/training_versions/<version>/ in the same format as v1-v7.

Copies <src>/{train,val}.jsonl (+ the builder's manifest.json if present) into
<version>/data/, verifies the copy by sha256, measures the composition, and
writes <version>/manifest.json + MANIFEST.md. The canonical originals stay in
data/combined_80_20_*; configs point there, and if a copy ever diverges the
original wins (data/training_versions/README.md).

Usage:
    venv/bin/python scripts/make_version_folder.py v11 data/combined_80_20_v11 \
        --story "..." --results "..."
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("version")
    ap.add_argument("src")
    ap.add_argument("--story", required=True)
    ap.add_argument("--results", default="pending")
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()

    src = ROOT / args.src
    dst = ROOT / "data" / "training_versions" / args.version
    (dst / "data").mkdir(parents=True, exist_ok=True)
    files = [f for f in ("train.jsonl", "val.jsonl", "manifest.json") if (src / f).exists()]
    hashes = {}
    for f in files:
        shutil.copy2(src / f, dst / "data" / f)
        a, b = sha(src / f), sha(dst / "data" / f)
        if a != b:
            raise SystemExit(f"FATAL: copy of {f} does not match its source")
        hashes[f] = a

    comp: Counter = Counter()
    rows = {}
    for split in ("train", "val"):
        n = 0
        for line in open(src / f"{split}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            n += 1
            if split == "train":
                comp[r.get("source_dataset", "unknown")] += 1
        rows[split] = n
    total = rows["train"] or 1
    synth = sum(v for k, v in comp.items() if k.startswith("AUG"))
    size_mb = round(sum((src / f).stat().st_size for f in files) / 1e6, 1)
    manifest = {
        "version": args.version, "dir": args.src,
        "train_rows": rows["train"], "val_rows": rows["val"],
        "synthetic_pct": round(100 * synth / total, 1),
        "general_rehearsal_pct": round(100 * comp.get("REHEARSAL", 0) / total, 1),
        "composition": dict(comp.most_common()),
        "story": args.story, "results": args.results,
        "results_dir": args.results_dir, "size_mb": size_mb,
        "sha256": hashes,
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (dst / "MANIFEST.md").write_text(
        f"# Training data — {args.version}\n\n"
        f"Dir: `{args.src}/` ({size_mb} MB) — train {rows['train']:,} / val {rows['val']:,} "
        f"| synthetic {manifest['synthetic_pct']}% | general rehearsal "
        f"{manifest['general_rehearsal_pct']}%\n\n{args.story}\n\n"
        f"Results: {args.results}\n\n"
        f"Full composition in `manifest.json`. A sha256-verified copy of the data is in "
        f"`data/`; the canonical dir is `{args.src}/` (configs point there — if the copy "
        f"ever diverges, the original wins).\n")
    print(f"{args.version}: {rows['train']:,} train / {rows['val']:,} val, "
          f"{manifest['synthetic_pct']}% synthetic -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
