"""Print FINAL/PROVISIONAL for a v13 build manifest (gate for run_schedule_v13.sh).

FINAL only when the manifest does not say provisional and it carries
non-programmatic rows (> 0). The builder writes `template_rows` (AUG_TPL3,
audit 2026-10-04 §1.1) and keeps `glm_rows` equal to it for older readers,
so `template_rows` is used when present -- summing both would double-count.
"""
from __future__ import annotations

import json
import sys


def build_status(manifest: dict) -> str:
    if "template_rows" in manifest:
        rows = int(manifest["template_rows"] or 0)
    else:
        rows = int(manifest.get("glm_rows", 0) or 0) + int(manifest.get("tpl_rows", 0) or 0)
    final = manifest.get("provisional") is not True and rows > 0
    return (f"{'FINAL' if final else 'PROVISIONAL'} provisional={manifest.get('provisional')} "
            f"non_programmatic_rows={rows}")


if __name__ == "__main__":
    print(build_status(json.load(open(sys.argv[1]))))
