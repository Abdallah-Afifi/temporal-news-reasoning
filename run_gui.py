#!/usr/bin/env python3
"""Open the Temporal RAG web app (Streamlit).

Steps for first use are in ``docs/demo.md``. For the text-only demo run:
``python3 -m temporal_rag.interactive_query``.

This script always starts Streamlit from the repo root folder so paths like
``encoder/``, ``output/``, and ``.streamlit/config.toml`` work.

Install Streamlit if needed: ``pip install streamlit pandas`` (see README for more).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT / "temporal_rag" / "temporal_rag_gui.py"


def main() -> None:
    """Start Streamlit using ``temporal_rag_gui.py`` and repo root as working directory."""
    if not APP.is_file():
        print(f"Missing {APP}", file=sys.stderr)
        sys.exit(1)
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP),
        "--browser.gatherUsageStats",
        "false",
        "--client.toolbarMode",
        "viewer",
    ]
    raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == "__main__":
    main()
