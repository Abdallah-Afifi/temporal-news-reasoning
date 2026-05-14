"""Shared folder paths for this project.

``PROJECT_ROOT`` is the repo folder (the one that contains ``encoder/`` and ``output/``).
Use these constants so scripts find files even when imports move around.
"""

from __future__ import annotations

from pathlib import Path

# Folder that holds this file (the ``temporal_rag`` package).
PACKAGE_DIR = Path(__file__).resolve().parent
# One level up: whole repository root.
PROJECT_ROOT = PACKAGE_DIR.parent
