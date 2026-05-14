"""
experiments/finetuning/shared/utils.py
============================
Common utilities shared across LLaMA / Qwen / Mistral fine-tuning pipelines.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import yaml
import torch


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create (or retrieve) a logger with a formatted StreamHandler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_yaml(path: str | Path) -> dict:
    """Load a YAML file and return its contents as a dict."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file and return a list of dicts."""
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_json(data: Any, path: str | Path) -> None:
    """Save data as formatted JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------

class Timer:
    """Simple wall-clock timer as a context manager."""

    def __init__(self, name: str = ""):
        self.name = name
        self.start_time: float = 0.0
        self.elapsed: float = 0.0
        self.elapsed_str: str = ""

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        hours, rem = divmod(int(self.elapsed), 3600)
        mins, secs = divmod(rem, 60)
        self.elapsed_str = f"{hours:02d}:{mins:02d}:{secs:02d}"


# ---------------------------------------------------------------------------
# GPU / device helpers
# ---------------------------------------------------------------------------

def get_device_info() -> str:
    """Return a one-line string describing the current GPU (or CPU)."""
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"CUDA | {gpu} | {mem:.1f} GB"
    return "CPU"


def print_gpu_memory(logger: Optional[logging.Logger] = None) -> None:
    """Log current GPU memory usage."""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    msg = f"GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved"
    if logger:
        logger.info(msg)
    else:
        print(msg)


def preflight_check(
    data_paths: list[Path] | None = None,
    output_dir: Path | None = None,
    min_vram_gb: float = 10.0,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Verify data files exist, output dir is writable, and VRAM is adequate."""
    log = logger or logging.getLogger(__name__)

    if data_paths:
        for p in data_paths:
            if not Path(p).exists():
                raise FileNotFoundError(f"Data file not found: {p}")
            log.info("✓ Data file found: %s", p)

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        log.info("✓ Output directory ready: %s", output_dir)

    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram < min_vram_gb:
            log.warning("Low VRAM: %.1f GB (recommended: %.1f GB)", vram, min_vram_gb)
        else:
            log.info("✓ VRAM: %.1f GB", vram)
    else:
        log.warning("No CUDA GPU detected — training will be very slow.")
