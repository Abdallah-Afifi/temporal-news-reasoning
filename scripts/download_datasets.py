"""
Download benchmark datasets and news corpora.

Usage:
    python scripts/download_datasets.py              # download all datasets
    python scripts/download_datasets.py time tram    # download specific datasets

Datasets:
    time          - SylvainWei/TIME (~1.1 GB) + TIME-Lite (~36 MB)
    timebench     - TimeBench (cloned from GitHub)
    tram          - TRAM-Benchmark (cloned from GitHub)
    cnn_dailymail - abisee/cnn_dailymail (~2.5 GB)
"""

import os
import subprocess
import sys


# ---------------------------------------------------------------------------
# Dataset output paths (relative to repo root)
# ---------------------------------------------------------------------------
DATASETS = {
    "time": {
        "hf_repos": [
            ("SylvainWei/TIME",      "./data/benchmarks/time/TIME"),
            ("SylvainWei/TIME-Lite", "./data/benchmarks/time/TIME-Lite"),
        ],
    },
    "timebench": {
        "git_url": "https://github.com/zchuz/TimeBench.git",
        "output":  "./data/benchmarks/timebench/TimeBench",
    },
    "tram": {
        "git_url": "https://github.com/EternityYW/TRAM-Benchmark.git",
        "output":  "./data/benchmarks/tram/TRAM-Benchmark",
    },
    "cnn_dailymail": {
        "hf_repos": [
            ("abisee/cnn_dailymail", "./data/corpus/cnn_dailymail"),
        ],
    },
}


def _hf_download(repo_id: str, local_dir: str):
    from huggingface_hub import snapshot_download
    os.makedirs(local_dir, exist_ok=True)
    print(f"  Downloading {repo_id}  ->  {local_dir}")
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local_dir)
    print(f"  Done.")


def _git_clone(url: str, local_dir: str):
    if os.path.isdir(local_dir):
        print(f"  Already exists at {local_dir}, skipping.")
        return
    os.makedirs(os.path.dirname(local_dir), exist_ok=True)
    print(f"  Cloning {url}  ->  {local_dir}")
    subprocess.run(["git", "clone", url, local_dir], check=True)
    print(f"  Done.")


def download_dataset(name: str):
    """Download a single dataset by name."""
    cfg = DATASETS.get(name)
    if not cfg:
        raise ValueError(f"Unknown dataset: '{name}'. Available: {list(DATASETS.keys())}")

    print(f"\n{'='*60}")
    print(f"  Dataset: {name}")
    print(f"{'='*60}")

    if "hf_repos" in cfg:
        for repo_id, local_dir in cfg["hf_repos"]:
            _hf_download(repo_id, local_dir)
    if "git_url" in cfg:
        _git_clone(cfg["git_url"], cfg["output"])


if __name__ == "__main__":
    names = sys.argv[1:] if len(sys.argv) > 1 else list(DATASETS.keys())
    print(f"Downloading dataset(s): {names}")
    for name in names:
        download_dataset(name)
    print(f"\n{'='*60}")
    print("All requested datasets downloaded.")
    print(f"{'='*60}")
