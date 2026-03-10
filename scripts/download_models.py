"""
Download pre-trained models from Hugging Face.

Usage:
    python scripts/download_models.py                # download all models
    python scripts/download_models.py qwen phi       # download specific models
    python scripts/download_models.py embedding      # download embedding model only

Note: LLaMA-3.2-3B requires accepting Meta's license at https://huggingface.co/meta-llama
"""

import os
import sys

MODELS = {
    "qwen":      ("Qwen/Qwen2.5-3B-Instruct",                 "Qwen2.5-3B-Instruct"),      # ~6 GB
    "phi":       ("microsoft/Phi-3-mini-4k-instruct",          "Phi-3-mini-4k-instruct"),   # ~7.5 GB
    "llama":     ("meta-llama/Llama-3.2-3B-Instruct",          "Llama-3.2-3B-Instruct"),   # ~6 GB (license required)
    "embedding": ("sentence-transformers/all-MiniLM-L6-v2",    "all-MiniLM-L6-v2"),        # ~80 MB
}


def _already_downloaded(local_dir: str) -> bool:
    """Return True if model weights already exist at local_dir."""
    return os.path.isdir(local_dir) and any(
        f.endswith((".safetensors", ".bin", ".model"))
        for f in os.listdir(local_dir)
    )


def download_model(name: str, output_dir: str = "./models"):
    """Download a model from Hugging Face Hub, skipping if already present."""
    from huggingface_hub import snapshot_download

    entry = MODELS.get(name)
    if not entry:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODELS.keys())}")

    repo_id, folder_name = entry
    local_dir = os.path.join(output_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {name}: {repo_id}  →  {local_dir}")
    print(f"{'='*60}")

    if _already_downloaded(local_dir):
        print(f"  Already exists, skipping.")
        return

    snapshot_download(repo_id=repo_id, local_dir=local_dir)
    print(f"  ✓ Done.")


if __name__ == "__main__":
    names = sys.argv[1:] if len(sys.argv) > 1 else list(MODELS.keys())
    print(f"Downloading {len(names)} model(s): {names}")
    for name in names:
        download_model(name)
    print(f"\n{'='*60}")
    print("All requested models downloaded.")
    print(f"{'='*60}")
