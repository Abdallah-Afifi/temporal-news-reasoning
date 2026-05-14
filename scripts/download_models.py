"""Download pre-trained models from Hugging Face."""

import os

MODELS = {
    "qwen": "Qwen/Qwen2.5-3B-Instruct",
    "phi": "microsoft/Phi-3-mini-4k-instruct",
    "llama": "meta-llama/Llama-3.2-3B-Instruct",
    "embedding": "sentence-transformers/all-MiniLM-L6-v2",
}


def download_model(name: str, output_dir: str = "./models"):
    """Download a model from Hugging Face Hub."""
    from huggingface_hub import snapshot_download

    model_id = MODELS.get(name)
    if not model_id:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODELS.keys())}")

    local_dir = os.path.join(output_dir, name)
    print(f"Downloading {model_id} to {local_dir}...")
    snapshot_download(repo_id=model_id, local_dir=local_dir)
    print(f"Done: {local_dir}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        for name in sys.argv[1:]:
            download_model(name)
    else:
        print("Downloading all models...")
        for name in MODELS:
            download_model(name)
