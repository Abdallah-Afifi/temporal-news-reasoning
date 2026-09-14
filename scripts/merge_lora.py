"""Merge a LoRA adapter into its base model (frozen venv, PEFT 0.18.1).

Produces a plain HF model directory that any engine (incl. vLLM) can load
without adapter support. Mirrors the eval-time merge in
src/models/inference.py (merge_and_unload) so weights are identical.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="models/Llama-3.2-3B-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    # Idempotency check must require BOTH config.json and at least one
    # weights shard: an interrupted merge can leave config.json written while
    # shards are incomplete, and vLLM would then load a corrupt half-merged
    # model (audit 2026-09-09 M7). save_pretrained writes config.json last,
    # so a complete merge always has shards present too.
    shards = list(out.glob("*.safetensors")) if out.exists() else []
    if (out / "config.json").exists() and shards:
        print(f"{out} already exists — skipping merge")
        return 0
    if shards or out.exists():
        print(
            f"WARNING: {out} exists but looks incomplete "
            f"(config.json={'yes' if (out / 'config.json').exists() else 'no'}, "
            f"{len(shards)} shard(s)) — re-merging into it"
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float16, device_map="cpu",
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)

    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        src = Path(args.base) / f
        if src.exists():
            shutil.copy(src, out / f)
    print(f"merged {args.adapter} into {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
