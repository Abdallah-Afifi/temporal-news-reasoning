#!/usr/bin/env python3
"""
experiments/finetuning/qwen3.5-9b-model/evaluate.py
=========================================
Evaluation script for the fine-tuned Qwen LoRA adapter.

NOTE: this directory name contains dots/dashes and is therefore NOT
importable as a Python package — this script is intentionally
self-contained and must be run as a file (``python experiments/finetuning/qwen3.5-9b-model/evaluate.py``).

Mirrors the LLaMA / Mistral evaluation pipeline:
  - Loads the base model (path from config.yaml) + attaches the LoRA adapter
  - Evaluates on a combined_80_20_split-format JSONL (default: val.jsonl)
  - Prompts use the SAME chat template as training
  - Batched greedy generation, multi-gold metrics, live progress log

Usage (from repo root)::

    python experiments/finetuning/qwen3.5-9b-model/evaluate.py
    python experiments/finetuning/qwen3.5-9b-model/evaluate.py --max-examples 100
    python experiments/finetuning/qwen3.5-9b-model/evaluate.py --load-in-4bit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from peft import PeftModel

from experiments.finetuning.shared.eval_runner import (
    load_eval_examples_from_jsonl,
    run_evaluation,
)
from experiments.finetuning.shared.utils import load_yaml, setup_logger

log = setup_logger("qwen3_5.evaluate")

MODEL_KEY = "qwen"
SCRIPT_DIR = Path(__file__).parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.yaml"
DEFAULT_CHECKPOINT = str(_REPO_ROOT / "checkpoints" / "qwen3.5-9b-model" / "final")
DEFAULT_VAL_DATA = str(_REPO_ROOT / "data" / "combined_80_20_split" / "val.jsonl")
RESULTS_DIR = SCRIPT_DIR / "results"


def load_tokenizer(model_name_or_path: str) -> AutoTokenizer:
    """Load and configure the Qwen tokenizer (mirrors qwen train.py)."""
    log.info("Loading Qwen tokenizer from: %s", model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, use_fast=True, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def resolve_base_model(config_path: Path) -> str:
    """Resolve the base model directory from config.yaml; fail clearly."""
    config = load_yaml(config_path)
    model_path = config.get("model_path", "models/qwen3.5-9b-model")
    resolved = (Path(model_path) if Path(model_path).is_absolute()
                else _REPO_ROOT / model_path)
    if not (resolved / "config.json").exists():
        raise FileNotFoundError(
            f"Qwen base model not found at '{resolved}'. "
            f"Set a valid 'model_path' in {config_path}."
        )
    return str(resolved)


def load_finetuned_model(
    base_model_name: str,
    adapter_path: str,
    load_in_4bit: bool = False,
) -> tuple:
    """Load the Qwen base model and attach the LoRA adapter."""
    tokenizer = load_tokenizer(base_model_name)

    load_kwargs: dict = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    log.info("Loading base model: %s", base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **load_kwargs)

    log.info("Attaching LoRA adapter: %s", adapter_path)
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
    model.eval()
    return model, tokenizer


def evaluate(
    adapter_path: str = DEFAULT_CHECKPOINT,
    base_model_name: Optional[str] = None,
    data_path: str = DEFAULT_VAL_DATA,
    stage_name: str = "final",
    max_examples: Optional[int] = None,
    output_path: Optional[str] = None,
    load_in_4bit: bool = False,
    max_new_tokens: int = 256,
    batch_size: int = 4,
) -> None:
    """Run evaluation of the fine-tuned Qwen adapter on a JSONL split."""
    if base_model_name is None:
        base_model_name = resolve_base_model(DEFAULT_CONFIG)

    model, tokenizer = load_finetuned_model(base_model_name, adapter_path, load_in_4bit)

    examples = load_eval_examples_from_jsonl(data_path, max_examples)
    log.info("Loaded %d eval examples from %s", len(examples), data_path)

    if output_path is None:
        out_dir = RESULTS_DIR / "combined_val"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / "qwen_report.json")
        md_path = str(out_dir / "report_table.md")
        progress_path = out_dir / "qwen_live_progress.log"
    else:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        md_path = str(output_path_obj.parent / "report_table.md")
        progress_path = output_path_obj.parent / "qwen_live_progress.log"

    run_evaluation(
        model=model,
        tokenizer=tokenizer,
        model_key=MODEL_KEY,
        examples=examples,
        stage_name=stage_name,
        benchmark_label="combined_val",
        output_path=output_path,
        md_path=md_path,
        progress_path=progress_path,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        logger=log,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate fine-tuned Qwen LoRA adapter")
    p.add_argument("--adapter-path", default=DEFAULT_CHECKPOINT,
                   help="Path to the LoRA adapter directory.")
    p.add_argument("--base-model", default=None,
                   help="Base model path (default: resolved from config.yaml).")
    p.add_argument("--data", default=DEFAULT_VAL_DATA,
                   help="JSONL eval data in combined_80_20_split format.")
    p.add_argument("--stage", default="final")
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    evaluate(
        adapter_path=args.adapter_path,
        base_model_name=args.base_model,
        data_path=args.data,
        stage_name=args.stage,
        max_examples=args.max_examples,
        output_path=args.output,
        load_in_4bit=args.load_in_4bit,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
