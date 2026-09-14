"""
experiments/finetuning/Mistral/inference.py
==================================
Quick inference testing for the fine-tuned Mistral-7B-Instruct-v0.3 LoRA adapter.

Mistral-specific behaviour:
  - [INST] / [/INST] chat format (no separate system-prompt token).
  - System instructions are folded into the user turn by
    format_inference_prompt(model_key="mistral"), matching the training
    data loader.
  - trust_remote_code NOT required.
  - Merging LoRA weights (merge_and_unload) makes inference faster on CPU/GPU
    but cannot be done with 4-bit quantized models.

Usage::

    python experiments/finetuning/Mistral/inference.py --demo
    python experiments/finetuning/Mistral/inference.py --interactive
    python experiments/finetuning/Mistral/inference.py --question "When did the merger happen?"
    python experiments/finetuning/Mistral/inference.py --input-file data/training/stage3_complex.jsonl --max-items 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from peft import PeftModel
from transformers import AutoModelForCausalLM

from experiments.finetuning.Mistral.data_loader import load_tokenizer, HF_MODEL_NAME, MODEL_KEY
from experiments.finetuning.shared.prompt_templates import format_inference_prompt
from experiments.finetuning.shared.utils import setup_logger, read_jsonl

log = setup_logger("mistral.inference")
DEFAULT_ADAPTER = str(_REPO_ROOT / "checkpoints" / "mistral" / "final")


def load_model_for_inference(
    adapter_path: str,
    base_model_name: str = HF_MODEL_NAME,
    load_in_4bit: bool = False,
):
    """Load Mistral-7B + LoRA adapter and optionally merge weights."""
    tokenizer   = load_tokenizer(base_model_name)
    load_kwargs: dict = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    log.info("Loading Mistral base: %s", base_model_name)
    base  = AutoModelForCausalLM.from_pretrained(base_model_name, **load_kwargs)
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    if not load_in_4bit:
        log.info("Merging LoRA weights...")
        model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


@torch.no_grad()
def run_inference(
    model, tokenizer, prompt: str,
    max_new_tokens: int = 256, do_sample: bool = False, temperature: float = 0.1,
) -> str:
    inputs  = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    gen_kw  = {"max_new_tokens": max_new_tokens, "do_sample": do_sample,
               "pad_token_id": tokenizer.eos_token_id, "eos_token_id": tokenizer.eos_token_id}
    if do_sample:
        gen_kw.update({"temperature": temperature, "top_p": 0.95})
    outputs = model.generate(**inputs, **gen_kw)
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def run_demo_questions(model, tokenizer) -> None:
    """Demo covering all three curriculum stages — Mistral [INST] format."""
    demos = [
        {
            "stage": "Stage 1 (Explicit)",
            "question": "The war ended on November 11, 1918 and began on July 28, 1914. How long did it last in years and months?",
            "context": "",
        },
        {
            "stage": "Stage 2 (Implicit)",
            "question": "The report published on September 1, 2024 says 'sanctions imposed last month'. When were the sanctions imposed?",
            "context": "The government announced sanctions last month following diplomatic talks.",
        },
        {
            "stage": "Stage 3 (Complex)",
            "question": "Based on the following articles, who led the country during the economic crisis?",
            "context": (
                "March 2020: President Adams declares national emergency over pandemic. "
                "July 2020: Economic crisis deepens; unemployment hits 15%. "
                "January 2021: President Baker inaugurated after November election. "
                "April 2021: IMF reports crisis is ongoing under new leadership."
            ),
        },
    ]
    for demo in demos:
        print(f"\n{'='*70}\n  {demo['stage']}\n{'='*70}")
        print(f"Q: {demo['question']}")
        if demo["context"]:
            print(f"C: {demo['context'][:200]}")
        prompt = format_inference_prompt(tokenizer, demo["question"], demo["context"], model_key=MODEL_KEY)
        print(f"\nMistral: {run_inference(model, tokenizer, prompt)}")


def run_batch_inference(model, tokenizer, input_path, max_items=None, output_path=None):
    records = read_jsonl(input_path)
    if max_items:
        records = records[:max_items]
    results = []
    for i, rec in enumerate(records):
        question = rec.get("question", rec.get("instruction", ""))
        context  = rec.get("context", rec.get("input", ""))
        gold     = rec.get("answer", rec.get("output", ""))
        prompt   = format_inference_prompt(tokenizer, question, context, model_key=MODEL_KEY)
        answer   = run_inference(model, tokenizer, prompt)
        print(f"[{i+1}/{len(records)}] Q: {question[:80]}...")
        print(f"  -> {answer[:150]}")
        results.append({"question": question, "gold": gold, "prediction": answer})
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nSaved: {output_path}")


def run_interactive(model, tokenizer) -> None:
    print("\nMistral-7B-Instruct-v0.3 (LoRA fine-tuned) — Temporal Reasoning")
    print("Type 'quit' to exit, 'demo' for demo questions.\n")
    while True:
        question = input("Question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if question.lower() == "demo":
            run_demo_questions(model, tokenizer)
            continue
        if not question:
            continue
        context  = input("Context (Enter to skip): ").strip()
        ref_date = input("Reference date (YYYY-MM-DD, Enter to skip): ").strip() or None
        prompt   = format_inference_prompt(tokenizer, question, context, model_key=MODEL_KEY, reference_date=ref_date)
        print(f"\n-> {run_inference(model, tokenizer, prompt)}\n")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mistral-7B LoRA inference")
    p.add_argument("--adapter-path", default=DEFAULT_ADAPTER)
    p.add_argument("--base-model", default=HF_MODEL_NAME)
    p.add_argument("--load-in-4bit", action="store_true")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--interactive", action="store_true")
    mode.add_argument("--demo", action="store_true")
    mode.add_argument("--question", type=str, default=None)
    mode.add_argument("--input-file", type=str, default=None)
    p.add_argument("--context", type=str, default="")
    p.add_argument("--reference-date", type=str, default=None)
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--output", type=str, default=None)
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    model, tokenizer = load_model_for_inference(
        args.adapter_path, args.base_model, args.load_in_4bit
    )
    if args.demo:
        run_demo_questions(model, tokenizer)
    elif args.interactive:
        run_interactive(model, tokenizer)
    elif args.question:
        prompt = format_inference_prompt(tokenizer, args.question, args.context, model_key=MODEL_KEY, reference_date=args.reference_date)
        print(run_inference(model, tokenizer, prompt))
    elif args.input_file:
        run_batch_inference(model, tokenizer, args.input_file, args.max_items, args.output)
    else:
        run_demo_questions(model, tokenizer)
