"""
experiments/finetuning/LLaMA/inference.py
================================
Quick inference testing script for the fine-tuned LLaMA-3.2-3B LoRA adapter.

Designed for manual verification after training — paste a custom question,
run the script, and confirm the model's temporal reasoning output looks correct
before committing to a full evaluation run.

Usage
-----
From the repo root::

    # Interactive mode (prompts for input)
    python experiments/finetuning/LLaMA/inference.py --interactive

    # Single question
    python experiments/finetuning/LLaMA/inference.py \\
        --question "The article was published on March 5, 2024. It says the CEO resigned recently. When did the CEO resign?" \\
        --context "The CEO of TechCorp announced his resignation last Tuesday."

    # Batch from a JSONL file
    python experiments/finetuning/LLaMA/inference.py \\
        --input-file data/training/stage1_explicit.jsonl \\
        --max-items 10

    # Use 4-bit quantization for low-VRAM inference
    python experiments/finetuning/LLaMA/inference.py --interactive --load-in-4bit

    # Compare base model vs fine-tuned (useful for ablation)
    python experiments/finetuning/LLaMA/inference.py --compare-base --question "..."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from peft import PeftModel
from transformers import AutoModelForCausalLM

from experiments.finetuning.LLaMA.data_loader import load_tokenizer, HF_MODEL_NAME, MODEL_KEY
from experiments.finetuning.shared.prompt_templates import format_inference_prompt
from experiments.finetuning.shared.utils import setup_logger, read_jsonl

# ---------------------------------------------------------------------------
log = setup_logger("llama.inference")
DEFAULT_ADAPTER = str(_REPO_ROOT / "checkpoints" / "llama" / "final")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_for_inference(
    adapter_path: str,
    base_model_name: str = HF_MODEL_NAME,
    load_in_4bit: bool = False,
):
    """Load the fine-tuned LLaMA model for inference.

    Merges the LoRA weights into the base model for faster inference
    (``merge_and_unload``). This is optional — keeping them separate uses
    less disk space but is slightly slower at inference time.

    Args:
        adapter_path:    Path to the saved LoRA adapter.
        base_model_name: Base model identifier.
        load_in_4bit:    Enable 4-bit quantization for inference.

    Returns:
        Tuple of (model, tokenizer).
    """
    tokenizer = load_tokenizer(base_model_name)

    load_kwargs: dict = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    log.info("Loading base LLaMA model: %s", base_model_name)
    base = AutoModelForCausalLM.from_pretrained(base_model_name, **load_kwargs)

    log.info("Loading LoRA adapter: %s", adapter_path)
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)

    # Merge weights for faster inference (skip if using 4-bit — merging with
    # quantized tensors is not supported by bitsandbytes).
    if not load_in_4bit:
        log.info("Merging LoRA weights into base model for faster inference…")
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Single inference call
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    do_sample: bool = False,
    num_beams: int = 1,
) -> str:
    """Generate a response for a single formatted prompt.

    Args:
        model:          Loaded (merged) LLaMA model.
        tokenizer:      LLaMA tokenizer.
        prompt:         Fully formatted chat-template string.
        max_new_tokens: Maximum generation length.
        temperature:    Sampling temperature (only used when do_sample=True).
        do_sample:      Nucleus sampling toggle.
        num_beams:      Beam search width (1 = greedy decoding).

    Returns:
        Decoded response string.
    """
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    input_len = inputs["input_ids"].shape[1]

    gen_kwargs: dict = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "num_beams": num_beams,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample and num_beams == 1:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.95

    outputs = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Demo helpers
# ---------------------------------------------------------------------------

def run_demo_questions(model, tokenizer) -> None:
    """Run a fixed set of demo questions covering all three curriculum stages.

    Useful for quickly verifying the model's temporal reasoning capability
    after training, without running the full evaluation suite.
    """
    demos = [
        # Stage 1 — Explicit date arithmetic
        {
            "stage": "Stage 1 (Explicit)",
            "question": "The peace treaty was signed on July 4, 2020. "
                        "The conflict had started on March 1, 2018. "
                        "How long did the conflict last?",
            "context": "",
        },
        # Stage 2 — Implicit temporal reference
        {
            "stage": "Stage 2 (Implicit)",
            "question": "The article says 'The CEO resigned last Tuesday'. "
                        "The article was published on Friday, March 14, 2025. "
                        "What is the approximate date of the resignation?",
            "context": "The CEO of GlobalTech announced his resignation last Tuesday, "
                       "citing personal reasons. The board has appointed an interim CEO.",
        },
        # Stage 3 — Complex multi-document reasoning
        {
            "stage": "Stage 3 (Complex)",
            "question": "Based on the articles below, who was Prime Minister "
                        "during the trade summit in 2023?",
            "context": "Article A (Jan 2022): John Smith became Prime Minister after the election. "
                       "Article B (Aug 2023): The trade summit opened in Geneva. "
                       "Article C (Nov 2023): Prime Minister Smith signed the agreement. "
                       "Article D (Feb 2024): New elections: Jane Doe replaces Smith.",
        },
    ]

    for demo in demos:
        print(f"\n{'='*70}")
        print(f"  {demo['stage']}")
        print(f"{'='*70}")
        print(f"QUESTION: {demo['question']}")
        if demo["context"]:
            print(f"CONTEXT : {demo['context'][:200]}…")

        prompt = format_inference_prompt(
            tokenizer,
            question=demo["question"],
            context=demo["context"],
            model_key=MODEL_KEY,
        )
        answer = run_inference(model, tokenizer, prompt)
        print(f"\nLLaMA ANSWER:\n{answer}")


# ---------------------------------------------------------------------------
# Batch inference from JSONL
# ---------------------------------------------------------------------------

def run_batch_inference(
    model,
    tokenizer,
    input_path: str,
    max_items: Optional[int] = None,
    output_path: Optional[str] = None,
) -> None:
    """Run inference on a JSONL file and optionally save results.

    Args:
        model:       Fine-tuned model.
        tokenizer:   LLaMA tokenizer.
        input_path:  Path to a JSONL file with {"question", "context"} records.
        max_items:   Maximum number of items to process.
        output_path: If set, save results to this JSONL file.
    """
    records = read_jsonl(input_path)
    if max_items:
        records = records[:max_items]

    results = []
    for i, rec in enumerate(records):
        question = rec.get("question", rec.get("instruction", ""))
        context  = rec.get("context", rec.get("input", ""))
        gold     = rec.get("answer", rec.get("output", ""))

        prompt = format_inference_prompt(tokenizer, question, context, model_key=MODEL_KEY)
        answer = run_inference(model, tokenizer, prompt)

        print(f"\n[{i+1}/{len(records)}] Q: {question[:100]}…")
        print(f"  → {answer[:200]}")

        results.append({
            "question": question,
            "context":  context,
            "gold":     gold,
            "prediction": answer,
        })

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nResults saved: {output_path}")


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def run_interactive(model, tokenizer) -> None:
    """Start an interactive inference session in the terminal."""
    print("\nLLaMA-3.2-3B-Instruct (LoRA fine-tuned) — Temporal Reasoning")
    print("Type 'quit' or 'exit' to stop. Type 'demo' to run demo questions.\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if question.lower() == "demo":
            run_demo_questions(model, tokenizer)
            continue
        if not question:
            continue

        context = input("Context (press Enter to skip): ").strip()
        ref_date = input("Reference date (YYYY-MM-DD, or Enter to skip): ").strip() or None

        prompt = format_inference_prompt(
            tokenizer, question, context, model_key=MODEL_KEY,
            reference_date=ref_date,
        )
        print("\nGenerating…")
        answer = run_inference(model, tokenizer, prompt)
        print(f"\n→ {answer}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Interactive / batch inference with fine-tuned LLaMA-3.2-3B"
    )
    p.add_argument("--adapter-path", default=DEFAULT_ADAPTER,
                   help="Path to LoRA adapter directory.")
    p.add_argument("--base-model", default=HF_MODEL_NAME,
                   help="Base model identifier or local path.")
    p.add_argument("--load-in-4bit", action="store_true",
                   help="Enable 4-bit QLoRA inference.")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--interactive", action="store_true",
                      help="Start interactive terminal session.")
    mode.add_argument("--demo", action="store_true",
                      help="Run fixed demo questions for all 3 curriculum stages.")
    mode.add_argument("--question", type=str, default=None,
                      help="Single question to answer.")
    mode.add_argument("--input-file", type=str, default=None,
                      help="JSONL file for batch inference.")

    p.add_argument("--context", type=str, default="",
                   help="Context for --question mode.")
    p.add_argument("--reference-date", type=str, default=None,
                   help="Reference date (YYYY-MM-DD) for --question mode.")
    p.add_argument("--max-items", type=int, default=None,
                   help="Cap items in --input-file mode.")
    p.add_argument("--output", type=str, default=None,
                   help="Output JSONL path for --input-file mode.")
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
        prompt = format_inference_prompt(
            tokenizer, args.question, args.context, model_key=MODEL_KEY,
            reference_date=args.reference_date,
        )
        print(run_inference(model, tokenizer, prompt))
    elif args.input_file:
        run_batch_inference(model, tokenizer, args.input_file, args.max_items, args.output)
    else:
        # Default: run demo
        run_demo_questions(model, tokenizer)
