"""
experiments/finetuning/LLaMA/evaluate.py
================================
Evaluation script for the fine-tuned LLaMA-3.2-3B LoRA adapter.

Runs the model on benchmark datasets (TIME / TIMEBENCH / TRAM) and computes:
  - Exact-match accuracy
  - Macro-averaged token-F1
  - Per-category accuracy breakdown
  - Per-difficulty breakdown
  - Temporal consistency score (for self-consistency experiments)

Usage
-----
From the repo root::

    # Evaluate final adapter on TIME benchmark
    python experiments/finetuning/LLaMA/evaluate.py --stage final --benchmark time

    # Evaluate a specific stage checkpoint on TIMEBENCH
    python experiments/finetuning/LLaMA/evaluate.py --stage stage3_complex --benchmark timebench

    # Limit examples for quick testing
    python experiments/finetuning/LLaMA/evaluate.py --max-examples 100

    # Save results to a custom path
    python experiments/finetuning/LLaMA/evaluate.py --output results/my_llama_eval.json
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from peft import PeftModel
from transformers import AutoModelForCausalLM

from experiments.finetuning.LLaMA.data_loader import load_tokenizer, HF_MODEL_NAME, MODEL_KEY
from experiments.finetuning.shared.evaluate_utils import TemporalEvaluator
from experiments.finetuning.shared.prompt_templates import build_temporal_cot_prompt
from experiments.finetuning.shared.utils import load_yaml, setup_logger, save_json

# ---------------------------------------------------------------------------
log = setup_logger("llama.evaluate")

DEFAULT_CHECKPOINT = str(_REPO_ROOT / "checkpoints" / "llama" / "final")
DEFAULT_CONFIG     = str(Path(__file__).parent / "config.yaml")
RESULTS_DIR        = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def load_finetuned_model(
    base_model_name: str,
    adapter_path: str,
    load_in_4bit: bool = False,
) -> tuple:
    """Load the base LLaMA model and attach the LoRA adapter.

    We load the adapter with ``is_trainable=False`` so weight merging and
    inference optimisations can be applied without accidentally continuing
    training.

    Args:
        base_model_name: HuggingFace ID or local path to the base model.
        adapter_path:    Path to the saved LoRA adapter directory.
        load_in_4bit:    Enable bitsandbytes 4-bit inference.

    Returns:
        Tuple of (model, tokenizer).
    """
    tokenizer = load_tokenizer(base_model_name)

    load_kwargs: dict = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
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


@torch.no_grad()
def generate_answer_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    do_sample: bool = False,
) -> list[str]:
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(model.device)

    gen_kwargs: dict = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.95

    outputs = model.generate(**inputs, **gen_kwargs)

    new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
    tokenizer.padding_side = original_padding_side
    return [tokenizer.decode(t, skip_special_tokens=True).strip() for t in new_tokens]


# ---------------------------------------------------------------------------
# Benchmark loading helper
# ---------------------------------------------------------------------------

def load_benchmark_examples(
    benchmark: str,
    task: Optional[str] = None,
    max_examples: Optional[int] = None,
) -> list:
    """Load benchmark examples using the project's BenchmarkLoader.

    Args:
        benchmark:    One of ``"time"``, ``"timebench"``, ``"tram"``.
        task:         Optional sub-task filter.
        max_examples: Cap the number of examples (useful for quick tests).

    Returns:
        List of :class:`src.data.data_loader.TemporalExample` objects.
    """
    from src.data.data_loader import BenchmarkLoader
    loader = BenchmarkLoader(str(_REPO_ROOT / "data" / "benchmarks"))
    examples = loader.load(benchmark, task=task)
    if max_examples:
        examples = examples[:max_examples]
    log.info("Benchmark '%s': loaded %d examples.", benchmark, len(examples))
    return examples


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    adapter_path: str,
    base_model_name: str = HF_MODEL_NAME,
    benchmark: str = "time",
    task: Optional[str] = None,
    stage_name: str = "final",
    max_examples: Optional[int] = None,
    output_path: Optional[str] = None,
    load_in_4bit: bool = False,
    max_new_tokens: int = 256,
) -> None:
    """Run full evaluation of the fine-tuned LLaMA adapter.

    Args:
        adapter_path:    Path to the LoRA adapter checkpoint directory.
        base_model_name: HuggingFace ID or local path of the base model.
        benchmark:       Benchmark to evaluate on (time|timebench|tram).
        task:            Optional sub-task filter.
        stage_name:      Stage label for the evaluation report.
        max_examples:    Cap examples for quick testing.
        output_path:     Where to save the JSON report.
        load_in_4bit:    Enable 4-bit inference quantization.
        max_new_tokens:  Token budget for each generated answer.
    """
    model, tokenizer = load_finetuned_model(base_model_name, adapter_path, load_in_4bit)

    examples = load_benchmark_examples(benchmark, task, max_examples)

    predictions: list[str] = []
    gold_labels: list[str] = []
    categories:  list[str] = []
    difficulties: list[Optional[str]] = []

    # Save report
    if output_path is None:
        benchmark_lower = benchmark.lower()
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        benchmark_dir = RESULTS_DIR / benchmark_lower
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = str(benchmark_dir / f"llama_report.json")
        md_path = str(benchmark_dir / "report_table.md")
        progress_path = benchmark_dir / "llama_live_progress.log"
    else:
        # If custom path is provided, try to make a corresponding md path
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        md_path = str(output_path_obj.parent / "report_table.md")
        progress_path = output_path_obj.parent / "llama_live_progress.log"

    log.info("Running inference on %d examples…", len(examples))
    
    batch_size = 4
    def log_eval_progress(prog_file, current: int, total: int, start_t: float):
        now = datetime.now()
        elapsed_sec = time.time() - start_t
        if elapsed_sec == 0: elapsed_sec = 0.001
        rate = current / elapsed_sec
        eta_sec = (total - current) / rate if rate > 0 else 0
        pct = (current / total) * 100
        
        msg = f"{now.strftime('%Y-%m-%d %H:%M:%S')} | progress={current}/{total} ({pct:6.3f}%) | run={current}/{total} ({pct:6.3f}%) | rate={rate:6.2f} ex/s | elapsed={timedelta(seconds=int(elapsed_sec))} | eta={timedelta(seconds=int(eta_sec))}\n"
        with open(prog_file, "a") as f:
            f.write(msg)

    with open(progress_path, "w") as f:
        f.write(f"Starting evaluation on {len(examples)} examples (Batch size = {batch_size})...\n")

    start_time = time.time()
    for i in range(0, len(examples), batch_size):
        batch_ex = examples[i : i + batch_size]
        
        prompts = [
            build_temporal_cot_prompt(
                model_key=MODEL_KEY,
                question=ex.question,
                context=ex.context or "",
            )
            for ex in batch_ex
        ]
        
        batch_preds = generate_answer_batch(model, tokenizer, prompts, max_new_tokens=max_new_tokens)
        predictions.extend(batch_preds)

        for ex in batch_ex:
            gold = ex.answer[0] if isinstance(ex.answer, list) else ex.answer
            gold_labels.append(str(gold))
            categories.append(ex.temporal_type or "unknown")
            difficulties.append(ex.difficulty)
            
        current_idx = i + len(batch_ex)
        if (i % (batch_size * 5) == 0) or (current_idx >= len(examples)):
            log_eval_progress(progress_path, current_idx, len(examples), start_time)

    # Compute metrics
    evaluator = TemporalEvaluator(
        model_key=MODEL_KEY,
        stage=stage_name,
        benchmark=benchmark,
    )
    result = evaluator.evaluate(predictions, gold_labels, categories, difficulties)
    evaluator.print_report(result)

    # Save report
    evaluator.save(result, output_path)
    evaluator.save_markdown(result, md_path)

    # Also save raw predictions alongside the report for error analysis.
    raw_preds_path = Path(output_path).with_name("predictions.json")
    save_json(
        [{"question": ex.question, "prediction": p, "gold": g, "category": c}
         for ex, p, g, c in zip(examples, predictions, gold_labels, categories)],
        raw_preds_path,
    )
    log.info("Raw predictions saved: %s", raw_preds_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate fine-tuned LLaMA-3.2-3B LoRA adapter")
    p.add_argument("--adapter-path", default=DEFAULT_CHECKPOINT,
                   help="Path to the LoRA adapter directory.")
    p.add_argument("--base-model", default=HF_MODEL_NAME,
                   help="Base model name or local path.")
    p.add_argument("--benchmark", default="time",
                   choices=["time", "timebench", "tram"],
                   help="Benchmark to evaluate on.")
    p.add_argument("--task", default=None,
                   help="Sub-task filter (e.g. 'TimeQA' for TIMEBENCH).")
    p.add_argument("--stage", default="final",
                   help="Stage label for the report (e.g. 'stage3_complex').")
    p.add_argument("--max-examples", type=int, default=None,
                   help="Cap number of examples (useful for quick testing).")
    p.add_argument("--output", default=None,
                   help="Output JSON path for the evaluation report.")
    p.add_argument("--load-in-4bit", action="store_true",
                   help="Enable 4-bit inference (reduces VRAM).")
    p.add_argument("--max-new-tokens", type=int, default=256,
                   help="Max tokens per generation call.")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    evaluate(
        adapter_path=args.adapter_path,
        base_model_name=args.base_model,
        benchmark=args.benchmark,
        task=args.task,
        stage_name=args.stage,
        max_examples=args.max_examples,
        output_path=args.output,
        load_in_4bit=args.load_in_4bit,
        max_new_tokens=args.max_new_tokens,
    )
