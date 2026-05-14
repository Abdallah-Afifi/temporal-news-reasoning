"""
experiments/finetuning/Mistral/evaluate.py
=================================
Evaluation script for the fine-tuned Mistral-7B-Instruct-v0.3 LoRA adapter.

Usage::

    python experiments/finetuning/Mistral/evaluate.py --benchmark time
    python experiments/finetuning/Mistral/evaluate.py --benchmark timebench --max-examples 200
    python experiments/finetuning/Mistral/evaluate.py --load-in-4bit --benchmark tram
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
import sys
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from peft import PeftModel
from transformers import AutoModelForCausalLM

from experiments.finetuning.Mistral.data_loader import load_tokenizer, HF_MODEL_NAME, MODEL_KEY
from experiments.finetuning.shared.evaluate_utils import TemporalEvaluator
from experiments.finetuning.shared.prompt_templates import build_temporal_cot_prompt
from experiments.finetuning.shared.utils import setup_logger, save_json

log = setup_logger("mistral.evaluate")
DEFAULT_CHECKPOINT = str(_REPO_ROOT / "checkpoints" / "mistral" / "final")
RESULTS_DIR        = Path(__file__).parent / "results"


def load_finetuned_model(
    base_model_name: str,
    adapter_path: str,
    load_in_4bit: bool = False,
) -> tuple:
    """Load Mistral base model + LoRA adapter for evaluation.

    No trust_remote_code needed. Adapter is loaded in non-trainable mode
    for pure inference.
    """
    tokenizer = load_tokenizer(base_model_name)
    load_kwargs: dict = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    log.info("Loading Mistral base model: %s", base_model_name)
    base  = AutoModelForCausalLM.from_pretrained(base_model_name, **load_kwargs)
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_answer_batch(model, tokenizer, prompts: list[str], max_new_tokens: int = 256) -> list[str]:
    """Generate answers for a batch of prompts.
    Uses left-padding to properly align batches for causal LM decoding.
    """
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
    
    tokenizer.padding_side = original_padding_side
    return [tokenizer.decode(t, skip_special_tokens=True).strip() for t in new_tokens]


def evaluate(
    adapter_path: str = DEFAULT_CHECKPOINT,
    base_model_name: str = HF_MODEL_NAME,
    benchmark: str = "time",
    task: Optional[str] = None,
    category: Optional[str] = None,
    stage_name: str = "final",
    max_examples: Optional[int] = None,
    output_path: Optional[str] = None,
    load_in_4bit: bool = False,
    max_new_tokens: int = 256,
    batch_size: int = 4,
) -> None:
    """Run benchmark evaluation for the fine-tuned Mistral model."""
    model, tokenizer = load_finetuned_model(base_model_name, adapter_path, load_in_4bit)

    from src.data.data_loader import BenchmarkLoader
    loader   = BenchmarkLoader(str(_REPO_ROOT / "data" / "benchmarks"))
    examples = loader.load(benchmark, task=task)
    if category:
        category_lower = category.lower()
        examples = [ex for ex in examples if ex.temporal_type.lower() == category_lower]
    if max_examples:
        examples = examples[:max_examples]
    log.info("Benchmark '%s': %d examples.", benchmark, len(examples))
    if category:
        log.info("Category filter: %s", category)

    predictions, gold_labels, categories, difficulties = [], [], [], []
    
    if output_path is None:
        benchmark_lower = benchmark.lower()
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        benchmark_dir = RESULTS_DIR / benchmark_lower
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(benchmark_dir / f"mistral_report.json")
        md_path = str(benchmark_dir / "report_table.md")
        progress_path = benchmark_dir / "mistral_live_progress.log"
    else:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        md_path = str(output_path_obj.parent / "report_table.md")
        progress_path = output_path_obj.parent / "mistral_live_progress.log"

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
        log.info(msg.strip())

    with open(progress_path, "w") as f:
        f.write(f"Starting Mistral evaluation on {len(examples)} examples (BS={batch_size})...\n")

    start_time = time.time()
    for i in range(0, len(examples), batch_size):
        batch_ex = examples[i : i + batch_size]
        prompts = [build_temporal_cot_prompt(MODEL_KEY, ex.question, ex.context or "") for ex in batch_ex]
        batch_preds = generate_answer_batch(model, tokenizer, prompts, max_new_tokens)
        predictions.extend(batch_preds)
        
        for ex in batch_ex:
            gold = ex.answer[0] if isinstance(ex.answer, list) else ex.answer
            gold_labels.append(str(gold))
            categories.append(ex.temporal_type or "unknown")
            difficulties.append(ex.difficulty)

        current_idx = i + len(batch_ex)
        if (i % (batch_size * 5) == 0) or (current_idx >= len(examples)):
            log_eval_progress(progress_path, current_idx, len(examples), start_time)

    evaluator = TemporalEvaluator(MODEL_KEY, stage_name, benchmark)
    result    = evaluator.evaluate(predictions, gold_labels, categories, difficulties)
    evaluator.print_report(result)

    evaluator.save(result, output_path)
    evaluator.save_markdown(result, md_path)

    raw_path = Path(output_path).with_name("predictions.json")
    save_json(
        [{"question": ex.question, "prediction": p, "gold": g}
         for ex, p, g in zip(examples, predictions, gold_labels)],
        raw_path,
    )
    log.info("Predictions saved: %s", raw_path)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate fine-tuned Mistral-7B LoRA adapter")
    p.add_argument("--adapter-path", default=DEFAULT_CHECKPOINT)
    p.add_argument("--base-model", default=HF_MODEL_NAME)
    p.add_argument("--benchmark", default="time", choices=["time", "timebench", "tram"])
    p.add_argument("--task", default=None)
    p.add_argument("--category", default=None, help="Optional temporal_type filter, e.g. temporal_qa.")
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
        benchmark=args.benchmark,
        task=args.task,
        category=args.category,
        stage_name=args.stage,
        max_examples=args.max_examples,
        output_path=args.output,
        load_in_4bit=args.load_in_4bit,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
