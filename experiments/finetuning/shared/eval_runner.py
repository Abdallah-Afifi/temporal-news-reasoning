"""
experiments/finetuning/shared/eval_runner.py
===================================
Shared evaluation loop used by the LLaMA / Qwen / Mistral evaluate scripts.

Runs chat-template inference (matching the training format) over a list of
examples, computes metrics via :class:`TemporalEvaluator`, and writes the
report, markdown summary, raw predictions, and a live progress log.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from experiments.finetuning.shared.evaluate_utils import TemporalEvaluator
from experiments.finetuning.shared.generation import generate_answer_batch
from experiments.finetuning.shared.prompt_templates import format_inference_prompt
from experiments.finetuning.shared.utils import read_jsonl, save_json


def load_eval_examples_from_jsonl(
    path: str | Path,
    max_examples: Optional[int] = None,
) -> list[dict]:
    """Load a combined_80_20_split-style JSONL into eval examples.

    Handles the TLQA / TimeQA schemas (Temprel records are skipped) via the
    LLaMA data loader's normalizer. Each example carries ALL gold variants
    (the joined answer string the model was trained on, plus individual
    answer parts) so metrics can credit any correct alternative.

    Returns a list of ``{question, context, golds, source}`` dicts.
    """
    from experiments.finetuning.LLaMA.data_loader import normalize_record

    records = read_jsonl(path)
    examples: list[dict] = []
    skipped = 0

    for rec in records:
        norm = normalize_record(rec)
        if norm is None:
            skipped += 1
            continue
        golds = [norm["answer"]]
        for part in norm.get("answer_parts", []):
            if part not in golds:
                golds.append(part)
        examples.append({
            "question": norm["question"],
            "context": norm["context"],
            "golds": golds,
            "source": rec.get("source_dataset", "unknown"),
        })

    if max_examples:
        examples = examples[:max_examples]
    return examples


def _log_progress(
    prog_file: Path,
    model_key: str,
    current: int,
    total: int,
    start_t: float,
) -> None:
    elapsed_sec = max(time.time() - start_t, 0.001)
    rate = current / elapsed_sec
    eta_sec = (total - current) / rate if rate > 0 else 0
    pct = (current / total) * 100 if total else 0.0
    msg = (
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {model_key} | "
        f"progress={current}/{total} ({pct:6.3f}%) | rate={rate:6.2f} ex/s | "
        f"elapsed={timedelta(seconds=int(elapsed_sec))} | eta={timedelta(seconds=int(eta_sec))}\n"
    )
    with open(prog_file, "a") as f:
        f.write(msg)


def run_evaluation(
    model,
    tokenizer,
    model_key: str,
    examples: list[dict],
    stage_name: str,
    benchmark_label: str,
    output_path: str | Path,
    md_path: str | Path,
    progress_path: str | Path,
    batch_size: int = 4,
    max_new_tokens: int = 256,
    logger=None,
) -> dict:
    """Run chat-template inference over ``examples`` and produce a report.

    Args:
        examples: List of ``{question, context, golds, source}`` dicts (see
            :func:`load_eval_examples_from_jsonl`).
        benchmark_label: Label for the report (e.g. "combined_val" or "time").
    Returns:
        The evaluation report dict.
    """
    predictions: list[str] = []
    gold_labels: list[list[str]] = []
    categories: list[str] = []

    progress_path = Path(progress_path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_path, "w") as f:
        f.write(f"Starting {model_key} evaluation on {len(examples)} examples (batch={batch_size})...\n")

    start_time = time.time()
    for i in range(0, len(examples), batch_size):
        batch = examples[i: i + batch_size]

        prompts = [
            format_inference_prompt(
                tokenizer, ex["question"], ex["context"] or "", model_key=model_key,
            )
            for ex in batch
        ]
        batch_preds = generate_answer_batch(
            model, tokenizer, prompts, max_new_tokens=max_new_tokens,
        )
        predictions.extend(batch_preds)

        for ex in batch:
            gold_labels.append([str(g) for g in ex["golds"]])
            categories.append(ex.get("source") or "unknown")

        current_idx = i + len(batch)
        if (i % (batch_size * 5) == 0) or (current_idx >= len(examples)):
            _log_progress(progress_path, model_key, current_idx, len(examples), start_time)

    evaluator = TemporalEvaluator(
        model_key=model_key, stage=stage_name, benchmark=benchmark_label,
    )
    result = evaluator.evaluate(predictions, gold_labels, categories)
    evaluator.print_report(result)
    evaluator.save(result, output_path)
    evaluator.save_markdown(result, md_path)

    raw_preds_path = Path(output_path).with_name("predictions.json")
    save_json(
        [
            {
                "question": ex["question"],
                "prediction": p,
                "golds": g,
                "category": c,
            }
            for ex, p, g, c in zip(examples, predictions, gold_labels, categories)
        ],
        raw_preds_path,
    )
    if logger:
        logger.info("Raw predictions saved: %s", raw_preds_path)

    return result
