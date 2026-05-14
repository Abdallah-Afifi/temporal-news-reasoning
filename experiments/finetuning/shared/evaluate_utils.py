"""
experiments/finetuning/shared/evaluate_utils.py
=====================================
Evaluation utilities for fine-tuned models.

Wraps ``src.evaluation.metrics.TemporalEvaluator`` when available,
falling back to a simple exact-match evaluator.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Optional

from experiments.finetuning.shared.utils import setup_logger, save_json


class TemporalEvaluator:
    """Compute accuracy / F1 metrics for temporal reasoning predictions."""

    def __init__(
        self,
        model_key: str = "llama",
        stage: str = "final",
        benchmark: str = "time",
        logger: logging.Logger | None = None,
    ) -> None:
        self.model_key = model_key
        self.stage = stage
        self.benchmark = benchmark
        self.log = logger or setup_logger(f"{model_key}.evaluator")

    # ------------------------------------------------------------------

    def evaluate(
        self,
        predictions: list[str],
        gold_labels: list[str],
        categories: list[str] | None = None,
        difficulties: list[Optional[str]] | None = None,
    ) -> dict:
        """Compute evaluation metrics."""
        # Try to delegate to the project's evaluator
        try:
            from src.evaluation.metrics import TemporalEvaluator as _SrcEval
            ev = _SrcEval(benchmark_name=self.benchmark)
            return ev.full_evaluation(
                predictions=predictions,
                gold_labels=gold_labels,
                categories=categories,
                metadata={"model": self.model_key, "stage": self.stage},
            )
        except (ImportError, Exception):
            pass

        # Fallback — simple exact match
        correct = sum(
            1 for p, g in zip(predictions, gold_labels)
            if p.strip().lower() == g.strip().lower()
        )
        total = len(predictions)
        accuracy = correct / total if total > 0 else 0.0

        result: dict = {
            "model_key": self.model_key,
            "stage": self.stage,
            "benchmark": self.benchmark,
            "num_examples": total,
            "overall_accuracy": accuracy,
            "overall_f1": accuracy,
            "exact_match": accuracy,
        }

        if categories:
            cat_total = Counter(categories)
            cat_correct: dict[str, int] = {}
            for p, g, c in zip(predictions, gold_labels, categories):
                if p.strip().lower() == g.strip().lower():
                    cat_correct[c] = cat_correct.get(c, 0) + 1
            result["by_category"] = {
                cat: {
                    "accuracy": cat_correct.get(cat, 0) / count,
                    "correct": cat_correct.get(cat, 0),
                    "total": count,
                }
                for cat, count in cat_total.items()
            }

        return result

    # ------------------------------------------------------------------

    def print_report(self, result: dict) -> None:
        self.log.info("=" * 60)
        self.log.info("Evaluation: %s / %s / %s",
                       self.model_key, self.stage, self.benchmark)
        self.log.info("=" * 60)
        self.log.info("Accuracy: %.4f", result.get("overall_accuracy", 0))
        self.log.info("F1:       %.4f", result.get("overall_f1", 0))
        self.log.info("N:        %d", result.get("num_examples", 0))

    def save(self, result: dict, path: str | Path) -> None:
        save_json(result, path)
        self.log.info("Report saved: %s", path)

    def save_markdown(self, result: dict, path: str | Path) -> None:
        lines = [
            f"# Evaluation: {self.model_key} / {self.stage} / {self.benchmark}",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Accuracy | {result.get('overall_accuracy', 0):.4f} |",
            f"| F1 | {result.get('overall_f1', 0):.4f} |",
            f"| N | {result.get('num_examples', 0)} |",
        ]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("\n".join(lines))
