"""
experiments/finetuning/shared/evaluate_utils.py
=====================================
Evaluation utilities for fine-tuned models.

Wraps ``src.evaluation.metrics.TemporalEvaluator`` when available,
falling back to a simple exact-match evaluator.
"""

from __future__ import annotations

import logging
from pathlib import Path

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
        gold_labels: list,
        categories: list[str] | None = None,
        difficulties: list | None = None,
    ) -> dict:
        """Compute evaluation metrics by delegating to src.evaluation.metrics.

        Fails loudly if the project evaluator cannot be used — a silent
        fallback would risk reporting fake numbers.

        Args:
            predictions: Model predictions.
            gold_labels: Gold answers; each entry may be a string or a list
                of alternative gold answers (any match counts).
            categories: Per-example category labels (e.g. source dataset).
            difficulties: Accepted for signature compatibility; currently unused.
        """
        from src.evaluation.metrics import TemporalEvaluator as _SrcEval

        ev = _SrcEval(benchmark_name=self.benchmark)
        return ev.full_evaluation(
            predictions=predictions,
            gold_labels=gold_labels,
            categories=categories,
            metadata={"model": self.model_key, "stage": self.stage},
        )

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
