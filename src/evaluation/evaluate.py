"""Evaluation harness — run models against benchmarks.

Task A1.7: Script that takes model predictions + gold labels, computes all
metrics, and produces report tables.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Sequence

import yaml

from src.evaluation.metrics import TemporalEvaluator

logger = logging.getLogger(__name__)


class EvaluationHarness:
    """Run evaluation across benchmarks and models.

    The harness reads ``evaluation_config.yaml``, iterates over the
    configured benchmarks / models, invokes :class:`TemporalEvaluator`
    for each run, and persists the results under ``results_dir``.

    Typical workflow::

        harness = EvaluationHarness("./configs/evaluation_config.yaml")
        report  = harness.evaluate("qwen", "timebench", task="temporal_qa")
        harness.evaluate_all()              # run everything in config
        harness.compare_results(["qwen", "phi", "llama"])
    """

    def __init__(self, config_path: str = "./configs/evaluation_config.yaml") -> None:
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.results_dir = Path(self.config.get("output", {}).get("results_dir", "./results"))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._all_reports: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    def _load_config(self) -> dict[str, Any]:
        """Load evaluation configuration from YAML."""
        if not self.config_path.exists():
            logger.warning("Config file %s not found; using defaults.", self.config_path)
            return {}
        with open(self.config_path) as f:
            return yaml.safe_load(f) or {}

    # ------------------------------------------------------------------
    # Prediction loading helpers
    # ------------------------------------------------------------------

    @staticmethod
    def load_predictions(path: str | Path) -> list[dict[str, Any]]:
        """Load prediction JSONL / JSON file.

        Each record is expected to have at least:
          - ``prediction``: model output string
          - ``reference`` / ``gold``: gold-standard answer
        Optional:
          - ``category``: temporal reasoning type
          - ``context``: input context
          - ``answer_chains``: list of sampled answer strings
        """
        path = Path(path)
        records: list[dict[str, Any]] = []
        if path.suffix == ".jsonl":
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        else:
            with open(path) as f:
                data = json.load(f)
                records = data if isinstance(data, list) else data.get("predictions", [])
        return records

    @staticmethod
    def _extract_fields(
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Extract parallel lists from prediction records."""
        predictions = [r.get("prediction", r.get("answer", "")) for r in records]
        references = [r.get("reference", r.get("gold", "")) for r in records]
        categories = [r.get("category") for r in records]
        has_categories = any(c is not None for c in categories)
        contexts = [r.get("context") for r in records]
        has_contexts = any(c is not None for c in contexts)

        # Answer chains for self-consistency
        answer_chains: list[list[str]] | None = None
        if any("answer_chains" in r for r in records):
            answer_chains = [r.get("answer_chains", []) for r in records]

        return {
            "predictions": predictions,
            "references": references,
            "categories": categories if has_categories else None,
            "contexts": contexts if has_contexts else None,
            "answer_chains": answer_chains,
        }

    # ------------------------------------------------------------------
    # Single evaluation run
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model_name: str,
        benchmark: str,
        task: str | None = None,
        predictions_path: str | Path | None = None,
        predictions: Sequence[str] | None = None,
        references: Sequence[str] | None = None,
        categories: Sequence[str] | None = None,
        contexts: Sequence[str] | None = None,
        answer_chains: Sequence[Sequence[str]] | None = None,
        efficiency_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate a model on a benchmark task.

        Supply data either via *predictions_path* (JSONL / JSON file) or
        directly via *predictions*, *references*, etc.

        Args:
            model_name: Name tag for the model being evaluated.
            benchmark: Benchmark name (``time``, ``timebench``, ``tram``).
            task: Optional sub-task within the benchmark.
            predictions_path: Path to predictions file.
            predictions: List of prediction strings (alternative).
            references: List of gold-standard strings (alternative).
            categories: Per-example temporal reasoning categories.
            contexts: Input contexts for error analysis.
            answer_chains: Sampled answer chains for consistency.
            efficiency_kwargs: Passed to efficiency metric computation.

        Returns:
            Evaluation report dict.
        """
        # If file-based, load and extract fields
        if predictions_path is not None:
            records = self.load_predictions(predictions_path)
            fields = self._extract_fields(records)
            predictions = fields["predictions"]
            references = fields["references"]
            categories = categories or fields["categories"]
            contexts = contexts or fields["contexts"]
            answer_chains = answer_chains or fields["answer_chains"]

        if predictions is None or references is None:
            raise ValueError("Must provide predictions and references (directly or via file).")

        label = f"{benchmark}/{task}" if task else benchmark
        evaluator = TemporalEvaluator(benchmark_name=label)

        logger.info(
            "Evaluating %s on %s (%d examples)…",
            model_name,
            label,
            len(predictions),
        )
        t0 = time.time()
        report = evaluator.full_evaluation(
            predictions=list(predictions),
            gold_labels=list(references),
            categories=list(categories) if categories is not None else None,
            metadata={"model": model_name, "task": task},
            answer_chains=(
                [list(c) for c in answer_chains] if answer_chains is not None else None
            ),
            contexts=list(contexts) if contexts is not None else None,
            efficiency_kwargs=efficiency_kwargs,
        )
        elapsed = time.time() - t0
        report["eval_wall_time_seconds"] = round(elapsed, 2)

        # Persist
        if self.config.get("output", {}).get("save_metrics", True):
            out_dir = self.results_dir / model_name / benchmark
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{task or 'all'}_report.json"
            evaluator.save_report(report, out_dir / fname)
            logger.info("Report saved to %s", out_dir / fname)

        self._all_reports.append(report)
        return report

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    def evaluate_all(
        self,
        predictions_dir: str | Path = "./results/predictions",
    ) -> list[dict[str, Any]]:
        """Run all configured evaluations.

        Expects prediction files at::

            <predictions_dir>/<model_name>/<benchmark>/<task>.jsonl

        Iterates over every (model, benchmark, task) tuple from the
        config and evaluates each.

        Returns:
            List of all evaluation report dicts.
        """
        predictions_dir = Path(predictions_dir)
        benchmarks = self.config.get("benchmarks", {})
        models = []
        # Collect model names from baselines config
        for baseline_cfg in self.config.get("baselines", {}).values():
            if isinstance(baseline_cfg, dict) and baseline_cfg.get("enabled", False):
                models.extend(baseline_cfg.get("models", []))
        models = sorted(set(models)) or ["qwen"]

        reports: list[dict[str, Any]] = []
        for model_name in models:
            for bench_name, bench_cfg in benchmarks.items():
                tasks = bench_cfg.get("tasks", [None])
                for task in tasks:
                    pred_file = predictions_dir / model_name / bench_name / f"{task or 'all'}.jsonl"
                    if not pred_file.exists():
                        logger.warning(
                            "Predictions file not found: %s — skipping.", pred_file
                        )
                        continue
                    report = self.evaluate(
                        model_name=model_name,
                        benchmark=bench_name,
                        task=task,
                        predictions_path=pred_file,
                    )
                    reports.append(report)

        self._all_reports.extend(reports)
        return reports

    # ------------------------------------------------------------------
    # Cross-model comparison
    # ------------------------------------------------------------------

    def compare_results(
        self,
        model_names: Sequence[str],
        benchmark: str | None = None,
    ) -> dict[str, dict[str, float]]:
        """Build a comparison table across models.

        Scans previously saved reports and returns a dict of model name
        to key metrics.

        Args:
            model_names: Models to compare.
            benchmark: Filter to a specific benchmark (optional).

        Returns:
            ``{model_name: {metric: value, ...}, ...}``
        """
        comparison: dict[str, dict[str, float]] = {}
        for model in model_names:
            model_dir = self.results_dir / model
            if not model_dir.exists():
                logger.warning("No results found for model %s", model)
                continue

            reports: list[dict[str, Any]] = []
            for report_path in sorted(model_dir.rglob("*_report.json")):
                report = TemporalEvaluator.load_report(report_path)
                if benchmark and report.get("benchmark", "").split("/")[0] != benchmark:
                    continue
                reports.append(report)

            if not reports:
                continue

            # Aggregate across tasks (macro-average)
            comparison[model] = {
                "accuracy": sum(r["overall_accuracy"] for r in reports) / len(reports),
                "f1": sum(r["overall_f1"] for r in reports) / len(reports),
                "temporal_f1": sum(r.get("temporal_f1", 0.0) for r in reports) / len(reports),
                "num_benchmarks": len(reports),
            }

        return comparison

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def summary_table(self) -> str:
        """Return a Markdown table summarising all collected reports."""
        if not self._all_reports:
            return "_No reports collected yet._"

        lines = [
            "| Model | Benchmark | Accuracy | F1 | Temporal F1 | N |",
            "|-------|-----------|----------|----|-------------|---|",
        ]
        for r in self._all_reports:
            model = r.get("metadata", {}).get("model", "?")
            bench = r.get("benchmark", "?")
            lines.append(
                f"| {model} | {bench} "
                f"| {r.get('overall_accuracy', 0):.3f} "
                f"| {r.get('overall_f1', 0):.3f} "
                f"| {r.get('temporal_f1', 0):.3f} "
                f"| {r.get('num_examples', 0)} |"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Report table generation (A1.7 + A1.8)
    # ------------------------------------------------------------------

    @staticmethod
    def generate_report_tables(
        report: dict[str, Any],
        output_dir: str | Path | None = None,
    ) -> str:
        """Generate Markdown report tables from an evaluation report.

        Produces tables for:
          - Overall metrics
          - By temporal reasoning category
          - Error analysis breakdown
          - Efficiency metrics (if present)

        If *output_dir* is given, the Markdown is also written to
        ``<output_dir>/report_tables.md``.

        Returns:
            The full Markdown string.
        """
        sections: list[str] = []
        model = report.get("metadata", {}).get("model", "unknown")
        bench = report.get("benchmark", "unknown")

        sections.append(f"# Evaluation Report: {model} on {bench}")
        sections.append(f"")
        sections.append(f"Generated: {report.get('timestamp', 'n/a')}")
        sections.append(f"Num examples: {report.get('num_examples', 0)}")
        sections.append("")

        # --- Overall metrics table ---
        sections.append("## Overall Metrics")
        sections.append("")
        sections.append("| Metric | Value |")
        sections.append("|--------|-------|")
        for key in ("overall_accuracy", "overall_f1", "exact_match", "temporal_f1"):
            if key in report:
                sections.append(f"| {key.replace('_', ' ').title()} | {report[key]:.4f} |")
        if "temporal_consistency" in report:
            sections.append(
                f"| Temporal Consistency | {report['temporal_consistency']:.4f} |"
            )
        sections.append("")

        # --- By category ---
        if "by_category" in report:
            sections.append("## Results by Temporal Reasoning Type")
            sections.append("")
            sections.append("| Category | Accuracy | Correct | Total |")
            sections.append("|----------|----------|---------|-------|")
            for cat, vals in sorted(report["by_category"].items()):
                sections.append(
                    f"| {cat} | {vals['accuracy']:.4f} | {vals['correct']} | {vals['total']} |"
                )
            sections.append("")

        # --- Error analysis ---
        if "error_analysis" in report:
            sections.append("## Error Analysis")
            sections.append("")
            sections.append("| Error Type | Count |")
            sections.append("|------------|-------|")
            for etype, count in sorted(report["error_analysis"].items()):
                sections.append(f"| {etype} | {count} |")
            sections.append("")

        # --- Efficiency ---
        if "efficiency" in report:
            sections.append("## Efficiency Metrics")
            sections.append("")
            sections.append("| Metric | Value |")
            sections.append("|--------|-------|")
            for k, v in report["efficiency"].items():
                sections.append(f"| {k.replace('_', ' ').title()} | {v} |")
            sections.append("")

        md = "\n".join(sections)

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "report_tables.md").write_text(md)

        return md
