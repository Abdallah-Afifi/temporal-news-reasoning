"""Evaluation metrics for temporal reasoning.

Implements the evaluation metrics defined in the project plan (Task A1.6):
  - Accuracy, F1, Exact-match (standard QA metrics)
  - Temporal F1 (date/time extraction correctness)
  - Timeline order accuracy (Kendall's tau)
  - Temporal consistency score (self-consistency across answer chains)
  - Category-wise breakdown
  - Efficiency metrics (cost / latency)
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import f1_score as sklearn_f1

# ---------------------------------------------------------------------------
# Temporal-expression extraction helpers
# ---------------------------------------------------------------------------

_DATE_PATTERNS: list[re.Pattern] = [
    # ISO-style: 2024-03-15, 2024/03/15
    re.compile(r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b"),
    # Written-out: March 15, 2024 or 15 March 2024
    re.compile(
        r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{4})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b((?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{1,2},?\s+\d{4})\b",
        re.IGNORECASE,
    ),
    # Year-month: March 2024
    re.compile(
        r"\b((?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{4})\b",
        re.IGNORECASE,
    ),
    # Standalone year: 2024
    re.compile(r"\b((?:19|20)\d{2})\b"),
]


def _coerce_text(value: Any) -> str:
    """Convert optional or non-string values into a safe text form."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def extract_temporal_expressions(text: Any) -> list[str]:
    """Extract date/time expressions from *text* using regex heuristics.

    Returns a deduplicated list of matched temporal strings.
    """
    text = _coerce_text(text)
    found: list[str] = []
    for pat in _DATE_PATTERNS:
        found.extend(pat.findall(text))
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for expr in found:
        normed = expr.strip().lower()
        if normed not in seen:
            seen.add(normed)
            unique.append(expr.strip())
    return unique


def _normalize_answer(text: Any) -> str:
    """SQuAD-style answer normalization: lowercase, drop articles, strip ALL
    punctuation (incl. internal commas), collapse whitespace.

    Fixed in the Sep-3-2026 audit: the previous version kept internal
    punctuation, so golds like ``Tuesday , 25th September , 2001`` never
    matched predictions like ``Tuesday, 25th September, 2001`` — an
    arm-dependent bias that inflated the v2/v3 TIME degradation.
    """
    text = _coerce_text(text).lower()
    # Letter-sequence golds (MCQ: "B,C,A") must NOT have article removal —
    # the letter "a" would be eaten and distinct sequences would collide.
    if re.fullmatch(r"[a-d][\s,;.\-]*(?:[a-d][\s,;.\-]*)*", text.strip()):
        text = re.sub(r"[^a-d]+", "", text)
        return " ".join(text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_references(reference: Any) -> list[str]:
    """Normalize a gold reference into a list of acceptable answer strings.

    Accepts either a single string/None or a list of alternative golds.
    A prediction matching ANY entry counts as correct.
    """
    if isinstance(reference, (list, tuple)):
        norms = [_normalize_answer(r) for r in reference if _coerce_text(r).strip()]
        return norms if norms else [""]
    return [_normalize_answer(reference)]


# ---------------------------------------------------------------------------
# Individual metric functions (referenced in evaluation_config.yaml)
# ---------------------------------------------------------------------------


def accuracy(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Compute accuracy (exact string match after normalisation).

    Args:
        predictions: Model predictions.
        references: Gold-standard answers.

    Returns:
        Accuracy in [0, 1].
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if len(predictions) == 0:
        return 0.0
    preds_norm = [_normalize_answer(p) for p in predictions]
    correct = sum(
        1 for p, r in zip(preds_norm, references) if p in _normalize_references(r)
    )
    return correct / len(preds_norm)


def f1_score(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Compute macro-averaged F1 score.

    Treats each unique answer string as a class label and computes
    sklearn macro F1 across all classes. References may be single strings
    or lists of alternative golds; for multi-gold examples the *effective*
    reference is the gold that matches the prediction (if any), else the
    first gold — so a correct alternative is never penalized.

    Args:
        predictions: Model predictions.
        references: Gold-standard answers (str or list[str] per example).

    Returns:
        Macro F1 in [0, 1].
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if len(predictions) == 0:
        return 0.0
    preds_norm = [_normalize_answer(p) for p in predictions]
    effective_refs: list[str] = []
    for p, r in zip(preds_norm, references):
        refs_norm = _normalize_references(r)
        effective_refs.append(p if p in refs_norm else refs_norm[0])
    labels = sorted(set(effective_refs) | set(preds_norm))
    return float(
        sklearn_f1(effective_refs, preds_norm, labels=labels, average="macro", zero_division=0)
    )


def exact_match(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Compute exact-match (EM) score.

    Identical to *accuracy* for single-answer QA but provided as a
    separate function so the config can reference it by name.
    """
    return accuracy(predictions, references)


def temporal_f1(
    predictions: Sequence[str],
    references: Sequence[str],
) -> float:
    """Compute temporal-specific F1.

    For each (prediction, reference) pair the function extracts temporal
    expressions and computes token-level precision / recall of the
    *temporal* tokens.  The final score is the average F1 across all
    examples.

    Args:
        predictions: Free-text model outputs.
        references: Gold-standard answers.

    Returns:
        Macro-averaged temporal F1 in [0, 1].
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if len(predictions) == 0:
        return 0.0

    f1_scores: list[float] = []
    for pred, ref in zip(predictions, references):
        pred_dates = set(d.lower() for d in extract_temporal_expressions(pred))
        per_ref: list[float] = []
        for ref_option in _normalize_references(ref):
            # Re-extract from the ORIGINAL reference strings where possible
            # (normalized refs still contain the date text).
            ref_dates = set(
                d.lower() for d in extract_temporal_expressions(ref_option)
            )
            if not ref_dates and not pred_dates:
                per_ref.append(1.0)  # Neither has dates → agree
                continue
            if not ref_dates or not pred_dates:
                per_ref.append(0.0)
                continue
            tp = len(pred_dates & ref_dates)
            precision = tp / len(pred_dates)
            recall = tp / len(ref_dates)
            if precision + recall == 0:
                per_ref.append(0.0)
            else:
                per_ref.append(2 * precision * recall / (precision + recall))
        f1_scores.append(max(per_ref) if per_ref else 0.0)
    return float(np.mean(f1_scores))


def timeline_order_accuracy(
    predicted_timeline: Sequence[str],
    reference_timeline: Sequence[str],
) -> float:
    """Compute timeline ordering accuracy via Kendall's tau.

    Both timelines are sequences of event identifiers.  The function
    measures how well the predicted ordering matches the reference
    ordering.  Returns (tau + 1) / 2 to map from [-1, 1] to [0, 1].

    Args:
        predicted_timeline: Event IDs in predicted order.
        reference_timeline: Event IDs in gold order.

    Returns:
        Normalised Kendall's tau in [0, 1].
    """
    if len(predicted_timeline) != len(reference_timeline):
        raise ValueError("Timelines must have the same length")
    if len(predicted_timeline) <= 1:
        return 1.0

    ref_index = {event: i for i, event in enumerate(reference_timeline)}
    pred_ranks = []
    ref_ranks = []
    for i, event in enumerate(predicted_timeline):
        if event not in ref_index:
            raise ValueError(f"Event '{event}' not found in reference timeline")
        pred_ranks.append(i)
        ref_ranks.append(ref_index[event])

    tau, _ = kendalltau(pred_ranks, ref_ranks)
    if np.isnan(tau):
        return 1.0  # identical rankings (all tied)
    return float((tau + 1.0) / 2.0)


# ---------------------------------------------------------------------------
# TemporalEvaluator — main evaluator class (from project plan sketch)
# ---------------------------------------------------------------------------


class TemporalEvaluator:
    """Evaluates model predictions on temporal reasoning benchmarks.

    Produces a comprehensive report including overall accuracy, F1,
    category-wise breakdown, temporal consistency, and efficiency
    metrics as outlined in the project proposal.
    """

    # Mapping from config metric names → callables
    METRIC_REGISTRY: dict[str, Any] = {
        "accuracy": accuracy,
        "f1": f1_score,
        "exact_match": exact_match,
        "temporal_f1": temporal_f1,
        "timeline_order_accuracy": timeline_order_accuracy,
    }

    def __init__(self, benchmark_name: str) -> None:
        self.benchmark_name = benchmark_name
        self.results: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Core metric helpers
    # ------------------------------------------------------------------

    def compute_accuracy(
        self, predictions: Sequence[str], gold_labels: Sequence[str]
    ) -> float:
        """Overall accuracy."""
        return accuracy(predictions, gold_labels)

    def compute_f1(
        self, predictions: Sequence[str], gold_labels: Sequence[str]
    ) -> float:
        """Macro F1 score."""
        return f1_score(predictions, gold_labels)

    def compute_by_category(
        self,
        predictions: Sequence[str],
        gold_labels: Sequence[str],
        categories: Sequence[str],
    ) -> dict[str, dict[str, float | int]]:
        """Accuracy broken down by temporal reasoning category.

        Returns a dict mapping each category to its accuracy and count.
        """
        cat_results: dict[str, dict[str, int]] = defaultdict(
            lambda: {"correct": 0, "total": 0}
        )
        for pred, gold, cat in zip(predictions, gold_labels, categories):
            cat = _coerce_text(cat) or "unknown"
            cat_results[cat]["total"] += 1
            if _normalize_answer(pred) in _normalize_references(gold):
                cat_results[cat]["correct"] += 1
        return {
            cat: {
                "accuracy": vals["correct"] / vals["total"] if vals["total"] else 0.0,
                "correct": vals["correct"],
                "total": vals["total"],
            }
            for cat, vals in cat_results.items()
        }

    def compute_temporal_consistency(
        self, answer_chains: Sequence[Sequence[str]]
    ) -> float:
        """Temporal consistency across multiple answer chains.

        Given *N* answer chains (e.g. from self-consistency sampling),
        extract temporal expressions from each and measure pairwise
        agreement.  The score is the fraction of chain-pairs that
        produce the same set of temporal expressions.

        Args:
            answer_chains: List of answer chains; each chain is a
                sequence of answer strings.

        Returns:
            Consistency score in [0, 1].
        """
        if len(answer_chains) <= 1:
            return 1.0

        chain_date_sets: list[frozenset[str]] = []
        for chain in answer_chains:
            dates: set[str] = set()
            for answer in chain:
                dates.update(d.lower() for d in extract_temporal_expressions(answer))
            chain_date_sets.append(frozenset(dates))

        agree = 0
        total = 0
        for a, b in combinations(chain_date_sets, 2):
            total += 1
            if a == b:
                agree += 1
        return agree / total if total else 1.0

    def compute_efficiency_metrics(
        self,
        total_tokens: int,
        total_time_seconds: float,
        cost_per_1k_tokens: float = 0.0002,
    ) -> dict[str, float]:
        """Compute efficiency / cost metrics.

        Mirrors the *Efficiency Metrics* table in the proposal.

        Args:
            total_tokens: Total tokens generated during evaluation.
            total_time_seconds: Wall-clock time for the evaluation run.
            cost_per_1k_tokens: Estimated USD cost per 1K tokens.

        Returns:
            Dict with throughput, latency and cost stats.
        """
        num_k_tokens = total_tokens / 1000.0
        return {
            "total_tokens": total_tokens,
            "total_time_seconds": round(total_time_seconds, 2),
            "tokens_per_second": (
                round(total_tokens / total_time_seconds, 1)
                if total_time_seconds > 0
                else 0.0
            ),
            "cost_per_1k_tokens": cost_per_1k_tokens,
            "total_cost_usd": round(num_k_tokens * cost_per_1k_tokens, 6),
        }

    # ------------------------------------------------------------------
    # Error analysis helpers
    # ------------------------------------------------------------------

    def classify_errors(
        self,
        predictions: Sequence[str],
        gold_labels: Sequence[str],
        contexts: Sequence[str] | None = None,
    ) -> dict[str, list[int]]:
        """Classify incorrect predictions into error categories.

        Error types (from proposal §Qualitative Error Analysis):
          - temporal_extraction: missed / wrong dates
          - hallucination: prediction contains dates not in context
          - timeline_ordering: event order incorrect
          - other: uncategorised

        Args:
            predictions: Model predictions.
            gold_labels: Gold-standard answers.
            contexts: Input contexts (for hallucination detection).

        Returns:
            Dict mapping error type → list of example indices.
        """
        errors: dict[str, list[int]] = {
            "temporal_extraction": [],
            "hallucination": [],
            "other": [],
        }

        for i, (pred, gold) in enumerate(zip(predictions, gold_labels)):
            if _normalize_answer(pred) in _normalize_references(gold):
                continue  # correct — skip

            pred_dates = set(d.lower() for d in extract_temporal_expressions(pred))
            gold_dates: set[str] = set()
            for gold_option in _normalize_references(gold):
                gold_dates.update(
                    d.lower() for d in extract_temporal_expressions(gold_option)
                )

            # Temporal extraction error: missed gold dates
            if gold_dates and not gold_dates.issubset(pred_dates):
                errors["temporal_extraction"].append(i)
                continue

            # Hallucination: predicted dates not present in context
            if contexts is not None and pred_dates:
                ctx_dates = set(
                    d.lower() for d in extract_temporal_expressions(contexts[i])
                )
                hallucinated = pred_dates - ctx_dates - gold_dates
                if hallucinated:
                    errors["hallucination"].append(i)
                    continue

            errors["other"].append(i)

        return errors

    # ------------------------------------------------------------------
    # Full evaluation report
    # ------------------------------------------------------------------

    def full_evaluation(
        self,
        predictions: Sequence[str],
        gold_labels: Sequence[str],
        categories: Sequence[str] | None = None,
        metadata: dict[str, Any] | None = None,
        answer_chains: Sequence[Sequence[str]] | None = None,
        contexts: Sequence[str] | None = None,
        efficiency_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run all metrics and return a comprehensive report.

        Args:
            predictions: Model predictions.
            gold_labels: Gold-standard answers.
            categories: Per-example temporal reasoning category.
            metadata: Arbitrary metadata to include in the report.
            answer_chains: Multiple answer chains for consistency.
            contexts: Input contexts for error analysis.
            efficiency_kwargs: Kwargs for :meth:`compute_efficiency_metrics`.

        Returns:
            Evaluation report dict.
        """
        report: dict[str, Any] = {
            "benchmark": self.benchmark_name,
            "timestamp": datetime.now().isoformat(),
            "num_examples": len(predictions),
            "overall_accuracy": self.compute_accuracy(predictions, gold_labels),
            "overall_f1": self.compute_f1(predictions, gold_labels),
            "exact_match": exact_match(predictions, gold_labels),
            "temporal_f1": temporal_f1(predictions, gold_labels),
        }

        if categories is not None:
            report["by_category"] = self.compute_by_category(
                predictions, gold_labels, categories
            )

        if answer_chains is not None:
            report["temporal_consistency"] = self.compute_temporal_consistency(
                answer_chains
            )

        if efficiency_kwargs is not None:
            report["efficiency"] = self.compute_efficiency_metrics(**efficiency_kwargs)

        # Error analysis
        error_indices = self.classify_errors(predictions, gold_labels, contexts)
        report["error_analysis"] = {
            etype: len(indices) for etype, indices in error_indices.items()
        }

        if metadata is not None:
            report["metadata"] = metadata

        self.results = report
        return report

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_report(self, report: dict[str, Any], path: str | Path) -> None:
        """Save evaluation report as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)

    @staticmethod
    def load_report(path: str | Path) -> dict[str, Any]:
        """Load a previously saved evaluation report."""
        with open(path) as f:
            return json.load(f)
