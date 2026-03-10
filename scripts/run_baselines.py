"""Run baseline evaluations (zero-shot, few-shot) on benchmarks."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any, TYPE_CHECKING

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import click
try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

if TYPE_CHECKING:
    from src.data.data_loader import TemporalExample

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


MODEL_ALIASES: dict[str, str] = {
    "qwen": "qwen2.5",
    "qwen2.5": "qwen2.5",
    "mistral": "mistral",
    "llama": "llama",
}

MODEL_RUNTIME_CONFIG: dict[str, dict[str, str]] = {
    "qwen2.5": {
        "inference_key": "qwen2.5-3b",
        "model_dir": "models/Qwen2.5-3B-Instruct",
    },
    "mistral": {
        "inference_key": "mistral",
        "model_dir": "models/Mistral-7B-Instruct-v0.3",
    },
    "llama": {
        "inference_key": "llama",
        "model_dir": "models/Llama-3.2-3B-Instruct",
    },
}


def _normalize_model_choice(model: str) -> list[str]:
    if model == "all":
        return ["qwen2.5", "mistral", "llama"]
    return [MODEL_ALIASES[model]]


def _normalize_benchmark_choice(benchmark: str) -> list[str]:
    return ["time", "timebench", "tram"] if benchmark == "all" else [benchmark]


def _build_zero_shot_prompt(example: "TemporalExample") -> str:
    prompt_parts: list[str] = [
        "You are a temporal reasoning assistant.",
        "Answer the question using only the provided context when available.",
        "Return only the final answer with no explanation.",
    ]
    if example.context:
        prompt_parts.append(f"Context:\n{example.context}")
    prompt_parts.append(f"Question: {example.question}")
    if example.choices:
        formatted_choices = "\n".join(
            f"{chr(65 + idx)}. {choice}" for idx, choice in enumerate(example.choices)
        )
        prompt_parts.append(f"Choices:\n{formatted_choices}")
        prompt_parts.append("Respond with the option text, or a single letter (A/B/C/D).")
    return "\n\n".join(prompt_parts)


def _canonical_gold(example: "TemporalExample") -> str:
    answer = example.answer
    if isinstance(answer, list):
        return str(answer[0]).strip() if answer else ""

    answer_str = str(answer).strip()
    if example.choices and len(answer_str) == 1 and answer_str.upper() in "ABCD":
        idx = ord(answer_str.upper()) - ord("A")
        if 0 <= idx < len(example.choices):
            return example.choices[idx]
    return answer_str


def _postprocess_prediction(raw_prediction: str, choices: list[str] | None) -> str:
    prediction = (raw_prediction or "").strip()
    if not prediction:
        return ""

    first_line = prediction.splitlines()[0].strip()
    if not choices:
        return first_line

    upper = first_line.upper()
    if upper and upper[0] in "ABCD":
        idx = ord(upper[0]) - ord("A")
        if 0 <= idx < len(choices):
            return choices[idx]

    for choice in choices:
        if first_line.lower() == choice.lower():
            return choice

    for choice in choices:
        if choice.lower() in first_line.lower():
            return choice

    return first_line


def _prediction_record(example: "TemporalExample", prediction: str, reference: str) -> dict[str, Any]:
    return {
        "id": example.id,
        "benchmark": example.source,
        "task": example.task,
        "question": example.question,
        "context": example.context,
        "category": example.temporal_type,
        "prediction": prediction,
        "reference": reference,
    }


def _template_for_benchmark(benchmark: str, results_dir: Path) -> dict[str, Any]:
    template_path = results_dir / f"qwen_{benchmark}_template.json"
    if template_path.exists():
        with open(template_path, encoding="utf-8") as f:
            return json.load(f)
    return {
        "benchmark": benchmark,
        "model": None,
        "configuration": "zero_shot",
        "timestamp": None,
        "num_examples": 0,
        "overall_accuracy": None,
        "overall_f1": None,
        "exact_match": None,
        "temporal_f1": None,
        "by_category": {},
        "error_analysis": {},
        "metadata": {},
    }


def _save_templated_report(
    report: dict[str, Any],
    benchmark: str,
    model_name: str,
    results_dir: Path,
    max_new_tokens: int,
    temperature: float,
) -> Path:
    merged = _template_for_benchmark(benchmark, results_dir)
    merged.update(
        {
            "benchmark": report.get("benchmark", benchmark),
            "model": model_name,
            "configuration": "zero_shot",
            "timestamp": report.get("timestamp"),
            "num_examples": report.get("num_examples", 0),
            "overall_accuracy": report.get("overall_accuracy"),
            "overall_f1": report.get("overall_f1"),
            "exact_match": report.get("exact_match"),
            "temporal_f1": report.get("temporal_f1"),
            "by_category": report.get("by_category", {}),
            "error_analysis": report.get("error_analysis", {}),
            "metadata": {
                **merged.get("metadata", {}),
                **report.get("metadata", {}),
                "model": model_name,
                "prompt_type": "zero_shot",
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            },
        }
    )

    output_path = results_dir / f"{model_name}_{benchmark}_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    return output_path


def _save_predictions(predictions: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


@click.command()
@click.option("--model", type=click.Choice(["qwen", "qwen2.5", "mistral", "llama", "all"]), default="all")
@click.option("--benchmark", type=click.Choice(["time", "timebench", "tram", "all"]), default="all")
@click.option("--mode", type=click.Choice(["zero_shot", "few_shot", "all"]), default="zero_shot")
@click.option("--data-dir", default="./data/benchmarks", show_default=True)
@click.option("--results-dir", default="./results/baseline/zero_shot", show_default=True)
@click.option("--model-root", default="./models", show_default=True)
@click.option("--eval-config", default=None, help="Optional path to evaluation YAML config.")
@click.option("--batch-size", default=4, show_default=True, type=int)
@click.option("--max-new-tokens", default=128, show_default=True, type=int)
@click.option("--temperature", default=0.0, show_default=True, type=float)
@click.option("--device", default="auto", show_default=True)
@click.option("--load-in-4bit/--no-load-in-4bit", default=False, show_default=True)
@click.option("--max-samples", default=None, type=int, help="Optional cap per benchmark for quick debug runs.")
def run_baselines(
    model: str,
    benchmark: str,
    mode: str,
    data_dir: str,
    results_dir: str,
    model_root: str,
    eval_config: str | None,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    device: str,
    load_in_4bit: bool,
    max_samples: int | None,
) -> None:
    """Run baseline evaluations for zero-shot temporal reasoning."""
    from src.data.data_loader import BenchmarkLoader
    from src.evaluation.evaluate import EvaluationHarness
    from src.models.inference import SLMInference

    if mode not in ("zero_shot", "all"):
        raise click.ClickException("Only zero_shot is currently implemented in this runner.")

    selected_models = _normalize_model_choice(model)
    selected_benchmarks = _normalize_benchmark_choice(benchmark)

    loader = BenchmarkLoader(data_dir)
    harness = EvaluationHarness(eval_config or "./configs/evaluation_config.yaml")
    if eval_config is None:
        harness.config = {
            "output": {
                "save_metrics": True,
                "save_predictions": True,
                "results_dir": str(results_dir),
            }
        }
        logger.info("Running with internal runtime evaluation settings (ignoring YAML config).")
    else:
        logger.info("Using evaluation config: %s", eval_config)

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    harness.results_dir = results_path

    logger.info("Running zero-shot baselines for models=%s on benchmarks=%s", selected_models, selected_benchmarks)

    for model_name in selected_models:
        runtime_cfg = MODEL_RUNTIME_CONFIG[model_name]
        model_dir = Path(model_root) / Path(runtime_cfg["model_dir"]).name
        logger.info("Loading model '%s' from %s", model_name, model_dir)
        inference = SLMInference(
            model_key=runtime_cfg["inference_key"],
            device=device,
            load_in_4bit=load_in_4bit,
            model_dir=str(model_dir),
        )

        for benchmark_name in selected_benchmarks:
            examples = loader.load(benchmark_name)
            if max_samples is not None:
                examples = examples[:max_samples]

            logger.info(
                "Benchmark '%s': %d examples for model '%s'",
                benchmark_name,
                len(examples),
                model_name,
            )

            prompts = [_build_zero_shot_prompt(ex) for ex in examples]
            references = [_canonical_gold(ex) for ex in examples]
            categories = [ex.temporal_type for ex in examples]
            contexts = [ex.context for ex in examples]

            predictions: list[str] = []
            prediction_records: list[dict[str, Any]] = []

            for start in tqdm(range(0, len(prompts), batch_size), desc=f"{model_name}/{benchmark_name}"):
                end = start + batch_size
                batch_prompts = prompts[start:end]
                batch_examples = examples[start:end]
                raw_outputs = inference.batch_generate(
                    batch_prompts,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                )

                for example, raw_output, reference in zip(batch_examples, raw_outputs, references[start:end]):
                    final_prediction = _postprocess_prediction(raw_output, example.choices)
                    predictions.append(final_prediction)
                    prediction_records.append(
                        _prediction_record(example, final_prediction, reference)
                    )

            report = harness.evaluate(
                model_name=model_name,
                benchmark=benchmark_name,
                task="zero_shot",
                predictions=predictions,
                references=references,
                categories=categories,
                contexts=contexts,
                efficiency_kwargs=None,
                answer_chains=None,
            )

            pred_path = results_path / f"{model_name}_{benchmark_name}_predictions.jsonl"
            _save_predictions(prediction_records, pred_path)
            report_path = _save_templated_report(
                report=report,
                benchmark=benchmark_name,
                model_name=model_name,
                results_dir=results_path,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

            md_path_dir = results_path / model_name / benchmark_name
            md = EvaluationHarness.generate_report_tables(report, md_path_dir)
            (md_path_dir / "zero_shot_report.md").write_text(md, encoding="utf-8")

            logger.info("Saved predictions: %s", pred_path)
            logger.info("Saved report: %s", report_path)

    logger.info("Zero-shot baseline evaluation complete.")


if __name__ == "__main__":
    run_baselines()
