"""
experiments/finetuning/LLaMA/evaluate.py
================================
Evaluation script for the fine-tuned LLaMA-3.2-3B LoRA adapter.

Two evaluation modes:

1. **JSONL mode (default, recommended)** — evaluate on a data split in the
   combined_80_20_split format (TLQA / TimeQA schemas). Prompts are built
   with the SAME chat template used during training, so eval-time inputs
   are in-distribution.
2. **Benchmark mode** — TIME / TIMEBENCH / TRAM via BenchmarkLoader.
   NOTE: the benchmark loaders are not implemented yet and the benchmark
   files are not in the repo, so this mode will raise a clear error.

Usage
-----
From the repo root::

    # Evaluate on the held-out combined validation split (default)
    python experiments/finetuning/LLaMA/evaluate.py

    # Custom data file / cap examples / custom output
    python experiments/finetuning/LLaMA/evaluate.py \\
        --data data/combined_80_20_split/val.jsonl --max-examples 100 \\
        --output results/my_llama_eval.json

    # Benchmark mode (requires implemented loader + downloaded data)
    python experiments/finetuning/LLaMA/evaluate.py --benchmark time
"""

from __future__ import annotations

import argparse
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
from experiments.finetuning.shared.eval_runner import (
    load_eval_examples_from_jsonl,
    run_evaluation,
)
from experiments.finetuning.shared.utils import setup_logger

# ---------------------------------------------------------------------------
log = setup_logger("llama.evaluate")

DEFAULT_CHECKPOINT = str(_REPO_ROOT / "checkpoints" / "llama" / "final")
DEFAULT_VAL_DATA   = str(_REPO_ROOT / "data" / "combined_80_20_split" / "val.jsonl")
DEFAULT_CONFIG     = str(Path(__file__).parent / "config.yaml")
RESULTS_DIR        = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_finetuned_model(
    base_model_name: str,
    adapter_path: str,
    load_in_4bit: bool = False,
) -> tuple:
    """Load the base LLaMA model and attach the LoRA adapter.

    The adapter is loaded with ``is_trainable=False`` so inference
    optimisations can be applied without accidentally continuing training.

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


# ---------------------------------------------------------------------------
# Benchmark loading (stub-aware)
# ---------------------------------------------------------------------------

def load_benchmark_examples(
    benchmark: str,
    task: Optional[str] = None,
    max_examples: Optional[int] = None,
) -> list:
    """Load benchmark examples via the project's BenchmarkLoader.

    Raises a clear error until the benchmark loaders are implemented and
    the benchmark data has been downloaded.
    """
    from src.data.data_loader import BenchmarkLoader
    loader = BenchmarkLoader(str(_REPO_ROOT / "data" / "benchmarks"))
    try:
        examples = loader.load(benchmark, task=task)
    except (NotImplementedError, FileNotFoundError) as exc:
        raise NotImplementedError(
            f"Benchmark '{benchmark}' cannot be evaluated yet: the "
            f"BenchmarkLoader is a stub and/or the benchmark data is not "
            f"downloaded. Use JSONL mode instead: "
            f"python experiments/finetuning/LLaMA/evaluate.py --data {DEFAULT_VAL_DATA}"
        ) from exc
    if max_examples:
        examples = examples[:max_examples]
    log.info("Benchmark '%s': loaded %d examples.", benchmark, len(examples))
    return examples


# ---------------------------------------------------------------------------
# Main evaluation entrypoint
# ---------------------------------------------------------------------------

def evaluate(
    adapter_path: str,
    base_model_name: str = HF_MODEL_NAME,
    benchmark: Optional[str] = None,
    data_path: str = DEFAULT_VAL_DATA,
    task: Optional[str] = None,
    stage_name: str = "final",
    max_examples: Optional[int] = None,
    output_path: Optional[str] = None,
    load_in_4bit: bool = False,
    max_new_tokens: int = 256,
    batch_size: int = 4,
) -> None:
    """Run evaluation of the fine-tuned LLaMA adapter.

    Args:
        adapter_path:    Path to the LoRA adapter checkpoint directory.
        base_model_name: HuggingFace ID or local path of the base model.
        benchmark:       If set, benchmark mode (time|timebench|tram).
        data_path:       JSONL mode input (combined_80_20_split format).
        task:            Optional benchmark sub-task filter.
        stage_name:      Stage label for the evaluation report.
        max_examples:    Cap examples for quick testing.
        output_path:     Where to save the JSON report.
        load_in_4bit:    Enable 4-bit inference quantization.
        max_new_tokens:  Token budget for each generated answer.
        batch_size:      Examples per generation call.
    """
    model, tokenizer = load_finetuned_model(base_model_name, adapter_path, load_in_4bit)

    if benchmark is not None:
        benchmark_examples = load_benchmark_examples(benchmark, task, max_examples)
        # Convert benchmark examples to the shared eval format.
        examples = [
            {
                "question": ex.question,
                "context": ex.context or "",
                "golds": (
                    [str(a) for a in ex.answer]
                    if isinstance(ex.answer, list) else [str(ex.answer)]
                ),
                "source": ex.temporal_type or benchmark,
            }
            for ex in benchmark_examples
        ]
        benchmark_label = benchmark.lower()
    else:
        examples = load_eval_examples_from_jsonl(data_path, max_examples)
        benchmark_label = "combined_val"
        log.info("Loaded %d eval examples from %s", len(examples), data_path)

    # Resolve output paths
    if output_path is None:
        out_dir = RESULTS_DIR / benchmark_label
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / "llama_report.json")
        md_path = str(out_dir / "report_table.md")
        progress_path = out_dir / "llama_live_progress.log"
    else:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        md_path = str(output_path_obj.parent / "report_table.md")
        progress_path = output_path_obj.parent / "llama_live_progress.log"

    run_evaluation(
        model=model,
        tokenizer=tokenizer,
        model_key=MODEL_KEY,
        examples=examples,
        stage_name=stage_name,
        benchmark_label=benchmark_label,
        output_path=output_path,
        md_path=md_path,
        progress_path=progress_path,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        logger=log,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate fine-tuned LLaMA-3.2-3B LoRA adapter")
    p.add_argument("--adapter-path", default=DEFAULT_CHECKPOINT,
                   help="Path to the LoRA adapter directory.")
    p.add_argument("--base-model", default=HF_MODEL_NAME,
                   help="Base model name or local path.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--benchmark", default=None,
                      choices=["time", "timebench", "tram"],
                      help="Benchmark to evaluate on (requires loader + data).")
    mode.add_argument("--data", default=DEFAULT_VAL_DATA,
                      help="JSONL eval data in combined_80_20_split format (default mode).")
    p.add_argument("--task", default=None,
                   help="Sub-task filter (benchmark mode only).")
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
    p.add_argument("--batch-size", type=int, default=4,
                   help="Examples per generation batch.")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    evaluate(
        adapter_path=args.adapter_path,
        base_model_name=args.base_model,
        benchmark=args.benchmark,
        data_path=args.data,
        task=args.task,
        stage_name=args.stage,
        max_examples=args.max_examples,
        output_path=args.output,
        load_in_4bit=args.load_in_4bit,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
