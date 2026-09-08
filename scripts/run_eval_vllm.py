"""Full vLLM evaluation runner — engine counterpart of run_baselines.py.

Reproduces the frozen HF pipeline exactly: same prompt builder, same NLI
instructions, same chat template + optional training system prompt, same
left-truncation at 4096, greedy, max_new_tokens 128, same postprocessing
and predictions.jsonl schema. Resume-safe by id. Run with venv_vllm.
Reports are computed afterwards in the frozen venv (compare_parity.py).
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_baselines import (  # noqa: E402
    _build_zero_shot_prompt,
    _canonical_golds,
    _postprocess_prediction,
    _prediction_record,
)
from src.data.data_loader import BenchmarkLoader  # noqa: E402

CHUNK = 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--benchmark", required=True, choices=["time", "timebench", "tram"])
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--model-name", default="llama")
    ap.add_argument("--adapter-dir", default=None,
                    help="evaluated adapter (uses its training system prompt)")
    ap.add_argument("--batch-size", default=32, type=int, help="compat no-op")
    ap.add_argument("--token-budget", default=0, type=int, help="compat no-op")
    ap.add_argument("--max-new-tokens", default=128, type=int)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-samples", default=None, type=int)
    args = ap.parse_args()

    system_prompt = None
    if args.adapter_dir:
        from experiments.finetuning.shared.prompt_templates import (
            TEMPORAL_SYSTEM_PROMPT,
        )
        system_prompt = TEMPORAL_SYSTEM_PROMPT

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    examples = BenchmarkLoader(str(PROJECT_ROOT / "data" / "benchmarks")).load(args.benchmark)
    if args.max_samples is not None:
        examples = examples[: args.max_samples]
    print(f"benchmark={args.benchmark} examples={len(examples)}")

    config_subdir = "finetuned" if args.adapter_dir else "zero_shot"
    pred_path = (
        Path(args.results_dir) / args.model_name / args.benchmark
        / config_subdir / "predictions.jsonl"
    )
    pred_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if pred_path.exists():
        with open(pred_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    continue
    todo = [ex for ex in examples if ex.id not in done_ids]
    print(f"resume: {len(done_ids)} done, {len(todo)} todo")

    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tok.truncation_side = "left"

    def encode(ex):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": _build_zero_shot_prompt(ex)})
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return tok(text, truncation=True, max_length=4096,
                   add_special_tokens=False)["input_ids"]

    llm = LLM(
        model=args.model_dir,
        dtype="float16",
        max_model_len=4096 + args.max_new_tokens,
        gpu_memory_utilization=args.gpu_mem,
        seed=42,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    t0 = time.time()
    with open(pred_path, "a", encoding="utf-8") as out:
        for ci in range(0, len(todo), CHUNK):
            chunk = todo[ci : ci + CHUNK]
            ids = [encode(ex) for ex in chunk]
            outs = llm.generate(
                prompts=[{"prompt_token_ids": i} for i in ids],
                sampling_params=sp,
            )
            for ex, o in zip(chunk, outs):
                raw = o.outputs[0].text.strip()
                pred = _postprocess_prediction(raw, ex.choices)
                rec = _prediction_record(ex, pred, _canonical_golds(ex), raw_prediction=raw)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            done = len(done_ids) + ci + len(chunk)
            el = time.time() - t0
            rate = (ci + len(chunk)) / el if el else 0.0
            print(
                f"{datetime.datetime.now():%H:%M:%S} | {done}/{len(examples)} "
                f"({100 * done / len(examples):.2f}%) | rate={rate:.2f} ex/s | "
                f"elapsed={datetime.timedelta(seconds=int(el))}",
                flush=True,
            )
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
