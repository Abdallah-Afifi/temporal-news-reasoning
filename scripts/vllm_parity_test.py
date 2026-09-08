"""vLLM parity test: same prompts, same decoding, different engine.

Runs a fixed random sample of a benchmark through vLLM with preprocessing
and postprocessing IDENTICAL to the frozen HF pipeline
(scripts/run_baselines.py), writing a predictions file that
scripts/compare_parity.py (frozen venv) scores against the reference HF
predictions. Run with the venv_vllm interpreter.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_baselines import _build_zero_shot_prompt, _postprocess_prediction  # noqa: E402
from src.data.data_loader import BenchmarkLoader  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/Llama-3.2-3B-Instruct")
    ap.add_argument("--benchmark", default="time")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reference", required=True,
                    help="frozen-engine predictions.jsonl to sample ids from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    ref = {}
    with open(args.reference, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ref[r["id"]] = r

    examples = BenchmarkLoader(str(PROJECT_ROOT / "data" / "benchmarks")).load(args.benchmark)
    rng = random.Random(args.seed)
    sample = rng.sample(examples, min(args.n, len(examples)))
    sample = [ex for ex in sample if ex.id in ref]
    print(f"parity sample: {len(sample)} examples with frozen-engine references")

    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tok.truncation_side = "left"

    token_ids = []
    for ex in sample:
        text = tok.apply_chat_template(
            [{"role": "user", "content": _build_zero_shot_prompt(ex)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tok(text, truncation=True, max_length=4096,
                  add_special_tokens=False)["input_ids"]
        token_ids.append(ids)

    llm = LLM(
        model=args.model_dir,
        dtype="float16",
        max_model_len=4096 + args.max_new_tokens,
        gpu_memory_utilization=args.gpu_mem,
        seed=42,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    outs = llm.generate(
        prompts=[{"prompt_token_ids": i} for i in token_ids],
        sampling_params=sp,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        for ex, out in zip(sample, outs):
            raw = out.outputs[0].text.strip()
            pred = _postprocess_prediction(raw, ex.choices)
            f.write(json.dumps({
                "id": ex.id,
                "benchmark": ex.source,
                "task": ex.task,
                "question": ex.question,
                "context": ex.context,
                "category": ex.temporal_type,
                "prediction": pred,
                "reference": ref[ex.id]["reference"],
                "hf_prediction": ref[ex.id]["prediction"],
            }, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
