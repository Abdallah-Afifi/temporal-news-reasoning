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
import fcntl
import json
import os
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
    ap.add_argument("--system-prompt", choices=["auto", "none"], default="auto", help=(
        "auto: legacy behaviour (training system prompt iff --adapter-dir). "
        "none: never add one -- REQUIRED for eval_parity arms, which were "
        "trained on the exact zero-shot prompt (shared/eval_parity.py)."))
    ap.add_argument("--date-string", default=None, help=(
        "Pin the chat template's 'Today Date' (eval_parity arms and their "
        "zero-shot reference use '26 Jul 2024'). Unset = wall-clock date, the "
        "legacy behaviour, which makes prompts differ between run days."))
    ap.add_argument("--ids-file", default=None, help=(
        "JSON {benchmark: [ids]}; evaluate ONLY these ids (the HPO dev split, "
        "data/hpo_dev/dev_ids.json). Unlike --max-samples, which takes a biased "
        "dataset-order prefix, this is the stratified subset."))
    args = ap.parse_args()

    system_prompt = None
    if args.adapter_dir:
        from experiments.finetuning.shared.prompt_templates import (
            TEMPORAL_SYSTEM_PROMPT,
        )
        if args.system_prompt == "auto":
            system_prompt = TEMPORAL_SYSTEM_PROMPT
        # vLLM loads ONLY what is at --model-dir. The adapter is NOT merged in
        # here: passing an un-merged adapter dir evaluates the BASE model with
        # the training prompt (audit 2026-09-09 M1). Merge first
        # (scripts/merge_lora.py) or verify --model-dir points at a merged
        # checkpoint.
        print(
            "WARNING: --adapter-dir selects the training system prompt only; "
            "LoRA weights are NOT loaded. --model-dir must be a MERGED "
            "checkpoint for a finetuned run."
        )
        # ...and now actually enforce it (audit 2026-09-12). The warning above
        # was the only thing standing between a typo and a full benchmark run
        # of the BASE model written into a `finetuned/` results dir -- the
        # failure mode is silent and costs a re-run of the whole leg. The
        # adapter records the base it was trained on; if --model-dir resolves
        # to that same directory, no LoRA weights can possibly be loaded.
        cfg = Path(args.adapter_dir) / "adapter_config.json"
        if cfg.exists():
            base = json.loads(cfg.read_text()).get("base_model_name_or_path")
            if base:
                try:
                    same = Path(base).resolve() == Path(args.model_dir).resolve()
                except OSError:
                    same = str(base) == str(args.model_dir)
                if same:
                    raise SystemExit(
                        f"FATAL: --model-dir ({args.model_dir}) is the BASE model "
                        f"this adapter was trained on, so this run would evaluate "
                        f"the base model and store it as a fine-tuned arm. "
                        f"Merge first: scripts/merge_lora.py --adapter-dir "
                        f"{args.adapter_dir}"
                    )
        else:
            print(f"WARNING: no adapter_config.json under {args.adapter_dir} — "
                  f"cannot verify that --model-dir is a merged checkpoint.")

    from transformers import AutoTokenizer

    examples = BenchmarkLoader(str(PROJECT_ROOT / "data" / "benchmarks")).load(args.benchmark)
    if args.ids_file:
        wanted = set(json.loads(Path(args.ids_file).read_text())[args.benchmark])
        examples = [ex for ex in examples if ex.id in wanted]
        if len(examples) != len(wanted):
            raise SystemExit(f"FATAL: {len(wanted) - len(examples)} ids in "
                             f"{args.ids_file} are not in {args.benchmark}")
    if args.max_samples is not None:
        examples = examples[: args.max_samples]
    print(f"benchmark={args.benchmark} examples={len(examples)}")

    config_subdir = "finetuned" if args.adapter_dir else "zero_shot"
    pred_path = (
        Path(args.results_dir) / args.model_name / args.benchmark
        / config_subdir / "predictions.jsonl"
    )
    pred_path.parent.mkdir(parents=True, exist_ok=True)

    # Single-writer lock (audit 2026-09-09 M9, fixed 2026-09-12). Nothing
    # stopped two drivers appending to one predictions.jsonl at once. The
    # damage is silent and survives the run: the rescore counts LINES, so
    # duplicated ids inflate `n` past `expected_n` and the arm is NOT flagged
    # as partial -- it is reported as complete, with some items scored twice.
    lock_path = pred_path.with_suffix(".jsonl.lock")
    lock_fh = open(lock_path, "w")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(
            f"FATAL: another process already holds the write lock on\n"
            f"  {pred_path}\n"
            f"Two writers would interleave appends into one arm. If you are "
            f"certain no other run is live, delete {lock_path} and retry."
        )
    lock_fh.write(f"pid={os.getpid()} model_dir={args.model_dir}\n")
    lock_fh.flush()

    done_ids: set[str] = set()
    prior: dict | None = None
    if pred_path.exists():
        with open(pred_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                done_ids.add(rec["id"])
                if prior is None and "model_dir" in rec:
                    prior = rec
    # Resume safety (audit 2026-09-09 M2, enforced 2026-09-12). The
    # predictions path is keyed only by results-dir/benchmark/mode, so two
    # different models pointed at one results-dir resume onto each other's
    # file and silently produce a blended arm. Records have carried
    # engine/model_dir/adapter_dir since 2026-09-09 -- check them.
    if prior is None and done_ids:
        print(
            f"WARNING: {pred_path} has {len(done_ids):,} rows but carries no "
            f"engine/model_dir tags (written before 2026-09-09), so this resume "
            f"CANNOT be checked for arm mixing. Verify by hand that it belongs "
            f"to --model-dir {args.model_dir}."
        )
    if prior is not None:
        mismatch = []
        if Path(prior["model_dir"]).resolve() != Path(args.model_dir).resolve():
            mismatch.append(f"model_dir {prior['model_dir']} != {args.model_dir}")
        if str(prior.get("adapter_dir") or "") != str(args.adapter_dir or ""):
            mismatch.append(
                f"adapter_dir {prior.get('adapter_dir')} != {args.adapter_dir}")
        if prior.get("engine", "vllm") != "vllm":
            mismatch.append(f"engine {prior.get('engine')} != vllm")
        if prior.get("date_string") != args.date_string:
            mismatch.append(f"date_string {prior.get('date_string')} != {args.date_string}")
        if bool(prior.get("system_prompt", bool(prior.get("adapter_dir")))) != bool(system_prompt):
            mismatch.append("system prompt on/off differs from the existing file")
        if mismatch:
            raise SystemExit(
                "FATAL: refusing to resume into a predictions file written by a "
                "different run — appending would blend two arms into one "
                "benchmark number:\n  " + "\n  ".join(mismatch) +
                f"\n  file: {pred_path}\nUse a different --results-dir, or move "
                f"the existing file aside if this is a deliberate re-run."
            )
    todo = [ex for ex in examples if ex.id not in done_ids]
    print(f"resume: {len(done_ids)} done, {len(todo)} todo")

    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tok.truncation_side = "left"

    def encode(ex):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": _build_zero_shot_prompt(ex)})
        kw = {"date_string": args.date_string} if args.date_string else {}
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kw
        )
        return tok(text, truncation=True, max_length=4096,
                   add_special_tokens=False)["input_ids"]

    # Imported here, after every cheap check above has passed: loading vLLM
    # costs ~30 s and allocates the GPU, so a bad --model-dir/--results-dir
    # should fail before it, not after (audit 2026-09-12).
    from vllm import LLM, SamplingParams

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
                # Provenance tag (audit 2026-09-09 M1/M3): predictions carry
                # no engine marker, so a results dir written by two engines
                # cannot be told apart after the fact. Extra keys are ignored
                # by every reader of this file (rescore, probes, audits).
                rec["engine"] = "vllm"
                rec["model_dir"] = str(args.model_dir)
                if args.adapter_dir:
                    rec["adapter_dir"] = str(args.adapter_dir)
                rec["system_prompt"] = bool(system_prompt)
                rec["date_string"] = args.date_string
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
