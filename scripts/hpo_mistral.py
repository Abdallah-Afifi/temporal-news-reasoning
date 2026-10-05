"""Hyperparameter search for the Mistral prompt-parity arm, selected on DEV
only. Ported 2026-09-28 from scripts/hpo_v11.py (LLaMA), same methodology,
smaller budget -- see docs/mistral_plan.md for the full writeup.

Methodology (same as hpo_v11.py, §ref docs/hpo_v11_protocol.md):

  * Objective: mean over TIME / TimeBench / TRAM of
        (trial v5 micro accuracy) - (zero-shot v5 micro accuracy)
    on the fixed dev split data/hpo_dev/dev_ids.json (the SAME dev split
    LLaMA's v11 search used -- it is carved from the benchmark test pools by
    item id, not by model, so it is valid for any model). Zero-shot = the
    SAME vLLM engine, run here on the same dev ids (Mistral's chat template
    has no date field, so no date pin is needed or passed).
  * Plan: an anchor (LLaMA's own HPO-winning hyperparameters -- t09,
    0.000177 / 1 epoch / lora_r 16 -- transplanted, since that is the best
    prior evidence for what this data+prompt combination rewards) plus N
    sampled configs drawn up front with a fixed seed and written to
    plan.json BEFORE the first trial runs.
  * Reduced scope vs v11's search (documented, not hidden): only
    (learning_rate, num_train_epochs, lora_r) are searched. v11 also swept
    an aug_fraction data knob; that question was already answered by v11/v12
    for THIS data (docs/v12_plan.md), so Mistral's search reuses
    data/combined_80_20_v12 whole, unchanged across trials, and N defaults
    to 5 sampled trials (+ anchor = 6) rather than v11's 10 (+ anchor = 11)
    -- a 7B model trains markedly slower per step than the 3B LLaMA arm, so
    the full v11 budget would cost several days on one GPU.
  * Selection: argmax of the objective. The final arm is then evaluated on
    test-minus-dev only (rescore_v5_protocol.py --exclude-ids ...), where no
    choice was ever made.

Usage (from a shell WITH real GPU access -- see docs/audit_2026_09_12.md,
this Claude Code session has none; launch from your own terminal):
    venv/bin/python scripts/hpo_mistral.py plan       # writes results/hpo_mistral/plan.json once
    venv/bin/python scripts/hpo_mistral.py run        # resumable; runs every planned trial
    venv/bin/python scripts/hpo_mistral.py report     # leaderboard + best.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "hpo_mistral"
CKPT = ROOT / "checkpoints" / "hpo_mistral"
BASE_CFG = ROOT / "experiments" / "finetuning" / "Mistral" / "config_mistral_parity.yaml"
DEV_IDS = ROOT / "data" / "hpo_dev" / "dev_ids.json"
BASE_MODEL = ROOT / "models" / "Mistral-7B-Instruct-v0.3"
PY, VLLM_PY = ROOT / "venv" / "bin" / "python", ROOT / "venv_vllm" / "bin" / "python"
BENCHES = ("time", "timebench", "tram")
PLAN_SEED = 20260928

# LLaMA's HPO-winning trial (t09), transplanted -- see module docstring.
ANCHOR = {"learning_rate": 1.77e-04, "num_train_epochs": 1, "lora_r": 16}


def sample_plan(n: int) -> list[dict]:
    rng = random.Random(PLAN_SEED)
    plan = [dict(ANCHOR, id="m00", note="anchor: LLaMA t09 hyperparameters, transplanted")]
    for i in range(1, n + 1):
        plan.append({
            "id": f"m{i:02d}",
            "learning_rate": float(f"{math.exp(rng.uniform(math.log(1e-5), math.log(5e-4))):.3g}"),
            "num_train_epochs": rng.choice([1, 2]),
            "lora_r": rng.choice([8, 16, 32]),
        })
    return plan


# --------------------------------------------------------------------- utils
def sh(cmd: list, log: Path, env: dict | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"  $ {' '.join(map(str, cmd))}  > {log.relative_to(ROOT)}", flush=True)
    with open(log, "a") as fh:
        r = subprocess.run(list(map(str, cmd)), cwd=ROOT, stdout=fh,
                           stderr=subprocess.STDOUT, env={**os.environ, **(env or {})})
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}); see {log}")


def eval_dev(model_dir: Path, results_dir: Path, adapter: Path | None, log: Path) -> None:
    for b in BENCHES:
        pred = results_dir / "mistral" / b / ("finetuned" if adapter else "zero_shot") / "predictions.jsonl"
        if pred.exists() and sum(1 for _ in open(pred)) >= len(json.loads(DEV_IDS.read_text())[b]):
            continue
        cmd = [VLLM_PY, "scripts/run_eval_vllm.py", "--model-dir", model_dir,
               "--benchmark", b, "--results-dir", results_dir, "--model-name", "mistral",
               "--ids-file", DEV_IDS, "--system-prompt", "none"]
        if adapter:
            cmd += ["--adapter-dir", adapter]
        sh(cmd, log, env={"VLLM_USE_FLASHINFER_SAMPLER": "0"})


_SCORING: dict = {}


def score(results_dir: Path, sub: str, ref: dict | None) -> dict:
    from scripts.rescore_v5_protocol import (abstain_option_ids, choice_map,
                                             rescore, setting_map)
    dev = json.loads(DEV_IDS.read_text())
    out = {}
    for b in BENCHES:
        if b not in _SCORING:
            keep = set(dev[b])
            cm = {k: v for k, v in choice_map(b, str(ROOT / "data" / "benchmarks")).items() if k in keep}
            _SCORING[b] = (cm, abstain_option_ids(cm), setting_map(b, str(ROOT / "data" / "benchmarks")))
        cm, ab, sm = _SCORING[b]
        path = results_dir / "mistral" / b / sub / "predictions.jsonl"
        s = rescore(path, cm, ab, want_correct_ids=True,
                    ref_correct=None if ref is None else ref[b]["_correct"], settings=sm)
        if s["n"] < len(cm):
            raise RuntimeError(f"{path}: {s['n']} scored < {len(cm)} dev items")
        out[b] = {k: s[k] for k in ("n", "v5_pct", "macro_pct", "no_abstain_pct", "by_setting")}
        out[b]["_correct"] = s["correct_ids"]
        if ref is not None:
            bb, cc = s["_paired"]
            out[b]["mcnemar_b_c"] = [bb, cc]
            out[b]["delta_se_pp"] = 100 * math.sqrt(bb + cc) / s["n"]
            for k in ("v5_pct", "macro_pct", "no_abstain_pct"):
                out[b][f"d_{k}"] = s[k] - ref[b][k]
    return out


def zs_reference() -> dict:
    zdir = OUT / "zs_dev"
    eval_dev(BASE_MODEL, zdir, None, OUT / "logs" / "zs_dev.log")
    return score(zdir, "zero_shot", None)


def run_trial(t: dict, ref: dict) -> dict:
    tdir, cdir = OUT / "trials" / t["id"], CKPT / t["id"]
    tdir.mkdir(parents=True, exist_ok=True)
    done = tdir / "score.json"
    if done.exists():
        return json.loads(done.read_text())
    adapter = cdir / "final"
    if not (adapter / "adapter_config.json").exists():
        cfg = yaml.safe_load(BASE_CFG.read_text())
        cfg.update(learning_rate=t["learning_rate"], num_train_epochs=t["num_train_epochs"],
                   lora_r=t["lora_r"], lora_alpha=2 * t["lora_r"],
                   output_dir=str(cdir.relative_to(ROOT)))
        (tdir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        t0 = time.time()
        sh([PY, "experiments/finetuning/Mistral/train.py", "--config", tdir / "config.yaml"],
           tdir / "train.log")
        (tdir / "train_seconds").write_text(str(int(time.time() - t0)))
    merged = cdir / "merged_fp16"
    if not (merged / "config.json").exists():
        sh([PY, "scripts/merge_lora.py", "--base", BASE_MODEL, "--adapter", adapter,
            "--out", merged], tdir / "merge.log")
    eval_dev(merged, tdir / "preds", adapter, tdir / "eval.log")
    s = score(tdir / "preds", "finetuned", ref)
    res = {"trial": t, "objective": sum(s[b]["d_v5_pct"] for b in BENCHES) / len(BENCHES),
           "objective_se_pp": math.sqrt(sum(s[b]["delta_se_pp"] ** 2 for b in BENCHES)) / len(BENCHES),
           "benchmarks": {b: {k: v for k, v in s[b].items() if k != "_correct"} for b in BENCHES}}
    done.write_text(json.dumps(res, indent=2))
    shutil.rmtree(merged, ignore_errors=True)
    return res


# ---------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "run", "report", "final-config"])
    ap.add_argument("--n", type=int, default=5, help="sampled configs besides the anchor")
    ap.add_argument("--only", nargs="*", help="run only these trial ids")
    ap.add_argument("--seed", type=int, default=42, help="final-config: training seed")
    args = ap.parse_args()
    plan_path = OUT / "plan.json"

    if args.cmd == "final-config":
        best = json.loads((OUT / "best.json").read_text())["trial"]
        if args.seed == 42:
            print(CKPT / best["id"] / "final")
            return 0
        cfg = yaml.safe_load((OUT / "trials" / best["id"] / "config.yaml").read_text())
        cfg.update(seed=args.seed, output_dir=f"checkpoints/mistral_best_s{args.seed}")
        p = OUT / f"final_s{args.seed}.yaml"
        p.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(p)
        return 0

    if args.cmd == "plan":
        if plan_path.exists():
            raise SystemExit(f"{plan_path} already exists; the plan is fixed once written.")
        OUT.mkdir(parents=True, exist_ok=True)
        plan = sample_plan(args.n)
        plan_path.write_text(json.dumps({"seed": PLAN_SEED, "created": time.strftime("%F %T"),
                                         "trials": plan}, indent=2))
        for t in plan:
            print(t)
        return 0

    plan = json.loads(plan_path.read_text())["trials"]
    if args.cmd == "run":
        ref = zs_reference()
        print("zero-shot dev:", {b: round(ref[b]["v5_pct"], 2) for b in BENCHES}, flush=True)
        for t in plan:
            if args.only and t["id"] not in args.only:
                continue
            print(f"\n=== {t['id']} {t}", flush=True)
            try:
                r = run_trial(t, ref)
                print(f"    objective {r['objective']:+.2f}pp (se {r['objective_se_pp']:.2f})", flush=True)
            except Exception as e:
                (OUT / "trials" / t["id"] / "FAILED").write_text(f"{time.strftime('%F %T')} {e}\n")
                print(f"    FAILED: {e}", flush=True)
        return 0

    rows = []
    for t in plan:
        p = OUT / "trials" / t["id"] / "score.json"
        if p.exists():
            rows.append(json.loads(p.read_text()))
        else:
            fail = OUT / "trials" / t["id"] / "FAILED"
            print(f"{t['id']}: {'FAILED ' + fail.read_text().strip() if fail.exists() else 'not run'}")
    rows.sort(key=lambda r: -r["objective"])
    print(f"\n{'id':5s}{'lr':>10s}{'ep':>4s}{'r':>4s}{'obj':>8s}{'±se':>6s}"
          + "".join(f"{b:>11s}" for b in BENCHES) + f"{'macroΔ':>9s}")
    for r in rows:
        t, B = r["trial"], r["benchmarks"]
        macro = sum(B[b]["d_macro_pct"] for b in BENCHES) / 3
        print(f"{t['id']:5s}{t['learning_rate']:10.2e}{t['num_train_epochs']:4d}{t['lora_r']:4d}"
              f"{r['objective']:+8.2f}{r['objective_se_pp']:6.2f}"
              + "".join(f"{B[b]['d_v5_pct']:+11.2f}" for b in BENCHES) + f"{macro:+9.2f}")
    if rows:
        best = rows[0]
        (OUT / "best.json").write_text(json.dumps(best, indent=2))
        print(f"\nbest = {best['trial']['id']} -> {OUT / 'best.json'}")
        if len(rows) > 1 and rows[0]["objective"] - rows[1]["objective"] < 2 * best["objective_se_pp"]:
            print("NOTE: the top two trials are within 2 SE on dev -- the ranking between "
                  "them is not resolved by this dev set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
