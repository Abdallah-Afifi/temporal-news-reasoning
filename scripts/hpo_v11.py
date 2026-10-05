"""Hyperparameter search for the v11 (prompt-parity) arm, selected on DEV only.

Methodology (pre-registered 2026-09-23, before any v11 trial existed; see
docs/hpo_v11_protocol.md):

  * Objective: mean over TIME / TimeBench / TRAM of
        (trial v5 micro accuracy) - (zero-shot v5 micro accuracy)
    on the fixed dev split data/hpo_dev/dev_ids.json (5,000 / 2,500 / 10,000
    items, stratified, carved out of the test pools because none of the three
    benchmarks ships a dev split). Zero-shot = the SAME vLLM engine with the
    SAME pinned template date, run here on the same dev ids. Macro and
    no-abstain deltas are recorded as secondary diagnostics, not selection.
  * Plan: an anchor (v10-glm's hyperparameters under the new prompt) plus N
    configurations drawn up front with a fixed seed and written to plan.json
    BEFORE the first trial runs. No adaptive sampling, so the search is
    reproducible and there is no selection-on-noise feedback loop.
  * Selection: argmax of the objective. The final arm is then evaluated on
    test-minus-dev only (rescore_v5_protocol.py --exclude-ids ...), where no
    choice was ever made.

Usage (from a shell WITH the GPU):
    venv/bin/python scripts/hpo_v11.py plan       # writes results/hpo_v11/plan.json once
    venv/bin/python scripts/hpo_v11.py run        # resumable; runs every planned trial
    venv/bin/python scripts/hpo_v11.py report     # leaderboard + best.json
"""
from __future__ import annotations

import argparse
import hashlib
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

OUT = ROOT / "results" / "hpo_v11"
CKPT = ROOT / "checkpoints" / "hpo_v11"
BASE_CFG = ROOT / "experiments" / "finetuning" / "LLaMA" / "config_v11_parity.yaml"
DEV_IDS = ROOT / "data" / "hpo_dev" / "dev_ids.json"
BASE_MODEL = ROOT / "models" / "Llama-3.2-3B-Instruct"
PY, VLLM_PY = ROOT / "venv" / "bin" / "python", ROOT / "venv_vllm" / "bin" / "python"
BENCHES = ("time", "timebench", "tram")
DATE = "26 Jul 2024"
PLAN_SEED = 20260923

ANCHOR = {"learning_rate": 4.62258900102083e-04, "num_train_epochs": 3,
          "lora_r": 32, "aug_fraction": 1.0}


def sample_plan(n: int) -> list[dict]:
    rng = random.Random(PLAN_SEED)
    plan = [dict(ANCHOR, id="t00", note="anchor: v10 hyperparameters, parity prompt")]
    for i in range(1, n + 1):
        plan.append({
            "id": f"t{i:02d}",
            # log-uniform: the effect of LR is multiplicative
            "learning_rate": float(f"{math.exp(rng.uniform(math.log(1e-5), math.log(5e-4))):.3g}"),
            # 3 epochs is covered by the anchor; sampled trials stay at 1-2 to
            # keep the plan within ~2 days of a single RTX 3090
            "num_train_epochs": rng.choice([1, 2]),
            "lora_r": rng.choice([8, 16, 32]),
            # how much of the synthetic AUG_GLM2 slice to keep -- the data knob
            "aug_fraction": rng.choice([0.0, 0.5, 1.0]),
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


def train_file(frac: float) -> Path:
    """v11 train with only `frac` of the AUG_GLM2 rows (deterministic subset)."""
    src = ROOT / "data" / "combined_80_20_v11" / "train.jsonl"
    if frac >= 1.0:
        return src
    dst = OUT / "data" / f"train_aug{frac:.2f}.jsonl"
    if dst.exists():
        return dst
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    aug = sorted((r for r in rows if r["source_dataset"] == "AUG_GLM2"),
                 key=lambda r: hashlib.sha1(r["question"].encode()).hexdigest())
    random.Random(f"aug-{frac}").shuffle(aug)
    keep = {id(r) for r in aug[: round(frac * len(aug))]}
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        for r in rows:
            if r["source_dataset"] != "AUG_GLM2" or id(r) in keep:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return dst


def eval_dev(model_dir: Path, results_dir: Path, adapter: Path | None, log: Path) -> None:
    for b in BENCHES:
        pred = results_dir / "llama" / b / ("finetuned" if adapter else "zero_shot") / "predictions.jsonl"
        if pred.exists() and sum(1 for _ in open(pred)) >= len(json.loads(DEV_IDS.read_text())[b]):
            continue
        cmd = [VLLM_PY, "scripts/run_eval_vllm.py", "--model-dir", model_dir,
               "--benchmark", b, "--results-dir", results_dir, "--model-name", "llama",
               "--ids-file", DEV_IDS, "--date-string", DATE, "--system-prompt", "none"]
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
        path = results_dir / "llama" / b / sub / "predictions.jsonl"
        s = rescore(path, cm, ab, want_correct_ids=True,
                    ref_correct=None if ref is None else ref[b]["_correct"], settings=sm)
        if s["n"] < len(cm):
            raise RuntimeError(f"{path}: {s['n']} scored < {len(cm)} dev items")
        out[b] = {k: s[k] for k in ("n", "v5_pct", "macro_pct", "no_abstain_pct", "by_setting")}
        out[b]["_correct"] = s["correct_ids"]
        if ref is not None:
            bb, cc = s["_paired"]
            out[b]["mcnemar_b_c"] = [bb, cc]
            # paired SE of the accuracy difference, in pp
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
                   train_data=str(train_file(t["aug_fraction"]).relative_to(ROOT)),
                   output_dir=str(cdir.relative_to(ROOT)))
        (tdir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        t0 = time.time()
        sh([PY, "experiments/finetuning/LLaMA/train.py", "--config", tdir / "config.yaml"],
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
    shutil.rmtree(merged, ignore_errors=True)  # 6 GB each; the adapter is kept
    return res


# ---------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "run", "report", "final-config"])
    ap.add_argument("--n", type=int, default=10, help="sampled configs besides the anchor")
    ap.add_argument("--only", nargs="*", help="run only these trial ids")
    ap.add_argument("--seed", type=int, default=42, help="final-config: training seed")
    args = ap.parse_args()
    plan_path = OUT / "plan.json"

    if args.cmd == "final-config":
        # The best trial's config with only the seed (and output dir) changed.
        # Seed 42 IS the best trial itself, so its adapter is reused, not retrained.
        best = json.loads((OUT / "best.json").read_text())["trial"]
        if args.seed == 42:
            print(CKPT / best["id"] / "final")
            return 0
        cfg = yaml.safe_load((OUT / "trials" / best["id"] / "config.yaml").read_text())
        cfg.update(seed=args.seed, output_dir=f"checkpoints/llama_v11_best_s{args.seed}")
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
            except Exception as e:  # recorded, never silently skipped
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
    print(f"\n{'id':5s}{'lr':>10s}{'ep':>4s}{'r':>4s}{'aug':>6s}{'obj':>8s}{'±se':>6s}"
          + "".join(f"{b:>11s}" for b in BENCHES) + f"{'macroΔ':>9s}")
    for r in rows:
        t, B = r["trial"], r["benchmarks"]
        macro = sum(B[b]["d_macro_pct"] for b in BENCHES) / 3
        print(f"{t['id']:5s}{t['learning_rate']:10.2e}{t['num_train_epochs']:4d}{t['lora_r']:4d}"
              f"{t['aug_fraction']:6.2f}{r['objective']:+8.2f}{r['objective_se_pp']:6.2f}"
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
