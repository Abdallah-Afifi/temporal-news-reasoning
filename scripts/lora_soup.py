"""Average LoRA adapters (a "soup") -- v13's zero-training arm (plan §5).

Exact uniform soup of the merged updates of the three v11 seeds (t09/s42,
s43, s44 -- same recipe, different seeds): the result's delta-W equals
(1/n) * sum_i (alpha/r) * B_i @ A_i for every module.

Done by CONCATENATION along the rank dim, not by averaging A and B
separately. A separate mean gives (1/n sum B)(1/n sum A) = 1/n^2 sum_ij B_i A_j:
each seed's own term weighted 1/n^2 plus cross terms between independently
initialised A's, i.e. a ~1/3-strength adapter plus noise
(docs/audit_2026_10_04.md §4.1). Concatenation:

    A' = [A_1; ...; A_n]      (n*r, in)
    B' = [B_1, ..., B_n] / n  (out, n*r)
    B' @ A' = (1/n) sum_i B_i @ A_i            exactly
    r' = n*r, alpha' = n*alpha  (keeps the alpha/r scale)

The output is a normal PEFT LoRA adapter of rank n*r that merge_lora.py
merges like any other. Kept in float32 (the seed adapters' dtype).
Uniform weights are the pre-registered choice (no dev-set weight tuning).

    venv/bin/python scripts/lora_soup.py
    venv/bin/python scripts/lora_soup.py --out checkpoints/llama_v11_soup/final \
        --adapters checkpoints/hpo_v11/t09/final \
                   checkpoints/llama_v11_best_s43/final \
                   checkpoints/llama_v11_best_s44/final
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]

DEFAULTS = [
    "checkpoints/hpo_v11/t09/final",        # v11-best seed 42
    "checkpoints/llama_v11_best_s43/final",
    "checkpoints/llama_v11_best_s44/final",
]


def check_config(cfg: dict) -> str | None:
    """Return a reason this adapter can't be soup'd by concatenation, or None."""
    if cfg.get("peft_type") != "LORA":
        return f"peft_type {cfg.get('peft_type')}"
    if cfg.get("use_rslora"):
        return "use_rslora (scale alpha/sqrt(r) does not survive r -> n*r)"
    if cfg.get("use_dora"):
        return "use_dora (magnitude vectors are not linear in B@A)"
    if cfg.get("bias", "none") != "none" or cfg.get("lora_bias"):
        return "bias terms"
    if cfg.get("modules_to_save"):
        return "modules_to_save"
    if cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        return "per-module rank_pattern/alpha_pattern"
    return None


def concat_soup(sds: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Concatenation soup of n LoRA state dicts with identical keys/shapes."""
    n = len(sds)
    out = {}
    for k in sds[0]:
        ts = [sd[k].to(torch.float32) for sd in sds]
        if ".lora_A." in k:
            out[k] = torch.cat(ts, dim=0)
        elif ".lora_B." in k:
            out[k] = torch.cat(ts, dim=1) / n
        else:
            raise ValueError(f"unexpected tensor in LoRA adapter: {k}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapters", nargs="*", default=None)
    ap.add_argument("--out", default="checkpoints/llama_v11_soup/final")
    args = ap.parse_args()
    dirs = [Path(a) if a.startswith("/") else ROOT / a
            for a in (args.adapters or DEFAULTS)]
    for d in dirs:
        if not (d / "adapter_model.safetensors").exists():
            print(f"missing adapter: {d}")
            return 1

    cfg = json.loads((dirs[0] / "adapter_config.json").read_text())
    bad = check_config(cfg)
    if bad:
        print(f"refusing to soup: {bad} ({dirs[0]})")
        return 1
    for d in dirs[1:]:
        other = json.loads((d / "adapter_config.json").read_text())
        bad = check_config(other)
        if bad:
            print(f"refusing to soup: {bad} ({d})")
            return 1
        # target_modules is stored in arbitrary order across saves; compare
        # as a set, everything else exactly.
        for k in ("r", "lora_alpha", "peft_type", "base_model_name_or_path"):
            if cfg.get(k) != other.get(k):
                print(f"adapter mismatch on {k}: {d}")
                return 1
        if set(cfg.get("target_modules") or []) != \
                set(other.get("target_modules") or []):
            print(f"adapter mismatch on target_modules: {d}")
            return 1

    sds = [load_file(str(d / "adapter_model.safetensors")) for d in dirs]
    for d, sd in zip(dirs[1:], sds[1:]):
        if set(sd) != set(sds[0]) or any(sd[k].shape != sds[0][k].shape for k in sd):
            print(f"tensor-set/shape mismatch: {d}")
            return 1
    n = len(dirs)
    out_sd = concat_soup(sds)

    out = Path(args.out) if args.out.startswith("/") else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    save_file(out_sd, str(out / "adapter_model.safetensors"),
              metadata={"format": "pt"})
    new_cfg = dict(cfg, r=n * cfg["r"], lora_alpha=n * cfg["lora_alpha"])
    (out / "adapter_config.json").write_text(json.dumps(new_cfg, indent=2))
    if (dirs[0] / "README.md").exists():
        shutil.copy(dirs[0] / "README.md", out / "README.md")
    meta = {
        "method": "concat (exact uniform mean of merged deltas)",
        "adapters": [str(d) for d in dirs], "weights": [1.0 / n] * n,
        "r": new_cfg["r"], "lora_alpha": new_cfg["lora_alpha"],
        "scale_alpha_over_r": new_cfg["lora_alpha"] / new_cfg["r"],
        "dtype": "float32",
        "replaces": "naive separate mean of A and B (audit_2026_10_04 §4.1)",
    }
    (out / "soup_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"concat soup of {n} adapters (r={new_cfg['r']}, "
          f"alpha={new_cfg['lora_alpha']}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
