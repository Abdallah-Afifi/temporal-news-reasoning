"""Run LoRA ablation trials sequentially from config files."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_CONFIGS = {
    "llama": [
        "configs/lora_llama_trial_a_r8_attn.yaml",
        "configs/lora_llama_trial_b_r16_attn.yaml",
        "configs/lora_llama_trial_c_r16_attn_mlp.yaml",
    ],
    "qwen": [
        "configs/lora_qwen_trial_a_r16_attn.yaml",
        "configs/lora_qwen_trial_b_r32_attn.yaml",
    ],
    "mistral": [
        "configs/lora_mistral_trial_a_r16_attn.yaml",
        "configs/lora_mistral_trial_b_r16_attn_mlp.yaml",
    ],
}


def _check_runtime_deps() -> bool:
    try:
        __import__("torch")
        __import__("transformers")
        __import__("peft")
    except Exception as exc:
        print(f"Dependency check failed: {exc}")
        print("Install required packages first, e.g. pip3 install -r requirements.txt")
        return False
    return True


def _collect_configs(group: str, custom_configs: list[str] | None) -> list[Path]:
    if custom_configs:
        return [Path(item) for item in custom_configs]

    if group == "all":
        combined: list[str] = []
        for model_group in ("llama", "qwen", "mistral"):
            combined.extend(DEFAULT_CONFIGS[model_group])
        return [Path(item) for item in combined]

    return [Path(item) for item in DEFAULT_CONFIGS[group]]


def _run_command(config_path: Path) -> int:
    cmd = [sys.executable, "src/training/train_lora.py", str(config_path)]
    print(f"\n=== Running: {' '.join(cmd)}")
    completed = subprocess.run(cmd)
    return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LoRA ablation trials from YAML configs")
    parser.add_argument(
        "--group",
        choices=["llama", "qwen", "mistral", "all"],
        default="all",
        help="Which default trial group to run",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help="Explicit config path(s) to run; can be repeated",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining trials even if one fails",
    )
    parser.add_argument(
        "--skip-dep-check",
        action="store_true",
        help="Skip import checks for torch/transformers/peft",
    )

    args = parser.parse_args()

    if not args.skip_dep_check and not _check_runtime_deps():
        return 1

    configs = _collect_configs(args.group, args.config)
    if not configs:
        print("No configs selected.")
        return 1

    missing = [item for item in configs if not item.exists()]
    if missing:
        print("Missing config files:")
        for item in missing:
            print(f" - {item}")
        return 1

    results: list[tuple[str, int]] = []

    for config in configs:
        code = _run_command(config)
        results.append((str(config), code))

        if code != 0 and not args.continue_on_error:
            print("\nStopping on first failure. Use --continue-on-error to keep running.")
            break

    print("\n=== Ablation Summary ===")
    for cfg, code in results:
        status = "OK" if code == 0 else f"FAIL({code})"
        print(f"{status:>9}  {cfg}")

    failed = [item for item in results if item[1] != 0]
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
