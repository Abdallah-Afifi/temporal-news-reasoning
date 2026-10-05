"""
experiments/finetuning/LLaMA/train.py
============================
Flat LoRA fine-tuning of LLaMA-3.2-3B-Instruct on the combined temporal
reasoning dataset (combined_80_20_split).

What this script does
----------------------
1. Pre-flight checks (data files, output dir, VRAM).
2. Loads the LLaMA tokenizer and model.
3. Wraps the model with PEFT LoRA adapters.
4. Loads train.jsonl + val.jsonl, normalising all data schemas.
5. Trains for N epochs with per-epoch validation, checkpointing,
   and optional early stopping based on eval_loss.
6. Saves the final (best) adapter to ``checkpoints/llama/final/``.
7. Writes a progress summary to ``experiments/finetuning/LLaMA/Progress/``.

Usage
-----
From the repo root::

    # Standard run
    python experiments/finetuning/LLaMA/train.py

    # Custom config
    python experiments/finetuning/LLaMA/train.py --config experiments/finetuning/LLaMA/config.yaml

    # 4-bit QLoRA (low VRAM)
    python experiments/finetuning/LLaMA/train.py --load-in-4bit

    # Dry-run: 5 training steps only (pipeline verification)
    python experiments/finetuning/LLaMA/train.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from experiments.finetuning.LLaMA.data_loader import (
    load_flat_datasets,
    load_tokenizer,
    HF_MODEL_NAME,
)
from experiments.finetuning.shared.utils import (
    Timer,
    get_device_info,
    load_yaml,
    preflight_check,
    print_gpu_memory,
    save_json,
    setup_logger,
)

# ---------------------------------------------------------------------------
log = setup_logger("llama.train")
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(
    model_name_or_path: str,
    dtype: str = "bfloat16",
    load_in_4bit: bool = False,
    attn_implementation: str = "sdpa",
) -> AutoModelForCausalLM:
    """Load the LLaMA base model with optional 4-bit quantization.

    ``attn_implementation`` selects the attention kernel. "sdpa" is PyTorch's
    fused implementation: same arithmetic as "eager", materially faster, and
    it needs no build step (flash-attention 2 would require nvcc, which this
    host does not have).
    """
    dtype_map = {
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float16":  torch.float16,  "fp16": torch.float16,
        "float32":  torch.float32,
    }
    torch_dtype = dtype_map.get(dtype.lower(), torch.bfloat16)

    load_kwargs: dict = {
        "torch_dtype": torch_dtype,
        "device_map": "auto",
        "attn_implementation": attn_implementation,
    }

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        log.info("4-bit QLoRA enabled (NF4 + double quantization)")

    log.info("Loading LLaMA model from: %s", model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
    return model


# ---------------------------------------------------------------------------
# LoRA configuration
# ---------------------------------------------------------------------------

def build_lora_config(cfg: dict) -> LoraConfig:
    """Build LoRA config from flat config dict."""
    return LoraConfig(
        r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        target_modules=list(cfg.get("target_modules",
                                     ["q_proj", "k_proj", "v_proj", "o_proj"])),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def init_progress_dir(progress_dir: Path, config: dict) -> Path:
    """Create a timestamped progress directory and save the config snapshot."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = progress_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    save_json(config, run_dir / "config_snapshot.json")

    readme = run_dir / "README.md"
    readme.write_text(
        f"# LLaMA LoRA Training Run\n\n"
        f"- **Started**: {datetime.now().isoformat()}\n"
        f"- **Model**: {config.get('model_path', 'N/A')}\n"
        f"- **Train data**: {config.get('train_data', 'N/A')}\n"
        f"- **Val data**: {config.get('val_data', 'N/A')}\n"
        f"- **Epochs**: {config.get('num_train_epochs', 'N/A')}\n"
        f"- **Learning rate**: {config.get('learning_rate', 'N/A')}\n"
        f"- **LoRA r**: {config.get('lora_r', 'N/A')}\n"
        f"- **Batch size**: {config.get('per_device_train_batch_size', 'N/A')}\n"
        f"- **Grad accum**: {config.get('gradient_accumulation_steps', 'N/A')}\n"
        f"- **Effective batch**: {config.get('per_device_train_batch_size', 4) * config.get('gradient_accumulation_steps', 1)}\n"
        f"- **Max seq length**: {config.get('max_seq_length', 'N/A')}\n\n"
        f"## Files\n\n"
        f"- `config_snapshot.json` — full config\n"
        f"- `data_stats.json` — dataset statistics\n"
        f"- `live_report.md` — **LIVE** training progress (refresh to see updates)\n"
        f"- `training_summary.json` — final results\n"
        f"- `final_eval_metrics.json` — final eval metrics\n"
        f"- `trainer_log_history.json` — per-step loss log\n",
    )

    log.info("Progress directory: %s", run_dir)
    return run_dir


def save_data_stats(run_dir: Path, train_size: int, val_size: int, skipped_info: str = "") -> None:
    stats = {
        "train_examples": train_size,
        "val_examples": val_size,
        "total": train_size + val_size,
        "skipped_info": skipped_info,
        "timestamp": datetime.now().isoformat(),
    }
    save_json(stats, run_dir / "data_stats.json")


def save_training_summary(run_dir: Path, train_result, elapsed_str: str, config: dict) -> None:
    summary = {
        "status": "completed",
        "elapsed": elapsed_str,
        "train_loss": getattr(train_result, "training_loss", None),
        "epochs_trained": config.get("num_train_epochs"),
        "global_step": getattr(train_result, "global_step", None),
        "metrics": getattr(train_result, "metrics", {}),
        "timestamp": datetime.now().isoformat(),
    }
    save_json(summary, run_dir / "training_summary.json")


# ---------------------------------------------------------------------------
# Live progress callback
# ---------------------------------------------------------------------------

class LiveProgressCallback(TrainerCallback):
    """Writes a ``live_report.md`` file to the progress directory that updates
    after every logging step and every evaluation.  ``tail -f`` or refresh
    this file to monitor training in real-time."""

    def __init__(self, run_dir: Path, config: dict, train_size: int, val_size: int):
        self.report_path = run_dir / "live_report.md"
        self.config = config
        self.train_size = train_size
        self.val_size = val_size
        self.start_time = datetime.now()
        self.train_losses: list[tuple[int, float]] = []   # (step, loss)
        self.eval_losses: list[tuple[float, float]] = []  # (epoch, eval_loss)
        self.best_eval_loss = float("inf")

    def _gpu_stats(self) -> str:
        if not torch.cuda.is_available():
            return "CPU"
        alloc = torch.cuda.memory_allocated() / 1e9
        resv = torch.cuda.memory_reserved() / 1e9
        return f"{alloc:.1f} GB alloc / {resv:.1f} GB reserved"

    def _write_report(self, state, is_eval: bool = False):
        now = datetime.now()
        elapsed = now - self.start_time
        hours, rem = divmod(int(elapsed.total_seconds()), 3600)
        mins, secs = divmod(rem, 60)
        elapsed_str = f"{hours:02d}:{mins:02d}:{secs:02d}"

        total_steps = state.max_steps if state.max_steps > 0 else int(
            self.train_size / (self.config.get('per_device_train_batch_size', 8) * self.config.get('gradient_accumulation_steps', 2)) * self.config.get('num_train_epochs', 3)
        )
        pct = (state.global_step / total_steps * 100) if total_steps > 0 else 0

        # ETA
        if state.global_step > 0:
            secs_per_step = elapsed.total_seconds() / state.global_step
            remaining_steps = total_steps - state.global_step
            eta_secs = int(remaining_steps * secs_per_step)
            eta_h, eta_rem = divmod(eta_secs, 3600)
            eta_m, eta_s = divmod(eta_rem, 60)
            eta_str = f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}"
        else:
            eta_str = "calculating..."

        lines = [
            "# 🔴 LIVE Training Progress",
            "",
            f"**Last updated**: {now.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Status**: {'🟢 Training' if not is_eval else '🔵 Evaluating'}  ",
            f"**Elapsed**: {elapsed_str} | **ETA**: {eta_str}  ",
            f"**Step**: {state.global_step} / {total_steps} ({pct:.1f}%)  ",
            f"**Epoch**: {state.epoch:.2f} / {self.config.get('num_train_epochs', 3)}  ",
            f"**GPU**: {self._gpu_stats()}",
            "",
            "---",
            "",
            "## Training Loss (every 10 steps)",
            "",
            "| Step | Train Loss |",
            "|-----:|-----------:|",
        ]

        # Show last 30 training loss entries
        for step, loss in self.train_losses[-30:]:
            lines.append(f"| {step} | {loss:.4f} |")

        lines.extend(["", "---", "", "## Eval Loss (per epoch)", ""])

        if self.eval_losses:
            lines.append("| Epoch | Eval Loss | vs Best | Status |")
            lines.append("|------:|----------:|--------:|:------:|")
            for epoch, eloss in self.eval_losses:
                diff = eloss - self.best_eval_loss
                status = "✅ best" if abs(diff) < 1e-6 else ("⬆️ +" + f"{diff:.4f}" if diff > 0 else "⬇️ " + f"{diff:.4f}")
                lines.append(f"| {epoch:.1f} | {eloss:.4f} | {diff:+.4f} | {status} |")
            lines.extend([
                "",
                f"**Best eval_loss**: {self.best_eval_loss:.4f}",
            ])
            # Overfitting warning
            if len(self.eval_losses) >= 2:
                last = self.eval_losses[-1][1]
                prev = self.eval_losses[-2][1]
                if last > prev:
                    lines.append("")  
                    lines.append("⚠️ **Warning**: eval_loss increased — possible overfitting.")
        else:
            lines.append("_No evaluation yet — first eval after epoch 1._")

        lines.extend([
            "",
            "---",
            "",
            "## Config Summary",
            "",
            f"- **LR**: {self.config.get('learning_rate')}",
            f"- **Batch**: {self.config.get('per_device_train_batch_size')} × {self.config.get('gradient_accumulation_steps')} = {self.config.get('per_device_train_batch_size', 8) * self.config.get('gradient_accumulation_steps', 2)} effective",
            f"- **Train**: {self.train_size} examples | **Val**: {self.val_size} examples",
            f"- **Max seq**: {self.config.get('max_seq_length')}",
            f"- **LoRA r**: {self.config.get('lora_r')} | α: {self.config.get('lora_alpha')}",
            f"- **Early stopping patience**: {self.config.get('early_stopping_patience', 'off')}",
        ])

        self.report_path.write_text("\n".join(lines) + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.train_losses.append((state.global_step, logs["loss"]))
            self._write_report(state)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            epoch = state.epoch or 0
            eloss = metrics["eval_loss"]
            self.eval_losses.append((epoch, eloss))
            if eloss < self.best_eval_loss:
                self.best_eval_loss = eloss
            self._write_report(state, is_eval=True)


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def train(
    config_path: str | Path,
    load_in_4bit: Optional[bool] = None,
    dry_run: bool = False,
    max_steps_override: int | None = None,
) -> None:
    """Run the flat LLaMA LoRA fine-tuning pipeline.

    Args:
        config_path:  Path to ``experiments/finetuning/LLaMA/config.yaml``.
        load_in_4bit: Force-enable/disable 4-bit QLoRA (None = use config).
        dry_run:      Run only 5 training steps (pipeline verification).
    """
    # ── Load config ───────────────────────────────────────────────
    config = load_yaml(config_path)
    set_seed(int(config.get("seed", 42)))
    log.info("Device info: %s", get_device_info())

    # ── Resolve paths (relative to repo root) ─────────────────────
    model_path = str((_REPO_ROOT / config.get("model_path", "models/Llama-3.2-3B-Instruct")).resolve())
    train_data = str((_REPO_ROOT / config["train_data"]).resolve())
    val_data   = str((_REPO_ROOT / config["val_data"]).resolve())
    output_dir = (_REPO_ROOT / config.get("output_dir", "checkpoints/llama")).resolve()
    if dry_run:
        # Never clobber real adapters with 5-step verification runs.
        output_dir = output_dir.with_name(output_dir.name + "_dry_run")
    progress_dir = (_REPO_ROOT / config.get("progress_dir", "experiments/finetuning/LLaMA/Progress")).resolve()

    # ── Pre-flight ────────────────────────────────────────────────
    # CLI flag (if given) overrides the config; otherwise defer to config.
    config_4bit = bool(config.get("load_in_4bit", False))
    use_4bit = config_4bit if load_in_4bit is None else load_in_4bit
    preflight_check(
        data_paths=[Path(train_data), Path(val_data)],
        output_dir=output_dir,
        min_vram_gb=6.0 if use_4bit else 10.0,
        logger=log,
    )

    # ── Progress directory ────────────────────────────────────────
    run_dir = init_progress_dir(progress_dir, config)

    # ── Load tokenizer & model ────────────────────────────────────
    model_source = model_path if (Path(model_path) / "config.json").exists() else HF_MODEL_NAME
    tokenizer = load_tokenizer(model_source)
    model = load_model(
        model_source,
        dtype=config.get("dtype", "bfloat16"),
        load_in_4bit=use_4bit,
        attn_implementation=config.get("attn_implementation", "sdpa"),
    )

    # ── Apply LoRA ────────────────────────────────────────────────
    peft_config = build_lora_config(config)
    model = get_peft_model(model, peft_config)
    # Frozen base embeddings + gradient checkpointing (reentrant) silently
    # drop gradients ("element 0 of tensors does not require grad").
    # Make embedding outputs require grad so checkpointed blocks backprop.
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # ── Load & tokenise data ──────────────────────────────────────
    max_length = int(config.get("max_seq_length", 2048))
    prompt_format = str(config.get("prompt_format", "legacy"))
    log.info("prompt_format: %s", prompt_format)
    train_ds, val_ds = load_flat_datasets(train_data, val_data, tokenizer, max_length, log,
                                          prompt_format=prompt_format)

    save_data_stats(
        run_dir, len(train_ds), len(val_ds),
        skipped_info="Temprel records skipped (no QA format); records with empty question/answer also skipped.",
    )
    log.info("Train: %d examples | Val: %d examples", len(train_ds), len(val_ds))

    # ── Training arguments ────────────────────────────────────────
    epochs = int(config.get("num_train_epochs", 3))
    max_steps = 5 if dry_run else -1
    if max_steps_override:
        max_steps = int(max_steps_override)
        output_dir = output_dir.with_name(output_dir.name + "_pilot")
        epochs = 1  # a step cap with 3 epochs would just be ignored
        # A pilot measures TRAINING throughput only. Without these, the
        # trainer runs a full evaluation over the whole val set after the
        # capped run -- on v7 that is 6,636 rows at eval-batch 2 = 3,319
        # steps, roughly 24 MINUTES, against a 9-minute 60-step measurement.
        # It also wrote a ~300 MB checkpoint nobody wants. Measured on the
        # 2026-09-08 pilot before this fix.
        config = dict(config)
        config["eval_strategy"] = "no"
        config["save_strategy"] = "no"
        config["load_best_model_at_end"] = False
        config["early_stopping_patience"] = 0

    training_args = TrainingArguments(
        # Without seed= the Trainer re-seeds with its default 42 at
        # construction, so the yaml seed only reached LoRA init and data order
        # / dropout were always seed 42 (audit 2026-09-23 §4).
        seed=int(config.get("seed", 42)),
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=int(config.get("per_device_train_batch_size", 8)),
        per_device_eval_batch_size=int(config.get("per_device_eval_batch_size", 8)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 2)),
        learning_rate=float(config.get("learning_rate", 4.62e-4)),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(config.get("warmup_ratio", 0.082)),
        weight_decay=float(config.get("weight_decay", 0.011)),
        logging_steps=int(config.get("logging_steps", 10)),
        logging_dir=str(run_dir / "logs"),
        eval_strategy=config.get("eval_strategy", "epoch"),
        save_strategy=config.get("save_strategy", "epoch"),
        save_total_limit=int(config.get("save_total_limit", 3)),
        load_best_model_at_end=bool(config.get("load_best_model_at_end", True)),
        metric_for_best_model=config.get("metric_for_best_model", "eval_loss"),
        greater_is_better=bool(config.get("greater_is_better", False)),
        fp16=False,
        bf16=bool(config.get("bf16", True)),
        tf32=bool(config.get("tf32", True)),
        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Length-grouped batching. HF's LengthGroupedSampler sorts within a
        # large megabatch and shuffles the megabatches, i.e. the
        # bucket-then-shuffle recipe -- so batches are length-homogeneous
        # without becoming content-homogeneous.
        #
        # Measured on v6's mixture (docs/performance_review_2026_09_07.md):
        # padding waste at micro-batch 2 is 40.6% in random order and 0.5%
        # length-sorted, so this is worth up to ~40% of training compute.
        #
        # DEFAULT FALSE so v1-v6 remain bit-reproducible. Turning it on
        # changes which examples share an optimizer step -- equivalent to a
        # reseed, not a recipe change (LR, epochs, batch shape, LoRA rank all
        # untouched) -- but it is a change, so it is opt-in per config.
        group_by_length=bool(config.get("group_by_length", False)),
        dataloader_num_workers=int(config.get("dataloader_num_workers", 4)),
        dataloader_pin_memory=bool(config.get("dataloader_pin_memory", True)),
        report_to=config.get("report_to", []),
        run_name=f"llama-lora-{datetime.now().strftime('%Y%m%d_%H%M')}",
    )

    # ── Callbacks ─────────────────────────────────────────────────
    callbacks = []
    patience = int(config.get("early_stopping_patience", 0))
    if patience > 0 and not dry_run:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
        log.info("Early stopping enabled (patience=%d epochs)", patience)

    # Live progress reporter
    live_cb = LiveProgressCallback(run_dir, config, len(train_ds), len(val_ds))
    callbacks.append(live_cb)
    log.info("Live progress report: %s", live_cb.report_path)

    # ── Trainer ───────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    log.info("=" * 70)
    log.info("Starting training | %d examples | %d epochs | LR=%.2e",
             len(train_ds), epochs, float(config.get("learning_rate", 4.62e-4)))
    log.info("=" * 70)
    print_gpu_memory(log)

    # ── Train ─────────────────────────────────────────────────────
    with Timer("training") as t:
        train_result = trainer.train()

    log.info("Training complete in %s", t.elapsed_str)
    print_gpu_memory(log)

    # ── Save final adapter ────────────────────────────────────────
    final_path = output_dir / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    log.info("✅ Final adapter saved: %s", final_path)

    # ── Save training summary ─────────────────────────────────────
    save_training_summary(run_dir, train_result, t.elapsed_str, config)

    # ── Log final eval loss ───────────────────────────────────────
    # HPO trials set final_eval: false -- they are selected on dev-set
    # ACCURACY, and a val-loss pass over ~2.9k rows costs ~20 min per trial.
    if not dry_run and config.get("final_eval", True):
        eval_metrics = trainer.evaluate()
        log.info("Final eval_loss: %.4f", eval_metrics.get("eval_loss", float("nan")))
        save_json(eval_metrics, run_dir / "final_eval_metrics.json")

    # ── Copy logs to progress dir for easy access ─────────────────
    log_history_path = run_dir / "trainer_log_history.json"
    save_json(trainer.state.log_history, log_history_path)
    log.info("Trainer log history saved: %s", log_history_path)

    log.info("=" * 70)
    log.info("Run complete! Progress directory: %s", run_dir)
    log.info("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LLaMA-3.2-3B LoRA flat fine-tuning",
    )
    p.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to config.yaml (default: experiments/finetuning/LLaMA/config.yaml).",
    )
    p.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force 4-bit QLoRA on/off (default: use config.yaml value).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run only 5 training steps (pipeline verification).",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop after N steps and write to <output_dir>_pilot. For "
             "throughput/OOM pilots: --dry-run's 5 steps catch an OOM (v4's "
             "hit on step 1) but are too warmup-dominated to time. Does "
             "nothing unless passed, so normal runs are unaffected.",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    train(
        config_path=args.config,
        load_in_4bit=args.load_in_4bit,
        dry_run=args.dry_run,
        max_steps_override=args.max_steps,
    )
