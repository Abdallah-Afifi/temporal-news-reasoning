#!/usr/bin/env python3
"""
experiments/finetuning/qwen3.5-9b-model/train.py
====================================
Flat LoRA fine-tuning of Qwen3.5-9B-Instruct on combined_80_20_split.

This mirrors the LLaMA/Mistral training logic:
- Shared normalization + chat-template formatting
- Label masking for prompt tokens
- Live progress report and checkpointing
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from experiments.finetuning.LLaMA.data_loader import load_flat_datasets
from experiments.finetuning.shared.utils import (
    Timer,
    get_device_info,
    load_yaml,
    preflight_check,
    print_gpu_memory,
    save_json,
    setup_logger,
)

log = setup_logger("qwen3_5.train")


def load_tokenizer(model_name_or_path: str) -> AutoTokenizer:
    log.info("Loading Qwen tokenizer from: %s", model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        log.info("pad_token set to eos_token (%r)", tokenizer.eos_token)
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(
    model_name_or_path: str,
    dtype: str = "bfloat16",
    load_in_4bit: bool = False,
    attn_implementation: Optional[str] = None,
) -> AutoModelForCausalLM:
    dtype_map = {
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float16": torch.float16, "fp16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype.lower(), torch.bfloat16)

    load_kwargs: dict = {
        "torch_dtype": torch_dtype,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation

    if load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        log.info("4-bit QLoRA enabled (NF4 + double quantization)")

    log.info("Loading Qwen3.5 model from: %s", model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)

    if load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
        log.info("Model prepared for k-bit training")

    return model


def build_lora_config(cfg: dict) -> LoraConfig:
    return LoraConfig(
        r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        target_modules=list(cfg.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
        ])),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def init_progress_dir(progress_dir: Path, config: dict) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = progress_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    save_json(config, run_dir / "config_snapshot.json")

    readme = run_dir / "README.md"
    readme.write_text(
        "# Qwen3.5-9B LoRA Training Run\n\n"
        f"- Started: {datetime.now().isoformat()}\n"
        f"- Model: {config.get('model_path', 'N/A')}\n"
        f"- Train data: {config.get('train_data', 'N/A')}\n"
        f"- Val data: {config.get('val_data', 'N/A')}\n"
        f"- Epochs: {config.get('num_train_epochs', 'N/A')}\n"
        f"- Learning rate: {config.get('learning_rate', 'N/A')}\n"
        f"- LoRA r: {config.get('lora_r', 'N/A')}\n"
        f"- Batch size: {config.get('per_device_train_batch_size', 'N/A')}\n"
        f"- Grad accum: {config.get('gradient_accumulation_steps', 'N/A')}\n"
        f"- Effective batch: {config.get('per_device_train_batch_size', 2) * config.get('gradient_accumulation_steps', 4)}\n"
        f"- Max seq length: {config.get('max_seq_length', 'N/A')}\n\n"
        "## Files\n\n"
        "- config_snapshot.json - full config\n"
        "- data_stats.json - dataset statistics\n"
        "- live_report.md - live training progress\n"
        "- training_summary.json - final results\n"
        "- final_eval_metrics.json - final eval metrics\n"
        "- trainer_log_history.json - per-step loss log\n",
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


class LiveProgressCallback(TrainerCallback):
    """Writes a live_report.md file with progress updates."""

    def __init__(self, run_dir: Path, config: dict, train_size: int, val_size: int):
        self.report_path = run_dir / "live_report.md"
        self.config = config
        self.train_size = train_size
        self.val_size = val_size
        self.start_time = datetime.now()
        self.train_losses: list[tuple[int, float]] = []
        self.eval_losses: list[tuple[float, float]] = []
        self.best_eval_loss = float("inf")

    def _gpu_stats(self) -> str:
        if not torch.cuda.is_available():
            return "CPU"
        alloc = torch.cuda.memory_allocated() / 1e9
        resv = torch.cuda.memory_reserved() / 1e9
        return f"{alloc:.1f} GB alloc / {resv:.1f} GB reserved"

    def _write_report(self, state, is_eval: bool = False) -> None:
        now = datetime.now()
        elapsed = now - self.start_time
        hours, rem = divmod(int(elapsed.total_seconds()), 3600)
        mins, secs = divmod(rem, 60)
        elapsed_str = f"{hours:02d}:{mins:02d}:{secs:02d}"

        total_steps = state.max_steps if state.max_steps > 0 else int(
            self.train_size / (self.config.get("per_device_train_batch_size", 2) * self.config.get("gradient_accumulation_steps", 4)) * self.config.get("num_train_epochs", 3)
        )
        pct = (state.global_step / total_steps * 100) if total_steps > 0 else 0

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
            "# LIVE Training Progress",
            "",
            f"Last updated: {now.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Status: {'training' if not is_eval else 'evaluating'}",
            f"Elapsed: {elapsed_str} | ETA: {eta_str}",
            f"Step: {state.global_step} / {total_steps} ({pct:.1f}%)",
            f"Epoch: {state.epoch:.2f} / {self.config.get('num_train_epochs', 3)}",
            f"GPU: {self._gpu_stats()}",
            "",
            "---",
            "",
            "## Training Loss (recent)",
            "",
            "| Step | Train Loss |",
            "|-----:|-----------:|",
        ]

        for step, loss in self.train_losses[-30:]:
            lines.append(f"| {step} | {loss:.4f} |")

        lines.extend(["", "---", "", "## Eval Loss (per epoch)", ""])

        if self.eval_losses:
            lines.append("| Epoch | Eval Loss | vs Best | Status |")
            lines.append("|------:|----------:|--------:|:------:|")
            for epoch, eloss in self.eval_losses:
                diff = eloss - self.best_eval_loss
                status = "best" if abs(diff) < 1e-6 else ("+" + f"{diff:.4f}" if diff > 0 else f"{diff:.4f}")
                lines.append(f"| {epoch:.1f} | {eloss:.4f} | {diff:+.4f} | {status} |")
            lines.extend(["", f"Best eval_loss: {self.best_eval_loss:.4f}"])
        else:
            lines.append("No evaluation yet - first eval after epoch 1.")

        lines.extend([
            "",
            "---",
            "",
            "## Config Summary",
            "",
            f"- Model: Qwen3.5-9B-Instruct",
            f"- LR: {self.config.get('learning_rate')}",
            f"- Batch: {self.config.get('per_device_train_batch_size')} x {self.config.get('gradient_accumulation_steps')} = {self.config.get('per_device_train_batch_size', 2) * self.config.get('gradient_accumulation_steps', 4)} effective",
            f"- Train: {self.train_size} examples | Val: {self.val_size} examples",
            f"- Max seq: {self.config.get('max_seq_length')}",
            f"- LoRA r: {self.config.get('lora_r')} | alpha: {self.config.get('lora_alpha')}",
            f"- Early stopping patience: {self.config.get('early_stopping_patience', 'off')}",
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


def train(
    config_path: str | Path,
    load_in_4bit_cli: bool = False,
    dry_run: bool = False,
) -> None:
    config = load_yaml(config_path)
    set_seed(int(config.get("seed", 42)))
    log.info("Device info: %s", get_device_info())

    model_path = str((_REPO_ROOT / config.get("model_path", "models/qwen3.5-9b-model")).resolve())
    train_data = str((_REPO_ROOT / config["train_data"]).resolve())
    val_data = str((_REPO_ROOT / config["val_data"]).resolve())
    output_dir = (_REPO_ROOT / config.get("output_dir", "checkpoints/qwen3.5-9b-model")).resolve()
    progress_dir = (_REPO_ROOT / config.get("progress_dir", "experiments/finetuning/qwen3.5-9b-model/Progress")).resolve()

    use_4bit = load_in_4bit_cli or config.get("load_in_4bit", False)
    preflight_check(
        data_paths=[Path(train_data), Path(val_data)],
        output_dir=output_dir,
        min_vram_gb=8.0 if use_4bit else 20.0,
        logger=log,
    )

    run_dir = init_progress_dir(progress_dir, config)

    model_source = model_path if (Path(model_path) / "config.json").exists() else model_path
    tokenizer = load_tokenizer(model_source)
    model = load_model(
        model_source,
        dtype=config.get("dtype", "bfloat16"),
        load_in_4bit=use_4bit,
        attn_implementation=config.get("attn_implementation"),
    )

    peft_config = build_lora_config(config)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    max_length = int(config.get("max_seq_length", 2048))
    train_ds, val_ds = load_flat_datasets(train_data, val_data, tokenizer, max_length, log)

    save_data_stats(
        run_dir, len(train_ds), len(val_ds),
        skipped_info="Temprel records skipped (no QA format); records with empty question/answer also skipped.",
    )
    log.info("Train: %d examples | Val: %d examples", len(train_ds), len(val_ds))

    epochs = int(config.get("num_train_epochs", 3))
    max_steps = 5 if dry_run else -1

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=int(config.get("per_device_train_batch_size", 2)),
        per_device_eval_batch_size=int(config.get("per_device_eval_batch_size", 2)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 4)),
        learning_rate=float(config.get("learning_rate", 3.0e-4)),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(config.get("warmup_ratio", 0.15)),
        weight_decay=float(config.get("weight_decay", 0.02)),
        logging_steps=int(config.get("logging_steps", 10)),
        logging_dir=str(run_dir / "logs"),
        eval_strategy=config.get("eval_strategy", "epoch"),
        save_strategy=config.get("save_strategy", "epoch"),
        save_total_limit=int(config.get("save_total_limit", 3)),
        load_best_model_at_end=bool(config.get("load_best_model_at_end", True)),
        metric_for_best_model=config.get("metric_for_best_model", "eval_loss"),
        greater_is_better=bool(config.get("greater_is_better", False)),
        fp16=bool(config.get("fp16", False)),
        bf16=bool(config.get("bf16", True)),
        tf32=bool(config.get("tf32", True)),
        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        dataloader_num_workers=int(config.get("dataloader_num_workers", 4)),
        dataloader_pin_memory=bool(config.get("dataloader_pin_memory", True)),
        report_to=config.get("report_to", []),
        run_name=f"qwen3-5-lora-{datetime.now().strftime('%Y%m%d_%H%M')}",
    )

    callbacks = []
    patience = int(config.get("early_stopping_patience", 0))
    if patience > 0 and not dry_run:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
        log.info("Early stopping enabled (patience=%d epochs)", patience)

    live_cb = LiveProgressCallback(run_dir, config, len(train_ds), len(val_ds))
    callbacks.append(live_cb)
    log.info("Live progress report: %s", live_cb.report_path)

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
             len(train_ds), epochs, float(config.get("learning_rate", 3.0e-4)))
    log.info("=" * 70)
    print_gpu_memory(log)

    with Timer("training") as t:
        train_result = trainer.train()

    log.info("Training complete in %s", t.elapsed_str)
    print_gpu_memory(log)

    final_path = output_dir / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    log.info("Final adapter saved: %s", final_path)

    save_training_summary(run_dir, train_result, t.elapsed_str, config)

    if not dry_run:
        eval_metrics = trainer.evaluate()
        log.info("Final eval_loss: %.4f", eval_metrics.get("eval_loss", float("nan")))
        save_json(eval_metrics, run_dir / "final_eval_metrics.json")

    log_history_path = run_dir / "trainer_log_history.json"
    save_json(trainer.state.log_history, log_history_path)
    log.info("Trainer log history saved: %s", log_history_path)

    log.info("=" * 70)
    log.info("Run complete. Progress directory: %s", run_dir)
    log.info("=" * 70)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Qwen3.5-9B LoRA flat fine-tuning",
    )
    p.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to config.yaml (default: experiments/finetuning/qwen3.5-9b-model/config.yaml).",
    )
    p.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Enable 4-bit QLoRA quantization (reduces VRAM).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run only 5 training steps (pipeline verification).",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    train(
        config_path=args.config,
        load_in_4bit_cli=args.load_in_4bit,
        dry_run=args.dry_run,
    )
