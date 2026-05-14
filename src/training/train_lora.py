"""LoRA fine-tuning script for temporal reasoning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping.")
    return data


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _messages_to_text(messages: list[dict[str, Any]]) -> str:
    chunks: list[str] = [  target_modules: [q_proj, k_proj, v_proj, o_proj]]
    for msg in messages:
        role = str(msg.get("role", "user")).upper()
        content = str(msg.get("content", "")).strip()
        if content:
            chunks.append(f"{role}: {content}")
    return "\n".join(chunks)


def _example_to_text(example: dict[str, Any]) -> str:
    if "text" in example and str(example["text"]).strip():
        return str(example["text"]).strip()

    if isinstance(example.get("messages"), list):
        text = _messages_to_text(example["messages"])
        if text:
            return text

    instruction = str(example.get("instruction", "")).strip()
    input_text = str(example.get("input", "")).strip()
    output_raw = example.get("output", "")
    if not output_raw:
        output_raw = example.get("answer", "")
    if isinstance(output_raw, list):
        output_text = ", ".join(str(item).strip() for item in output_raw if str(item).strip())
    else:
        output_text = str(output_raw).strip()
    question = str(example.get("question", "")).strip()
    context = str(example.get("context", "")).strip()

    if instruction or output_text:
        prompt = instruction
        if input_text:
            prompt = f"{prompt}\n{input_text}" if prompt else input_text
        if prompt and output_text:
            return f"### Instruction\n{prompt}\n\n### Response\n{output_text}"

    if question and output_text:
        if context:
            return f"Context: {context}\nQuestion: {question}\nAnswer: {output_text}"
        return f"Question: {question}\nAnswer: {output_text}"

    if output_text:
        return output_text

    serialized = json.dumps(example, ensure_ascii=False)
    return serialized


def _resolve_model_source(model_cfg: dict[str, Any]) -> str:
    local_path = model_cfg.get("local_path")
    if local_path:
        local_path = Path(local_path)
        if (local_path / "config.json").exists():
            return str(local_path)
    model_name = model_cfg.get("name")
    if not model_name:
        raise ValueError("model.name is required in config.")
    return str(model_name)


def _dtype_from_config(dtype_value: str | None) -> torch.dtype:
    if dtype_value is None:
        return torch.bfloat16
    value = dtype_value.lower()
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if value in {"float16", "fp16", "half"}:
        return torch.float16
    if value in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_value}")


def _build_stage_texts(stage: dict[str, Any]) -> tuple[list[str], list[str] | None]:
    stage_name = stage.get("name", "unknown")
    data_path = stage.get("data")
    if not data_path:
        raise ValueError(f"Stage {stage_name} is missing 'data'.")

    records = _read_jsonl(data_path)
    texts = [_example_to_text(item) for item in records]
    texts = [item for item in texts if item.strip()]

    if not texts:
        raise ValueError(f"No usable examples found in stage data: {data_path}")

    # Optional external eval data for this stage
    eval_path = stage.get("eval_data")
    eval_texts: list[str] | None = None
    if eval_path:
        eval_records = _read_jsonl(eval_path)
        eval_texts = [_example_to_text(item) for item in eval_records]
        eval_texts = [t for t in eval_texts if t.strip()]
        if not eval_texts:
            eval_texts = None

    return texts, eval_texts


def _tokenize_dataset(
    tokenizer: Any,
    texts: list[str],
    max_length: int,
    eval_ratio: float,
    seed: int,
    eval_texts: list[str] | None = None,
) -> tuple[Dataset, Dataset | None]:
    """Tokenize texts and optionally use an explicit eval_texts list.

    If `eval_texts` is provided and non-empty, it will be used as the evaluation
    dataset instead of performing a train/test split.
    """
    dataset = Dataset.from_dict({"text": texts})

    if eval_texts:
        train_dataset = dataset
        eval_dataset = Dataset.from_dict({"text": eval_texts})
    else:
        if len(dataset) >= 100 and eval_ratio > 0:
            split = dataset.train_test_split(test_size=eval_ratio, seed=seed)
            train_dataset = split["train"]
            eval_dataset = split["test"]
        else:
            train_dataset = dataset
            eval_dataset = None

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    train_dataset = train_dataset.map(tokenize_batch, batched=True, remove_columns=["text"])
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(tokenize_batch, batched=True, remove_columns=["text"])

    return train_dataset, eval_dataset


def train(config_path: str):
    """Run LoRA fine-tuning with the given configuration.

    Args:
        config_path: Path to YAML configuration file.
    """
    config = _load_yaml(config_path)
    model_cfg = config.get("model", {})
    lora_cfg = config.get("lora", {})
    training_cfg = config.get("training", {})
    wandb_cfg = config.get("wandb", {})

    set_seed(int(training_cfg.get("seed", 42)))

    model_source = _resolve_model_source(model_cfg)
    output_dir = Path(training_cfg.get("output_dir", "./checkpoints/lora"))
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=_dtype_from_config(model_cfg.get("dtype")),
        device_map="auto",
        trust_remote_code=True,
    )

    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
        target_modules=list(lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        bias=str(lora_cfg.get("bias", "none")),
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    curriculum = config.get("curriculum", {})
    stages = curriculum.get("stages", [])
    if not stages:
        raise ValueError("curriculum.stages must contain at least one stage.")

    eval_ratio = float(training_cfg.get("eval_ratio", 0.02))
    seed = int(training_cfg.get("seed", 42))
    max_length = int(training_cfg.get("max_seq_length", 2048))

    report_to: list[str] = []
    run_name = None
    if isinstance(wandb_cfg, dict) and wandb_cfg.get("project"):
        report_to = ["wandb"]
        run_name = wandb_cfg.get("run_name")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    for stage in stages:
        stage_name = str(stage.get("name", "stage"))
        stage_epochs = float(stage.get("epochs", training_cfg.get("num_train_epochs", 1)))
        stage_max_steps = int(stage.get("max_steps", training_cfg.get("max_steps", -1)))
        stage_texts, stage_eval_texts = _build_stage_texts(stage)
        train_dataset, eval_dataset = _tokenize_dataset(
            tokenizer=tokenizer,
            texts=stage_texts,
            max_length=max_length,
            eval_ratio=eval_ratio,
            seed=seed,
            eval_texts=stage_eval_texts,
        )

        stage_output_dir = output_dir / stage_name
        stage_run_name = f"{run_name}-{stage_name}" if run_name else stage_name

        print(f"\n=== Training curriculum stage: {stage_name} ({len(stage_texts)} examples) ===")

        training_args = TrainingArguments(
            output_dir=str(stage_output_dir),
            num_train_epochs=stage_epochs,
            max_steps=stage_max_steps,
            per_device_train_batch_size=int(training_cfg.get("per_device_train_batch_size", 4)),
            gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 4)),
            learning_rate=float(training_cfg.get("learning_rate", 2e-4)),
            warmup_ratio=float(training_cfg.get("warmup_ratio", 0.1)),
            weight_decay=float(training_cfg.get("weight_decay", 0.01)),
            logging_steps=int(training_cfg.get("logging_steps", 10)),
            save_steps=int(training_cfg.get("save_steps", 500)),
            fp16=bool(training_cfg.get("fp16", False)),
            bf16=bool(training_cfg.get("bf16", True)),
            gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", True)),
            report_to=report_to,
            run_name=stage_run_name,
            eval_strategy="steps" if eval_dataset is not None else "no",
            eval_steps=int(training_cfg.get("eval_steps", 250)) if eval_dataset is not None else None,
            save_total_limit=int(training_cfg.get("save_total_limit", 3)),
            logging_dir=str(stage_output_dir / "logs"),
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            processing_class=tokenizer,
        )

        trainer.train()
        trainer.save_model(str(stage_output_dir / "final"))

    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning entrypoint")
    parser.add_argument("config", type=str, help="Path to YAML config")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    train(args.config)
