#!/usr/bin/env python3
"""
Optuna-based Hyperparameter Search for LLaMA-3.2-3B-Instruct
Fine-tuning on combined_80_20_split QA dataset

Optimizes:
- Learning rate (1e-5 to 1e-3, log scale)
- Batch size (2, 4, 8)
- Warmup ratio (0.05 to 0.3)
- Weight decay (0.0 to 0.1)
- LoRA rank (8, 16, 32)

Usage:
    python hyperparameter_search.py --n-trials 20
    python hyperparameter_search.py --n-trials 50 --seed 42

SUPERSEDED 2026-09-23 (docs/audit_2026_09_23.md §5) -- kept for provenance of
the learning rate every arm v2-v10 used. Its objective is eval loss on the
in-distribution val split of the OLD combined_80_20_split, on 500 train / 200
val rows for 1 epoch at max_seq_length 1024; the chosen LR was then applied to
3 epochs over 12k rows. Batch size and LoRA rank, listed above as searched,
are in fact held fixed by the code. Use scripts/hpo_v11.py, which selects on
benchmark-format dev accuracy (docs/hpo_v11_protocol.md).
"""

import os
import json
import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import get_peft_model, LoraConfig, TaskType
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.finetuning.LLaMA.data_loader import (
    load_tokenizer as load_llama_tokenizer,
    normalize_and_tokenize,
)
from experiments.finetuning.shared.utils import read_jsonl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLaMAHyperparameterSearcher:
    """Optuna-based hyperparameter optimizer for LLaMA-3.2-3B."""
    
    def __init__(
        self,
        model_path: str = "models/Llama-3.2-3B-Instruct",
        data_root: str = ".",
        results_dir: str = "results/hyperparam_search_llama",
        n_trials: int = 20,
        seed: int = 42,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_train_examples: int = 500,
        max_val_examples: int = 200,
        max_seq_length: int = 1024,
        batch_size: int = 4,
        lora_r: int = 16,
    ):
        self.model_path = model_path
        self.data_root = data_root
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.n_trials = n_trials
        self.seed = seed
        self.device = device
        self.dtype = dtype
        self.max_train_examples = max_train_examples
        self.max_val_examples = max_val_examples
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.lora_r = lora_r
        self.live_report_path = self.results_dir / "live_report.md"
        
        self.tokenizer = None
        self._train_records = None
        self._val_records = None
        self.train_dataset = None
        self.val_dataset = None
        self.trial_results = []
        
        logger.info(f"Model: {model_path}")
        logger.info(f"Data: {Path(data_root) / 'data/combined_80_20_split'}")
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"Number of trials: {n_trials}")
        logger.info(f"Train subset: {max_train_examples} examples")
        logger.info(f"Val subset: {max_val_examples} examples")
        logger.info(f"Max sequence length: {max_seq_length}")
        logger.info(f"Fixed batch size: {batch_size}")
        logger.info(f"Fixed LoRA rank: {lora_r}")
    
    def load_datasets(self):
        """Load + subsample raw QA records from combined_80_20_split.

        Keeps the records in RAW form here; ``tokenize_datasets`` applies the
        SAME normalization + chat-template formatting + label masking as the
        real training pipeline, so the HPO objective matches training.
        """
        if self.train_dataset is not None:
            return  # Already loaded

        train_path = Path(self.data_root) / "data/combined_80_20_split/train.jsonl"
        val_path = Path(self.data_root) / "data/combined_80_20_split/val.jsonl"

        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError(f"Dataset not found at {train_path} or {val_path}")

        train_records = read_jsonl(train_path)
        val_records = read_jsonl(val_path)

        if len(train_records) > self.max_train_examples:
            train_records = random.Random(self.seed).sample(train_records, self.max_train_examples)
        if len(val_records) > self.max_val_examples:
            val_records = random.Random(self.seed + 1).sample(val_records, self.max_val_examples)

        logger.info(f"Loaded {len(train_records)} training records")
        logger.info(f"Loaded {len(val_records)} validation records")

        self._train_records = train_records
        self._val_records = val_records

    def tokenize_datasets(self):
        """Tokenize with the real training pipeline (chat template + label masking).

        Handles all three data schemas (TLQA / TimeQA; Temprel skipped) and
        produces ``input_ids`` / ``attention_mask`` / ``labels`` with prompt
        tokens masked to -100 — identical to what ``train.py`` trains on.
        """
        self.tokenizer = load_llama_tokenizer(self.model_path)

        self.train_dataset = normalize_and_tokenize(
            self._train_records, self.tokenizer, self.max_seq_length,
        )
        self.val_dataset = normalize_and_tokenize(
            self._val_records, self.tokenizer, self.max_seq_length,
        )
        logger.info(f"Tokenized: train={len(self.train_dataset)}, val={len(self.val_dataset)} examples")
    
    def create_model(self, lora_r: int = 16):
        """Create LLaMA-3.2-3B model with LoRA."""
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device,
            torch_dtype=torch.bfloat16 if self.dtype == "bfloat16" else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        model.config.use_cache = False
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model
    
    def generate_results_table(self):
        """Generate CSV table of all trials."""
        df = pd.DataFrame(self.trial_results)
        df = df.sort_values('eval_loss')
        
        csv_path = self.results_dir / "hyperparameter_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to {csv_path}")
        
        return df

    def _write_live_report(self):
        """Write a compact live report that updates after every completed trial."""
        report_lines = [
            "# Live HPO Report",
            "",
            f"Updated: {datetime.now().isoformat(timespec='seconds')}",
            f"Model: {self.model_path}",
            f"Dataset: combined_80_20_split ({self.max_train_examples} train / {self.max_val_examples} val)",
            f"Fixed batch size: {self.batch_size}",
            f"Fixed LoRA rank: {self.lora_r}",
            f"Max sequence length: {self.max_seq_length}",
            f"Completed trials: {len(self.trial_results)} / {self.n_trials}",
            "",
        ]

        if self.trial_results:
            df = pd.DataFrame(self.trial_results).sort_values('eval_loss')
            best = df.iloc[0]
            report_lines.extend([
                "## Best So Far",
                f"- Trial: {int(best['trial'])}",
                f"- Eval loss: {best['eval_loss']:.4f}",
                f"- Learning rate: {best['learning_rate']:.2e}",
                f"- Warmup ratio: {best['warmup_ratio']:.3f}",
                f"- Weight decay: {best['weight_decay']:.4f}",
                "",
                "## Completed Trials",
                "| Trial | Learning Rate | Warmup | Weight Decay | Eval Loss |",
                "|:---:|:---:|:---:|:---:|:---:|",
            ])

            for _, row in df.iterrows():
                report_lines.append(
                    f"| {int(row['trial'])} | {row['learning_rate']:.2e} | {row['warmup_ratio']:.3f} | {row['weight_decay']:.4f} | {row['eval_loss']:.4f} |"
                )
        else:
            report_lines.append("No completed trials yet.")

        self.live_report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    
    def generate_report(self, df: pd.DataFrame):
        """Generate human-readable analysis report."""
        report_path = self.results_dir / "hyperparameter_analysis.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("LLAMA-3.2-3B HYPERPARAMETER SEARCH ANALYSIS\n")
            f.write("Dataset: combined_80_20_split\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Search Date: {datetime.now().isoformat()}\n")
            f.write(f"Total Trials: {len(df)}\n")
            f.write(f"Best Loss: {df['eval_loss'].min():.4f}\n\n")
            
            f.write("LOSS STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean:     {df['eval_loss'].mean():.4f}\n")
            f.write(f"Median:   {df['eval_loss'].median():.4f}\n")
            f.write(f"Std Dev:  {df['eval_loss'].std():.4f}\n")
            f.write(f"Min:      {df['eval_loss'].min():.4f}\n")
            f.write(f"Max:      {df['eval_loss'].max():.4f}\n\n")
            
            f.write("TOP 10 CONFIGURATIONS\n")
            f.write("-" * 80 + "\n")
            top_10 = df.head(10)
            for idx, (_, row) in enumerate(top_10.iterrows(), 1):
                f.write(f"\n{idx}. Loss: {row['eval_loss']:.4f}\n")
                f.write(f"   Learning Rate: {row['learning_rate']:.2e}\n")
                f.write(f"   Batch Size: {int(row['batch_size'])}\n")
                f.write(f"   Warmup Ratio: {row['warmup_ratio']:.3f}\n")
                f.write(f"   Weight Decay: {row['weight_decay']:.4f}\n")
                f.write(f"   LoRA r: {int(row['lora_r'])}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("PARAMETER IMPORTANCE\n")
            f.write("=" * 80 + "\n\n")
            
            # Learning rate impact
            f.write("Learning Rate Impact:\n")
            lr_groups = df.groupby(pd.cut(df['learning_rate'], bins=5))['eval_loss'].agg(['count', 'mean', 'std', 'min'])
            f.write(lr_groups.to_string())
            f.write("\n\n")
            
            # Batch size impact
            f.write("Batch Size Impact:\n")
            bs_groups = df.groupby('batch_size')['eval_loss'].agg(['count', 'mean', 'std', 'min'])
            f.write(bs_groups.to_string())
            f.write("\n\n")
            
            # LoRA rank impact
            f.write("LoRA Rank Impact:\n")
            r_groups = df.groupby('lora_r')['eval_loss'].agg(['count', 'mean', 'std', 'min'])
            f.write(r_groups.to_string())
            f.write("\n")
        
        logger.info(f"Report saved to {report_path}")
    
    def save_best_hyperparameters(self, df: pd.DataFrame):
        """Save best hyperparameters to JSON."""
        best_trial = df.iloc[0]
        
        best_params = {
            "learning_rate": float(best_trial['learning_rate']),
            "per_device_train_batch_size": int(best_trial['batch_size']),
            "warmup_ratio": float(best_trial['warmup_ratio']),
            "weight_decay": float(best_trial['weight_decay']),
            "lora_r": int(best_trial['lora_r']),
            "eval_loss": float(best_trial['eval_loss']),
            "trial_number": int(best_trial['trial']),
        }
        
        json_path = self.results_dir / "best_hyperparameters.json"
        with open(json_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        
        logger.info(f"Best parameters saved to {json_path}")
        return best_params
    
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function - single trial."""
        # Suggest hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        warmup_ratio = trial.suggest_float('warmup_ratio', 0.05, 0.2)
        weight_decay = trial.suggest_float('weight_decay', 0.0, 0.06)
        batch_size = self.batch_size
        lora_r = self.lora_r
        
        try:
            # Load datasets once
            if self.train_dataset is None:
                self.load_datasets()
                self.tokenize_datasets()
            
            # Create fresh model for this trial
            model = self.create_model(lora_r=lora_r)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=str(self.results_dir / f"trial_{trial.number}"),
                learning_rate=learning_rate,
                per_device_train_batch_size=int(batch_size),
                per_device_eval_batch_size=int(batch_size),
                num_train_epochs=1,
                warmup_ratio=warmup_ratio,
                weight_decay=weight_decay,
                logging_steps=25,
                eval_strategy="epoch",
                save_strategy="no",
                bf16=True,
                gradient_checkpointing=True,
                gradient_accumulation_steps=4,
                lr_scheduler_type="cosine",
                report_to=[],
            )
            
            # Create trainer — collator pads pre-tokenized input_ids/labels
            # (labels keep their -100 prompt mask; pads get -100 too)
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                padding=True,
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.val_dataset,
                data_collator=data_collator,
            )
            
            # Train
            logger.info(f"Trial {trial.number}: LR={learning_rate:.2e}, BS={batch_size}, WU={warmup_ratio:.3f}, WD={weight_decay:.4f}, r={lora_r}")
            train_result = trainer.train()
            
            # Evaluate
            eval_result = trainer.evaluate()
            eval_loss = eval_result['eval_loss']
            
            # Store results
            self.trial_results.append({
                'trial': trial.number,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'warmup_ratio': warmup_ratio,
                'weight_decay': weight_decay,
                'lora_r': lora_r,
                'eval_loss': eval_loss,
            })
            
            logger.info(f"Trial {trial.number} completed - Eval Loss: {eval_loss:.4f}")
            
            # Clean up
            del model, trainer
            torch.cuda.empty_cache()

            self._write_live_report()
            
            return eval_loss
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {str(e)}")
            raise optuna.TrialPruned()
    
    def run_search(self):
        """Run hyperparameter search."""
        sampler = TPESampler(seed=self.seed)
        pruner = MedianPruner(n_warmup_steps=1)

        self._write_live_report()
        
        study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
            pruner=pruner,
        )
        
        logger.info(f"Starting hyperparameter search with {self.n_trials} trials")
        study.optimize(self.objective, n_trials=self.n_trials)
        
        # Generate results
        df = self.generate_results_table()
        self.generate_report(df)
        best_params = self.save_best_hyperparameters(df)
        
        return df, best_params


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search for LLaMA-3.2-3B")
    parser.add_argument('--n-trials', type=int, default=8, help='Number of trials')
    parser.add_argument('--model-path', type=str, default='models/Llama-3.2-3B-Instruct')
    parser.add_argument('--data-root', type=str, default='.')
    parser.add_argument('--results-dir', type=str, default='experiments/finetuning/LLaMA/results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dtype', type=str, default='bfloat16')
    parser.add_argument('--max-train-examples', type=int, default=500)
    parser.add_argument('--max-val-examples', type=int, default=200)
    parser.add_argument('--max-seq-length', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lora-r', type=int, default=16)
    
    args = parser.parse_args()
    
    searcher = LLaMAHyperparameterSearcher(
        model_path=args.model_path,
        data_root=args.data_root,
        results_dir=args.results_dir,
        n_trials=args.n_trials,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        max_train_examples=args.max_train_examples,
        max_val_examples=args.max_val_examples,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        lora_r=args.lora_r,
    )
    
    df, best_params = searcher.run_search()
    
    print("\n" + "=" * 80)
    print("SEARCH COMPLETED")
    print("=" * 80)
    print(f"\nBest Configuration:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
