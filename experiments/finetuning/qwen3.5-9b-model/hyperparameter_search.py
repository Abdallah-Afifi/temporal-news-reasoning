#!/usr/bin/env python3
"""
Optuna-based Hyperparameter Search for Qwen3.5-9B-Instruct
Fine-Tuning on combined_80_20_split QA Dataset

Optimizes:
- Learning rate (1e-5 to 1e-3, log scale)
- Batch size (2, 4, 8)
- Warmup ratio (0.05 to 0.3)
- Weight decay (0.0 to 0.1)
- LoRA rank (8, 16, 32)

Usage:
    python hyperparameter_search.py --n-trials 20
    python hyperparameter_search.py --n-trials 50 --seed 42
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import csv
from datetime import datetime

import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import get_peft_model, LoraConfig, TaskType
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HyperparameterSearcher:
    """Optuna-based hyperparameter optimizer for Qwen3.5-9B."""
    
    def __init__(
        self,
        model_path: str = "models/qwen3.5-9b-model",
        data_root: str = ".",
        results_dir: str = "results/hyperparam_search_9b",
        n_trials: int = 20,
        seed: int = 42,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ):
        self.model_path = model_path
        self.data_root = data_root
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.n_trials = n_trials
        self.seed = seed
        self.device = device
        self.dtype = dtype
        
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.trial_results = []
        
        logger.info(f"Model: {model_path}")
        logger.info(f"Data: {Path(data_root) / 'data/combined_80_20_split'}")
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"Number of trials: {n_trials}")
    
    def load_datasets(self):
        """Load QA datasets from combined_80_20_split."""
        if self.train_dataset is not None:
            return  # Already loaded
        
        import jsonlines
        
        train_path = Path(self.data_root) / "data/combined_80_20_split/train.jsonl"
        val_path = Path(self.data_root) / "data/combined_80_20_split/val.jsonl"
        
        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError(f"Dataset not found at {train_path} or {val_path}")
        
        # Load training data
        train_examples = []
        with jsonlines.open(train_path) as reader:
            for obj in reader:
                train_examples.append(obj)
        
        # Load validation data
        val_examples = []
        with jsonlines.open(val_path) as reader:
            for obj in reader:
                val_examples.append(obj)
        
        logger.info(f"Loaded {len(train_examples)} training examples")
        logger.info(f"Loaded {len(val_examples)} validation examples")
        
        # Format datasets
        self.train_dataset = self._format_examples(train_examples)
        self.val_dataset = self._format_examples(val_examples)
    
    def _format_examples(self, examples):
        """Format QA examples to training text."""
        formatted = []
        for ex in examples:
            question = ex.get('question', '')
            answers = ex.get('answers', [])
            subject = ex.get('subject', '')
            
            # Format as QA pairs
            answers_text = "\n".join([f"  - {ans}" for ans in answers])
            
            text = f"""Question: {question}
Subject: {subject}
Answers:
{answers_text}"""
            
            formatted.append(text)
        
        return formatted
    
    def tokenize_datasets(self, max_seq_length: int = 2048):
        """Tokenize datasets."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        
        # Tokenize
        train_tokens = [
            self.tokenizer.encode(
                text,
                max_length=max_seq_length,
                truncation=True,
                return_tensors="pt"
            ) for text in self.train_dataset
        ]
        
        val_tokens = [
            self.tokenizer.encode(
                text,
                max_length=max_seq_length,
                truncation=True,
                return_tensors="pt"
            ) for text in self.val_dataset
        ]
        
        return train_tokens, val_tokens
    
    def create_model(self, lora_r: int = 16):
        """Create Qwen3.5-9B model with LoRA."""
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device,
            torch_dtype=torch.bfloat16 if self.dtype == "bfloat16" else torch.float32,
            trust_remote_code=True,
        )
        
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
    
    def generate_report(self, df: pd.DataFrame):
        """Generate human-readable analysis report."""
        report_path = self.results_dir / "hyperparameter_analysis.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("QWEN3.5-9B HYPERPARAMETER SEARCH ANALYSIS\n")
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
        batch_size = trial.suggest_categorical('batch_size', [2, 4, 8])
        warmup_ratio = trial.suggest_float('warmup_ratio', 0.05, 0.3)
        weight_decay = trial.suggest_float('weight_decay', 0.0, 0.1)
        lora_r = trial.suggest_categorical('lora_r', [8, 16, 32])
        
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
                logging_steps=50,
                eval_strategy="epoch",
                save_strategy="no",
                bf16=True,
                gradient_checkpointing=True,
                gradient_accumulation_steps=4,
                lr_scheduler_type="cosine",
                report_to=[],
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.val_dataset,
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
            
            return eval_loss
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {str(e)}")
            raise optuna.TrialPruned()
    
    def run_search(self):
        """Run hyperparameter search."""
        sampler = TPESampler(seed=self.seed)
        pruner = MedianPruner(n_warmup_steps=1)
        
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
    parser = argparse.ArgumentParser(description="Hyperparameter search for Qwen3.5-9B")
    parser.add_argument('--n-trials', type=int, default=20, help='Number of trials')
    parser.add_argument('--model-path', type=str, default='models/qwen3.5-9b-model')
    parser.add_argument('--data-root', type=str, default='.')
    parser.add_argument('--results-dir', type=str, default='experiments/finetuning/qwen3.5-9b-model/results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dtype', type=str, default='bfloat16')
    
    args = parser.parse_args()
    
    searcher = HyperparameterSearcher(
        model_path=args.model_path,
        data_root=args.data_root,
        results_dir=args.results_dir,
        n_trials=args.n_trials,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
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
