#!/usr/bin/env python3
"""
Evaluation script for Qwen3.5-9B fine-tuned models
Evaluates on combined_80_20_split validation set using various metrics
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict

import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import jsonlines
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QwenEvaluator:
    """Evaluator for Qwen3.5-9B models."""
    
    def __init__(self, model_path: str, data_root: str = "."):
        """Initialize evaluator."""
        self.model_path = model_path
        self.data_root = data_root
        self.tokenizer = None
        self.model = None
        
        logger.info(f"Model: {model_path}")
        logger.info(f"Data root: {data_root}")
    
    def load_model(self):
        """Load tokenizer and model."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        logger.info("Model and tokenizer loaded")
    
    def load_val_dataset(self):
        """Load validation dataset from combined_80_20_split."""
        val_path = Path(self.data_root) / "data/combined_80_20_split/val.jsonl"
        
        if not val_path.exists():
            raise FileNotFoundError(f"Validation dataset not found at {val_path}")
        
        val_examples = []
        with jsonlines.open(val_path) as reader:
            for obj in reader:
                val_examples.append(obj)
        
        logger.info(f"Loaded {len(val_examples)} validation examples")
        return val_examples
    
    def format_example(self, ex: Dict) -> str:
        """Format single example to training text."""
        question = ex.get('question', '')
        answers = ex.get('answers', [])
        subject = ex.get('subject', '')
        
        # Format as QA pair
        answers_text = "\n".join([f"  - {ans}" for ans in answers])
        
        text = f"""Question: {question}
Subject: {subject}
Answers:
{answers_text}"""
        
        return text
    
    def compute_loss(self, examples):
        """Compute loss on examples."""
        losses = []
        
        for ex in tqdm(examples, desc="Computing loss"):
            text = self.format_example(ex)
            
            # Tokenize
            inputs = self.tokenizer.encode(text, return_tensors="pt")
            inputs = inputs.to(self.model.device)
            
            # Compute loss
            with torch.no_grad():
                outputs = self.model(inputs, labels=inputs)
                loss = outputs.loss.item()
            
            losses.append(loss)
        
        return losses
    
    def generate_completions(self, examples: list, max_length: int = 100, num_samples: int = 5):
        """Generate model completions on examples."""
        completions = []
        
        for ex in tqdm(examples[:num_samples], desc="Generating completions"):
            question = ex.get('question', '')
            subject = ex.get('subject', '')
            
            prompt = f"""Question: {question}
Subject: {subject}
Answer:"""
            
            # Tokenize
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            inputs = inputs.to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    num_beams=1,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            completions.append({
                'question': question,
                'subject': subject,
                'prompt': prompt,
                'generated_text': generated_text,
            })
        
        return completions
    
    def evaluate(self, output_dir: Optional[str] = None):
        """Run full evaluation."""
        # Load model
        self.load_model()
        
        # Load dataset
        val_examples = self.load_val_dataset()
        
        # Compute loss
        logger.info("Computing validation loss...")
        losses = self.compute_loss(val_examples)
        
        # Generate sample completions
        logger.info("Generating sample completions...")
        completions = self.generate_completions(val_examples, num_samples=5)
        
        # Compute statistics
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        min_loss = np.min(losses)
        max_loss = np.max(losses)
        
        # Print results
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        print(f"\nValidation Loss:")
        print(f"  Mean: {mean_loss:.4f}")
        print(f"  Std:  {std_loss:.4f}")
        print(f"  Min:  {min_loss:.4f}")
        print(f"  Max:  {max_loss:.4f}")
        
        print(f"\nTotal Examples Evaluated: {len(losses)}")
        
        # Save results if output dir specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save metrics
            metrics = {
                'mean_loss': float(mean_loss),
                'std_loss': float(std_loss),
                'min_loss': float(min_loss),
                'max_loss': float(max_loss),
                'num_examples': len(losses),
            }
            
            with open(output_dir / "metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Save sample completions
            with open(output_dir / "sample_completions.json", 'w') as f:
                json.dump(completions, f, indent=2)
            
            logger.info(f"Results saved to {output_dir}")
        
        return {
            'losses': losses,
            'metrics': metrics,
            'completions': completions,
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Qwen3.5-9B on combined_80_20_split")
    parser.add_argument('--model-path', type=str, required=True, help='Path to fine-tuned model')
    parser.add_argument('--data-root', type=str, default='.', help='Data root directory')
    parser.add_argument('--output-dir', type=str, help='Output directory for results')
    
    args = parser.parse_args()
    
    evaluator = QwenEvaluator(
        model_path=args.model_path,
        data_root=args.data_root,
    )
    
    evaluator.evaluate(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
