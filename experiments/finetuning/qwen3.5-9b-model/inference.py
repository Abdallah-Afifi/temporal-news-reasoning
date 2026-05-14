#!/usr/bin/env python3
"""
Inference script for Qwen3.5-9B fine-tuned model
"""

import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path, lora_path=None):
    """Load base model and optionally LoRA."""
    logger.info(f"Loading model from {model_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    if lora_path:
        logger.info(f"Loading LoRA from {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_tokens=100):
    """Generate response for a prompt."""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True,
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def main():
    parser = argparse.ArgumentParser(description="Run inference with Qwen3.5-9B model")
    parser.add_argument('--model-path', type=str, default='checkpoints/qwen3.5-9b-model')
    parser.add_argument('--lora-path', type=str, default=None)
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--max-tokens', type=int, default=100)
    
    args = parser.parse_args()
    
    logger.info("Loading model...")
    model, tokenizer = load_model(args.model_path, args.lora_path)
    
    if args.interactive:
        logger.info("Starting interactive inference mode. Type 'quit' to exit.")
        while True:
            try:
                prompt = input("\nPrompt: ").strip()
                if prompt.lower() == 'quit':
                    break
                
                response = generate_response(model, tokenizer, prompt, args.max_tokens)
                print(f"Response: {response}\n")
                
            except KeyboardInterrupt:
                break
    
    elif args.prompt:
        logger.info(f"Prompt: {args.prompt}")
        response = generate_response(model, tokenizer, args.prompt, args.max_tokens)
        print(f"Response: {response}")
    
    else:
        # Default example
        example_prompt = """Question: When was the Berlin Wall built?

Options:
A) 1961
B) 1975
C) 1989
D) 1995

Answer: """
        
        logger.info("Running example inference...")
        logger.info(f"Prompt: {example_prompt}")
        response = generate_response(model, tokenizer, example_prompt)
        print(f"Response: {response}")


if __name__ == "__main__":
    main()
