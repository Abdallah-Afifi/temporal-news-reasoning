"""Perturbation-consistency probe across models (NOT a unit test).

Asks each local model a base question and a minimally perturbed variant whose
correct answer changes, and reports base accuracy, consistency (both right)
and a per-category error breakdown.

Moved out of ``tests/`` on 2026-09-12: it defines no ``test_*`` function, so
pytest collected 0 tests from it while still importing torch/peft at
collection time. It needs a GPU and local model weights; run it directly:

    ./venv/bin/python scripts/probe_perturbation_consistency.py
"""

import sys
import time
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.inference import SLMInference

ENABLE_4BIT = os.getenv("ENABLE_4BIT", "1") == "1" # Default to 4-bit to fit easily into memory

# ==============================================================================
# 1. QA Databanks
# ==============================================================================

EXPLICIT_BANK = [
    {
        "id": "exp_1_duration",
        "category": "duration",
        "context": "The meeting started on January 5, 2023, and ended on January 9, 2023.",
        "base": {
            "question": "How many days did the meeting last?",
            "expected_regex": r"\b(4|four)\b"
        },
        "perturbed": {
            "context": "The meeting started on January 5, 2023, and ended on January 10, 2023.",
            "question": "How many days did the meeting last?",
            "expected_regex": r"\b(5|five)\b"
        }
    },
    {
        "id": "exp_2_order",
        "category": "order",
        "context": "The package arrived at 10:00 AM on Monday. The letter was delivered at 9:00 AM on the same day.",
        "base": {
            "question": "Which item was delivered first, the package or the letter?",
            "expected_regex": r"(?i)letter"
        },
        "perturbed": {
            "context": "The package arrived at 8:00 AM on Monday. The letter was delivered at 9:00 AM on the same day.",
            "question": "Which item was delivered first, the package or the letter?",
            "expected_regex": r"(?i)package"
        }
    },
    {
        "id": "exp_3_boundary",
        "category": "boundary",
        "context": "John was employed from 2010 to 2015. He was unemployed until 2018.",
        "base": {
            "question": "Was John employed in 2016?",
            "expected_regex": r"(?i)no\b"
        },
        "perturbed": {
            "context": "John was employed from 2010 to 2017. He was unemployed until 2018.",
            "question": "Was John employed in 2016?",
            "expected_regex": r"(?i)yes\b"
        }
    },
    {
        "id": "exp_4_arithmetic",
        "category": "arithmetic",
        "context": "The project took exactly 14 days to complete. It was finished on March 20.",
        "base": {
            "question": "On what date did the project start? Provide just the month and day.",
            "expected_regex": r"(?i)march\s*6"
        },
        "perturbed": {
            "context": "The project took exactly 10 days to complete. It was finished on March 20.",
            "question": "On what date did the project start? Provide just the month and day.",
            "expected_regex": r"(?i)march\s*10"
        }
    }
]

IMPLICIT_BANK = [
    {
        "id": "imp_1_order",
        "category": "order",
        "context": "I got an email from Alice yesterday. Tomorrow, I will reply to her.",
        "base": {
            "question": "Did I reply to Alice before or after I received her email?",
            "expected_regex": r"(?i)after"
        },
        "perturbed": {
            "context": "I will get an email from Alice tomorrow. Yesterday, I sent her an email.",
            "question": "Did I send an email to Alice before or after receiving one from her?",
            "expected_regex": r"(?i)before"
        }
    },
    {
        "id": "imp_2_boundary",
        "category": "boundary",
        "context": "The festival is happening next weekend. Today is Wednesday.",
        "base": {
            "question": "Is the festival happening today?",
            "expected_regex": r"(?i)no\b"
        },
        "perturbed": {
            "context": "The festival is happening this week. Today is Thursday.",
            "question": "Is it possible the festival is happening today?",
            "expected_regex": r"(?i)(yes|possible)"
        }
    },
    {
        "id": "imp_3_duration",
        "category": "duration",
        "context": "She left her house earlier this morning and returned late at night.",
        "base": {
            "question": "Did she spend more than a few hours away from home?",
            "expected_regex": r"(?i)yes\b"
        },
        "perturbed": {
            "context": "She left her house earlier this morning and returned shortly after lunch.",
            "question": "Did she spend more than 12 hours away from home?",
            "expected_regex": r"(?i)no\b"
        }
    }
]

COMPLEX_BANK = [
    {
        "id": "cmp_1_order",
        "category": "order",
        "context": "Event A happened before Event B. Event C happened after Event B.",
        "base": {
            "question": "Did Event A happen before Event C?",
            "expected_regex": r"(?i)yes\b"
        },
        "perturbed": {
            "context": "Event A happened after Event B. Event B happened after Event C.",
            "question": "Did Event A happen before Event C?",
            "expected_regex": r"(?i)no\b"
        }
    },
    {
        "id": "cmp_2_arithmetic",
        "category": "arithmetic",
        "context": "The first phase took 2 months. The second phase took twice as long as the first. The third phase took half as long as the first.",
        "base": {
            "question": "How many months did all three phases take in total?",
            "expected_regex": r"\b(7|seven)\b"
        },
        "perturbed": {
            "context": "The first phase took 4 months. The second phase took twice as long as the first. The third phase took half as long as the first.",
            "question": "How many months did all three phases take in total?",
            "expected_regex": r"\b(14|fourteen)\b"
        }
    },
    {
        "id": "cmp_3_duration",
        "category": "duration",
        "context": "Alice is 5 years older than Bob, who was born 10 years after Charlie. Charlie is currently 30 years old.",
        "base": {
            "question": "How old is Alice?",
            "expected_regex": r"\b(25|twenty[- ]five)\b"
        },
        "perturbed": {
            "context": "Alice is 3 years younger than Bob, who was born 5 years after Charlie. Charlie is currently 30 years old.",
            "question": "How old is Alice?",
            "expected_regex": r"\b(22|twenty[- ]two)\b"
        }
    }
]

MODELS = [
    {"label": "Mistral-7B", "key": "mistral", "path": "models/Mistral-7B-Instruct-v0.3"},
    {"label": "LLaMA-3.2-3B", "key": "llama", "path": "models/Llama-3.2-3B-Instruct"},
    {"label": "Qwen2.5-3B", "key": "qwen2.5-3b", "path": "models/Qwen2.5-3B-Instruct"},
]

# ==============================================================================
# 2. Evaluation Logic
# ==============================================================================

def format_prompt(context: str, question: str) -> str:
    if context:
        return f"Context: {context}\nQuestion: {question}\nAnswer clearly and concisely."
    return f"Question: {question}\nAnswer clearly and concisely."

def evaluate_models_on_bank(bank_name: str, bank_data: list):
    print(f"\n{'='*50}")
    print(f"  Evaluating {bank_name.upper()} Reasoning Bank")
    print(f"{'='*50}\n")
    
    results = {
        model["label"]: {
            "base_correct": 0,
            "perturb_correct": 0,
            "consistent": 0, 
            "total": len(bank_data),
            "total_time": 0.0,
            "errors": {"order": 0, "duration": 0, "boundary": 0, "arithmetic": 0}
        }
        for model in MODELS
    }
    
    for model_info in MODELS:
        label = model_info["label"]
        model_dir = PROJECT_ROOT / model_info["path"]
        
        if not model_dir.exists():
            print(f"[!] Skipping {label} - Model path not found: {model_dir}")
            continue
            
        print(f"--- Loading {label} ---")
        try:
            model = SLMInference(
                model_info["key"], 
                model_dir=str(model_dir), 
                load_in_4bit=ENABLE_4BIT
            )
        except Exception as e:
            print(f"Failed to load {label}: {e}")
            continue
            
        for item in bank_data:
            cat = item["category"]
            
            # --- Test BASE scenario ---
            p_base = format_prompt(item["context"], item["base"]["question"])
            t0 = time.perf_counter()
            resp_base = model.generate(p_base, max_new_tokens=64, temperature=0.0, do_sample=False)
            time_base = time.perf_counter() - t0
            results[label]["total_time"] += time_base
            
            base_match = bool(re.search(item["base"]["expected_regex"], resp_base))
            if base_match:
                results[label]["base_correct"] += 1
            else:
                results[label]["errors"][cat] += 1
                
            # --- Test PERTURBED scenario ---
            p_pert = format_prompt(item["perturbed"]["context"], item["perturbed"]["question"])
            t0 = time.perf_counter()
            resp_pert = model.generate(p_pert, max_new_tokens=64, temperature=0.0, do_sample=False)
            time_pert = time.perf_counter() - t0
            results[label]["total_time"] += time_pert
            
            pert_match = bool(re.search(item["perturbed"]["expected_regex"], resp_pert))
            if pert_match:
                results[label]["perturb_correct"] += 1
                
            # Record an error if base was right but perturbation failed (model logic is brittle/inconsistent)
            if base_match and not pert_match:
                results[label]["errors"][cat] += 1
                
            # Consistency requires BOTH to be logically correct
            if base_match and pert_match:
                results[label]["consistent"] += 1
                
            print(f"[{item['id']}] Base: {'PASS' if base_match else 'FAIL'} ({time_base:.2f}s) | Perturb: {'PASS' if pert_match else 'FAIL'} ({time_pert:.2f}s)")
            
            # Print failure logs to help debug
            if not base_match:
                print(f"  -> Base failed. Expected regex: {item['base']['expected_regex']}, Got: '{resp_base}'")
            if not pert_match:
                print(f"  -> Perturb failed. Expected regex: {item['perturbed']['expected_regex']}, Got: '{resp_pert}'")
            
        # Unload model to save memory before loading the next one
        del model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    # ==============================================================================
    # 3. Output Generation
    # ==============================================================================
    report_path = PROJECT_ROOT / "results" / "analysis" / f"{bank_name}_reasoning_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write(f"# {bank_name.capitalize()} Reasoning Evaluation Report\n\n")
        f.write("This report evaluates SLMs on generalization over time, temporal consistency, and detailed error breakdown (order, duration, boundary, arithmetic) using baseline and perturbed inference testing.\n\n")
        
        f.write("## 1. Top-Level Metrics\n\n")
        f.write("| Model | Base Accuracy | Temporal Consistency | Avg Latency/Query |\n")
        f.write("|-------|---------------|----------------------|-------------------|\n")
        
        for label, st in results.items():
            tot = st["total"]
            base_acc = (st["base_correct"] / tot) * 100 if tot > 0 else 0
            consist = (st["consistent"] / tot) * 100 if tot > 0 else 0
            avg_lat = st["total_time"] / (tot * 2) if tot > 0 else 0
            f.write(f"| {label} | {base_acc:.1f}% | {consist:.1f}% | {avg_lat:.2f}s |\n")
            
        f.write("\n## 2. Categorized Error Breakdown\n\n")
        f.write("Total errors observed across both baseline and perturbed prompts by reasoning category.\n\n")
        f.write("| Model | Order Errors | Duration Errors | Boundary Errors | Arithmetic Errors |\n")
        f.write("|-------|--------------|-----------------|-----------------|-------------------|\n")
        
        for label, st in results.items():
            errs = st["errors"]
            f.write(f"| {label} | {errs['order']} | {errs['duration']} | {errs['boundary']} | {errs['arithmetic']} |\n")
            
    print(f"\n[+] Report saved to {report_path}\n")


if __name__ == "__main__":
    evaluate_models_on_bank("explicit", EXPLICIT_BANK)
    evaluate_models_on_bank("implicit", IMPLICIT_BANK)
    evaluate_models_on_bank("complex", COMPLEX_BANK)
