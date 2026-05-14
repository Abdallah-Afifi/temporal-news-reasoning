import sys
import os
from pathlib import Path
import time
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reduce progress bars during model load
os.environ["TQDM_DISABLE"] = "1"

from src.models.inference import SLMInference
from tests.test_inference import EXPLICIT_BANK, IMPLICIT_BANK, COMPLEX_BANK, MODELS, format_prompt

def run():
    log_file = PROJECT_ROOT / "tests" / "full_inference_responses.log"
    
    with open(log_file, "w") as f:
        f.write("="*80 + "\n")
        f.write("FULL INFERENCE RESPONSES LOG\n")
        f.write("="*80 + "\n\n")

    banks = [
        ("EXPLICIT", EXPLICIT_BANK),
        ("IMPLICIT", IMPLICIT_BANK),
        ("COMPLEX", COMPLEX_BANK),
    ]

    for model_info in MODELS:
        label = model_info["label"]
        model_dir = PROJECT_ROOT / model_info["path"]
        
        with open(log_file, "a") as f:
            f.write(f"\n{'#'*80}\n")
            f.write(f"MODEL: {label}\n")
            f.write(f"{'#'*80}\n\n")

        if not model_dir.exists():
            print(f"Skipping {label}, path not found.")
            continue
            
        print(f"Loading {label}...")
        model = SLMInference(model_info["key"], model_dir=str(model_dir), load_in_4bit=True)
        
        for bank_name, bank_data in banks:
            with open(log_file, "a") as f:
                f.write(f"\n--- {bank_name} REASONING BANK ---\n\n")
                
            for item in bank_data:
                # Base
                p_base = format_prompt(item["context"], item["base"]["question"])
                resp_base = model.generate(p_base, max_new_tokens=64, temperature=0.0, do_sample=False)
                
                # Perturbed
                p_pert = format_prompt(item["perturbed"]["context"], item["perturbed"]["question"])
                resp_pert = model.generate(p_pert, max_new_tokens=64, temperature=0.0, do_sample=False)
                
                with open(log_file, "a") as f:
                    f.write(f"--- ID: {item['id']} (Category: {item['category']}) ---\n")
                    f.write(f"[BASE PROMPT]\n{p_base}\n\n")
                    f.write(f"[EXPECTED REGEX]: {item['base']['expected_regex']}\n")
                    f.write(f"[MODEL ANSWER]\n{resp_base}\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"[PERTURBED PROMPT]\n{p_pert}\n\n")
                    f.write(f"[EXPECTED REGEX]: {item['perturbed']['expected_regex']}\n")
                    f.write(f"[MODEL ANSWER]\n{resp_pert}\n")
                    f.write("=" * 80 + "\n\n")
                    
        # Clean up memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    print(f"Finished! Log written to {log_file}")

if __name__ == '__main__':
    run()
