import sys
import time
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.models.inference import SLMInference

ENABLE_4BIT = os.getenv("ENABLE_4BIT", "0") == "1"

def run_model(label: str, model_key: str, model_dir: str, prompt: str) -> None:
	load_start = time.perf_counter()
	if ENABLE_4BIT:
		try:
			model = SLMInference(model_key, model_dir=model_dir, load_in_4bit=True)
			load_mode = "4bit"
		except Exception as error:
			print(f"[{label}] 4-bit load failed ({error}); falling back to full precision.")
			model = SLMInference(model_key, model_dir=model_dir)
			load_mode = "full"
	else:
		model = SLMInference(model_key, model_dir=model_dir)
		load_mode = "full"
	load_seconds = time.perf_counter() - load_start

	gen_start = time.perf_counter()
	response = model.generate(
		prompt,
		max_new_tokens=96,
		temperature=0.0,
		do_sample=False,
	)
	gen_seconds = time.perf_counter() - gen_start

	print(f"{label} ({load_mode}) load: {load_seconds:.2f}s, generate: {gen_seconds:.2f}s")
	print(f"{label}: {response}\n")


# Mistral-7B-Instruct
run_model(
	"Mistral",
	"mistral",
	str(PROJECT_ROOT / "models" / "Mistral-7B-Instruct-v0.3"),
	"What is the capital of France?",
)

# LLaMA-3.2-3B-Instruct
run_model(
	"LLaMA",
	"llama",
	str(PROJECT_ROOT / "models" / "Llama-3.2-3B-Instruct"),
	"When did World War II end?",
)

# Qwen2.5-3B-Instruct
run_model(
	"Qwen2.5-3B",
	"qwen2.5-3b",
	str(PROJECT_ROOT / "models" / "Qwen2.5-3B-Instruct"),
	"Who is the president of the United States?",
)
