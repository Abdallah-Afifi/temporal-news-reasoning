#!/bin/bash
# run_full_finetuned_eval.sh
#
# Evaluates the fine-tuned LLaMA LoRA adapter on TIME / TimeBench.
# Requires benchmark data:  python scripts/download_datasets.py time timebench
#
# NOTE: with --adapter-dir set, the runner now injects the TRAINING system
# prompt (matching the fine-tuned chat format), and reports
# prompt_type="training_format_chat" in the metadata.

set -e

# Activate a virtual environment if one exists (conda users: activate manually)
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
fi

echo "================================================================="
echo "Starting Fine-Tuned Evaluation for LLaMA-3.2-3B on TIME Benchmark"
echo "================================================================="

python scripts/run_baselines.py \
  --model llama \
  --benchmark time \
  --results-dir ./results/finetuned \
  --adapter-dir ./checkpoints/llama/final \
  --batch-size 16

echo ""
echo "================================================================="
echo "Starting Fine-Tuned Evaluation for LLaMA-3.2-3B on TimeBench Benchmark"
echo "================================================================="

python scripts/run_baselines.py \
  --model llama \
  --benchmark timebench \
  --results-dir ./results/finetuned \
  --adapter-dir ./checkpoints/llama/final \
  --batch-size 16

echo ""
echo "================================================================="
echo "Evaluation Complete!"
echo "================================================================="
