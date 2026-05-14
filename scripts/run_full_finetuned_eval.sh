#!/bin/bash
# run_full_finetuned_eval.sh

# Activate the python virtual environment
source venv/bin/activate

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
