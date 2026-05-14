# Scripts

This folder contains utility and setup scripts used by the project.

Common scripts
- `download_models.py` — download pretrained model files referenced by pipelines.
- `download_datasets.py` — download benchmark datasets (TIME, TimeBench, TRAM).
- `install_heideltime.sh` — install TreeTagger + py_heideltime integration.
- `run_baselines.py` — run zero-shot baseline evaluations.
- `run_full_finetuned_eval.sh` — orchestrate full evaluation over finetuned adapters.

Usage
Run scripts from the repository root, for example:

```bash
python scripts/download_datasets.py time
bash scripts/install_heideltime.sh
python scripts/run_baselines.py --model qwen --benchmark time
```

Notes
- Many scripts assume data/models are under `data/` and `models/` respectively.
- Avoid committing large outputs produced by scripts; results and checkpoints are gitignored.
