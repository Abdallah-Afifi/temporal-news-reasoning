`src/data/` — data layout and utilities

Purpose
- Load benchmarks (TIME, TIMEBENCH, TRAM)
- Provide dataset interfaces for training and evaluation
- Utilities for generating synthetic training data

Key files
- `src/data/data_loader.py`: `BenchmarkLoader` used by evaluation and baselines
- `src/data/corpus_processor.py`: corpus ingestion helpers
- `src/data/synthetic_templates.py`: synthetic dataset templates and generator helpers

Data layout (repo root)
- `data/benchmarks/` : TIME, TimeBench, TRAM
- `data/corpus/`     : news corpora (gitignored)
- `data/training/`   : fine-tuning / curriculum files

Useful scripts
```bash
# download benchmark datasets
python scripts/download_datasets.py time timebench tram

# generate synthetic training data
python scripts/generate_synthetic_dataset.py --out data/training/synthetic.jsonl
```

Guidance
- Keep large corpora and embeddings out of git; use `data/` entries managed by scripts.
