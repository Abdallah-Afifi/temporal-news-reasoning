# Efficient Temporal Reasoning for News Understanding

> Does specialized LoRA fine-tuning on out-of-domain temporal QA transfer to
> temporal reasoning over real-world news in a 3B-parameter model?

## Overview

This repository holds two largely independent bodies of work. **Read this
section before any other: the repository is bigger than the evaluated
system.**

**1. The evaluated chain (what every published number comes from).** A
LoRA fine-tuning campaign on **LLaMA-3.2-3B-Instruct only**, eleven arms
(v1 → v6d), evaluated on TIME / TimeBench / TRAM under a frozen scoring
protocol. The single reference is
[`docs/results_and_methodology.md`](docs/results_and_methodology.md).

Headline finding, stated honestly: **no fine-tuned arm beats the zero-shot
baseline on TIME or TimeBench** once a TIME benchmark artifact is excluded
(2,427 items where a "There is no answer" option is always the gold — see §1.1
of the methodology document). On TRAM, an audit on 2026-09-12 found that 85.3% of the
column had been generated from prompts that omitted the question (the NLI
hypothesis and the storytelling passage never reached the model). The loader is
fixed and **all ten arms were re-run on 2026-09-13**: zero-shot scores 46.95%
(not 35.20%) and no fine-tuned arm beats it, by −3.0 to −8.2pp. The campaign's value is the negative
result and its attribution: fine-tuning on Wikipedia-style temporal QA moves
only the categories the synthetic slices directly target, and costs broad
temporal ability everywhere else.

**2. The RAG system** (`temporal_rag/`, `src/rag/`, `src/temporal/`,
`src/pipeline/`) — FAISS indexing, temporal filtering, a Neo4j temporal
knowledge graph, GLiNER entity extraction, and a Gradio UI. It is a
substantial, separately developed component and is **not part of the
evaluated chain**: no number in the results tables passes through it.

### Status of the components

| Component | Status |
|---|---|
| LoRA fine-tuning + evaluation harness (`experiments/`, `scripts/`, `src/{data,evaluation,models}`) | **Evaluated.** Produces every published number |
| Temporal-aware RAG (`temporal_rag/`) | Built and runnable; **not evaluated**, no tests, needs Postgres + Neo4j + a JVM for HeidelTime |
| `src/rag/` retrieval + timeline construction | Prototype; zero non-test importers |
| `src/prompting/` (timeline CoT, self-consistency) | **Not implemented** — the methods `raise NotImplementedError` |
| `src/training/` | **Do not use** — a divergent second trainer; the real one is `experiments/finetuning/LLaMA/train.py` |

### Not claimed

- **No large-model comparison exists in this repository.** There is no GPT-4 /
  Claude / 70B baseline and no cost or latency comparison, so the project does
  not and cannot claim to "approach large model performance at a fraction of
  the cost".
- **One model, one seed.** Every corrected number is LLaMA-3.2-3B-Instruct
  with seed 42, single run, greedy decoding, no confidence intervals. Qwen and
  Mistral artifacts exist in `results/` but predate the frozen protocol and
  are not comparable.

### Key Components

| Module | Description | State |
|--------|-------------|-------|
| **LoRA Fine-Tuned SLMs** | Parameter-efficient fine-tuning on temporal reasoning tasks | **Evaluated** |
| **Evaluation harness** | Frozen scoring protocol, HF + vLLM engines, CPU rescoring from immutable predictions | **Evaluated** |
| **Temporal-Aware RAG** | Dense retrieval (Sentence-BERT) + temporal filtering + re-ranking | Built, not evaluated |
| **Temporal Intent Detection** | Classifies queries as recency/past/future/atemporal | Prototype |
| **Timeline Construction** | Builds chronological event timelines from retrieved documents | Prototype |
| **Temporal Prompting** | Timeline-based Chain-of-Thought + temporal context injection | **Not implemented** (stubs) |
| **Consistency Checking** | Cross-references temporal facts and detects contradictions | **Not implemented** (stub) |

### RAG System Components (`temporal_rag/`, not in the evaluated chain)

The `temporal_rag/` directory contains a complete but **unevaluated and
untested** RAG system featuring:

- **Indexing**: FAISS-based semantic indexing with multiple chunking strategies (recursive & character-based)
- **Embedding**: Advanced encoder supporting various embedding models and preprocessing
- **Retrieval**: Multi-stage retrieval with semantic search and temporal filtering
- **Knowledge Graphs**: Neo4j-compatible temporal knowledge graph storage
- **Entity Extraction**: GLiNER-based named entity recognition for temporal relations
- **Coreference Resolution**: Entity linking and pronoun resolution
- **Interactive Query Engine**: Full-stack query processing pipeline
- **Web UI**: Gradio-based interface for interactive RAG exploration

## Repository Structure

```
temporal-news-reasoning/
├── configs/              # Model & pipeline configuration files
├── data/                 # Datasets & benchmarks (partially gitignored)
│   ├── benchmarks/       # TIME, TIMEBENCH, TRAM benchmarks
│   ├── corpus/           # News articles (gitignored)
│   └── training/         # Fine-tuning data
├── src/                  # Source code
│   ├── data/             # Data loading & processing
│   ├── temporal/         # Temporal intent detection & extraction
│   ├── rag/              # Retrieval-Augmented Generation pipeline
│   ├── models/           # Model inference
│   ├── training/         # LoRA fine-tuning
│   ├── prompting/        # Prompt templates & strategies
│   ├── evaluation/       # Metrics & evaluation harness
│   ├── pipeline/         # End-to-end pipeline
│   └── demo/             # Gradio web demo
├── temporal_rag/         # Comprehensive RAG system (merged from Retrieval-augmented-generation branch)
│   ├── bbc_to_json.py    # BBC dataset processing
│   ├── build_faiss_database.py  # FAISS index builder
│   ├── embed.py          # Embedding pipeline
│   ├── encoder.py        # Advanced encoder with chunking
│   ├── chunk_splitter.py # Text chunking strategies
│   ├── recursive_splitter.py  # Recursive document splitting
│   ├── temporal_ie.py    # Temporal information extraction
│   ├── kg_store.py       # Knowledge graph storage
│   ├── sql_store.py      # SQL database backend
│   ├── interactive_query.py    # Interactive query interface
│   ├── query_faiss.py    # FAISS query engine
│   ├── query_full_stack.py     # Full RAG pipeline
│   ├── temporal_rag_gui.py     # Web UI for RAG system
│   └── utils/            # Utility modules (HeidelTime wrapper, etc.)
├── scripts/              # Utility & setup scripts
├── notebooks/            # Jupyter notebooks for exploration
├── experiments/          # Experimental training/eval pipelines
│   └── finetuning/        # LoRA finetuning runs and utilities
├── results/              # Experiment results
├── checkpoints/          # Model checkpoints (gitignored)
├── docs/                 # Documentation & meeting notes
└── tests/                # Unit tests
```

## Quick Start

### 1. Environment Setup

```bash
git clone https://github.com/Abdallah-Afifi/temporal-news-reasoning.git
cd temporal-news-reasoning

# Python 3.12 (the version the published numbers were produced with)
python3.12 -m venv venv
venv/bin/python -m pip install -U pip
venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128

# For a fresh install:
venv/bin/python -m pip install -r requirements.txt
# To REPRODUCE the published numbers, use the exact environment instead:
venv/bin/python -m pip install -r requirements.lock.venv.txt
```

`requirements.txt` pins only 2 of 47 dependencies, so it will not reproduce
the evaluated environment — `requirements.lock.venv.txt` (and
`requirements.lock.venv_vllm.txt` for the vLLM engine) will.

### 2. Download Models & Data

```bash
python scripts/download_models.py
python scripts/download_datasets.py
bash scripts/install_heideltime.sh
```

### 3. Verify Installation

```bash
python scripts/verify_install.py
```

### 4. Run an evaluation

```bash
# HF engine (the path used for TIME/TimeBench on arms v1-v6)
./venv/bin/python scripts/run_baselines.py --model llama --benchmark timebench \
    --results-dir ./results/baseline/smoke --batch-size 32 \
    --token-budget 57344 --max-samples 5

# Re-score every arm from the stored predictions (CPU only, ~8 min).
# --tram-root IS REQUIRED for a quotable TRAM column: without it the scorer
# reads the pre-2026-09-12 predictions, whose prompts omitted the NLI
# hypothesis and the storytelling passage for 85.3% of items. It prints a
# STALE banner in that case — do not quote a run that shows one.
./venv/bin/python scripts/rescore_v5_protocol.py --tram-root results/tram_fixed
```

The protocol is frozen. Read `docs/results_and_methodology.md` §5 before
running anything that will be quoted — in particular, a fine-tuned arm must be
evaluated from a **merged** checkpoint on vLLM (`scripts/merge_lora.py`);
`--adapter-dir` alone selects the training system prompt and loads no LoRA
weights.

## Benchmarks

| Benchmark | Tasks | Focus |
|-----------|-------|-------|
| **TIME** | News temporal QA, timeline construction | Real-world news temporal reasoning |
| **TIMEBENCH** | 16 temporal reasoning subtasks | Comprehensive temporal understanding |
| **TRAM** | Temporal reasoning across modalities | Multi-modal temporal reasoning |

## Models

| Model | Parameters | Notes |
|-------|-----------|-------|
| LLaMA-3.2-3B-Instruct | 3B | **The only model in the evaluated campaign** — every published number |
| Mistral-7B-Instruct-v0.3 | 7B | Side experiment; scored under a superseded protocol, **not comparable**, and banned from vLLM (parity failure) |
| Qwen3.5-9B | 9B | Side experiment; not in the evaluated chain |
| all-MiniLM-L6-v2 | 22M | Embedding model for the (unevaluated) RAG system |

Qwen2.5-3B and Phi-3-mini appear in early planning documents but were never
run under the frozen protocol; they are not present in `models/`.

## Branch Strategy

| Branch | Purpose | Status |
|--------|---------|--------|
| `main` | Stable, tested code with integrated RAG system | ✅ Active |
| `develop` | Integration branch for feature merging | Active |
| `feature/data-pipeline` | Data loading & preprocessing (Team A) | Feature |
| `feature/temporal-rag` | RAG system development (Team B) | Feature |
| `feature/model-finetuning` | LoRA fine-tuning & prompting (Team C) | Feature |
| `feature/evaluation` | Evaluation harness & metrics | Feature |
| `Retrieval-augmented-generation` | ⭐ Complete RAG system (now merged to main) | ✅ Merged |
| `experiment/*` | Individual experiment branches | Experimental |
| `docs/*` | Documentation updates | Documentation |

**Recent Changes:** The `Retrieval-augmented-generation` branch has been successfully merged into `main` (commit 176bb44), bringing the complete RAG system including FAISS indexing, temporal filtering, knowledge graphs, and web UI into the main production branch.


## Team

- **Supervisor:** Dr. Nouri Sakr, Dr. Alia El Bolock
- **Sponsor:** Microsoft (Dr. Ahmed Tawfik)

## License

This project is for academic research purposes.
