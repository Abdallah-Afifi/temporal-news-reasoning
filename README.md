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
fixed and **all ten arms were re-run on 2026-09-13**: zero-shot scores 46.93%
(not 35.20%) and no fine-tuned arm beats it, by −3.0 to −8.2pp. The campaign's value is the negative
result and its attribution: fine-tuning on Wikipedia-style temporal QA moves
only the categories the synthetic slices directly target, and costs broad
temporal ability everywhere else.

**2026-09-23 audit — that reading needs a correction.** Every fine-tuned arm
was trained on a different prompt from the one it was evaluated on (custom
system prompt + bare Context/Question at training; the zero-shot instruction
prompt with a separate Choices block and an NLI Premise/Hypothesis layout at
evaluation), so the deltas above measure fine-tuning *plus a prompt switch*.
The audit also found benchmark contamination through shared TimeQA/TimeBench
passages and a wall-clock date stamped into every prompt. Details:
[`docs/audit_2026_09_23.md`](docs/audit_2026_09_23.md). The fixed arm, **v11**
(train/eval prompt parity, decontaminated data, hyperparameters selected on a
held-out dev split and reported on test-minus-dev only), is specified in
[`docs/hpo_v11_protocol.md`](docs/hpo_v11_protocol.md) and runs with
`scripts/run_schedule_v11.sh`. **Result (test-minus-dev, same
prompt/engine/date as zero-shot, three seeds): v11 beats zero-shot on all
three benchmarks, with every seed** — mean TIME +5.27pp (±0.43), TimeBench
+1.48pp (±0.39), TRAM +6.54pp (±0.68); seed 42 McNemar z = +39 / +4.2 /
+139. The first arm in the campaign to do so, and in every TIME retrieval
setting. It still regresses on some categories (TimeBench temporal_dialogue
−18, TRAM storytelling −11), addressed in v12. Details:
`docs/results_and_methodology.md` §10.

**v12 (2026-09-28)** tested three single-variable fixes against v11-best
(2 epochs; corrected storytelling/relation/ordering/temporal_dialogue/
duration training cards; +4096 context). All three still beat zero-shot
on all three benchmarks. Measured against v11's three-seed mean (not the
lucky seed 42), **2 epochs is a clean win** (TimeBench +2.97pp, TRAM
+2.29pp, TIME neutral); the card fixes and longer context give only
marginal gains. Details: `docs/results_and_methodology.md` §11 and
`docs/audit_2026_10_04.md` §5.1.

**Mistral-7B-Instruct-v0.3 (2026-10-02)** — the same audit fixes and
protocol, ported to a second base model. HPO-lite search (re-run from
scratch rather than reusing LLaMA's hyperparameters) picked a learning
rate ~5x lower than LLaMA's winner, confirming that choice. **Beats
zero-shot on all three benchmarks with every seed** (42/43/44, mean ± sd):
TIME +8.53pp (±0.40), TimeBench +10.41pp (±0.40), TRAM +4.61pp (±0.35),
every seed McNemar z > 28. This is a model-internal result only —
Mistral's and LLaMA's zero-shot baselines differ in both directions
across benchmarks, so no LLaMA-vs-Mistral ranking is licensed. Details:
`docs/results_and_methodology.md` §12.

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

- **No like-for-like large-model comparison exists.** GPT-4/GPT-3.5/70B were
  never run under this project's protocol. The benchmark papers' own published
  GPT numbers are set beside v11 in `docs/results_and_methodology.md` §10.4,
  with the reasons they are not comparable (different prompts, samples and
  metrics): on TRAM and TimeBench v11 is ~20–28pp below GPT-4. So the project
  does not claim to "approach large model performance at a fraction of the
  cost".
- **Mostly one model, one exception.** Every number through v12 is
  LLaMA-3.2-3B-Instruct, greedy decoding. Arms v1–v10 are single runs
  (seed 42, no confidence intervals); v11 is three seeds (42/43/44), sd
  0.4–0.7pp; v12 is a single seed per arm, borrowing v11's noise floor
  (§11.3). Mistral-7B-Instruct-v0.3 **is** under the frozen protocol as of
  2026-10-02 (`docs/results_and_methodology.md` §12) — three seeds, sd
  0.35–0.40pp, not directly comparable to LLaMA's numbers (different zero-shot
  baselines), only comparable to its own zero-shot. Qwen artifacts in
  `results/` still predate the frozen protocol and are not comparable.

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
