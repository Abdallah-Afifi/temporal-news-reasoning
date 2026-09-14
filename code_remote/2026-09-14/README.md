# Efficient Temporal Reasoning for News Understanding

> Can small language models (3B parameters) achieve competitive temporal reasoning performance on real-world news understanding through specialized fine-tuning and temporal-aware retrieval?

## Overview

This project investigates whether combining **LoRA fine-tuning**, **temporal-aware RAG**, and **specialized prompting** can enable small language models (Qwen2.5-3B, Phi-3-mini, LLaMA-3.2-3B) to perform temporal reasoning on news articles — approaching large model performance at a fraction of the cost.

**Latest Update:** The comprehensive Retrieval-Augmented Generation (RAG) system from the `Retrieval-augmented-generation` branch has been successfully merged into main, bringing production-ready RAG capabilities including FAISS semantic search, temporal filtering, knowledge graph support, and an interactive web UI.

### Key Components

| Module | Description |
|--------|-------------|
| **Temporal Intent Detection** | Classifies queries as recency/past/future/atemporal and extracts temporal constraints |
| **Temporal-Aware RAG** | Dense retrieval (Sentence-BERT) + temporal filtering + re-ranking |
| **Timeline Construction** | Builds chronological event timelines from retrieved documents |
| **LoRA Fine-Tuned SLMs** | Parameter-efficient fine-tuning on temporal reasoning tasks |
| **Temporal Prompting** | Timeline-based Chain-of-Thought + temporal context injection |
| **Consistency Checking** | Cross-references temporal facts and detects contradictions |

### ✅ RAG System Components (Now in Main)

The `temporal_rag/` directory contains a complete, production-ready RAG system featuring:

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
# Clone the repository
git clone https://github.com/Abdallah-Afifi/temporal-news-reasoning.git
cd temporal-news-reasoning

# Create conda environment
conda create -n temporal python=3.10 -y
conda activate temporal

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys
```

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

### 4. Run Baselines

```bash
python scripts/run_baselines.py --model qwen --benchmark timebench
```

## Benchmarks

| Benchmark | Tasks | Focus |
|-----------|-------|-------|
| **TIME** | News temporal QA, timeline construction | Real-world news temporal reasoning |
| **TIMEBENCH** | 16 temporal reasoning subtasks | Comprehensive temporal understanding |
| **TRAM** | Temporal reasoning across modalities | Multi-modal temporal reasoning |

## Models

| Model | Parameters | Notes |
|-------|-----------|-------|
| Qwen2.5-3B-Instruct | 3B | Primary model |
| Phi-3-mini-4k-instruct | 3.8B | Secondary model |
| LLaMA-3.2-3B-Instruct | 3B | Tertiary model |
| all-MiniLM-L6-v2 | 22M | Embedding model for RAG |

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
