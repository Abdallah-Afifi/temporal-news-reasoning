# RAG (Retrieval-Augmented Generation) utilities

This package contains RAG-related code used by the project:

- `faiss_index.py` — helpers for building/loading FAISS indices.
- `semantic_retriever.py` — embedding-based retrieval interface.
- `temporal_filter.py` — temporal filtering of retrieval results.
- `temporal_reranker.py` — re-ranking candidates with temporal signals.
- `timeline_builder.py` — assemble timelines from retrieved documents.
- `heideltime_wrapper.py` — wrapper for HeidelTime temporal tagger integration.

Usage
- The high-level pipeline uses `src/rag/semantic_retriever.py` to fetch candidates, then applies `temporal_filter` and `temporal_reranker` before passing context to the model.
- See `scripts/build_index.py` and `scripts/run_baselines.py` for example usages.

Notes
- Some components expect external binaries (HeidelTime / TreeTagger) to be installed; use `scripts/install_heideltime.sh` to set up the environment.
- The FAISS index files and large corpora are stored under `data/` and are gitignored.
