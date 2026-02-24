# RAG System Design

## Architecture

The temporal-aware RAG system operates in three stages:

1. **Dense Retrieval** — Sentence-BERT encodes queries and corpus, FAISS finds top-k similar documents
2. **Temporal Filtering** — Filters documents based on temporal constraints extracted from the query
3. **Temporal Re-ranking** — Re-ranks remaining documents using a weighted combination of semantic + temporal scores

## Component Details

_To be expanded as implementation progresses._
