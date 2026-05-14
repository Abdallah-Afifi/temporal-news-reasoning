# Evaluation Plan

## Benchmarks
- **TIME**: News-specific temporal QA, timeline construction, temporal NLI
- **TIMEBENCH**: 16 subtasks covering temporal expression processing, commonsense, event relations, forecasting, QA
- **TRAM**: Cross-modal temporal reasoning

## Baselines
- Zero-shot (all 3 SLMs)
- Few-shot (1, 3, 5 examples)
- GPT-4 upper bound (sampled subset)

## Ablation Components
1. Base model only
2. + LoRA fine-tuning
3. + Temporal RAG
4. + Temporal prompting
5. All combinations
6. Full system

## Metrics
- Accuracy, F1, Exact Match
- Temporal F1 (temporal expression extraction)
- Timeline ordering accuracy (Kendall's tau)

_To be expanded as evaluation implementation progresses._
