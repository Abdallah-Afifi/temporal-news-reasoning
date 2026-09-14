# Evaluation Report: llama on timebench/finetuned

Generated: 2026-09-05T21:47:06.820291
Num examples: 21188

## Overall Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | 0.4580 |
| Overall F1 | 0.2409 |
| Exact Match | 0.4580 |
| Temporal F1 | 0.9722 |

## Results by Temporal Reasoning Type

| Category | Accuracy | Correct | Total |
|----------|----------|---------|-------|
| arithmetic | 0.1588 | 635 | 4000 |
| duration | 0.7744 | 1191 | 1538 |
| situated_generation | 0.0000 | 0 | 115 |
| temporal_dialogue | 0.5906 | 854 | 1446 |
| temporal_nli | 0.4656 | 3243 | 6965 |
| temporal_ordering | 0.5447 | 2314 | 4248 |
| temporal_qa | 0.5570 | 557 | 1000 |
| temporal_reasoning | 0.4851 | 910 | 1876 |

## Error Analysis

| Error Type | Count |
|------------|-------|
| hallucination | 52 |
| other | 10935 |
| temporal_extraction | 497 |
