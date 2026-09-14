# Evaluation Report: llama on timebench/finetuned

Generated: 2026-09-07T18:58:34.128699
Num examples: 21188

## Overall Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | 0.4393 |
| Overall F1 | 0.2035 |
| Exact Match | 0.4393 |
| Temporal F1 | 0.9632 |

## Results by Temporal Reasoning Type

| Category | Accuracy | Correct | Total |
|----------|----------|---------|-------|
| arithmetic | 0.0968 | 387 | 4000 |
| duration | 0.7250 | 1115 | 1538 |
| situated_generation | 0.0000 | 0 | 115 |
| temporal_dialogue | 0.6528 | 944 | 1446 |
| temporal_nli | 0.4389 | 3057 | 6965 |
| temporal_ordering | 0.5388 | 2289 | 4248 |
| temporal_qa | 0.5230 | 523 | 1000 |
| temporal_reasoning | 0.5288 | 992 | 1876 |

## Error Analysis

| Error Type | Count |
|------------|-------|
| hallucination | 309 |
| other | 11219 |
| temporal_extraction | 353 |
