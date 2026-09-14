# Evaluation Report: llama on timebench/finetuned

Generated: 2026-09-06T19:52:52.109709
Num examples: 21188

## Overall Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | 0.4602 |
| Overall F1 | 0.2250 |
| Exact Match | 0.4602 |
| Temporal F1 | 0.9701 |

## Results by Temporal Reasoning Type

| Category | Accuracy | Correct | Total |
|----------|----------|---------|-------|
| arithmetic | 0.1485 | 594 | 4000 |
| duration | 0.7971 | 1226 | 1538 |
| situated_generation | 0.0000 | 0 | 115 |
| temporal_dialogue | 0.6058 | 876 | 1446 |
| temporal_nli | 0.4663 | 3248 | 6965 |
| temporal_ordering | 0.5518 | 2344 | 4248 |
| temporal_qa | 0.5520 | 552 | 1000 |
| temporal_reasoning | 0.4856 | 911 | 1876 |

## Error Analysis

| Error Type | Count |
|------------|-------|
| hallucination | 82 |
| other | 10864 |
| temporal_extraction | 491 |
