# Evaluation Report: mistral on timebench/zero_shot

Generated: 2026-09-09T21:32:22.588541
Num examples: 21188

## Overall Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | 0.3551 |
| Overall F1 | 0.0632 |
| Exact Match | 0.3551 |
| Temporal F1 | 0.8263 |

## Results by Temporal Reasoning Type

| Category | Accuracy | Correct | Total |
|----------|----------|---------|-------|
| arithmetic | 0.0222 | 89 | 4000 |
| duration | 0.5696 | 876 | 1538 |
| situated_generation | 0.0000 | 0 | 115 |
| temporal_dialogue | 0.6058 | 876 | 1446 |
| temporal_nli | 0.5167 | 3599 | 6965 |
| temporal_ordering | 0.4868 | 2068 | 4248 |
| temporal_qa | 0.0100 | 10 | 1000 |
| temporal_reasoning | 0.0027 | 5 | 1876 |

## Error Analysis

| Error Type | Count |
|------------|-------|
| hallucination | 2377 |
| other | 10599 |
| temporal_extraction | 689 |
