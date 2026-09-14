# Evaluation Report: mistral on timebench/zero_shot

Generated: 2026-09-08T15:00:34.180338
Num examples: 21188

## Overall Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | 0.4490 |
| Overall F1 | 0.1670 |
| Exact Match | 0.4490 |
| Temporal F1 | 0.9299 |

## Results by Temporal Reasoning Type

| Category | Accuracy | Correct | Total |
|----------|----------|---------|-------|
| arithmetic | 0.0925 | 370 | 4000 |
| duration | 0.7614 | 1171 | 1538 |
| situated_generation | 0.0000 | 0 | 115 |
| temporal_dialogue | 0.6432 | 930 | 1446 |
| temporal_nli | 0.5094 | 3548 | 6965 |
| temporal_ordering | 0.5946 | 2526 | 4248 |
| temporal_qa | 0.3320 | 332 | 1000 |
| temporal_reasoning | 0.3390 | 636 | 1876 |

## Error Analysis

| Error Type | Count |
|------------|-------|
| hallucination | 614 |
| other | 10561 |
| temporal_extraction | 500 |
