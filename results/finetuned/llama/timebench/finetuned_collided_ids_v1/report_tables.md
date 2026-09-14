# Evaluation Report: llama on timebench/zero_shot

Generated: 2026-08-30T15:56:54.312232
Num examples: 21188

## Overall Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | 0.3819 |
| Overall F1 | 0.1786 |
| Exact Match | 0.3819 |
| Temporal F1 | 0.9650 |

## Results by Temporal Reasoning Type

| Category | Accuracy | Correct | Total |
|----------|----------|---------|-------|
| arithmetic | 0.0345 | 138 | 4000 |
| duration | 0.7315 | 1125 | 1538 |
| situated_generation | 0.0000 | 0 | 115 |
| temporal_dialogue | 0.6840 | 989 | 1446 |
| temporal_nli | 0.3624 | 2524 | 6965 |
| temporal_ordering | 0.5075 | 2156 | 4248 |
| temporal_qa | 0.3760 | 376 | 1000 |
| temporal_reasoning | 0.4179 | 784 | 1876 |

## Error Analysis

| Error Type | Count |
|------------|-------|
| hallucination | 276 |
| other | 12357 |
| temporal_extraction | 463 |
