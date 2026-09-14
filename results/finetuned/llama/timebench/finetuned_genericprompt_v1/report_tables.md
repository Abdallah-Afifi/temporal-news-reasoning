# Evaluation Report: llama on timebench/zero_shot

Generated: 2026-08-30T14:50:23.871607
Num examples: 21188

## Overall Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | 0.1614 |
| Overall F1 | 0.1223 |
| Exact Match | 0.1614 |
| Temporal F1 | 0.9479 |

## Results by Temporal Reasoning Type

| Category | Accuracy | Correct | Total |
|----------|----------|---------|-------|
| arithmetic | 0.0343 | 137 | 4000 |
| duration | 0.7341 | 1129 | 1538 |
| situated_generation | 0.0000 | 0 | 115 |
| temporal_dialogue | 0.6860 | 992 | 1446 |
| temporal_nli | 0.0000 | 0 | 6965 |
| temporal_ordering | 0.0000 | 0 | 4248 |
| temporal_qa | 0.3780 | 378 | 1000 |
| temporal_reasoning | 0.4179 | 784 | 1876 |

## Error Analysis

| Error Type | Count |
|------------|-------|
| hallucination | 501 |
| other | 16804 |
| temporal_extraction | 463 |
