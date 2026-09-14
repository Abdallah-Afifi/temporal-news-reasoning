# Evaluation Report: mistral on timebench/zero_shot

Generated: 2026-08-31T04:31:58.522661
Num examples: 21188

## Overall Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | 0.4148 |
| Overall F1 | 0.1313 |
| Exact Match | 0.4148 |
| Temporal F1 | 0.8746 |

## Results by Temporal Reasoning Type

| Category | Accuracy | Correct | Total |
|----------|----------|---------|-------|
| arithmetic | 0.0025 | 10 | 4000 |
| duration | 0.7204 | 1108 | 1538 |
| situated_generation | 0.0000 | 0 | 115 |
| temporal_dialogue | 0.8181 | 1183 | 1446 |
| temporal_nli | 0.4770 | 3322 | 6965 |
| temporal_ordering | 0.5548 | 2357 | 4248 |
| temporal_qa | 0.2760 | 276 | 1000 |
| temporal_reasoning | 0.2841 | 533 | 1876 |

## Error Analysis

| Error Type | Count |
|------------|-------|
| hallucination | 2246 |
| other | 9800 |
| temporal_extraction | 353 |
