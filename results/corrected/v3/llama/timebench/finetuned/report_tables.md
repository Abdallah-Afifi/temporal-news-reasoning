# Evaluation Report: llama on timebench/zero_shot

Generated: 2026-09-03T23:18:50.692986
Num examples: 21188

## Overall Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | 0.4574 |
| Overall F1 | 0.2194 |
| Exact Match | 0.4574 |
| Temporal F1 | 0.9684 |

## Results by Temporal Reasoning Type

| Category | Accuracy | Correct | Total |
|----------|----------|---------|-------|
| arithmetic | 0.1403 | 561 | 4000 |
| duration | 0.7867 | 1210 | 1538 |
| situated_generation | 0.0000 | 0 | 115 |
| temporal_dialogue | 0.6058 | 876 | 1446 |
| temporal_nli | 0.4890 | 3406 | 6965 |
| temporal_ordering | 0.5534 | 2351 | 4248 |
| temporal_qa | 0.5610 | 561 | 1000 |
| temporal_reasoning | 0.3870 | 726 | 1876 |

## Error Analysis

| Error Type | Count |
|------------|-------|
| hallucination | 44 |
| other | 10881 |
| temporal_extraction | 572 |
