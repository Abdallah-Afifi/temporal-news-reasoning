# Evaluation Report: llama on timebench/zero_shot

Generated: 2026-09-03T03:27:25.278099
Num examples: 21188

## Overall Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | 0.4550 |
| Overall F1 | 0.2165 |
| Exact Match | 0.4550 |
| Temporal F1 | 0.9820 |

## Results by Temporal Reasoning Type

| Category | Accuracy | Correct | Total |
|----------|----------|---------|-------|
| arithmetic | 0.1403 | 561 | 4000 |
| duration | 0.7289 | 1121 | 1538 |
| situated_generation | 0.0000 | 0 | 115 |
| temporal_dialogue | 0.6508 | 941 | 1446 |
| temporal_nli | 0.4892 | 3407 | 6965 |
| temporal_ordering | 0.5527 | 2348 | 4248 |
| temporal_qa | 0.5380 | 538 | 1000 |
| temporal_reasoning | 0.3859 | 724 | 1876 |

## Error Analysis

| Error Type | Count |
|------------|-------|
| hallucination | 45 |
| other | 11159 |
| temporal_extraction | 344 |
