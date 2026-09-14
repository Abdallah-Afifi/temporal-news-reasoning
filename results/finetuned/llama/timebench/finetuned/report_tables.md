# Evaluation Report: llama on timebench/zero_shot

Generated: 2026-08-31T03:40:45.235355
Num examples: 21188

## Overall Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | 0.3806 |
| Overall F1 | 0.1773 |
| Exact Match | 0.3806 |
| Temporal F1 | 0.9647 |

## Results by Temporal Reasoning Type

| Category | Accuracy | Correct | Total |
|----------|----------|---------|-------|
| arithmetic | 0.0335 | 134 | 4000 |
| duration | 0.7334 | 1128 | 1538 |
| situated_generation | 0.0000 | 0 | 115 |
| temporal_dialogue | 0.6826 | 987 | 1446 |
| temporal_nli | 0.3605 | 2511 | 6965 |
| temporal_ordering | 0.5066 | 2152 | 4248 |
| temporal_qa | 0.3700 | 370 | 1000 |
| temporal_reasoning | 0.4174 | 783 | 1876 |

## Error Analysis

| Error Type | Count |
|------------|-------|
| hallucination | 281 |
| other | 12377 |
| temporal_extraction | 465 |
