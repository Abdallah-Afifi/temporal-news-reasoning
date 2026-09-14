# Evaluation Report: llama on timebench/zero_shot

Generated: 2026-09-03T22:54:11.345413
Num examples: 21188

## Overall Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | 0.3864 |
| Overall F1 | 0.1880 |
| Exact Match | 0.3864 |
| Temporal F1 | 0.9534 |

## Results by Temporal Reasoning Type

| Category | Accuracy | Correct | Total |
|----------|----------|---------|-------|
| arithmetic | 0.0428 | 171 | 4000 |
| duration | 0.7328 | 1127 | 1538 |
| situated_generation | 0.0000 | 0 | 115 |
| temporal_dialogue | 0.6867 | 993 | 1446 |
| temporal_nli | 0.3637 | 2533 | 6965 |
| temporal_ordering | 0.5085 | 2160 | 4248 |
| temporal_qa | 0.4020 | 402 | 1000 |
| temporal_reasoning | 0.4275 | 802 | 1876 |

## Error Analysis

| Error Type | Count |
|------------|-------|
| hallucination | 247 |
| other | 12032 |
| temporal_extraction | 721 |
