# Evaluation Report: llama on timebench/zero_shot

Generated: 2026-09-13T23:22:17.677631
Num examples: 21188

## Overall Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | 0.4444 |
| Overall F1 | 0.1702 |
| Exact Match | 0.4444 |
| Temporal F1 | 0.9052 |

## Results by Temporal Reasoning Type

| Category | Accuracy | Correct | Total |
|----------|----------|---------|-------|
| arithmetic | 0.0688 | 275 | 4000 |
| duration | 0.4948 | 761 | 1538 |
| situated_generation | 0.0000 | 0 | 115 |
| temporal_dialogue | 0.6895 | 997 | 1446 |
| temporal_nli | 0.5197 | 3620 | 6965 |
| temporal_ordering | 0.5405 | 2296 | 4248 |
| temporal_qa | 0.4220 | 422 | 1000 |
| temporal_reasoning | 0.5576 | 1046 | 1876 |

## Error Analysis

| Error Type | Count |
|------------|-------|
| hallucination | 1702 |
| other | 9637 |
| temporal_extraction | 432 |
