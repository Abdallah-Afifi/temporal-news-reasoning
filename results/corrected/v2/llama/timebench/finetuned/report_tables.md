# Evaluation Report: llama on timebench/zero_shot

Generated: 2026-09-03T23:06:39.086057
Num examples: 21188

## Overall Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | 0.4380 |
| Overall F1 | 0.2128 |
| Exact Match | 0.4380 |
| Temporal F1 | 0.9712 |

## Results by Temporal Reasoning Type

| Category | Accuracy | Correct | Total |
|----------|----------|---------|-------|
| arithmetic | 0.1643 | 657 | 4000 |
| duration | 0.7731 | 1189 | 1538 |
| situated_generation | 0.0000 | 0 | 115 |
| temporal_dialogue | 0.4620 | 668 | 1446 |
| temporal_nli | 0.4616 | 3215 | 6965 |
| temporal_ordering | 0.5259 | 2234 | 4248 |
| temporal_qa | 0.5480 | 548 | 1000 |
| temporal_reasoning | 0.4104 | 770 | 1876 |

## Error Analysis

| Error Type | Count |
|------------|-------|
| hallucination | 24 |
| other | 11344 |
| temporal_extraction | 539 |
