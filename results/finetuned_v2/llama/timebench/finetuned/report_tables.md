# Evaluation Report: llama on timebench/zero_shot

Generated: 2026-09-02T14:40:08.702060
Num examples: 21188

## Overall Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | 0.4375 |
| Overall F1 | 0.2132 |
| Exact Match | 0.4375 |
| Temporal F1 | 0.9849 |

## Results by Temporal Reasoning Type

| Category | Accuracy | Correct | Total |
|----------|----------|---------|-------|
| arithmetic | 0.1618 | 647 | 4000 |
| duration | 0.7211 | 1109 | 1538 |
| situated_generation | 0.0000 | 0 | 115 |
| temporal_dialogue | 0.5318 | 769 | 1446 |
| temporal_nli | 0.4617 | 3216 | 6965 |
| temporal_ordering | 0.5250 | 2230 | 4248 |
| temporal_qa | 0.5330 | 533 | 1000 |
| temporal_reasoning | 0.4083 | 766 | 1876 |

## Error Analysis

| Error Type | Count |
|------------|-------|
| hallucination | 26 |
| other | 11586 |
| temporal_extraction | 306 |
