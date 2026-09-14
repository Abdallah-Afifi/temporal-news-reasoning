# Evaluation Report: llama on timebench/zero_shot

Generated: 2026-09-04T06:07:00.610866
Num examples: 21188

## Overall Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | 0.4655 |
| Overall F1 | 0.2247 |
| Exact Match | 0.4655 |
| Temporal F1 | 0.9741 |

## Results by Temporal Reasoning Type

| Category | Accuracy | Correct | Total |
|----------|----------|---------|-------|
| arithmetic | 0.1393 | 557 | 4000 |
| duration | 0.7685 | 1182 | 1538 |
| situated_generation | 0.0000 | 0 | 115 |
| temporal_dialogue | 0.5636 | 815 | 1446 |
| temporal_nli | 0.4838 | 3370 | 6965 |
| temporal_ordering | 0.5742 | 2439 | 4248 |
| temporal_qa | 0.5660 | 566 | 1000 |
| temporal_reasoning | 0.4979 | 934 | 1876 |

## Error Analysis

| Error Type | Count |
|------------|-------|
| hallucination | 18 |
| other | 10858 |
| temporal_extraction | 449 |
