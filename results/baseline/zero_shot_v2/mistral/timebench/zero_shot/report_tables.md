# Evaluation Report: mistral on timebench/zero_shot

Generated: 2026-09-02T15:15:44.973712
Num examples: 21188

## Overall Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | 0.4448 |
| Overall F1 | 0.1589 |
| Exact Match | 0.4448 |
| Temporal F1 | 0.9420 |

## Results by Temporal Reasoning Type

| Category | Accuracy | Correct | Total |
|----------|----------|---------|-------|
| arithmetic | 0.0772 | 309 | 4000 |
| duration | 0.7601 | 1169 | 1538 |
| situated_generation | 0.0000 | 0 | 115 |
| temporal_dialogue | 0.6452 | 933 | 1446 |
| temporal_nli | 0.5097 | 3550 | 6965 |
| temporal_ordering | 0.5944 | 2525 | 4248 |
| temporal_qa | 0.3060 | 306 | 1000 |
| temporal_reasoning | 0.3374 | 633 | 1876 |

## Error Analysis

| Error Type | Count |
|------------|-------|
| hallucination | 641 |
| other | 10899 |
| temporal_extraction | 223 |
