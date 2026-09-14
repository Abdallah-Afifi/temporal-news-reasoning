# Evaluation Report: llama on timebench/zero_shot

Generated: 2026-09-09T18:54:22.519689
Num examples: 21188

## Overall Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | 0.4688 |
| Overall F1 | 0.1873 |
| Exact Match | 0.4688 |
| Temporal F1 | 0.9036 |

## Results by Temporal Reasoning Type

| Category | Accuracy | Correct | Total |
|----------|----------|---------|-------|
| arithmetic | 0.1285 | 514 | 4000 |
| duration | 0.6743 | 1037 | 1538 |
| situated_generation | 0.0000 | 0 | 115 |
| temporal_dialogue | 0.5823 | 842 | 1446 |
| temporal_nli | 0.5400 | 3761 | 6965 |
| temporal_ordering | 0.5513 | 2342 | 4248 |
| temporal_qa | 0.3890 | 389 | 1000 |
| temporal_reasoning | 0.5581 | 1047 | 1876 |

## Error Analysis

| Error Type | Count |
|------------|-------|
| hallucination | 1464 |
| other | 9278 |
| temporal_extraction | 514 |
