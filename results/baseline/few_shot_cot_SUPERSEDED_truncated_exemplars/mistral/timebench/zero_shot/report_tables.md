# Evaluation Report: mistral on timebench/zero_shot

Generated: 2026-09-14T09:20:32.053442
Num examples: 21075

## Overall Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | 0.4319 |
| Overall F1 | 0.1535 |
| Exact Match | 0.4319 |
| Temporal F1 | 0.8987 |

## Results by Temporal Reasoning Type

| Category | Accuracy | Correct | Total |
|----------|----------|---------|-------|
| arithmetic | 0.0578 | 231 | 4000 |
| duration | 0.5891 | 906 | 1538 |
| situated_generation | 0.0000 | 0 | 115 |
| temporal_dialogue | 0.6791 | 982 | 1446 |
| temporal_nli | 0.4830 | 3364 | 6965 |
| temporal_ordering | 0.5673 | 2410 | 4248 |
| temporal_qa | 0.4555 | 404 | 887 |
| temporal_reasoning | 0.4296 | 806 | 1876 |

## Error Analysis

| Error Type | Count |
|------------|-------|
| hallucination | 1832 |
| other | 9886 |
| temporal_extraction | 254 |
