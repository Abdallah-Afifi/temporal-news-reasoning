# Evaluation Report: llama on timebench/zero_shot

Generated: 2026-09-01T21:06:30.596012
Num examples: 21188

## Overall Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | 0.4370 |
| Overall F1 | 0.1746 |
| Exact Match | 0.4370 |
| Temporal F1 | 0.9242 |

## Results by Temporal Reasoning Type

| Category | Accuracy | Correct | Total |
|----------|----------|---------|-------|
| arithmetic | 0.0155 | 62 | 4000 |
| duration | 0.7841 | 1206 | 1538 |
| situated_generation | 0.0000 | 0 | 115 |
| temporal_dialogue | 0.7863 | 1137 | 1446 |
| temporal_nli | 0.4679 | 3259 | 6965 |
| temporal_ordering | 0.5155 | 2190 | 4248 |
| temporal_qa | 0.3880 | 388 | 1000 |
| temporal_reasoning | 0.5426 | 1018 | 1876 |

## Error Analysis

| Error Type | Count |
|------------|-------|
| hallucination | 1269 |
| other | 10294 |
| temporal_extraction | 365 |
