# Evaluation Report: llama on timebench/zero_shot

Generated: 2026-09-03T14:58:19.006762
Num examples: 21188

## Overall Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | 0.4480 |
| Overall F1 | 0.2047 |
| Exact Match | 0.4480 |
| Temporal F1 | 0.9187 |

## Results by Temporal Reasoning Type

| Category | Accuracy | Correct | Total |
|----------|----------|---------|-------|
| arithmetic | 0.0612 | 245 | 4000 |
| duration | 0.7861 | 1209 | 1538 |
| situated_generation | 0.0000 | 0 | 115 |
| temporal_dialogue | 0.7863 | 1137 | 1446 |
| temporal_nli | 0.4688 | 3265 | 6965 |
| temporal_ordering | 0.5155 | 2190 | 4248 |
| temporal_qa | 0.4280 | 428 | 1000 |
| temporal_reasoning | 0.5432 | 1019 | 1876 |

## Error Analysis

| Error Type | Count |
|------------|-------|
| hallucination | 1129 |
| other | 9955 |
| temporal_extraction | 611 |
