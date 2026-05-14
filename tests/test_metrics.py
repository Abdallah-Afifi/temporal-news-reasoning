from src.evaluation.metrics import TemporalEvaluator, extract_temporal_expressions, temporal_f1


def test_extract_temporal_expressions_handles_none() -> None:
    assert extract_temporal_expressions(None) == []


def test_temporal_f1_handles_none_values() -> None:
    score = temporal_f1([None, "March 2024"], ["", None])
    assert score == 0.5


def test_classify_errors_handles_none_contexts() -> None:
    evaluator = TemporalEvaluator("time")

    errors = evaluator.classify_errors(
        predictions=["March 2024", "April 2024"],
        gold_labels=["March 2024", "March 2024"],
        contexts=[None, None],
    )

    assert errors["temporal_extraction"] == [1]
    assert errors["hallucination"] == []
    assert errors["other"] == []