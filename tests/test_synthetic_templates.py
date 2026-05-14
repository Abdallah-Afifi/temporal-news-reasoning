"""Tests for synthetic news temporal dataset generation."""

from src.data.synthetic_templates import NewsArticle, SyntheticDataGenerator


def test_generate_from_article_creates_targeted_examples():
    generator = SyntheticDataGenerator(seed=7)
    article = NewsArticle(
        source="ccnews",
        article_id="sample-1",
        title="Deal announced in March",
        text=(
            "On March 4, 2024, the company announced a new deal. "
            "Seven days later, on March 11, 2024, the board approved it. "
            "The announcement came after weeks of talks."
        ),
    )

    examples = generator.generate_from_article(article)
    categories = {example["category"] for example in examples}

    assert examples
    assert "explicit_date_extraction" in categories
    assert "duration" in categories or "temporal_ordering" in categories
    for example in examples:
        assert example["source_id"] == "sample-1"
        assert example["messages"][0]["role"] == "user"
        assert example["messages"][1]["role"] == "assistant"
        assert example["answer"]
        assert example["rationale"]


def test_split_examples_groups_by_source_id():
    generator = SyntheticDataGenerator(seed=11)
    examples = [
        {
            "id": "a",
            "category": "duration",
            "question": "q1",
            "answer": "1",
            "rationale": "r1",
            "source_id": "article-1",
            "source_dataset": "ccnews",
        },
        {
            "id": "b",
            "category": "duration",
            "question": "q2",
            "answer": "2",
            "rationale": "r2",
            "source_id": "article-1",
            "source_dataset": "ccnews",
        },
        {
            "id": "c",
            "category": "temporal_nli",
            "question": "q3",
            "answer": "false",
            "rationale": "r3",
            "source_id": "article-2",
            "source_dataset": "ccnews",
        },
    ]

    splits = generator.split_examples(examples, train_ratio=1 / 3, val_ratio=1 / 3, test_ratio=1 / 3)
    source_locations = {}
    for split_name, split_records in splits.items():
        for record in split_records:
            source_id = record["source_id"]
            if source_id in source_locations:
                assert source_locations[source_id] == split_name
            else:
                source_locations[source_id] = split_name

    assert set(source_locations) == {"article-1", "article-2"}


def test_validate_examples_returns_summary():
    generator = SyntheticDataGenerator(seed=3)
    examples = [
        {
            "id": "a",
            "category": "duration",
            "question": "How many days?",
            "answer": "3",
            "rationale": "The gap is 3 days.",
            "source_id": "article-1",
            "source_dataset": "ccnews",
            "benchmark_targets": ["TIME"],
        }
    ]

    summary = generator.validate_examples(examples)
    assert summary["num_examples"] == 1
    assert summary["missing_required_fields"] == 0
    assert summary["duplicate_examples"] == 0
    assert summary["category_distribution"]["duration"] == 1
