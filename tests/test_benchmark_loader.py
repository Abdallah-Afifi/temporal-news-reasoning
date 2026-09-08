import json

import pytest

from src.data.data_loader import BenchmarkLoader, TemporalExample


@pytest.fixture
def bench_dir(tmp_path):
    """Create a minimal benchmark tree with split files and a .git dir."""
    time_dir = tmp_path / "time" / "TIME"
    time_dir.mkdir(parents=True)

    (time_dir / "test.jsonl").write_text(
        "\n".join([
            json.dumps({
                "id": "t1",
                "question": "When did the event happen?",
                "context": "Article text",
                "answer": "March 2024",
                "temporal_type": "temporal_qa",
            }),
            json.dumps({
                "id": "t2",
                "query": "Who was president?",
                "targets": ["Obama", "Biden"],
                "choices": ["Obama", "Biden", "Bush"],
                "type": "ordering",
            }),
        ]),
        encoding="utf-8",
    )
    (time_dir / "train.jsonl").write_text(
        json.dumps({"question": "train only?", "answer": "yes"}),
        encoding="utf-8",
    )

    # Noise that must be ignored
    (tmp_path / "time" / "TIME" / ".git").mkdir()
    (tmp_path / "time" / "TIME" / ".git" / "HEAD").write_text("ref", encoding="utf-8")
    (tmp_path / "time" / "README.md").write_text("docs", encoding="utf-8")
    (tmp_path / "time" / "package.json").write_text("{}", encoding="utf-8")
    return tmp_path


def test_load_filters_to_test_split_and_ignores_noise(bench_dir):
    loader = BenchmarkLoader(str(bench_dir))
    examples = loader.load("time")

    assert len(examples) == 2  # test.jsonl only; train.jsonl, .git, README ignored
    assert all(not e.id.startswith("train") for e in examples)


def test_record_normalization(bench_dir):
    loader = BenchmarkLoader(str(bench_dir))
    examples = {e.id: e for e in loader.load("time")}

    t1 = examples["t1"]
    assert t1.question == "When did the event happen?"
    assert t1.answer == "March 2024"
    assert t1.context == "Article text"
    assert t1.temporal_type == "temporal_qa"
    assert t1.source == "time"
    assert t1.split == "test"

    # 'query'/'targets'/'type' fallback keys
    t2 = examples["t2"]
    assert t2.question == "Who was president?"
    assert t2.answer == ["Obama", "Biden"]
    assert t2.choices == ["Obama", "Biden", "Bush"]
    assert t2.temporal_type == "ordering"
    assert t2.task == "test"  # file stem used as fallback task


def test_split_none_loads_everything(bench_dir):
    loader = BenchmarkLoader(str(bench_dir))
    examples = loader.load("time", split=None)
    assert len(examples) == 3


def test_unknown_benchmark_raises(bench_dir):
    loader = BenchmarkLoader(str(bench_dir))
    with pytest.raises(ValueError):
        loader.load("mmlu")


def test_missing_data_raises_actionable_error(tmp_path):
    loader = BenchmarkLoader(str(tmp_path / "empty"))
    with pytest.raises(FileNotFoundError, match="download_datasets"):
        loader.load("time")


def test_task_filter(bench_dir):
    loader = BenchmarkLoader(str(bench_dir))
    examples = loader.load("time", task="ordering")
    assert [e.id for e in examples] == ["t2"]


def test_fallback_ids_are_stable(bench_dir):
    (bench_dir / "time" / "TIME" / "extra.jsonl").write_text(
        json.dumps({"question": "no id?", "answer": "x"}) + "\n",
        encoding="utf-8",
    )
    loader = BenchmarkLoader(str(bench_dir))
    first = loader.load("time", split=None)
    second = loader.load("time", split=None)
    assert [e.id for e in first] == [e.id for e in second]
    no_id = next(e for e in first if e.question == "no id?")
    assert no_id.id == "extra.jsonl-0"


def test_tram_ids_are_unique() -> None:
    """TRAM ids collided 31,626 times (3.2%) until 2026-09-07.

    `_load_tram` indexed rows as `len(examples) + row_idx`, which double-counts:
    within one CSV it produced 0, 2, 4, 6 ... and the next CSV restarted at
    len(examples), landing back inside the range already used.

    This was not cosmetic. `run_baselines.py` keys its resume set on `id`, so
    every duplicate after the first would be skipped as "already completed" and
    a full TRAM run would silently evaluate ~31,626 fewer items than it
    reported. Pinned here because TRAM's first real run depends on it.
    """
    from collections import Counter
    from pathlib import Path

    import pytest

    from src.data.data_loader import BenchmarkLoader

    data_dir = Path(__file__).resolve().parents[1] / "data" / "benchmarks"
    if not (data_dir / "tram" / "TRAM-Benchmark" / "datasets").exists():
        pytest.skip("TRAM data not present")

    examples = BenchmarkLoader(data_dir=str(data_dir)).load("tram")
    counts = Counter(e.id for e in examples)
    duplicates = {i: c for i, c in counts.items() if c > 1}
    assert not duplicates, (
        f"{sum(c - 1 for c in duplicates.values())} duplicate TRAM ids; "
        f"resume would silently skip them"
    )
    assert len(counts) == len(examples)
