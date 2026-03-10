"""Unified benchmark data loader for TIME, TIMEBENCH, and TRAM.

All loaders convert native formats into a common TemporalExample schema:

    {
        "id":            str            – unique example identifier
        "question":      str            – the question / prompt text
        "context":       str            – supporting passage (empty string if none)
        "answer":        str | list     – gold answer(s)
        "choices":       list | None    – MCQ options (None for open-ended)
        "temporal_type": str            – coarse category of temporal reasoning
        "difficulty":    str | None     – difficulty tier when available
        "source":        str            – benchmark name ("time", "timebench", "tram")
        "task":          str            – sub-task / dataset name
        "metadata":      dict           – any remaining original fields
    }
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Canonical data structure
# ---------------------------------------------------------------------------

@dataclass
class TemporalExample:
    id: str
    question: str
    context: str
    answer: str | list[str]
    choices: list[str] | None
    temporal_type: str
    difficulty: str | None
    source: str
    task: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

class BenchmarkLoader:
    """Load and standardize TIME, TIMEBENCH, and TRAM benchmark datasets.

    Usage::

        loader = BenchmarkLoader("./data/benchmarks")

        # All examples from a benchmark
        examples = loader.load("timebench")

        # Specific task only
        examples = loader.load("timebench", task="TimeQA")

        # List available tasks
        print(loader.available_tasks("tram"))
    """

    SUPPORTED_BENCHMARKS = ["time", "timebench", "tram"]

    def __init__(self, data_dir: str = "./data/benchmarks"):
        self.data_dir = Path(data_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        benchmark: str,
        task: str | None = None,
        split: str = "test",
    ) -> list[TemporalExample]:
        """Load a benchmark dataset into the unified format.

        Args:
            benchmark: One of ``'time'``, ``'timebench'``, ``'tram'``.
            task:      Sub-task name to filter (None = load all tasks).
            split:     For TIME benchmark: ``'full'`` or ``'lite'``.
                       For TIMEBENCH / TRAM: not currently used (all data
                       returned) – kept for API compatibility.

        Returns:
            List of :class:`TemporalExample` objects.
        """
        benchmark = benchmark.lower()
        if benchmark not in self.SUPPORTED_BENCHMARKS:
            raise ValueError(
                f"Unknown benchmark: {benchmark!r}. "
                f"Supported: {self.SUPPORTED_BENCHMARKS}"
            )
        loader_fn = {
            "time": self._load_time,
            "timebench": self._load_timebench,
            "tram": self._load_tram,
        }[benchmark]
        examples = loader_fn(task=task, split=split)
        return examples

    def available_tasks(self, benchmark: str) -> list[str]:
        """Return the list of sub-task names available for a benchmark."""
        benchmark = benchmark.lower()
        if benchmark == "time":
            return ["newest", "lite", "all"]
        if benchmark == "timebench":
            # Prefer full zip
            zip_path = self.data_dir / "timebench" / "TimeBench" / "TimeBench-full-19000.zip"
            if zip_path.exists():
                import zipfile
                with zipfile.ZipFile(zip_path) as z:
                    tasks = sorted(set(
                        n.split("/")[2]
                        for n in z.namelist()
                        if n.endswith(".jsonl") and len(n.split("/")) >= 3
                    ))
                return tasks
            # Fall back to extracted subset
            subset_dir = self.data_dir / "timebench" / "TimeBench" / "TimeBench-subset-7553"
            if not subset_dir.exists():
                return []
            return sorted(p.name for p in subset_dir.iterdir() if p.is_dir())
        if benchmark == "tram":
            datasets_dir = self.data_dir / "tram" / "TRAM-Benchmark" / "datasets"
            if not datasets_dir.exists():
                return []
            return sorted(
                p.stem for p in datasets_dir.glob("*.zip")
                if "shots" not in p.stem
            )
        return []

    # ------------------------------------------------------------------
    # TIME benchmark
    # ------------------------------------------------------------------

    # Temporal type heuristics based on question keywords
    _TIME_TEMPORAL_KEYWORDS: list[tuple[str, str]] = [
        ("before", "temporal_ordering"),
        ("after", "temporal_ordering"),
        ("at the same time", "temporal_simultaneous"),
        ("when", "temporal_qa"),
        ("how long", "duration"),
        ("how many year", "arithmetic"),
        ("how many month", "arithmetic"),
        ("how many day", "arithmetic"),
        ("at the time", "temporal_context"),
        ("during", "temporal_context"),
        ("while", "temporal_context"),
        ("since", "temporal_context"),
        ("until", "temporal_context"),
        ("recent", "temporal_recency"),
        ("latest", "temporal_recency"),
        ("current", "temporal_recency"),
        ("first", "temporal_ordering"),
        ("last", "temporal_ordering"),
    ]

    def _infer_time_type(self, question: str) -> str:
        q = question.lower()
        for keyword, ttype in self._TIME_TEMPORAL_KEYWORDS:
            if keyword in q:
                return ttype
        return "news_temporal_mcq"

    def _load_time(self, task: str | None, split: str) -> list[TemporalExample]:
        """Load the TIME / TIME-Lite benchmark (news MCQ).

        split / task options:
          - ``'newest'`` or ``'full'`` → TIME_Newest.json only
          - ``'lite'``               → TIME-Lite.json only
          - ``None`` / ``'all'``     → both files (default)
        """
        time_dir = self.data_dir / "time"
        newest_path = time_dir / "TIME" / "TIME_Newest.json"
        lite_path   = time_dir / "TIME-Lite" / "TIME-Lite.json"

        want = (task or split or "all").lower()

        candidate_pairs: list[tuple[str, Path]] = []
        if want in ("newest", "full"):
            candidate_pairs = [("newest", newest_path)]
        elif want == "lite":
            candidate_pairs = [("lite", lite_path)]
        else:  # 'all' or None — load both
            candidate_pairs = [("newest", newest_path), ("lite", lite_path)]

        files_to_load = [(v, p) for v, p in candidate_pairs if p.exists()]

        if not files_to_load:
            raise FileNotFoundError(
                f"TIME benchmark files not found under {time_dir}. "
                "Run scripts/download_datasets.py first."
            )

        examples: list[TemporalExample] = []
        for variant, path in files_to_load:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)

            for idx, item in enumerate(raw):
                question_text = item.get("Question", "")
                gold = str(item.get("Gold Answer", ""))
                context_raw = item.get("Context", "")

                # Parse choices embedded in question text (lines starting A. B. C. D.)
                lines = question_text.split("\n")
                question_line = lines[0].strip()
                choices = []
                for line in lines[1:]:
                    stripped = line.strip()
                    if stripped and stripped[0] in "ABCD" and len(stripped) > 2 and stripped[1] in ".):":
                        choices.append(stripped[2:].strip())

                examples.append(TemporalExample(
                    id=f"time-{variant}-{idx}",
                    question=question_line,
                    context=context_raw,
                    answer=gold,
                    choices=choices if choices else None,
                    temporal_type=self._infer_time_type(question_line),
                    difficulty=None,
                    source="time",
                    task=variant,
                    metadata={"raw_question": question_text},
                ))
        return examples

    # ------------------------------------------------------------------
    # TIMEBENCH
    # ------------------------------------------------------------------

    # Map lowercase task folder name → temporal_type
    _TIMEBENCH_TYPE_MAP: dict[str, str] = {
        # full zip names (lowercase)
        "timeqa":       "temporal_qa",
        "menatqa":      "temporal_qa",
        "tempreason":   "temporal_reasoning",
        "timedial":     "temporal_dialogue",
        "durationqa":   "duration",
        "mctaco":       "duration",
        "nli":          "temporal_nli",
        "timex-nli":    "temporal_nli",
        "tracie":       "temporal_ordering",
        "situatedgen":  "situated_generation",
        "date_arith":   "arithmetic",
        # subset names (Title-case) kept for fallback
        "TimeQA":       "temporal_qa",
        "MenatQA":      "temporal_qa",
        "TempReason":   "temporal_reasoning",
        "TimeDial":     "temporal_dialogue",
        "DurationQA":   "duration",
        "McTaco":       "duration",
        "TimeX-NLI":    "temporal_nli",
        "TRACIE":       "temporal_ordering",
        "SituatedGen":  "situated_generation",
    }

    # File stem keywords → difficulty label
    _TIMEBENCH_DIFFICULTY: dict[str, str] = {
        "easy":           "easy",
        "hard":           "hard",
        "l1":             "level_1",
        "l2":             "level_2",
        "l3":             "level_3",
        "f2":             "format_2",
        "cs1":            "cs1",
        "cs2":            "cs2",
        "cs3":            "cs3",
        "counterfactual": "counterfactual",
        "order":          "order",
        "scope":          "scope",
        "event-event":    "event-event",
        "event-time":     "event-time",
        "s1":             "s1",
        "s2":             "s2",
        "s3":             "s3",
    }

    def _parse_timebench_answer(self, item: dict, task: str = "") -> str | list[str]:
        # TRACIE / NLI with Premise+Hypothesis format
        if "Label" in item:
            return str(item["Label"])
        ans = item.get("answer", item.get("labels", ""))
        if isinstance(ans, list):
            if all(a in ("yes", "no") for a in ans):
                opts = item.get("options", [])
                correct = [opts[i] for i, a in enumerate(ans) if a == "yes"]
                return correct if len(correct) != 1 else correct[0]
            return ans
        return str(ans)

    def _parse_timebench_context(self, item: dict) -> str:
        # tempreason full zip may have both context + fact_context
        parts = []
        for key in ("context", "fact_context", "Premise"):
            val = item.get(key, "")
            if isinstance(val, list):
                val = " ".join(str(v) for v in val)
            if val:
                parts.append(str(val))
        return " ".join(parts)

    def _infer_timebench_difficulty(self, stem: str) -> str | None:
        stem_lower = stem.lower()
        for key, val in self._TIMEBENCH_DIFFICULTY.items():
            if key in stem_lower:
                return val
        return None

    def _timebench_item_to_example(
        self,
        item: dict,
        task_name: str,
        file_stem: str,
        line_no: int,
    ) -> TemporalExample:
        """Convert a single JSONL record to a TemporalExample."""
        task_key = task_name.lower()
        ttype = self._TIMEBENCH_TYPE_MAP.get(task_name, self._TIMEBENCH_TYPE_MAP.get(task_key, "temporal_reasoning"))
        difficulty = self._infer_timebench_difficulty(file_stem)

        item_id = item.get("idx", item.get("qid", f"{file_stem}-{line_no}"))

        # ── SituatedGen ──────────────────────────────────────────────
        if task_key == "situatedgen":
            question = "Generate a temporally-grounded statement using: " + ", ".join(item.get("keywords", []))
            answer: str | list[str] = item.get("statement", "")
            return TemporalExample(
                id=f"timebench-{task_key}-{item_id}",
                question=question,
                context="",
                answer=answer,
                choices=None,
                temporal_type=ttype,
                difficulty=difficulty,
                source="timebench",
                task=task_name,
                metadata={k: v for k, v in item.items() if k not in ("keywords", "statement")},
            )

        # ── TRACIE / NLI  (Premise + Hypothesis + Label) ─────────────
        if "Hypothesis" in item:
            question = str(item.get("Hypothesis", ""))
            context  = str(item.get("Premise", ""))
            answer   = self._parse_timebench_answer(item, task_key)
            return TemporalExample(
                id=f"timebench-{task_key}-{item_id}",
                question=question,
                context=context,
                answer=answer,
                choices=None,
                temporal_type=ttype,
                difficulty=difficulty,
                source="timebench",
                task=task_name,
                metadata={k: v for k, v in item.items() if k not in ("Hypothesis", "Premise", "Label")},
            )

        # ── TimeDial  (context is the dialogue, question is implicit) ─
        if task_key == "timedial":
            question = item.get("context", "")
            opts: list[str] | None = item.get("options")
            answer = self._parse_timebench_answer(item, task_key)
            return TemporalExample(
                id=f"timebench-{task_key}-{item_id}",
                question=str(question),
                context="",
                answer=answer,
                choices=opts,
                temporal_type=ttype,
                difficulty=difficulty,
                source="timebench",
                task=task_name,
                metadata={k: v for k, v in item.items() if k not in ("context", "options", "labels", "qid")},
            )

        # ── General case (TimeQA, MenatQA, TempReason, DurationQA, McTaco, date_arith) ──
        question = item.get("question", item.get("statement", ""))
        opts = item.get("options")
        answer = self._parse_timebench_answer(item, task_key)
        context = self._parse_timebench_context(item)

        return TemporalExample(
            id=f"timebench-{task_key}-{item_id}",
            question=str(question),
            context=context,
            answer=answer,
            choices=opts,
            temporal_type=ttype,
            difficulty=difficulty,
            source="timebench",
            task=task_name,
            metadata={
                k: v for k, v in item.items()
                if k not in ("question", "context", "fact_context", "answer",
                             "options", "labels", "statement", "keywords", "idx", "qid")
            },
        )

    def _load_timebench(self, task: str | None, split: str) -> list[TemporalExample]:
        """Load TIMEBENCH — prefers TimeBench-full-19000.zip, falls back to extracted subset."""
        timebench_dir = self.data_dir / "timebench" / "TimeBench"
        full_zip = timebench_dir / "TimeBench-full-19000.zip"
        subset_dir = timebench_dir / "TimeBench-subset-7553"

        if full_zip.exists():
            return self._load_timebench_from_zip(full_zip, task)
        if subset_dir.exists():
            return self._load_timebench_from_dir(subset_dir, task)
        raise FileNotFoundError(
            f"TIMEBENCH data not found under {timebench_dir}. "
            "Expected TimeBench-full-19000.zip or TimeBench-subset-7553/. "
            "Run scripts/download_datasets.py first."
        )

    def _load_timebench_from_zip(
        self,
        zip_path: Path,
        task_filter: str | None,
    ) -> list[TemporalExample]:
        """Read TIMEBENCH directly from the full zip archive."""
        import zipfile as _zipfile
        examples: list[TemporalExample] = []

        # task_filter is compared against lowercased folder name
        task_filter_lower = task_filter.lower() if task_filter else None

        with _zipfile.ZipFile(zip_path) as z:
            jsonl_names = sorted(n for n in z.namelist() if n.endswith(".jsonl"))

            if not jsonl_names:
                raise FileNotFoundError(f"No .jsonl files found inside {zip_path}")

            available = sorted(set(
                n.split("/")[2] for n in jsonl_names if len(n.split("/")) >= 3
            ))

            if task_filter_lower and task_filter_lower not in [t.lower() for t in available]:
                raise ValueError(
                    f"Task {task_filter!r} not found in TIMEBENCH. "
                    f"Available: {available}"
                )

            for name in jsonl_names:
                parts = name.split("/")
                if len(parts) < 3:
                    continue
                task_name = parts[2]     # e.g. 'timeqa', 'date_arith', 'nli'
                file_stem = parts[-1].replace(".jsonl", "")

                if task_filter_lower and task_name.lower() != task_filter_lower:
                    continue

                with z.open(name) as f:
                    for line_no, raw_line in enumerate(f):
                        raw_line = raw_line.strip()
                        if not raw_line:
                            continue
                        item = json.loads(raw_line)
                        examples.append(
                            self._timebench_item_to_example(item, task_name, file_stem, line_no)
                        )
        return examples

    def _load_timebench_from_dir(
        self,
        subset_dir: Path,
        task_filter: str | None,
    ) -> list[TemporalExample]:
        """Read TIMEBENCH from an extracted subset directory (fallback)."""
        if task_filter is not None:
            task_dir = subset_dir / task_filter
            if not task_dir.exists():
                raise ValueError(
                    f"Task {task_filter!r} not found in TIMEBENCH. "
                    f"Available: {self.available_tasks('timebench')}"
                )
            jsonl_files = sorted(task_dir.glob("*.jsonl"))
        else:
            jsonl_files = sorted(subset_dir.rglob("*.jsonl"))

        examples: list[TemporalExample] = []
        for jsonl_path in jsonl_files:
            task_name = jsonl_path.parent.name
            file_stem = jsonl_path.stem
            with open(jsonl_path, encoding="utf-8") as f:
                for line_no, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    examples.append(
                        self._timebench_item_to_example(item, task_name, file_stem, line_no)
                    )
        return examples

    # ------------------------------------------------------------------
    # TRAM
    # ------------------------------------------------------------------

    # Map zip stem → temporal_type
    _TRAM_TYPE_MAP: dict[str, str] = {
        "nli_mcq": "temporal_nli",
        "nli_saq": "temporal_nli",
        "duration": "duration",
        "ordering": "temporal_ordering",
        "arithmetic": "arithmetic",
        "relation": "temporal_relation",
        "frequency": "frequency",
        "causality": "causality",
        "ambiguity_resolution": "ambiguity",
        "typical_time": "typical_time",
        "storytelling": "storytelling",
    }

    def _tram_csv_to_example(
        self,
        idx: int,
        row: dict[str, str],
        task: str,
        ttype: str,
    ) -> TemporalExample:
        """Convert a TRAM CSV row to a TemporalExample."""
        question = row.get("Question", "").strip()
        # Some NLI tasks have Premise + Hypothesis instead of just Question
        if not question:
            premise = row.get("Premise", "")
            hyp = row.get("Hypothesis", "")
            question = row.get("Question", "")
            if not question and premise:
                question = hyp  # hypothesis is what we're evaluating

        # Collect options (A, B, C, D …)
        option_keys = [k for k in ("Option A", "Option B", "Option C", "Option D") if k in row and row[k].strip()]
        choices = [row[k].strip() for k in option_keys] if option_keys else None

        # Answer: letter → actual option text when possible
        answer_letter = row.get("Answer", "").strip()
        answer: str
        letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
        if choices and answer_letter.upper() in letter_to_idx:
            cidx = letter_to_idx[answer_letter.upper()]
            answer = choices[cidx] if cidx < len(choices) else answer_letter
        else:
            answer = answer_letter

        # Context: for relation / NLI tasks use Premise field
        context = row.get("Premise", row.get("Source", "")).strip()

        difficulty = row.get("Category", row.get("Difficulty", None))

        metadata = {
            k: v for k, v in row.items()
            if k not in ("Question", "Option A", "Option B", "Option C", "Option D", "Answer", "Premise", "Category")
            and v.strip()
        }

        return TemporalExample(
            id=f"tram-{task}-{idx}",
            question=question,
            context=context,
            answer=answer,
            choices=choices,
            temporal_type=ttype,
            difficulty=difficulty if isinstance(difficulty, str) and difficulty else None,
            source="tram",
            task=task,
            metadata=metadata,
        )

    def _load_tram(self, task: str | None, split: str) -> list[TemporalExample]:
        """Load TRAM benchmark tasks (CSV files inside zip archives)."""
        datasets_dir = self.data_dir / "tram" / "TRAM-Benchmark" / "datasets"
        if not datasets_dir.exists():
            raise FileNotFoundError(
                f"TRAM data not found at {datasets_dir}. "
                "Run scripts/download_datasets.py first."
            )

        zip_files: list[Path]
        if task is not None:
            zip_path = datasets_dir / f"{task}.zip"
            if not zip_path.exists():
                raise ValueError(
                    f"Task {task!r} not found in TRAM. "
                    f"Available: {self.available_tasks('tram')}"
                )
            zip_files = [zip_path]
        else:
            zip_files = sorted(
                p for p in datasets_dir.glob("*.zip")
                if "shots" not in p.stem
            )

        examples: list[TemporalExample] = []
        for zip_path in zip_files:
            task_name = zip_path.stem
            ttype = self._TRAM_TYPE_MAP.get(task_name, "temporal_reasoning")

            with zipfile.ZipFile(zip_path) as z:
                csv_names = [
                    n for n in z.namelist()
                    if n.endswith(".csv") and "shots" not in n
                ]
                for csv_name in csv_names:
                    with z.open(csv_name) as f:
                        text = f.read().decode("utf-8", errors="replace")
                    reader = csv.DictReader(io.StringIO(text))
                    for row_idx, row in enumerate(reader):
                        ex = self._tram_csv_to_example(
                            idx=len(examples) + row_idx,
                            row=row,
                            task=task_name,
                            ttype=ttype,
                        )
                        examples.append(ex)
        return examples


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def load_all_benchmarks(
    data_dir: str = "./data/benchmarks",
    time_split: str = "all",
) -> list[TemporalExample]:
    """Load every available benchmark and return a combined list.

    By default loads both ``TIME_Newest.json`` and ``TIME-Lite.json``
    (``time_split='all'``).  Pass ``'newest'`` or ``'lite'`` to restrict.
    """
    loader = BenchmarkLoader(data_dir)
    combined: list[TemporalExample] = []
    errors: list[str] = []

    for benchmark in BenchmarkLoader.SUPPORTED_BENCHMARKS:
        try:
            split = time_split if benchmark == "time" else "test"
            examples = loader.load(benchmark, split=split)
            combined.extend(examples)
        except FileNotFoundError as exc:
            errors.append(f"{benchmark}: {exc}")

    if errors:
        import warnings
        warnings.warn(
            "Some benchmarks could not be loaded:\n" + "\n".join(errors),
            stacklevel=2,
        )
    return combined


def get_benchmark_stats(examples: list[TemporalExample]) -> dict:
    """Return a summary dictionary of example counts by source / task / type."""
    from collections import Counter
    return {
        "total": len(examples),
        "by_source": dict(Counter(e.source for e in examples)),
        "by_task": dict(Counter(e.task for e in examples)),
        "by_temporal_type": dict(Counter(e.temporal_type for e in examples)),
    }
