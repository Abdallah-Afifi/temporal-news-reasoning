"""Unified benchmark data loader for TIME, TIMEBENCH, and TRAM.

Loads benchmark files downloaded by ``scripts/download_datasets.py`` into a
standardized :class:`TemporalExample` format used by the evaluation harness
and baseline/fine-tuned evaluation runners.

Expected on-disk layout (created by ``scripts/download_datasets.py``)::

    data/benchmarks/time/TIME/...              (HF snapshot: parquet/json/jsonl)
    data/benchmarks/time/TIME-Lite/...
    data/benchmarks/timebench/TimeBench/...    (git clone)
    data/benchmarks/tram/TRAM-Benchmark/...    (git clone)

The loader is intentionally format-tolerant: benchmark sources package their
data differently (HF parquet snapshots vs. task-JSON git repos), so records
are normalized through a flexible key-mapping. Files inside ``.git`` working
directories and obvious metadata files (README, package.json, ...) are
skipped.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standardized example format
# ---------------------------------------------------------------------------

@dataclass
class TemporalExample:
    """A single benchmark question in standardized form.

    Attributes:
        id: Stable identifier (used for resume/dedup in eval runners).
        question: Question text.
        answer: Gold answer — string or list of acceptable answers.
        context: Optional supporting passage.
        choices: Optional multiple-choice options.
        temporal_type: Temporal reasoning category (for per-category metrics).
        task: Sub-task name within the benchmark.
        source: Benchmark / source-dataset label.
        difficulty: Optional difficulty label.
        split: Source split, when identifiable from the file layout.
    """

    id: str
    question: str
    answer: Any
    context: str = ""
    choices: Optional[list[str]] = None
    temporal_type: Optional[str] = None
    task: Optional[str] = None
    source: Optional[str] = None
    difficulty: Optional[str] = None
    split: Optional[str] = None
    metadata: Optional[dict] = None


# ---------------------------------------------------------------------------
# Flexible field mapping
# ---------------------------------------------------------------------------

QUESTION_KEYS = ("question", "query", "input", "prompt")
CONTEXT_KEYS = ("context", "passage", "article", "background", "paragraph")
ANSWER_KEYS = (
    "answer", "answers", "gold", "gold answer", "gold_answer",
    "targets", "label", "output", "final_answers",
)
CHOICE_KEYS = ("choices", "options", "candidates")
CATEGORY_KEYS = ("temporal_type", "category", "type", "task_type", "reasoning_type")
TASK_KEYS = ("task", "task_name", "subtask", "dataset_name")
ID_KEYS = ("id", "example_id", "uid", "index", "idx")
DIFFICULTY_KEYS = ("difficulty", "level")

_IGNORED_FILE_PATTERNS = (
    re.compile(r"/\.git(/|$)"),
    re.compile(r"(^|/)(README|LICENSE|package|dataset_infos|\.ds_store)", re.IGNORECASE),
    re.compile(r"\.(py|md|txt|sh|yaml|yml|toml|cfg|ini|git|lock)$", re.IGNORECASE),
)


def _first_value(record: dict, keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    # case-insensitive fallback
    lowered = {k.lower(): v for k, v in record.items()}
    for key in keys:
        if lowered.get(key) is not None:
            return lowered[key]
    return None


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return "\n".join(str(v).strip() for v in value if str(v).strip())
    return str(value).strip()


def _infer_split_from_path(path: Path) -> Optional[str]:
    """Guess the split (train/dev/test) from file/directory names, if any.

    Tokens are split on '-', '_', '.' so HF names like
    ``test-00000-of-00001.parquet`` or ``dev_split.json`` resolve correctly.
    The file name takes priority over its directories.
    """
    for part in reversed(path.parts):
        for token in re.split(r"[-_.]", part.lower()):
            if token in ("dev", "val"):
                return "validation"
            if token in ("train", "validation", "test"):
                return token
    return None


def _record_to_example(
    record: dict,
    benchmark: str,
    default_split: Optional[str],
    fallback_id: str,
) -> Optional[TemporalExample]:
    """Normalize one raw record; returns None if it has no question/answer."""
    question = _as_text(_first_value(record, QUESTION_KEYS))
    if not question:
        return None

    answer = _first_value(record, ANSWER_KEYS)
    if answer is None:
        return None

    choices_raw = _first_value(record, CHOICE_KEYS)
    choices: Optional[list[str]] = None
    if isinstance(choices_raw, (list, tuple)) and choices_raw:
        choices = [str(c) for c in choices_raw]
    elif isinstance(choices_raw, str) and "|" in choices_raw:
        choices = [c.strip() for c in choices_raw.split("|") if c.strip()]

    # Some benchmarks (e.g. TIME) embed the MCQ options directly in the
    # question text as "A. ... / B. ... / C. ... / D. ..." lines. Split
    # them out so the prompt builder can format them and gold letters can
    # be mapped to option text by the multi-gold scorer.
    if not choices and "\n" in question:
        lines = question.split("\n")
        options: list[str] = []
        for line in lines[1:]:
            stripped = line.strip()
            if len(stripped) > 2 and stripped[0] in "ABCD" and stripped[1] in ".):":
                options.append(stripped[2:].strip())
        if len(options) >= 2:
            question = lines[0].strip()
            choices = options

    return TemporalExample(
        id=str(_first_value(record, ID_KEYS) or fallback_id),
        question=question,
        answer=answer,
        context=_as_text(_first_value(record, CONTEXT_KEYS)),
        choices=choices,
        temporal_type=(
            _as_text(_first_value(record, CATEGORY_KEYS)) or None
        ),
        task=_as_text(_first_value(record, TASK_KEYS)) or None,
        source=str(record.get("source_dataset") or benchmark),
        difficulty=_as_text(_first_value(record, DIFFICULTY_KEYS)) or None,
        split=str(record.get("split") or default_split or ""),
        # Carry the fields that identify the EVALUATION CONDITION, not just
        # the item (added 2026-09-12). TIME ships `Setting` — `base` (gold
        # context, 40.4% of rows) vs `bm25`/`vector`/`hybrid` (a retriever
        # supplied the context, 59.6%) — and dropping it made every TIME
        # number a pooled average over four conditions ~23pp apart, in which
        # the fine-tuning effect changes SIGN (audit 2026-09-12 §0a-bis:
        # v7c +2.76pp on `base`, −3.0pp on retrieved, pooled −0.64pp).
        # Preserved here so predictions carry it and analyses can stratify
        # without joining back to TIME_Newest.json.
        metadata={k: record[k] for k in ("Setting", "Dataset Name", "Level")
                  if k in record and record[k] not in (None, "")} or None,
    )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class BenchmarkLoader:
    """Load and standardize benchmark datasets."""

    SUPPORTED_BENCHMARKS = ["time", "timebench", "tram"]

    # Some benchmark directories ship more than one distribution of the same
    # data. TIME ships TIME_Newest.json (104,951 items) alongside
    # TIME-Lite.json (1,549), and every TIME-Lite question is a duplicate of a
    # TIME_Newest question -- 118 of them carrying a *different* gold answer.
    # Loading both double-counts 1.45% of the benchmark and scores 118 items
    # against contradictory references, so name the canonical file per
    # benchmark and use only that.
    CANONICAL_FILES: dict[str, str] = {"time": "TIME_Newest.json"}

    def __init__(self, data_dir: str = "./data/benchmarks"):
        self.data_dir = Path(data_dir)

    # ------------------------------------------------------------------

    def _candidate_files(self, benchmark: str) -> list[Path]:
        bench_dir = self.data_dir / benchmark
        if not bench_dir.is_dir():
            return []
        files: list[Path] = []
        for path in sorted(bench_dir.rglob("*")):
            if not path.is_file():
                continue
            posix = path.as_posix()
            if any(p.search(posix) for p in _IGNORED_FILE_PATTERNS):
                continue
            if path.suffix.lower() in {".json", ".jsonl", ".parquet"}:
                files.append(path)
        return files

    @staticmethod
    def _read_file(path: Path) -> list[dict]:
        """Read a .json / .jsonl / .parquet file into a list of dicts."""
        suffix = path.suffix.lower()
        try:
            if suffix == ".jsonl":
                records = []
                with path.open(encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            records.append(json.loads(line))
                return [r for r in records if isinstance(r, dict)]

            if suffix == ".json":
                with path.open(encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return [r for r in data if isinstance(r, dict)]
                if isinstance(data, dict):
                    # {split: [records]} / {task: [records]} style containers
                    nested = [
                        r for v in data.values() if isinstance(v, list)
                        for r in v if isinstance(r, dict)
                    ]
                    return nested or ([data] if any(
                        _first_value(data, QUESTION_KEYS) for _ in [0]
                    ) else [])

            if suffix == ".parquet":
                try:
                    import pandas as pd
                except ImportError:
                    logger.warning("pandas not installed — skipping %s", path)
                    return []
                return pd.read_parquet(path).to_dict(orient="records")
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read %s: %s", path, exc)
        return []

    # ------------------------------------------------------------------

    def load(
        self,
        benchmark: str,
        task: str | None = None,
        split: str | None = "test",
    ) -> list[TemporalExample]:
        """Load a benchmark dataset.

        Args:
            benchmark: One of 'time', 'timebench', 'tram'.
            task: Optional sub-task filter (matches record task or file stem).
            split: Preferred split ('test' by default). Files whose names
                identify a split are filtered to it; if no file identifies
                the requested split, ALL files are loaded (with a warning),
                matching the historical behavior of loading everything.

        Returns:
            List of examples in standardized format.
        """
        if benchmark not in self.SUPPORTED_BENCHMARKS:
            raise ValueError(
                f"Unknown benchmark: {benchmark}. "
                f"Supported: {self.SUPPORTED_BENCHMARKS}"
            )

        # TimeBench / TRAM ship as zip archives with per-task layouts that
        # the generic field-mapper cannot reconstruct (task folders inside
        # the full zip; CSVs with Option A-D columns). Use the dedicated
        # loaders ported from the reference implementation so the eval sets
        # match the committed baselines exactly.
        if benchmark == "timebench":
            return self._load_timebench(task)
        if benchmark == "tram":
            return self._load_tram(task)

        files = self._candidate_files(benchmark)
        canonical_name = self.CANONICAL_FILES.get(benchmark)
        if canonical_name:
            canonical = [f for f in files if f.name == canonical_name]
            if canonical:
                if len(canonical) < len(files):
                    logger.info(
                        "Benchmark '%s': using canonical file %s "
                        "(ignoring %d duplicate distribution file(s)).",
                        benchmark, canonical_name, len(files) - len(canonical),
                    )
                files = canonical
            else:
                logger.warning(
                    "Benchmark '%s': canonical file %s not found; "
                    "falling back to all %d discovered file(s).",
                    benchmark, canonical_name, len(files),
                )
        if not files:
            raise FileNotFoundError(
                f"No benchmark data files found under {self.data_dir / benchmark}. "
                f"Download them first: python scripts/download_datasets.py {benchmark}"
            )

        split_map: dict[Path, Optional[str]] = {
            f: _infer_split_from_path(f) for f in files
        }
        selected: list[Path]
        if split and any(s == split for s in split_map.values()):
            selected = [f for f, s in split_map.items() if s == split]
        else:
            selected = files
            if split:
                logger.warning(
                    "No files identified with split '%s' for benchmark '%s'; "
                    "loading all files (%d).", split, benchmark, len(files),
                )

        examples: list[TemporalExample] = []
        for path in selected:
            records = self._read_file(path)
            file_split = split_map.get(path)
            file_stem = path.stem
            for i, record in enumerate(records):
                if task is not None:
                    record_task = (
                        _as_text(_first_value(record, TASK_KEYS))
                        or _as_text(_first_value(record, CATEGORY_KEYS))
                    )
                    if task.lower() not in (record_task or "").lower() and task.lower() not in file_stem.lower():
                        continue
                example = _record_to_example(
                    record,
                    benchmark=benchmark,
                    default_split=file_split,
                    fallback_id=f"{path.name}-{i}",
                )
                if example is not None:
                    # Dataset defect (audit 2026-09-09 H3): 12 TIME Timeline
                    # rows ship an empty-string gold in the source. Scored
                    # items need a non-empty gold; skip for all arms alike.
                    golds = (
                        example.answer if isinstance(example.answer, list)
                        else [example.answer]
                    )
                    if golds and all(not str(g).strip() for g in golds):
                        logger.warning(
                            "Benchmark '%s': skipping empty-gold item %s "
                            "(gold is blank in source data)", benchmark, example.id,
                        )
                        continue
                    if example.task is None:
                        example.task = file_stem
                    examples.append(example)

        if not examples:
            raise FileNotFoundError(
                f"Benchmark '{benchmark}' files under {self.data_dir / benchmark} "
                f"contained no usable QA records "
                f"({'task filter: ' + task if task else 'no filter'})."
            )

        logger.info(
            "Benchmark '%s': loaded %d examples from %d files%s.",
            benchmark, len(examples), len(selected),
            f" (task={task})" if task else "",
        )
        return examples

    # ------------------------------------------------------------------
    # TimeBench (full zip / extracted subset) — ported from the reference
    # loader so the eval set matches the committed baselines exactly.
    # ------------------------------------------------------------------

    _TIMEBENCH_TYPE_MAP: dict[str, str] = {
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

    _TIMEBENCH_DIFFICULTY: dict[str, str] = {
        "easy": "easy", "hard": "hard",
        "l1": "level_1", "l2": "level_2", "l3": "level_3",
        "f2": "format_2",
        "cs1": "cs1", "cs2": "cs2", "cs3": "cs3",
        "counterfactual": "counterfactual",
        "order": "order", "scope": "scope",
        "event-event": "event-event", "event-time": "event-time",
        "s1": "s1", "s2": "s2", "s3": "s3",
    }

    def _load_timebench(self, task: str | None) -> list[TemporalExample]:
        """Load TIMEBENCH — prefers TimeBench-full-19000.zip, falls back to
        the extracted TimeBench-subset-7553 directory."""
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
        self, zip_path: Path, task_filter: str | None,
    ) -> list[TemporalExample]:
        import zipfile as _zipfile

        task_filter_lower = task_filter.lower() if task_filter else None
        examples: list[TemporalExample] = []

        with _zipfile.ZipFile(zip_path) as z:
            jsonl_names = sorted(n for n in z.namelist() if n.endswith(".jsonl"))
            if not jsonl_names:
                raise FileNotFoundError(f"No .jsonl files found inside {zip_path}")

            for name in jsonl_names:
                parts = name.split("/")
                if len(parts) < 3:
                    continue
                task_name = parts[2]
                file_stem = parts[-1].replace(".jsonl", "")

                if task_filter_lower and task_name.lower() != task_filter_lower:
                    continue

                with z.open(name) as f:
                    for line_no, raw_line in enumerate(f):
                        raw_line = raw_line.strip()
                        if not raw_line:
                            continue
                        item = json.loads(raw_line)
                        example = self._timebench_item_to_example(
                            item, task_name, file_stem, line_no
                        )
                        if example is not None:
                            examples.append(example)
        return examples

    def _load_timebench_from_dir(
        self, subset_dir: Path, task_filter: str | None,
    ) -> list[TemporalExample]:
        if task_filter is not None:
            task_dir = subset_dir / task_filter
            if not task_dir.exists():
                raise ValueError(
                    f"Task {task_filter!r} not found in TIMEBENCH subset."
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
                example = self._timebench_item_to_example(
                    item, task_name, file_stem, line_no
                )
                if example is not None:
                    examples.append(example)
        return examples

    @staticmethod
    def _parse_timebench_answer(item: dict, task: str = "") -> Any:
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

    @staticmethod
    def _parse_timebench_context(item: dict) -> str:
        parts = []
        for key in ("context", "fact_context", "Premise"):
            val = item.get(key, "")
            if isinstance(val, list):
                val = " ".join(str(v) for v in val)
            if val:
                parts.append(str(val))
        return " ".join(parts)

    def _infer_timebench_difficulty(self, stem: str) -> Optional[str]:
        stem_lower = stem.lower()
        for key, val in self._TIMEBENCH_DIFFICULTY.items():
            if key in stem_lower:
                return val
        return None

    def _timebench_item_to_example(
        self, item: dict, task_name: str, file_stem: str, line_no: int,
    ) -> TemporalExample | None:
        task_key = task_name.lower()
        ttype = self._TIMEBENCH_TYPE_MAP.get(
            task_name, self._TIMEBENCH_TYPE_MAP.get(task_key, "temporal_reasoning")
        )
        difficulty = self._infer_timebench_difficulty(file_stem)
        item_id = item.get("idx", item.get("qid", f"{file_stem}-{line_no}"))

        # Dataset defect (audit 2026-09-09 H3): 113 rows in TimeBench-full's
        # timeqa_easy_500 / timeqa_hard_500 ship answer == [''] — the gold is
        # an empty string in the SOURCE data, unrecoverable by parsing. Such
        # items are unscoreable (an empty prediction scores correct on them;
        # a real answer scores wrong) and are skipped for every arm alike.
        if task_key != "situatedgen":
            ans = self._parse_timebench_answer(item, task_key)
            golds = ans if isinstance(ans, list) else [ans]
            if golds and all(not str(g).strip() for g in golds):
                logger.warning(
                    "TimeBench: skipping empty-gold item %s/%s line %d "
                    "(answer == [''] in source data)", task_name, file_stem, line_no,
                )
                return None

        # SituatedGen
        if task_key == "situatedgen":
            question = "Generate a temporally-grounded statement using: " + ", ".join(
                item.get("keywords", [])
            )
            return TemporalExample(
                id=f"timebench-{task_key}-{file_stem}-{item_id}",
                question=question,
                answer=item.get("statement", ""),
                choices=None,
                temporal_type=ttype,
                difficulty=difficulty,
                source="timebench",
                task=task_name,
                metadata={k: v for k, v in item.items() if k not in ("keywords", "statement")},
            )

        # TRACIE / NLI (Premise + Hypothesis + Label)
        if "Hypothesis" in item:
            return TemporalExample(
                id=f"timebench-{task_key}-{file_stem}-{item_id}",
                question=str(item.get("Hypothesis", "")),
                context=str(item.get("Premise", "")),
                answer=self._parse_timebench_answer(item, task_key),
                choices=None,
                temporal_type=ttype,
                difficulty=difficulty,
                source="timebench",
                task=task_name,
                metadata={k: v for k, v in item.items() if k not in ("Hypothesis", "Premise", "Label")},
            )

        # TimeDial (context is the dialogue, question is implicit)
        if task_key == "timedial":
            return TemporalExample(
                id=f"timebench-{task_key}-{file_stem}-{item_id}",
                question=str(item.get("context", "")),
                answer=self._parse_timebench_answer(item, task_key),
                choices=item.get("options"),
                temporal_type=ttype,
                difficulty=difficulty,
                source="timebench",
                task=task_name,
                metadata={k: v for k, v in item.items() if k not in ("context", "options", "labels", "qid")},
            )

        # General case (TimeQA, MenatQA, TempReason, DurationQA, McTaco, date_arith)
        return TemporalExample(
            id=f"timebench-{task_key}-{file_stem}-{item_id}",
            question=str(item.get("question", item.get("statement", ""))),
            context=self._parse_timebench_context(item),
            answer=self._parse_timebench_answer(item, task_key),
            choices=item.get("options"),
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

    # ------------------------------------------------------------------
    # TRAM (CSV files inside per-task zips) — ported from the reference
    # loader so the eval set matches the committed baselines exactly.
    # ------------------------------------------------------------------

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

    def _load_tram(self, task: str | None) -> list[TemporalExample]:
        import csv
        import io
        import zipfile as _zipfile

        datasets_dir = self.data_dir / "tram" / "TRAM-Benchmark" / "datasets"
        if not datasets_dir.exists():
            raise FileNotFoundError(
                f"TRAM data not found at {datasets_dir}. "
                "Run scripts/download_datasets.py first."
            )

        if task is not None:
            zip_path = datasets_dir / f"{task}.zip"
            if not zip_path.exists():
                raise ValueError(f"Task {task!r} not found in TRAM.")
            zip_files = [zip_path]
        else:
            zip_files = sorted(
                p for p in datasets_dir.glob("*.zip") if "shots" not in p.stem
            )

        examples: list[TemporalExample] = []
        for zip_path in zip_files:
            task_name = zip_path.stem
            ttype = self._TRAM_TYPE_MAP.get(task_name, "temporal_reasoning")

            with _zipfile.ZipFile(zip_path) as z:
                csv_names = [
                    n for n in z.namelist()
                    if n.endswith(".csv") and "shots" not in n
                ]
                for csv_name in csv_names:
                    with z.open(csv_name) as f:
                        text = f.read().decode("utf-8", errors="replace")
                    reader = csv.DictReader(io.StringIO(text))
                    for row in reader:
                        # The index MUST be a single running counter.
                        # It used to be `len(examples) + row_idx`, which
                        # double-counts: within one CSV it produced 0, 2, 4,
                        # 6 ... and the next CSV restarted at len(examples),
                        # landing back inside the range the previous file had
                        # already used. That collided 31,626 ids (3.2%).
                        #
                        # This was not cosmetic. `run_baselines.py` keys its
                        # resume set on `id`, so every duplicate id after the
                        # first would be skipped as "already completed" and a
                        # full TRAM run would silently evaluate ~31,626 fewer
                        # items than it reported. Fixed 2026-09-07, before
                        # TRAM's first run — no stored TRAM predictions exist,
                        # so nothing is invalidated.
                        examples.append(
                            self._tram_csv_to_example(
                                len(examples), row, task_name, ttype
                            )
                        )
        if not examples:
            raise FileNotFoundError(
                f"TRAM zips under {datasets_dir} contained no usable records."
            )
        return examples

    @staticmethod
    def _tram_csv_to_example(
        idx: int, row: dict, task: str, ttype: str,
    ) -> TemporalExample:
        # PROMPT-CONTENT BUG, fixed 2026-09-12 (audit). The old code was:
        #
        #     question = row["Question"]
        #     if not question:            # never true -- see below
        #         question = row["Hypothesis"]
        #
        # Every TRAM NLI row HAS a "Question" column, but it holds a constant
        # instruction ("Do the hypothesis and premise logically entail each
        # other, contradict each other, or remain undetermined?"), so the
        # fallback never fired and the `Hypothesis` -- half of the problem --
        # never reached the prompt. Three consecutive rows sharing a premise
        # produced BYTE-IDENTICAL prompts with different golds. Measured:
        # nli_mcq + nli_saq = 564,268 items (57.5% of TRAM) had only 3 unique
        # (question, answer) pairs each, and every arm scored 30-35% on a
        # 3-way task whose chance rate is 33.3%.
        #
        # `storytelling_mcq` (67,204 items) had the same defect via a
        # different column: its passage is in `Story`, which was never read,
        # while the question asks "which of the two endings is the most
        # plausible ending to the story?".
        #
        # And `context` fell back to the `Source` column, so `relation_*`
        # (204,914 items) rendered `Context: TempEval-3` -- a dataset name --
        # as if it were the passage.
        question = row.get("Question", "").strip()
        premise = row.get("Premise", "").strip()
        hypothesis = row.get("Hypothesis", "").strip()
        story = row.get("Story", "").strip()

        # The hypothesis IS the classified sentence: `_nli_instruction` in
        # run_baselines.py renders "Premise: <context> / Hypothesis:
        # <question>" for choice-less NLI rows, which is the contract this
        # restores. The constant instruction column is kept in `metadata`.
        if hypothesis:
            question = hypothesis

        option_keys = [
            k for k in ("Option A", "Option B", "Option C", "Option D")
            if k in row and row[k].strip()
        ]
        choices = [row[k].strip() for k in option_keys] if option_keys else None

        # Answer: letter → actual option text when possible
        answer_letter = row.get("Answer", "").strip()
        letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
        if choices and answer_letter.upper() in letter_to_idx:
            cidx = letter_to_idx[answer_letter.upper()]
            answer = choices[cidx] if cidx < len(choices) else answer_letter
        else:
            answer = answer_letter

        difficulty = row.get("Category", row.get("Difficulty", None))

        return TemporalExample(
            id=f"tram-{task}-{idx}",
            question=question,
            # Premise or Story -- never `Source`, which is provenance
            # ("SNLI", "TempEval-3", "ROC"), not text.
            context=premise or story,
            answer=answer,
            choices=choices,
            temporal_type=ttype,
            difficulty=difficulty if isinstance(difficulty, str) and difficulty else None,
            source="tram",
            task=task,
            metadata={
                k: v for k, v in row.items()
                if k not in ("Question", "Option A", "Option B", "Option C",
                             "Option D", "Answer", "Premise", "Category")
                and v.strip()
            },
        )
