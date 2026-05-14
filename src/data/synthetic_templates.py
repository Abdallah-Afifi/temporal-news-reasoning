"""Synthetic training data generation for news temporal reasoning."""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from itertools import cycle
from pathlib import Path
from typing import Any, Iterable

from dateutil import parser as date_parser

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover - optional dependency path
    load_dataset = None

_DATE_PATTERNS = [
    re.compile(r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b"),
    re.compile(
        r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{4})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b((?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{1,2},?\s+\d{4})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b((?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{4})\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b((?:19|20)\d{2})\b"),
]


@dataclass(frozen=True)
class NewsArticle:
    source: str
    article_id: str
    title: str
    text: str
    url: str | None = None
    date_publish: str | None = None


@dataclass(frozen=True)
class DateCandidate:
    expression: str
    parsed_date: date | None
    sentence: str


class SyntheticDataGenerator:
    """Generate synthetic temporal reasoning training examples."""

    CATEGORY_WEIGHTS = {
        "arithmetic": 0.18,
        "duration": 0.14,
        "temporal_ordering": 0.18,
        "implicit_reference": 0.18,
        "temporal_nli": 0.16,
        "timeline_construction": 0.16,
    }

    def __init__(
        self,
        data_root: str | Path = "/home/abdallah/Documents/Thesis/Datasets",
        seed: int = 13,
        max_articles_per_source: int = 4000,
    ) -> None:
        self.data_root = Path(data_root)
        self.random = random.Random(seed)
        self.max_articles_per_source = max_articles_per_source

    def generate(
        self,
        num_examples: int = 10000,
        sources: Iterable[str] | None = None,
        include_cot: bool = True,
    ) -> list[dict[str, Any]]:
        """Generate synthetic temporal reasoning examples."""
        articles = self.load_articles(sources=sources)
        if not articles:
            raise ValueError("No news articles found for synthetic generation.")

        examples: list[dict[str, Any]] = []
        source_cycle = cycle(articles)
        while len(examples) < num_examples:
            article = next(source_cycle)
            generated = self.generate_from_article(article, include_cot=include_cot)
            if not generated:
                continue
            self.random.shuffle(generated)
            for example in generated:
                examples.append(example)
                if len(examples) >= num_examples:
                    break
        return examples

    def load_articles(self, sources: Iterable[str] | None = None) -> list[NewsArticle]:
        """Load articles from the local news corpora."""
        allowed = {s.lower() for s in sources} if sources else {"ccnews", "cnn_dailymail", "cnn_stories"}
        articles: list[NewsArticle] = []
        if "ccnews" in allowed:
            articles.extend(self._load_ccnews())
        if "cnn_dailymail" in allowed:
            articles.extend(self._load_cnn_dailymail())
        if "cnn_stories" in allowed:
            articles.extend(self._load_cnn_stories())
        return articles

    def generate_from_article(
        self,
        article: NewsArticle,
        include_cot: bool = True,
    ) -> list[dict[str, Any]]:
        """Create multiple training examples from one article."""
        sentences = self._split_sentences(article.text)
        context = self._build_context(article, sentences)
        candidates = self._extract_date_candidates(article)
        examples: list[dict[str, Any]] = []

        if candidates:
            examples.append(
                self._build_date_extraction_example(article, context, candidates[0], include_cot)
            )

        extras: list[dict[str, Any]] = []
        if len(candidates) >= 2:
            pair = self.random.sample(candidates, 2)
            extras.append(
                self._build_arithmetic_example(article, context, pair[0], include_cot)
            )
            extras.append(
                self._build_duration_example(article, context, pair[0], pair[1], include_cot)
            )
            extras.append(
                self._build_ordering_example(article, context, pair[0], pair[1], include_cot)
            )
            extras.append(
                self._build_nli_example(article, context, pair[0], pair[1], include_cot)
            )

        if candidates:
            extras.append(
                self._build_implicit_reference_example(article, context, candidates[0], include_cot)
            )

        if len(sentences) >= 3:
            extras.append(
                self._build_timeline_example(article, context, sentences[:3], include_cot)
            )

        self.random.shuffle(extras)
        examples.extend(extras[: max(0, 3 - len(examples))])
        return examples

    def validate_examples(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        """Validate generated examples and return summary statistics."""
        required = {"id", "category", "question", "answer", "rationale", "source_id", "source_dataset"}
        missing = sum(1 for example in examples if not required.issubset(example))
        duplicates = len(examples) - len(
            {self._fingerprint(example) for example in examples}
        )
        counts = Counter(example.get("category", "unknown") for example in examples)
        benchmark_counts = Counter()
        for example in examples:
            for benchmark in example.get("benchmark_targets", []):
                benchmark_counts[benchmark] += 1

        answer_with_date = sum(
            1 for example in examples if self._extract_temporal_expressions(example.get("answer", ""))
        )

        return {
            "num_examples": len(examples),
            "missing_required_fields": missing,
            "duplicate_examples": duplicates,
            "category_distribution": dict(counts),
            "benchmark_distribution": dict(benchmark_counts),
            "answers_with_temporal_expressions": answer_with_date,
            "duplicate_rate": duplicates / len(examples) if examples else 0.0,
        }

    def split_examples(
        self,
        examples: list[dict[str, Any]],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> dict[str, list[dict[str, Any]]]:
        """Split by source_id so no article leaks across splits."""
        if round(train_ratio + val_ratio + test_ratio, 6) != 1.0:
            raise ValueError("Split ratios must sum to 1.0")

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for example in examples:
            grouped[example["source_id"]].append(example)

        source_ids = list(grouped)
        self.random.shuffle(source_ids)
        total = len(source_ids)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        splits = {
            "train": source_ids[:train_end],
            "validation": source_ids[train_end:val_end],
            "test": source_ids[val_end:],
        }

        return {
            name: [item for source_id in source_list for item in grouped[source_id]]
            for name, source_list in splits.items()
        }

    def save_jsonl(self, examples: list[dict[str, Any]], output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

    def _load_ccnews(self) -> list[NewsArticle]:
        articles: list[NewsArticle] = []
        for path in sorted(self.data_root.glob("ccnews/ccnews_*_en.jsonl")):
            with open(path, "r", encoding="utf-8") as f:
                for index, line in enumerate(f):
                    if len(articles) >= self.max_articles_per_source:
                        return articles
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    title = str(record.get("title", "")).strip()
                    text = str(record.get("text", "")).strip()
                    if not title or not text:
                        continue
                    articles.append(
                        NewsArticle(
                            source="ccnews",
                            article_id=f"{path.stem}:{index}",
                            title=title,
                            text=text,
                            url=(record.get("url") or None),
                            date_publish=(record.get("date_publish") or None),
                        )
                    )
        return articles

    def _load_cnn_dailymail(self) -> list[NewsArticle]:
        if load_dataset is None:
            return []
        articles: list[NewsArticle] = []
        data_dir = self.data_root / "cnn_dailymail"
        parquet_files = sorted(data_dir.glob("3.0.0/*.parquet")) or sorted(data_dir.glob("2.0.0/*.parquet")) or sorted(data_dir.glob("1.0.0/*.parquet"))
        if not parquet_files:
            return []

        dataset = load_dataset("parquet", data_files={"train": [str(p) for p in parquet_files]}, split="train")
        for index, row in enumerate(dataset):
            if len(articles) >= self.max_articles_per_source:
                break
            article = str(row.get("article", "")).strip()
            highlights = str(row.get("highlights", "")).strip()
            if not article:
                continue
            title = self._derive_title(highlights, article)
            articles.append(
                NewsArticle(
                    source="cnn_dailymail",
                    article_id=str(row.get("id", f"cnn_dailymail:{index}")),
                    title=title,
                    text=article,
                )
            )
        return articles

    def _load_cnn_stories(self) -> list[NewsArticle]:
        articles: list[NewsArticle] = []
        story_root = self.data_root / "cnn_stories"
        if not story_root.exists():
            return articles
        for index, path in enumerate(sorted(story_root.rglob("*.story"))):
            if len(articles) >= self.max_articles_per_source:
                break
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
            if not raw:
                continue
            parts = raw.split("@highlight")
            article = parts[0].strip()
            highlights = " ".join(part.strip() for part in parts[1:] if part.strip())
            if not article:
                continue
            articles.append(
                NewsArticle(
                    source="cnn_stories",
                    article_id=f"{path.stem}:{index}",
                    title=self._derive_title(highlights, article),
                    text=article,
                )
            )
        return articles

    def _extract_date_candidates(self, article: NewsArticle) -> list[DateCandidate]:
        text = f"{article.title}. {article.text}"
        candidates: list[DateCandidate] = []
        for expression in self._extract_temporal_expressions(text):
            parsed = self._parse_date(expression)
            if parsed is None:
                continue
            sentence = self._find_sentence_with_expression(article.text, expression)
            candidates.append(DateCandidate(expression=expression, parsed_date=parsed, sentence=sentence))
        deduped: list[DateCandidate] = []
        seen = set()
        for candidate in candidates:
            key = (candidate.expression.lower(), candidate.parsed_date.isoformat() if candidate.parsed_date else "")
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped

    def _extract_temporal_expressions(self, text: str) -> list[str]:
        found: list[str] = []
        for pattern in _DATE_PATTERNS:
            found.extend(pattern.findall(text))
        unique: list[str] = []
        seen: set[str] = set()
        for expression in found:
            normalised = expression.strip().lower()
            if normalised in seen:
                continue
            seen.add(normalised)
            unique.append(expression.strip())
        return unique

    def _build_date_extraction_example(
        self,
        article: NewsArticle,
        context: str,
        candidate: DateCandidate,
        include_cot: bool,
    ) -> dict[str, Any]:
        answer = candidate.parsed_date.isoformat() if candidate.parsed_date else candidate.expression
        question = f"According to the article, what date is mentioned in the sentence about {self._short_event_hint(candidate.sentence)}?"
        rationale = f"The article explicitly mentions '{candidate.expression}' in the supporting sentence, so the answer is {answer}."
        return self._build_example(
            article=article,
            category="explicit_date_extraction",
            question=question,
            context=context,
            answer=answer,
            rationale=rationale,
            benchmark_targets=["TIME", "TimeBench", "TRAM"],
            include_cot=include_cot,
            temporal_anchor={"expression": candidate.expression, "date": answer},
        )

    def _build_arithmetic_example(
        self,
        article: NewsArticle,
        context: str,
        candidate: DateCandidate,
        include_cot: bool,
    ) -> dict[str, Any]:
        base_date = candidate.parsed_date or date.today()
        offset = self.random.choice([1, 2, 3, 5, 7, 10, 14])
        target_date = base_date + timedelta(days=offset)
        question = (
            f"If the event happened {offset} days after {candidate.expression}, what date was it?"
        )
        rationale = (
            f"{candidate.expression} resolves to {base_date.isoformat() if candidate.parsed_date else candidate.expression}; "
            f"adding {offset} days gives {target_date.isoformat()}."
        )
        return self._build_example(
            article=article,
            category="arithmetic",
            question=question,
            context=context,
            answer=target_date.isoformat(),
            rationale=rationale,
            benchmark_targets=["TIME", "TRAM"],
            include_cot=include_cot,
            temporal_anchor={"base_expression": candidate.expression, "offset_days": str(offset)},
        )

    def _build_duration_example(
        self,
        article: NewsArticle,
        context: str,
        first: DateCandidate,
        second: DateCandidate,
        include_cot: bool,
    ) -> dict[str, Any]:
        earlier, later = sorted([first, second], key=lambda c: c.parsed_date or date.min)
        delta = abs((later.parsed_date - earlier.parsed_date).days) if earlier.parsed_date and later.parsed_date else 0
        question = (
            f"How many days passed between {earlier.expression} and {later.expression} in the article?"
        )
        rationale = (
            f"{earlier.expression} normalizes to {earlier.parsed_date.isoformat() if earlier.parsed_date else earlier.expression} "
            f"and {later.expression} normalizes to {later.parsed_date.isoformat() if later.parsed_date else later.expression}; "
            f"the difference is {delta} days."
        )
        return self._build_example(
            article=article,
            category="duration",
            question=question,
            context=context,
            answer=str(delta),
            rationale=rationale,
            benchmark_targets=["TIME", "TRAM"],
            include_cot=include_cot,
            temporal_anchor={
                "start_expression": earlier.expression,
                "end_expression": later.expression,
                "duration_days": str(delta),
            },
        )

    def _build_ordering_example(
        self,
        article: NewsArticle,
        context: str,
        first: DateCandidate,
        second: DateCandidate,
        include_cot: bool,
    ) -> dict[str, Any]:
        earlier, later = sorted([first, second], key=lambda c: c.parsed_date or date.min)
        question = f"Which happened first in the article: {first.expression} or {second.expression}?"
        answer = first.expression if (first.parsed_date or date.min) <= (second.parsed_date or date.min) else second.expression
        rationale = (
            f"{first.expression} and {second.expression} normalize to different dates, and the earlier date comes first."
        )
        return self._build_example(
            article=article,
            category="temporal_ordering",
            question=question,
            context=context,
            answer=answer,
            rationale=rationale,
            benchmark_targets=["TIME", "TimeBench", "TRAM"],
            include_cot=include_cot,
            temporal_anchor={"first": first.expression, "second": second.expression},
        )

    def _build_nli_example(
        self,
        article: NewsArticle,
        context: str,
        first: DateCandidate,
        second: DateCandidate,
        include_cot: bool,
    ) -> dict[str, Any]:
        earlier, later = sorted([first, second], key=lambda c: c.parsed_date or date.min)
        claim = f"The event tied to {later.expression} happened before the event tied to {earlier.expression}."
        question = (
            f"Given the article, is the claim true, false, or unknown: '{claim}'?"
        )
        answer = "false"
        rationale = (
            f"The article places {earlier.expression} before {later.expression}, so the claim reverses the order and is false."
        )
        return self._build_example(
            article=article,
            category="temporal_nli",
            question=question,
            context=context,
            answer=answer,
            rationale=rationale,
            benchmark_targets=["TimeBench", "TRAM"],
            include_cot=include_cot,
            temporal_anchor={"claim": claim},
        )

    def _build_implicit_reference_example(
        self,
        article: NewsArticle,
        context: str,
        candidate: DateCandidate,
        include_cot: bool,
    ) -> dict[str, Any]:
        base_date = candidate.parsed_date or date.today()
        offset = self.random.choice([1, 2, 3, 7, 14])
        reference_phrase = self._relative_phrase(offset)
        resolved_date = base_date + timedelta(days=offset)
        question = (
            f"If the article says the event happened {reference_phrase} after {candidate.expression}, what date was it?"
        )
        rationale = (
            f"{candidate.expression} resolves to {base_date.isoformat() if candidate.parsed_date else candidate.expression}; "
            f"adding {offset} days gives {resolved_date.isoformat()}."
        )
        return self._build_example(
            article=article,
            category="implicit_reference",
            question=question,
            context=context,
            answer=resolved_date.isoformat(),
            rationale=rationale,
            benchmark_targets=["TIME", "TimeBench"],
            include_cot=include_cot,
            temporal_anchor={"base_expression": candidate.expression, "offset_days": str(offset)},
        )

    def _build_timeline_example(
        self,
        article: NewsArticle,
        context: str,
        sentences: list[str],
        include_cot: bool,
    ) -> dict[str, Any]:
        question = "Put the following events in chronological order as they appear in the article: " + "; ".join(
            f"Event {index + 1}: {self._short_event_hint(sentence)}" for index, sentence in enumerate(sentences)
        )
        answer = " > ".join(f"Event {index + 1}" for index in range(len(sentences)))
        rationale = "The article presents the events in narrative order, so the timeline follows their appearance from first to last."
        return self._build_example(
            article=article,
            category="timeline_construction",
            question=question,
            context=context,
            answer=answer,
            rationale=rationale,
            benchmark_targets=["TIME", "TimeBench", "TRAM"],
            include_cot=include_cot,
            temporal_anchor={"num_events": str(len(sentences))},
        )

    def _build_example(
        self,
        article: NewsArticle,
        category: str,
        question: str,
        context: str,
        answer: str,
        rationale: str,
        benchmark_targets: list[str],
        include_cot: bool,
        temporal_anchor: dict[str, str],
    ) -> dict[str, Any]:
        example_id = f"{article.source}:{article.article_id}:{category}:{self.random.randint(0, 10**9)}"
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
        assistant_response = f"Answer: {answer}"
        if include_cot:
            assistant_response += f"\nReasoning: {rationale}"
        return {
            "id": example_id,
            "source_dataset": article.source,
            "source_id": article.article_id,
            "source_title": article.title,
            "category": category,
            "benchmark_targets": benchmark_targets,
            "question": question,
            "context": context,
            "answer": answer,
            "rationale": rationale,
            "temporal_anchor": temporal_anchor,
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt,
                },
                {
                    "role": "assistant",
                    "content": assistant_response,
                },
            ],
        }

    def _build_context(self, article: NewsArticle, sentences: list[str]) -> str:
        intro = article.title.strip()
        snippet = " ".join(sentences[:4]) if sentences else article.text[:800]
        return f"{intro}. {snippet}"[:1500]

    def _split_sentences(self, text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text.strip()))
        return [sentence.strip() for sentence in sentences if len(sentence.strip()) > 30]

    def _find_sentence_with_expression(self, text: str, expression: str) -> str:
        for sentence in self._split_sentences(text):
            if expression.lower() in sentence.lower():
                return sentence
        return text[:240]

    def _parse_date(self, expression: str) -> date | None:
        expression = expression.strip()
        year_match = re.fullmatch(r"((?:19|20)\d{2})", expression)
        if year_match:
            return date(int(year_match.group(1)), 1, 1)
        try:
            parsed = date_parser.parse(expression, fuzzy=True, default=datetime(1900, 1, 1))
        except Exception:
            return None
        if parsed is None:
            return None
        return parsed.date()

    def _derive_title(self, highlights: str, article: str) -> str:
        candidate = highlights.split(".")[0].strip() if highlights else article.split(".")[0].strip()
        return candidate[:120] if candidate else "Untitled news article"

    def _short_event_hint(self, sentence: str) -> str:
        words = sentence.split()
        return " ".join(words[:8]) if words else "the event"

    def _relative_phrase(self, offset_days: int) -> str:
        if offset_days == 1:
            return "the next day"
        if offset_days == 2:
            return "two days later"
        if offset_days == 7:
            return "a week later"
        return f"{offset_days} days later"

    def _fingerprint(self, example: dict[str, Any]) -> str:
        return "|".join(
            [
                example.get("category", ""),
                example.get("question", "").strip().lower(),
                example.get("answer", "").strip().lower(),
                example.get("source_id", ""),
            ]
        )
