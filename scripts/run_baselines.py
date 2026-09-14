"""Run baseline evaluations (zero-shot, few-shot) on benchmarks."""

from __future__ import annotations

import functools
import json
import logging
import re
from pathlib import Path
import sys
from typing import Any, TYPE_CHECKING

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import click
try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

if TYPE_CHECKING:
    from src.data.data_loader import TemporalExample

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


MODEL_ALIASES: dict[str, str] = {
    "qwen": "qwen2.5",
    "qwen2.5": "qwen2.5",
    "qwen3.5": "qwen3.5",
    "mistral": "mistral",
    "llama": "llama",
}

MODEL_RUNTIME_CONFIG: dict[str, dict[str, str]] = {
    "qwen2.5": {
        "inference_key": "qwen2.5-3b",
        "model_dir": "models/Qwen2.5-3B-Instruct",
    },
    "qwen3.5": {
        "inference_key": "qwen3.5-9b",
        "model_dir": "models/qwen3.5-9b-model",
    },
    "mistral": {
        "inference_key": "mistral",
        "model_dir": "models/Mistral-7B-Instruct-v0.3",
    },
    "llama": {
        "inference_key": "llama",
        "model_dir": "models/Llama-3.2-3B-Instruct",
    },
}


def _normalize_model_choice(model: str) -> list[str]:
    if model == "all":
        return ["qwen2.5", "qwen3.5", "mistral", "llama"]
    return [MODEL_ALIASES[model]]


def _normalize_benchmark_choice(benchmark: str) -> list[str]:
    return ["time", "timebench", "tram"] if benchmark == "all" else [benchmark]


def _nli_instruction(example: "TemporalExample") -> str | None:
    """Task-specific instruction for premise/hypothesis NLI-style tasks.

    TimeBench's TRACIE and TimeX-NLI subtasks are classification tasks over
    a bare hypothesis sentence. With the generic "answer the question"
    framing the model free-completes the sentence instead of classifying
    it, which pinned ~53% of timebench at exactly 0% accuracy.
    """
    if example.choices:
        return None
    task = (example.task or "").lower()
    ttype = (example.temporal_type or "").lower()
    if "tracie" in task:
        return (
            "Read the premise. Decide whether the hypothesis is consistent "
            "with it. Answer with exactly one word: positive or negative."
        )
    if ttype == "temporal_nli" or "nli" in task:
        return (
            "Read the premise. Does the premise entail the hypothesis? "
            "Answer with exactly one word: Entailment, Contradiction, or Neutral."
        )
    return None


# Questions that carry their own output-format requirement (TIME Timeline:
# "Requirements: You must output a sequence of uppercase letters ...").
_SELF_SPECIFIED_FORMAT_RE = re.compile(
    r"Requirements:|must output|output format", re.IGNORECASE
)


def _build_zero_shot_prompt(example: "TemporalExample") -> str:
    nli_instruction = _nli_instruction(example)
    if nli_instruction is not None:
        prompt_parts: list[str] = [
            "You are a temporal reasoning assistant.",
            nli_instruction,
        ]
        if example.context:
            prompt_parts.append(f"Premise:\n{example.context}")
        prompt_parts.append(f"Hypothesis: {example.question}")
        prompt_parts.append("Return only the final answer with no explanation.")
        return "\n\n".join(prompt_parts)

    prompt_parts = [
        "You are a temporal reasoning assistant.",
        "Answer the question using only the provided context when available.",
        "Return only the final answer with no explanation.",
    ]
    if example.context:
        prompt_parts.append(f"Context:\n{example.context}")
    prompt_parts.append(f"Question: {example.question}")
    if example.choices:
        formatted_choices = "\n".join(
            f"{chr(65 + idx)}. {choice}" for idx, choice in enumerate(example.choices)
        )
        prompt_parts.append(f"Choices:\n{formatted_choices}")
        # Some tasks state their own output format inside the question. TIME's
        # Timeline items (11,361 of them) require "a sequence of uppercase
        # letters separated by commas, such as 'A,B,C'"; appending the generic
        # single-letter instruction after that contradicts the task in the very
        # last line the model reads. Only add the generic instruction when the
        # question has not already specified a format.
        if not _SELF_SPECIFIED_FORMAT_RE.search(example.question or ""):
            prompt_parts.append(
                "Respond with the option text, or a single letter (A/B/C/D)."
            )
    return "\n\n".join(prompt_parts)


# System prompt for the zero-shot-CoT baseline arm. Verbatim from the STaR
# teacher (scripts/generate_cot_data.py) so the baseline stays in the same
# prompt family the v9/STaR training data is generated with -- comparing v9
# against a differently-prompted zero-shot would be prompt-confounded.
COT_SYSTEM_PROMPT = (
    "You are an expert temporal-reasoning teacher. You reason about dates, "
    "durations, event order, and timelines in news text, then state the "
    "final answer in the exact requested format."
)

_COT_FINISH = (
    "Write your reasoning as 'Step 1: ...', 'Step 2: ...', etc., only as "
    "many steps as the question needs. Finish with a line of exactly the "
    "form 'ANSWER: <answer>' where <answer> is the final answer phrased as "
    "directly as possible."
)


def _build_cot_prompt(example: "TemporalExample") -> str:
    """Zero-shot self-rationalisation prompt (STaR arm-A family, no gold).

    Same input formatting as _build_zero_shot_prompt -- same context/
    question/choices layout, NLI-aware -- with the answer-format instruction
    replaced by the step-then-anchor instruction. NEVER includes the gold.
    """
    nli_instruction = _nli_instruction(example)
    if nli_instruction is not None:
        prompt_parts: list[str] = [
            "You are a temporal reasoning assistant.",
            nli_instruction,
        ]
        if example.context:
            prompt_parts.append(f"Premise:\n{example.context}")
        prompt_parts.append(f"Hypothesis: {example.question}")
        prompt_parts.append(_COT_FINISH)
        return "\n\n".join(prompt_parts)

    prompt_parts = [
        "You are a temporal reasoning assistant.",
        "Answer the question using only the provided context when available.",
    ]
    if example.context:
        prompt_parts.append(f"Context:\n{example.context}")
    prompt_parts.append(f"Question: {example.question}")
    if example.choices:
        formatted_choices = "\n".join(
            f"{chr(65 + idx)}. {choice}" for idx, choice in enumerate(example.choices)
        )
        prompt_parts.append(f"Choices:\n{formatted_choices}")
        # Mirror the standard prompt's self-specified-format guard: TIME's
        # Timeline items already dictate 'A,B,C' output, so the generic
        # option instruction would contradict the task. The ANSWER anchor
        # instruction always applies (scoring depends on it).
        if not _SELF_SPECIFIED_FORMAT_RE.search(example.question or ""):
            prompt_parts.append(
                "In your final ANSWER line, respond with the option text, "
                "or a single letter (A/B/C/D)."
            )
    prompt_parts.append(_COT_FINISH)
    return "\n\n".join(prompt_parts)


def _load_cot_exemplars(path: str) -> list[dict]:
    """Load and validate the few-shot CoT exemplar file (fail loudly)."""
    import json
    import os

    if not os.path.exists(path):
        raise click.ClickException(f"--prompt-style cot_fewshot: exemplar file not found: {path}")
    with open(path) as fh:
        data = json.load(fh)
    exemplars = data.get("exemplars")
    if not isinstance(exemplars, list) or not exemplars:
        raise click.ClickException(f"{path}: 'exemplars' must be a non-empty list")
    for idx, ex in enumerate(exemplars):
        for field in ("question", "steps", "answer"):
            if not ex.get(field):
                raise click.ClickException(f"{path}: exemplar[{idx}] missing '{field}'")
        if not isinstance(ex["steps"], list) or not all(s.strip() for s in ex["steps"]):
            raise click.ClickException(f"{path}: exemplar[{idx}].steps must be non-empty strings")
        if ex.get("choices") is not None and not isinstance(ex["choices"], list):
            raise click.ClickException(f"{path}: exemplar[{idx}].choices must be a list or null")

    # PREAMBLE BUDGET (2026-09-14b). The prompt is truncated to 4,096 tokens
    # from the LEFT at inference (src/models/inference.py), and the exemplars
    # sit at the very front — so an oversized exemplar file does not error, it
    # silently deletes itself along with COT_SYSTEM_PROMPT and leaves the model
    # a mid-sentence fragment. The first mined file was 9,393 tokens and this
    # went unnoticed through three completed legs. chars/4 is the same estimator
    # the batch packer uses; the exact check lives in mine_cot_exemplars.py.
    preamble_chars = sum(len(_format_cot_exemplar(ex)) for ex in exemplars) + 64
    est_tokens = preamble_chars // 4
    if est_tokens > 1600:
        raise click.ClickException(
            f"{path}: exemplar preamble is ~{est_tokens:,} tokens, over the 1,600 "
            f"budget. At inference the prompt is left-truncated to 4,096 tokens, so "
            f"this preamble would be silently cut — removing the system prompt and "
            f"the leading exemplars and leaving a fragment, not {len(exemplars)} "
            f"worked examples. Re-mine with scripts/mine_cot_exemplars.py, which "
            f"now enforces the budget at selection time."
        )
    return exemplars


def _format_cot_exemplar(ex: dict) -> str:
    """Render one worked example: context/question/choices, then trace + ANSWER."""
    parts: list[str] = []
    if ex.get("context"):
        parts.append(f"Context:\n{ex['context']}")
    parts.append(f"Question: {ex['question']}")
    if ex.get("choices"):
        parts.append("Choices:\n" + "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(ex["choices"])))
    parts.extend(ex["steps"])
    parts.append(f"ANSWER: {ex['answer']}")
    return "\n".join(parts)


def _build_cot_fewshot_prompt(example: "TemporalExample", exemplars: list[dict]) -> str:
    """Few-shot CoT: k worked traces, then the target under the cot builder.

    The target block is exactly _build_cot_prompt(example) -- same persona,
    layout, NLI-awareness and ANSWER anchor -- so the only delta vs the
    zero_shot_reasoning arm is the visible exemplars. The target question is
    ALWAYS last (recency) and its gold NEVER appears in the exemplars.
    """
    blocks = [_format_cot_exemplar(ex) for ex in exemplars]
    return (
        "Worked examples:\n\n"
        + "\n\n".join(blocks)
        + "\n\nNow solve this one, the same way.\n\n"
        + _build_cot_prompt(example)
    )


def _flatten_answers(answer: Any) -> list[Any]:
    """Flatten arbitrarily nested answer structures into a list of values."""
    if isinstance(answer, (list, tuple)):
        flat: list[Any] = []
        for item in answer:
            flat.extend(_flatten_answers(item))
        return flat
    return [answer]


def _canonical_golds(example: "TemporalExample") -> list[str]:
    """Return ALL acceptable gold strings for an example (multi-gold).

    Every gold variant is kept so the metrics can credit any correct
    alternative. For multiple-choice examples, a single-letter gold (e.g.
    "A" / "(A)") is additionally mapped to its option text, and both
    variants stay acceptable.
    """
    golds: list[str] = []
    for value in _flatten_answers(example.answer):
        text = "" if value is None else str(value).strip()
        if not text:
            continue
        variants = [text]
        letter = text.strip("()")
        if example.choices and len(letter) == 1 and letter.upper() in "ABCD":
            idx = ord(letter.upper()) - ord("A")
            if 0 <= idx < len(example.choices):
                variants.append(example.choices[idx].strip())
        elif example.choices and isinstance(value, int) and 0 <= value < len(example.choices):
            variants.append(example.choices[value].strip())
        for variant in variants:
            if variant and variant not in golds:
                golds.append(variant)
    return golds or [""]


# A standalone option letter: "B", "(B)", "B.", "B)" — the whole line.
_BARE_LETTER_RE = re.compile(r"^\(?([A-D])\)?[.,:]?$", re.IGNORECASE)
# An option letter used as a prefix: "B) text", "B. text", "B - text".
# Requires a delimiter AND whitespace, so the article in "A man arrived"
# and the sequence "C,B,A" are both left alone.
_PREFIXED_LETTER_RE = re.compile(r"^\(?([A-D])[).:,\-]\s+", re.IGNORECASE)

# "<option text>; B" -- the model states its answer and then names the option
# letter. Anchored to the very end so it cannot fire on prose containing a
# semicolon mid-sentence.
_TRAILING_LETTER_RE = re.compile(r"^(?P<text>.+?)\s*;\s*\(?(?P<letter>[A-D])\)?[.]?$", re.IGNORECASE | re.DOTALL)

# A stem made only of option letters and separators — "A", "A; B", "C, D".
_LETTERS_ONLY_RE = re.compile(r"^\s*\(?[A-D]\)?\s*(?:[;,]\s*\(?[A-D]\)?\s*)*$", re.IGNORECASE)

# A negation cue anywhere in the answer line. The containment rule must not
# fire on these: "It was not December 1994, it was March 1995" contains BOTH
# options as substrings and returned whichever appeared first in the choices
# list — crediting an option the model explicitly denied. When a cue is
# present the line falls through to the prefix/trailing rules instead, which
# have their own (stricter) guards.
_NEGATION_CUE_RE = re.compile(
    r"\b(?:not|no|neither|nor|never|cannot|can'?t|don'?t|doesn'?t|didn'?t|"
    r"isn'?t|wasn'?t|weren'?t|rather\s+than|instead\s+of)\b",
    re.IGNORECASE,
)


def _postprocess_prediction(raw_prediction: str, choices: list[str] | None) -> str:
    prediction = (raw_prediction or "").strip()
    if not prediction:
        return ""

    first_line = prediction.splitlines()[0].strip()
    # STaR/CoT forward-compat: if the model emits an anchored final answer
    # marker, score that instead of the first reasoning line. No effect on
    # current arms (verified: 0 predictions contain the marker).
    if "ANSWER:" in prediction:
        tail = prediction.rsplit("ANSWER:", 1)[1]
        first_line = tail.strip().splitlines()[0].strip() if tail.strip() else first_line
    if not choices:
        return first_line

    # Exact choice text wins over any letter reading: a choice that happens to
    # begin with "A" is answer text, not option A.
    for choice in choices:
        if first_line.lower() == choice.lower():
            return choice

    # A bare option letter — "B", "(B)", "B.", "B) ..." — but NOT a word that
    # merely starts with A-D. The previous rule tested first_line[0] alone, so
    # "Aralvaimozhi" became option A and "December 1994" became option D. That
    # rewrote roughly one in seven free-text answers into a choice the model
    # never named, and the rewrite was stored, not just scored.
    match = _BARE_LETTER_RE.match(first_line) or _PREFIXED_LETTER_RE.match(first_line)
    if match:
        idx = ord(match.group(1).upper()) - ord("A")
        if 0 <= idx < len(choices):
            return choices[idx]

    # Containment: the option text appears inside the answer line. Two guards
    # (2026-09-09 audit H1; the previous plain substring loop is preserved in
    # git history):
    #   1. no negation cue in the line (see _NEGATION_CUE_RE);
    #   2. (?<!\w)...(?!\w) boundaries so "No" cannot match inside "nothing";
    #   3. when several options appear, the LAST one in the line wins —
    #      English self-corrections affirm the final mention ("not X, but Y").
    if not _NEGATION_CUE_RE.search(first_line):
        lowered = first_line.lower()
        best = None
        for choice in choices:
            m = re.search(rf"(?<!\w){re.escape(choice.lower())}(?!\w)", lowered)
            if m and (best is None or m.start() >= best[0]):
                best = (m.start(), choice)
        if best is not None:
            return best[1]

    # An unambiguous partial naming of one option. v4 answers "Fact 1" where
    # the option reads "Fact 1 happened earlier." — enough to identify the
    # option and no other, but shorter than the option, so neither the exact
    # rule nor the containment rule above can see it. Requires >= 5 chars (a
    # bare letter or article can never qualify) and requires the prefix to
    # match exactly one option, so an ambiguous stem like "Fact" is left
    # alone. Measured inert for zero-shot/v1/v2/v3-corrected (+0.00-0.04pp,
    # 0 items on zero-shot) and worth +21.2pp on v4's Order_Compare.
    if len(first_line) >= 5:
        lowered = first_line.lower()
        named = [c for c in choices if c.lower().startswith(lowered)]
        if len(named) == 1:
            return named[0]

    # A self-declared choice: "<option text>; B". v5 emits this shape on
    # 26,192 TIME MCQ items (v3-corrected 10,559, zero-shot 0) and the text
    # and the letter agree 98.5% of the time wherever the text resolves at
    # all -- the model is naming its answer twice, not hedging. The trailing
    # letter is only read HERE, after every text-based rule above has already
    # failed on the full line, so a resolvable text answer always wins and
    # this can never override one. The text half is retried on its own first,
    # because the "; B" suffix defeats the exact and containment rules by
    # itself. Recovers 1,935 v5 items and 1,577 v3-corrected items; inert for
    # zero-shot, v1 and v2, which never emit the shape.
    trailing = _TRAILING_LETTER_RE.match(first_line)
    if trailing and not _LETTERS_ONLY_RE.match(trailing.group("text")):
        # Guard: if the stem is itself nothing but option letters ("A", "A; B"),
        # the semicolons are SEPARATORS between letters, not a
        # "<text>; <letter>" self-declaration. Some TIME golds are
        # space-separated letter sequences ("A C", "A B C") and the model
        # answers "A; C" -- reading the last letter as the choice there threw
        # away 114 correct v3-corrected answers before this guard.
        stem = trailing.group("text").strip()
        for choice in choices:
            if stem.lower() == choice.lower():
                return choice
        for choice in choices:
            if choice.lower() in stem.lower():
                return choice
        if len(stem) >= 5:
            lowered = stem.lower()
            named = [c for c in choices if c.lower().startswith(lowered)]
            if len(named) == 1:
                return named[0]
        idx = ord(trailing.group("letter").upper()) - ord("A")
        if 0 <= idx < len(choices):
            return choices[idx]

    return first_line


# Filled in by run_benchmarks() before any record is built; see
# _prediction_record. Module-level so the record builder keeps its signature.
_RUN_PROVENANCE: dict[str, str] = {}


def _prediction_record(
    example: "TemporalExample",
    prediction: str,
    reference: str,
    raw_prediction: str | None = None,
) -> dict[str, Any]:
    """Build one stored prediction row.

    ``raw_prediction`` is the untouched model output. Keeping it means a later
    change to _postprocess_prediction can be applied by rescoring on CPU
    instead of re-running generation — the letter-extraction bug cost a full
    re-run precisely because only the postprocessed string was ever stored.
    """
    record = {
        "id": example.id,
        "benchmark": example.source,
        "task": example.task,
        "question": example.question,
        "context": example.context,
        # TIME's Timeline items (13,171) carry task="Timeline" with
        # temporal_type=None, so keying category on temporal_type alone
        # dropped the largest structured category out of every by-category
        # report. Fall back to the task name.
        "category": example.temporal_type or example.task,
        "prediction": prediction,
        "reference": reference,
    }
    if raw_prediction is not None:
        record["raw_prediction"] = raw_prediction
    # Provenance (audit 2026-09-09 M1/M3; added to the HF path 2026-09-12).
    # run_eval_vllm has tagged its records since 2026-09-09; without the same
    # tags here a results dir written by BOTH engines cannot be told apart
    # after the fact, and the resume guard below has nothing to check.
    record["engine"] = "hf"
    if _RUN_PROVENANCE:
        record.update(_RUN_PROVENANCE)
    return record


def _template_for_benchmark(benchmark: str, results_dir: Path,
                            model_name: str | None = None) -> dict[str, Any]:
    """Report skeleton: the model's own template if one exists, else qwen's.

    The path was hardcoded to ``qwen_*`` for every model (audit 2026-09-09),
    which is harmless today only because ``_save_templated_report`` overwrites
    every field it reads — but it meant a llama run loaded a file whose
    ``model`` said ``qwen2.5-3b-instruct``, one forgotten ``update`` key away
    from mislabelling a report. Prefer the model's own template
    (2026-09-12 fix).
    """
    candidates = []
    if model_name:
        candidates.append(results_dir / f"{model_name}_{benchmark}_template.json")
    candidates.append(results_dir / f"qwen_{benchmark}_template.json")
    for template_path in candidates:
        if template_path.exists():
            with open(template_path, encoding="utf-8") as f:
                return json.load(f)
    return {
        "benchmark": benchmark,
        "model": None,
        "configuration": "zero_shot",
        "timestamp": None,
        "num_examples": 0,
        "overall_accuracy": None,
        "overall_f1": None,
        "exact_match": None,
        "temporal_f1": None,
        "by_category": {},
        "error_analysis": {},
        "metadata": {},
    }


def _save_templated_report(
    report: dict[str, Any],
    benchmark: str,
    model_name: str,
    results_dir: Path,
    max_new_tokens: int,
    temperature: float,
    prompt_type: str = "zero_shot",
) -> Path:
    merged = _template_for_benchmark(benchmark, results_dir, model_name)
    merged.update(
        {
            "benchmark": report.get("benchmark", benchmark),
            "model": model_name,
            "configuration": "fine_tuned" if report.get("metadata", {}).get("adapter_dir") else "zero_shot",
            "timestamp": report.get("timestamp"),
            "num_examples": report.get("num_examples", 0),
            "overall_accuracy": report.get("overall_accuracy"),
            "overall_f1": report.get("overall_f1"),
            "exact_match": report.get("exact_match"),
            "temporal_f1": report.get("temporal_f1"),
            "by_category": report.get("by_category", {}),
            "error_analysis": report.get("error_analysis", {}),
            "metadata": {
                **merged.get("metadata", {}),
                **report.get("metadata", {}),
                "model": model_name,
                "prompt_type": prompt_type,
                "adapter_dir": report.get("metadata", {}).get("adapter_dir"),
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            },
        }
    )

    config_subdir = "finetuned" if prompt_type != "zero_shot" else "zero_shot"
    output_path = (
        results_dir / model_name / benchmark / config_subdir / "zero_shot_report.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    return output_path


def _save_predictions(predictions: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


@click.command()
@click.option("--model", type=click.Choice(["qwen", "qwen2.5", "qwen3.5", "mistral", "llama", "all"]), default="all")
@click.option("--benchmark", type=click.Choice(["time", "timebench", "tram", "all"]), default="all")
@click.option("--mode", type=click.Choice(["zero_shot", "few_shot", "all"]), default="zero_shot")
@click.option("--data-dir", default="./data/benchmarks", show_default=True)
@click.option("--results-dir", default="./results/baseline/zero_shot", show_default=True)
@click.option("--model-root", default="./models", show_default=True)
@click.option("--eval-config", default=None, help="Optional path to evaluation YAML config.")
@click.option("--batch-size", default=4, show_default=True, type=int)
@click.option(
    "--token-budget",
    default=65_536,
    show_default=True,
    type=int,
    help="Max batch prompt-tokens (batch_size x longest prompt). VRAM-aware: "
    "~32k for 7B models on 24 GB GPUs, ~65k for 3B.",
)
@click.option("--max-new-tokens", default=128, show_default=True, type=int)
@click.option(
    "--prompt-style",
    type=click.Choice(["standard", "cot", "cot_fewshot"]),
    default="standard",
    show_default=True,
    help="standard: the frozen zero-shot prompt. cot: the STaR arm-A "
    "self-rationalisation family (reason as 'Step 1: ...' then a final "
    "'ANSWER: <answer>' line; the postprocessor already scores the "
    "anchored line). For the zero-shot-CoT baseline arm that v9/STaR is "
    "compared against -- results must go to their own --results-dir. "
    "cot_fewshot: the same family with k worked-trace exemplars prefixed "
    "to the prompt (--cot-exemplars), the few_shot_cot arm.",
)
@click.option(
    "--cot-exemplars",
    default="data/cot_fewshot_exemplars.json",
    show_default=True,
    type=str,
    help="Exemplar file for --prompt-style cot_fewshot (k=4 handwritten "
    "trace exemplars; validated on load).",
)

@click.option(
    "--allow-budget-change",
    is_flag=True,
    default=False,
    help="Permit resuming a predictions file whose rows were generated with a "
         "different --max-new-tokens. The ONLY sound use is a re-run after "
         "scripts/salvage_cot_truncated.py has dropped the truncated rows; "
         "without it, resuming blends two token budgets into one arm.",
)
@click.option("--temperature", default=0.0, show_default=True, type=float)
@click.option("--device", default="auto", show_default=True)
@click.option("--load-in-4bit/--no-load-in-4bit", default=False, show_default=True)
@click.option("--max-samples", default=None, type=int, help="Optional cap per benchmark for quick debug runs.")
@click.option("--category", default=None, type=str, help="Optional temporal_type filter (e.g. temporal_qa).")
@click.option("--adapter-dir", default=None, type=str, help="Optional path to a PEFT LoRA adapter directory.")
def run_baselines(
    model: str,
    benchmark: str,
    mode: str,
    data_dir: str,
    results_dir: str,
    model_root: str,
    eval_config: str | None,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    device: str,
    load_in_4bit: bool,
    max_samples: int | None,
    category: str | None,
    adapter_dir: str | None,
    token_budget: int,
    prompt_style: str,
    cot_exemplars: str,
    allow_budget_change: bool,
) -> None:
    """Run baseline evaluations for zero-shot temporal reasoning."""
    from src.data.data_loader import BenchmarkLoader
    from src.evaluation.evaluate import EvaluationHarness
    from src.models.inference import SLMInference

    # When evaluating a fine-tuned LoRA adapter, use the SAME system prompt
    # the models were trained with, so eval-time prompts stay in-distribution.
    adapter_system_prompt = None
    if adapter_dir:
        from experiments.finetuning.shared.prompt_templates import TEMPORAL_SYSTEM_PROMPT

        adapter_system_prompt = TEMPORAL_SYSTEM_PROMPT
    elif prompt_style in ("cot", "cot_fewshot"):
        adapter_system_prompt = COT_SYSTEM_PROMPT

    # Few-shot exemplars: validate eagerly so a bad file fails in seconds,
    # not hours into a leg.
    if prompt_style == "cot_fewshot":
        cot_exemplars = _load_cot_exemplars(cot_exemplars)
        logger.info("Loaded %d CoT exemplars", len(cot_exemplars))

    if mode not in ("zero_shot", "all"):
        raise click.ClickException("Only zero_shot is currently implemented in this runner.")

    selected_models = _normalize_model_choice(model)
    selected_benchmarks = _normalize_benchmark_choice(benchmark)

    loader = BenchmarkLoader(data_dir)
    harness = EvaluationHarness(eval_config or "./configs/evaluation_config.yaml")
    if eval_config is None:
        harness.config = {
            "output": {
                "save_metrics": True,
                "save_predictions": True,
                "results_dir": str(results_dir),
            }
        }
        logger.info("Running with internal runtime evaluation settings (ignoring YAML config).")
    else:
        logger.info("Using evaluation config: %s", eval_config)

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    harness.results_dir = results_path

    logger.info("Running zero-shot baselines for models=%s on benchmarks=%s", selected_models, selected_benchmarks)


    for model_name in selected_models:
        runtime_cfg = MODEL_RUNTIME_CONFIG[model_name]
        model_dir = Path(model_root) / Path(runtime_cfg["model_dir"]).name
        logger.info("Loading model '%s' from %s", model_name, model_dir)
        inference = SLMInference(
            model_key=runtime_cfg["inference_key"],
            device=device,
            load_in_4bit=load_in_4bit,
            model_dir=str(model_dir),
            adapter_dir=adapter_dir,
            system_prompt=adapter_system_prompt,
        )

        for benchmark_name in selected_benchmarks:
            examples = loader.load(benchmark_name)
            if category:
                category_lower = category.lower()
                examples = [
                    ex for ex in examples
                    if category_lower in {
                        (ex.temporal_type or "").lower(), (ex.task or "").lower()
                    }
                ]
            if max_samples is not None:
                examples = examples[:max_samples]

            logger.info(
                "Benchmark '%s': %d examples for model '%s'",
                benchmark_name,
                len(examples),
                model_name,
            )
            if category:
                logger.info("Category filter: %s", category)

            if prompt_style == "cot":
                _prompt_fn = _build_cot_prompt
            elif prompt_style == "cot_fewshot":
                _prompt_fn = functools.partial(_build_cot_fewshot_prompt, exemplars=cot_exemplars)
            else:
                _prompt_fn = _build_zero_shot_prompt
            prompts = [_prompt_fn(ex) for ex in examples]
            references = [_canonical_golds(ex) for ex in examples]
            # TIME records carry Task, not temporal_type — reading the latter
            # alone filed all 104,951 items under "unknown" in the live report
            # (audit 2026-09-09 M10). Same fallback order as the stored
            # per-row category in _prediction_record.
            categories = [
                ex.temporal_type or ex.task or "unknown" for ex in examples
            ]
            contexts = [ex.context for ex in examples]

            # Batch in prompt-length order. Random-length batches waste
            # roughly half the GPU compute on padding (median prompt ~2k
            # tokens, some capped at 4096). Sorting groups similar lengths
            # so batches are dense.
            #
            # This ordering is part of the protocol, not a free optimisation:
            # left-padding plus non-associative float accumulation means a
            # sequence's greedy output depends on what else shares its batch
            # (measured: 89.25% agreement when the same items are regenerated
            # in different batches). The order below is deterministic -- a
            # stable sort by prompt length -- so the same input set always
            # produces the same batches, and therefore the same predictions.
            # Changing the sort, the batch size or the token budget changes
            # the numbers and breaks comparability with existing arms.
            order = sorted(
                range(len(examples)),
                key=lambda i: len(prompts[i]),
                reverse=True,  # longest first: peak memory hit early
            )
            loop_examples = [examples[i] for i in order]
            loop_prompts = [prompts[i] for i in order]
            loop_references = [references[i] for i in order]

            # Keep zero-shot and fine-tuned runs in separate directories so
            # the resume logic never reuses predictions across configurations
            # (a LoRA adapter run must not inherit zero-shot predictions and
            # vice versa).
            config_subdir = "finetuned" if adapter_dir else "zero_shot"
            pred_path = (
                results_path / model_name / benchmark_name / config_subdir / "predictions.jsonl"
            )
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            completed_ids: set[str] = set()
            prediction_by_id: dict[str, str] = {}

            _RUN_PROVENANCE.clear()
            _RUN_PROVENANCE["model_dir"] = str(model_dir or model_name)
            if adapter_dir:
                _RUN_PROVENANCE["adapter_dir"] = str(adapter_dir)
            # GENERATION CONFIG (audit 2026-09-14b). model_dir/adapter_dir/
            # engine were tagged on 2026-09-12, but the two settings that
            # decide what the model actually emits — the prompt style and the
            # token budget — were not. That is exactly how the few-shot CoT
            # incident happened: a leg launched from a pre-fix script ran at
            # --max-new-tokens 256 while the script on disk said 768, and
            # nothing in the predictions file recorded which budget produced
            # which row. Tag both so a mixed-budget arm is detectable after
            # the fact instead of invisible.
            _RUN_PROVENANCE["prompt_style"] = str(prompt_style)
            _RUN_PROVENANCE["max_new_tokens"] = str(max_new_tokens)

            # Resume logic: load existing predictions if file exists
            if pred_path.exists():
                # Refuse to resume into a file written by a DIFFERENT model,
                # adapter or engine (audit 2026-09-09 M2, enforced on the HF
                # path 2026-09-12 — run_eval_vllm got the same guard). The
                # path is keyed only by results-dir/benchmark/mode, so two
                # adapters sharing a results dir silently blend into one arm.
                _prior = None
                with open(pred_path, "r", encoding="utf-8") as _f:
                    for _line in _f:
                        try:
                            _r = json.loads(_line)
                        except Exception:
                            continue
                        if "model_dir" in _r:
                            _prior = _r
                            break
                if _prior is not None:
                    _mis = []
                    if str(_prior.get("model_dir")) != _RUN_PROVENANCE["model_dir"]:
                        _mis.append(f"model_dir {_prior.get('model_dir')} != "
                                    f"{_RUN_PROVENANCE['model_dir']}")
                    if str(_prior.get("adapter_dir") or "") != str(adapter_dir or ""):
                        _mis.append(f"adapter_dir {_prior.get('adapter_dir')} != {adapter_dir}")
                    if _prior.get("engine", "hf") != "hf":
                        _mis.append(f"engine {_prior.get('engine')} != hf")
                    # A prompt-style change is never a legitimate resume: the
                    # rows already on disk answered a different prompt.
                    if "prompt_style" in _prior and \
                            str(_prior["prompt_style"]) != str(prompt_style):
                        _mis.append(f"prompt_style {_prior['prompt_style']} != {prompt_style}")
                    # A budget change IS legitimate in exactly one workflow:
                    # salvage_cot_truncated.py drops every row that ran out of
                    # budget, then the leg is re-run with a larger one. Greedy
                    # decoding makes the kept rows byte-identical under the
                    # larger cap, so the blend is sound — but only when it is
                    # deliberate. Require it to be said out loud.
                    if "max_new_tokens" in _prior and \
                            str(_prior["max_new_tokens"]) != str(max_new_tokens) \
                            and not allow_budget_change:
                        _mis.append(
                            f"max_new_tokens {_prior['max_new_tokens']} != {max_new_tokens}"
                            " — if this is a post-salvage re-run, pass"
                            " --allow-budget-change; if it is not, you are about"
                            " to blend two token budgets into one arm"
                        )
                    if _mis:
                        raise SystemExit(
                            "FATAL: refusing to resume into a predictions file "
                            "written by a different run — appending would blend "
                            "two arms into one benchmark number:\n  "
                            + "\n  ".join(_mis) + f"\n  file: {pred_path}"
                        )
                elif pred_path.stat().st_size > 0:
                    logger.warning(
                        "%s has rows but no engine/model_dir tags (written "
                        "before 2026-09-12), so this resume CANNOT be checked "
                        "for arm mixing, prompt style or token budget — verify "
                        "by hand.", pred_path,
                    )
                invalid_lines = 0
                duplicate_ids = 0
                with open(pred_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                            record_id = record.get("id")
                            prediction = record.get("prediction")
                            if not isinstance(record_id, str):
                                invalid_lines += 1
                                continue
                            if not isinstance(prediction, str):
                                invalid_lines += 1
                                continue
                            if record_id in prediction_by_id:
                                duplicate_ids += 1
                            prediction_by_id[record_id] = prediction
                        except Exception:
                            invalid_lines += 1
                            continue
                completed_ids = set(prediction_by_id.keys())
                logger.info(
                    "Resuming: %d unique predictions already completed for %s/%s (duplicates=%d, invalid_lines=%d).",
                    len(completed_ids),
                    model_name,
                    benchmark_name,
                    duplicate_ids,
                    invalid_lines,
                )

            import time, datetime
            start_time = time.time()
            progress_path = results_path / model_name / f"{model_name.capitalize()}_live_progress.log"

            def log_zs_progress(current: int, total: int, start_t: float):
                now = datetime.datetime.now()
                elapsed = time.time() - start_t
                if elapsed == 0: elapsed = 0.001
                rate = current / elapsed
                eta = (total - current) / rate if rate > 0 else 0
                pct = (current / total) * 100
                msg = f"{now.strftime('%Y-%m-%d %H:%M:%S')} | progress={current}/{total} ({pct:6.3f}%) | rate={rate:6.2f} ex/s | elapsed={datetime.timedelta(seconds=int(elapsed))} | eta={datetime.timedelta(seconds=int(eta))}\n"
                with open(progress_path, "a") as fp:
                    fp.write(msg)

            # NOTE: do NOT drop completed examples before batching. Batch
            # composition changes predictions -- regenerating 800 held-out
            # items in freshly packed batches reproduced only 89.25% of the
            # originals (86/800 differed, including outright different
            # answers). Batches are therefore always formed over the FULL
            # example list in the same deterministic order, and a batch is
            # skipped only when every member is already complete, so a
            # resumed run reproduces an uninterrupted one exactly. This is
            # slower when the gaps are scattered; that cost buys
            # reproducibility and is not optional.

            # Dynamic batch sizing: batches are capped by BOTH --batch-size
            # and a prompt-token budget (~65k batch-tokens ≈ the proven-safe
            # 16 x 4096 envelope), so 4k-token prompts run in small batches
            # while short prompts run in big ones. Prevents OOM on long-
            # prompt regions without slowing down short-prompt regions.
            TOKEN_BUDGET = max(1024, int(token_budget))

            def _est_tokens(p: str) -> int:
                # chars/4 ≈ tokens; capped at the 4096 truncation limit
                # (+128 generated tokens of KV headroom).
                return min(len(p) // 4 + 1, 4_224)

            batches: list[tuple[int, int]] = []
            start = 0
            while start < len(loop_prompts):
                end = start + 1
                max_tok = _est_tokens(loop_prompts[start])
                while end < len(loop_prompts) and end - start < batch_size:
                    cand_tok = max(max_tok, _est_tokens(loop_prompts[end]))
                    if (end - start + 1) * cand_tok > TOKEN_BUDGET:
                        break
                    max_tok = cand_tok
                    end += 1
                batches.append((start, end))
                start = end

            # Main loop: skip already-completed examples
            for bi, (start, end) in enumerate(batches):
                batch_examples = loop_examples[start:end]
                batch_prompts = loop_prompts[start:end]
                batch_references = loop_references[start:end]

                # Skip batch if all examples in batch are already completed
                if all(ex.id in completed_ids for ex in batch_examples):
                    continue

                raw_outputs = inference.batch_generate(
                    batch_prompts,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                )

                new_records = []
                for example, raw_output, reference in zip(batch_examples, raw_outputs, batch_references):
                    if example.id in completed_ids:
                        continue
                    final_prediction = _postprocess_prediction(raw_output, example.choices)
                    prediction_by_id[example.id] = final_prediction
                    record = _prediction_record(
                        example, final_prediction, reference, raw_prediction=raw_output,
                    )
                    new_records.append(record)

                # Append new predictions to file after each batch
                if new_records:
                    with open(pred_path, "a", encoding="utf-8") as f:
                        for rec in new_records:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                current_idx = end if end < len(loop_prompts) else len(loop_prompts)
                if (bi % 10 == 0) or (current_idx >= len(loop_prompts)):
                    log_zs_progress(current_idx, len(loop_prompts), start_time)

            missing_ids = [ex.id for ex in examples if ex.id not in prediction_by_id]
            if missing_ids:
                raise click.ClickException(
                    "Cannot evaluate %s/%s because %d predictions are still missing (example id: %s)."
                    % (model_name, benchmark_name, len(missing_ids), missing_ids[0])
                )

            ordered_predictions = [prediction_by_id[ex.id] for ex in examples]

            # Evaluate and save report as before
            report = harness.evaluate(
                model_name=model_name,
                benchmark=benchmark_name,
                task="finetuned" if adapter_dir else "zero_shot",
                predictions=ordered_predictions,
                references=references,
                categories=categories,
                contexts=contexts,
                efficiency_kwargs=None,
                answer_chains=None,
            )

            if adapter_dir:
                report["metadata"]["adapter_dir"] = adapter_dir

            report_path = _save_templated_report(
                report=report,
                benchmark=benchmark_name,
                model_name=model_name,
                results_dir=results_path,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                prompt_type="training_format_chat" if adapter_dir else "zero_shot",
            )

            md_path_dir = results_path / model_name / benchmark_name / config_subdir
            md = EvaluationHarness.generate_report_tables(report, md_path_dir)
            (md_path_dir / "report_tables.md").write_text(md, encoding="utf-8")

            logger.info("Saved predictions: %s", pred_path)
            logger.info("Saved report: %s", report_path)

    logger.info("Zero-shot baseline evaluation complete.")


if __name__ == "__main__":
    run_baselines()
