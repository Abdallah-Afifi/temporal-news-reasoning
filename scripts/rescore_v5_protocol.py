"""Rescore every corrected arm under the v5 scoring protocol. Zero GPU cost.

Two defects were found in the v4 post-mortem, both of which made a correct
answer score as wrong whenever the model's output STYLE changed:

1. **Unambiguous partial option naming** — v4 answers "Fact 1" where the
   option reads "Fact 1 happened earlier."  Fixed in
   `scripts/run_baselines._postprocess_prediction`; applied here by
   re-running that function over the stored `raw_prediction`.
2. **Date reformatting** — v4 answers "18 March 1934" where the gold reads
   "March 18, 1934."  Fixed in `src.evaluation.date_equivalence`, applied
   here at scoring time.

Both are applied to EVERY arm — scoring protocol is identical across arms or
the comparison is void. Neither can credit a less precise or different
answer; see the two test modules for the guards.

Writes results/rescored/v5_protocol.json and prints the official table.

Usage: ./venv/bin/python scripts/rescore_v5_protocol.py
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_baselines import _postprocess_prediction  # noqa: E402
from src.data.data_loader import BenchmarkLoader  # noqa: E402
from src.evaluation.date_equivalence import matches_any  # noqa: E402
from src.evaluation.generation_metrics import best_token_f1  # noqa: E402
from src.evaluation.metrics import _normalize_answer  # noqa: E402

ARMS = {
    "time": [
        ("zero-shot", "results/baseline/zero_shot_v3/llama/time/zero_shot/predictions.jsonl"),
        ("v1-ft", "results/corrected/v1/llama/time/finetuned/predictions.jsonl"),
        ("v2-ft", "results/corrected/v2/llama/time/finetuned/predictions.jsonl"),
        ("v3-corr", "results/corrected/v3_fixed/llama/time/finetuned/predictions.jsonl"),
        ("v4", "results/corrected/v4/llama/time/finetuned/predictions.jsonl"),
        ("v5", "results/corrected/v5/llama/time/finetuned/predictions.jsonl"),
        ("v6", "results/corrected/v6/llama/time/finetuned/predictions.jsonl"),
        # ENGINE-SUFFIXED ARMS. run_schedule_v7.sh sends every vLLM leg to
        # results/corrected/<arm>_vllm, NOT results/corrected/<arm>. Before
        # 2026-09-08 the v7 entry pointed at the non-vllm path and v7 was
        # absent from `time` and `timebench` altogether, so the final rescore
        # would have reported v7 as "missing" on all three benchmarks.
        # Labels carry the engine because this table now MIXES engines and
        # D53 requires every cell to say which one produced it.
        ("v6-vllm", "results/corrected/v6_vllm/llama/time/finetuned/predictions.jsonl"),
        ("v7-vllm", "results/corrected/v7_vllm/llama/time/finetuned/predictions.jsonl"),
        ("v7c-vllm", "results/corrected/v7_corrected_vllm/llama/time/finetuned/predictions.jsonl"),
        ("v6d-vllm", "results/corrected/v6d_vllm/llama/time/finetuned/predictions.jsonl"),
        # v9 — the synthetic-swap arm (AUG_GLM2). vLLM on all three
        # benchmarks; see docs/audit_2026_09_16.md for the data caveats.
        ("v9-vllm", "results/corrected/v9_vllm/llama/time/finetuned/predictions.jsonl"),
        ("v9-glm-vllm", "results/corrected/v9_glm_vllm/llama/time/finetuned/predictions.jsonl"),
        # v10-glm (added 2026-09-22): AUG_GLM2 doubled to 6,000 rows, news-shape
        # near-miss-passage mechanism added, lora_r 16->32/alpha 32->64. A KNOWN
        # three-way confound vs v9-glm -- see run_schedule_v10_glm.sh.
        ("v10-glm-vllm", "results/corrected/v10_glm_vllm/llama/time/finetuned/predictions.jsonl"),
        # ZERO-SHOT vLLM (added 2026-09-21, scripts/run_zs_vllm_time_timebench.sh).
        # No adapter, base model straight to vLLM. Removes the need to invoke HF/vLLM
        # parity to compare a vLLM-only arm (v7/v7c/v6d/v9/v9-glm) against zero-shot.
        ("zs-vllm", "results/baseline/zero_shot_vllm/llama/time/zero_shot/predictions.jsonl"),
        # CoT arms (added 2026-09-14). These existed since 2026-09-09 with
        # complete predictions and were NOT in this list, so they never
        # reached the canonical table — the same defect class as the
        # v7-corrected omission fixed on 2026-09-09 (F1). They are base-model
        # prompting arms, not fine-tunes: no adapter, no training.
        # Mistral zero-shot: complete since the zero_shot_v3 era and quoted in
        # session_state, but never in this list until 2026-09-14, so it was
        # never scored by the frozen scorer. Mistral is a SIDE experiment --
        # banned from vLLM for a 2.20pp parity failure (D46/D53) -- so its
        # numbers are not comparable to the llama campaign, but they should at
        # least be produced by the same scorer as everything else.
        ("zs-mistral", "results/baseline/zero_shot_v3/mistral/time/zero_shot/predictions.jsonl"),
        ("zsCoT-llama", "results/baseline/zero_shot_cot/llama/time/zero_shot/predictions.jsonl"),
        ("zsCoT-mistr", "results/baseline/zero_shot_cot/mistral/time/zero_shot/predictions.jsonl"),
        ("fsCoT-llama", "results/baseline/few_shot_cot/llama/time/zero_shot/predictions.jsonl"),
        ("fsCoT-mistr", "results/baseline/few_shot_cot/mistral/time/zero_shot/predictions.jsonl"),
],
    # TRAM: added 2026-09-07 for its first ever run (D48). Arms appear here
    # only once their predictions exist; a missing file is reported and
    # skipped, so this is safe to carry before the run.
    "tram": [
        ("zero-shot", "results/baseline/zero_shot_v3/llama/tram/zero_shot/predictions.jsonl"),
        ("v1-vllm", "results/corrected/v1_vllm/llama/tram/finetuned/predictions.jsonl"),
        ("v2-vllm", "results/corrected/v2_vllm/llama/tram/finetuned/predictions.jsonl"),
        ("v3c-vllm", "results/corrected/v3c_vllm/llama/tram/finetuned/predictions.jsonl"),
        ("v4-vllm", "results/corrected/v4_vllm/llama/tram/finetuned/predictions.jsonl"),
        ("v5-vllm", "results/corrected/v5_vllm/llama/tram/finetuned/predictions.jsonl"),
        ("v6-vllm", "results/corrected/v6_vllm/llama/tram/finetuned/predictions.jsonl"),
        ("v7-vllm", "results/corrected/v7_vllm/llama/tram/finetuned/predictions.jsonl"),
        ("v7c-vllm", "results/corrected/v7_corrected_vllm/llama/tram/finetuned/predictions.jsonl"),
        ("v6d-vllm", "results/corrected/v6d_vllm/llama/tram/finetuned/predictions.jsonl"),
        # v9 — the synthetic-swap arm (AUG_GLM2). vLLM on all three
        # benchmarks; see docs/audit_2026_09_16.md for the data caveats.
        ("v9-vllm", "results/corrected/v9_vllm/llama/tram/finetuned/predictions.jsonl"),
        ("v9-glm-vllm", "results/corrected/v9_glm_vllm/llama/tram/finetuned/predictions.jsonl"),
        # v10-glm (added 2026-09-22): AUG_GLM2 doubled to 6,000 rows, news-shape
        # near-miss-passage mechanism added, lora_r 16->32/alpha 32->64. A KNOWN
        # three-way confound vs v9-glm -- see run_schedule_v10_glm.sh.
        ("v10-glm-vllm", "results/corrected/v10_glm_vllm/llama/tram/finetuned/predictions.jsonl"),
],
    "timebench": [
        ("zero-shot", "results/baseline/zero_shot_v3/llama/timebench/zero_shot/predictions.jsonl"),
        ("v1-ft", "results/corrected/v1/llama/timebench/finetuned/predictions.jsonl"),
        ("v2-ft", "results/corrected/v2/llama/timebench/finetuned/predictions.jsonl"),
        ("v3-corr", "results/corrected/v3_fixed/llama/timebench/finetuned/predictions.jsonl"),
        ("v4", "results/corrected/v4/llama/timebench/finetuned/predictions.jsonl"),
        ("v5", "results/corrected/v5/llama/timebench/finetuned/predictions.jsonl"),
        ("v6", "results/corrected/v6/llama/timebench/finetuned/predictions.jsonl"),
        ("v6-vllm", "results/corrected/v6_vllm/llama/timebench/finetuned/predictions.jsonl"),
        ("v7-vllm", "results/corrected/v7_vllm/llama/timebench/finetuned/predictions.jsonl"),
        ("v7c-vllm", "results/corrected/v7_corrected_vllm/llama/timebench/finetuned/predictions.jsonl"),
        ("v6d-vllm", "results/corrected/v6d_vllm/llama/timebench/finetuned/predictions.jsonl"),
        # v9 — the synthetic-swap arm (AUG_GLM2). vLLM on all three
        # benchmarks; see docs/audit_2026_09_16.md for the data caveats.
        ("v9-vllm", "results/corrected/v9_vllm/llama/timebench/finetuned/predictions.jsonl"),
        ("v9-glm-vllm", "results/corrected/v9_glm_vllm/llama/timebench/finetuned/predictions.jsonl"),
        # v10-glm (added 2026-09-22): AUG_GLM2 doubled to 6,000 rows, news-shape
        # near-miss-passage mechanism added, lora_r 16->32/alpha 32->64. A KNOWN
        # three-way confound vs v9-glm -- see run_schedule_v10_glm.sh.
        ("v10-glm-vllm", "results/corrected/v10_glm_vllm/llama/timebench/finetuned/predictions.jsonl"),
        # ZERO-SHOT vLLM (added 2026-09-21, scripts/run_zs_vllm_time_timebench.sh).
        # No adapter, base model straight to vLLM. Removes the need to invoke HF/vLLM
        # parity to compare a vLLM-only arm (v7/v7c/v6d/v9/v9-glm) against zero-shot.
        ("zs-vllm", "results/baseline/zero_shot_vllm/llama/timebench/zero_shot/predictions.jsonl"),
        # CoT arms (added 2026-09-14). These existed since 2026-09-09 with
        # complete predictions and were NOT in this list, so they never
        # reached the canonical table — the same defect class as the
        # v7-corrected omission fixed on 2026-09-09 (F1). They are base-model
        # prompting arms, not fine-tunes: no adapter, no training.
        # Mistral zero-shot: complete since the zero_shot_v3 era and quoted in
        # session_state, but never in this list until 2026-09-14, so it was
        # never scored by the frozen scorer. Mistral is a SIDE experiment --
        # banned from vLLM for a 2.20pp parity failure (D46/D53) -- so its
        # numbers are not comparable to the llama campaign, but they should at
        # least be produced by the same scorer as everything else.
        ("zs-mistral", "results/baseline/zero_shot_v3/mistral/timebench/zero_shot/predictions.jsonl"),
        ("zsCoT-llama", "results/baseline/zero_shot_cot/llama/timebench/zero_shot/predictions.jsonl"),
        ("zsCoT-mistr", "results/baseline/zero_shot_cot/mistral/timebench/zero_shot/predictions.jsonl"),
        ("fsCoT-llama", "results/baseline/few_shot_cot/llama/timebench/zero_shot/predictions.jsonl"),
        ("fsCoT-mistr", "results/baseline/few_shot_cot/mistral/timebench/zero_shot/predictions.jsonl"),
],
}


# TIME ships ONE category under two spellings — "Co_temporality" (8,289 items)
# and "Co-temporality" (1,800). It is an upstream inconsistency in
# TIME_Newest.json, not a code bug, but reporting it as two rows splits a
# single category across every per-category table in the thesis. Normalised
# here, at report time only: the underlying data is untouched.
_CATEGORY_ALIASES = {"co-temporality": "Co_temporality"}

# Categories whose gold is a sentence, where exact match is the wrong
# instrument. TimeBench's situated_generation asks for a "temporally grounded
# statement" and its 115 golds are multi-sentence texts a correct answer can
# paraphrase, so EM reports 0.00% for EVERY arm including zero-shot — a metric
# artifact, not a result. Reported ADDITIONALLY as token-F1, never folded into
# the headline: the headline stays exact match for every category so one
# campaign is scored by one rule (see src/evaluation/generation_metrics.py).
_GENERATION_CATEGORIES = {"situated_generation"}


def canon_category(raw: str) -> str:
    """Fold TIME's duplicate category spellings into one label."""
    return _CATEGORY_ALIASES.get(str(raw).strip().lower(), str(raw))


# An option that offers abstention. TIME's MCQ items carry one on 2,427 items
# and it is the GOLD on 2,427 of 2,427 -- the benchmark never presents an
# abstain option as a wrong answer. "If an abstain option is present, pick it"
# therefore scores ~100% on that bucket with no temporal reasoning at all
# (D49, 2026-09-08). Arms trained with AUG_NOANS learn exactly that rule, so
# the bucket is not arm-neutral: zero-shot 52.95%, v6 99.88%, v7c 99.18%.
# Reported as a SEPARATE column below; the headline column is untouched.
_ABSTAIN_RE = re.compile(
    r"\b(?:there\s+is\s+no\s+answer|no\s+answer|unanswerable|"
    r"not\s+enough\s+information|cannot\s+be\s+determined)\b",
    re.IGNORECASE,
)


def abstain_option_ids(choices_by_id: dict[str, list[str] | None]) -> set[str]:
    """Ids whose choice list offers an abstain option (see _ABSTAIN_RE)."""
    return {i for i, ch in choices_by_id.items()
            if ch and any(_ABSTAIN_RE.search(str(c) or "") for c in ch)}


def mcnemar(b: int, c: int) -> tuple[float, float]:
    """Paired test for two arms on the SAME items.

    ``b`` = reference right / arm wrong, ``c`` = arm right / reference wrong.
    The unpaired two-proportion z below (``ztest``) ignores the ~80% of items
    on which two arms agree and therefore over-states the standard error; it
    is kept because every published number quotes it, but McNemar is the
    correct test for this design and is reported alongside.

    Returns (signed z, chi-square with continuity correction).
    """
    n = b + c
    if n == 0:
        return 0.0, 0.0
    return (c - b) / math.sqrt(n), (abs(b - c) - 1) ** 2 / n


def choice_map(benchmark: str, data_dir: str) -> dict[str, list[str] | None]:
    loader = BenchmarkLoader(data_dir=data_dir)
    return {ex.id: ex.choices for ex in loader.load(benchmark)}


# Directories that are deliberately not arms: smoke tests, superseded runs,
# and the stale TRAM predictions kept for the record.
_NOT_ARMS = (
    "smoke", "zero_shot_v2", "few_shot/", "finetuned_collided",
    "finetuned_genericprompt", "/v3/", "results_rerun",
    # FROZEN-ERA runs (106,500-row TIME / pre-fix TimeBench): the original
    # broken harness, superseded in full by results/corrected/. Kept for the
    # three-stage comparison in docs/mistake_ledger.md, never scored as arms.
    "results/finetuned/", "results/finetuned_v2/", "results/finetuned_v3/",
    "results/finetuned_v3_vllm/",
    # Stubs from interrupted smoke runs (3-13 rows).
    "results/baseline/zero_shot/llama/", "results/baseline/zero_shot/qwen2.5/",
    "results/corrected/v2/llama/time/zero_shot/",
    "results/baseline/zero_shot/mistral/",
    # The first few-shot CoT attempt (2026-09-13/14). Its 9,393-token exemplar
    # preamble was left-truncated away by the 4,096-token inference limit, so
    # the model never saw the worked examples OR the system prompt, and all
    # three legs additionally ran at the retired 256-token budget. Superseded
    # in full; see that directory's WHY_SUPERSEDED.md.
    "few_shot_cot_SUPERSEDED_truncated_exemplars",
)

# The TRAM paths that --tram-root supersedes. Without this the check would
# flag all ten stale-prompt files as "missing" on every --tram-root run, and a
# check that cries wolf is a check nobody reads.
_SUPERSEDED_TRAM: set[str] = set()


def unlisted_arms(bench: str, listed: set[str],
                  superseded: set[str] | None = None) -> list[tuple[str, int]]:
    """Complete predictions files under results/ that ARMS does not mention.

    A hand-maintained arm list has now silently dropped an arm three times:
    v7-corrected (F1, 2026-09-09), the v7 path bug (2026-09-08), and the CoT
    arms (2026-09-14, complete since 09-09 and never scored). The list stays
    hand-written -- it encodes which directory is canonical for an arm, which
    cannot be inferred -- but it is now CHECKED against the tree, so a
    complete-but-unlisted arm is reported instead of being invisible.
    """
    out: list[tuple[str, int]] = []
    for path in sorted(PROJECT_ROOT.glob(f"results/**/{bench}/*/predictions.jsonl")):
        rel = str(path.relative_to(PROJECT_ROOT))
        if rel in listed or rel in (superseded or set()):
            continue
        if any(k in rel for k in _NOT_ARMS):
            continue
        try:
            with open(path, "rb") as f:
                n = sum(1 for _ in f)
        except OSError:
            continue
        out.append((rel, n))
    return out


def setting_map(benchmark: str, data_dir: str) -> dict[str, str]:
    """id -> TIME `Setting` (the evaluation condition), empty for others.

    TIME is 40.4% `base` (gold context) and 59.6% `bm25`/`vector`/`hybrid`
    (a retriever supplied the context). The conditions sit ~23pp apart AND the
    fine-tuning effect changes sign between them — v7c is +2.76pp on `base`
    and -3.0pp on retrieved, which pools to -0.64pp (audit 2026-09-12
    §0a-bis). A pooled TIME number is therefore a cancellation, not a summary.
    """
    if benchmark != "time":
        return {}
    loader = BenchmarkLoader(data_dir=data_dir)
    return {ex.id: str((ex.metadata or {}).get("Setting") or "")
            for ex in loader.load(benchmark)}


def rescore(path: Path, choices_by_id: dict[str, list[str] | None],
            abstain_ids: set[str] | None = None,
            want_correct_ids: bool = False,
            ref_correct: set[str] | None = None,
            settings: dict[str, str] | None = None) -> dict:
    """Score one arm under the v5 protocol.

    The headline number (``v5_pct``) is unchanged. Four diagnostics are
    reported alongside it, all additive (2026-09-12 audit):

    - ``rules_fired`` / ``strict_pct`` -- how much of the score depends on
      ``_postprocess_prediction``'s answer-extraction rules, measured against
      the bare first line of the raw output. NOTE: this is what "scorer-rule
      firings" means. The older ``rewritten`` field counts something else --
      disagreement between the postprocessor NOW and the one that ran at
      generation time -- so it necessarily reads 0 for any arm generated after
      the postprocessor froze. It is kept under the clearer name
      ``protocol_drift`` (``rewritten`` retained as an alias for old readers).
    - ``no_abstain_pct`` -- the score with the abstain-option bucket removed
      (see ``abstain_option_ids``), the artifact-free comparison D49 requires.
    - ``macro_pct`` -- unweighted mean over categories. TRAM is 57.5% one task
      (temporal_nli) and 20.9% a second, so its micro-average is close to "NLI
      accuracy" and the sign of the zero-shot comparison depends on this
      choice. Both are reported; neither is privileged.
    """
    abstain_ids = abstain_ids or set()
    settings = settings or {}
    # (correct, n) per evaluation condition, and the same excluding the
    # abstain bucket -- the combination is the only honest TIME comparison.
    per_set: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    per_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    gen_f1: dict[str, list[float]] = defaultdict(list)
    stored = fixed = n = rewritten = torn = empty_gold = 0
    fired = strict = 0
    abst_hit = abst_n = 0
    # McNemar cells vs the reference arm (zero-shot): b = reference right and
    # this arm wrong, c = this arm right and reference wrong.
    pair_b = pair_c = 0
    # CoT arms instruct the model to finish with a line "ANSWER: <answer>";
    # scoring reads that anchor. An output that runs out of `--max-new-tokens`
    # before reaching it scores 0 for a budget reason, not a reasoning one, and
    # nothing in the pipeline made that visible (audit 2026-09-14: 22.1% of
    # llama's TIME CoT outputs were truncated, and they score 0.08%).
    anchor_have = anchor_missing = 0
    correct_ids: set[str] = set()
    # errors="replace", not the default "strict". FOUND 2026-09-22: v4's TIME
    # predictions.jsonl (dated 2026-09-06, untouched since) has one invalid
    # UTF-8 byte near its start. Strict decoding raises INSIDE this iteration,
    # before the per-line json.loads try/except below ever runs, so it isn't
    # caught by the "torn line" handling that already exists for a mid-write
    # kill -- it aborted the whole rescore, for every arm still to come. Same
    # philosophy as that handling: one corrupt byte in one row of a
    # multi-hundred-thousand-row arm should not block scoring everything
    # after it. A replaced byte inside a JSON string still parses; if it lands
    # somewhere that breaks JSON structure instead, the existing
    # JSONDecodeError handler below already counts it as torn and continues.
    for line in open(path, encoding="utf-8", errors="replace"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            # A leg killed mid-write leaves a torn line somewhere in
            # predictions.jsonl. Count it and keep going: stopping at the
            # first bad line would silently discard everything after it and
            # under-report a 99.99%-complete run as a tiny partial.
            torn += 1
            continue
        refs = r["reference"] if isinstance(r["reference"], list) else [r["reference"]]
        gold = [_normalize_answer(g) for g in refs]
        if not any(gold):
            # Empty gold in the source dataset (113 TimeBench timeqa rows,
            # audit 2026-09-09 H3). The loader now skips these at load time;
            # old prediction files still contain the rows, so skip them here
            # too — for every arm alike — to keep the denominator consistent.
            empty_gold += 1
            continue
        raw = r.get("raw_prediction")
        pred = _postprocess_prediction(raw, choices_by_id.get(r["id"])) \
            if raw is not None else r["prediction"]
        rewritten += pred != r["prediction"]
        s = _normalize_answer(r["prediction"]) in gold
        f = matches_any(pred, refs)
        # Strict reading: the model's own first line, with no option-matching,
        # letter-extraction, containment or trailing-letter rule applied.
        first = ""
        if raw is not None and str(raw).strip():
            first = str(raw).strip().splitlines()[0].strip()
            fired += pred != first
            strict += matches_any(first, refs)
        else:
            strict += f
        stored += s
        fixed += f
        n += 1
        if (setting := settings.get(r["id"])):
            e = per_set[setting]
            e[0] += f
            e[1] += 1
            if r["id"] not in abstain_ids:
                e[2] += f
                e[3] += 1
        if r["id"] in abstain_ids:
            abst_n += 1
            abst_hit += f
        if want_correct_ids and f:
            correct_ids.add(r["id"])
        if raw is not None and str(raw).strip():
            if "ANSWER:" in str(raw):
                anchor_have += 1
            else:
                anchor_missing += 1
        if ref_correct is not None:
            ref = r["id"] in ref_correct
            pair_b += ref and not f
            pair_c += f and not ref
        cat = canon_category(r.get("category") or r.get("task") or "unknown")
        c = per_cat[cat]
        c[0] += s
        c[1] += f
        c[2] += 1
        if cat in _GENERATION_CATEGORIES:
            gen_f1[cat].append(best_token_f1(pred, refs))
    cats = {k: v for k, v in per_cat.items()}
    rest_n, rest_hit = n - abst_n, fixed - abst_hit
    return {"n": n, "torn": torn, "empty_gold": empty_gold,
            "stored_pct": 100 * stored / n if n else 0.0,
            "v5_pct": 100 * fixed / n if n else 0.0,
            "stored_correct": stored, "v5_correct": fixed,
            # see the docstring: protocol_drift != rules_fired
            "protocol_drift": rewritten, "rewritten": rewritten,
            "rules_fired": fired,
            "rules_fired_pct": 100 * fired / n if n else 0.0,
            "strict_correct": strict,
            "strict_pct": 100 * strict / n if n else 0.0,
            "rule_credit_pp": (100 * fixed / n - 100 * strict / n) if n else 0.0,
            "abstain_n": abst_n,
            "abstain_pct": 100 * abst_hit / abst_n if abst_n else float("nan"),
            "no_abstain_n": rest_n,
            "no_abstain_correct": rest_hit,
            "no_abstain_pct": 100 * rest_hit / rest_n if rest_n else 0.0,
            "macro_pct": (sum(100 * v[1] / v[2] for v in cats.values())
                          / len(cats)) if cats else 0.0,
            "by_setting": {k: {"v5_pct": round(100 * v[0] / v[1], 2), "n": v[1],
                               "no_abstain_pct": round(100 * v[2] / v[3], 2) if v[3] else None,
                               "no_abstain_n": v[3]}
                           for k, v in per_set.items()},
            "correct_ids": correct_ids if want_correct_ids else None,
            "_paired": [pair_b, pair_c] if ref_correct is not None else None,
            # Reported only for anchor-style (CoT) arms: for a standard arm
            # almost nothing carries the anchor and the number is meaningless.
            "anchor_missing": anchor_missing if anchor_have > 0.1 * n else None,
            "anchor_missing_pct": (100 * anchor_missing / n
                                   if n and anchor_have > 0.1 * n else None),
            "by_category": {k: {"stored_pct": round(100 * v[0] / v[2], 2),
                                "v5_pct": round(100 * v[1] / v[2], 2), "n": v[2]}
                            for k, v in per_cat.items()},
            "generation_f1": {k: {"token_f1": round(100 * sum(v) / len(v), 2),
                                  "n": len(v)}
                              for k, v in gen_f1.items() if v}}


def ztest(c1: int, n1: int, c2: int, n2: int) -> float:
    p1, p2 = c1 / n1, c2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return (p1 - p2) / se if se else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data/benchmarks")
    ap.add_argument("--out", default="results/rescored/v5_protocol.json")
    ap.add_argument("--tram-root", default=None, help=(
        "Score the TRAM arms from <root>/<arm>/llama/tram/... instead of the "
        "stale results/corrected paths. Used by "
        "scripts/run_schedule_tram_rerun.sh after the 2026-09-12 loader fix; "
        "TIME and TimeBench are unaffected either way."))
    args = ap.parse_args()

    if args.tram_root:
        # arm label -> directory name under --tram-root. zero-shot writes to
        # a `zero_shot/` config subdir (no adapter); every other arm writes to
        # `finetuned/`.
        rerun = {"zero-shot": ("zero_shot", "zero_shot"), "v1-vllm": ("v1", "finetuned"),
                 "v2-vllm": ("v2", "finetuned"), "v3c-vllm": ("v3c", "finetuned"),
                 "v4-vllm": ("v4", "finetuned"), "v5-vllm": ("v5", "finetuned"),
                 "v6-vllm": ("v6", "finetuned"), "v7-vllm": ("v7", "finetuned"),
                 "v7c-vllm": ("v7c", "finetuned"), "v6d-vllm": ("v6d", "finetuned"),
                 # v9 writes its TRAM leg straight to the fixed root, so it is
                 # listed here as well as in ARMS["tram"]. Omitting it here
                 # would make --tram-root drop the arm silently, which is the
                 # v7-omission defect (F1) in a new place.
                 "v9-vllm": ("v9", "finetuned"),
                 "v9-glm-vllm": ("v9_glm", "finetuned"),
                 "v10-glm-vllm": ("v10_glm", "finetuned")}
        _SUPERSEDED_TRAM.update(p for _, p in ARMS["tram"])
        ARMS["tram"] = [
            (label, f"{args.tram_root}/{d}/llama/tram/{sub_}/predictions.jsonl")
            for label, (d, sub_) in rerun.items()
        ]
        print(f"TRAM arms repointed at {args.tram_root} (2026-09-12 fixed prompts)")

    report: dict = {}
    for bench in ("time", "timebench", "tram"):
        cmap = choice_map(bench, args.data_dir)
        smap = setting_map(bench, args.data_dir)
        abst = abstain_option_ids(cmap)
        arms: dict = {}
        partial: dict = {}
        zs_correct: set[str] | None = None
        print(f"\n{'=' * 72}\n{bench.upper()}  (stored -> v5 protocol)\n{'=' * 72}")
        if bench == "tram" and not args.tram_root:
            # The stored TRAM predictions were generated BEFORE the
            # 2026-09-12 loader fix, from prompts that omitted the NLI
            # `Hypothesis` and the storytelling `Story` and that carried the
            # `Source` string as `relation`'s context. The ids still match, so
            # this table computes cleanly and looks fine -- which is exactly
            # why it has to say so out loud.
            print("*** STALE: 836,386 of 980,918 TRAM items (85.3%) were "
                  "generated from prompts")
            print("*** the 2026-09-12 loader fix has changed. Do NOT quote "
                  "this column until it is")
            print("*** re-run. Only the 144,532 unaffected items "
                  "(ambiguity/arithmetic/causality/")
            print("*** duration/frequency/ordering/typical_time) remain valid "
                  "— see docs/audit_2026_09_12.md §0a.")
        if abst:
            print(f"abstain-option bucket: {len(abst):,} items "
                  f"({100 * len(abst) / len(cmap):.1f}% of {len(cmap):,})")
        print(f"{'arm':12s}{'n':>8s}{'stored':>10s}{'v5':>10s}{'delta':>8s}"
              f"{'drift':>8s}{'fired':>8s}{'strict':>9s}{'no-abst':>9s}{'macro':>8s}")
        for label, p in ARMS[bench]:
            path = PROJECT_ROOT / p
            if not path.exists():
                print(f"{label:12s}  missing: {p}")
                continue
            # zero-shot is scored first (ARMS lists it first in every
            # benchmark) so its per-item correctness is available as the
            # reference side of the paired McNemar test below.
            s = rescore(path, cmap, abst,
                        want_correct_ids=(label == "zero-shot"),
                        ref_correct=(None if label == "zero-shot" else zs_correct),
                        settings=smap)
            if label == "zero-shot":
                zs_correct = s.pop("correct_ids", None)
            s.pop("correct_ids", None)
            s["expected_n"] = len(cmap)
            # A run that COVERED the benchmark but lost a line to a torn
            # write is not the thing PARTIAL protects against (2026-09-14).
            # PARTIAL exists to reject an ABORTED leg, whose file is a biased
            # prefix -- length-sorted on HF, a category-blocked prefix on
            # vLLM. A complete run missing a handful of scattered ids to torn
            # writes is neither: it is the whole benchmark minus noise, and
            # excluding it entirely loses a 99.999%-complete arm. Score it,
            # and say so. (Found on the mistral TIME leg: 104,938 scored +
            # 1 torn = 104,939 expected.)
            _covered = s["n"] + s["torn"] >= len(cmap)
            if s["n"] < len(cmap) and _covered:
                print(f"{label:12s}  NOTE: {s['torn']} torn line(s); "
                      f"{s['n']:,}/{len(cmap):,} scored. The run covered the "
                      f"benchmark, so it is scored, not excluded.")
            if s["n"] < len(cmap) and not _covered:
                # A leg that is still running, or one that died part-way.
                # Its number is NOT comparable to a complete arm's, on either
                # engine, but for different reasons (corrected 2026-09-12 —
                # the old comment gave only the HF rationale and asserted it
                # for vLLM legs too, where it is false):
                #   HF   (run_baselines) sorts prompts by length, so a partial
                #        file is the SHORT-prompt end of the benchmark.
                #   vLLM (run_eval_vllm) runs dataset-order chunks, so a
                #        partial file is a dataset-order prefix — and TIME
                #        arrives in large contiguous category blocks (107
                #        runs over 12 categories, ~980 items per run), so a
                #        prefix is a wildly unbalanced category mix.
                # Either way it is a biased subset. Report it, never fold it
                # into the table.
                partial[label] = s
                print(f"{label:12s}{s['n']:8d}{'':10s}{'PARTIAL':>9s}"
                      f"   {100 * s['n'] / len(cmap):5.1f}% of {len(cmap):,} "
                      f"— EXCLUDED (in progress or aborted; length-sorted, so "
                      f"not comparable)")
                continue
            arms[label] = s
            print(f"{label:12s}{s['n']:8d}{s['stored_pct']:9.2f}%{s['v5_pct']:9.2f}%"
                  f"{s['v5_pct'] - s['stored_pct']:+7.2f}{s['protocol_drift']:8d}"
                  f"{s['rules_fired_pct']:7.1f}%{s['strict_pct']:8.2f}%"
                  f"{s['no_abstain_pct']:8.2f}%{s['macro_pct']:7.2f}%")
            if s.get("anchor_missing_pct"):
                print(f"{'':12s}  ^ CoT arm: {s['anchor_missing_pct']:.1f}% of outputs "
                      f"never reached the 'ANSWER:' anchor ({s['anchor_missing']:,} items, "
                      f"token budget exhausted) — those score ~0 for a BUDGET reason, "
                      f"not a reasoning one.")

        # Tree-vs-list check (2026-09-14). See unlisted_arms().
        unlisted = unlisted_arms(bench, {p for _, p in ARMS[bench]},
                                 _SUPERSEDED_TRAM)
        if unlisted:
            print(f"\n*** {len(unlisted)} predictions file(s) for {bench} are NOT in ARMS "
                  f"and are therefore absent from this table:")
            for rel, n in unlisted:
                print(f"***   {rel}  ({n:,} rows)")
            print("*** Add them to ARMS, or to _NOT_ARMS if they are deliberately "
                  "excluded.")

        if not arms:
            print(f"  (no complete arm for {bench}; nothing to compare)")
            report[bench] = {"arms": {}, "partial": {k: v["n"] for k, v in partial.items()}}
            continue

        # Compare the NEWEST arm present against the two references, so this
        # keeps working as cycles are added without editing the formatting.
        new = next((c for c in ("v6d-vllm", "v7c-vllm", "v7-vllm", "v6-vllm", "v6", "v5", "v4")
                      if c in arms), next(iter(arms)))
        cats = sorted(arms[new]["by_category"],
                      key=lambda c: -arms[new]["by_category"][c]["n"])
        print(f"\nper-category, v5 protocol:\n{'category':24s}{'n':>7s}"
              + "".join(f"{a:>11s}" for a in arms)
              + f"{new + '-v3':>9s}{new + '-zs':>9s}")
        for cat in cats:
            n = arms[new]["by_category"][cat]["n"]
            row = f"{cat:24s}{n:7d}"
            acc = {}
            for a in arms:
                e = arms[a]["by_category"].get(cat)
                acc[a] = e["v5_pct"] if e else float("nan")
                row += f"{acc[a]:10.1f}%" if e else f"{'—':>11s}"
            ref = 'v3-corr' if 'v3-corr' in acc else new
            print(row + f"{acc[new] - acc[ref]:+8.1f}"
                        f"{acc[new] - acc['zero-shot']:+9.1f}")
        # Generation categories, reported ADDITIONALLY — never folded into the
        # headline. Exact match gives 0.00% for every arm here, which is a
        # metric artifact; token-F1 says what the arms actually do.
        gen_cats = sorted({c for a in arms for c in arms[a].get("generation_f1", {})})
        if gen_cats:
            print(f"\ngeneration categories — token-F1 (ADDITIONAL metric; the")
            print(f"headline above stays exact match, which reads 0.00% for all arms):")
            print(f"{'category':24s}{'n':>7s}" + "".join(f"{a:>11s}" for a in arms))
            for cat in gen_cats:
                e0 = next((arms[a]["generation_f1"][cat] for a in arms
                           if cat in arms[a].get("generation_f1", {})), None)
                row_g = f"{cat:24s}{e0['n'] if e0 else 0:7d}"
                for a in arms:
                    e = arms[a].get("generation_f1", {}).get(cat)
                    row_g += f"{e['token_f1']:10.2f}%" if e else f"{'—':>11s}"
                print(row_g)

        row = f"{'OVERALL':24s}{arms[new]['n']:7d}"
        for a in arms:
            row += f"{arms[a]['v5_pct']:10.2f}%"
        print("-" * (31 + 11 * len(arms)))
        # TRAM has no v3-corr arm (v3-corrected was never run on TRAM), so the
        # v3 delta/z-test must be optional. Before 2026-09-09 these lines
        # indexed arms['v3-corr'] unconditionally and the whole rescore died
        # with KeyError on the TRAM section, taking the TIME and TimeBench
        # tables down with it even though both had already been computed.
        has_v3 = "v3-corr" in arms
        print(row
              + (f"{arms[new]['v5_pct'] - arms['v3-corr']['v5_pct']:+8.2f}"
                 if has_v3 else f"{'—':>8s}")
              + f"{arms[new]['v5_pct'] - arms['zero-shot']['v5_pct']:+9.2f}")

        nw, zs = arms[new], arms["zero-shot"]
        if has_v3:
            v3 = arms["v3-corr"]
            print(f"\n{new} vs v3-corrected: z = "
                  f"{ztest(nw['v5_correct'], nw['n'], v3['v5_correct'], v3['n']):+.2f}")
        else:
            print(f"\n{new} vs v3-corrected: n/a (v3-corrected has no {bench} run)")
        print(f"{new} vs zero-shot   : z = "
              f"{ztest(nw['v5_correct'], nw['n'], zs['v5_correct'], zs['n']):+.2f}"
              f"  (unpaired -- conservative on this design; McNemar below)")

        # ---- the artifact-free and paired views (2026-09-12 audit) --------
        # Every arm vs zero-shot on the SAME items, with the abstain-option
        # bucket removed. This is the comparison D49 requires and it is not
        # the same call as the headline one: v6/v7c lead zero-shot overall
        # and trail it here.
        print(f"\n{'arm':12s}{'headline':>10s}{'no-abstain':>12s}"
              f"{'d(headline)':>13s}{'d(no-abst)':>12s}{'McNemar z':>11s}")
        for a, st in arms.items():
            if a == "zero-shot":
                continue
            mcn = ""
            if zs_correct is not None and st.get("_paired") is not None:
                b, c = st["_paired"]
                z, _chi = mcnemar(b, c)
                mcn = f"{z:+10.2f}"
            print(f"{a:12s}{st['v5_pct']:9.2f}%{st['no_abstain_pct']:11.2f}%"
                  f"{st['v5_pct'] - zs['v5_pct']:+12.2f}"
                  f"{st['no_abstain_pct'] - zs['no_abstain_pct']:+11.2f}"
                  f"{mcn:>11s}")
        # ---- per evaluation condition (TIME only) ------------------------
        conds = sorted({c for a in arms.values() for c in a.get("by_setting", {})})
        if conds:
            zsb = arms["zero-shot"]["by_setting"]
            print(f"\nby retrieval setting, abstain bucket EXCLUDED "
                  f"(delta vs zero-shot):")
            print(f"{'arm':12s}" + "".join(f"{c:>12s}" for c in conds))
            print(f"{'zero-shot':12s}"
                  + "".join(f"{zsb[c]['no_abstain_pct']:11.2f}%" for c in conds))
            for a, st in arms.items():
                if a == "zero-shot":
                    continue
                row = f"{a:12s}"
                for c in conds:
                    e = st.get("by_setting", {}).get(c)
                    row += (f"{e['no_abstain_pct'] - zsb[c]['no_abstain_pct']:+12.2f}"
                            if e else f"{'—':>12s}")
                print(row)
            print("  base = gold context; bm25/vector/hybrid = retriever-supplied.")
            print("  A POOLED TIME NUMBER AVERAGES THESE AND CANCELS THEM — see")
            print("  docs/results_and_methodology.md §1.0.")

        print("  headline = published v5-protocol number (unchanged).")
        print("  no-abstain = same protocol, abstain-option bucket excluded "
              "for every arm alike.")
        if new == "v5":
            v4a = arms["v4"]
            print(f"v5 vs v4          : z = "
                  f"{ztest(nw['v5_correct'], nw['n'], v4a['v5_correct'], v4a['n']):+.2f}"
                  f"  ({nw['v5_pct'] - v4a['v5_pct']:+.2f}pp — v5's own variable)")
        report[bench] = arms

    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
