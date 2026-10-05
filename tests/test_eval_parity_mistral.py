"""Train/eval prompt parity for Mistral (ported 2026-09-28 from
tests/test_eval_parity.py, which covers LLaMA only).

experiments/finetuning/shared/eval_parity.py is model-agnostic -- it only
calls tokenizer.apply_chat_template on the shared zero-shot builder's output
-- so the same contract (a training row renders to the SAME string
run_eval_vllm.py builds for the analogous benchmark item) must hold for
Mistral too. Two differences from the LLaMA tests, both intentional:
  - Mistral's eos token is "</s>", not "<|eot_id|>".
  - Mistral-7B-Instruct-v0.3's chat template has no "Today Date" field at
    all (verified 2026-09-28), so date pinning is a no-op for this model --
    tested here as "the string never appears", not "it appears pinned".
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from experiments.finetuning.shared.eval_parity import (  # noqa: E402
    PINNED_DATE, render_prompt, row_to_example,
)
from scripts.run_baselines import _build_zero_shot_prompt  # noqa: E402
from src.data.data_loader import TemporalExample  # noqa: E402

MODEL = ROOT / "models" / "Mistral-7B-Instruct-v0.3"


@pytest.fixture(scope="module")
def tok():
    if not MODEL.exists():
        pytest.skip("local Mistral tokenizer not present")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(MODEL))


def eval_side(tok, ex: TemporalExample) -> str:
    # Mirrors run_eval_vllm.encode with --system-prompt none --date-string.
    return tok.apply_chat_template(
        [{"role": "user", "content": _build_zero_shot_prompt(ex)}],
        tokenize=False, add_generation_prompt=True, date_string=PINNED_DATE)


def test_mcq_row_renders_like_a_loaded_mcq_item(tok):
    row = {"category": "Co_temporality"}
    q = "While X was chair, what role did Y hold?\nChoices:\nA. clerk\nB. bursar\nC. dean"
    train_ex = row_to_example(row, q, "Some passage.", "bursar")
    bench_ex = TemporalExample(id="t", question="While X was chair, what role did Y hold?",
                               answer="B", context="Some passage.",
                               choices=["clerk", "bursar", "dean"], task="Co_temporality")
    assert render_prompt(tok, train_ex) == eval_side(tok, bench_ex)


def test_nli_saq_row_takes_trams_premise_hypothesis_layout(tok):
    row = {"category": "nli_saq"}
    q = "Premise: The dam opened in 1961.\nHypothesis: The dam was open by 1970."
    train_ex = row_to_example(row, q, "[1] Title: unrelated passage", "entailment")
    bench_ex = TemporalExample(id="t", question="The dam was open by 1970.",
                               answer="entailment", context="The dam opened in 1961.",
                               task="nli_saq", temporal_type="temporal_nli")
    got = render_prompt(tok, train_ex)
    assert got == eval_side(tok, bench_ex)
    assert "Premise:\nThe dam opened in 1961." in got


def test_date_string_has_no_effect_on_mistral(tok):
    # Mistral-7B-Instruct-v0.3's template has no date field (verified
    # 2026-09-28) -- confirm the pin doesn't leak in some other form and
    # that rendering is unaffected by which date_string is passed.
    ex = row_to_example({}, "When?", "", "1990")
    got_pinned = render_prompt(tok, ex, date_string=PINNED_DATE)
    got_other = render_prompt(tok, ex, date_string="1 Jan 2000")
    assert PINNED_DATE not in got_pinned
    assert got_pinned == got_other
    assert "You are a temporal reasoning assistant specialized" not in got_pinned


def test_parity_labels_are_exactly_the_answer_plus_eos(tok):
    from experiments.finetuning.Mistral.data_loader import tokenize_example_parity
    row = {"category": "Localization", "source_dataset": "AUG_GLM2"}
    norm = {"question": "When did it open?", "context": "It opened on May 1, 1990.",
            "answer": "May 1, 1990", "answer_parts": ["May 1, 1990"]}
    out, why = tokenize_example_parity(row, norm, tok, 2048)
    assert why == "ok"
    kept = [t for t, l in zip(out["input_ids"], out["labels"]) if l != -100]
    assert tok.decode(kept) == "May 1, 1990</s>"
    prompt = render_prompt(tok, row_to_example(row, norm["question"], norm["context"],
                                               norm["answer"]))
    n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
    assert out["labels"][:n_prompt] == [-100] * n_prompt


def test_row_whose_gold_is_truncated_away_is_dropped(tok):
    from experiments.finetuning.Mistral.data_loader import tokenize_example_parity
    ctx = ("Filler sentence number one. " * 400) + "The answer is Zanzibar."
    norm = {"question": "Where?", "context": ctx, "answer": "Zanzibar",
            "answer_parts": ["Zanzibar"]}
    out, why = tokenize_example_parity({"source_dataset": "TimeQA"}, norm, tok, 512)
    assert out is None and why == "gold_truncated"
