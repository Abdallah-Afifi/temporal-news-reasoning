"""Train/eval prompt parity (experiments/finetuning/shared/eval_parity.py).

The defect these guard against: every arm through v10-glm was trained on one
prompt and evaluated on another (audit 2026-09-23 §1). The contract is that a
training row renders to the SAME string run_eval_vllm.py builds for the
analogous benchmark item, so zero-shot and fine-tuned differ only in weights.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from experiments.finetuning.shared.eval_parity import (  # noqa: E402
    PINNED_DATE, _split_choices, render_prompt, row_to_example,
)
from scripts.run_baselines import _build_zero_shot_prompt  # noqa: E402
from src.data.data_loader import TemporalExample  # noqa: E402

MODEL = ROOT / "models" / "Llama-3.2-3B-Instruct"


@pytest.fixture(scope="module")
def tok():
    if not MODEL.exists():
        pytest.skip("local Llama tokenizer not present")
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
    assert "Answer with exactly one word: Entailment, Contradiction, or Neutral." in got


def test_nli_mcq_row_keeps_its_choices_like_tram_nli_mcq(tok):
    row = {"category": "nli_mcq"}
    q = ("Premise: A.\nHypothesis: B.\nChoices:\nA. entailment\nB. neutral\n"
         "C. contradiction")
    ex = row_to_example(row, q, "", "neutral")
    assert ex.choices == ["entailment", "neutral", "contradiction"]
    assert (ex.context, ex.question) == ("A.", "B.")


def test_timeline_keeps_its_own_format_instruction(tok):
    q = ("Below are 3 facts. Requirements: You must output a sequence of uppercase "
         "letters separated by commas, such as 'A,B,C'.\nChoices:\nA. x\nB. y\nC. z")
    got = render_prompt(tok, row_to_example({"category": "Timeline"}, q, "ctx", "A,B,C"))
    assert "Respond with the option text" not in got
    assert "Choices:\nA. x\nB. y\nC. z" in got


def test_date_is_pinned_not_wall_clock(tok):
    got = render_prompt(tok, row_to_example({}, "When?", "", "1990"))
    assert f"Today Date: {PINNED_DATE}" in got
    assert "You are a temporal reasoning assistant specialized" not in got  # no legacy system prompt


def test_malformed_choice_block_is_left_untouched():
    q = "Q?\nChoices:\nA. one\nC. skipped a letter"
    assert _split_choices(q) == (q, None)


def test_parity_labels_are_exactly_the_answer_plus_eot(tok):
    from experiments.finetuning.LLaMA.data_loader import tokenize_example_parity
    row = {"category": "Localization", "source_dataset": "AUG_GLM2"}
    norm = {"question": "When did it open?", "context": "It opened on May 1, 1990.",
            "answer": "May 1, 1990", "answer_parts": ["May 1, 1990"]}
    out, why = tokenize_example_parity(row, norm, tok, 2048)
    assert why == "ok"
    kept = [t for t, l in zip(out["input_ids"], out["labels"]) if l != -100]
    assert tok.decode(kept) == "May 1, 1990<|eot_id|>"
    prompt = render_prompt(tok, row_to_example(row, norm["question"], norm["context"],
                                               norm["answer"]))
    n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
    assert out["labels"][:n_prompt] == [-100] * n_prompt


def test_row_whose_gold_is_truncated_away_is_dropped(tok):
    from experiments.finetuning.LLaMA.data_loader import tokenize_example_parity
    ctx = ("Filler sentence number one. " * 400) + "The answer is Zanzibar."
    norm = {"question": "Where?", "context": ctx, "answer": "Zanzibar",
            "answer_parts": ["Zanzibar"]}
    out, why = tokenize_example_parity({"source_dataset": "TimeQA"}, norm, tok, 512)
    assert out is None and why == "gold_truncated"
