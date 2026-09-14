"""Few-shot CoT prompt tests (the few_shot_cot arm, D61).

Covers: exemplar file validation, prompt construction (exemplars present,
target LAST, anchor instruction intact), NLI-awareness, and that the
exemplar file itself never changed shape.
"""

import json
from pathlib import Path

import pytest

from scripts.run_baselines import (
    COT_SYSTEM_PROMPT,
    _build_cot_fewshot_prompt,
    _build_cot_prompt,
    _format_cot_exemplar,
    _load_cot_exemplars,
)

REPO = Path(__file__).resolve().parents[1]
EXEMPLAR_FILE = REPO / "data" / "cot_fewshot_exemplars.json"


from src.data.data_loader import TemporalExample  # noqa: E402


def _ex(question, context=None, choices=None, task="Order_Compare", temporal_type=None):
    return TemporalExample(
        id="t",
        question=question,
        answer="unused",
        context=context,
        choices=choices,
        temporal_type=temporal_type,
        task=task,
        source="timebench",
    )


def test_exemplar_file_loads_and_validates():
    exs = _load_cot_exemplars(str(EXEMPLAR_FILE))
    assert len(exs) == 4
    kinds = {e.get("kind") for e in exs}
    assert kinds == {"phrase_qa", "ordering", "duration_option", "nli"}


def test_load_rejects_missing_file(tmp_path):
    with pytest.raises(Exception):
        _load_cot_exemplars(str(tmp_path / "nope.json"))


def test_load_rejects_bad_schema(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"exemplars": [{"question": "q"}]}))
    with pytest.raises(Exception):
        _load_cot_exemplars(str(p))


def test_exemplar_render_contains_steps_and_answer():
    exs = _load_cot_exemplars(str(EXEMPLAR_FILE))
    rendered = _format_cot_exemplar(exs[0])
    assert "Step 1:" in rendered
    assert rendered.rstrip().endswith(f"ANSWER: {exs[0]['answer']}")


def test_fewshot_prompt_has_exemplars_then_target_last():
    exs = _load_cot_exemplars(str(EXEMPLAR_FILE))
    target = _ex("When did the bridge open?", context="The bridge opened in 2019.")
    prompt = _build_cot_fewshot_prompt(target, exemplars=exs)
    for ex in exs:
        assert ex["question"] in prompt
    assert "Worked examples:" in prompt
    assert "Now solve this one, the same way." in prompt
    # target question appears AFTER every exemplar question (recency)
    pos = [prompt.rindex(ex["question"]) for ex in exs]
    assert all(p < prompt.rindex("When did the bridge open?") for p in pos)


def test_fewshot_prompt_keeps_cot_anchor_and_choices():
    exs = _load_cot_exemplars(str(EXEMPLAR_FILE))
    target = _ex("Which is longer?", choices=["300 days", "eleven months"])
    prompt = _build_cot_fewshot_prompt(target, exemplars=exs)
    assert "ANSWER: <answer>" in prompt  # the _COT_FINISH anchor survives
    assert "A. 300 days" in prompt and "B. eleven months" in prompt


def test_fewshot_target_block_equals_cot_builder_plus_prefix():
    """The target block must be byte-identical to the cot arm's prompt."""
    exs = _load_cot_exemplars(str(EXEMPLAR_FILE))
    target = _ex("Hypothesis-style Q?", context="Premise text.")
    fewshot = _build_cot_fewshot_prompt(target, exemplars=exs)
    cot = _build_cot_prompt(target)
    assert fewshot.endswith(cot)


def test_nli_target_gets_nli_layout():
    exs = _load_cot_exemplars(str(EXEMPLAR_FILE))
    target = _ex("The event happened before 2020.", context="It happened in 2018.")
    prompt = _build_cot_fewshot_prompt(target, exemplars=exs)
    # _build_cot_prompt routes NLI-ish contexts through Premise/Hypothesis
    # (its own test pins that); here we only require the persona + anchor
    # survive and the NLI exemplar's Contradiction trace is visible.
    assert "ANSWER: Contradiction" in prompt.split("Now solve this one")[0]


def test_cot_system_prompt_unchanged_for_fewshot_arm():
    # The few-shot arm shares the STaR teacher persona (prompt-family lock).
    assert "expert temporal-reasoning teacher" in COT_SYSTEM_PROMPT
