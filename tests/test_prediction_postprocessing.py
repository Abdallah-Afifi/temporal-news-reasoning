"""Regression tests for MCQ letter extraction in the evaluation postprocessor.

The original rule mapped a prediction to a choice whenever its first character
was A-D, so free-text answers like "Aralvaimozhi" or "December 1994" were
rewritten into options the model never named. The rewritten string was what
got stored -- the raw generation was discarded -- so the corruption could only
be undone by re-running inference. These tests pin the letter rule, and the
loader now persists ``raw_prediction`` so a future change is a rescore.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_baselines import _postprocess_prediction, _prediction_record

CHOICES = ["Cherthala", "Berlin", "Nuclear deal with Iran", "December 1994"]


def test_word_beginning_with_option_letter_is_not_a_letter_answer() -> None:
    assert _postprocess_prediction("Aralvaimozhi", CHOICES) == "Aralvaimozhi"
    assert _postprocess_prediction("December 1994", CHOICES) == "December 1994"


def test_leading_article_is_not_option_a() -> None:
    assert _postprocess_prediction("A man arrived on Tuesday", CHOICES) == (
        "A man arrived on Tuesday"
    )


def test_timeline_letter_sequence_is_left_alone() -> None:
    """Timeline golds are ordered letters; they must not collapse to one option."""
    assert _postprocess_prediction("C,B,A", CHOICES) == "C,B,A"


def test_bare_option_letter_maps_to_choice_text() -> None:
    for raw in ("B", "(B)", "B.", "b"):
        assert _postprocess_prediction(raw, CHOICES) == "Berlin", raw


def test_letter_prefix_maps_to_choice_text() -> None:
    assert _postprocess_prediction("B) Berlin", CHOICES) == "Berlin"
    assert _postprocess_prediction("B - Berlin", CHOICES) == "Berlin"


def test_exact_choice_text_wins_over_letter_reading() -> None:
    """A choice that itself starts with an option letter is answer text."""
    assert _postprocess_prediction("December 1994", CHOICES) == "December 1994"
    assert _postprocess_prediction("Cherthala", CHOICES) == "Cherthala"


def test_raw_prediction_is_persisted_when_supplied() -> None:
    class _Example:
        id = "TIME-1"
        source = "time"
        task = "MCQ"
        question = "q"
        context = ""
        temporal_type = "ordering"

    record = _prediction_record(_Example(), "Berlin", "Berlin", raw_prediction="B) Berlin")
    assert record["raw_prediction"] == "B) Berlin"
    assert record["prediction"] == "Berlin"


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------
from run_baselines import _build_zero_shot_prompt  # noqa: E402


class _Ex:
    def __init__(self, question, choices=None, context=""):
        self.question, self.choices, self.context = question, choices, context
        self.id, self.source, self.task, self.temporal_type = "x", "time", "t", "t"
        self.answer = ""


TIMELINE_Q = (
    "Below are 3 facts. You need to sort these facts in chronological order. "
    "Requirements: You must output a sequence of uppercase letters separated "
    "by commas, such as 'A,B,C', without any other characters."
)


def test_timeline_prompt_does_not_contradict_the_task() -> None:
    """The task asks for 'A,B,C'; a trailing single-letter instruction fought it."""
    prompt = _build_zero_shot_prompt(_Ex(TIMELINE_Q, ["fact one", "fact two", "fact three"]))
    assert "Choices:" in prompt
    assert "single letter" not in prompt


def test_ordinary_mcq_still_gets_the_generic_instruction() -> None:
    prompt = _build_zero_shot_prompt(_Ex("Which came first?", ["Alpha", "Beta"]))
    assert "Respond with the option text, or a single letter (A/B/C/D)." in prompt


def test_free_form_prompt_has_no_choices_block() -> None:
    prompt = _build_zero_shot_prompt(_Ex("When did it happen?"))
    assert "Choices:" not in prompt and "single letter" not in prompt


def test_category_falls_back_to_task_when_temporal_type_is_missing() -> None:
    """TIME Timeline items have task='Timeline' and temporal_type=None."""
    ex = _Ex("Sort these facts.", ["a", "b"])
    ex.temporal_type, ex.task = None, "Timeline"
    from run_baselines import _prediction_record
    assert _prediction_record(ex, "A,B", "A,B")["category"] == "Timeline"


# --- unambiguous partial naming of an option (added after the v4 post-mortem) ---
#
# v4 answers "Fact 1" where the option reads "Fact 1 happened earlier." — it has
# named the option unambiguously, but the string is SHORTER than the option, so
# neither the exact-match rule nor the containment rule can see it. Left
# unhandled this cost v4 21.2pp on TIME/Order_Compare (2.0pp overall) as a pure
# scoring artifact. The rule must stay strictly conservative: it fires only when
# the prediction prefixes exactly one option.

ORDER_CHOICES = [
    "Fact 1 happened earlier.",
    "Fact 2 happened earlier.",
    "They happened at the same time.",
]


def test_unambiguous_partial_option_resolves_to_that_option() -> None:
    assert _postprocess_prediction("Fact 1", ORDER_CHOICES) == "Fact 1 happened earlier."
    assert _postprocess_prediction("Fact 2", ORDER_CHOICES) == "Fact 2 happened earlier."


def test_ambiguous_partial_option_is_left_alone() -> None:
    """"Fact" prefixes two options, so it names neither."""
    assert _postprocess_prediction("Fact", ORDER_CHOICES) == "Fact"


def test_partial_option_rule_requires_five_characters() -> None:
    """Short stems can never resolve, so an article or bare token is safe."""
    assert _postprocess_prediction("They", ORDER_CHOICES) == "They"


def test_partial_option_rule_does_not_rewrite_free_text() -> None:
    """The v1/v2/v3 arms must be unaffected: no prefix, no rewrite."""
    assert _postprocess_prediction("Aralvaimozhi", CHOICES) == "Aralvaimozhi"
    assert _postprocess_prediction("December 1994", CHOICES) == "December 1994"
    assert _postprocess_prediction("Nuclear", CHOICES) == "Nuclear deal with Iran"


def test_partial_option_rule_never_fires_without_choices() -> None:
    assert _postprocess_prediction("Fact 1", None) == "Fact 1"


# --- "<option text>; B" self-declared choice (v5's dominant MCQ shape) -------


def test_trailing_letter_resolves_via_the_text_half_first() -> None:
    """The text names the option; the "; B" suffix must not defeat it."""
    assert (
        _postprocess_prediction("Nuclear deal with Iran; C", CHOICES)
        == "Nuclear deal with Iran"
    )


def test_trailing_letter_used_when_the_text_half_resolves_to_nothing() -> None:
    """v5 paraphrases the option, then names it. The letter is the answer."""
    assert _postprocess_prediction("some paraphrase of it; B", CHOICES) == "Berlin"


def test_trailing_letter_never_overrides_a_resolvable_text_answer() -> None:
    """Text and letter disagree (1.5% of cases): the text wins, as everywhere."""
    assert (
        _postprocess_prediction("Cherthala; D", CHOICES) == "Cherthala"
    )


def test_trailing_letter_requires_the_letter_to_be_last() -> None:
    """Prose with a mid-sentence semicolon is left completely alone."""
    assert (
        _postprocess_prediction("he left; then she arrived", CHOICES)
        == "he left; then she arrived"
    )
    assert _postprocess_prediction("a; b; c and d", CHOICES) == "a; b; c and d"


def test_trailing_letter_out_of_range_is_left_alone() -> None:
    two = ["Yes", "No"]
    assert _postprocess_prediction("unresolvable text; D", two) == "unresolvable text; D"


def test_trailing_letter_never_fires_without_choices() -> None:
    assert _postprocess_prediction("whatever; B", None) == "whatever; B"


def test_trailing_letter_does_not_rewrite_plain_free_text() -> None:
    """Zero-shot/v1/v2 emit no such shape; they must be untouched."""
    assert _postprocess_prediction("Aralvaimozhi", CHOICES) == "Aralvaimozhi"
    assert _postprocess_prediction("December 1994", CHOICES) == "December 1994"


def test_letters_only_stem_is_not_a_text_plus_letter_answer() -> None:
    """Some TIME golds are space-separated letter sequences ("A C").

    The model answers "A; C" there, so the semicolon separates two LETTERS —
    it is not a "<text>; <letter>" self-declaration. Reading the last letter
    as the choice threw away 114 correct v3-corrected answers before this
    guard existed.
    """
    assert _postprocess_prediction("A; C", CHOICES) == "A; C"
    assert _postprocess_prediction("A; B; C", CHOICES) == "A; B; C"
    assert _postprocess_prediction("C; D", CHOICES) == "C; D"
    assert _postprocess_prediction("(A); (B)", CHOICES) == "(A); (B)"


def test_guard_does_not_block_a_real_text_plus_letter_answer() -> None:
    assert _postprocess_prediction("some paraphrase; B", CHOICES) == "Berlin"
