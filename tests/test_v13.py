"""v13 tests: prog-generator guarantees + the extended ingest gate.

Covers the three things most likely to silently corrupt a data wave:
  1. every AUG_PROG row's gold re-derives from its text (verifier round
     trip), on a fresh generator run at smoke scale;
  2. the ingest gate's new `extract` path accepts a correct multi-select
     row and rejects the three real failure shapes (wrong join, correct
     option missing from context, distractor present in context);
  3. the v13 GLM brief carries the two new/extended cards.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from random import Random

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import importlib.util

spec = importlib.util.spec_from_file_location(
    "generate_v13_prog_data", ROOT / "scripts" / "generate_v13_prog_data.py")
prog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prog)

import ingest_glm_batch as ingest  # noqa: E402


def _rows(cat: str, n: int = 25):
    gen = prog.GENS[cat]
    return gen(Random(f"test-{cat}"), n, prog.letter_cycle_abc())


def test_prog_rows_verify_roundtrip():
    for cat in prog.GENS:
        rows = _rows(cat)
        assert rows, cat
        for r in rows:
            assert r["targets"] and isinstance(r["targets"], list)
            assert len(r["targets"]) == 1
            assert prog.VERIFIERS[cat](r), (cat, r["question"][:80],
                                             r["targets"])


def test_prog_mcq_letters_cycle():
    rows = _rows("prog_relation", 60)
    letters = [prog.split_options(r["question"]) and
               chr(65 + prog.split_options(r["question"]).index(r["targets"][0]))
               for r in rows if "Choices:" in r["question"]]
    assert letters, "expected MCQ relation rows"
    # cycling A,B,C means no letter exceeds 60%
    from collections import Counter
    top = Counter(letters).most_common(1)[0]
    assert top[1] / len(letters) <= 0.6


def test_prog_extract_gold_shape():
    import re
    for r in _rows("prog_extract"):
        gold = r["targets"][0]
        assert re.fullmatch(r"[A-E](  [A-E])*", gold), gold
        letters = gold.split("  ")
        assert letters == sorted(set(letters))


def _extract_row(gold="B  C", ctx_extra="yes"):
    ctx = ("Session 1 happened at 12:04 am on 18 January, 2020.\n"
           "A: Hey! How are you?\nB: We booked the venue on January 27, "
           "2020, and the caterer confirmed the next day by phone.\n"
           "A: Perfect. My sister arrives on the morning train and I still "
           "need to pick up the decorations before then.\nB: The rental "
           "company said everything would be ready a week ahead of the "
           "event, so there is slack in the schedule if anything slips.\n"
           "A: Good. Last time the delivery came late and we had to "
           "improvise the whole afternoon.\nB: Then let us confirm the "
           "final numbers early and send everyone a reminder note.")
    q = ("Which of the following are time expressions mentioned in the "
         "context? (Note: There may be one or more correct options. And the "
         "time expressions are mentioned directly or indirectly in the "
         "context.)\nChoices:\nA. March 3, 2021\nB. 18 January, 2020\n"
         "C. January 27, 2020\nD. February 7, 2020")
    return {"source_dataset": "AUG_GLM2", "slice": "C", "category": "extract",
            "provenance": "dial", "question": q, "context": ctx,
            "targets": [gold], "rationale": "A and C are mentioned; the "
            "others are not.", "source": "augmented", "source_id": ""}


def test_ingest_extract_accepts_valid_row():
    assert ingest.check(_extract_row()) == []


def test_ingest_extract_rejects_wrong_join():
    assert any("two-space" in e for e in ingest.check(_extract_row("B, C")))


def test_ingest_extract_rejects_correct_option_missing_from_context():
    r = _extract_row("A  B")
    r["context"] = r["context"].replace("18 January, 2020", "19 January, 2020")
    errs = ingest.check(r)
    assert any("not found in context" in e for e in errs), errs


def test_ingest_extract_rejects_distractor_in_context():
    r = _extract_row("B  C")
    r["context"] += " See you on March 3, 2021!"
    errs = ingest.check(r)
    assert any("appears in context" in e for e in errs), errs


def test_ingest_accepts_bare_relation_row():
    """v13's event-to-time relation shape is bare (TRAM's own surface);
    the gate must accept it and still validate the label space."""
    r = {"source_dataset": "AUG_GLM2", "slice": "B", "category": "relation",
         "provenance": "none",
         "question": ("The festival ran from January 5, 2018 to April 22, "
                      "2018, the longest in its history. KUALA LUMPUR, "
                      "February 2018 (AFP). What is the relationship between "
                      "the event 'ran' and the time 'February 2018'?"),
         "context": "", "targets": ["IS_INCLUDED"],
         "rationale": "The event's span contains February 2018.",
         "source": "augmented", "source_id": ""}
    assert ingest.check(r) == []
    bad = dict(r)
    bad["targets"] = ["DURING"]     # TRAM golds only, per the v12 correction
    assert any("gold must be one of" in e for e in ingest.check(bad))


def test_v13_master_has_new_cards():
    spec2 = importlib.util.spec_from_file_location(
        "make_v13_glm_orders", ROOT / "scripts" / "make_v13_glm_orders.py")
    m = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(m)
    master = m.build_master_v13()
    assert "### extract" in master
    assert "EVENT-TO-TIME" in master
    assert "TWO SPACES" in " ".join(master.split())
    total = sum(r for _, r, _, _ in m.PLAN)
    assert total == 4500, total
