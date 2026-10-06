"""Quality filters of scripts/build_v13_training_data.py (audit 2026-10-04)."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_v13_training_data as b  # noqa: E402


def rel(q_events: str, gold: str) -> dict:
    return {"category": "relation", "targets": [gold],
            "question": q_events + " What is the relationship between the events?\n"
                        "Choices:\nA. BEFORE\nB. AFTER\nC. SIMULTANEOUS"}


def story(gold: str, other: str) -> dict:
    return {"category": "storytelling", "targets": [gold],
            "question": f"Which ending?\nChoices:\nA. {gold}\nB. {other}"}


def test_relation_interval_and_event_to_time():
    assert b.computed_relation(
        "The campaign ran from 4 January 2016 to 29 April 2016. "
        "The registry was updated on 12 March 2016. What is the relationship") == "INCLUDES"
    assert b.computed_relation(
        "The drive collected 340 units on 3 February 2016. What is the relationship "
        "between the event 'collected' and the time 'March 2016'?") == "BEFORE"
    assert b.computed_relation(
        "On 3 August 2019 two things happened the same day. What is the relationship") is None


def test_prog_timeline_year_only_collision_and_duration_compare():
    under = {"category": "prog_timeline", "targets": ["C,A,B"],
             "question": "Sort.\nA. The opera house recorded its survey in 1947.\n"
                         "B. The expansion on Oct 8, 1953.\n"
                         "C. The rail operator recorded its inspection in 1947."}
    mixed = {"category": "prog_timeline", "targets": ["B,C,A"],
             "question": "Sort.\nA. The library recorded its opening in 1979.\n"
                         "B. The inspection on November 6, 1979.\n"
                         "C. The restoration on December 15, 1985."}
    ok = {"category": "prog_timeline", "targets": ["A,B"],
          "question": "Sort.\nA. The library recorded its opening in 1970.\n"
                      "B. The inspection on November 6, 1979."}
    incoherent = {"category": "prog_duration_compare", "targets": ["x"],
                  "question": "*Duration 1:* Between the opening of X and Y. "
                              "*Duration 2:* Between the closing of Z and W."}
    dated = {"category": "prog_duration_compare", "targets": ["x"],
             "question": "*Duration 1:* Between Apr 9, 2016 and 21 April 2021. "
                         "*Duration 2:* Between December 19, 2021 and Dec 22, 2023."}
    c = Counter()
    assert b.filter_prog_rows([under, mixed, ok, incoherent, dated], c) == [ok, dated]
    assert c["prog_timeline_year_only_collision"] == 2
    assert c["prog_duration_compare_event_named_incoherent"] == 1


def test_timeqa_cot_skips_are_excluded(tmp_path):
    raw = tmp_path / "cot_raw"
    raw.mkdir()
    (raw / "001.txt").write_text('{"id": "g0001", "cot": "..."}\n'
                                 "# SKIP g0002: passage contradicts gold\n")
    key = tmp_path / "answer_key.json"
    key.write_text('{"g0001": {"question": "q1", "context": "c1", "gold": "a"},'
                   ' "g0002": {"question": "q2", "context": "c2", "gold": "b"}}')
    keys, skips = b.cot_skip_keys(raw, key)
    assert keys == {("q2", "c2")} and list(skips) == ["g0002"]
    rows = [{"source_dataset": "TimeQA", "question": "q1", "context": "c1"},
            {"source_dataset": "TimeQA", "question": "q2", "context": "c2"},
            {"source_dataset": "TLQA", "question": "q2", "context": "c2"}]
    c = Counter()
    kept = b.filter_base_rows(rows, keys, c, "train")
    assert [r["question"] for r in kept] == ["q1", "q2"] and kept[1]["source_dataset"] == "TLQA"
    assert c["timeqa_cot_skip_wrong_gold_train"] == 1


def test_build_refuses_without_template_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "TPL_NEW", tmp_path / "empty")
    monkeypatch.delenv("V13_ALLOW_NO_GLM", raising=False)
    monkeypatch.setattr(sys, "argv", ["build_v13_training_data.py"])
    with pytest.raises(SystemExit, match="FATAL: no rows"):
        b.main()


# --- template-source rule (researcher decision 2026-10-05) -------------------
def test_non_math_template_rows_are_removed_math_kept():
    rows = [{"source_dataset": "AUG_TPL3", "category": c, "targets": ["x"], "question": f"q {c}"}
            for c in ("relation", "storytelling", "duration", "nli_saq", "extract",
                      "Computation", "Timeline", "Order_Compare")]
    c = Counter()
    kept = b.filter_template_rows(rows, c)
    assert {r["category"] for r in kept} == {"Computation", "Timeline", "Order_Compare"}
    assert c["tpl_non_math_template_removed_relation"] == 1
    assert c["tpl_non_math_template_removed_storytelling"] == 1


def test_glm_written_non_math_rows_are_kept():
    rows = [{"source_dataset": "AUG_GLM3", "category": c, "targets": ["x"], "question": f"g {c}"}
            for c in ("nli_saq", "storytelling", "extract")]
    assert len(b.filter_template_rows(rows, Counter())) == 3


def test_math_template_rows_failing_independent_verify_are_dropped():
    ok = {"category": "Computation", "targets": ["24 days"], "question": "good"}
    bad = {"category": "Computation", "targets": ["25 days"], "question": "bad"}
    c = Counter()
    assert b.filter_template_rows([ok, bad], c, failed={"bad"}) == [ok]
    assert c["tpl_math_failed_independent_verify_Computation"] == 1


def test_script_built_base_rows_non_math_removed_math_tagged():
    scripted = {"s-nli": "nli_mcq", "s-oc": "Order_Compare", "s-bad": "Duration_Compare"}
    rows = [{"source_dataset": "AUG_GLM2", "category": "nli_mcq", "question": "s-nli"},
            {"source_dataset": "AUG_GLM2", "category": "Order_Compare", "question": "s-oc"},
            {"source_dataset": "AUG_GLM2", "category": "Duration_Compare", "question": "s-bad"},
            {"source_dataset": "AUG_GLM2", "category": "nli_mcq", "question": "llm-written"}]
    c = Counter()
    kept = b.filter_base_rows(rows, set(), c, "train", scripted, failed={"s-bad"})
    assert [r["question"] for r in kept] == ["s-oc", "llm-written"]
    assert kept[0]["provenance"] == "agent-template"
    assert c["v12_script_built_non_math_removed_train"] == 1
    assert c["v12_script_built_math_failed_verify_train"] == 1


def test_prog_rows_failing_independent_verify_are_dropped():
    rows = [{"category": "prog_computation", "question": "p1"},
            {"category": "prog_computation", "question": "p2"}]
    c = Counter()
    assert [r["question"] for r in b.filter_prog_rows(rows, c, failed={"p2"})] == ["p1"]



def test_any_base_row_failing_verify_is_dropped():
    rows = [{"source_dataset": "AUG_SEQ", "question": "seq-bad"},
            {"source_dataset": "TimeQA", "question": "tq-ok", "context": ""}]
    c = Counter()
    kept = b.filter_base_rows(rows, set(), c, "val", {}, failed={"seq-bad"})
    assert [r["question"] for r in kept] == ["tq-ok"]
    assert c["v12_failed_verify_or_review_AUG_SEQ_val"] == 1
