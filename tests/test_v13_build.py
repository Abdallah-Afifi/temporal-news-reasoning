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


def test_relation_point_point_reversed_label_is_dropped():
    bad = rel("The union ratified its deal on 6 December 2018. "
              "Talks began on 11 September 2018.", "BEFORE")
    good = rel("The union ratified its deal on 6 December 2018. "
               "Talks began on 11 September 2018.", "AFTER")
    c = Counter()
    kept = b.filter_template_rows([bad, good], c)
    assert kept == [good]
    assert c["tpl_relation_label_disagrees_with_dates"] == 1


def test_relation_interval_and_event_to_time():
    assert b.computed_relation(
        "The campaign ran from 4 January 2016 to 29 April 2016. "
        "The registry was updated on 12 March 2016. What is the relationship") == "INCLUDES"
    assert b.computed_relation(
        "The drive collected 340 units on 3 February 2016. What is the relationship "
        "between the event 'collected' and the time 'March 2016'?") == "BEFORE"
    assert b.computed_relation(
        "On 3 August 2019 two things happened the same day. What is the relationship") is None


def test_relation_unparseable_rows_are_kept():
    r = rel("Two things happened on the same day.", "SIMULTANEOUS")
    assert b.filter_template_rows([r], Counter()) == [r]


def test_duration_broken_template_and_unit_only_options():
    broken = {"category": "duration", "targets": ["ten minutes"],
              "question": "How long did it take the speech to last?\nChoices:\n"
                          "A. ten minutes\nB. two hours"}
    unit_only = {"category": "duration", "targets": ["ten minutes"],
                 "question": "How long to read the letter?\nChoices:\nA. ten minutes\n"
                             "B. ten seconds\nC. ten hours\nD. ten years"}
    fine = {"category": "duration", "targets": ["eight years"],
            "question": "The partnership ran from 2006 to 2014. How long?\nChoices:\n"
                        "A. eight years\nB. nine years\nC. seven years"}
    c = Counter()
    assert b.filter_template_rows([broken, unit_only, fine], c) == [fine]
    assert c["tpl_duration_broken_template"] == 1
    assert c["tpl_duration_options_differ_only_by_unit"] == 1


def test_storytelling_gold_longer_share_forced_to_half():
    longer = [story(f"a much longer and more detailed ending number {i}", "short")
              for i in range(10)]
    shorter = [story("short", f"a much longer and more detailed distractor {i}")
               for i in range(3)]
    c = Counter()
    kept = b.filter_template_rows(longer + shorter, c)
    assert len(kept) == 6
    assert sum(b.story_gold_longer(r) for r in kept) == 3
    assert c["tpl_storytelling_gold_longer_dropped_for_50pct_balance"] == 7
    assert kept == b.filter_template_rows(longer + shorter, Counter())  # deterministic


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
