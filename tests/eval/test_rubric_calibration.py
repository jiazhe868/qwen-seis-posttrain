"""Agreement of a rubric judge with gold labels (dpt.eval.rubric_calibration, 2026-11-15). All data here is made up."""

from __future__ import annotations

import json

import pytest

from dpt.common.io import write_jsonl
from dpt.eval.rubric_calibration import (answer_pairs, confusion, criterion_pairs, derived_scores, format_comparison,
                                         gate_index, level_table, main, same_question_order, spearman, split_of,
                                         summarize)


def test_confusion_counts_false_passes_and_false_fails():
    c = confusion([(True, True), (True, False), (False, False), (False, False), (False, True)])
    assert (c["tp"], c["fp"], c["tn"], c["fn"]) == (1, 1, 2, 1)
    assert c["fp_rate"] == pytest.approx(1 / 3) and c["fn_rate"] == pytest.approx(1 / 2)
    assert c["balanced_accuracy"] == pytest.approx(1 - (1 / 3 + 1 / 2) / 2)
    assert confusion([(True, True), (False, False)])["kappa"] == pytest.approx(1.0)
    assert confusion([])["n"] == 0 and confusion([])["kappa"] is None


def test_the_split_is_fixed_by_the_question_id():
    ids = [f"d_report-{i:04d}" for i in range(400)]
    assert [split_of(i) for i in ids] == [split_of(i) for i in ids]
    assert 0.5 < sum(split_of(i) == "tune" for i in ids) / len(ids) < 0.7


def test_spearman_handles_ties_and_order():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 2], [5, 5, 6]) == pytest.approx(1.0) and spearman([1, 2], [1, 2]) is None


def test_partly_and_flagged_criteria_are_set_apart():
    labels = {("sft", "q"): {"criteria": [{"id": 1, "met": "yes"}, {"id": 2, "met": "no"}, {"id": 3, "met": "partly"},
                                          {"id": 4, "met": "no"}], "criterion_issues": [{"id": 4, "issue": "wrong"}]}}
    scores = {("sft", "q"): {"kind": "criteria", "per_criterion": [8, 7, 9, 2]}}
    pairs, apart = criterion_pairs(labels, scores, 7, [("sft", "q")])
    assert pairs == [(True, True), (True, False)] and apart == {"partly": 1, "criterion_issue": 1, "missing": 0}
    pairs, apart = criterion_pairs(labels, scores, 7, [("sft", "q")], partly_as_no=True)
    assert pairs == [(True, True), (True, False), (True, False)] and apart["partly"] == 0
    checks = {("sft", "q"): {"kind": "checks", "per_criterion": [1, 0, 1, 1]}}
    assert criterion_pairs(labels, checks, 7, [("sft", "q")])[0] == [(True, True), (False, False)]


def test_answers_pass_at_the_mark_and_unanswered_ones_fail_on_both_sides():
    labels = {("a", "q"): {"acceptable": True, "score": 80}, ("b", "q"): {"no_answer": True, "acceptable": False,
                                                                          "score": 0}}
    scores = {("a", "q"): {"score": 0.72}, ("b", "q"): {"score": 0.0, "no_answer": True}}
    pairs, js, gs = answer_pairs(labels, scores, 70, [("a", "q"), ("b", "q")])
    assert pairs == [(True, True), (False, False)] and js == [pytest.approx(72.0), 0.0] and gs == [80.0, 0.0]
    assert answer_pairs(labels, scores, 70, [("a", "q")], gold_cut=85)[0] == [(True, False)]


def test_summarize_chooses_thresholds_on_the_tune_split_and_reports_both(tmp_path):
    """A lenient judge (v1: every criterion 9/10) against one that follows the gold marks (v2)."""
    items, labels, scores = [], [], []
    for k in range(40):
        iid = f"d_report-{k:03d}"
        items.append({"id": iid, "answer": {"rubric": [{"criterion": "A", "weight": 2}, {"criterion": "B", "weight": 1}]}})
        good = k % 2 == 0
        labels.append({"row": "sft", "id": iid, "acceptable": good, "score": 80 if good else 40,
                       "criteria": [{"id": 1, "met": "yes" if good else "no"}, {"id": 2, "met": "yes" if good else "no"}]})
        scores.append({"variant": "v1", "kind": "criteria", "row": "sft", "id": iid, "score": 0.9, "per_criterion": [9, 9]})
        follow = [9, 9] if good else [2, 2]
        scores.append({"variant": "v2", "kind": "criteria", "row": "sft", "id": iid,
                       "score": sum(follow) / 20, "per_criterion": follow})
    res = summarize(labels, scores, items)
    v1, v2 = res["variants"]["v1"], res["variants"]["v2"]
    assert v1["test"]["answer_at_70_vs_gold60"]["fp_rate"] == 1.0 and v1["test"]["criterion"]["fp_rate"] == 1.0
    assert v2["test"]["answer_vs_gold60"]["fp_rate"] == 0.0 and v2["test"]["answer_vs_gold60"]["fn_rate"] == 0.0
    assert v2["test"]["answer_vs_accept"]["fn_rate"] == 0.0
    assert v2["test"]["criterion"]["kappa"] == pytest.approx(1.0)
    assert res["rubric_validity"]["tune"] == pytest.approx(1.0)

    for name, rows in (("labels.jsonl", labels), ("scores.jsonl", scores), ("items.jsonl", items)):
        write_jsonl(tmp_path / name, rows)
    assert main(["--labels", str(tmp_path / "labels.jsonl"), "--scores", str(tmp_path / "scores.jsonl"),
                 "--d-report", str(tmp_path / "items.jsonl"), "--out", str(tmp_path / "out.json")]) == 0
    assert set(json.loads((tmp_path / "out.json").read_text())["variants"]) == {"v1", "v1+gate", "v2", "v2+gate"}


def test_the_gate_is_the_main_result_criterion_and_caps_a_report_that_misses_it():
    """Rubric v2 (2026-11-15): partial credit for coverage must not lift a report whose main result is wrong."""
    item = {"id": "q", "answer": {"rubric": [{"criterion": "Explains the physics", "weight": 3},
                                             {"criterion": "Computes Q of about 110", "weight": 3},
                                             {"criterion": "Notes a limitation", "weight": 1}]}}
    assert gate_index(item) == 1                                        # the numeric criterion, not the first
    assert gate_index({"answer": {"rubric": [{"criterion": "a", "weight": 1}, {"criterion": "b", "weight": 2}]}}) == 1
    scores = {("sft", "q"): {"kind": "criteria", "score": 0.0, "per_criterion": [9, 2, 9]},
              ("rl", "q"): {"kind": "checks", "score": 0.0, "per_criterion": [1, 1, 0]}}
    items = {"q": item}
    plain = derived_scores(scores, items)
    assert plain[("sft", "q")]["score"] == pytest.approx((27 + 6 + 9) / 70)
    gated = derived_scores(scores, items, gate_cap=0.3)
    assert gated[("sft", "q")]["score"] == 0.3 and gated[("rl", "q")]["score"] == pytest.approx(6 / 7)
    clean = derived_scores(scores, items, drop={"q": {3}})
    assert clean[("sft", "q")]["score"] == pytest.approx(33 / 60) and clean[("sft", "q")]["kept_ids"] == [1, 2]
    labels = {("sft", "q"): {"criteria": [{"id": 1, "met": "yes"}, {"id": 2, "met": "no"}, {"id": 3, "met": "yes"}]}}
    assert criterion_pairs(labels, clean, 7, [("sft", "q")])[0] == [(True, True), (False, False)]


def test_three_level_verdicts_are_met_only_when_full_and_tabled_against_the_gold():
    """2026-11-16: a three-level judge (full 1, partial 0.5, none 0) counts a criterion as met only when full; the
    reward's partial credit is rescored from the same verdicts, and the levels are tabled against the gold marks."""
    item = {"id": "q", "answer": {"rubric": [{"criterion": "a", "weight": 2}, {"criterion": "b", "weight": 1},
                                             {"criterion": "c", "weight": 1}]}}
    labels = {("sft", "q"): {"criteria": [{"id": 1, "met": "yes"}, {"id": 2, "met": "partly"}, {"id": 3, "met": "no"}]}}
    scores = {("sft", "q"): {"kind": "levels", "score": 0.0, "per_criterion": [1.0, 0.5, 0.5]}}
    assert criterion_pairs(labels, scores, 7, [("sft", "q")], partly_as_no=True)[0] == [
        (True, True), (False, False), (False, False)]
    assert derived_scores(scores, {"q": item})[("sft", "q")]["score"] == pytest.approx(3.0 / 4)
    assert derived_scores(scores, {"q": item}, partial_credit=0.25)[("sft", "q")]["score"] == pytest.approx(2.5 / 4)
    assert derived_scores(scores, {"q": item}, partial_credit=0.0)[("sft", "q")]["score"] == pytest.approx(2.0 / 4)
    table = level_table(labels, scores, [("sft", "q")])
    assert table["yes"]["full"] == 1 and table["partly"]["partial"] == 1 and table["no"]["partial"] == 1


def test_same_question_order_counts_pairs_the_judge_orders_as_the_gold():
    """GRPO uses only the order of the answers to one question (2026-11-16): near ties in the gold are left out, and
    a judge tie counts half."""
    labels = {("base", "q"): {"score": 30}, ("sft", "q"): {"score": 70}, ("rl", "q"): {"score": 75},
              ("base", "r"): {"score": 20}, ("sft", "r"): {"score": 60}}
    scores = {("base", "q"): {"score": 0.5}, ("sft", "q"): {"score": 0.8}, ("rl", "q"): {"score": 0.6},
              ("base", "r"): {"score": 0.7}, ("sft", "r"): {"score": 0.7}}
    order = same_question_order(labels, scores, list(labels))
    # q: base < sft (agrees), base < rl (agrees), sft ~ rl (5 apart, left out); r: a judge tie (half)
    assert order == {"pairs": 3, "agreement": pytest.approx(2.5 / 3)}
    assert same_question_order(labels, {}, list(labels)) == {"pairs": 0, "agreement": None}


def test_the_format_comparison_counts_partly_met_criteria_judged_met():
    """Judged met among the gold's partly-met criteria = the false passes added by counting "partly" as not met."""
    criterion = {"tp": 194, "fn": 2, "fp": 10, "tn": 33, "set_apart": {"partly": 167}}
    answers = {"n": 93, "tp": 17, "fp": 52, "tn": 23, "fn": 1, "fp_rate": 0.69, "fn_rate": 0.06,
               "balanced_accuracy": 0.63}
    result = {"answers": 300, "questions": {"tune": 69, "test": 31}, "variants": {"v1": {
        "pass_mark_vs_gold60": 90, "same_question_order": {"pairs": 2, "agreement": 0.5},
        "test": {"criterion": criterion, "criterion_partly_as_no": {"fp": 149}, "answer_vs_gold60": answers}}}}
    out = format_comparison(result)
    assert list(out["formats"]) == ["0-10 per criterion"]
    v1 = out["formats"]["0-10 per criterion"]
    assert v1["criteria"] == {"yes": 196, "partly": 167, "no": 43}
    assert v1["judged_met"] == {"yes": 194, "partly": 139, "no": 10}
    assert v1["answers"]["fp"] == 52 and v1["pass_mark"] == 90 and v1["levels"] is None
