"""T1 numeric / T4 math graders, including the full adversarial suite (report §第二层)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dpt.rewards.numeric import NumericConfig, grade_math, grade_numeric, parse_quantity, safe_eval
from dpt.rewards.router import RewardRouter

CASES = Path(__file__).with_name("adversarial_cases.jsonl")
T20 = {"value": 20.0, "unit": "s", "rel_tol": 0.01, "abs_tol": 0.0}


def test_safe_eval_subset():
    assert safe_eval(r"\frac{1}{2}") == 0.5
    assert safe_eval(r"2\sqrt{2}") == pytest.approx(2 * 2**0.5)
    assert safe_eval(r"3 \times 10^{-2}") == pytest.approx(0.03)
    assert safe_eval("2pi") == pytest.approx(6.283185307)
    assert safe_eval(r"\sqrt[3]{27}") == pytest.approx(3)


@pytest.mark.parametrize("expr", ["__import__('os')", "1/0", "2^100000", "foo(3)", "1 +"])
def test_safe_eval_rejects_unsafe_or_bad(expr):
    with pytest.raises(ValueError):
        safe_eval(expr)


def test_parse_quantity_splits_value_and_unit():
    q = parse_quantity(r"t = \frac{120}{6} = 20\,\mathrm{s}")
    assert q.value == 20 and q.unit == "s"


def test_binary_reward_no_partial_credit():
    """Ablation A7: a close-but-wrong answer earns exactly 0, never a fraction."""
    r = grade_numeric(r"\boxed{20.5 s}", T20)
    assert r.score == 0.0 and r.reason == "wrong_value" and r.components["t1_rel_err"] > 0


def test_unit_conversion_and_dimension():
    assert grade_numeric(r"\boxed{0.02 ks}", T20).score == 1.0
    assert grade_numeric(r"\boxed{20 m}", T20).reason == "unit_dimension_mismatch"


def test_missing_unit_policy_is_configurable():
    assert grade_numeric(r"\boxed{20}", T20).reason == "missing_unit"
    assert grade_numeric(r"\boxed{20}", T20, NumericConfig(require_unit=False)).score == 1.0


def test_bad_ground_truth_fails_closed():
    assert grade_numeric(r"\boxed{20 s}", {"unit": "s"}).reason == "bad_ground_truth"


def test_math_candidates_must_agree():
    assert grade_math(r"\boxed{1/2} \boxed{0.5}", {"kind": "math", "value": "0.5"}).score == 1.0
    assert grade_math(r"\boxed{1/2} \boxed{0.6}", {"kind": "math", "value": "0.5"}).reason == "multi_answer"


def _cases():
    return [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_adversarial_suite_is_large_enough():
    cases = _cases()
    assert len(cases) >= 120
    assert {
        "equivalent_forms",
        "unit_conversion",
        "multi_answer",
        "interval",
        "sci_notation",
        "buried_answer",
    } <= {c["category"] for c in cases}


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["id"])
def test_adversarial_case(case):
    router = RewardRouter({})
    res = router.grade(
        case["task"], case["response"], case["ground_truth"], {"task": case["task"], "response_length": 0}
    )
    assert res.components["base_score"] == pytest.approx(case["expected"]), (case, res.reason)


def test_partial_credit_is_off_by_default_and_ablation_arm_grades_closeness():
    """Ablation A7: partial credit exists only as an ablation arm; production stays binary."""
    gt = {"value": 20.0, "unit": "s", "rel_tol": 0.01}
    assert grade_numeric(r"\boxed{21 s}", gt).score == 0.0  # default: binary
    a7 = NumericConfig(partial_credit={"enabled": True, "max_rel_err": 0.10})
    r = grade_numeric(r"\boxed{21 s}", gt, a7)  # rel_err = 0.05
    assert r.score == pytest.approx(0.5) and r.reason == "partial_credit" and r.correct is False
    assert grade_numeric(r"\boxed{20.1 s}", gt, a7).score == 1.0  # within tolerance → 1
    assert grade_numeric(r"\boxed{30 s}", gt, a7).score == 0.0  # beyond max_rel_err
    assert grade_numeric(r"\boxed{21000 ms}", gt, a7).score == pytest.approx(0.5)  # units still converted
    # structural failures stay 0 even in the A7 arm
    assert grade_numeric(r"\boxed{20 s} or \boxed{21 s}", gt, a7).score == 0.0
    assert grade_numeric(r"\boxed{19–21 s}", gt, a7).score == 0.0
    assert grade_numeric(r"\boxed{21 m}", gt, a7).score == 0.0
    assert grade_numeric(r"\boxed{21}", gt, a7).score == 0.0


def test_shipped_config_keeps_partial_credit_disabled():
    from dpt.rewards.router import load_rewards_config

    pc = load_rewards_config()["t1_num"]["partial_credit"]
    assert pc == {"enabled": False, "max_rel_err": 0.10}


@pytest.mark.parametrize("ans", ["7.97\\ \\text{unitless}", "7.97 \\text{(dimensionless)}", "7.97 dimensionless",
                                 "7.97\\ \\text{aftershocks}", "7.97\\ \\text{magnitude (dimensionless)}"])
def test_a_no_unit_note_on_a_dimensionless_answer_is_accepted(ans):
    import json

    from dpt.rewards.router import compute_score

    gold = json.dumps({"value": 7.9719, "unit": "", "rel_tol": 0.01, "abs_tol": 0.0})
    assert compute_score("t1_num", f"</think>\\boxed{{{ans}}}", gold, {"uid": "x", "task": "t1_num"})["score"] == 1.0
    wrong = json.dumps({"value": 7.97, "unit": "km", "rel_tol": 0.01, "abs_tol": 0.0})   # a dimensioned label
    assert compute_score("t1_num", f"</think>\\boxed{{{ans}}}", wrong, {"uid": "x", "task": "t1_num"})["score"] == 0.0


def test_a_count_word_does_not_hide_a_wrong_dimension():
    import json

    from dpt.rewards.router import compute_score

    gold = json.dumps({"value": 800.0, "unit": "", "rel_tol": 0.01, "abs_tol": 0.0})
    assert compute_score("t1_num", "</think>\\boxed{800 aftershocks/day}", gold, {"uid": "x", "task": "t1_num"})["score"] == 0.0
