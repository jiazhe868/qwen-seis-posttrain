"""Adversarial cases for the eval-side numeric grader (report §第二层-1: grader unit-test set).

The report's grader test set covers equivalent forms (1/2 vs 0.5), units
(km vs m), multiple candidate answers, interval answers, scientific notation
and answers buried in long text. Each category has cases here, and every
fail-closed reason is checked.
"""

from __future__ import annotations

import pytest

from dpt.eval.graders.numeric import grade_numeric, parse_quantity
from dpt.eval.graders.units import parse_unit, to_si

T20S = {"value": 20.0, "unit": "s", "rel_tol": 0.01, "abs_tol": 0.0}
V6 = {"value": 6.0, "unit": "km/s", "rel_tol": 0.01, "abs_tol": 0.0}
HALF = {"value": 0.5, "unit": "", "rel_tol": 1e-6, "abs_tol": 0.0}


@pytest.mark.parametrize(
    "ans",
    [
        r"\boxed{20 s}",
        "Final answer: 20 s",
        "20",
        "20.0 s",
        "0.3333 min",
        "20000 ms",
        "t = 20 s",
        r"\boxed{\frac{40}{2}\,\mathrm{s}}",
        r"2.0\times10^{1} s",
        "2e1 s",
        r"$20\text{ s}$",
        "≈ 20 s",
    ],
)
def test_equivalent_forms_accepted(ans):
    r = grade_numeric(ans, T20S)
    assert r.correct, (ans, r)


@pytest.mark.parametrize("ans", ["1/2", r"\frac{1}{2}", "0.5", "5e-1", r"\boxed{0.50}", "50%"])
def test_fraction_decimal_percent(ans):
    assert grade_numeric(ans, HALF).correct


@pytest.mark.parametrize(
    "ans",
    [
        "6 km/s",
        "6000 m/s",
        "6.0 km s^-1",
        "6 km·s⁻¹",
        "6000 m s⁻¹",
        r"\boxed{6\,\mathrm{km}\,\mathrm{s}^{-1}}",
        "6 km per s",
        "600000 cm/s",
    ],
)
def test_compound_units(ans):
    assert grade_numeric(ans, V6).correct, ans


@pytest.mark.parametrize(
    "ans,reason",
    [
        (r"\boxed{20} and \boxed{21}", "multiple_answers"),
        ("Answer: 19\nAnswer: 20", "multiple_answers"),
        ("19 or 20 s", "multiple_answers"),
        ("about 19.9–20.1 s", "interval"),  # step-140 hack
        ("19.9-20.1 s", "interval"),
        ("between 19 and 21 s", "interval"),
        ("[19, 21] s", "interval"),
        ("20 ± 1 s", "interval"),
        ("> 19 s", "interval"),
        ("20 km", "unit_mismatch"),
        ("", "no_answer"),
        (
            "The travel time is twenty seconds, which I computed carefully over several lines\nof reasoning.",
            "no_answer",
        ),
    ],
)
def test_fail_closed_reasons(ans, reason):
    r = grade_numeric(ans, T20S)
    assert not r.correct and r.score == 0.0
    assert r.reason == reason, (ans, r)


def test_answer_buried_in_long_text_uses_marker():
    text = "We first compute 12 s for the first leg, then 8 s.\nThe sum is relevant.\nFinal answer: 20 s"
    assert grade_numeric(text, T20S).correct


def test_wrong_value_and_tolerance():
    assert not grade_numeric("20.5 s", T20S).correct  # 2.5% off, tol 1%
    assert grade_numeric("20.1 s", T20S).correct
    assert grade_numeric("20.4", {"value": 20, "unit": "s", "rel_tol": 0, "abs_tol": 0.5}).correct


def test_missing_unit_policy():
    assert grade_numeric("20", T20S).correct
    r = grade_numeric("20", T20S, missing_unit="strict")
    assert not r.correct and r.reason == "missing_unit"


def test_magnitude_labels_and_negative_values():
    mag = {"value": 6.5, "unit": "", "rel_tol": 0, "abs_tol": 0.05}
    for a in ["Mw 6.5", "M 6.5", "6.5"]:
        assert grade_numeric(a, mag).correct, a
    neg = {"value": -4.2, "unit": "", "rel_tol": 0.01, "abs_tol": 0}
    assert grade_numeric("x = -4.2", neg).correct and grade_numeric("−4.2", neg).correct
    assert not grade_numeric("4.2", neg).correct


def test_offset_temperatures():
    g = {"value": 25.0, "unit": "°C", "rel_tol": 0, "abs_tol": 0.5}
    for a in ["298.15 K", "25 °C", "77 °F"]:
        assert grade_numeric(a, g).correct, a


def test_unit_table():
    assert to_si(1.0, "km")[0] == pytest.approx(1000.0)
    assert to_si(1.0, "g/cm^3")[0] == pytest.approx(1000.0)  # kg/m^3
    assert to_si(1.0, "mGal")[0] == pytest.approx(1e-5)  # m/s^2
    assert to_si(1.0, "Ma")[0] == pytest.approx(3.15576e13)
    assert to_si(1.0, "dyn·cm")[0] == pytest.approx(1e-7)  # N·m
    assert parse_unit("kg/(m s^2)").dim == parse_unit("Pa").dim
    assert to_si(1.0, "hPa")[0] == pytest.approx(100.0)


def test_parse_quantity_splits_value_and_unit():
    q = parse_quantity(r"3.2 \times 10^{4} km/s")
    assert q.value == pytest.approx(32000.0) and q.unit == "km/s"


@pytest.mark.parametrize("ans", ["7.97\\ \\text{unitless}", "7.97 \\text{(dimensionless)}",
                                 "1.8\\times10^{-4}\\ \\text{(dimensionless)}", "7.97\\ \\text{aftershocks}",
                                 "7.97\\ \\text{magnitude (dimensionless)}"])
def test_a_no_unit_note_on_a_dimensionless_answer_is_accepted(ans):
    from dpt.eval.graders.numeric import grade_numeric

    v = 7.9719 if ans.startswith("7.97") else 1.8e-4
    assert grade_numeric(f"\\boxed{{{ans}}}", {"value": v, "unit": "", "rel_tol": 0.01}, missing_unit="strict").correct
    assert not grade_numeric(f"\\boxed{{{ans}}}", {"value": v, "unit": "km", "rel_tol": 0.01},
                             missing_unit="strict").correct


def test_a_count_word_does_not_hide_a_wrong_dimension():
    from dpt.eval.graders.numeric import grade_numeric

    assert not grade_numeric("\\boxed{800 aftershocks/day}", {"value": 800.0, "unit": "", "rel_tol": 0.01},
                             missing_unit="strict").correct


@pytest.mark.parametrize("ans", ["8.07\\ \\mathrm{M_s}", "8.07 M_s", "8.07 M_{w}", "8.07 Ms"])
def test_subscripted_magnitude_labels_are_dimensionless(ans):
    from dpt.eval.graders.numeric import grade_numeric

    assert grade_numeric(f"\\boxed{{{ans}}}", {"value": 8.07, "unit": "", "abs_tol": 0.05, "rel_tol": 0.0},
                         missing_unit="strict").correct
