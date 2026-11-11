"""Final-answer extraction and structural checks (step-95 multi-answer and step-140 interval hacks)."""

from __future__ import annotations

import pytest

from dpt.rewards.answer_extract import (
    extract_final_answer,
    find_boxed,
    has_multiple_values,
    is_interval,
    skeleton,
)


def test_find_boxed_handles_nested_braces():
    assert find_boxed(r"x \boxed{\frac{1}{\sqrt{2}}} y \boxed{3}") == [r"\frac{1}{\sqrt{2}}", "3"]
    assert find_boxed(r"\boxed{unterminated") == []


def test_extraction_priority_and_disagreeing_marker():
    ex = extract_final_answer("work\nAnswer: 25 s\n" + r"\boxed{20 s}")
    assert ex.method == "boxed+marker" and sorted(ex.distinct) == ["20 s", "25 s"]


def test_long_unmarked_answer_is_not_fished():
    assert extract_final_answer("The time is 20 s because " + "x" * 100).candidates == []
    assert extract_final_answer("20 s").candidates == ["20 s"]


@pytest.mark.parametrize(
    "c",
    [
        "3.1–3.4 km",
        "about 3.1-3.4 km",
        "between 3 and 4",
        "3 ± 0.1",
        r"3 \pm 0.1",
        "[3, 4]",
        "> 3 km",
        r"\geq 3",
        "at least 3 km",
        "3 km以上",
        "3 km or more",
        "3 to 4 s",
        "2.9~3.1",
        "从 2.9 到 3.1",
    ],
)
def test_intervals_detected(c):
    """Regression for the step-140 hack: ranges must never be graded as their first number."""
    assert is_interval(c)


@pytest.mark.parametrize(
    "c", ["6.0×10^-3 m", "5 s^-1", "6.0e-3 m", "-3.2 K", "t = 20 s", "~20 s", "1,000 m", "5 N"]
)
def test_single_values_not_intervals(c):
    assert not is_interval(c)
    assert not has_multiple_values(c)


def test_math_mode_allows_subtraction():
    assert is_interval("2 - 1")
    assert not is_interval(r"2\sqrt{2} - 1", math_mode=True)


@pytest.mark.parametrize("c", ["20 s or 25 s", "20 s, 25 s", "x = 3 or x = 4", "3; 4", "20 s 或 25 s"])
def test_multiple_values_detected(c):
    """Regression for the step-95 hack: several candidates in one answer earn nothing."""
    assert has_multiple_values(c)


def test_skeleton_never_confuses_newton_with_number():
    assert skeleton("5 N") == "# N"
    assert skeleton(r"\frac{\sqrt{3}}{2}") == "#"
