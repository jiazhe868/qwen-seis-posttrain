"""Training and evaluation graders are independent implementations, but they must agree on
the data contract. A disagreement on answers that follow the prompt instructions means one
side mis-grades (report §评测体系 第二层)."""

from __future__ import annotations

import pytest

from dpt.common.ifeval_schema import normalize_constraint
from dpt.eval.graders.ifeval import grade_ifeval
from dpt.eval.graders.numeric import grade_numeric
from dpt.rewards.instruction import InstructionConfig, grade_instruction
from dpt.rewards.numeric import grade_numeric as train_numeric

NO_RELEVANCE = InstructionConfig(relevance_check=False)

IFEVAL_CASES = [
    # (constraints, text, expected)
    ([{"type": "no_commas"}], "No commas here at all.", True),
    ([{"type": "no_comma"}], "One, two.", False),
    ([{"type": "bullets", "relation": "exactly", "value": 3}], "- a\n- b\n- c", True),
    ([{"type": "bullet_count", "n": 3}], "- a\n- b", False),
    ([{"type": "num_bullets", "value": 2}], "* a\n* b", True),
    ([{"type": "max_words", "value": 5}], "one two three four", True),
    ([{"type": "max_words", "n": 3}], "one two three four", False),
    ([{"type": "forbid_keywords", "value": ["magnitude"]}], "The quake was strong.", True),
    ([{"type": "exclude_keywords", "keywords": ["magnitude"]}], "Its magnitude was 6.", False),
    ([{"type": "include_keywords", "value": ["epicenter"]}], "The epicenter was offshore.", True),
    ([{"type": "lowercase"}], "all lower case text", True),
    ([{"type": "all_uppercase"}], "Not Upper", False),
]


@pytest.mark.parametrize("constraints,text,expected", IFEVAL_CASES)
def test_ifeval_graders_agree(constraints, text, expected):
    train = grade_instruction(text, {"constraints": constraints}, cfg=NO_RELEVANCE).score >= 1.0
    ev = grade_ifeval(text, {"constraints": constraints}).correct
    assert train == ev == expected, (constraints, text, train, ev)


def test_alias_normalization():
    assert normalize_constraint({"type": "bullet_count", "n": 3}) == {"type": "bullets", "value": 3,
                                                                      "relation": "exactly"}
    assert normalize_constraint({"type": "min_sentences", "value": 2})["relation"] == "at_least"


NUMERIC_CASES = [
    ("\\boxed{20 s}", {"value": 20.0, "unit": "s", "rel_tol": 0.01, "abs_tol": 0.0}, True),
    ("\\boxed{0.02 km}", {"value": 20.0, "unit": "m", "rel_tol": 0.01, "abs_tol": 0.0}, True),
    ("\\boxed{1.2 \\times 10^{18} N·m}", {"value": 1.2e18, "unit": "N·m", "rel_tol": 0.01, "abs_tol": 0.0}, True),
    ("\\boxed{3.1–3.4 km}", {"value": 3.2, "unit": "km", "rel_tol": 0.01, "abs_tol": 0.0}, False),
    ("\\boxed{20 s} or \\boxed{21 s}", {"value": 20.0, "unit": "s", "rel_tol": 0.01, "abs_tol": 0.0}, False),
    ("\\boxed{25 s}", {"value": 20.0, "unit": "s", "rel_tol": 0.01, "abs_tol": 0.0}, False),
    ("Answering this requires t=d/v.\n\\boxed{20 s}",
     {"value": 20.0, "unit": "s", "rel_tol": 0.01, "abs_tol": 0.0}, True),
]


@pytest.mark.parametrize("answer,gold,expected", NUMERIC_CASES)
def test_numeric_graders_agree(answer, gold, expected):
    train = train_numeric(answer, gold).score >= 1.0
    ev = grade_numeric(answer, gold, missing_unit="strict").correct
    assert train == ev == expected, (answer, train, ev)


MATH_CASES = [
    ("\\boxed{\\frac{\\pi}{2}}", 1.5707963267948966, True),
    ("\\boxed{\\sqrt{2}+1}", 2.414213562373095, True),
    ("\\boxed{2^{10}}", 1024.0, True),
    ("\\boxed{\\frac{1}{2}}", 0.5, True),
    ("\\boxed{0.5}", 0.5, True),
    ("\\boxed{1.5708}", 1.5707963267948966, False),     # T4 math is exact: 1e-6 relative tolerance
]


@pytest.mark.parametrize("answer,value,expected", MATH_CASES)
def test_math_graders_agree(answer, value, expected):
    from dpt.rewards.numeric import grade_math

    gold = {"kind": "math", "value": value, "unit": "", "rel_tol": 1e-6, "abs_tol": 0.0}
    train = grade_math(answer, gold).score >= 1.0
    ev = grade_numeric(answer, gold, missing_unit="strict").correct
    assert train == ev == expected, (answer, train, ev)


UNIT_SPELLINGS = [
    ("\\boxed{20 seconds}", {"value": 20.0, "unit": "s"}, True),
    ("\\boxed{3 kilometres}", {"value": 3.0, "unit": "km"}, True),
    ("\\boxed{30 degrees}", {"value": 30.0, "unit": "deg"}, True),
    ("\\boxed{6.5 Mw}", {"value": 6.5, "unit": ""}, True),
    ("\\boxed{6.5 km}", {"value": 6.5, "unit": ""}, False),            # dimensioned answer, dimensionless gold
    ("\\boxed{20}", {"value": 20.0, "unit": "s"}, False),              # missing unit: strict in train and gold
]


@pytest.mark.parametrize("answer,gold,expected", UNIT_SPELLINGS)
def test_unit_spellings_agree(answer, gold, expected):
    gold = {**gold, "rel_tol": 0.01, "abs_tol": 0.0}
    train = train_numeric(answer, gold).score >= 1.0
    ev = grade_numeric(answer, gold, missing_unit="strict").correct
    assert train == ev == expected, (answer, train, ev)
