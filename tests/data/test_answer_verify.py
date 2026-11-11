"""Answer verification: three-way agreement, suspect labels, perturbation consistency (report §第一层)."""

from __future__ import annotations

import math

import pytest

from dpt.data.answer_verify import (
    VerifyConfig,
    extract_value,
    family_majority,
    last_boxed,
    parse_quantity,
    perturbation_consistency,
    to_label_unit,
    verify_numeric,
    verify_problem,
)
from dpt.data.backends import FakeSolverFamily
from dpt.data.schema import Problem
from dpt.data.solvers import solve

CFG = VerifyConfig()
HD = {"name": "hypocentral_distance", "params": {"epicentral_km": 30, "depth_km": 40}}


def _prob(value, gen=HD, task="t1_num", unit="km"):
    meta = {"generator": gen} if gen else {}
    return Problem.from_dict({"task": task, "question": f"hypocentral distance? {value}",
                              "answer": {"value": value, "unit": unit, "rel_tol": 0.01}, "meta": meta})


def _resp(v, unit="km"):
    return f"thinking\n</think>\n\nThe answer is \\boxed{{{v} {unit}}}"


@pytest.mark.parametrize("s,expected", [
    ("20 s", (20.0, "s")),
    ("3.2\\times10^{-4} m", (3.2e-4, "m")),
    ("1,200 \\mathrm{km}", (1200.0, "km")),
    ("-4.5", (-4.5, "")),
    ("6.0 \\, km/s", (6.0, "km/s")),
])
def test_parse_quantity(s, expected):
    v, u = parse_quantity(s)
    assert math.isclose(v, expected[0]) and u == expected[1]


def test_boxed_extraction_uses_last_box_and_requires_end_of_thinking():
    assert last_boxed("\\boxed{1} then \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert extract_value("no end marker \\boxed{50 km}", "km") is None
    assert extract_value(_resp("50000", "m"), "km") == pytest.approx(50.0)
    assert extract_value(_resp("50", "s"), "km") is None  # wrong dimension


def test_unit_conversion_table():
    assert to_label_unit(1.0, "min", "s") == 60.0
    assert to_label_unit(3.0, "km/s", "m/s") == 3000.0
    assert to_label_unit(2.0, "MPa", "Pa") == 2e6
    assert to_label_unit(1.0, "furlong", "m") is None


def test_family_majority_counts_misses():
    v, share = family_majority([50.0] * 5 + [None] * 3, rel_tol=0.01, abs_tol=0)
    assert v == 50.0 and share == pytest.approx(5 / 8)


def test_all_three_agree_accepts():
    out = verify_numeric(_prob(50.0), {"a": [_resp(50)] * 8, "b": [_resp(50.2)] * 7 + [_resp(70)]}, CFG)
    assert out.status == "accepted" and out.votes["solver"] == pytest.approx(50.0)


def test_everyone_against_the_label_is_suspect_label_with_high_priority():
    out = verify_numeric(_prob(70.0), {"a": [_resp(50)] * 8, "b": [_resp(50)] * 8}, CFG)
    assert out.status == "rejected" and out.reason == "suspect_label" and out.priority == "high"


def test_undecided_family_rejects():
    noisy = [_resp(v) for v in (50, 51.5, 53, 55, 57, 59, 61, 63)]
    out = verify_numeric(_prob(50.0), {"a": [_resp(50)] * 8, "b": noisy}, CFG)
    assert out.status == "rejected" and out.reason == "family_undecided"


def test_solver_required_for_t1_but_not_t4():
    out = verify_numeric(_prob(50.0, gen=None), {"a": [_resp(50)] * 8, "b": [_resp(50)] * 8}, CFG)
    assert out.reason == "no_solver"
    t4 = Problem.from_dict({"task": "t4_general", "question": "2+3?", "answer": {"kind": "math", "value": 5}})
    out4 = verify_numeric(t4, {"a": [_resp(5, "")] * 8, "b": [_resp(5, "")] * 8}, CFG)
    assert out4.status == "accepted"


def test_non_numeric_tasks_not_applicable():
    p = Problem.from_dict({"task": "t4_general", "question": "No commas.",
                           "answer": {"kind": "ifeval", "constraints": [{"type": "no_commas"}]}})
    assert verify_problem(p, {}, CFG).status == "not_applicable"


def test_verify_problem_with_fake_families():
    truth = lambda p: solve(p.meta["generator"])  # noqa: E731
    fams = {"a": FakeSolverFamily("a", truth, 0.1), "b": FakeSolverFamily("b", truth, 0.1)}
    assert verify_problem(_prob(50.0), fams, CFG).status == "accepted"
    assert verify_problem(_prob(80.0), fams, CFG).reason == "suspect_label"


def test_perturbation_scaling_and_smoothness_pass_for_physical_solvers():
    tt = {"name": "travel_time", "params": {"distance_km": 120, "velocity_km_s": 6},
          "scaling": {"distance_km": 1, "velocity_km_s": -1}}
    assert perturbation_consistency(tt).ok
    assert perturbation_consistency(HD).ok  # no declared scaling → smoothness + sensitivity checks


def test_perturbation_catches_buggy_solvers():
    gen = {"name": "fake", "params": {"x": 2.0, "y": 3.0}, "scaling": {"x": 2}}
    # answer ignores y entirely, and scales linearly in x although x^2 is declared
    res = perturbation_consistency(gen, solver=lambda g: g["params"]["x"] * 10)
    assert not res.ok and "scaling_violation:x" in res.issues and "insensitive:y" in res.issues

    def erratic(g):
        x = g["params"]["x"]
        return x if x <= 2.0 else 5 * x  # kink exactly at the operating point
    assert "erratic:x" in perturbation_consistency({"name": "e", "params": {"x": 2.0}}, solver=erratic).issues

    inv = {"name": "i", "params": {"x": 2.0, "station": 7}, "invariant": ["station"]}
    res = perturbation_consistency(inv, solver=lambda g: g["params"]["x"] + g["params"]["station"])
    assert "depends_on_invariant:station" in res.issues


def test_label_that_fails_perturbation_is_rejected():
    bad = {"name": "travel_time", "params": {"distance_km": 120, "velocity_km_s": 6},
           "scaling": {"distance_km": 2, "velocity_km_s": -1}}  # wrong declared physics
    p = _prob(20.0, gen=bad, unit="s")
    fams = {"a": lambda p, n: [_resp(20, "s")] * n, "b": lambda p, n: [_resp(20, "s")] * n}
    out = verify_problem(p, fams, CFG)
    assert out.status == "rejected" and out.reason.startswith("perturbation:")


def test_audit_verifies_a_sample_per_template_and_decides_for_the_whole_template():
    from dpt.data.answer_verify import AuditConfig, VerifyConfig, audit_verify
    from dpt.data.schema import Problem

    def prob(template, d):
        return Problem.from_dict({
            "task": "t1_num", "question": f"Template {template}: a wave travels {d} km at 5 km/s. Time in s?",
            "answer": {"value": d / 5, "unit": "s"},
            "meta": {"family": "travel_time", "template": template,
                     "generator": {"name": "travel_time", "params": {"distance_km": d, "velocity_km_s": 5.0}}}})

    probs = [prob(t, d) for t in (0, 1, 2) for d in range(10, 60)]
    probs.append(Problem.from_dict({"task": "t3_report", "question": "Explain coda waves.",
                                    "answer": {"rubric": [{"criterion": "c", "weight": 1}], "reference_length": 500}}))
    calls = []

    def family(offset_for_template):
        def fn(p, n):
            calls.append(p.uid)
            t = p.meta["template"]
            v = p.answer["value"] * offset_for_template[t]
            return [f"</think>\\boxed{{{v:.6g} s}}"] * n
        return fn

    fams = {"a": family({0: 1.0, 1: 2.0, 2: 1.0}), "b": family({0: 1.0, 1: 2.0, 2: 3.0})}
    res = audit_verify(probs, fams, VerifyConfig(family_samples=2), AuditConfig(per_group=10), max_workers=1)
    g = res.groups
    assert g["t1_num:travel_time:t0"]["kept"] and g["t1_num:travel_time:t0"]["accepted"] == 10
    assert not g["t1_num:travel_time:t1"]["kept"] and "contested" in g["t1_num:travel_time:t1"]["why"]
    assert not g["t1_num:travel_time:t2"]["kept"] and "accept 0/10" in g["t1_num:travel_time:t2"]["why"]
    assert len(calls) == 2 * 30                                   # 10 audited per template × 2 families
    kept_templates = {p.meta.get("template") for p in res.kept if p.task == "t1_num"}
    assert kept_templates == {0} and sum(p.task == "t1_num" for p in res.kept) == 50
    assert any(p.task == "t3_report" for p in res.kept)          # not numeric: passes through
    assert {r["reason"] for r in res.removed} == {"verify:template_rejected"}
    assert all(p.meta["verified"]["audit"] == "t1_num:travel_time:t0" for p in res.kept if p.task == "t1_num")


def test_family_solvers_are_found_without_importing_the_generators():
    """Bank verification never imports dpt.data.gen; the T1 family solvers must still be found (2026-10-25:
    every T1 template was rejected with solver_error:KeyError)."""
    import os
    import subprocess
    import sys
    from pathlib import Path
    src = str(Path(__file__).resolve().parents[2] / "src")
    code = ("import sys; from dpt.data.solvers import solve; assert 'dpt.data.gen.families' not in sys.modules; "
            "print(solve({'name': 'direct_travel_time', 'params': {'distance_km': 202.0, 'velocity_km_s': 5.0}}))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env={**os.environ, "PYTHONPATH": src})
    assert float(out.stdout) == pytest.approx(40.4)


def test_a_family_that_prints_its_thinking_unmarked_still_votes():
    """Gemma, served without a reasoning parser, writes "thought\\n..." and no </think> (2026-10-25: every
    judge vote was None, so every problem came out family_undecided)."""
    from dpt.data.answer_verify import extract_value, unmarked_reply
    gemma = "thought\n*   t = d / v = 120 / 6\n    *   so maybe \\boxed{2 s}? no: 20 s.\n\nThe trip takes 20 s.\n\n\\boxed{20 s}"
    assert extract_value(gemma, "s") is None                       # the Qwen rule: no </think> = cut off
    assert extract_value(unmarked_reply(gemma), "s") == pytest.approx(20.0)   # the last \boxed
    qwen = "t = 120/6\n</think>\n\n\\boxed{20 s}"
    assert unmarked_reply(qwen) == qwen
