"""Layer-3 diagnostics: reasoning-answer consistency, perturbation gap, pass@k floor."""

from __future__ import annotations

import pytest

from dpt.eval.consistency import (
    consistency_report,
    is_consistent,
    last_derived_number,
    pass_at_k_floor,
    perturbation_gap,
)


def test_last_derived_number_prefers_results():
    assert last_derived_number("v = 6 km/s, d = 120 km, so t = 120/6 = 20 s. Check with 6.") == 20.0
    assert last_derived_number("numbers 3 and 4") == 4.0
    assert last_derived_number("no numbers") is None
    assert last_derived_number("M0 = 3.2×10^{19} N m") == pytest.approx(3.2e19)


def test_consistency_allows_unit_scaling():
    assert is_consistent(20.0, 20.1)
    assert is_consistent(20.0, 20000.0)  # s vs ms
    assert not is_consistent(20.0, 25.0)
    assert not is_consistent(None, 20.0)


def test_correct_but_inconsistent_rate():
    samples = [
        {"thinking": "t = 20", "submitted": 20.0, "correct": True},
        {"thinking": "t = 17", "submitted": 20.0, "correct": True},  # guessed / copied
        {"thinking": "t = 25", "submitted": 25.0, "correct": False},
    ]
    r = consistency_report(samples)
    assert r.n_correct == 2 and r.correct_but_inconsistent_rate == pytest.approx(0.5)
    assert r.consistent_rate == pytest.approx(2 / 3)


def test_perturbation_gap_matches_report_example():
    # Final model: 71.4 original vs 68.9 perturbed (gap 2.5); base gap 2.1 → widening 0.4, not flagged.
    orig = [1.0] * 714 + [0.0] * 286
    pert = [1.0] * 689 + [0.0] * 311
    r = perturbation_gap(orig, pert, base_gap=2.1, max_widening=1.6)
    assert r.gap == pytest.approx(2.5) and r.gap_widening == pytest.approx(0.4) and not r.flagged
    r2 = perturbation_gap(orig, [1.0] * 600 + [0.0] * 400, base_gap=2.1, max_widening=1.6)
    assert r2.flagged


def test_pass_at_k_floor():
    assert pass_at_k_floor({"64": 91.0}, {64: 90.2})
    assert not pass_at_k_floor({64: 89.0}, {64: 90.2})
