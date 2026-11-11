"""Metrics and statistics: pass@k correctness, CI coverage, paired tests, noise floor."""

from __future__ import annotations

import itertools
from math import comb

import numpy as np
import pytest

from dpt.eval import metrics as M
from dpt.eval import stats as S


def _bruteforce_pass_at_k(n, c, k):
    samples = [1] * c + [0] * (n - c)
    combos = list(itertools.combinations(range(n), k))
    return sum(any(samples[i] for i in cmb) for cmb in combos) / len(combos)


@pytest.mark.parametrize("n,c,k", [(8, 0, 1), (8, 3, 1), (8, 3, 4), (8, 8, 8), (10, 2, 5), (6, 5, 2)])
def test_pass_at_k_unbiased_matches_bruteforce(n, c, k):
    assert M.pass_at_k_unbiased(n, c, k) == pytest.approx(_bruteforce_pass_at_k(n, c, k))


def test_pass_at_1_equals_mean_accuracy():
    rng = np.random.default_rng(0)
    corr = rng.random((50, 8)) < 0.6
    assert M.pass_at_k(corr, 1) == pytest.approx(100 * corr.mean())
    curve = M.pass_at_k_curve(corr)
    assert list(curve) == [1, 2, 4, 8] and all(np.diff(list(curve.values())) >= 0)


def test_pass_at_k_requires_enough_samples():
    with pytest.raises(ValueError):
        M.pass_at_k_unbiased(4, 1, 8)


def test_avg_at_k_weights_items_equally():
    s = np.array([[1, 1, 1, 1], [0, 0, 0, 0]], float)
    assert M.avg_at_k(s) == 50.0


def test_win_rate_ties_count_half():
    assert M.win_rate(["win", "tie", "loss", "win"]) == pytest.approx(62.5)
    with pytest.raises(ValueError):
        M.win_rate(["draw"])


def test_truncation_and_tokens():
    assert M.truncation_rate(["stop", "length", "stop", "stop"]) == 0.25
    assert M.mean_output_tokens([100, 300]) == 200


def test_bootstrap_ci_covers_true_mean():
    rng = np.random.default_rng(1)
    hits = 0
    for t in range(40):
        x = rng.random(300) < 0.7
        ci = S.bootstrap_ci(x.astype(float), n_boot=1000, seed=t)
        hits += ci.lo <= 70.0 <= ci.hi
    assert hits >= 34  # ~95% nominal coverage, loose bound for 40 trials


def test_sampling_ci_is_narrower_than_item_ci_on_heterogeneous_items():
    rng = np.random.default_rng(0)
    p = rng.beta(0.6, 0.6, size=600)  # many items near always-right / always-wrong
    s = (rng.random((600, 8)) < p[:, None]).astype(float)
    assert S.sampling_ci(s).halfwidth < S.bootstrap_ci(s.mean(1)).halfwidth
    with pytest.raises(ValueError):
        S.sampling_ci(s[:, :1])


def test_paired_bootstrap_detects_real_gain_and_not_noise():
    rng = np.random.default_rng(2)
    a = (rng.random(600) < 0.55).astype(float)
    b = np.where(rng.random(600) < 0.15, 1.0, a)  # b fixes ~15% of items → real gain
    d = S.paired_bootstrap_diff(a, b, n_boot=2000)
    assert d.significant and d.diff > 0
    same = S.paired_bootstrap_diff(a, a.copy(), n_boot=500)
    assert not same.significant


def test_noise_floor_and_significance():
    nf = S.noise_floor([71.0, 71.9, 70.6, 72.1, 71.4])
    assert nf.max_pairwise_abs_diff == pytest.approx(1.5)
    assert not S.is_significant_improvement(72.0, 71.0, noise=nf)
    assert S.is_significant_improvement(71.4, 55.3, ci_halfwidth=1.6, required_delta=8.0)
    assert not S.is_significant_improvement(62.0, 55.3, ci_halfwidth=1.6, required_delta=8.0)


def test_wilson_and_combined_halfwidth():
    ci = S.wilson_ci(136, 200)
    assert ci.point == pytest.approx(68.0) and ci.lo < 68 < ci.hi
    assert S.combined_halfwidth(3.0, 4.0) == pytest.approx(5.0)


def test_comb_sanity():
    assert comb(8, 0) == 1
