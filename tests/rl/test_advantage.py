"""Tests for group-mean advantages, per-task scaling and per-task loss-share multipliers."""

from __future__ import annotations

import numpy as np
import pytest

from dpt.rl.advantage import (
    TaskAdvantageScaler,
    TaskLossNormalizer,
    group_mean_advantage,
    largest_remainder_counts,
    token_shares,
)

STAGE_A = {"t1_num": 0.50, "t2_code": 0.25, "t3_report": 0.0, "t4_general": 0.25}
STAGE_B = {"t1_num": 0.40, "t2_code": 0.25, "t3_report": 0.15, "t4_general": 0.20}


def test_group_mean_without_std_division():
    scores = [1, 0, 0, 0, 1, 1, 1, 0]
    gids = ["a"] * 4 + ["b"] * 4
    adv = group_mean_advantage(scores, gids)
    np.testing.assert_allclose(adv, [0.75, -0.25, -0.25, -0.25, 0.25, 0.25, 0.25, -0.75])
    # A2: no std division, so a 1/4 group and a 3/4 group get mirrored, equally scaled advantages.
    assert adv[0] == pytest.approx(-adv[7])


def test_invalid_samples_excluded_from_baseline_and_get_zero():
    # sample 3 truncated: excluded from the mean, gets advantage 0
    adv = group_mean_advantage([1, 0, 0, 0], ["g"] * 4, valid=[True, True, True, False])
    np.testing.assert_allclose(adv, [2 / 3, -1 / 3, -1 / 3, 0.0])


def test_singleton_or_all_invalid_group_has_zero_advantage():
    adv = group_mean_advantage([1, 0.3, 0.5], ["a", "b", "b"], valid=[True, True, False])
    np.testing.assert_allclose(adv, [0, 0, 0])


def test_scaler_brings_low_variance_task_to_reference_scale():
    rng = np.random.default_rng(0)
    s = TaskAdvantageScaler(["t3_report"], momentum=0.0, max_gain=20.0)
    t1 = rng.choice([-0.5, 0.5], size=400)          # binary-like advantages, std 0.5
    t3 = rng.normal(0, 0.05, size=400)               # narrow rubric scores, std 0.05
    adv = np.concatenate([t1, t3])
    tasks = ["t1_num"] * 400 + ["t3_report"] * 400
    out = s.update_and_scale(adv, tasks)
    assert np.std(out[400:]) == pytest.approx(np.std(out[:400]), rel=0.15)
    np.testing.assert_allclose(out[:400], t1)        # unscaled task untouched


def test_scaler_gain_is_capped_and_state_round_trips():
    s = TaskAdvantageScaler(["t3_report"], max_gain=2.0, momentum=0.5)
    s.update(np.array([0.5, -0.5, 0.001, -0.001]), ["t1_num", "t1_num", "t3_report", "t3_report"])
    assert s.gains()["t3_report"] == 2.0
    s2 = TaskAdvantageScaler(["t3_report"], max_gain=2.0, momentum=0.5)
    s2.load_state_dict(s.state_dict())
    assert s2.gains() == s.gains()
    with pytest.raises(ValueError):
        TaskAdvantageScaler(["t1_num"]).load_state_dict(s.state_dict())


def test_largest_remainder_is_exact_and_deterministic():
    expected = {"t1_num": 64, "t2_code": 32, "t3_report": 0, "t4_general": 32}
    assert largest_remainder_counts(STAGE_A, 128) == expected
    c = largest_remainder_counts(STAGE_B, 128)
    assert sum(c.values()) == 128 and c == {"t1_num": 51, "t2_code": 32, "t3_report": 19, "t4_general": 26}


def test_token_mean_lets_long_tasks_dominate_the_gradient():
    """§多任务 诊断-2: T1 with 50% of prompts gets far less than 50% of the gradient under token loss."""
    tasks = ["t1_num"] * 50 + ["t2_code"] * 25 + ["t4_general"] * 25
    toks = [2500] * 50 + [12000] * 25 + [3000] * 25
    assert token_shares(tasks, toks)["t1_num"] < 0.30


def test_multipliers_restore_target_gradient_shares():
    norm = TaskLossNormalizer({"t1_num": 2500, "t2_code": 12000, "t3_report": 8000, "t4_general": 3000},
                              group_size=8, prompts_per_step=128, num_updates=2)
    counts = largest_remainder_counts(STAGE_A, 128)
    tasks, toks = [], []
    for t, c in counts.items():
        for _ in range(c * 8 // 2):  # one mini-batch at exactly the budgeted lengths
            tasks.append(t)
            toks.append(norm.response_tokens[t])
    shares = norm.gradient_shares(tasks, toks, STAGE_A)
    for t, w in STAGE_A.items():
        assert shares[t] == pytest.approx(w, abs=1e-9)


def test_multipliers_zero_for_absent_task_and_apply_rejects_it():
    norm = TaskLossNormalizer({"t1_num": 1, "t2_code": 1, "t3_report": 1, "t4_general": 1})
    assert norm.multipliers(STAGE_A)["t3_report"] == 0.0
    with pytest.raises(ValueError):
        norm.apply(np.ones(2), ["t1_num", "t3_report"], STAGE_A)


def test_budgets_do_not_depend_on_the_batch():
    norm = TaskLossNormalizer({"t1_num": 4000, "t2_code": 9000, "t3_report": 8000, "t4_general": 3500})
    b = norm.budgets(STAGE_B)
    assert b["t1_num"] == 51 * 8 * 4000 / 2
    # constant C scales multipliers linearly
    m1 = norm.multipliers(STAGE_B)
    norm2 = TaskLossNormalizer(norm.response_tokens, constant=1024.0)
    m2 = norm2.multipliers(STAGE_B)
    for t in m1:
        assert m2[t] == pytest.approx(1024.0 * m1[t])
