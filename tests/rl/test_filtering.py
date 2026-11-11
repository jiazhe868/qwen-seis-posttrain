"""Tests for zero-advantage group filtering and truncation masking."""

from __future__ import annotations

import numpy as np
import pytest

from dpt.rl.filtering import filter_zero_advantage, truncation_loss_mask, zero_advantage_mask


def test_zero_advantage_mask_flags_whole_degenerate_groups():
    scores = [1, 1, 0, 0, 0.7, 0.7]
    m = zero_advantage_mask(scores, ["a", "a", "b", "b", "c", "c"])
    assert m.tolist() == [True, True, True, True, True, True]
    m = zero_advantage_mask([1, 0, 0, 0], ["a", "a", "b", "b"])
    assert m.tolist() == [False, False, True, True]


def test_filter_stats_per_task():
    scores = np.array([1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
    gids = np.repeat([0, 1, 2], 4)
    tasks = ["t1_num"] * 8 + ["t2_code"] * 4
    lens = [100] * 4 + [200] * 4 + [1000] * 4
    r = filter_zero_advantage(scores, gids, task_ids=tasks, response_lengths=lens)
    assert r.keep_idx.tolist() == [4, 5, 6, 7]
    assert r.stats["filter/t1_num/zero_adv_group_frac"] == 0.5
    assert r.stats["filter/t1_num/all_solved_group_frac"] == 0.5
    assert r.stats["filter/t2_code/zero_adv_group_frac"] == 1.0
    assert r.stats["filter/t2_code/all_solved_group_frac"] == 0.0
    assert r.stats["filter/zero_adv_token_frac"] == pytest.approx(4400 / 5200)


def test_keep_multiple_of_readmits_whole_cheapest_groups():
    scores = np.array([1, 0, 1, 1, 0, 0, 1, 1])        # group 0 useful; 1, 2, 3 degenerate
    gids = np.repeat([0, 1, 2, 3], 2)
    lens = [10, 10, 50, 50, 5, 5, 99, 99]
    r = filter_zero_advantage(scores, gids, response_lengths=lens, keep_multiple_of=4)
    assert len(r.keep_idx) % 4 == 0
    assert r.keep_idx.tolist() == [0, 1, 4, 5]          # cheapest degenerate group (2) re-admitted


def test_keep_multiple_impossible_raises():
    with pytest.raises(ValueError):
        filter_zero_advantage([1, 0, 1], ["a", "a", "b"], keep_multiple_of=4)


def test_truncation_mask():
    m = truncation_loss_mask(np.ones((3, 4)), [False, True, False])
    assert m.sum(1).tolist() == [4, 0, 4]
