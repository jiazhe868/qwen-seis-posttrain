"""Per-step advantage pipeline used by the verl adapter."""

from __future__ import annotations

import numpy as np
import pytest

from dpt.rl.advantage import TaskAdvantageScaler, TaskLossNormalizer
from dpt.rl.pipeline import compute_step_advantages, panel_metrics

MIX = {"t1_num": 0.5, "t2_code": 0.25, "t3_report": 0.0, "t4_general": 0.25}


def _norm():
    return TaskLossNormalizer(response_tokens={"t1_num": 4000, "t2_code": 9000, "t3_report": 8000, "t4_general": 3500},
                              group_size=4, prompts_per_step=4, num_updates=1)


def test_truncated_excluded_from_baseline_and_loss():
    # one group of 4 T1 samples; the truncated one scored 0 but must not lower the baseline
    out = compute_step_advantages(
        [1, 0, 1, 0], ["g"] * 4, ["t1_num"] * 4, [100, 200, 100, 24576],
        max_response=24576, scaler=TaskAdvantageScaler(), normalizer=None, nominal_mix=MIX)
    assert out.truncated.tolist() == [False, False, False, True]
    assert out.seq_adv[3] == 0.0 and not out.valid[3]
    # baseline over the 3 valid samples = 2/3
    np.testing.assert_allclose(out.seq_adv[:3], [1 / 3, -2 / 3, 1 / 3])


def test_padding_rows_ignored_and_never_raise():
    out = compute_step_advantages(
        [1, 0, 0], ["g", "g", "pad1"], ["t1_num", "t1_num", ""], [10, 10, 128],
        max_response=100, scaler=TaskAdvantageScaler(), normalizer=_norm(), nominal_mix=MIX,
        is_padding=[False, False, True])
    assert out.seq_adv[2] == 0 and not out.valid[2]
    assert out.seq_adv[0] > 0 > out.seq_adv[1]


def test_task_multipliers_equalize_gradient_share_despite_length():
    # T2 answers are ~2× longer; with multipliers, grad share follows the nominal weights, not tokens
    scores = [1, 0] * 4
    gids = ["a", "a", "b", "b", "c", "c", "d", "d"]
    tasks = ["t1_num"] * 4 + ["t2_code"] * 4
    lens = [4000] * 4 + [9000] * 4
    mix = {"t1_num": 0.5, "t2_code": 0.5, "t3_report": 0.0, "t4_general": 0.0}
    norm = TaskLossNormalizer(response_tokens={"t1_num": 4000, "t2_code": 9000}, group_size=2,
                              prompts_per_step=4, num_updates=1)
    out = compute_step_advantages(scores, gids, tasks, lens, max_response=24576,
                                  scaler=TaskAdvantageScaler(), normalizer=norm, nominal_mix=mix)
    m = out.metrics
    assert m["task/t2_code/trained_token_share"] == pytest.approx(9 / 13)
    assert m["task/t1_num/grad_share"] == pytest.approx(0.5)
    assert m["task/t2_code/grad_share"] == pytest.approx(0.5)


def test_panel_metrics_use_grader_verdict_and_full_batch():
    # sample 2 is correct but long (score reduced by the overlong penalty): still counted as correct
    m = panel_metrics(scores=[1.0, 0.0, 0.7, 0.0], correct=[True, False, True, False],
                      task_ids=["t1_num"] * 4, group_ids=["g"] * 4, response_lengths=[100, 300, 22000, 24576],
                      truncated=[False, False, False, True])
    assert m["task/t1_num/acc"] == 0.5
    assert m["task/t1_num/len_correct"] == (100 + 22000) / 2
    assert m["task/t1_num/len_incorrect"] == 300          # truncated sample excluded from length split
    assert m["task/t1_num/truncation_rate"] == 0.25
    assert m["task/t1_num/prompt_share"] == 1.0


def test_std_normalization_only_for_ablation_a2():
    kw = dict(max_response=1000, scaler=TaskAdvantageScaler(), normalizer=None, nominal_mix=MIX)
    plain = compute_step_advantages([1, 1, 1, 0], ["g"] * 4, ["t1_num"] * 4, [10] * 4, **kw)
    std = compute_step_advantages([1, 1, 1, 0], ["g"] * 4, ["t1_num"] * 4, [10] * 4,
                                  normalize_by_group_std=True, **kw)
    # nearly-all-correct group: std normalization inflates the advantages (the difficulty bias A2 removes)
    assert abs(std.seq_adv[3]) > abs(plain.seq_adv[3]) * 2
