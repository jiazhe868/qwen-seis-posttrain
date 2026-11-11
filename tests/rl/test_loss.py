"""Tests for the reference PPO loss: homogeneity, clip-higher, dual-clip, and the two invariance
properties the training design relies on (report §单步时间预算 设计说明-2, §多任务 调整-1, ablation A10).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dpt.rl.advantage import TaskLossNormalizer, group_mean_advantage, largest_remainder_counts
from dpt.rl.filtering import filter_zero_advantage
from dpt.rl.loss import aggregate, grouped_task_loss, ppo_clip_token_loss

torch.manual_seed(0)


def _rand(B=6, T=9):
    logp = torch.randn(B, T) * 0.3 - 2.0
    old = logp + torch.randn(B, T) * 0.4
    adv = torch.randn(B)
    mask = (torch.rand(B, T) > 0.2).float()
    return logp, old, adv, mask


@pytest.mark.parametrize("c", [0.01, 0.5, 3.7, 1e4])
@pytest.mark.parametrize("dual", [None, 3.0])
def test_ppo_clip_is_positively_homogeneous_in_advantage(c, dual):
    logp, old, adv, _ = _rand()
    w = torch.rand_like(logp) * 2
    base, _ = ppo_clip_token_loss(logp, old, adv, dual_clip_c=dual, is_weights=w)
    scaled, _ = ppo_clip_token_loss(logp, old, c * adv, dual_clip_c=dual, is_weights=w)
    torch.testing.assert_close(scaled, c * base, rtol=1e-5, atol=1e-6)


def test_clip_higher_keeps_gradient_for_ratio_between_upper_bounds():
    """A3: a positive-advantage token at ratio 1.25 is clipped (no grad) with ε_high=0.2, not with 0.28."""
    old = torch.zeros(1, 1)
    for eps_high, expect_grad in [(0.2, False), (0.28, True)]:
        logp = torch.full((1, 1), float(np.log(1.25)), requires_grad=True)
        loss, _ = ppo_clip_token_loss(logp, old, torch.ones(1), clip_high=eps_high)
        loss.sum().backward()
        assert (logp.grad.abs().item() > 0) == expect_grad


def test_dual_clip_bounds_negative_advantage_loss():
    logp = torch.full((1, 1), 5.0)  # ratio e^5 ≈ 148
    loss, _ = ppo_clip_token_loss(logp, torch.zeros(1, 1), -torch.ones(1), dual_clip_c=3.0)
    assert loss.item() == pytest.approx(3.0)


def test_per_task_normalization_equals_advantage_prescaling():
    """L = Σ_k (w_k/B_k) Σ_{t∈k} ℓ_t  ==  aggregate(ℓ(A·m), fixed-constant C) with m_k = w_k·C/B_k."""
    mix = {"t1_num": 0.4, "t2_code": 0.25, "t3_report": 0.15, "t4_general": 0.2}
    norm = TaskLossNormalizer({"t1_num": 7, "t2_code": 20, "t3_report": 15, "t4_general": 6},
                              group_size=2, prompts_per_step=8, num_updates=1, constant=50.0)
    tasks = ["t1_num", "t1_num", "t2_code", "t2_code", "t3_report", "t4_general"]
    logp, old, adv, mask = _rand(B=len(tasks), T=12)
    logp.requires_grad_(True)

    direct_mat, _ = ppo_clip_token_loss(logp, old, adv)
    direct = grouped_task_loss(direct_mat, mask, tasks, norm.target_weights(mix), norm.budgets(mix))
    g_direct = torch.autograd.grad(direct, logp)[0]

    adv_scaled = torch.as_tensor(norm.apply(adv.numpy(), tasks, mix), dtype=torch.float32)
    via_mat, _ = ppo_clip_token_loss(logp, old, adv_scaled)
    via = aggregate(via_mat, mask, "fixed-constant", constant=norm.constant)
    g_via = torch.autograd.grad(via, logp)[0]

    torch.testing.assert_close(via, direct, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(g_via, g_direct, rtol=1e-5, atol=1e-8)


class TinyPolicy(torch.nn.Module):
    """A toy policy: per-position features → logits over a 7-token vocabulary."""

    def __init__(self, d=5, vocab=7):
        super().__init__()
        self.proj = torch.nn.Linear(d, vocab)

    def logp(self, feats, tokens):
        return torch.log_softmax(self.proj(feats), -1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)


def _grads(policy, feats, tokens, old, adv, mask, mode, constant=None):
    policy.zero_grad()
    mat, _ = ppo_clip_token_loss(policy.logp(feats, tokens), old, adv)
    aggregate(mat, mask, mode, constant=constant).backward()
    return [p.grad.clone() for p in policy.parameters()]


def test_zero_adv_filter_keeps_gradient_under_fixed_constant():
    """The report's pre-launch unit test: dropping zero-advantage groups must not change the gradient
    when the loss is normalized by a fixed constant, and it DOES change it under token-mean."""
    torch.manual_seed(1)
    G, n, T = 6, 4, 10
    B = G * n
    policy = TinyPolicy()
    feats = torch.randn(B, T, 5)
    tokens = torch.randint(0, 7, (B, T))
    mask = (torch.rand(B, T) > 0.3).float()
    with torch.no_grad():
        old = policy.logp(feats, tokens) + 0.05 * torch.randn(B, T)  # slightly off-policy
    scores = np.array([1, 0, 0, 1,  1, 1, 1, 1,  0, 0, 1, 0,  0, 0, 0, 0,  1, 0, 1, 1,  0.5, 0.5, 0.5, 0.5])
    gids = np.repeat(np.arange(G), n)
    adv = torch.as_tensor(group_mean_advantage(scores, gids), dtype=torch.float32)

    res = filter_zero_advantage(scores, gids)
    assert len(res.drop_idx) == 3 * n  # groups 1 (all 1), 3 (all 0), 5 (all 0.5)
    k = torch.as_tensor(res.keep_idx)

    C = 4096.0
    full = _grads(policy, feats, tokens, old, adv, mask, "fixed-constant", C)
    kept = _grads(policy, feats[k], tokens[k], old[k], adv[k], mask[k], "fixed-constant", C)
    for a, b in zip(full, kept):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-9)

    full_tm = _grads(policy, feats, tokens, old, adv, mask, "token-mean")
    kept_tm = _grads(policy, feats[k], tokens[k], old[k], adv[k], mask[k], "token-mean")
    assert not torch.allclose(full_tm[0], kept_tm[0], rtol=1e-3)


def test_task_multipliers_compose_with_filtering():
    """Budgets are fixed, so filtering and per-task multipliers commute (grads unchanged)."""
    torch.manual_seed(2)
    mix = {"t1_num": 0.5, "t2_code": 0.25, "t3_report": 0.0, "t4_general": 0.25}
    counts = largest_remainder_counts(mix, 4)
    assert counts == {"t1_num": 2, "t2_code": 1, "t3_report": 0, "t4_general": 1}
    norm = TaskLossNormalizer({"t1_num": 10, "t2_code": 30, "t3_report": 1, "t4_general": 8},
                              group_size=4, prompts_per_step=4, num_updates=1)
    tasks = ["t1_num"] * 8 + ["t2_code"] * 4 + ["t4_general"] * 4
    gids = np.repeat(np.arange(4), 4)
    scores = np.array([1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0], dtype=float)
    B, T = 16, 8
    policy = TinyPolicy()
    feats, tokens = torch.randn(B, T, 5), torch.randint(0, 7, (B, T))
    mask = torch.ones(B, T)
    with torch.no_grad():
        old = policy.logp(feats, tokens)
    adv = norm.apply(group_mean_advantage(scores, gids), tasks, mix)
    adv_t = torch.as_tensor(adv, dtype=torch.float32)
    k = torch.as_tensor(filter_zero_advantage(scores, gids).keep_idx)
    full = _grads(policy, feats, tokens, old, adv_t, mask, "token-sum")
    kept = _grads(policy, feats[k], tokens[k], old[k], adv_t[k], mask[k], "token-sum")
    for a, b in zip(full, kept):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-10)


def test_aggregate_modes():
    mat = torch.tensor([[1.0, 2.0, 3.0], [4.0, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    assert aggregate(mat, mask, "token-sum").item() == 7.0
    assert aggregate(mat, mask, "token-mean").item() == pytest.approx(7 / 3)
    assert aggregate(mat, mask, "fixed-constant", constant=14).item() == 0.5
    assert aggregate(mat, mask, "seq-mean-token-mean").item() == pytest.approx((1.5 + 4) / 2)
    with pytest.raises(ValueError):
        aggregate(mat, mask, "fixed-constant")
