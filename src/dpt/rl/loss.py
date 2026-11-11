"""Reference policy-loss implementation (PPO-clip with clip-higher, dual-clip, TIS, aggregation).

This is the *specification* of the loss we train with. The production loss
runs inside verl's Megatron workers (``policy_loss.loss_mode=vanilla``). This
module exists so that

* every property we rely on is a unit test on CPU: positive homogeneity in the
  advantage, invariance of the gradient to dropping zero-advantage groups
  under fixed-constant normalization, and equivalence of per-task
  normalization and advantage pre-scaling;
* the adapter can be checked against it. Feeding the same tensors to verl's
  ``compute_policy_loss_vanilla`` and to :func:`ppo_clip_token_loss` must give
  the same token losses.

Token loss (for one token with ratio ``ρ = exp(logp - old_logp)``, advantage ``A``):

    ℓ = max(-A·ρ, -A·clip(ρ, 1-ε_low, 1+ε_high))          (clipped surrogate)
    ℓ = min(ℓ, -A·c)            if A < 0 and dual-clip    (dual-clip, c = 3)
    ℓ = w_tis · ℓ               (truncated IS weight, a constant w.r.t. θ)

Clip-higher (ε_low = 0.2, ε_high = 0.28; ablation A3): with a symmetric clip, a
token with ρ already > 1.2 and A > 0 gets no gradient, so a token at
probability 0.01 can rise to only 0.012 per update while a token at 0.9 is
already saturated. This asymmetry starves low-probability exploratory tokens
and drives entropy collapse. Raising the upper bound to 0.28 kept 62% of
initial entropy at step 200 (vs 35% with 0.2) and gave +1.4 D-Num on 9B.

Aggregation modes (``aggregate``):

* ``token-mean``: ``Σ ℓ / (#valid tokens in this batch)``. The divisor
  depends on the batch, so removing samples, including zero-advantage ones
  whose ``ℓ`` has zero gradient, changes the scale of every remaining token's
  gradient. Long answers also dominate. We do **not** use it.
* ``fixed-constant``: ``Σ ℓ / C`` with ``C`` a configured constant (the
  per-update token budget). Removing zero-gradient samples cannot change the
  gradient (ablation A10, report §单步时间预算 设计说明-2).
* ``token-sum``: the ``C = 1`` special case.
* ``seq-mean-token-mean``: provided for comparison and tests only.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch


def _broadcast_adv(adv: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    if adv.dim() == 1:
        return adv.unsqueeze(-1).expand_as(like)
    if adv.shape != like.shape:
        raise ValueError(f"advantage shape {tuple(adv.shape)} incompatible with {tuple(like.shape)}")
    return adv


def ppo_clip_token_loss(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    adv: torch.Tensor,
    *,
    clip_low: float = 0.2,
    clip_high: float = 0.28,
    dual_clip_c: float | None = 3.0,
    is_weights: torch.Tensor | None = None,
    log_ratio_clamp: float = 20.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Token-level PPO-clip loss matrix ``(B, T)`` and diagnostic metrics.

    ``adv`` may be per-sequence ``(B,)`` or per-token ``(B, T)``. ``is_weights``
    are the TIS weights from :func:`dpt.rl.tis.tis_weights`. They are treated as
    constants (detached), exactly like verl's ``rollout_is_weights``.

    Positively homogeneous in ``adv``: ``loss(c·adv) == c·loss(adv)`` for ``c > 0``.
    """
    if dual_clip_c is not None and dual_clip_c <= 1.0:
        raise ValueError("dual_clip_c must be > 1")
    a = _broadcast_adv(adv.to(logp.dtype), logp)
    log_ratio = torch.clamp(logp - old_logp, -log_ratio_clamp, log_ratio_clamp)
    ratio = torch.exp(log_ratio)
    l1 = -a * ratio
    l2 = -a * torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)
    clipped = torch.maximum(l1, l2)
    loss = clipped
    if dual_clip_c is not None:
        l3 = -a * dual_clip_c
        loss = torch.where(a < 0, torch.minimum(clipped, l3), clipped)
    if is_weights is not None:
        loss = loss * is_weights.detach().to(loss.dtype)
    with torch.no_grad():
        metrics = {
            "clipfrac": float((l2 > l1).float().mean()),
            "clipfrac_high": float(((ratio > 1 + clip_high) & (a > 0)).float().mean()),
            "clipfrac_low": float(((ratio < 1 - clip_low) & (a < 0)).float().mean()),
            "ratio_mean": float(ratio.mean()),
        }
    return loss, metrics


def aggregate(
    loss_mat: torch.Tensor,
    mask: torch.Tensor,
    mode: str,
    *,
    constant: float | None = None,
) -> torch.Tensor:
    """Reduce a ``(B, T)`` token-loss matrix to a scalar. See the module docstring for the modes."""
    m = mask.to(loss_mat.dtype)
    total = (loss_mat * m).sum()
    if mode == "token-mean":
        return total / m.sum().clamp_min(1.0)
    if mode == "token-sum":
        return total
    if mode == "fixed-constant":
        if constant is None or constant <= 0:
            raise ValueError("fixed-constant aggregation needs a positive `constant`")
        return total / constant
    if mode == "seq-mean-token-mean":
        tok = m.sum(-1)
        seq = (loss_mat * m).sum(-1) / tok.clamp_min(1.0)
        has = (tok > 0).to(loss_mat.dtype)
        return (seq * has).sum() / has.sum().clamp_min(1.0)
    raise ValueError(f"unknown aggregation mode {mode!r}")


def grouped_task_loss(
    loss_mat: torch.Tensor,
    mask: torch.Tensor,
    task_ids: Sequence[str],
    weights: Mapping[str, float],
    budgets: Mapping[str, float],
) -> torch.Tensor:
    """Per-task normalized objective ``Σ_k (w_k / B_k) Σ_{t ∈ task k} ℓ_t`` computed directly.

    This is the form described in §多任务 RL 难调 调整-1. Production gets the
    same value by scaling advantages with
    :meth:`dpt.rl.advantage.TaskLossNormalizer.multipliers` and keeping a
    fixed-constant aggregation. ``tests/rl/test_loss.py`` checks that the two
    agree.
    """
    m = mask.to(loss_mat.dtype)
    per_seq = (loss_mat * m).sum(-1)
    tids = np.asarray(task_ids)
    total = loss_mat.new_zeros(())
    for t in set(task_ids):
        b = float(budgets.get(t, 0.0))
        w = float(weights.get(t, 0.0))
        if b <= 0 or w <= 0:
            continue
        sel = torch.as_tensor(tids == t)
        total = total + (w / b) * per_seq[sel].sum()
    return total
