"""Batch filtering before the expensive phases of an RL step.

Zero-advantage groups
---------------------
A group whose 8 rewards are all equal (all correct, all wrong, or the same
rubric score) has advantage 0 for every token. Its policy-gradient
contribution is exactly zero, but under the baseline pipeline it still paid
for an old-logprob forward pass and a full forward/backward. Dropping such
groups *before* old-logprob recomputation removed ~35% of tokens and saved
4.4 min/step (ablation A10; report §单步时间预算).

Precondition: the loss must be normalized by a fixed constant, not by the
batch's token count. Otherwise dropping samples rescales the gradient of
every remaining token (see :mod:`dpt.rl.loss` and the unit test
``test_zero_adv_filter_keeps_gradient_under_fixed_constant``).

The zero-advantage fraction per task is also a monitoring signal. 20–45% is
healthy. More than 60% for 30 steps means a task is too easy or too hard
and triggers curriculum auto-reweighting (§多任务 RL 难调 调整-6).

Truncated responses
-------------------
Responses that hit the 24K cap are excluded from the loss (§算法配置 长度控制:
"被截断的回答不进 loss"). Unfinished is not the same as wrong, and training on
a cut-off trace teaches the model to stop mid-thought.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass
class FilterResult:
    keep_idx: np.ndarray                  # indices of samples to keep (sorted)
    drop_idx: np.ndarray                  # indices of dropped samples (sorted)
    stats: dict[str, float] = field(default_factory=dict)


def zero_advantage_mask(
    scores: Sequence[float] | np.ndarray,
    group_ids: Sequence[Any],
    *,
    valid: Sequence[bool] | np.ndarray | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """``True`` for every sample whose group carries no learning signal.

    A group is degenerate when its valid members have score range ≤ ``eps``
    or it has fewer than two valid members. Matches the zero-advantage
    condition of :func:`dpt.rl.advantage.group_mean_advantage`.
    """
    r = np.asarray(scores, dtype=np.float64)
    v = np.ones(len(r), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    groups: dict[Any, list[int]] = {}
    for i, g in enumerate(group_ids):
        groups.setdefault(g, []).append(i)
    out = np.zeros(len(r), dtype=bool)
    for idx in groups.values():
        members = [i for i in idx if v[i]]
        if len(members) < 2 or (r[members].max() - r[members].min()) <= eps:
            out[idx] = True
    return out


def filter_zero_advantage(
    scores: Sequence[float] | np.ndarray,
    group_ids: Sequence[Any],
    *,
    task_ids: Sequence[str] | None = None,
    response_lengths: Sequence[int] | None = None,
    valid: Sequence[bool] | np.ndarray | None = None,
    eps: float = 1e-6,
    keep_multiple_of: int = 1,
) -> FilterResult:
    """Drop all samples of degenerate groups; return kept/dropped indices and statistics.

    ``keep_multiple_of`` keeps the batch divisible by the data-parallel size
    (or ``dp × micro-batch``) the trainer needs. When the kept count is not a
    multiple, the smallest degenerate groups are re-admitted whole until it
    is. Re-admitted groups contribute zero gradient, so correctness is
    unchanged; we only pay a little compute. Groups are re-admitted whole so
    each group keeps all its members.
    """
    r = np.asarray(scores, dtype=np.float64)
    gids = list(group_ids)
    degenerate = zero_advantage_mask(r, gids, valid=valid, eps=eps)

    if keep_multiple_of > 1:
        kept = int((~degenerate).sum())
        if kept % keep_multiple_of:
            # Re-admit whole degenerate groups (smallest token cost first) until divisible.
            dgroups: dict[Any, list[int]] = {}
            for i, g in enumerate(gids):
                if degenerate[i]:
                    dgroups.setdefault(g, []).append(i)
            lens = np.asarray(response_lengths) if response_lengths is not None else np.ones(len(r))
            order = sorted(dgroups.items(), key=lambda kv: (float(lens[kv[1]].sum()), str(kv[0])))
            for _, idx in order:
                if kept % keep_multiple_of == 0:
                    break
                degenerate[idx] = False
                kept += len(idx)
            if kept % keep_multiple_of:
                raise ValueError(
                    f"cannot make kept batch ({kept}) divisible by {keep_multiple_of}; "
                    "group size must divide the trainer's batch granularity"
                )

    keep_idx = np.flatnonzero(~degenerate)
    drop_idx = np.flatnonzero(degenerate)
    stats: dict[str, float] = {
        "filter/zero_adv_sample_frac": float(degenerate.mean()) if len(r) else 0.0,
    }
    if response_lengths is not None:
        lens = np.asarray(response_lengths, dtype=np.float64)
        tot = lens.sum()
        stats["filter/zero_adv_token_frac"] = float(lens[drop_idx].sum() / tot) if tot > 0 else 0.0
    if task_ids is not None:
        tids = np.asarray(task_ids)
        # Use the raw degenerate mask (before re-admission) for monitoring.
        raw_degenerate = zero_advantage_mask(r, gids, valid=valid, eps=eps)
        members: dict[Any, list[int]] = {}
        for i, g in enumerate(gids):
            members.setdefault(g, []).append(i)
        for t in sorted(set(task_ids)):
            sel = tids == t
            # Fraction of this task's *groups* that are degenerate (the monitored quantity).
            g_all = {gids[i] for i in np.flatnonzero(sel)}
            g_deg = {gids[i] for i in np.flatnonzero(sel & raw_degenerate)}
            stats[f"filter/{t}/zero_adv_group_frac"] = len(g_deg) / max(1, len(g_all))
            # Split degenerate groups into "all solved" vs "all failed" (too easy vs too hard).
            solved = sum(1 for g in g_deg if r[members[g]].min() >= 1.0 - eps)
            stats[f"filter/{t}/all_solved_group_frac"] = solved / max(1, len(g_all))
    return FilterResult(keep_idx=keep_idx, drop_idx=drop_idx, stats=stats)


def truncation_loss_mask(response_mask: np.ndarray, truncated: Sequence[bool] | np.ndarray) -> np.ndarray:
    """Zero the loss mask of truncated responses (they stay in the batch for logging)."""
    m = np.asarray(response_mask).copy()
    t = np.asarray(truncated, dtype=bool)
    m[t] = 0
    return m
