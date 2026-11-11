"""The per-step advantage pipeline: one framework-free function the verl adapter calls.

Order of operations (report §阶段二 RL 设计 + §多任务 RL 难调 调整):

1. **Valid mask.** A sequence takes part in the loss only if it is a real
   sample (not one of verl's divisibility pads) and was not truncated at the
   response cap. Truncated samples are also left out of their group's
   baseline: "not finished" is not "wrong", so they must not drag the mean
   down for the other members (§算法配置 长度控制).
2. **Group-mean advantage** ``A_i = r_i − mean_valid(r_group)`` with no division
   by the group std (ablation A2).
3. **Per-task scale** (T3 only by default). A running std maps the
   continuous judge reward onto the scale of the binary tasks (调整-2). The
   statistics skip invalid samples and degenerate groups.
4. **Per-task loss normalization** (调整-1). Multiply by ``w_k·C/B_k`` (fixed
   budgets from the stage's *nominal* mix). Under a token-sum loss each task's
   gradient share is then set by ``w_k``, not by how long its answers are.
5. **Zero** the advantage of every invalid sequence. The adapter also zeroes
   its loss mask.

Everything returned is plain numpy. Two metric helpers:

* :func:`panel_metrics`: the per-task dashboard values (reward, accuracy,
  truncation, length split by correct/incorrect, prompt and token shares). It
  is computed on the **full, unfiltered batch** in the replay buffer. Computing
  it after zero-advantage filtering would hide exactly what it must show:
  all-wrong groups (forgetting) and truncated responses are what the filter
  removes.
* the ``metrics`` of :class:`StepAdvantages`: values about the *trained* batch
  (advantage gains, gradient shares), named per :mod:`dpt.monitoring.keys`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from dpt.common.types import TASKS
from dpt.rl.advantage import TaskAdvantageScaler, TaskLossNormalizer, group_mean_advantage, token_shares
from dpt.rl.filtering import zero_advantage_mask


def _group_std(r: np.ndarray, gids: Sequence[str], valid: np.ndarray) -> np.ndarray:
    """Per-sample std of its group's valid rewards (population std; 0 for singleton groups)."""
    out = np.zeros_like(r)
    groups: dict[str, list[int]] = {}
    for i, g in enumerate(gids):
        groups.setdefault(g, []).append(i)
    for idx in groups.values():
        v = [i for i in idx if valid[i]]
        if len(v) > 1:
            out[idx] = float(np.std(r[v]))
    return out


@dataclass
class StepAdvantages:
    seq_adv: np.ndarray            # [B] final per-sequence advantage (0 where invalid)
    valid: np.ndarray              # [B] bool: sequence contributes to the loss
    truncated: np.ndarray          # [B] bool
    metrics: dict[str, float] = field(default_factory=dict)


def compute_step_advantages(
    scores: Sequence[float] | np.ndarray,
    group_ids: Sequence[str],
    task_ids: Sequence[str],
    response_lengths: Sequence[int] | np.ndarray,
    *,
    max_response: int,
    scaler: TaskAdvantageScaler,
    normalizer: TaskLossNormalizer | None,
    nominal_mix: Mapping[str, float],
    is_padding: Sequence[bool] | np.ndarray | None = None,
    eps: float = 1e-6,
    normalize_by_group_std: bool = False,
) -> StepAdvantages:
    r = np.asarray(scores, dtype=np.float64)
    n = len(r)
    lens = np.asarray(response_lengths, dtype=np.int64)
    tids = np.asarray([str(t) for t in task_ids], dtype=object)
    pad = np.zeros(n, dtype=bool) if is_padding is None else np.asarray(is_padding, dtype=bool)
    gids = [str(g) for g in group_ids]

    truncated = (lens >= max_response) & ~pad
    valid = ~truncated & ~pad

    adv = group_mean_advantage(r, gids, valid=valid)
    degenerate = zero_advantage_mask(r, gids, valid=valid, eps=eps)
    if normalize_by_group_std:
        # Ablation A2 only (original GRPO). Production keeps it off: dividing by the group std
        # inflates nearly-all-correct / nearly-all-wrong groups and biases learning by difficulty.
        adv = adv / (_group_std(r, gids, valid) + eps)
    stat_mask = valid & ~degenerate
    real = ~pad
    adv_real = scaler.update_and_scale(adv[real], list(tids[real]), stat_mask[real])
    if normalizer is not None:
        adv_real = normalizer.apply(adv_real, list(tids[real]), nominal_mix)
    adv = adv.copy()
    adv[real] = adv_real
    adv[~valid] = 0.0

    # Metrics of the TRAINED batch only (panel metrics such as reward/accuracy/length/truncation are
    # computed on the full, unfiltered batch by panel_metrics() in the replay buffer).
    metrics: dict[str, float] = {}
    present = [t for t in TASKS if np.any(tids[real] == t)]
    for t, g in scaler.gains().items():
        if t in present:
            metrics[f"task/{t}/adv_scale_gain"] = g
    if present:
        ts = token_shares(list(tids[valid]), list(lens[valid]))
        for t in present:
            metrics[f"task/{t}/trained_token_share"] = ts[t]
        if normalizer is not None:
            gs = normalizer.gradient_shares(list(tids[valid]), list(lens[valid]), nominal_mix)
            for t in present:
                metrics[f"task/{t}/grad_share"] = gs[t]
    metrics["batch/trained_valid_frac"] = float(valid[real].mean()) if real.any() else 0.0
    return StepAdvantages(seq_adv=adv, valid=valid, truncated=truncated, metrics=metrics)


def panel_metrics(
    scores: Sequence[float] | np.ndarray,
    correct: Sequence[bool] | np.ndarray,
    task_ids: Sequence[str],
    group_ids: Sequence[str],
    response_lengths: Sequence[int] | np.ndarray,
    truncated: Sequence[bool] | np.ndarray,
) -> dict[str, float]:
    """Per-task panel values on the full rollout batch (report §默认监控面板, §多任务 诊断-1/2).

    ``correct`` must come from the grader's own verdict (reward ``acc``), not from
    ``score >= 1``: the final score includes the overlong penalty, so a correct answer of
    20K-24K tokens would otherwise be counted as wrong.
    """
    r = np.asarray(scores, dtype=np.float64)
    ok = np.asarray(correct, dtype=bool)
    lens = np.asarray(response_lengths, dtype=np.int64)
    trunc = np.asarray(truncated, dtype=bool)
    tids = np.asarray([str(t) for t in task_ids], dtype=object)
    gids = [str(g) for g in group_ids]
    out: dict[str, float] = {}
    n_groups = max(1, len(set(gids)))
    ts = token_shares(list(tids), list(lens))
    for t in TASKS:
        sel = tids == t
        if not sel.any():
            continue
        out[f"task/{t}/reward_mean"] = float(r[sel].mean())
        out[f"task/{t}/acc"] = float(ok[sel].mean())
        out[f"task/{t}/truncation_rate"] = float(trunc[sel].mean())
        out[f"task/{t}/len_mean"] = float(lens[sel].mean())
        for name, m in (("correct", ok), ("incorrect", ~ok)):
            s = sel & m & ~trunc
            if s.any():
                out[f"task/{t}/len_{name}"] = float(lens[s].mean())
        out[f"task/{t}/n_samples"] = float(sel.sum())
        out[f"task/{t}/prompt_share"] = len({gids[i] for i in np.flatnonzero(sel)}) / n_groups
        out[f"task/{t}/token_share"] = ts[t]
    if len(r):
        out["batch/truncation_rate"] = float(trunc.mean())
    return out
