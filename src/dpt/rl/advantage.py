"""Advantage computation for multi-task GRPO.

Three pieces, applied in this order:

1. :func:`group_mean_advantage`: ``A_i = r_i - mean_{j in G(i)} r_j`` over the
   8 samples of a prompt, **without** dividing by the group std.
   Ablation A2: dividing by std inflates near-solved groups (7/8 correct, std
   ≈ 0.35) and near-impossible groups (1/8 correct) by up to ~3×. That is a
   difficulty bias. It also makes a lucky guess on a hopeless problem the
   loudest gradient in the batch. Not dividing gave +0.9 D-Num and +2.3
   hard-problem pass@8 on 9B, and the effect held on 27B.

2. :class:`TaskAdvantageScaler`: for tasks with continuous, low-variance
   rewards (T3 rubric scores sit in a narrow band), divide that task's
   advantages by its *running* advantage std and multiply by a reference
   scale, so a "clearly better" T3 sample moves the policy about as much as a
   correct T1 answer (§多任务 RL 难调, 调整-2). Within a group we still do not
   divide by the group std; this is a slow per-task constant, not a
   per-group normalizer.

3. :class:`TaskLossNormalizer`: per-task loss-share multipliers. Under a
   token-level loss, a task's gradient share is proportional to its *token*
   count. T1 had 50% of prompts but only 27% of tokens, and T2+T3 long
   trajectories took 58% (§多任务 RL 难调, 诊断-2). We want

       L = Σ_k (w_k / B_k) Σ_{t ∈ task k} ℓ_t

   with fixed per-task token budgets ``B_k`` and target weights ``w_k``.

   **Implementation trick.** The PPO-clip token loss (including dual-clip) is
   *positively homogeneous* in the advantage: ``ℓ(c·A) = c·ℓ(A)`` for every
   ``c > 0`` (see :func:`dpt.rl.loss.ppo_clip_token_loss`; ``max``/``min`` of
   terms that are all linear in ``A`` commute with positive scaling, and the
   sign of ``A``, which selects the dual-clip branch, is unchanged). So if the
   framework aggregates ``Σ_t ℓ_t / C`` for a *fixed* constant ``C``,
   multiplying each sample's advantage by ``m_k = w_k · C / B_k`` gives exactly
   the objective above. The trainer workers therefore keep the stock loss.
   Only the driver-side advantage changes, so no custom code has to be
   registered inside Ray worker processes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from dpt.common.types import TASKS


# ----------------------------------------------------------------------------------------------
# 1. Group-mean advantage
# ----------------------------------------------------------------------------------------------
def group_mean_advantage(
    scores: Sequence[float] | np.ndarray,
    group_ids: Sequence[Any],
    valid: Sequence[bool] | np.ndarray | None = None,
) -> np.ndarray:
    """Per-sample advantage ``r_i - mean(r_group)``, no std division (ablation A2).

    Args:
        scores: scalar reward per sample.
        group_ids: group key per sample (verl's ``uid``: one per prompt).
        valid: optional mask. Invalid samples (e.g. truncated at the 24K cap,
            which "do not enter the loss") are excluded from the group mean and
            receive advantage 0. This keeps "unfinished" from being treated as
            "wrong" for the other members of the group.

    Groups with fewer than two valid members get advantage 0. With one sample
    there is no baseline, so any non-zero value would be pure noise.
    """
    r = np.asarray(scores, dtype=np.float64)
    n = r.shape[0]
    if len(group_ids) != n:
        raise ValueError(f"len(group_ids)={len(group_ids)} != len(scores)={n}")
    v = np.ones(n, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    adv = np.zeros(n, dtype=np.float64)
    groups: dict[Any, list[int]] = {}
    for i, g in enumerate(group_ids):
        groups.setdefault(g, []).append(i)
    for idx in groups.values():
        members = [i for i in idx if v[i]]
        if len(members) < 2:
            continue
        mean = r[members].mean()
        adv[members] = r[members] - mean
    return adv


# ----------------------------------------------------------------------------------------------
# 2. Per-task running-std scaling
# ----------------------------------------------------------------------------------------------
@dataclass
class _RunningMoments:
    """EMA of the first two moments; ``count`` tracks how many updates have been folded in."""

    mean: float = 0.0
    sq_mean: float = 0.0
    count: int = 0

    def update(self, x: np.ndarray, momentum: float) -> None:
        if x.size == 0:
            return
        m, s = float(x.mean()), float((x**2).mean())
        if self.count == 0:
            self.mean, self.sq_mean = m, s
        else:
            self.mean = momentum * self.mean + (1 - momentum) * m
            self.sq_mean = momentum * self.sq_mean + (1 - momentum) * s
        self.count += 1

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self.sq_mean - self.mean**2, 0.0)))


class TaskAdvantageScaler:
    """Rescale advantages of selected tasks to a common scale using running std estimates.

    ``A'_i = A_i * target_std / max(std_k, min_std)`` for samples of a scaled task ``k``.

    ``target_std`` is either a float or ``None``. With ``None``, the target is
    the pooled running std of the *unscaled* tasks (e.g. binary T1/T2/T4), so
    T3 advantages are mapped onto whatever scale the verifiable tasks currently
    have. Statistics only use samples from non-degenerate groups. Zero-advantage
    groups would otherwise drag the std toward 0 and blow the scale up.

    Running statistics are tracked for every task (they are also useful
    monitoring signals); only ``scaled_tasks`` are rescaled.
    """

    def __init__(
        self,
        scaled_tasks: Sequence[str] = ("t3_report",),
        *,
        momentum: float = 0.9,
        min_std: float = 0.05,
        target_std: float | None = None,
        max_gain: float = 4.0,
    ) -> None:
        unknown = set(scaled_tasks) - set(TASKS)
        if unknown:
            raise ValueError(f"unknown tasks in scaled_tasks: {unknown}")
        self.scaled_tasks = tuple(scaled_tasks)
        self.momentum = momentum
        self.min_std = min_std
        self.target_std = target_std
        self.max_gain = max_gain  # cap the multiplier: early in training the std estimate is noisy
        self.moments: dict[str, _RunningMoments] = {t: _RunningMoments() for t in TASKS}

    def update(self, adv: np.ndarray, task_ids: Sequence[str], valid: np.ndarray | None = None) -> None:
        adv = np.asarray(adv, dtype=np.float64)
        tids = np.asarray(task_ids)
        v = np.ones_like(adv, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
        for t in TASKS:
            sel = (tids == t) & v
            self.moments[t].update(adv[sel], self.momentum)

    def _reference_std(self) -> float | None:
        if self.target_std is not None:
            return self.target_std
        stds = [
            self.moments[t].std
            for t in TASKS
            if t not in self.scaled_tasks and self.moments[t].count > 0
        ]
        return float(np.mean(stds)) if stds else None

    def gains(self) -> dict[str, float]:
        """Current multiplier per task (1.0 for unscaled tasks or before statistics exist)."""
        ref = self._reference_std()
        out = {t: 1.0 for t in TASKS}
        if ref is None:
            return out
        for t in self.scaled_tasks:
            m = self.moments[t]
            if m.count == 0:
                continue
            out[t] = float(min(ref / max(m.std, self.min_std), self.max_gain))
        return out

    def scale(self, adv: np.ndarray, task_ids: Sequence[str]) -> np.ndarray:
        g = self.gains()
        mult = np.array([g[t] for t in task_ids], dtype=np.float64)
        return np.asarray(adv, dtype=np.float64) * mult

    def update_and_scale(
        self, adv: np.ndarray, task_ids: Sequence[str], valid: np.ndarray | None = None
    ) -> np.ndarray:
        """Fold the current batch into the running stats, then rescale it."""
        self.update(adv, task_ids, valid)
        return self.scale(adv, task_ids)

    # -- checkpointing (Incident 2: every stateful component must round-trip exactly) --------
    def state_dict(self) -> dict[str, Any]:
        return {
            "scaled_tasks": list(self.scaled_tasks),
            "momentum": self.momentum,
            "min_std": self.min_std,
            "target_std": self.target_std,
            "max_gain": self.max_gain,
            "moments": {t: vars(m).copy() for t, m in self.moments.items()},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if tuple(state["scaled_tasks"]) != self.scaled_tasks:
            raise ValueError(
                f"checkpoint scaled_tasks {state['scaled_tasks']} != configured {self.scaled_tasks}"
            )
        self.momentum = state["momentum"]
        self.min_std = state["min_std"]
        self.target_std = state["target_std"]
        self.max_gain = state["max_gain"]
        self.moments = {t: _RunningMoments(**m) for t, m in state["moments"].items()}


# ----------------------------------------------------------------------------------------------
# 3. Per-task loss-share multipliers (fixed token budgets)
# ----------------------------------------------------------------------------------------------
def largest_remainder_counts(mix: Mapping[str, float], total: int) -> dict[str, int]:
    """Integer allocation of ``total`` items proportional to ``mix`` that sums exactly to ``total``.

    Ties are broken by task order in :data:`TASKS`, so the result is deterministic.
    """
    weights = {t: max(float(mix.get(t, 0.0)), 0.0) for t in TASKS}
    s = sum(weights.values())
    if s <= 0:
        raise ValueError("mix has no positive weight")
    raw = {t: total * w / s for t, w in weights.items()}
    counts = {t: int(np.floor(x)) for t, x in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(TASKS, key=lambda t: (-(raw[t] - counts[t]), TASKS.index(t)))
    for t in order[:remainder]:
        counts[t] += 1
    return counts


@dataclass
class TaskLossNormalizer:
    """Compute fixed per-task token budgets ``B_k`` and the advantage multipliers ``w_k·C/B_k``.

    Budgets are a *design* quantity that never depends on the current batch:

        B_k = (nominal prompts of task k per step) × group_size × L_k / num_updates

    where ``L_k`` is the configured typical response length of task k. They
    are recomputed only when the curriculum *stage* changes (the nominal mix
    changes), so they stay constant within a stage. Filtering groups, a
    batch with unusually long answers, or auto-reweighting do not change them.
    That is the precondition for zero-advantage filtering to leave the
    gradient unchanged (report §单步时间预算 设计说明-2; ablation A10).

    ``constant`` is the fixed ``C`` the framework divides the summed token loss
    by. It is 1 for a pure token-sum aggregation. For verl's
    ``seq-mean-token-sum-norm`` it is ``global_batch_size × loss_scale_factor``,
    both configured constants. Set it to whatever the adapter's aggregation
    divides by.
    """

    response_tokens: Mapping[str, float]
    group_size: int = 8
    prompts_per_step: int = 128
    num_updates: int = 2
    constant: float = 1.0
    weights: Mapping[str, float] | None = None  # None → w_k = nominal mix share

    def budgets(self, mix: Mapping[str, float]) -> dict[str, float]:
        counts = largest_remainder_counts(mix, self.prompts_per_step)
        return {
            t: counts[t] * self.group_size * float(self.response_tokens.get(t, 0.0)) / self.num_updates
            for t in TASKS
        }

    def target_weights(self, mix: Mapping[str, float]) -> dict[str, float]:
        src = self.weights if self.weights is not None else mix
        w = {t: max(float(src.get(t, 0.0)), 0.0) if float(mix.get(t, 0.0)) > 0 else 0.0 for t in TASKS}
        s = sum(w.values())
        if s <= 0:
            raise ValueError("no task has positive weight under this mix")
        return {t: v / s for t, v in w.items()}

    def multipliers(self, mix: Mapping[str, float]) -> dict[str, float]:
        """``m_k = w_k · C / B_k`` (0 for tasks absent from the mix)."""
        b = self.budgets(mix)
        w = self.target_weights(mix)
        return {t: (w[t] * self.constant / b[t]) if b[t] > 0 and w[t] > 0 else 0.0 for t in TASKS}

    def apply(self, adv: np.ndarray, task_ids: Sequence[str], mix: Mapping[str, float]) -> np.ndarray:
        m = self.multipliers(mix)
        missing = {t for t in task_ids if m.get(t, 0.0) == 0.0}
        if missing:
            # A sample of a task with zero weight in this stage should never have been sampled.
            raise ValueError(f"samples from tasks with zero loss weight in this stage: {sorted(missing)}")
        return np.asarray(adv, dtype=np.float64) * np.array([m[t] for t in task_ids], dtype=np.float64)

    def gradient_shares(
        self, task_ids: Sequence[str], token_counts: Sequence[int], mix: Mapping[str, float]
    ) -> dict[str, float]:
        """Fraction of total loss weight each task receives for an actual batch (monitoring).

        Under a plain token loss this would equal each task's token share. With
        the multipliers it tracks ``w_k`` up to the realized/budgeted token ratio.
        """
        m = self.multipliers(mix)
        mass = {t: 0.0 for t in TASKS}
        for t, n in zip(task_ids, token_counts):
            mass[t] += m.get(t, 0.0) * float(n)
        s = sum(mass.values())
        return {t: (v / s if s > 0 else 0.0) for t, v in mass.items()}


def token_shares(task_ids: Sequence[str], token_counts: Sequence[int]) -> dict[str, float]:
    """Each task's share of loss tokens. This equals its gradient share under token-mean loss."""
    mass = {t: 0.0 for t in TASKS}
    for t, n in zip(task_ids, token_counts):
        mass[t] += float(n)
    s = sum(mass.values())
    return {t: (v / s if s > 0 else 0.0) for t, v in mass.items()}
