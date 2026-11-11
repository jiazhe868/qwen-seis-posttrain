"""Per-task metric panels and gradient-share decomposition.

The report's multi-task lesson: aggregate reward was fine while T1 had
stalled at step 140 and general ability was eroding. Nothing was visible
until every panel was split by task (report §多任务 RL 难调-诊断-1). This
module turns per-sample arrays from one RL step into ``task/<task>/<metric>``
scalars.

The gradient-share decomposition (diagnosis step 2) explains *why* T1
stalled. T1 made up 50% of prompts but only 27% of tokens, while T2 and T3's
long trajectories and reports took 58% of tokens. Under a token-level loss,
gradient follows tokens, so T1 received far less gradient than its mixture
weight implied. :func:`gradient_share` logs both shares every step, so that
gap is visible.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

from dpt.monitoring import keys as K


def _arr(x: Sequence | np.ndarray | None, dtype=float) -> np.ndarray | None:
    return None if x is None else np.asarray(x, dtype=dtype)


def zero_adv_fraction(rewards: np.ndarray, group_ids: Sequence) -> float:
    """Fraction of prompt groups whose samples all received the same reward (no learning signal)."""
    groups: dict[object, list[float]] = defaultdict(list)
    for r, g in zip(rewards, group_ids):
        groups[g].append(float(r))
    if not groups:
        return 0.0
    return sum(1 for v in groups.values() if np.ptp(v) == 0.0) / len(groups)


def per_task_panels(
    task_ids: Sequence[str],
    rewards: Sequence[float],
    response_lengths: Sequence[int],
    *,
    correct: Sequence[bool] | None = None,
    truncated: Sequence[bool] | None = None,
    entropies: Sequence[float] | None = None,
    clip_fracs: Sequence[float] | None = None,
    group_ids: Sequence | None = None,
) -> dict[str, float]:
    """Aggregate one step's per-sample arrays into per-task scalars.

    ``entropies`` and ``clip_fracs`` are per-sample means over response
    tokens. ``group_ids`` identifies samples of the same prompt; it is needed
    for the zero-advantage fraction.
    """
    t = np.asarray(task_ids, dtype=object)
    r = np.asarray(rewards, dtype=float)
    L = np.asarray(response_lengths, dtype=float)
    c = _arr(correct, bool)
    tr = _arr(truncated, bool)
    ent = _arr(entropies)
    cf = _arr(clip_fracs)
    g = None if group_ids is None else np.asarray(group_ids, dtype=object)
    out: dict[str, float] = {}
    for task in sorted(set(t.tolist())):
        m = t == task
        out[K.task_key(K.TASK_COUNT, task)] = float(m.sum())
        out[K.task_key(K.TASK_REWARD, task)] = float(r[m].mean())
        out[K.task_key(K.TASK_LEN, task)] = float(L[m].mean())
        if c is not None:
            out[K.task_key(K.TASK_ACC, task)] = float(c[m].mean())
            if (m & c).any():
                out[K.task_key(K.TASK_LEN_CORRECT, task)] = float(L[m & c].mean())
            if (m & ~c).any():
                out[K.task_key(K.TASK_LEN_INCORRECT, task)] = float(L[m & ~c].mean())
        if tr is not None:
            out[K.task_key(K.TASK_TRUNC, task)] = float(tr[m].mean())
        if ent is not None:
            out[K.task_key(K.TASK_ENTROPY, task)] = float(ent[m].mean())
        if cf is not None:
            out[K.task_key(K.TASK_CLIP, task)] = float(cf[m].mean())
        if g is not None:
            out[K.task_key(K.TASK_ZERO_ADV, task)] = zero_adv_fraction(r[m], g[m].tolist())
    return out


def gradient_share(
    task_ids: Sequence[str],
    token_counts: Sequence[int],
    *,
    group_ids: Sequence | None = None,
    loss_weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Prompt share vs token share per task, i.e. who actually receives the gradient.

    ``loss_weights`` (optional) is a per-task multiplier applied to tokens,
    for example the per-task normalization ``w_k / B_k`` from
    ``dpt.rl.advantage``. With it, the token share becomes the *effective*
    gradient share after re-weighting, so the panel shows whether the fix
    worked.
    """
    t = np.asarray(task_ids, dtype=object)
    tok = np.asarray(token_counts, dtype=float)
    w = np.array([1.0 if loss_weights is None else float(loss_weights.get(x, 1.0)) for x in t])
    eff = tok * w
    if group_ids is not None:
        firsts: dict[object, str] = {}
        for gid, task in zip(group_ids, t):
            firsts.setdefault(gid, task)
        prompt_tasks = np.asarray(list(firsts.values()), dtype=object)
    else:
        prompt_tasks = t
    out: dict[str, float] = {}
    for task in sorted(set(t.tolist())):
        out[K.task_key(K.TASK_PROMPT_SHARE, task)] = float((prompt_tasks == task).mean())
        out[K.task_key(K.TASK_TOKEN_SHARE, task)] = float(eff[t == task].sum() / max(eff.sum(), 1e-12))
    return out


def memory_panel(per_rank_peak_gb: Sequence[float]) -> dict[str, float]:
    """Max and argmax of per-rank peak memory (incident 4: one PP-stage-2 rank OOMed)."""
    a = np.asarray(per_rank_peak_gb, dtype=float)
    return {K.MEMORY_PEAK_MAX_RANK: float(a.max()), K.MEMORY_PEAK_ARGMAX_RANK: float(a.argmax())}
