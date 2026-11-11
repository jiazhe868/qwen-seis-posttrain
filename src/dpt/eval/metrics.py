"""Evaluation metrics: avg@k, unbiased pass@k, pass@k curves, win rates, token and truncation stats.

All score-like metrics are returned on a 0–100 scale, matching the report's tables.

``scores`` arrays have shape ``[n_items, n_samples]``. An entry is the
per-sample score in [0, 1] (1/0 for exact graders). ``avg@k`` averages over
samples first and then over items, so every item has equal weight no matter
how many samples it has.
"""

from __future__ import annotations

from math import comb
from typing import Iterable, Sequence

import numpy as np


def avg_at_k(scores: np.ndarray) -> float:
    """Mean over items of the per-item mean over samples (×100). D-Num/D-Fresh use avg@8."""
    s = np.asarray(scores, dtype=float)
    if s.ndim != 2 or s.size == 0:
        raise ValueError("scores must be a non-empty [n_items, n_samples] array")
    return float(100.0 * s.mean(axis=1).mean())


def per_item_means(scores: np.ndarray) -> np.ndarray:
    """Per-item mean over samples (0–1). This is the unit the item-level bootstrap resamples."""
    return np.asarray(scores, dtype=float).mean(axis=1)


def pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for one item with ``c`` correct out of ``n`` samples (Chen et al., 2021).

    ``1 - C(n-c, k) / C(n, k)``, computed exactly with integer binomials.
    Requires ``k <= n``.
    """
    if k > n:
        raise ValueError(f"k={k} > n={n}; draw at least k samples per item")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_at_k(correct: np.ndarray, k: int) -> float:
    """Mean unbiased pass@k over items (×100). ``correct`` is a boolean ``[n_items, n]`` array."""
    c = np.asarray(correct, dtype=bool)
    n = c.shape[1]
    return float(100.0 * np.mean([pass_at_k_unbiased(n, int(row.sum()), k) for row in c]))


def pass_at_k_curve(correct: np.ndarray, ks: Sequence[int] = (1, 2, 4, 8, 16, 32, 64)) -> dict[int, float]:
    """pass@k for every ``k`` in ``ks`` that does not exceed the number of samples drawn.

    RL should raise pass@1 while pass@64 stays at or above base. Otherwise RL
    only sharpened the distribution and traded away exploration
    (report §第三层-3).
    """
    n = np.asarray(correct).shape[1]
    return {k: pass_at_k(correct, k) for k in ks if k <= n}


def win_rate(outcomes: Iterable[str]) -> float:
    """(wins + ½·ties) / n × 100 over outcomes in {"win", "loss", "tie"}."""
    o = list(outcomes)
    if not o:
        raise ValueError("no outcomes")
    bad = set(o) - {"win", "loss", "tie"}
    if bad:
        raise ValueError(f"unknown outcomes {bad}")
    return 100.0 * (o.count("win") + 0.5 * o.count("tie")) / len(o)


def win_rate_items(outcomes: Iterable[str]) -> np.ndarray:
    """Per-item win credit (1 / 0.5 / 0), the unit for bootstrapping a win-rate CI."""
    m = {"win": 1.0, "tie": 0.5, "loss": 0.0}
    return np.array([m[o] for o in outcomes], dtype=float)


def truncation_rate(finish_reasons: Iterable[str]) -> float:
    """Fraction (0–1) of samples whose generation hit the length limit (``finish_reason == "length"``).

    It is always reported because it bounds scores. Base at xhigh truncates
    11% of samples at 32K (report §harness 规则).
    """
    f = list(finish_reasons)
    return sum(x == "length" for x in f) / max(1, len(f))


def mean_output_tokens(token_counts: Iterable[int]) -> float:
    t = list(token_counts)
    return float(np.mean(t)) if t else 0.0
