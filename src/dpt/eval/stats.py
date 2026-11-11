"""Statistics: confidence intervals, paired comparisons, noise floor, Wilson intervals.

The rule this module enforces (report §评测体系): a difference between two
results counts as an improvement only if it exceeds the 95% confidence
interval. Changes smaller than the measured noise floor are noise.

Two kinds of uncertainty, two kinds of interval
-----------------------------------------------
1. **Generation noise on a fixed eval set.** Sampling at temperature 1.0
   gives a different score on every rerun of the same checkpoint. This is
   what the report's ``±`` numbers measure. Ten reruns on D-Num gave pairwise
   differences averaging 0.9 (std ≈ 0.8), so the 95% half-width is
   ≈ 1.96 × 0.8 ≈ 1.6. :func:`sampling_ci` estimates it from one run by
   resampling the k samples *within* each item, and :func:`noise_floor`
   measures it directly from repeated runs. **Gates and the ``±`` columns
   use this interval**, because the question is whether two measurements on
   the same items differ by more than rerun noise.
2. **Item sampling.** The 600 items are one draw from all possible domain
   problems. :func:`bootstrap_ci` resamples items and is the right interval
   for "how well would this generalize to new items of the same kind". It is
   wider, typically about 2× on heterogeneous sets.

For model-vs-model claims on the same items, use :func:`paired_bootstrap_diff`.
It resamples the same items for both models, so shared item difficulty
cancels, and it is the right test for "did RL improve over SFT on this set".
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class CI:
    point: float
    lo: float
    hi: float
    level: float = 0.95

    @property
    def halfwidth(self) -> float:
        return (self.hi - self.lo) / 2.0

    def to_dict(self) -> dict[str, float]:
        d = asdict(self)
        d["halfwidth"] = self.halfwidth
        return d


def bootstrap_ci(
    per_item: Sequence[float] | np.ndarray,
    *,
    n_boot: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
    scale: float = 100.0,
    stat: Callable[[np.ndarray], float] = np.mean,
) -> CI:
    """Percentile bootstrap CI of ``stat`` over items (values are per-item means in [0, 1]; scaled ×100)."""
    x = np.asarray(per_item, dtype=float)
    if x.size == 0:
        raise ValueError("empty sample")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = np.apply_along_axis(stat, 1, x[idx]) if stat is not np.mean else x[idx].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(boots, [alpha, 1.0 - alpha])
    return CI(point=float(scale * stat(x)), lo=float(scale * lo), hi=float(scale * hi), level=level)


def sampling_ci(
    scores: Sequence[Sequence[float]] | np.ndarray,
    *,
    n_boot: int = 2_000,
    level: float = 0.95,
    seed: int = 0,
    scale: float = 100.0,
) -> CI:
    """CI for generation noise with the item set fixed: resample the k samples within each item.

    ``scores`` is ``[n_items, k]``. With ``k == 1`` there is nothing to
    resample within an item. Use :func:`noise_floor` over reruns instead, or
    :func:`bootstrap_ci` for item variance.
    """
    s = np.asarray(scores, dtype=float)
    if s.ndim != 2 or s.shape[1] < 2:
        raise ValueError("sampling_ci needs [n_items, k>=2] scores")
    n, k = s.shape
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_boot, n, k))
    boots = np.take_along_axis(np.broadcast_to(s, (n_boot, n, k)), idx, axis=2).mean(axis=(1, 2))
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(boots, [alpha, 1.0 - alpha])
    return CI(point=float(scale * s.mean()), lo=float(scale * lo), hi=float(scale * hi), level=level)


@dataclass(frozen=True)
class DiffResult:
    diff: float  # mean(b) - mean(a), ×scale
    lo: float
    hi: float
    p_value: float  # two-sided bootstrap p-value for "no difference"
    significant: bool  # CI excludes 0

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def paired_bootstrap_diff(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    *,
    n_boot: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
    scale: float = 100.0,
) -> DiffResult:
    """Paired bootstrap of ``mean(b - a)`` over the same items (e.g. a=SFT, b=SFT+RL)."""
    xa, xb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if xa.shape != xb.shape:
        raise ValueError("paired comparison requires the same items in the same order")
    d = xb - xa
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boots = d[idx].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(boots, [alpha, 1.0 - alpha])
    # Two-sided p-value: how often the bootstrap distribution, centered at 0, is as extreme as observed.
    centered = boots - d.mean()
    p = float(np.mean(np.abs(centered) >= abs(d.mean())))
    return DiffResult(
        diff=float(scale * d.mean()),
        lo=float(scale * lo),
        hi=float(scale * hi),
        p_value=p,
        significant=bool(lo > 0 or hi < 0),
    )


@dataclass(frozen=True)
class NoiseFloor:
    n_runs: int
    mean_pairwise_abs_diff: float
    max_pairwise_abs_diff: float
    std: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def noise_floor(run_scores: Sequence[float]) -> NoiseFloor:
    """Noise floor from repeated evaluations of the *same* checkpoint with different seeds.

    The report's protocol: 10 repeats on D-Num gave a mean pairwise
    difference of 0.9 and a max of 2.1. Any change smaller than that is
    treated as noise.
    """
    s = [float(x) for x in run_scores]
    if len(s) < 2:
        raise ValueError("need at least two runs")
    diffs = [abs(x - y) for x, y in itertools.combinations(s, 2)]
    return NoiseFloor(len(s), float(np.mean(diffs)), float(np.max(diffs)), float(np.std(s, ddof=1)))


def is_significant_improvement(
    new: float,
    old: float,
    *,
    ci_halfwidth: float | None = None,
    noise: NoiseFloor | None = None,
    required_delta: float = 0.0,
) -> bool:
    """``new - old`` must reach ``required_delta`` *and* exceed the noise (CI half-width and noise-floor max).

    For gates expressed as "≥ base + 8", ``required_delta=8`` and the
    observed gain must also be outside the noise band. For a gate with no
    margin (``required_delta=0``), only the noise test applies.
    """
    gain = new - old
    band = max(ci_halfwidth or 0.0, noise.max_pairwise_abs_diff if noise else 0.0)
    return gain >= required_delta and gain > band


def wilson_ci(successes: float, n: int, level: float = 0.95) -> CI:
    """Wilson score interval for a proportion, ×100 (good for win rates near 50% and small n)."""
    if n <= 0:
        raise ValueError("n must be positive")
    from statistics import NormalDist

    z = NormalDist().inv_cdf(1 - (1 - level) / 2)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return CI(point=100 * p, lo=100 * (center - half), hi=100 * (center + half), level=level)


def combined_halfwidth(*halfwidths: float) -> float:
    """CI half-width of a difference of independent estimates: sqrt(sum(h²))."""
    return math.sqrt(sum(h * h for h in halfwidths))
