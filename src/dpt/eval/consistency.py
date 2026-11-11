"""Layer-3 checks: is a correct answer evidence of real skill? (report §第三层：答对是不是"真的会")

Three diagnostics. None is a gate by itself; together they separate real
capability gains from guessing, format luck and memorization.

1. **Reasoning-answer consistency.** Take the last numeric value derived
   in the thinking and compare it with the submitted answer. If the answer
   is correct but the derivation ends somewhere else, the model guessed or
   copied a remembered number. This "correct but inconsistent" rate is
   tracked separately. It rose while the step-95 multi-answer hack was
   active.
2. **Perturbation gap.** 200 D-Num items get numerically perturbed twins
   with recomputed answers. A memorizing model drops on the twins. The
   check compares the final model's (original − perturbed) gap with
   base's gap. Report: 71.4 vs 68.9 (gap 2.5) against base gap 2.1, so no
   meaningful widening.
3. **pass@k floor.** After RL, pass@64 must not fall below base. Otherwise
   RL only sharpened the distribution and lost exploration (report:
   91.0 vs 90.2).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

_NUM_RE = re.compile(
    r"(?<![\w.])[-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"(?:\s*(?:[eE][-+−]?\d+|(?:×|\\times|x)\s*10\^\{?[-+−]?\d+\}?))?"
)
# Prefer numbers that follow an equals/approx sign: those are results, not inputs being restated.
_RESULT_RE = re.compile(r"(?:=|≈|\\approx|is about|gives|yields|so)\s*\$?\s*(" + _NUM_RE.pattern + r")")


def _to_float(tok: str) -> float | None:
    t = tok.replace("−", "-").replace(",", "").replace(" ", "")
    m = re.match(r"^([-+]?\d+(?:\.\d+)?)(?:(?:×|\\times|x)10\^\{?([-+]?\d+)\}?)$", t)
    try:
        if m:
            return float(m.group(1)) * 10 ** int(m.group(2))
        return float(t)
    except ValueError:
        return None


def last_derived_number(thinking: str) -> float | None:
    """Last number that appears as a *result* (after ``=``/``≈``) in the thinking, else the last number."""
    results = _RESULT_RE.findall(thinking)
    if results:
        return _to_float(results[-1])
    nums = _NUM_RE.findall(thinking)
    return _to_float(nums[-1]) if nums else None


def is_consistent(
    derived: float | None,
    submitted_si_or_raw: float | None,
    *,
    rel_tol: float = 1e-2,
    allow_power_of_ten: bool = True,
) -> bool:
    """Whether a derived value agrees with the submitted value.

    Unit conversions often differ by a power of ten (``km`` vs ``m``), so
    with ``allow_power_of_ten`` the comparison also accepts matches after
    scaling by 10^±3, 10^±6 or 10^±9.
    """
    if derived is None or submitted_si_or_raw is None:
        return False
    scales = [1.0]
    if allow_power_of_ten:
        scales += [10.0**e for e in (-9, -6, -3, 3, 6, 9)]
    for s in scales:
        a, b = derived * s, submitted_si_or_raw
        if abs(a - b) <= rel_tol * max(abs(b), 1e-12):
            return True
    return False


@dataclass(frozen=True)
class ConsistencyReport:
    n: int
    consistent_rate: float  # over all gradable samples
    correct_but_inconsistent_rate: float  # over correct samples: the "guessed / format-lucky" signal
    n_correct: int

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def consistency_report(
    samples: Iterable[Mapping[str, object]], *, rel_tol: float = 1e-2
) -> ConsistencyReport:
    """``samples``: dicts with ``thinking: str``, ``submitted: float | None``, ``correct: bool``."""
    n = n_cons = n_corr = n_corr_incons = 0
    for s in samples:
        n += 1
        submitted = s.get("submitted")
        cons = is_consistent(
            last_derived_number(str(s["thinking"])),
            None if submitted is None else float(submitted),  # type: ignore[arg-type]
            rel_tol=rel_tol,
        )
        n_cons += cons
        if s["correct"]:
            n_corr += 1
            n_corr_incons += not cons
    return ConsistencyReport(
        n=n,
        consistent_rate=n_cons / max(1, n),
        correct_but_inconsistent_rate=n_corr_incons / max(1, n_corr),
        n_correct=n_corr,
    )


@dataclass(frozen=True)
class PerturbationResult:
    original: float
    perturbed: float
    gap: float
    base_gap: float | None
    gap_widening: float | None
    flagged: bool

    def to_dict(self) -> dict[str, float | bool | None]:
        return asdict(self)


def perturbation_gap(
    original_per_item: Sequence[float],
    perturbed_per_item: Sequence[float],
    *,
    base_gap: float | None = None,
    max_widening: float = 2.0,
) -> PerturbationResult:
    """Score gap (×100) between original items and their perturbed twins (same order).

    ``flagged`` is set when the gap exceeds base's gap by more than
    ``max_widening`` points. Set ``max_widening`` to the D-Num CI half-width
    so that only a widening beyond noise counts.
    """
    o, p = np.asarray(original_per_item, float), np.asarray(perturbed_per_item, float)
    if o.shape != p.shape:
        raise ValueError("original and perturbed must align item-by-item")
    orig, pert = 100 * o.mean(), 100 * p.mean()
    gap = orig - pert
    widen = None if base_gap is None else gap - base_gap
    return PerturbationResult(
        float(orig),
        float(pert),
        float(gap),
        base_gap,
        None if widen is None else float(widen),
        bool(widen is not None and widen > max_widening),
    )


def pass_at_k_floor(
    model_curve: Mapping[int | str, float],
    base_curve: Mapping[int | str, float],
    k: int = 64,
    tolerance: float = 0.0,
) -> bool:
    """True if model pass@k ≥ base pass@k − tolerance (keys may be int or str)."""

    def get(c: Mapping[int | str, float]) -> float:
        return float(c[k] if k in c else c[str(k)])  # type: ignore[index]

    return get(model_curve) >= get(base_curve) - tolerance
