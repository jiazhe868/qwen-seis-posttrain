"""Programmatic solvers: the first of the three independent answer sources for T1 problems.

Report §RL 题库: T1 ground truth comes from "程序化求解器给出真值，带单位和容差比对". A
problem generated from a template stores::

    meta.generator = {
        "name": "travel_time",                          # key in SOLVERS
        "params": {"distance_km": 120, "velocity_km_s": 6},
        "scaling": {"distance_km": 1, "velocity_km_s": -1},   # optional power-law exponents
        "invariant": []                                  # optional params the answer must NOT depend on
    }

Each solver is a pure function ``params -> float`` in the problem's answer unit.
The optional ``scaling`` exponents state the physical relation (answer ∝
param^k). The perturbation check in ``answer_verify`` uses them to confirm the
answer "changes according to the physics" (report §第一层 item 4).

The registry ships with a few seismology examples that are used by the toy
data. Register project solvers with :func:`register_solver`.
"""

from __future__ import annotations

import math
from typing import Callable

Solver = Callable[[dict[str, float]], float]
SOLVERS: dict[str, Solver] = {}


def register_solver(name: str) -> Callable[[Solver], Solver]:
    def deco(fn: Solver) -> Solver:
        if name in SOLVERS and SOLVERS[name] is not fn:
            raise ValueError(f"solver {name!r} already registered")
        SOLVERS[name] = fn
        return fn

    return deco


def solve(generator: dict) -> float:
    """Evaluate ``generator['name']`` on ``generator['params']``; raises ``KeyError`` for unknown solvers.

    The T1 families (``dpt.data.gen.families``) register their solvers when that module is imported. A
    name not registered yet triggers that import, so a stage that never imports the generators still finds
    them: bank verification did not, and rejected every T1 template with ``solver_error:KeyError``
    (2026-10-25).
    """
    name = generator["name"]
    if name not in SOLVERS:
        import dpt.data.gen.families  # noqa: F401  (registers every family's solver on import)
    return float(SOLVERS[name](dict(generator["params"])))


# --- examples (seismology) -------------------------------------------------------------------
@register_solver("travel_time")
def _travel_time(p: dict[str, float]) -> float:
    """t [s] = distance [km] / velocity [km/s]."""
    return p["distance_km"] / p["velocity_km_s"]


@register_solver("wavelength")
def _wavelength(p: dict[str, float]) -> float:
    """λ [m] = v [m/s] / f [Hz]."""
    return p["velocity_m_s"] / p["frequency_hz"]


@register_solver("hypocentral_distance")
def _hypocentral_distance(p: dict[str, float]) -> float:
    """R [km] = sqrt(Δ² + h²) for epicentral distance Δ and depth h (flat-earth approximation)."""
    return math.hypot(p["epicentral_km"], p["depth_km"])


@register_solver("sp_distance")
def _sp_distance(p: dict[str, float]) -> float:
    """Distance [km] from S−P time: d = Δt · vp · vs / (vp − vs)."""
    vp, vs = p["vp_km_s"], p["vs_km_s"]
    return p["ts_minus_tp_s"] * vp * vs / (vp - vs)


@register_solver("moment_magnitude")
def _moment_magnitude(p: dict[str, float]) -> float:
    """Mw = (2/3)(log10 M0 − 9.1), M0 in N·m (IASPEI standard form)."""
    return (2.0 / 3.0) * (math.log10(p["m0_nm"]) - 9.1)


@register_solver("seismic_moment")
def _seismic_moment(p: dict[str, float]) -> float:
    """M0 [N·m] = μ · A · D with rigidity μ [Pa], fault area A [m²], average slip D [m]."""
    return p["rigidity_pa"] * p["area_m2"] * p["slip_m"]
