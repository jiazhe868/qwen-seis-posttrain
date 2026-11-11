"""T1 numeric problem families (seismology): parameters, solver, question templates.

A family is one kind of problem. Each instance draws parameters, rounds them to the figures shown in the
question, and computes the answer from the *rounded* values, so the stated numbers and the label agree::

    f = FAMILIES["sp_distance"]
    params = f.sample(rng)              # {"ts_minus_tp_s": 12.4, "vp_km_s": 6.1, "vs_km_s": 3.5}
    f.solve(params)                     # answer in f.unit
    f.render(params, template=0)        # question text

The solver is registered in ``dpt.data.solvers.SOLVERS`` under the family name, so the verify stage's
perturbation check (``dpt.data.answer_verify``) can re-evaluate it from ``meta.generator``.

Multi-step families (``steps >= 2``) chain formulas: the question gives the inputs of the first step and
asks for the output of the last, and no intermediate value is stated.

``split="heldout"`` families exist only for D-Num. The eval's claim is transfer to problem *kinds* the
policy never trained on, so a held-out family must not be a rewording of a training family.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Callable

from dpt.data.gen.common import fmt_num, round_sig
from dpt.data.solvers import SOLVERS, register_solver

ParamSolver = Callable[[dict[str, float]], float]


@dataclass(frozen=True)
class Param:
    """One input. Drawn uniformly in [lo, hi] (log-uniformly if ``log``), then rounded to ``digits`` figures."""

    name: str
    lo: float
    hi: float
    log: bool = False
    digits: int = 3
    integer: bool = False
    decimals: int | None = None     # fixed decimals instead of significant figures (angles, magnitudes)

    def draw(self, rng: random.Random) -> float:
        if self.log:
            x = math.exp(rng.uniform(math.log(self.lo), math.log(self.hi)))
        else:
            x = rng.uniform(self.lo, self.hi)
        return self.round(x)

    def round(self, x: float) -> float:
        if self.integer:
            return float(round(x))
        if self.decimals is not None:
            return round(x, self.decimals)
        return round_sig(x, self.digits)

    def show(self, x: float) -> str:
        if self.integer:
            return str(int(x))
        if self.decimals is not None:
            return f"{x:.{self.decimals}f}"
        return fmt_num(x, self.digits)


@dataclass(frozen=True)
class Family:
    name: str
    topic: str
    solver: ParamSolver
    params: tuple[Param, ...]
    unit: str                                   # answer unit; "" = dimensionless
    templates: tuple[str, ...]                  # str.format over the shown parameter values
    steps: int = 1
    rel_tol: float = 0.01
    abs_tol: float = 0.0                        # for answers that can be near zero (magnitudes, angles, R)
    split: str = "train"                        # "train" | "heldout"
    scaling: dict[str, float] | None = None     # answer ∝ param^k (perturbation check)
    invariant: tuple[str, ...] = ()             # params the answer must not depend on
    valid: Callable[[dict[str, float]], bool] | None = None   # rejects unphysical draws
    # Draws shown values that must be mutually consistent (e.g. arrival times from one origin): returns
    # every parameter, already rounded. Replaces the independent per-Param draws.
    make: Callable[[random.Random], dict[str, float]] | None = None
    notes: str = ""
    extra: dict[str, str] = field(default_factory=dict)
    bank: str = "v1"                            # v1: the bank of 2026-10; hard: dpt.data.gen.seismology_hard

    def sample(self, rng: random.Random, *, max_tries: int = 200) -> dict[str, float]:
        for _ in range(max_tries):
            p = self.make(rng) if self.make else {q.name: q.draw(rng) for q in self.params}
            if self.is_valid(p) and _finite(self.solve(p)):
                return p
        raise RuntimeError(f"family {self.name}: no valid parameter draw in {max_tries} tries")

    def is_valid(self, params: dict[str, float]) -> bool:
        """``valid`` with the solver's domain errors (a zero denominator, asin > 1) counted as invalid."""
        if self.valid is None:
            return True
        try:
            return bool(self.valid(params))
        except (ValueError, ZeroDivisionError, OverflowError):
            return False

    def solve(self, params: dict[str, float]) -> float:
        try:
            return float(self.solver(dict(params)))
        except (ValueError, ZeroDivisionError, OverflowError):
            return math.nan

    def render(self, params: dict[str, float], template: int) -> str:
        shown = {q.name: q.show(params[q.name]) for q in self.params}
        return _an(self.templates[template % len(self.templates)].format(**shown))

    def answer(self, params: dict[str, float]) -> dict:
        return {"value": self.solve(params), "unit": self.unit, "rel_tol": self.rel_tol, "abs_tol": self.abs_tol}

    def generator_meta(self, params: dict[str, float]) -> dict:
        g = {"name": self.name, "params": dict(params)}
        if self.scaling:
            g["scaling"] = dict(self.scaling)
        if self.invariant:
            g["invariant"] = list(self.invariant)
        return g


_AN = re.compile(r"\b([Aa]) (?=(?:8|11(?![0-9])|18(?![0-9])))")


def _an(text: str) -> str:
    """'a 8 km layer' → 'an 8 km layer' (also 11 and 18): templates cannot know the number in advance."""
    return _AN.sub(r"\1n ", text)


def _finite(x: float) -> bool:
    return isinstance(x, float) and math.isfinite(x)


FAMILIES: dict[str, Family] = {}


def register(f: Family) -> Family:
    if f.name in FAMILIES:
        raise ValueError(f"family {f.name!r} already registered")
    if f.split not in ("train", "heldout"):
        raise ValueError(f"{f.name}: split must be train or heldout")
    names = {q.name for q in f.params}
    for t in f.templates:
        missing = {k for k in _fields(t)} - names
        if missing:
            raise ValueError(f"{f.name}: template uses unknown parameters {sorted(missing)}")
    if f.name in SOLVERS:
        if SOLVERS[f.name] is not f.solver:
            raise ValueError(f"solver {f.name!r} already registered with another function")
    else:
        register_solver(f.name)(f.solver)
    FAMILIES[f.name] = f
    return f


def _fields(template: str) -> list[str]:
    import string

    return [name for _, name, _, _ in string.Formatter().parse(template) if name]


def by_split(split: str, bank: str = "v1") -> list[Family]:
    return [f for f in FAMILIES.values() if f.split == split and f.bank == bank]


# The seismology families register themselves on import (kept in their own modules for length).
from dpt.data.gen import seismology  # noqa: E402,F401
from dpt.data.gen import seismology_hard  # noqa: E402,F401
from dpt.data.gen import seismology_long  # noqa: E402,F401


def heldout_families_are_present() -> bool:
    """True where the private modules with the held-out D-Num and D-Num-hard families are present.

    Those families are not published (README, "Data & licensing"). Each public module imports its private
    module as its own last statement, because the private module uses its helpers; importing them here
    instead broke whenever seismology_hard was imported before this module. Asked as a function, after
    the imports, so that the answer never depends on which module happened to be imported first.
    """
    return seismology.heldout_dnum_families_present and seismology_hard.heldout_dnum_hard_families_present
