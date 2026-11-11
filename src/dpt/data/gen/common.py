"""Shared pieces of the generators: candidate records, the license tag, deterministic sampling, output.

A candidate is a :class:`dpt.data.schema.Problem` dict. ``candidate()`` builds it and validates it with
``Problem.from_dict``, so a generator fails at write time, not three stages later in the pipeline.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Iterable

from dpt.common.io import write_jsonl
from dpt.common.seeding import derive_seed
from dpt.data.schema import Problem

# License tag of everything we generate (templates, solvers, our own code). It is on the ingest allow-list
# (configs/data/pipeline.yaml). Model-written items also record the model in ``source.model``.
GENERATED_LICENSE = "internal-generated"

SPLITS = ("train", "heldout")


def candidate(task: str, question: str, answer: dict[str, Any], *, origin: str, split: str = "train",
              source: dict[str, Any] | None = None, **meta: Any) -> dict[str, Any]:
    """A validated candidate record. ``meta`` keys are stored as-is (``family``, ``topic``, ``generator`` …)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, not {split!r}")
    rec = {"task": task, "question": question, "answer": dict(answer),
           "source": {"license": GENERATED_LICENSE, "origin": origin, **(source or {})},
           "meta": {"split": split, **meta}}
    return Problem.from_dict(rec).to_dict()


def rng_for(seed: int, *names: object) -> random.Random:
    """An independent, reproducible RNG per (seed, names): adding a family never changes another's draws."""
    return random.Random(derive_seed(seed, *names))


def round_sig(x: float, digits: int) -> float:
    """``x`` rounded to ``digits`` significant figures (0 stays 0)."""
    if x == 0 or not math.isfinite(x):
        return x
    return round(x, digits - 1 - int(math.floor(math.log10(abs(x)))))


def fmt_num(x: float, digits: int = 3) -> str:
    """A number as a person writes it in a problem: no float noise, no exponent for everyday magnitudes."""
    x = round_sig(x, digits)
    if x == int(x) and abs(x) < 1e7:
        return str(int(x))
    if 1e-3 <= abs(x) < 1e7:
        return f"{x:.{digits}g}" if abs(x) >= 1 else f"{x:.{digits + 3}f}".rstrip("0").rstrip(".")
    mant, exp = f"{x:.{digits - 1}e}".split("e")
    return f"{mant} × 10^{int(exp)}"


def write_candidates(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    rows = list(rows)
    write_jsonl(path, rows)
    return len(rows)


# The moment-magnitude rule stated in every question that converts between M0 and Mw. It lives here, not in
# seismology.py, because the D-Num-hard families use it too, and importing it from seismology.py failed with a
# circular import whenever seismology.py was the first module imported (seismology → families → seismology_hard).
MW_RULE = "Use Mw = (2/3)(log10 M0 − 9.1) with M0 in N·m."


def import_private_module_if_present(module_name: str) -> bool:
    """Import a module that is kept out of git, such as the held-out eval families; False when it is absent.

    The held-out families and kinds behind D-Num, D-Num-hard and D-Code are not published (README, "Data &
    licensing"): the public repository runs without them, with an empty held-out split. Only the module's
    own absence counts as absent. An import error raised inside a private module that is present is
    re-raised, so a broken private module can never look like a missing one.
    """
    import importlib

    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as missing_module:
        if missing_module.name != module_name:
            raise
        return False
    return True
