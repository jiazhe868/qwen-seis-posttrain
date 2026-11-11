"""T2 code tasks for training and the D-Code eval set (README §3.1). Scope: seismology.

    python -m dpt.data.gen.t2 build --out-dir $DATA/gen --eval-out-dir $SECURE --n-train 3000 --n-heldout 150 \\
        [--seed 1] [--check] [--sandbox-sif $DPT_SANDBOX_SIF]
    python -m dpt.data.gen.t2 kinds                    # which kinds and families are training / held out

Two sources of tasks:

* **Physics functions** from the T1 families (``dpt.data.gen.families``): one worked problem from the family,
  "write ``f(args)`` that does this calculation for any inputs". Variants change the template, the argument
  order and, explicitly stated, the unit of one input or of the result. Expected values come from the
  family's solver.
* **Data-processing kinds** (``dpt.data.gen.t2_tasks``): seismic array and catalog processing with a
  reference implementation, a fully specified spec and input generators.

For every task the generator runs the reference on 1–2 small example inputs (shown in the prompt, and the
visible tests) and on ~6 hidden inputs, and writes the hidden tests as a file
``<hidden_tests_id>.py`` (module-level ``test_*`` functions, the solution imported as ``solution``,
``math`` only, inputs as literals): the format of ``dpt.rewards.code_sandbox.HiddenTestStore``.

Checks (``--check``, in the sandbox): the reference passes its visible and hidden tests, and output
mutants (zero, ×1.1, ×1000, negated, off by one, reversed, last element dropped, None) that change at least
one expected value fail the hidden tests; a task needs two such mutants. Physics-function tasks have no
reference source (the solver lives in this repo, not in the sandbox), so they are checked with a lookup
table of the expected values: that checks the test file, the expected values are the solver's.

Outputs: ``<out-dir>/candidates_t2.jsonl`` (pipeline candidates) and ``<out-dir>/hidden_tests/``; D-Code
(held-out kinds and families only) as ``<eval-out-dir>/d_code.jsonl`` with a canary and a manifest, and
its hidden tests in ``<eval-out-dir>/hidden_tests/`` (what ``scripts/slurm/eval.slurm`` serves). The two
directories must not overlap, and an existing D-Code file is never overwritten without ``--force``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import types
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from dpt.common.io import write_json, write_jsonl
from dpt.common.logging import get_logger
from dpt.data.gen.common import candidate, rng_for
from dpt.data.gen.t2_tasks import KINDS, Kind, Variant, kinds_by_split, pick

log = get_logger(__name__)

ORIGIN = "dpt.data.gen.t2"
NOTE = "You may use the Python standard library and numpy."

COMPARE_SRC = '''
def _same(got, exp, rel, tol):
    if exp is None:
        return got is None
    if isinstance(exp, bool):
        if got is None or isinstance(got, (str, bytes, list, tuple, dict)):
            return False
        return bool(got) == exp
    if isinstance(exp, (list, tuple)):
        if got is None or isinstance(got, (str, bytes, dict)):
            return False
        try:
            got = list(got)
        except TypeError:
            return False
        return len(got) == len(exp) and all(_same(g, e, rel, tol) for g, e in zip(got, exp))
    if isinstance(exp, str):
        return got == exp
    if got is None or isinstance(got, (str, bytes, list, tuple, dict)):
        return False
    try:
        g = float(got)
    except (TypeError, ValueError):
        return False
    if isinstance(exp, int):
        return g == exp
    return math.isfinite(g) and math.isclose(g, exp, rel_tol=rel, abs_tol=tol)
'''

TEST_HEADER = "import math\n\nimport solution\n\n" + COMPARE_SRC

# Output mutants: each must change some expected value (else it is skipped as equivalent) and then fail.
MUTATIONS: dict[str, str] = {
    "zero": '''
def _dpt_mut(x):
    if isinstance(x, bool):
        return not x
    if isinstance(x, (int, float)):
        return 0 * x
    if isinstance(x, (list, tuple)):
        return type(x)(_dpt_mut(v) for v in x)
    return x
''',
    "scale_1p1": '''
def _dpt_mut(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return x * 1.1
    if isinstance(x, (list, tuple)):
        return type(x)(_dpt_mut(v) for v in x)
    return x
''',
    "scale_1000": '''
def _dpt_mut(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return x * 1000
    if isinstance(x, (list, tuple)):
        return type(x)(_dpt_mut(v) for v in x)
    return x
''',
    "negate": '''
def _dpt_mut(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return -x
    if isinstance(x, (list, tuple)):
        return type(x)(_dpt_mut(v) for v in x)
    return x
''',
    "off_by_one": '''
def _dpt_mut(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, int):
        return x + 1
    if isinstance(x, (list, tuple)):
        return type(x)(_dpt_mut(v) for v in x)
    return x
''',
    "reverse": '''
def _dpt_mut(x):
    if isinstance(x, (list, tuple)):
        return type(x)(reversed(x))
    return x
''',
    "drop_last": '''
def _dpt_mut(x):
    if isinstance(x, (list, tuple)) and x:
        return type(x)(list(x)[:-1])
    return x
''',
    "none": '''
def _dpt_mut(x):
    return None
''',
}
MIN_MUTANTS = 2


# --------------------------------------------------------------------------- tasks


@dataclass
class Case:
    args: tuple
    expected: Any
    tol: float


@dataclass
class Task:
    uid: str                         # = hidden_tests_id
    split: str
    source: str                      # library kind name, or "family:<name>"
    kind: str
    topic: str
    family: str | None
    entry: str
    question: str
    visible_tests: str
    hidden_tests: str
    reference: str
    reference_is_table: bool
    key: str
    rel: float
    hidden_cases: list[Case] = field(default_factory=list)
    example_cases: list[Case] = field(default_factory=list)


def lit(x: Any) -> str:
    """A Python literal for ``x`` (floats via repr, so values round-trip exactly)."""
    if isinstance(x, tuple):
        return "(" + ", ".join(lit(v) for v in x) + ("," if len(x) == 1 else "") + ")"
    if isinstance(x, list):
        return "[" + ", ".join(lit(v) for v in x) + "]"
    if isinstance(x, float):
        if not math.isfinite(x):
            raise ValueError("non-finite literal")
        return repr(x)
    if x is None or isinstance(x, (bool, int, str)):
        return repr(x)
    raise TypeError(f"cannot write a literal of type {type(x).__name__}")


def show(x: Any) -> str:
    """A readable value for the prompt's examples (6 significant figures)."""
    if isinstance(x, tuple):
        return "(" + ", ".join(show(v) for v in x) + ")"
    if isinstance(x, list):
        return "[" + ", ".join(show(v) for v in x) + "]"
    if isinstance(x, float):
        return f"{x:.6g}"
    return repr(x)


def _plain(x: Any) -> Any:
    """Expected values as plain Python (numpy scalars and arrays never reach the test literals)."""
    if isinstance(x, bool) or x is None or isinstance(x, (int, str)):
        return x
    if isinstance(x, float):
        return float(x)
    if isinstance(x, tuple):
        return tuple(_plain(v) for v in x)
    if isinstance(x, list):
        return [_plain(v) for v in x]
    if hasattr(x, "item"):
        return _plain(x.item())
    raise TypeError(f"unsupported expected value {type(x).__name__}")


def _finite(x: Any) -> bool:
    if isinstance(x, float):
        return math.isfinite(x)
    if isinstance(x, (list, tuple)):
        return all(_finite(v) for v in x)
    return True


def _maxabs(x: Any) -> float:
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return 0.0
    if isinstance(x, (int, float)):
        return abs(float(x))
    return max((_maxabs(v) for v in x), default=0.0)


def _exec_reference(src: str, entry: str) -> Callable[..., Any]:
    ns: dict[str, Any] = {"__name__": "t2_reference"}
    exec(compile(src, "<reference>", "exec"), ns)  # our own reference code, never model output
    return ns[entry]


def _cases(v: Variant, fn: Callable[..., Any], inputs: Sequence[tuple]) -> list[Case]:
    out = []
    for args in inputs:
        exp = _plain(fn(*copy.deepcopy(args)))
        if not _finite(exp):
            raise ValueError("non-finite expected value")
        out.append(Case(tuple(args), exp, max(v.abs_tol, v.abs_scale * _maxabs(exp))))
    return out


def render_tests(entry: str, cases: Sequence[Case], rel: float) -> str:
    parts = [TEST_HEADER]
    for k, c in enumerate(cases, 1):
        call = f"solution.{entry}({', '.join(lit(a) for a in c.args)})"
        parts.append(f"\n\ndef test_{k:02d}():\n    assert _same({call}, {lit(c.expected)}, {rel!r}, {c.tol!r})\n")
    return "".join(parts)


def table_reference(entry: str, cases: Sequence[Case]) -> str:
    rows = "".join(f"    {lit(c.args)}: {lit(c.expected)},\n" for c in cases)
    return f"_TABLE = {{\n{rows}}}\n\n\ndef {entry}(*args):\n    return _TABLE[tuple(args)]\n"


def render_question(v: Variant, examples: Sequence[Case]) -> str:
    lines = [v.spec.strip(), "", NOTE, "", "Example:" if len(examples) == 1 else "Examples:"]
    for c in examples:
        lines.append(f"    {v.entry}({', '.join(lit(a) for a in c.args)}) returns {show(c.expected)}")
    return "\n".join(lines)


def make_task(v: Variant, *, split: str, source: str, kind: str, topic: str) -> Task:
    fn = v.fn if v.reference is None else _exec_reference(v.reference, v.entry)
    ex = _cases(v, fn, v.examples)
    hid = _cases(v, fn, v.hidden)
    question = render_question(v, ex)
    visible, hidden = render_tests(v.entry, ex, v.rel_tol), render_tests(v.entry, hid, v.rel_tol)
    reference = v.reference if v.reference is not None else table_reference(v.entry, ex + hid)
    prefix = "t2" if split == "train" else "dcode"
    uid = f"{prefix}-{hashlib.sha256((question + hidden).encode()).hexdigest()[:16]}"
    return Task(uid, split, source, kind, topic, v.family, v.entry, question, visible, hidden, reference,
                v.reference is None, v.key, v.rel_tol, hid, ex)


# --------------------------------------------------------------------------- physics functions from families

# Input-unit suffixes of family parameter names: suffix → (label, [(alternative suffix, label, factor)]); the
# alternative value is value·factor. Longest suffix first.
_IN_UNITS: list[tuple[str, str, list[tuple[str, str, float]]]] = sorted([
    ("_km_s", "km/s", [("_m_s", "m/s", 1000.0)]),
    ("_m_s", "m/s", [("_km_s", "km/s", 1e-3)]),
    ("_cm_s", "cm/s", [("_m_s", "m/s", 0.01)]),
    ("_m_s2", "m/s²", [("_cm_s2", "cm/s²", 100.0)]),
    ("_kg_m3", "kg/m³", [("_g_cm3", "g/cm³", 1e-3)]),
    ("_g_cm3", "g/cm³", [("_kg_m3", "kg/m³", 1000.0)]),
    ("_km2", "km²", [("_m2", "m²", 1e6)]),
    ("_m2", "m²", [("_km2", "km²", 1e-6)]),
    ("_gpa", "GPa", [("_pa", "Pa", 1e9)]),
    ("_mpa", "MPa", [("_pa", "Pa", 1e6), ("_bar", "bar", 10.0)]),
    ("_pa", "Pa", [("_mpa", "MPa", 1e-6)]),
    ("_nm", "N·m", [("_dyne_cm", "dyne·cm", 1e7)]),
    ("_km", "km", [("_m", "m", 1000.0)]),
    ("_m", "m", [("_km", "km", 1e-3)]),
    ("_deg", "degrees", [("_rad", "radians", math.pi / 180.0)]),
    ("_hz", "Hz", []),
    ("_days", "days", [("_hours", "hours", 24.0)]),
    ("_mm_yr", "mm/yr", [("_m_yr", "m/yr", 1e-3)]),
    ("_yr", "years", []),
    ("_s", "s", [("_ms", "ms", 1000.0)]),
], key=lambda u: -len(u[0]))

# Result units: unit → [(alternative, factor)].
_OUT_UNITS: dict[str, list[tuple[str, float]]] = {
    "s": [("ms", 1000.0)], "km": [("m", 1000.0)], "m": [("cm", 100.0), ("km", 1e-3)],
    "km/s": [("m/s", 1000.0)], "m/s": [("cm/s", 100.0)], "cm/s": [("m/s", 0.01)],
    "N·m": [("dyne·cm", 1e7)], "MPa": [("Pa", 1e6), ("bar", 10.0)], "Pa": [("MPa", 1e-6)],
    "GPa": [("Pa", 1e9)], "deg": [("rad", math.pi / 180.0)], "°": [("rad", math.pi / 180.0)],
    "degrees": [("radians", math.pi / 180.0)], "days": [("hours", 24.0)], "J": [("erg", 1e7)],
    "km²": [("m²", 1e6)], "m²": [("km²", 1e-6)],
}

_SOLVER_STYLES = (
    "The problem below is one instance of a calculation. Write a Python function `{sig}` that carries out the "
    "same calculation for any physically valid inputs and returns the quantity the problem asks for {ret}, as "
    "a float.\n\nProblem: {statement}\n\nArguments:\n{args}",
    "Turn this worked problem into a reusable function.\n\nProblem: {statement}\n\nImplement `{sig}`. It must "
    "return the quantity the problem asks for {ret}, as a float, for any physically valid values of its "
    "arguments:\n{args}",
)
_ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth")


def _in_unit(name: str) -> tuple[str, str, list[tuple[str, str, float]]] | None:
    for suffix, label, alts in _IN_UNITS:
        if name.endswith(suffix) and len(name) > len(suffix):
            return suffix, label, alts
    return None


def _template_fields(template: str) -> list[str]:
    import string

    return [f for _, f, _, _ in string.Formatter().parse(template) if f]


def family_variant(fam: Any, rng) -> Variant:
    """One "implement this calculation" variant of a T1 family (see module docstring)."""
    names = [q.name for q in fam.params]
    usable = [k for k, t in enumerate(fam.templates) if set(names) <= set(_template_fields(t))]
    if not usable:
        raise ValueError(f"{fam.name}: no template mentions every parameter")
    k = pick(rng, usable)
    order = list(names)
    reverse = len(order) > 1 and rng.random() < 0.4
    if reverse:
        order.reverse()
    integer = {q.name for q in fam.params if q.integer}
    conv: dict[str, tuple[str, str, str, float]] = {}            # param → (arg name, label, alt label, factor)
    convertible = [n for n in names if n not in integer and (_in_unit(n) or (None, None, []))[2]]
    if convertible and rng.random() < 0.35:
        n = pick(rng, convertible)
        suffix, label, alts = _in_unit(n)
        alt_suffix, alt_label, factor = pick(rng, alts)
        new = n[: -len(suffix)] + alt_suffix
        if new not in names:
            conv[n] = (new, label, alt_label, factor)
    out_alts = _OUT_UNITS.get(fam.unit, [])
    out_unit, out_factor = fam.unit, 1.0
    if out_alts and rng.random() < 0.3:
        out_unit, out_factor = pick(rng, out_alts)
    arg_of = {n: conv[n][0] if n in conv else n for n in names}
    factor_of = {n: conv[n][3] if n in conv else 1.0 for n in names}
    entry = pick(rng, (fam.name, f"compute_{fam.name}", f"calc_{fam.name}"))

    def to_args(params: dict[str, float]) -> tuple:
        out = []
        for n in order:
            x = params[n]
            if n in integer:
                out.append(int(x))
            else:
                out.append(float(f"{x * factor_of[n]:.12g}"))
        return tuple(out)

    def fn(*args: float) -> float:
        params = {n: (args[order.index(n)] / factor_of[n]) for n in names}
        return out_factor * fam.solve(params)

    ex_params = fam.sample(rng)
    statement = fam.render(ex_params, k).strip()
    template = fam.templates[k]
    shown = {q.name: q.show(ex_params[q.name]) for q in fam.params}
    dup = Counter(shown.values())
    first_pos = sorted(names, key=lambda n: template.index("{" + n + "}"))
    arg_lines = []
    for n in order:
        u = _in_unit(n)
        given = f"{shown[n]} {u[1]}" if u else shown[n]
        line = f"- `{arg_of[n]}`: the value given as {given} in the problem"
        if dup[shown[n]] > 1:
            line += f" (the {_ORDINALS[min(first_pos.index(n), 9)]} quantity stated)"
        if n in conv:
            line += f", but passed in {conv[n][2]}"
        if n in integer:
            line += " (an int)"
        arg_lines.append(line)
    if not fam.unit:
        ret = "(a dimensionless number)"
    elif out_unit != fam.unit:
        ret = f"in {out_unit} (not {fam.unit})"
    else:
        ret = f"in {fam.unit}"
    sig = f"{entry}({', '.join(arg_of[n] for n in order)})"
    spec = pick(rng, _SOLVER_STYLES).format(sig=sig, ret=ret, statement=statement, args="\n".join(arg_lines))
    hidden: list[tuple] = []
    seen = {to_args(ex_params)}
    for _ in range(40):
        if len(hidden) >= 6:
            break
        a = to_args(fam.sample(rng))
        if a not in seen:
            seen.add(a)
            hidden.append(a)
    rel = max(1e-6, fam.rel_tol / 5.0) if fam.rel_tol > 0 else 1e-6
    key = f"family:{fam.name}|t{k}|{'rev' if reverse else 'fwd'}|{sorted(conv)}|{out_unit}"
    return Variant(entry, spec, None, [to_args(ex_params)], hidden, key, rel_tol=rel, abs_scale=0.0,
                   abs_tol=max(1e-12, getattr(fam, "abs_tol", 0.0) / 5.0 * out_factor), fn=fn, family=fam.name)


# --------------------------------------------------------------------------- generation


@dataclass(frozen=True)
class Source:
    name: str
    kind: str
    topic: str
    split: str
    family: str | None
    build: Callable[[Any], Variant]


def library_sources(split: str, kinds: Sequence[Kind] | None = None) -> list[Source]:
    ks = kinds if kinds is not None else kinds_by_split(split)
    return [Source(k.name, k.name, k.topic, split, None, k.build) for k in sorted(ks, key=lambda k: k.name)]


def family_sources(split: str, families: Sequence[Any]) -> list[Source]:
    out = []
    for f in sorted(families, key=lambda f: f.name):
        if f.split != split:
            continue
        out.append(Source(f"family:{f.name}", "physics_function", f.topic, split, f.name,
                          (lambda fam: (lambda rng: family_variant(fam, rng)))(f)))
    return out


@dataclass
class Generated:
    tasks: list[Task]
    failures: Counter
    exhausted: list[str]


def generate(sources: Sequence[Source], n: int, *, seed: int, cap_per_key: int = 4, tries: int = 30,
             seen_questions: set[str] | None = None, per_key: Counter | None = None) -> Generated:
    """Up to ``n`` tasks, round-robin over ``sources``, at most ``cap_per_key`` per semantic variant.

    ``seen_questions`` and ``per_key`` may be shared between calls, so a top-up never repeats a question or
    exceeds the cap.
    """
    rngs = {s.name: rng_for(seed, "t2", s.split, s.name) for s in sources}
    seen = seen_questions if seen_questions is not None else set()
    per_key = per_key if per_key is not None else Counter()
    failures: Counter = Counter()
    dead: set[str] = set()
    out: list[Task] = []
    while len(out) < n and len(dead) < len(sources):
        for s in sources:
            if s.name in dead or len(out) >= n:
                continue
            for _ in range(tries):
                try:
                    v = s.build(rngs[s.name])
                    if per_key[v.key] >= cap_per_key:
                        continue
                    t = make_task(v, split=s.split, source=s.name, kind=s.kind, topic=s.topic)
                except Exception as e:  # an unlucky draw (empty selection, degenerate fit): try another one
                    failures[f"{s.name}:{type(e).__name__}"] += 1
                    continue
                if len(effective_mutants(t)) < MIN_MUTANTS:   # hidden tests too weak to tell wrong code apart
                    failures[f"{s.name}:weak_tests"] += 1
                    continue
                if t.question in seen:
                    continue
                seen.add(t.question)
                per_key[v.key] += 1
                out.append(t)
                break
            else:
                dead.add(s.name)
    return Generated(out, failures, sorted(dead))


def build(n_train: int, n_heldout: int, *, seed: int = 1, families: Sequence[Any] | None = None,
          kinds: Sequence[Kind] | None = None, solver_share: float = 0.4, heldout_solver_share: float = 0.35,
          cap_per_key: int = 4, heldout_cap_per_key: int = 2) -> tuple[Generated, Generated]:
    """(training tasks, D-Code tasks). Held-out tasks come only from held-out kinds and families.

    ``solver_share`` of the tasks are physics functions; if the data-processing kinds run out of distinct
    variants, the families fill the rest. D-Code uses a lower cap per variant, so its items are not
    near-copies of each other.
    """
    fams = list(families) if families is not None else _registered_families()
    ks = list(kinds) if kinds is not None else list(KINDS.values())
    seen: set[str] = set()
    result = []
    for split, n, share, cap in (("train", n_train, solver_share, cap_per_key),
                                 ("heldout", n_heldout, heldout_solver_share, heldout_cap_per_key)):
        lib = library_sources(split, [k for k in ks if k.split == split])
        sol = family_sources(split, fams)
        per_key: Counter = Counter()
        n_sol = round(n * share) if sol else 0
        a = generate(sol, n_sol, seed=seed, cap_per_key=cap, seen_questions=seen, per_key=per_key)
        b = generate(lib, n - len(a.tasks), seed=seed, cap_per_key=cap, seen_questions=seen, per_key=per_key)
        if len(a.tasks) + len(b.tasks) < n and sol:           # library exhausted: top up from the families
            extra = generate(sol, n - len(a.tasks) - len(b.tasks), seed=seed + 7919, cap_per_key=cap,
                             seen_questions=seen, per_key=per_key)
            a.tasks.extend(extra.tasks)
            a.failures.update(extra.failures)
        exhausted = sorted(set(b.exhausted) | (set(a.exhausted) if len(a.tasks) + len(b.tasks) < n else set()))
        result.append(Generated(a.tasks + b.tasks, a.failures + b.failures, exhausted))
    return result[0], result[1]


def _registered_families() -> list[Any]:
    from dpt.data.gen import families as F

    return list(F.FAMILIES.values())


# --------------------------------------------------------------------------- checks


def inprocess_check(t: Task) -> list[str]:
    """The reference passes its own visible and hidden tests when run in this process (fast, serial)."""
    issues = []
    mod = types.ModuleType("solution")
    exec(compile(t.reference, "<reference>", "exec"), mod.__dict__)  # our own reference code
    saved = sys.modules.get("solution")
    sys.modules["solution"] = mod
    try:
        for label, src in (("visible", t.visible_tests), ("hidden", t.hidden_tests)):
            ns: dict[str, Any] = {"__name__": "t2_check"}
            exec(compile(src, "<tests>", "exec"), ns)
            names = sorted(k for k in ns if k.startswith("test_") and callable(ns[k]))
            if not names:
                issues.append(f"{label}:no_tests")
            for name in names:
                try:
                    ns[name]()
                except AssertionError:
                    issues.append(f"{label}:{name}_fails")
    finally:
        if saved is None:
            sys.modules.pop("solution", None)
        else:
            sys.modules["solution"] = saved
    return issues


_COMPILED: dict[str, Callable[..., Any]] = {}


def _compiled(name: str, src: str, fn: str) -> Callable[..., Any]:
    """The function ``fn`` defined by ``src`` (the same source the test and mutant files contain)."""
    if name not in _COMPILED:
        ns: dict[str, Any] = {"math": math}
        exec(src, ns)
        _COMPILED[name] = ns[fn]
    return _COMPILED[name]


def effective_mutants(t: Task) -> list[str]:
    """Mutations that change at least one hidden expected value beyond the test tolerance."""
    same = _compiled("_same", COMPARE_SRC, "_same")
    out = []
    for name, src in MUTATIONS.items():
        mut = _compiled(f"mut:{name}", src, "_dpt_mut")
        if any(not same(mut(copy.deepcopy(c.expected)), c.expected, t.rel, c.tol) for c in t.hidden_cases):
            out.append(name)
    return out


def mutant_source(t: Task, mutation: str) -> str:
    return (t.reference + "\n\n_dpt_ref = " + t.entry + "\n" + MUTATIONS[mutation] +
            f"\n\ndef {t.entry}(*args, **kwargs):\n    return _dpt_mut(_dpt_ref(*args, **kwargs))\n")


def check_task(t: Task, sandbox: Any, *, timeout_s: float = 30.0) -> list[str]:
    """Run the reference and the effective mutants in the sandbox; [] if the task is sound."""
    from dpt.rewards.code_sandbox import SandboxRequest

    def run(code: str, tests: str, kind: str):
        r = sandbox.run(SandboxRequest(code=code, tests=tests, timeout_s=timeout_s, kind=kind))
        if r.status == "infra_error":
            raise RuntimeError(f"sandbox unavailable: {r.detail}")          # fail the run, not the task
        return r

    issues = []
    for label, tests in (("visible", t.visible_tests), ("hidden", t.hidden_tests)):
        r = run(t.reference, tests, label)
        if r.status != "ok" or r.total == 0 or r.passed != r.total or r.tamper:
            issues.append(f"reference_{label}:{r.status}:{r.passed}/{r.total}")
    muts = effective_mutants(t)
    if len(muts) < MIN_MUTANTS:
        issues.append(f"too_few_mutants:{len(muts)}")
    for m in muts:
        r = run(mutant_source(t, m), t.hidden_tests, "hidden")
        if r.status == "ok" and r.total > 0 and r.passed == r.total:
            issues.append(f"mutant_survives:{m}")
    return issues


def check_all(tasks: Sequence[Task], sandbox: Any | None, *, workers: int = 16) -> dict[str, list[str]]:
    """uid → issues for every task with a problem (in-process for all; in the sandbox if one is given)."""
    bad: dict[str, list[str]] = {}
    for t in tasks:
        iss = inprocess_check(t)
        if iss:
            bad[t.uid] = iss
    if sandbox is not None:
        todo = [t for t in tasks if t.uid not in bad]
        with ThreadPoolExecutor(max(1, workers)) as ex:
            for t, iss in zip(todo, ex.map(lambda t: check_task(t, sandbox), todo)):
                if iss:
                    bad[t.uid] = iss
    return bad


def build_sandbox(sif: str | None) -> Any:
    from dpt.rewards.code_sandbox import LocalSubprocessSandbox

    python = os.environ.get("DPT_SANDBOX_PYTHON") or sys.executable
    return LocalSubprocessSandbox(python=python, container_image=sif or None, cpu_s=20, mem_mb=2048)


# --------------------------------------------------------------------------- outputs


def _record_meta(t: Task) -> dict[str, Any]:
    # Not "generator": the pipeline reserves meta.generator for a T1 solver spec ({name, params, ...}).
    return {"kind": t.kind, "family": t.family, "topic": t.topic, "variant_key": t.key}


def write_training(tasks: Sequence[Task], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    hdir = out / "hidden_tests"
    hdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in tasks:
        (hdir / f"{t.uid}.py").write_text(t.hidden_tests, encoding="utf-8")
        rows.append(candidate("t2_code", t.question,
                              {"entry_point": t.entry, "visible_tests": t.visible_tests, "hidden_tests_id": t.uid,
                               "language": "python"},
                              origin=ORIGIN, split="train", source={"kind": t.kind, "family": t.family},
                              **_record_meta(t)))
    path = out / "candidates_t2.jsonl"
    write_jsonl(path, rows)
    return path


def write_dcode(tasks: Sequence[Task], eval_dir: str | Path, *, force: bool = False,
                stage_stats: list[dict[str, Any]] | None = None) -> Path:
    from dpt.common.manifest import build_manifest, write_manifest
    from dpt.data.decontam import make_canary

    out = Path(eval_dir) / "d_code.jsonl"
    if out.exists() and not force:
        raise FileExistsError(f"{out} exists (it may be frozen); pass --force to replace it before the freeze")
    hdir = Path(eval_dir) / "hidden_tests"
    hdir.mkdir(parents=True, exist_ok=True)
    canary = make_canary("d_code/v1")
    recs = []
    for t in tasks:
        (hdir / f"{t.uid}.py").write_text(t.hidden_tests, encoding="utf-8")
        recs.append({"id": f"d_code-{t.uid.split('-', 1)[1]}", "question": t.question, "task": "t2_code",
                     "answer": {"entry_point": t.entry, "hidden_tests_id": t.uid, "language": "python",
                                "visible_tests": t.visible_tests},
                     "meta": {**_record_meta(t), "canary": canary}})
    write_jsonl(out, recs)
    write_manifest(build_manifest(out, "d_code", "v1", stage_stats=stage_stats or [],
                                  notes="generated by dpt.data.gen.t2 from held-out kinds and families; hidden "
                                        "tests in hidden_tests/ next to this file"))
    return out


def _overlap(a: Path, b: Path) -> bool:
    a, b = a.resolve(), b.resolve()
    return a == b or a in b.parents or b in a.parents


def _counts(tasks: Sequence[Task]) -> dict[str, int]:
    return dict(sorted(Counter(t.source for t in tasks).items()))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="T2 code tasks and D-Code")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out-dir", required=True, help="training candidates + hidden_tests/")
    b.add_argument("--eval-out-dir", required=True, help="secure eval dir: d_code.jsonl + hidden_tests/")
    b.add_argument("--n-train", type=int, default=3000)
    b.add_argument("--n-heldout", type=int, default=150)
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--solver-share", type=float, default=0.4, help="share of physics-function tasks (training)")
    b.add_argument("--cap-per-key", type=int, default=4, help="instances per semantic variant")
    b.add_argument("--check", action="store_true", help="run every reference and its mutants in the sandbox")
    b.add_argument("--sandbox-sif", default=os.environ.get("DPT_SANDBOX_SIF") or None)
    b.add_argument("--workers", type=int, default=16)
    b.add_argument("--force", action="store_true", help="replace an existing d_code.jsonl (before the freeze)")
    sub.add_parser("kinds")
    args = ap.parse_args(argv)

    if args.cmd == "kinds":
        fams = _registered_families()
        for split in ("train", "heldout"):
            print(f"{split}: kinds {[k.name for k in kinds_by_split(split)]}")
            print(f"{split}: families {sorted(f.name for f in fams if f.split == split)}")
        return 0

    out, ev = Path(args.out_dir), Path(args.eval_out_dir)
    if _overlap(out, ev):
        raise SystemExit("--out-dir and --eval-out-dir must not overlap: D-Code's hidden tests stay with the eval sets")
    if (ev / "d_code.jsonl").exists() and not args.force:
        raise SystemExit(f"{ev / 'd_code.jsonl'} exists (it may be frozen); pass --force before the freeze")
    train, held = build(args.n_train, args.n_heldout, seed=args.seed, solver_share=args.solver_share,
                        cap_per_key=args.cap_per_key)
    sandbox = build_sandbox(args.sandbox_sif) if args.check else None
    summary: dict[str, Any] = {"seed": args.seed, "checked_in_sandbox": bool(args.check)}
    for name, g in (("train", train), ("heldout", held)):
        bad = check_all(g.tasks, sandbox, workers=args.workers)
        for uid, iss in sorted(bad.items())[:20]:
            log.warning("%s task %s dropped: %s", name, uid, iss)
        g.tasks[:] = [t for t in g.tasks if t.uid not in bad]
        summary[name] = {"tasks": len(g.tasks), "dropped_by_check": len(bad),
                         "drop_reasons": dict(Counter(i.split(":")[0] for v in bad.values() for i in v)),
                         "per_source": _counts(g.tasks), "build_failures": dict(g.failures),
                         "exhausted_sources": g.exhausted}
    path = write_training(train.tasks, out)
    stats = [{"stage": "generated", "count": summary["heldout"]["tasks"] + summary["heldout"]["dropped_by_check"]},
             {"stage": "dropped:check", "count": summary["heldout"]["dropped_by_check"]},
             {"stage": "kept", "count": len(held.tasks)}]
    dpath = write_dcode(held.tasks, ev, force=args.force, stage_stats=stats)
    write_json(out / "t2_summary.json", summary)
    log.info("T2: %d training tasks → %s; D-Code: %d → %s", len(train.tasks), path, len(held.tasks), dpath)
    print(json.dumps({k: summary[k]["tasks"] for k in ("train", "heldout")}))
    if len(train.tasks) < args.n_train or len(held.tasks) < args.n_heldout:
        log.warning("fewer tasks than requested (train %d/%d, heldout %d/%d): see t2_summary.json",
                    len(train.tasks), args.n_train, len(held.tasks), args.n_heldout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
