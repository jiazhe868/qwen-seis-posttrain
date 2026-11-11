"""T1 numeric grader and T4 math grader. Binary rewards, fail closed.

T1 rule (report §各任务的 reward): 1 if the single final answer is within
tolerance of the solver's ground truth after unit conversion, else 0.

* **No partial credit.** Ablation A7: rewarding relative error made training
  reward rise faster but D-Num fall by 1.8, because the model learned to give
  "roughly right" answers. Closeness never earns reward in production. The
  A7 arm can be reproduced with ``partial_credit.enabled`` (off by default).
* **Several candidate final answers → 0** (step-95 hack), even if one is right.
* **Ranges / ± / inequalities → 0** (step-140 hack).
* **Missing unit when the reference has one → 0** by default
  (``require_unit``). "20" is not an answer to "how long, in what unit?".
* Unit conversion is done by the grader (:mod:`dpt.rewards.units`), so "2 km"
  and "2000 m" are equally correct. A wrong *dimension* is wrong.

The arithmetic evaluator below is a small recursive-descent parser. It never
calls ``eval``, because model output is untrusted input.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from dpt.common.types import T1_NUM, T4_GENERAL
from dpt.rewards.answer_extract import (
    Extraction,
    extract_final_answer,
    final_rhs,
    has_multiple_values,
    is_interval,
    normalize_candidate,
)
from dpt.rewards.base import RewardResult, fail_closed, zero
from dpt.rewards.units import DimensionError, UnitError, compatible, parse_unit, from_si, to_si


@dataclass
class NumericConfig:
    default_rel_tol: float = 1e-2
    default_abs_tol: float = 0.0
    require_unit: bool = True
    max_unmarked_chars: int = 64
    # Trailing labels accepted as "no unit" when the reference is dimensionless
    # (seismic magnitude scales, ratio words).
    dimensionless_labels: list[str] = field(
        default_factory=lambda: ["Mw", "ML", "Ms", "mb", "M_w", "M_L", "M_s", "m_b", "M", "times", "倍", "级"]
    )
    math_rel_tol: float = 1e-6
    math_abs_tol: float = 1e-9
    # ABLATION A7 ONLY. When enabled, a parsed, unit-compatible single answer outside tolerance
    # earns max(0, 1 - rel_err / max_rel_err) instead of 0. A7 showed this raises training reward
    # but lowers D-Num by 1.8 (the policy learns "close enough"), so production keeps it off.
    partial_credit: dict[str, Any] = field(default_factory=lambda: {"enabled": False, "max_rel_err": 0.10})

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "NumericConfig":
        d = dict(d or {})
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


# =============================================================================================
# Safe arithmetic evaluation
# =============================================================================================
class EvalError(ValueError):
    pass


def _match_brace(s: str, i: int) -> int:
    """Index just past the brace group starting at ``s[i] == '{'``."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return j + 1
    raise EvalError("unbalanced braces")


def _latex_to_plain(s: str) -> str:
    """Rewrite the LaTeX subset that appears in numeric answers into plain infix."""
    s = s.replace("\\left", "").replace("\\right", "").replace("\\displaystyle", "")
    s = s.replace("−", "-").replace("÷", "/").replace("\\div", "/")
    s = s.replace("\\times", "*").replace("\\cdot", "*").replace("×", "*").replace("·", "*")
    s = (
        s.replace("\\pi", " pi ")
        .replace("π", " pi ")
        .replace("\\,", "")
        .replace("\\!", "")
        .replace("\\ ", "")
    )
    # \frac{a}{b}, \dfrac, \tfrac (nested-safe)
    while True:
        m = re.search(r"\\[dt]?frac\s*\{", s)
        if not m:
            break
        a0 = m.end() - 1
        a1 = _match_brace(s, a0)
        b0 = a1
        while b0 < len(s) and s[b0].isspace():
            b0 += 1
        if b0 >= len(s) or s[b0] != "{":
            raise EvalError("\\frac without second argument")
        b1 = _match_brace(s, b0)
        s = s[: m.start()] + "((" + s[a0 + 1 : a1 - 1] + ")/(" + s[b0 + 1 : b1 - 1] + "))" + s[b1:]
    # \sqrt[n]{x} and \sqrt{x}
    while True:
        m = re.search(r"\\sqrt\s*(\[[^\]]*\])?\s*\{", s)
        if not m:
            break
        a0 = m.end() - 1
        a1 = _match_brace(s, a0)
        inner = s[a0 + 1 : a1 - 1]
        if m.group(1):
            n = m.group(1)[1:-1]
            rep = f"(({inner})^(1/({n})))"
        else:
            rep = f"sqrt({inner})"
        s = s[: m.start()] + rep + s[a1:]
    s = s.replace("{", "(").replace("}", ")")
    return s


_TOK = re.compile(
    r"\s*(?:(\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?)|(\*\*|[-+*/^()])|([A-Za-z_]+))"
)
_FUNCS = {"sqrt": math.sqrt, "ln": math.log, "log": math.log10, "exp": math.exp}
_CONSTS = {"pi": math.pi, "e": math.e}


class _Eval:
    def __init__(self, text: str):
        self.toks: list[tuple[str, str]] = []
        pos = 0
        text = text.strip()
        while pos < len(text):
            m = _TOK.match(text, pos)
            if not m or m.end() == pos:
                raise EvalError(f"bad token at {text[pos:pos + 10]!r}")
            pos = m.end()
            if m.group(1) is not None:
                self.toks.append(("num", m.group(1)))
            elif m.group(2) is not None:
                self.toks.append(("op", "^" if m.group(2) == "**" else m.group(2)))
            else:
                self.toks.append(("id", m.group(3)))
        self.i = 0

    def peek(self) -> tuple[str, str] | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self) -> tuple[str, str]:
        t = self.toks[self.i]
        self.i += 1
        return t

    def run(self) -> float:
        v = self.expr()
        if self.peek() is not None:
            raise EvalError(f"trailing token {self.peek()}")
        if not math.isfinite(v):
            raise EvalError("non-finite result")
        return v

    def expr(self) -> float:
        v = self.term()
        while self.peek() in (("op", "+"), ("op", "-")):
            op = self.take()[1]
            r = self.term()
            v = v + r if op == "+" else v - r
        return v

    def _starts_atom(self) -> bool:
        t = self.peek()
        return t is not None and (t[0] in ("num", "id") or t == ("op", "("))

    def term(self) -> float:
        v = self.unary()
        while True:
            t = self.peek()
            if t in (("op", "*"), ("op", "/")):
                op = self.take()[1]
                r = self.unary()
                if op == "/" and r == 0:
                    raise EvalError("division by zero")
                v = v * r if op == "*" else v / r
            elif self._starts_atom():  # implicit multiplication: 2pi, 3sqrt(2), 2(3)
                v = v * self.unary()
            else:
                return v

    def unary(self) -> float:
        t = self.peek()
        if t in (("op", "-"), ("op", "+")):
            self.take()
            v = self.unary()
            return -v if t[1] == "-" else v
        return self.power()

    def power(self) -> float:
        base = self.atom()
        if self.peek() == ("op", "^"):
            self.take()
            exp = self.unary()
            if abs(exp) > 1000:
                raise EvalError("exponent too large")
            try:
                return float(base**exp)
            except (OverflowError, ZeroDivisionError) as e:
                raise EvalError(str(e)) from e
        return base

    def atom(self) -> float:
        if self.peek() is None:
            raise EvalError("unexpected end")
        kind, val = self.take()
        if kind == "num":
            return float(val)
        if kind == "id":
            if val in _CONSTS:
                return _CONSTS[val]
            if val in _FUNCS:
                if self.peek() != ("op", "("):
                    raise EvalError(f"{val} needs parentheses")
                self.take()
                arg = self.expr()
                if self.peek() != ("op", ")"):
                    raise EvalError("missing )")
                self.take()
                try:
                    return _FUNCS[val](arg)
                except ValueError as e:
                    raise EvalError(str(e)) from e
            raise EvalError(f"unknown identifier {val!r}")
        if val == "(":
            v = self.expr()
            if self.peek() != ("op", ")"):
                raise EvalError("missing )")
            self.take()
            return v
        raise EvalError(f"unexpected {val!r}")


def safe_eval(expr: str) -> float:
    """Evaluate a numeric expression (LaTeX subset allowed) without ``eval``."""
    plain = _latex_to_plain(expr)
    plain = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", plain)  # thousands separators
    return _Eval(plain).run()


# =============================================================================================
# Quantity parsing
# =============================================================================================
@dataclass
class Quantity:
    value: float
    unit: str  # raw unit string as written ("" if none)


_SCI_TAIL = (
    r"(?:\s*[eE]\s*[-+]?\s*\d+|\s*(?:\*|×|\\times|\\cdot|·|x)\s*10\s*(?:\^|\*\*)\s*\{?\s*[-+−]?\s*\d+\s*\}?)?"
)
_NUM_HEAD = re.compile(
    r"^\s*(?P<num>[-+−]?\s*(?:"
    r"\\[dt]?frac\s*\{(?:[^{}]|\{[^{}]*\})*\}\s*\{(?:[^{}]|\{[^{}]*\})*\}"  # \frac{..}{..}
    r"|(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+)(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?"  # 12, 1,000, 3.5, 1/2
    r"|\.\d+"
    rf"){_SCI_TAIL})"
    r"(?P<rest>.*)$",
    re.S,
)


def _clean(c: str) -> str:
    t = normalize_candidate(c)
    t = t.replace("\\%", "%").replace("\\left", "").replace("\\right", "").replace("\\displaystyle", "")
    t = re.sub(r"^\$+|\$+$", "", t).strip()
    return t


def parse_quantity(candidate: str) -> Quantity:
    """Split ``candidate`` into a numeric value and a unit string.

    Raises :class:`EvalError` when no leading number can be parsed.
    """
    rhs = final_rhs(_clean(candidate))
    rhs = rhs.replace("\\approx", "").replace("~", " ").strip()
    m = _NUM_HEAD.match(rhs)
    if not m:
        raise EvalError(f"no leading number in {rhs!r}")
    num = m.group("num").replace("−", "-")
    num = re.sub(r"\s*x\s*10", "*10", num)
    value = safe_eval(num)
    unit = m.group("rest").strip().rstrip(".。,，;；").strip()
    unit = re.sub(r"^\\[,;! ]\s*", "", unit).strip()
    return Quantity(value=value, unit=unit)


def _close(x: float, ref: float, rel_tol: float, abs_tol: float) -> bool:
    return abs(x - ref) <= max(abs_tol, rel_tol * abs(ref))


def _structural_checks(
    task: str, ex: Extraction, *, expression_ok: bool = False, allow_interval: bool = False
) -> RewardResult | None:
    if not ex.candidates or not ex.distinct:
        return zero(task, "no_final_answer")
    if not allow_interval:
        for c in ex.distinct:
            if is_interval(c, math_mode=expression_ok):
                return zero(task, "interval_answer", flags=["interval_answer"])
    for c in ex.distinct:
        if has_multiple_values(c, expression_ok=expression_ok):
            return zero(task, "multi_answer", flags=["multi_answer"])
    return None


# =============================================================================================
# Graders
# =============================================================================================
@fail_closed(task=T1_NUM)
def grade_numeric(
    answer_text: str, ground_truth: dict[str, Any], cfg: NumericConfig | None = None, *, task: str = T1_NUM
) -> RewardResult:
    """Grade a numeric answer with units. ``answer_text`` is the post-``</think>`` text."""
    cfg = cfg or NumericConfig()
    try:
        ref_value = (
            float(ground_truth["value"])
            if not isinstance(ground_truth["value"], str)
            else safe_eval(ground_truth["value"])
        )
    except (KeyError, ValueError, TypeError):
        return zero(task, "bad_ground_truth")
    ref_unit_str = str(ground_truth.get("unit", "") or "")
    rel_tol = float(ground_truth.get("rel_tol", cfg.default_rel_tol))
    abs_tol = float(ground_truth.get("abs_tol", cfg.default_abs_tol))
    ref_unit = parse_unit(ref_unit_str)

    ex = extract_final_answer(answer_text, max_unmarked_chars=cfg.max_unmarked_chars)
    bad = _structural_checks(task, ex)
    if bad is not None:
        return bad

    values_in_ref_unit: list[float] = []
    for c in ex.distinct:
        if ref_unit.is_dimensionless:
            # "M 6.5" / "Mw 6.5": a magnitude-scale label written before the value.
            for lab in cfg.dimensionless_labels:
                if re.match(rf"^{re.escape(lab)}\s+[-+−]?[\d.]", c):
                    c = c[len(lab) :].strip()
                    break
        try:
            q = parse_quantity(c)
        except EvalError:
            return zero(task, "parse_fail")
        unit_str = q.unit
        if not unit_str or (ref_unit.is_dimensionless and unit_str in cfg.dimensionless_labels):
            if ref_unit_str and not ref_unit.is_dimensionless and cfg.require_unit:
                return zero(task, "missing_unit")
            if ref_unit_str and ref_unit.is_dimensionless and ref_unit.factor != 1.0 and cfg.require_unit:
                # e.g. reference "%" and answer "35": ambiguous between 35 % and 0.35
                return zero(task, "missing_unit")
            values_in_ref_unit.append(q.value)  # interpreted in the reference unit
            continue
        try:
            u = parse_unit(unit_str)
        except UnitError:
            return zero(task, "unit_parse_fail")
        if not compatible(u, ref_unit):
            return zero(task, "unit_dimension_mismatch")
        try:
            values_in_ref_unit.append(from_si(to_si(q.value, u), ref_unit))
        except DimensionError:
            return zero(task, "unit_dimension_mismatch")

    # Several candidates are allowed only if they all denote the same value.
    v0 = values_in_ref_unit[0]
    if any(not _close(v, v0, rel_tol, abs_tol) for v in values_in_ref_unit[1:]):
        return zero(task, "multi_answer", flags=["multi_answer"])

    correct = _close(v0, ref_value, rel_tol, abs_tol)
    rel_err = abs(v0 - ref_value) / max(abs(ref_value), 1e-300)
    score, reason = (1.0, "ok") if correct else (0.0, "wrong_value")
    pc = cfg.partial_credit or {}
    if not correct and pc.get("enabled"):
        # Ablation A7 arm only: structural failures above already returned 0; this grades closeness.
        score = max(0.0, 1.0 - rel_err / float(pc.get("max_rel_err", 0.10)))
        reason = "partial_credit" if score > 0 else "wrong_value"
    return RewardResult(
        score=score,
        task=task,
        reason=reason,
        correct=correct,
        # rel_err is logged for analysis; in production it never enters the reward (ablation A7).
        components={"t1_rel_err": min(rel_err, 1e6)},
    )


def _norm_symbolic(s: str) -> str:
    t = _clean(s)
    t = re.sub(r"\s+", "", t).replace("\\left", "").replace("\\right", "").replace("\\dfrac", "\\frac")
    t = t.replace("\\tfrac", "\\frac").replace("{", "(").replace("}", ")")
    return t.lower()


@fail_closed(task=T4_GENERAL)
def grade_math(
    answer_text: str, ground_truth: dict[str, Any], cfg: NumericConfig | None = None
) -> RewardResult:
    """T4 general-replay math: numeric equivalence (exact-ish tolerance) or normalized string match."""
    cfg = cfg or NumericConfig()
    task = T4_GENERAL
    if ground_truth.get("unit"):
        gt = dict(ground_truth)
        gt.setdefault("rel_tol", cfg.math_rel_tol)
        gt.setdefault("abs_tol", cfg.math_abs_tol)
        return grade_numeric(answer_text, gt, cfg, task=task)

    ref = ground_truth.get("value")
    ex = extract_final_answer(answer_text, max_unmarked_chars=cfg.max_unmarked_chars)
    # General math answers may be expressions; an inequality is acceptable only if the reference is one.
    bad = _structural_checks(
        task, ex, expression_ok=True, allow_interval=is_interval(str(ref), math_mode=True)
    )
    if bad is not None:
        return bad
    rel_tol = float(ground_truth.get("rel_tol", cfg.math_rel_tol))
    abs_tol = float(ground_truth.get("abs_tol", cfg.math_abs_tol))
    try:
        ref_v = float(ref) if not isinstance(ref, str) else safe_eval(ref)
    except (EvalError, ValueError, TypeError):
        ref_v = None

    results: list[bool] = []
    for c in ex.distinct:
        rhs = final_rhs(_clean(c))
        if ref_v is not None:
            try:
                results.append(_close(safe_eval(rhs), ref_v, rel_tol, abs_tol))
                continue
            except EvalError:
                results.append(False)
                continue
        results.append(_norm_symbolic(rhs) == _norm_symbolic(str(ref)))
    if len(set(results)) > 1:
        # Candidates disagree with each other (one right, one wrong): a hedge, not an answer.
        return zero(task, "multi_answer", flags=["multi_answer"])
    correct = bool(results and results[0])
    return RewardResult(
        score=float(correct), task=task, reason="ok" if correct else "wrong_value", correct=correct
    )
