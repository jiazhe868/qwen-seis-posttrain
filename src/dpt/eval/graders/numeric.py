"""Numeric-answer grader for D-Num / D-Fresh (and math items in general benchmarks).

Rules (report §评测体系, Reward hacking cases at steps 95 and 140):

* The answer must be **one** numeric value, optionally with a unit.
  A range ("about 3.1–3.4 km", "between 3 and 4", "[3, 4]"), an uncertainty
  ("3.2 ± 0.1"), an inequality ("> 3") or several alternatives ("3 or 4")
  scores 0.
* Values are compared in SI after unit conversion, with the item's tolerance:
  ``|pred - gold| <= max(abs_tol, rel_tol * |gold|)``, where ``abs_tol`` is
  given in the gold unit.
* A dimension mismatch (pred in km, gold in seconds) scores 0. A missing
  unit either scores 0 (``missing_unit="strict"``) or is read as the gold
  unit (``"assume_gold"``). The eval suite and the dev gold grader use
  ``strict``, the same policy as training and the prompt instruction.
  ``assume_gold`` stays available for external benchmarks whose prompts do
  not ask for units.
* Value expressions are evaluated with sympy, so ``1/2``, ``\\frac{1}{2}``,
  ``3.2\\times10^{4}``, ``3.2e4``, ``2\\sqrt{2}`` and whole unit-free expressions
  (``\\frac{\\pi}{2}``, ``\\sqrt{2}+1``, ``2^{10}``) all work. Spelled-out units
  ("seconds", "kilometres") and magnitude labels ("6.5 Mw") are understood. This parser is
  independent of the training grader's regex-based parser.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from dpt.eval.graders import GradeResult
from dpt.eval.graders.extract import extract_final_answer
from dpt.eval.graders.units import UnitParseError, normalize_unit_text, to_si

_NUM = r"[-+−]?\s*(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d*)?|\.\d+)"
_SCI = (
    rf"{_NUM}(?:\s*[eE]\s*[-+−]?\d+)?"
    r"(?:\s*(?:\*|×|x|\\times|\\cdot|·)\s*10\s*(?:\^|\*\*)\s*\{?\s*[-+−]?\d+\s*\}?)?"
)
_FRAC_LATEX = r"[-+−]?\s*\\[dt]?frac\s*\{[^{}]+\}\s*\{[^{}]+\}"
_FRAC_PLAIN = rf"{_NUM}\s*/\s*\d+(?:\.\d+)?(?![A-Za-z])"
_SQRT = rf"(?:{_NUM}\s*)?\\sqrt\s*\{{[^{{}}]+\}}"
_VALUE_RE = re.compile(rf"^\s*(?P<val>{_FRAC_LATEX}|{_SQRT}|{_FRAC_PLAIN}|{_SCI})(?P<rest>.*)$", re.DOTALL)

_N = r"\d+(?:\.\d+)?"
_INTERVAL_RES = [
    re.compile(rf"{_N}\s*(?:–|—|~|〜|～|\bto\b|至|到)\s*[-+]?{_N}"),
    re.compile(rf"(?<=\d)\s*-\s*{_N}"),  # 3.1-3.4; "1e-3"/"10^-3" have no digit before "-"
    re.compile(rf"\bbetween\s+{_N}.*\band\s+{_N}", re.IGNORECASE),
    re.compile(rf"[\[\(]\s*[-+]?{_N}\s*,\s*[-+]?{_N}\s*[\]\)]"),
    re.compile(r"±|\+/-|\\pm|\\mp"),
    re.compile(r"^\s*(?:<|>|≤|≥|\\le|\\ge|\\lt|\\gt)"),
]
_ALTERNATIVE_RE = re.compile(rf"{_N}[^\d]{{0,12}}\b(?:or|and)\b\s*[-+]?{_N}|{_N}\s*或\s*{_N}", re.IGNORECASE)
_LEADING_LABEL_RE = re.compile(
    r"^\s*(?:\\?[A-Za-z][A-Za-z_\\{}]{0,8}\s*(?:=|≈|\\approx)|≈|\\approx|~|约|approximately|about|roughly)\s*",
    re.IGNORECASE,
)
_MAGNITUDE_LABEL_RE = re.compile(r"^\s*(?:M_?[wWsLbe]?|m_?[bB]|Mw|ML|Ms|mb)\s+(?=[-+]?\d)")


_SCALE_UNITS = {"%", "‰", "ppm", "ppb"}


@dataclass(frozen=True)
class ParsedQuantity:
    value: float
    unit: str


class ValueParseError(ValueError):
    pass


def _to_sympy_syntax(expr: str) -> str:
    s = expr.replace("−", "-").replace(" ", "")
    s = re.sub(r"(\d),(?=\d{3}\b)", r"\1", s)  # thousands separators
    s = re.sub(r"\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"*sqrt(\1)", s)
    s = re.sub(r"([+\-/(])\*sqrt", r"\1sqrt", s)          # "1+\sqrt{2}" → "1+sqrt(2)"
    s = s.lstrip("*")
    s = re.sub(r"^([-+])\*", r"\1", s)
    s = s.replace("\\times", "*").replace("\\cdot", "*").replace("×", "*").replace("·", "*")
    s = re.sub(r"(?<=\d)x(?=10)", "*", s)
    s = s.replace("^", "**").replace("{", "(").replace("}", ")").replace("\\pi", "pi")
    return s


def evaluate_number(expr: str) -> float:
    """Evaluate a numeric expression string with sympy (falls back to ``float``)."""
    s = _to_sympy_syntax(expr)
    try:
        return float(s)
    except ValueError:
        pass
    try:
        import sympy

        val = sympy.sympify(s, rational=False)
        out = float(sympy.N(val))
    except Exception as e:  # sympy raises many exception types on garbage input
        raise ValueParseError(f"cannot evaluate {expr!r}: {e}") from e
    if not math.isfinite(out):
        raise ValueParseError(f"non-finite value {expr!r}")
    return out


def _clean(answer: str) -> str:
    a = answer.strip().strip("$").strip()
    a = re.sub(r"\\text\{\s*\}", "", a)
    a = re.sub(r"\\(?:left|right)", "", a)
    a = a.replace("\\!", "").replace("\\,", " ")
    a = _MAGNITUDE_LABEL_RE.sub("", a)
    for _ in range(2):  # "t = ≈ 20 s" → "20 s"
        a = _LEADING_LABEL_RE.sub("", a)
    return a.strip().rstrip(".。").strip()


def check_single_value(answer: str) -> str | None:
    """Return a failure reason if ``answer`` is a range/uncertainty/inequality/alternatives, else None."""
    for rx in _INTERVAL_RES:
        if rx.search(answer):
            return "interval"
    if _ALTERNATIVE_RE.search(answer):
        return "multiple_answers"
    return None


# A unit-free math expression: digits, operators, braces/parentheses and a few LaTeX commands.
_MATH_CMDS_RE = re.compile(r"\\(?:[dt]?frac|sqrt|pi|times|cdot|left|right)")
_PURE_MATH_RE = re.compile(r"^[\d\s.+\-−*/^(){}×·,]+$")


def _pure_math_value(a: str) -> float | None:
    """Value of an answer that is a whole math expression (``\\pi/2``, ``\\sqrt{2}+1``, ``2^{10}``), else None."""
    stripped = _MATH_CMDS_RE.sub(" ", a)
    if not _PURE_MATH_RE.match(stripped) or not re.search(r"\d|\\pi", a):
        return None
    s = re.sub(r"\\sqrt\s*\{", r"\\sqrt{", a)
    try:
        return evaluate_number(s)
    except ValueParseError:
        return None


def parse_quantity(answer: str) -> ParsedQuantity:
    """Split ``"3.2 \\times 10^{4} km/s"`` into ``ParsedQuantity(32000.0, "km/s")``.

    A unit-free expression (T4 math: ``\\frac{\\pi}{2}``, ``\\sqrt{2}+1``, ``2^{10}``) is evaluated as a
    whole first; otherwise the leading number is split from the unit text.
    """
    a = _clean(answer)
    whole = _pure_math_value(a)
    if whole is not None:
        return ParsedQuantity(value=whole, unit="")
    m = _VALUE_RE.match(a)
    if not m:
        raise ValueParseError(f"no leading numeric value in {answer!r}")
    value = evaluate_number(m.group("val"))
    # LaTeX wrappers (\mathrm{km}, \text{ s}) are handled by the unit normalizer.
    return ParsedQuantity(value=value, unit=m.group("rest").strip())


def grade_numeric(
    answer_text: str, gold: dict[str, Any], *, missing_unit: str = "assume_gold"
) -> GradeResult:
    """Grade a final-answer string against ``gold = {value, unit, rel_tol, abs_tol}``."""
    ext = extract_final_answer(answer_text)
    if ext.answer is None:
        return GradeResult.fail(ext.reason, candidates=list(ext.candidates))
    reason = check_single_value(ext.answer)
    if reason:
        return GradeResult.fail(reason, answer=ext.answer)
    try:
        pred = parse_quantity(ext.answer)
    except ValueParseError as e:
        return GradeResult.fail("parse_error", answer=ext.answer, error=str(e))

    gold_value = float(gold["value"])
    gold_unit = str(gold.get("unit", "") or "")
    rel_tol = float(gold.get("rel_tol", 1e-2))
    abs_tol = float(gold.get("abs_tol", 0.0))
    pred_unit = pred.unit
    if not normalize_unit_text(pred_unit):
        if gold_unit and missing_unit == "strict":
            return GradeResult.fail("missing_unit", answer=ext.answer)
        pred_unit = gold_unit
    try:
        g_si, g_dim = to_si(gold_value, gold_unit)
        if not normalize_unit_text(gold_unit):
            # Dimensionless gold: compare the raw number; the pred unit must also be dimensionless.
            p_scaled, p_dim = to_si(pred.value, pred_unit)
            if any(p_dim):
                return GradeResult.fail("unit_mismatch", answer=ext.answer, pred_unit=pred.unit)
            # Pure scale units (%, ‰, ppm) are converted; angle units are compared as written
            # because a unitless angle gold value is ambiguous between degrees and radians.
            scale_unit = normalize_unit_text(pred_unit) in _SCALE_UNITS
            p_si, g_si = (p_scaled if scale_unit else pred.value), gold_value
            tol = max(abs_tol, rel_tol * abs(gold_value))
        else:
            p_si, p_dim = to_si(pred.value, pred_unit)
            if p_dim != g_dim:
                return GradeResult.fail(
                    "unit_mismatch", answer=ext.answer, pred_unit=pred.unit, gold_unit=gold_unit
                )
            tol_unit = to_si(abs_tol, gold_unit)[0] - to_si(0.0, gold_unit)[0]
            tol = max(abs(tol_unit), rel_tol * abs(g_si))
    except UnitParseError as e:
        return GradeResult.fail("unit_parse_error", answer=ext.answer, error=str(e))
    ok = abs(p_si - g_si) <= tol
    return GradeResult(
        correct=ok,
        score=1.0 if ok else 0.0,
        reason="ok" if ok else "wrong_value",
        details={"answer": ext.answer, "pred_si": p_si, "gold_si": g_si, "tol": tol},
    )
