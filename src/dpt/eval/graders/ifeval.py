"""IFEval-style verifiable-instruction checker (eval side).

Each item lists machine-checkable constraints, such as
``{"type": "min_words", "value": 300}`` or
``{"type": "include_keywords", "value": ["moment magnitude"]}`` (canonical schema:
``dpt.common.ifeval_schema``; other vocabularies are normalized on entry). Two
accuracies are reported, following the IFEval convention:

* **strict**: every constraint holds on the answer as written.
* **loose**: every constraint holds on at least one lenient variant of the
  answer (markdown emphasis removed, first or last line dropped). This
  absorbs harmless preambles like "Sure, here it is:".

The item counts as correct under strict accuracy, which is the headline
number. Loose accuracy and per-instruction pass rates go into ``details``.
This module shares no code with the training-side checker in
``dpt.rewards.instruction``. That checker also scores content relevance (the
step-60 filler hack), which IFEval itself does not measure.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from dpt.common.ifeval_schema import normalize_constraints
from dpt.eval.graders import GradeResult

Check = Callable[[str, dict[str, Any]], bool]


def _words(t: str) -> list[str]:
    return re.findall(r"\b[\w'-]+\b", t)


def _paragraphs(t: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n|\*\*\*", t) if p.strip()]


def _strip_fence(t: str) -> str:
    m = re.match(r"^\s*```(?:json)?\s*\n(.*)\n```\s*$", t, re.DOTALL)
    return m.group(1) if m else t


def _json_ok(t: str, _: dict[str, Any]) -> bool:
    try:
        json.loads(_strip_fence(t))
        return True
    except ValueError:
        return False


def _rel(n: float, c: dict[str, Any], default: str) -> bool:
    v, r = float(c["value"]), c.get("relation", default)
    return {"at_least": n >= v, ">=": n >= v, "at_most": n <= v, "<=": n <= v, "exactly": n == v,
            "==": n == v, "less_than": n < v, "<": n < v, "more_than": n > v, ">": n > v}.get(r, False)


def _kw(c: dict[str, Any]) -> list[str]:
    v = c["value"]
    return [str(x) for x in (v if isinstance(v, (list, tuple)) else [v])]


def _n_sentences(t: str) -> int:
    return len(re.findall(r"[.!?。！？](?:\s|$)", t))


# Keyed on the canonical schema (dpt.common.ifeval_schema); the checks themselves are this module's own.
CHECKS: dict[str, Check] = {
    "words": lambda t, c: _rel(len(_words(t)), c, "exactly"),
    "min_words": lambda t, c: len(_words(t)) >= int(c["value"]),
    "max_words": lambda t, c: len(_words(t)) <= int(c["value"]),
    "sentences": lambda t, c: _rel(_n_sentences(t), c, "at_least"),
    "no_commas": lambda t, c: "," not in t and "，" not in t,
    "all_lowercase": lambda t, c: t == t.lower() and any(ch.isalpha() for ch in t),
    "all_uppercase": lambda t, c: t == t.upper() and any(ch.isalpha() for ch in t),
    "include_keywords": lambda t, c: all(k.lower() in t.lower() for k in _kw(c)),
    "forbid_keywords": lambda t, c: not any(k.lower() in t.lower() for k in _kw(c)),
    "keyword_frequency": lambda t, c: _rel(t.lower().count(str(c["keyword"]).lower()), c, "at_least"),
    "bullets": lambda t, c: _rel(len(re.findall(r"^\s*[*\-•]\s+", t, re.MULTILINE)), c, "exactly"),
    "paragraphs": lambda t, c: _rel(len(_paragraphs(t)), c, "exactly"),
    "sections": lambda t, c: _rel(
        len(re.findall(rf"^\s*{re.escape(c.get('marker', 'Section'))}\s*\d+", t, re.MULTILINE)), c, "exactly"),
    "json_format": _json_ok,
    "start_with": lambda t, c: t.strip().lower().startswith(str(c["value"]).strip().lower()),
    "end_with": lambda t, c: t.strip().lower().endswith(str(c["value"]).strip().lower()),
    "title_in_double_angular": lambda t, c: re.search(r"<<[^<>\n]+>>", t) is not None,
    "quotation": lambda t, c: t.strip().startswith('"') and t.strip().endswith('"'),
    "placeholders": lambda t, c: len(re.findall(r"\[[^\[\]\n]+\]", t)) >= int(c.get("value", 1)),
    "highlighted_sections": lambda t, c: len(re.findall(r"\*[^*\n]+\*", t)) >= int(c.get("value", 1)),
    "postscript": lambda t, c: re.search(rf"^\s*{re.escape(c.get('marker', 'P.S.'))}", t, re.MULTILINE)
    is not None,
}


def _variants(t: str) -> list[str]:
    lines = t.split("\n")
    base = [t, t.replace("*", "")]
    out = list(base)
    for b in base:
        bl = b.split("\n")
        if len(lines) > 1:
            out += ["\n".join(bl[1:]), "\n".join(bl[:-1]), "\n".join(bl[1:-1])]
    return out


def check_constraints(answer: str, constraints: list[dict[str, Any]]) -> tuple[list[bool], list[bool]]:
    """Return (strict_pass_per_constraint, loose_pass_per_constraint). Unknown types fail closed."""
    strict, loose = [], []
    variants = _variants(answer)
    for c in normalize_constraints(constraints):
        fn = CHECKS.get(c.get("type", ""))
        if fn is None:
            strict.append(False)
            loose.append(False)
            continue
        try:
            strict.append(bool(fn(answer, c)))
            loose.append(any(fn(v, c) for v in variants))
        except (KeyError, TypeError, ValueError):   # malformed constraint: fail closed
            strict.append(False)
            loose.append(False)
    return strict, loose


def grade_ifeval(answer_text: str, gold: dict[str, Any]) -> GradeResult:
    constraints = list(gold.get("constraints", []))
    if not constraints:
        return GradeResult.fail("no_constraints")
    if not answer_text.strip():
        return GradeResult.fail("no_answer")
    strict, loose = check_constraints(answer_text, constraints)
    ok = all(strict)
    return GradeResult(
        correct=ok,
        score=1.0 if ok else 0.0,
        reason="ok" if ok else "constraint_failed",
        details={
            "strict": strict,
            "loose": loose,
            "loose_correct": all(loose),
            "instruction_strict_frac": sum(strict) / len(strict),
        },
    )
