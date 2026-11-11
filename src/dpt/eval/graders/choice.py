"""Multiple-choice grader (GPQA-Diamond, MMLU-Pro).

The answer must name exactly one option letter. Accepted forms:
``\\boxed{C}``, ``Answer: C``, ``(C)``, or a bare ``C``. Naming two
different letters fails closed with ``multiple_answers``.
"""

from __future__ import annotations

import re
import string
from typing import Any

from dpt.eval.graders import GradeResult
from dpt.eval.graders.extract import find_boxed

_MARKER_RE = re.compile(
    r"(?:final\s+answer|answer|答案)\s*(?:is)?\s*[:：]?\s*\(?\s*([A-J])\s*\)?(?![A-Za-z])", re.IGNORECASE
)


def extract_choice(answer_text: str, n_options: int = 10) -> tuple[str | None, str]:
    letters = set(string.ascii_uppercase[:n_options])
    boxed = {b.strip().strip("()").upper() for b in find_boxed(answer_text)}
    boxed = {b for b in boxed if b}
    if boxed:
        if len(boxed) > 1:
            return None, "multiple_answers"
        b = next(iter(boxed))
        return (b, "ok") if b in letters else (None, "invalid_choice")
    marked = {m.group(1).upper() for m in _MARKER_RE.finditer(answer_text)}
    if marked:
        if len(marked) > 1:
            return None, "multiple_answers"
        m = next(iter(marked))
        return (m, "ok") if m in letters else (None, "invalid_choice")
    bare = answer_text.strip().strip(".()").upper()
    if len(bare) == 1 and bare in letters:
        return bare, "ok"
    return None, "no_answer"


def grade_choice(answer_text: str, gold: dict[str, Any]) -> GradeResult:
    letter, reason = extract_choice(answer_text, int(gold.get("n_options", 10)))
    if letter is None:
        return GradeResult.fail(reason)
    ok = letter == str(gold["letter"]).upper()
    return GradeResult(
        correct=ok, score=float(ok), reason="ok" if ok else "wrong_choice", details={"pred": letter}
    )
