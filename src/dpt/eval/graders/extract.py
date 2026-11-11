"""Final-answer extraction for evaluation (independent of ``dpt.rewards.answer_extract``).

Accepted answer conventions, in priority order:

1. ``\\boxed{...}``: every boxed expression in the answer text (nested braces handled)
2. Explicit marker lines: ``Final answer: ...``, ``Answer: ...``, ``最终答案：...``, ``答案：...``
3. A short bare answer: the whole answer text if it is one line of ≤ 80 characters

If the answer has **more than one distinct** candidate, extraction fails with
``multiple_answers``. Listing several candidates and hoping one matches was a
reward hack seen at RL step 95 (report §Reward hacking), and the eval grader
must not reward it either. Candidates that differ only in whitespace or
``$`` delimiters count as the same answer.

The input is the *final answer* (thinking already stripped by the harness).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_MARKER_RE = re.compile(
    r"^\s*(?:\*\*)?(?:final\s+answer|answer|最终答案|答案)(?:\*\*)?\s*(?:is)?\s*[:：]\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class Extraction:
    answer: str | None
    reason: str  # "ok" | "no_answer" | "multiple_answers"
    candidates: tuple[str, ...] = ()


def find_boxed(text: str) -> list[str]:
    """Contents of every ``\\boxed{...}`` / ``\\fbox{...}`` with balanced braces."""
    out: list[str] = []
    for m in re.finditer(r"\\(?:boxed|fbox)\s*\{", text):
        depth, i = 1, m.end()
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            out.append(text[m.end() : i - 1].strip())
    return out


def _canon(s: str) -> str:
    s = s.strip().strip("$").strip()
    s = re.sub(r"\s+", "", s)
    return s.rstrip(".。")


def _dedupe(cands: list[str]) -> list[str]:
    seen: dict[str, str] = {}
    for c in cands:
        k = _canon(c)
        if k and k not in seen:
            seen[k] = c.strip()
    return list(seen.values())


def extract_final_answer(
    answer_text: str, *, allow_bare: bool = True, bare_max_chars: int = 80
) -> Extraction:
    """Extract exactly one final answer from ``answer_text`` or explain why not."""
    text = answer_text.strip()
    if not text:
        return Extraction(None, "no_answer")
    boxed = _dedupe(find_boxed(text))
    if boxed:
        if len(boxed) > 1:
            return Extraction(None, "multiple_answers", tuple(boxed))
        return Extraction(boxed[0], "ok", tuple(boxed))
    marked = _dedupe([m.group(1) for m in _MARKER_RE.finditer(text)])
    if marked:
        if len(marked) > 1:
            return Extraction(None, "multiple_answers", tuple(marked))
        return Extraction(marked[0].rstrip(".。"), "ok", tuple(marked))
    if allow_bare and "\n" not in text and len(text) <= bare_max_chars:
        return Extraction(text.rstrip(".。"), "ok", (text,))
    return Extraction(None, "no_answer")
