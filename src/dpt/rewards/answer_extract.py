"""Final-answer extraction and the structural checks that defeat known answer-format hacks.

Input is the text **after** ``</think>`` (see :mod:`dpt.chat.thinking`). The
caller has already failed responses without an end-of-thinking marker.

Extraction order (strictest signal first):

1. every ``\\boxed{…}`` / ``\\fbox{…}`` (nested braces handled);
2. plus every "Final answer:" / "Answer:" / "最终答案：" line that does not
   itself contain a box;
3. only if 1 and 2 found nothing, the whole answer text, *if it is short*
   (``max_unmarked_chars``). A long answer with no marker is
   ``no_final_answer``. We do not fish a number out of the prose. That old
   behaviour is what "answer buried in a long passage" adversarial cases
   exploit (report §第二层 grader 单元测试集).

Structural checks (each hard-zeroes a T1 answer, report §各任务的 reward):

* :func:`is_interval`: ranges ("3.1–3.4 km", "between 3 and 4", "3 ± 0.2",
  "[3, 4]", inequalities such as "> 3"). Step-140 hack: the old parser took
  the first number of "about 3.1–3.4 km".
* :func:`has_multiple_values`: more than one value in one candidate ("20 s or
  25 s", "20 s, 25 s"). Distinct values *across* candidates are checked by the
  numeric grader with tolerance. Step-95 hack: the old parser gave credit if
  any of several candidates matched.

Implementation note: numbers, including scientific notation, are first
collapsed into the placeholder ``#`` and exponents into ``^E``. The checks then
run as regexes over that skeleton, so ``6.0×10^-3`` or ``s^-1`` can never be
mistaken for a range ``a - b``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --- number skeletonization -------------------------------------------------------------------
_MANTISSA = r"(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+)(?:\.\d+)?|\.\d+"
_SCI_SUFFIX = (
    r"(?:\s*[eE]\s*[-+−]?\s*\d+"
    r"|\s*(?:×|x|X|\*|\\times|\\cdot|·)\s*10\s*(?:\^|\*\*)\s*\{?\s*[-+−]?\s*\d+\s*\}?)"
)
SCI_NUMBER_RE = re.compile(rf"(?:{_MANTISSA}){_SCI_SUFFIX}")
POW10_RE = re.compile(r"(?<![\d.])10\s*\^\s*\{?\s*[-+−]?\s*\d+\s*\}?")
EXPONENT_RE = re.compile(r"(?:\^|\*\*)\s*\{?\s*[-+−]?\s*\d+(?:\.\d+)?\s*\}?")
PLAIN_NUMBER_RE = re.compile(_MANTISSA)
FRAC_RE = re.compile(r"\\[dt]?frac\s*\{[^{}]*\}\s*\{[^{}]*\}")
SQRT_RE = re.compile(r"\\sqrt\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}")


def _collapse_latex_values(t: str) -> str:
    """Collapse innermost-first ``\\sqrt{…}`` / ``\\frac{…}{…}`` groups so nested forms count as one value."""
    for _ in range(10):
        new = SQRT_RE.sub(" #SQ ", t)
        new = FRAC_RE.sub(" # ", new)
        if new == t:
            break
        t = new
    return t.replace("#SQ", "#")


def skeleton(s: str) -> str:
    """Replace numbers with ``#`` and unit exponents with ``^E`` (order matters).

    ``#`` is used instead of a letter so that unit symbols (N = newton) never
    collide with the placeholder.
    """
    t = s.replace("#", " ")
    t = _collapse_latex_values(t)
    t = SCI_NUMBER_RE.sub(" # ", t)
    t = POW10_RE.sub(" # ", t)
    t = EXPONENT_RE.sub("^E", t)
    t = PLAIN_NUMBER_RE.sub(" # ", t)
    t = re.sub(r"#\s*/\s*#", "#", t)  # simple fractions 1/2 count as one value
    return re.sub(r"\s+", " ", t).strip()


# --- boxed / marker extraction ----------------------------------------------------------------
_BOX_START = re.compile(r"\\(?:boxed|fbox)\s*\{")
# A marker line must be the marker word(s) as whole words followed by a colon ("Answer: 20 s",
# "**Final answer:** 20 s", "答案：20 s") or by "is" ("The final answer is 20 s"). Without the
# word boundary and the required separator, prose such as "Answering this requires t = d/v"
# was read as an answer line and outranked the \boxed{} that followed.
_MARKER_RE = re.compile(
    r"(?im)^[ \t>*_#-]*(?:\*\*)?(?:"
    r"(?:the\s+)?final\s+answer\s+is|(?:final\s+answer|answer)\b(?:\*\*)?\s*(?:is\b)?\s*[:：]"
    r"|(?:最终答案|答案|答)\s*(?:是)?\s*[:：]"
    r")\s*(?:\*\*)?\s*(.+?)\s*$"
)


def find_boxed(text: str) -> list[str]:
    """Contents of every ``\\boxed{…}`` with balanced nested braces (unterminated boxes ignored)."""
    out: list[str] = []
    for m in _BOX_START.finditer(text):
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


def normalize_candidate(s: str) -> str:
    """Light normalization for de-duplicating candidates (not for value comparison)."""
    t = s.strip().strip("$").strip()
    t = re.sub(r"^\\\[|\\\]$", "", t).strip()
    for _ in range(2):  # unwrap (possibly nested) text/emphasis wrappers
        t = re.sub(
            r"\\(?:text|mathrm|textrm|rm|mathbf|textbf|bm|boldsymbol|mathit|emph)\s*\{([^{}]*)\}", r"\1", t
        )
    t = re.sub(r"(?<=\d)\\,(?=\d{3}(?!\d))", "", t)  # thin-space thousands separator: 12\,500
    t = t.replace("\\,", " ").replace("\\ ", " ").replace("\\!", "").replace("\\;", " ")
    t = re.sub(r"[\s。．]+$", "", t).rstrip(".")
    return re.sub(r"\s+", " ", t).strip()


@dataclass
class Extraction:
    candidates: list[str] = field(default_factory=list)
    method: str = "none"  # "boxed" | "marker" | "boxed+marker" | "short_text" | "none"

    @property
    def distinct(self) -> list[str]:
        seen: dict[str, None] = {}
        for c in self.candidates:
            seen.setdefault(normalize_candidate(c), None)
        return [c for c in seen if c]


def extract_final_answer(answer_text: str, *, max_unmarked_chars: int = 64) -> Extraction:
    """Collect candidate final answers from post-thinking text (see module docstring)."""
    boxed = find_boxed(answer_text)
    marker_hits = []
    for m in _MARKER_RE.finditer(answer_text):
        line = m.group(1).strip()
        if not line or _BOX_START.search(line):
            continue
        marker_hits.append(line)
    if boxed and marker_hits:
        return Extraction(boxed + marker_hits, "boxed+marker")
    if boxed:
        return Extraction(boxed, "boxed")
    if marker_hits:
        return Extraction(marker_hits, "marker")
    text = answer_text.strip()
    if text and len(text) <= max_unmarked_chars and "\n" not in text:
        return Extraction([text], "short_text")
    return Extraction([], "none")


# --- structural checks ------------------------------------------------------------------------
_UNITCH = r"[A-Za-zµμΩÅ°%‰/·\.\\\s\^E()]{0,16}?"
# Hyphen/minus ranges ("3.1-3.4") are separate so that math mode can treat "a - b" as subtraction.
_HYPHEN_RANGE = re.compile(rf"#\s*{_UNITCH}\s*(?<!\^)(?:-|−)\s*[-−]?\s*#")
_INTERVAL_PATTERNS = [
    re.compile(rf"#\s*{_UNITCH}\s*(?<!\^)(?:–|—|~|～|〜|\bto\b|\buntil\b|至|到)\s*[-−]?\s*#"),
    re.compile(r"(?i)\bbetween\b.*?#.*?\band\b.*?#"),
    re.compile(r"(?:从|介于|在)\s*[-−]?\s*#.*?(?:到|至|和|与|之间)"),
    re.compile(r"#\s*(?:±|∓|\+/-|\+-|\\pm|\\mp)\s*#"),
    re.compile(r"[\[\(]\s*[-−]?\s*#\s*[,;，]\s*[-−]?\s*#\s*[\]\)]"),
    re.compile(
        r"(?i)(?:>=|<=|≥|≤|⩾|⩽|(?<![=\-])>|<(?![=\-])|\\geq?|\\leq?|\\gt|\\lt|\bat least\b|\bat most\b|"
        r"\bmore than\b|\bless than\b|\bgreater than\b|\bup to\b|超过|不少于|不超过|小于|大于)\s*[-−]?\s*#"
    ),
    re.compile(
        rf"#\s*{_UNITCH}\s*(?:以上|以下|或更多|或更少|\+(?!\s*#)|(?i:\bor more\b|\bor less\b|\band above\b|"
        r"\band below\b|\bor above\b|\bor below\b|\bor so\b))"
    ),
]
_ALT_SPLIT = re.compile(r"(?i)\s+or\s+|\s+and/or\s+|或者|或|;|；|\beither\b")
_RHS_SPLIT = re.compile(r"=|≈|≃|\\approx|\\simeq|≅")


def is_interval(candidate: str, *, math_mode: bool = False) -> bool:
    """True if ``candidate`` expresses a range, an uncertainty band or an inequality.

    ``math_mode=True`` (T4 general math) does not read ``a - b`` as a range,
    since subtraction is a legitimate expression there. T1 answers are single
    quantities, so in T1 "3.1-3.4 km" is a range.
    """
    sk = skeleton(candidate)
    if not math_mode and _HYPHEN_RANGE.search(sk):
        return True
    return any(p.search(sk) for p in _INTERVAL_PATTERNS)


def final_rhs(candidate: str) -> str:
    """Right-hand side after the last '=' / '≈' (``"t = d/v = 20 s"`` → ``"20 s"``)."""
    parts = _RHS_SPLIT.split(candidate)
    return parts[-1].strip() if parts else candidate


def has_multiple_values(candidate: str, *, expression_ok: bool = False) -> bool:
    """True if one candidate string carries more than one value.

    Alternatives ("3 or 4", "3; 4") are checked on the full string, so that
    ``"x = 3 or x = 4"`` cannot hide behind the right-hand-side rule. For
    single-quantity answers (T1) the number of values on the right-hand side of
    the final equality must then be exactly one. ``expression_ok=True`` (T4
    math) skips that count, because ``2\\sqrt{2} + 3`` is one value written
    with three numbers.
    """
    alts = [a for a in _ALT_SPLIT.split(candidate) if a and "#" in skeleton(a)]
    if len(alts) > 1:
        return True
    if expression_ok:
        return False
    return count_values(candidate) > 1


def count_values(candidate: str) -> int:
    return skeleton(final_rhs(candidate)).split().count("#")
