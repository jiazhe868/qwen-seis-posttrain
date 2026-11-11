"""T4 instruction-following reward: verifiable rule checks plus a content-relevance check.

T4 is general replay that guards against forgetting (report §RL 题库). Its
instruction-following half rewards responses that satisfy machine-checkable
constraints ("at least 200 words", "exactly 3 bullet points", "end with
'Any questions?'", …).

The step-60 hack: the policy met word-count and format rules by padding with
irrelevant text. T4 reward rose while IFEval did not move. The fix adds a
**relevance gate**. A response must share enough content vocabulary with the
prompt and must not be filler (lorem ipsum, heavy n-gram repetition, repeated
sentences). Otherwise the score is 0 whatever the format.

Independence: this checker is written separately from the IFEval grader used
in evaluation (``dpt.eval.graders``). They share no code, so the policy cannot
overfit the evaluation grader through the training reward (report §T4 row: "与评测用的
grader 是不同实现").

Scoring is binary by default (all constraints must pass), matching the
"no partial credit" stance used for T1. ``scoring: fraction`` is available
for ablations.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from dpt.common.ifeval_schema import normalize_constraints
from dpt.common.text import normalize_text, tokenize
from dpt.common.types import T4_GENERAL
from dpt.rewards.base import RewardResult, fail_closed, zero
from dpt.rewards.hack_patterns import _is_filler

_STOP = set(
    "a an the of to in on for and or is are was were be been with as by at from that this these those it its "
    "your you we our they their i me my he she his her them can could should would will shall may might must "
    "do does did not no yes please write answer response respond give provide make use using about into than "
    "then there here what which who whom how why when where all any each every some such only also very more "
    "most least words word sentences sentence paragraphs paragraph bullet bullets points point format least "
    "exactly at end start begin title section sections json lowercase uppercase letters letter".split()
)


def _words(text: str) -> list[str]:
    return [t for t in tokenize(text) if re.match(r"[A-Za-z一-鿿]", t)]


def _cjk_count(text: str) -> int:
    return len(re.findall(r"[一-鿿]", text))


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?。！？])\s+", text.strip()) if s.strip()]


def _paragraphs(text: str, sep: str = "\n\n") -> list[str]:
    if sep == "***":
        parts = re.split(r"\n\s*\*\*\*\s*\n", text)
    else:
        parts = re.split(r"\n\s*\n", text)
    return [p for p in parts if p.strip()]


def _relation(n: float, relation: str, value: float) -> bool:
    if relation in ("at_least", ">="):
        return n >= value
    if relation in ("at_most", "<="):
        return n <= value
    if relation in ("exactly", "=="):
        return n == value
    if relation in ("less_than", "<"):
        return n < value
    if relation in ("more_than", ">"):
        return n > value
    raise ValueError(f"unknown relation {relation!r}")


def _count_words(text: str) -> int:
    # CJK text has no spaces; count each CJK char as a word (a common convention for length rules).
    cjk = _cjk_count(text)
    latin = len([w for w in _words(text) if not re.match(r"[一-鿿]", w)])
    return latin + cjk


# Each checker: (response_text, constraint_dict) -> bool
def _c_words(t: str, c: dict) -> bool:
    return _relation(_count_words(t), c.get("relation", "at_least"), float(c["value"]))


def _c_min_words(t: str, c: dict) -> bool:
    return _count_words(t) >= float(c["value"])


def _c_max_words(t: str, c: dict) -> bool:
    return _count_words(t) <= float(c["value"])


def _c_sentences(t: str, c: dict) -> bool:
    return _relation(len(_sentences(t)), c.get("relation", "at_least"), float(c["value"]))


def _c_paragraphs(t: str, c: dict) -> bool:
    return _relation(
        len(_paragraphs(t, c.get("separator", "\n\n"))), c.get("relation", "exactly"), float(c["value"])
    )


def _c_bullets(t: str, c: dict) -> bool:
    n = len(re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+\S", t))
    return _relation(n, c.get("relation", "exactly"), float(c["value"]))


def _c_include(t: str, c: dict) -> bool:
    low = t.lower()
    return all(str(k).lower() in low for k in c["value"])


def _c_forbid(t: str, c: dict) -> bool:
    low = t.lower()
    return not any(re.search(rf"\b{re.escape(str(k).lower())}\b", low) for k in c["value"])


def _c_keyword_freq(t: str, c: dict) -> bool:
    n = len(re.findall(rf"(?i)\b{re.escape(str(c['keyword']))}\b", t))
    return _relation(n, c.get("relation", "at_least"), float(c["value"]))


def _c_json(t: str, c: dict) -> bool:
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", t.strip())
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError:
        return False


def _c_end_with(t: str, c: dict) -> bool:
    return t.rstrip().endswith(str(c["value"]).strip())


def _c_start_with(t: str, c: dict) -> bool:
    return t.lstrip().startswith(str(c["value"]).strip())


def _c_lower(t: str, c: dict) -> bool:
    return t == t.lower() and any(ch.isalpha() for ch in t)


def _c_upper(t: str, c: dict) -> bool:
    return t == t.upper() and any(ch.isalpha() for ch in t)


def _c_language(t: str, c: dict) -> bool:
    letters = len(re.findall(r"[A-Za-z]", t))
    cjk = _cjk_count(t)
    total = max(1, letters + cjk)
    lang = str(c["value"]).lower()
    if lang in ("zh", "chinese"):
        return cjk / total >= 0.5
    if lang in ("en", "english"):
        return letters / total >= 0.8
    raise ValueError(f"unsupported language {lang}")


def _c_no_commas(t: str, c: dict) -> bool:
    return "," not in t and "，" not in t


def _c_title(t: str, c: dict) -> bool:
    return bool(re.search(r"<<[^<>\n]+>>", t))


def _c_sections(t: str, c: dict) -> bool:
    marker = re.escape(str(c.get("marker", "Section")))
    n = len(re.findall(rf"(?m)^\s*{marker}\s*\d+", t))
    return _relation(n, c.get("relation", "exactly"), float(c["value"]))


def _c_placeholders(t: str, c: dict) -> bool:
    return len(re.findall(r"\[[^\[\]\n]+\]", t)) >= int(c["value"])


def _c_quotation(t: str, c: dict) -> bool:
    s = t.strip()
    return len(s) >= 2 and s[0] in '"“' and s[-1] in '"”'


def _c_postscript(t: str, c: dict) -> bool:
    return bool(re.search(rf"(?m)^\s*{re.escape(str(c.get('marker', 'P.S.')))}", t))


def _c_highlights(t: str, c: dict) -> bool:
    return len(re.findall(r"\*[^*\n]+\*", t)) >= int(c["value"])


CHECKERS: dict[str, Callable[[str, dict], bool]] = {
    "words": _c_words,
    "min_words": _c_min_words,
    "max_words": _c_max_words,
    "sentences": _c_sentences,
    "paragraphs": _c_paragraphs,
    "bullets": _c_bullets,
    "include_keywords": _c_include,
    "forbid_keywords": _c_forbid,
    "keyword_frequency": _c_keyword_freq,
    "json_format": _c_json,
    "end_with": _c_end_with,
    "start_with": _c_start_with,
    "all_lowercase": _c_lower,
    "all_uppercase": _c_upper,
    "language": _c_language,
    "no_commas": _c_no_commas,
    "title_in_double_angular": _c_title,
    "sections": _c_sections,
    "placeholders": _c_placeholders,
    "quotation": _c_quotation,
    "postscript": _c_postscript,
    "highlighted_sections": _c_highlights,
}


@dataclass
class InstructionConfig:
    scoring: str = "all"  # "all" (binary) | "fraction"
    relevance_check: bool = True
    min_prompt_overlap: float = 0.15  # fraction of prompt content words that must reappear
    min_prompt_content_words: int = 4  # skip relevance gate for very short prompts
    filler_repetition_threshold: float = 0.3
    # Constraint types whose output is intrinsically content-free; relevance gate is skipped for these.
    relevance_exempt: list[str] = field(default_factory=lambda: ["json_format"])

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "InstructionConfig":
        d = dict(d or {})
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def content_words(text: str) -> set[str]:
    return {
        w for w in _words(normalize_text(text)) if w not in _STOP and (len(w) > 2 or re.match(r"[一-鿿]", w))
    }


def relevance(prompt: str, response: str) -> float:
    """Fraction of the prompt's content words that reappear in the response."""
    p = content_words(prompt)
    if not p:
        return 1.0
    return len(p & content_words(response)) / len(p)


@fail_closed(task=T4_GENERAL)
def grade_instruction(
    answer_text: str, ground_truth: dict[str, Any], *, prompt: str = "", cfg: InstructionConfig | None = None
) -> RewardResult:
    cfg = cfg or InstructionConfig()
    task = T4_GENERAL
    # One schema for data, training grader and eval grader (dpt.common.ifeval_schema).
    constraints = normalize_constraints(ground_truth.get("constraints") or [])
    if not constraints:
        return zero(task, "bad_ground_truth")
    text = answer_text.strip()
    if not text:
        return zero(task, "no_final_answer")

    results = []
    for c in constraints:
        fn = CHECKERS.get(c.get("type", ""))
        if fn is None:
            return zero(task, "bad_ground_truth")
        try:
            results.append(bool(fn(text, c)))
        except (KeyError, TypeError, ValueError):     # malformed constraint: fail closed, never raise
            return zero(task, "bad_ground_truth")

    rel = relevance(prompt, text) if prompt else 1.0
    components = {"t4_if_frac": sum(results) / len(results), "t4_relevance": rel}
    exempt = any(c.get("type") in cfg.relevance_exempt for c in constraints)
    if cfg.relevance_check and not exempt:
        if _is_filler(text, cfg.filler_repetition_threshold):
            return zero(task, "irrelevant_content", flags=["filler_text"], components=components)
        if len(content_words(prompt)) >= cfg.min_prompt_content_words and rel < cfg.min_prompt_overlap:
            return zero(task, "irrelevant_content", components=components)

    if cfg.scoring == "fraction":
        score = components["t4_if_frac"]
        correct = all(results)
    else:
        correct = all(results)
        score = 1.0 if correct else 0.0
    return RewardResult(
        score=score,
        task=task,
        reason="ok" if correct else "constraint_failed",
        correct=correct,
        components=components,
    )
