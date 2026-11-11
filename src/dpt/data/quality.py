"""Cheap quality filters applied before anything expensive.

The motivating failure (report §阶段一踩到的坑, "Loss spike"): a loss spike near
SFT step 400 was traced to a few dozen long samples where OCR had failed and
the same lines repeated many times. A long, highly repetitive sample has
abnormally low per-token loss on its repeats and abnormal gradients on its
first occurrence. The fix that stuck is a 4-gram repetition filter: drop the
sample if more than 30% of its 4-gram occurrences repeat an earlier 4-gram.

Why 0.30. Normal technical prose and long CoT score roughly 0.02–0.15 on this
metric. Code is somewhat higher because of boilerplate, but well below 0.3.
OCR loops score 0.6–0.95. 0.30 sits in the wide empty gap, so it is
insensitive to the exact value. Check the histogram in the stage stats after
every new source anyway.

The same checks run on RL questions, which should not contain OCR garbage
either, and on SFT assistant turns.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

from dpt.common.text import ngram_repetition_ratio

REPLACEMENT_CHAR = "�"


@dataclass
class QualityConfig:
    ngram_n: int = 4
    max_ngram_repetition: float = 0.30
    min_chars: int = 20
    max_chars: int = 200_000
    max_replacement_char_frac: float = 0.01   # U+FFFD = undecodable bytes from OCR / bad encoding
    min_repetition_tokens: int = 50           # repetition ratio is noise on very short texts

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "QualityConfig":
        return cls(**(d or {}))


def quality_issues(text: str, cfg: QualityConfig) -> list[str]:
    """Return a list of issue codes; an empty list means the text passes."""
    issues: list[str] = []
    stripped = text.strip()
    if len(stripped) < cfg.min_chars:
        issues.append("too_short")
    if len(stripped) > cfg.max_chars:
        issues.append("too_long")
    if stripped and stripped.count(REPLACEMENT_CHAR) / len(stripped) > cfg.max_replacement_char_frac:
        issues.append("encoding_garbage")
    if len(stripped.split()) >= cfg.min_repetition_tokens or len(stripped) >= 4 * cfg.min_repetition_tokens:
        if ngram_repetition_ratio(stripped, cfg.ngram_n) > cfg.max_ngram_repetition:
            issues.append(f"repetition_{cfg.ngram_n}gram")
    return issues


@dataclass
class FilterResult:
    kept: list[Any]
    dropped: list[dict[str, Any]] = field(default_factory=list)  # {"uid", "reason"}
    reasons: Counter = field(default_factory=Counter)


def filter_by_quality(
    items: Iterable[Any], text_fn, cfg: QualityConfig, uid_fn=lambda x: x.uid
) -> FilterResult:
    """Generic filter. ``text_fn(item) -> str`` selects what to check (question, assistant turns, …)."""
    res = FilterResult(kept=[])
    for it in items:
        issues = quality_issues(text_fn(it), cfg)
        if issues:
            res.dropped.append({"uid": uid_fn(it), "reason": ",".join(issues)})
            for i in issues:
                res.reasons[i] += 1
        else:
            res.kept.append(it)
    return res
