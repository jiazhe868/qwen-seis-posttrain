"""Canonical schema for instruction-following (IFEval-style) constraints.

The *data* defines constraints once. The training grader (``dpt.rewards.instruction``)
and the evaluation grader (``dpt.eval.graders.ifeval``) are deliberately independent
*implementations*, but they must read the same *schema*. Otherwise a constraint
written one way passes in training and silently fails in evaluation, or the
reverse. That is exactly the train/eval grader disagreement the report's
"第二层" checks exist to catch.

Canonical form: ``{"type": <name>, "value": <number|str|list>, "relation": <at_least|at_most|exactly|...>,
...extra}``. :func:`normalize_constraint` maps every known alias (other
benchmark vocabularies, the toy fixtures' ``bullet_count``/``n``) onto it.
Unknown types pass through unchanged, and both graders then fail closed on them.
"""

from __future__ import annotations

from typing import Any

CANONICAL_TYPES: frozenset[str] = frozenset({
    "words", "min_words", "max_words", "sentences", "paragraphs", "bullets", "include_keywords",
    "forbid_keywords", "keyword_frequency", "json_format", "end_with", "start_with", "all_lowercase",
    "all_uppercase", "language", "no_commas", "title_in_double_angular", "sections", "placeholders",
    "quotation", "postscript", "highlighted_sections",
})

# alias type → (canonical type, default relation or None)
_TYPE_ALIASES: dict[str, tuple[str, str | None]] = {
    "bullet_count": ("bullets", "exactly"),
    "num_bullets": ("bullets", "exactly"),
    "num_paragraphs": ("paragraphs", "exactly"),
    "num_sections": ("sections", "exactly"),
    "min_sentences": ("sentences", "at_least"),
    "max_sentences": ("sentences", "at_most"),
    "no_comma": ("no_commas", None),
    "lowercase": ("all_lowercase", None),
    "uppercase": ("all_uppercase", None),
    "exclude_keywords": ("forbid_keywords", None),
    "title": ("title_in_double_angular", None),
}
# parameter names that mean "value" in other vocabularies
_VALUE_ALIASES = ("n", "count", "num", "keywords", "text", "min_count", "words")


def normalize_constraint(c: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical form of one constraint (input is not modified)."""
    out = dict(c)
    t = str(out.get("type", ""))
    if t in _TYPE_ALIASES:
        canon, rel = _TYPE_ALIASES[t]
        out["type"] = canon
        if rel and "relation" not in out:
            out["relation"] = rel
    if "value" not in out:
        for k in _VALUE_ALIASES:
            if k in out:
                out["value"] = out.pop(k)
                break
    if out.get("type") == "keyword_frequency" and "max_count" in out and "min_count" not in c:
        out.setdefault("relation", "at_most")
        out.setdefault("value", out.pop("max_count"))
    return out


def normalize_constraints(cs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [normalize_constraint(c) for c in cs]
