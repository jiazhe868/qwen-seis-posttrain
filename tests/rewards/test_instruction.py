"""T4 instruction-following checks and the relevance gate (step-60 filler hack)."""

from __future__ import annotations

import pytest

from dpt.rewards.instruction import CHECKERS, InstructionConfig, grade_instruction, relevance

PROMPT = "Explain why seismic P waves travel faster than S waves in the crust. Use at least 30 words."


def test_passes_when_constraints_and_relevance_hold():
    ans = (
        "P waves are compressional and depend on both the bulk and shear moduli, while S waves depend only on "
        "the shear modulus, so in crustal rock P waves travel faster than S waves, typically by a factor of "
        "about 1.7 in velocity."
    )
    r = grade_instruction(ans, {"constraints": [{"type": "min_words", "value": 30}]}, prompt=PROMPT)
    assert r.score == 1.0 and r.correct


def test_filler_meets_word_count_but_scores_zero():
    """Regression for the step-60 hack: padding satisfied the length rule while IFEval did not move."""
    filler = "The weather is pleasant today and the garden looks lovely. " * 8
    r = grade_instruction(filler, {"constraints": [{"type": "min_words", "value": 30}]}, prompt=PROMPT)
    assert r.score == 0.0 and r.reason == "irrelevant_content"


def test_off_topic_text_fails_relevance():
    off = " ".join(f"word{i}" for i in range(40))
    assert relevance(PROMPT, off) == 0.0
    r = grade_instruction(off, {"constraints": [{"type": "min_words", "value": 30}]}, prompt=PROMPT)
    assert r.reason == "irrelevant_content"


@pytest.mark.parametrize(
    "ctype,c,text,ok",
    [
        ("bullets", {"value": 2}, "- a\n- b\n", True),
        ("bullets", {"value": 3}, "- a\n- b\n", False),
        ("include_keywords", {"value": ["fault", "slip"]}, "Fault slip was large.", True),
        ("forbid_keywords", {"value": ["tsunami"]}, "No tsunami was generated.", False),
        ("json_format", {}, '{"a": 1}', True),
        ("end_with", {"value": "Any questions?"}, "Done. Any questions?", True),
        ("all_lowercase", {}, "all lower case", True),
        ("language", {"value": "zh"}, "这是中文回答。", True),
        ("paragraphs", {"value": 2}, "one\n\ntwo", True),
        ("no_commas", {}, "a, b", False),
        ("title_in_double_angular", {}, "<<Title>>\nbody", True),
        ("keyword_frequency", {"keyword": "slip", "relation": "at_least", "value": 2}, "slip and slip", True),
    ],
)
def test_checkers(ctype, c, text, ok):
    assert CHECKERS[ctype](text, {"type": ctype, **c}) is ok


def test_binary_vs_fraction_scoring():
    gt = {"constraints": [{"type": "include_keywords", "value": ["fault"]}, {"type": "all_uppercase"}]}
    text = "fault mechanics of crustal p waves and s waves"
    assert grade_instruction(text, gt, prompt="").score == 0.0
    assert grade_instruction(text, gt, prompt="", cfg=InstructionConfig(scoring="fraction")).score == 0.5


def test_unknown_constraint_fails_closed():
    assert grade_instruction("x", {"constraints": [{"type": "nope"}]}).reason == "bad_ground_truth"
