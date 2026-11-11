"""Prompt contract shared by RL export and the eval harness."""

from __future__ import annotations

import pytest

from dpt.data.prompts import PROMPT_VERSION, build_messages, messages_for_problem
from dpt.data.schema import Problem


def test_numeric_prompt_demands_single_boxed_answer():
    msgs = build_messages("t1_num", "  How long?  ")
    assert msgs[0]["role"] == "system" and "\\boxed{<number> <unit>}" in msgs[0]["content"]
    assert "range" in msgs[0]["content"]
    assert msgs[-1] == {"role": "user", "content": "How long?"}


def test_code_prompt_names_last_python_block_and_entry_point():
    msgs = build_messages("t2_code", "Write rms.", extra={"entry_point": "rms"})
    assert "```python" in msgs[0]["content"] and "last block" in msgs[0]["content"]
    assert msgs[-1]["content"].endswith("Implement the function `rms`.")


def test_report_prompt_requires_references_section():
    assert "References" in build_messages("t3_report", "Write a report.")[0]["content"]


def test_ifeval_has_no_system_prompt_and_math_does():
    assert build_messages("t4_general", "No commas.", extra={"kind": "ifeval"}) == [
        {"role": "user", "content": "No commas."}]
    assert build_messages("t4_general", "1+1?", extra={"kind": "math"})[0]["role"] == "system"
    with pytest.raises(ValueError):
        build_messages("t4_general", "?")


def test_messages_for_problem_reads_kind_and_entry_point():
    p = Problem.from_dict({"task": "t4_general", "question": "Haiku, no commas.",
                           "answer": {"kind": "ifeval", "constraints": [{"type": "no_commas"}]}})
    assert len(messages_for_problem(p)) == 1
    assert isinstance(PROMPT_VERSION, str)


def test_prompts_render_with_real_template_at_medium_effort():
    from dpt.chat.render import render_chat
    from tests.fakes import CharTokenizer

    text = render_chat(CharTokenizer(), build_messages("t1_num", "Q?"), effort="medium", add_generation_prompt=True)
    assert "Reasoning effort" not in text and text.endswith("<think>\n")


def test_t3_length_hint_reaches_training_prompts_like_eval_prompts():
    p = Problem.from_dict({"task": "t3_report", "question": "Explain induced seismicity.",
                           "answer": {"rubric": [{"criterion": "c", "weight": 1}],
                                      "reference_length": 700, "prompt_extra": {"length_hint": 550}}})
    msgs = messages_for_problem(p)
    assert msgs[-1]["content"].endswith("Aim for roughly 550 words.")
