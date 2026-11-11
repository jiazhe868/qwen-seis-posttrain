"""Tests for chat rendering, effort control and assistant-only loss masks (report §阶段一踩到的坑)."""

from __future__ import annotations

import pytest

from dpt.chat.render import (
    EffortRenderingError,
    check_effort_rendering,
    render_chat,
    tokenize_with_assistant_mask,
)
from dpt.chat.thinking import split_thinking, strip_thinking
from tests.fakes import CharTokenizer

TOK = CharTokenizer()
CONV = [
    {"role": "system", "content": "You are a geophysics assistant."},
    {"role": "user", "content": "P-wave speed 6 km/s, distance 120 km. Travel time?"},
    {"role": "assistant", "reasoning_content": "t = d / v = 120 / 6 = 20 s", "content": "20 s"},
]


@pytest.mark.parametrize("effort", ["low", "medium", "xhigh"])
def test_render_carries_requested_effort(effort):
    text = render_chat(TOK, CONV, effort=effort)
    if effort == "medium":
        assert "Reasoning effort is set to" not in text
    else:
        assert f"Reasoning effort is set to {effort}." in text


def test_template_default_is_xhigh_which_is_why_we_never_rely_on_it():
    """Regression for the SFT incident: omitting the kwarg silently yields the xhigh sentence."""
    text = TOK.apply_chat_template(CONV, tokenize=False)
    assert "Reasoning effort is set to xhigh." in text
    with pytest.raises(EffortRenderingError):
        check_effort_rendering(text, "medium")


def test_disabled_thinking_has_no_effort_sentence():
    text = render_chat(TOK, CONV, effort="xhigh", enable_thinking=False, add_generation_prompt=False)
    assert "Reasoning effort" not in text


def test_generation_prompt_opens_think_block():
    text = render_chat(TOK, CONV[:2], effort="medium", add_generation_prompt=True)
    assert text.endswith("<|im_start|>assistant\n<think>\n")


def test_loss_mask_covers_only_assistant_including_thinking():
    out = tokenize_with_assistant_mask(TOK, CONV, effort="medium")
    trained = "".join(chr(i) for i, m in zip(out["input_ids"], out["loss_mask"]) if m)
    assert trained == "<think>\nt = d / v = 120 / 6 = 20 s\n</think>\n\n20 s<|im_end|>"


def test_loss_mask_excludes_tool_outputs():
    conv = [
        {"role": "user", "content": "Compute the epicentral distance."},
        {"role": "assistant", "reasoning_content": "use tool", "content": "",
         "tool_calls": [{"function": {"name": "dist", "arguments": {"lat": 1}}}]},
        {"role": "tool", "content": "TOOL_SECRET_OUTPUT 111.2 km"},
        {"role": "assistant", "reasoning_content": "done", "content": "111.2 km"},
    ]
    tools = [{"type": "function", "function": {"name": "dist", "parameters": {}}}]
    out = tokenize_with_assistant_mask(TOK, conv, effort="medium", tools=tools)
    trained = "".join(chr(i) for i, m in zip(out["input_ids"], out["loss_mask"]) if m)
    assert "TOOL_SECRET_OUTPUT" not in trained
    assert "<function=dist>" in trained and trained.endswith("111.2 km<|im_end|>")


def test_split_thinking_handles_prompt_side_open_tag():
    s = split_thinking("reasoning here\n</think>\n\nAnswer: 20 s")
    assert s.has_end and s.thinking == "reasoning here" and s.answer == "Answer: 20 s"


def test_missing_end_marker_is_ungradable():
    assert strip_thinking("I am still thinking about 20 s ...") is None
    assert not split_thinking("<think>still going").gradable


def test_last_end_marker_wins():
    s = split_thinking("I might write </think> literally\n</think>\nfinal")
    assert s.answer == "final"
