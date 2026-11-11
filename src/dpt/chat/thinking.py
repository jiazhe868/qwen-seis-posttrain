"""Splitting a Qwen3.8 response into its thinking part and its final answer.

Format facts (from the Qwen3.8-27B ``chat_template.jinja``):

* With thinking enabled, the *generation prompt* already ends with
  ``"<|im_start|>assistant\\n<think>\\n"``. The sampled response therefore
  usually does **not** contain the opening ``<think>`` tag. It looks like
  ``"<reasoning>\\n</think>\\n\\n<final answer>"``.
* Rendered training conversations do contain both tags:
  ``"<think>\\n<reasoning>\\n</think>\\n\\n<answer>"``.

Rules that every grader and the eval harness follow (report §harness 规则):

1. Only the final answer is graded. Thinking is stripped before grading, and
   judges never see it (§防范原则: "reward 只看最终答案").
2. If the end-of-thinking marker is missing, the response is a **failure**.
   Usually the response was truncated mid-thought. We never grade raw
   thinking, because that would reward answers that only appear mid-reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass

THINK_START = "<think>"
THINK_END = "</think>"


@dataclass(frozen=True)
class ThinkSplit:
    thinking: str      # reasoning text without tags (stripped)
    answer: str        # final answer text after </think> (stripped); "" when has_end is False
    has_end: bool      # whether a </think> marker was found

    @property
    def gradable(self) -> bool:
        """A response is gradable only if thinking terminated and an answer follows."""
        return self.has_end and bool(self.answer)


def split_thinking(response: str) -> ThinkSplit:
    """Split ``response`` at the **last** ``</think>``.

    The last marker is used because a model can quote the tag inside its own
    reasoning. Text after the final marker is what the user sees.
    """
    idx = response.rfind(THINK_END)
    if idx < 0:
        thinking = response
        if thinking.lstrip().startswith(THINK_START):
            thinking = thinking.lstrip()[len(THINK_START):]
        return ThinkSplit(thinking=thinking.strip(), answer="", has_end=False)
    head, tail = response[:idx], response[idx + len(THINK_END):]
    head = head.lstrip()
    if head.startswith(THINK_START):
        head = head[len(THINK_START):]
    return ThinkSplit(thinking=head.strip(), answer=tail.strip(), has_end=True)


def strip_thinking(response: str) -> str | None:
    """Return the final answer, or ``None`` if the response has no end-of-thinking marker."""
    s = split_thinking(response)
    return s.answer if s.has_end else None
