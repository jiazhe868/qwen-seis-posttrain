"""Chat-template rendering with explicit reasoning-effort control and assistant-only loss masks.

Why this module exists
----------------------
The report names a wrong chat template as the most common cause of capability
dropping after SFT for no obvious reason. Its first SFT run lost
reasoning-effort control because the effort field was not rendered. All
training data, rollout prompts and eval prompts therefore go through
:func:`render_chat`, which

1. always passes ``reasoning_effort`` explicitly (never relies on the
   template default, which is ``xhigh``), and
2. verifies that the rendered text has the effort instruction the official
   template produces for that level (:func:`check_effort_rendering`).

How Qwen3.8 encodes effort (verified against ``chat_template.jinja``)
--------------------------------------------------------------------
With thinking on, the template prepends a sentence to the system turn:

* ``xhigh`` → "Reasoning effort is set to xhigh. Please think carefully …"
* ``low``   → "Reasoning effort is set to low. Keep your thinking brief …"
* ``medium`` → **no sentence at all**

So "medium" means "no effort sentence present". If a renderer forgets the
kwarg, the template silently inserts the *xhigh* sentence. Data meant to teach
medium-length reasoning then carries an xhigh label, and the model stops
respecting effort levels. The check below catches exactly that.

Loss mask
---------
Only assistant turns are trained, including their ``<think>`` sections and the
closing ``<|im_end|>``. Tool outputs are excluded. The template renders tool
results inside *user* turns (``<tool_response>…``), so an "assistant spans
only" rule excludes them automatically. Spans are found on the rendered string
and mapped to tokens via the fast tokenizer's offset mapping. This does not
depend on how the template re-renders prefixes, and it is exact for
multi-turn tool trajectories.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from dpt.common.types import check_effort

EFFORT_SENTENCE_PREFIX = "Reasoning effort is set to"
_EFFORT_RE = re.compile(re.escape(EFFORT_SENTENCE_PREFIX) + r" (\w+)\.")

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
ASSISTANT_HEADER = IM_START + "assistant\n"


class EffortRenderingError(ValueError):
    """Raised when rendered text does not carry the requested reasoning-effort signal."""


def expected_effort_sentence(effort: str) -> str | None:
    """The effort sentence the official template emits, or ``None`` for ``medium``."""
    check_effort(effort)
    return None if effort == "medium" else f"{EFFORT_SENTENCE_PREFIX} {effort}."


def check_effort_rendering(text: str, effort: str, *, enable_thinking: bool = True) -> None:
    """Assert that ``text`` carries exactly the effort signal for ``effort``.

    * thinking disabled → no effort sentence may appear
    * ``medium``        → no effort sentence may appear
    * ``low``/``xhigh`` → exactly one sentence, naming that level
    """
    found = _EFFORT_RE.findall(text)
    want = None if not enable_thinking else expected_effort_sentence(effort)
    if want is None:
        if found:
            raise EffortRenderingError(
                f"requested effort={effort!r} (thinking={enable_thinking}) but rendered text "
                f"contains effort sentence(s) for {found}; was reasoning_effort passed to the template?"
            )
        return
    if found != [effort]:
        raise EffortRenderingError(
            f"requested effort={effort!r} but rendered text has effort sentences {found}"
        )


def render_chat(
    tokenizer: Any,
    messages: Sequence[dict[str, Any]],
    *,
    effort: str,
    tools: Sequence[dict[str, Any]] | None = None,
    add_generation_prompt: bool = False,
    enable_thinking: bool = True,
    preserve_thinking: bool = True,
    check: bool = True,
) -> str:
    """Render ``messages`` with the model's own chat template and a verified effort level.

    ``tokenizer`` is a Hugging Face tokenizer (or anything exposing a compatible
    ``apply_chat_template``). Use ``add_generation_prompt=True`` for rollout and
    eval prompts. Use ``False`` for complete SFT conversations.
    """
    check_effort(effort)
    text = tokenizer.apply_chat_template(
        list(messages),
        tools=list(tools) if tools else None,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        reasoning_effort=effort,
        enable_thinking=enable_thinking,
        preserve_thinking=preserve_thinking,
    )
    if check:
        check_effort_rendering(text, effort, enable_thinking=enable_thinking)
    return text


def assistant_char_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of assistant turns, starting right after the role header.

    Each span ends just after the matching ``<|im_end|>``, so the stop token is
    trained. An unterminated final assistant turn runs to the end of the text.
    """
    spans: list[tuple[int, int]] = []
    pos = 0
    while True:
        start = text.find(ASSISTANT_HEADER, pos)
        if start < 0:
            break
        body_start = start + len(ASSISTANT_HEADER)
        end = text.find(IM_END, body_start)
        body_end = len(text) if end < 0 else end + len(IM_END)
        spans.append((body_start, body_end))
        pos = body_end
    return spans


def tokenize_with_assistant_mask(
    tokenizer: Any,
    messages: Sequence[dict[str, Any]],
    *,
    effort: str,
    tools: Sequence[dict[str, Any]] | None = None,
    max_length: int | None = None,
) -> dict[str, list[int]]:
    """Render and tokenize a full conversation; return ``input_ids`` and a 0/1 ``loss_mask``.

    A token is trained iff its character span lies inside an assistant turn.
    Raises if the conversation exceeds ``max_length``. Truncating a
    conversation would cut off the final answer and teach the model to stop
    mid-thought, so oversized samples are dropped upstream and never truncated.
    """
    text = render_chat(tokenizer, messages, effort=effort, tools=tools, add_generation_prompt=False)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids: list[int] = list(enc["input_ids"])
    offsets: list[tuple[int, int]] = [tuple(o) for o in enc["offset_mapping"]]
    spans = assistant_char_spans(text)

    loss_mask = [0] * len(input_ids)
    j = 0
    for i, (s, e) in enumerate(offsets):
        while j < len(spans) and spans[j][1] <= s:
            j += 1
        if j < len(spans) and s >= spans[j][0] and e <= spans[j][1] and e > s:
            loss_mask[i] = 1

    if max_length is not None and len(input_ids) > max_length:
        raise ValueError(f"conversation has {len(input_ids)} tokens > max_length={max_length}")
    if not any(loss_mask):
        raise ValueError("conversation has no assistant tokens to train on")
    return {"input_ids": input_ids, "loss_mask": loss_mask}
