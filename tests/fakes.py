"""Test doubles shared across test packages.

``CharTokenizer`` renders conversations with the **real** Qwen3.8 chat
template (vendored at ``tests/fixtures/qwen38_chat_template.jinja``, Apache-2.0,
from the Qwen/Qwen3.8-27B repository) using the same Jinja settings as
Hugging Face ``transformers``, and tokenizes at character level with offsets.
That lets chat-rendering and loss-mask logic be tested against the true
template without downloading the 248K-entry tokenizer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

TEMPLATE_PATH = Path(__file__).parent / "fixtures" / "qwen38_chat_template.jinja"


def _raise(msg: str):
    raise jinja2.exceptions.TemplateError(msg)


class CharTokenizer:
    """One token per character; ids are code points. Mimics the HF tokenizer API we use."""

    def __init__(self, template_path: Path = TEMPLATE_PATH):
        env = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
        )
        env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False, **kw)
        env.globals["raise_exception"] = _raise
        self.template = env.from_string(template_path.read_text())

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False, **kwargs):
        assert tokenize is False
        return self.template.render(
            messages=messages, tools=tools, add_generation_prompt=add_generation_prompt, **kwargs
        )

    def __call__(self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False):
        out: dict[str, Any] = {"input_ids": [ord(c) for c in text]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out

    def decode(self, ids) -> str:
        return "".join(chr(i) for i in ids)
