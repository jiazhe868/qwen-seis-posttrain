"""Tool-calling benchmark grader: parse Qwen3.8 XML tool calls and compare them to gold calls.

Qwen3.8 emits tool calls in this format (from its chat template):

    <tool_call>
    <function=NAME>
    <parameter=ARG>
    VALUE
    </parameter>
    </function>
    </tool_call>

A sample is correct if the sequence of (name, arguments) pairs matches the
gold sequence exactly. Argument values are JSON-decoded when possible, so
``3`` matches ``"3"`` only when both decode to the same value.
"""

from __future__ import annotations

import json
import re
from typing import Any

from dpt.eval.graders import GradeResult

_CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.DOTALL)


def _decode(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return v.strip()
    return v


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    calls = []
    for name, body in _CALL_RE.findall(text):
        args = {k.strip(): _decode(v) for k, v in _PARAM_RE.findall(body)}
        calls.append({"name": name.strip(), "arguments": args})
    return calls


def grade_tool_calls(answer_text: str, gold: dict[str, Any]) -> GradeResult:
    want = gold["calls"] if "calls" in gold else [gold]
    want_norm = [
        {"name": c["name"], "arguments": {k: _decode(v) for k, v in c.get("arguments", {}).items()}}
        for c in want
    ]
    got = parse_tool_calls(answer_text)
    if not got:
        return GradeResult.fail("no_tool_call")
    ok = got == want_norm
    return GradeResult(
        correct=ok, score=float(ok), reason="ok" if ok else "call_mismatch", details={"pred": got}
    )
