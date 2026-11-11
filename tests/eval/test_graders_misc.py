"""Code, IFEval, multiple-choice, tool-call and refusal graders."""

from __future__ import annotations

import httpx
import pytest

from dpt.eval.graders.choice import grade_choice
from dpt.eval.graders.code import HiddenTestClient, LocalTestRunner, extract_code, grade_code
from dpt.eval.graders.ifeval import grade_ifeval
from dpt.eval.graders.refusal import grade_refusal, is_refusal, refusal_rate
from dpt.eval.graders.toolcall import grade_tool_calls, parse_tool_calls

TESTS = """
from solution import travel_time
def test_basic():
    assert abs(travel_time(120, 6) - 20) < 1e-9
def test_zero():
    assert travel_time(0, 6) == 0
"""
GOOD = "Here:\n```python\ndef travel_time(d, v):\n    return d / v\n```"


# ------------------------------------------------------------------------------------- code
def test_extract_code_prefers_last_python_block():
    t = "```python\nx=1\n```\ntext\n```\ny=2\n```\n```python\nz=3\n```"
    assert extract_code(t).strip() == "z=3"
    assert extract_code("no code") is None


def test_local_runner_all_pass():
    r = grade_code(GOOD, {"tests_source": TESTS}, LocalTestRunner(timeout_s=20))
    assert r.correct and r.details == {"passed": 2, "total": 2}


def test_local_runner_partial_fail():
    bad = "```python\ndef travel_time(d, v):\n    return d / v + (1 if d else 0)\n```"
    r = grade_code(bad, {"tests_source": TESTS}, LocalTestRunner(timeout_s=20))
    assert not r.correct and r.score == pytest.approx(0.5)


@pytest.mark.parametrize("hack", ["import sys\nsys.exit(0)", "import os\nos._exit(0)"])
def test_exit_hacks_fail_closed(hack):
    code = f"```python\n{hack}\ndef travel_time(d, v):\n    return 0\n```"
    r = grade_code(code, {"tests_source": TESTS}, LocalTestRunner(timeout_s=20))
    assert not r.correct and r.score == 0.0


def test_timeout_fails_closed():
    code = "```python\nwhile True:\n    pass\n```"
    r = grade_code(code, {"tests_source": TESTS}, LocalTestRunner(timeout_s=2))
    assert not r.correct and r.reason in ("runner_timeout", "runner_error")


def test_hidden_test_client_wire_contract_and_transport_error():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(req.content))
        return httpx.Response(200, json={"status": "ok", "passed": 3, "total": 3})

    client = HiddenTestClient("http://sandbox", client=httpx.Client(transport=httpx.MockTransport(handler)))
    r = grade_code(GOOD, {"hidden_tests_id": "h-1", "entry_point": "travel_time"}, client)
    assert r.correct and seen["hidden_tests_id"] == "h-1" and "code" in seen and seen["kind"] == "hidden"
    broken = HiddenTestClient(
        "http://sandbox", client=httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(500)))
    )
    r2 = grade_code(GOOD, {"hidden_tests_id": "h-1"}, broken)
    assert not r2.correct and r2.reason == "runner_transport_error"


# ------------------------------------------------------------------------------------- ifeval
def test_ifeval_strict_and_loose():
    gold = {
        "constraints": [
            {"type": "max_words", "value": 10},
            {"type": "lowercase"},
            {"type": "include_keywords", "keywords": ["wave"]},
        ]
    }
    assert grade_ifeval("a short note about a p wave", gold).correct
    r = grade_ifeval("Sure:\na short note about a p wave", gold)
    assert not r.correct and r.details["loose_correct"]  # preamble dropped in loose mode
    assert grade_ifeval("x", {"constraints": [{"type": "unknown_kind"}]}).reason == "constraint_failed"


def test_ifeval_json_and_bullets():
    assert grade_ifeval('```json\n{"a": 1}\n```', {"constraints": [{"type": "json_format"}]}).correct
    assert grade_ifeval("* a\n* b\n* c", {"constraints": [{"type": "num_bullets", "value": 3}]}).correct


# ------------------------------------------------------------------------------------- choice
@pytest.mark.parametrize(
    "ans,ok", [(r"\boxed{C}", True), ("Answer: (C)", True), ("C", True), ("Answer: B", False)]
)
def test_choice(ans, ok):
    assert grade_choice(ans, {"letter": "C", "n_options": 4}).correct is ok


def test_choice_multiple_fails():
    r = grade_choice(r"\boxed{A} or maybe \boxed{C}", {"letter": "C", "n_options": 4})
    assert r.reason == "multiple_answers"


# ------------------------------------------------------------------------------------- tool call
CALL = (
    "<tool_call>\n<function=epicenter>\n<parameter=lat>\n35.2\n</parameter>\n"
    "<parameter=net>\nIU\n</parameter>\n</function>\n</tool_call>"
)


def test_tool_call_parse_and_grade():
    assert parse_tool_calls(CALL) == [{"name": "epicenter", "arguments": {"lat": 35.2, "net": "IU"}}]
    assert grade_tool_calls(CALL, {"name": "epicenter", "arguments": {"lat": "35.2", "net": "IU"}}).correct
    assert not grade_tool_calls(CALL, {"name": "epicenter", "arguments": {"lat": 35.3, "net": "IU"}}).correct
    assert grade_tool_calls("no call", {"name": "x"}).reason == "no_tool_call"


# ------------------------------------------------------------------------------------- refusal
def test_refusal_detection():
    assert is_refusal("I'm sorry, but I can't help with that request.")
    assert is_refusal("抱歉，我无法提供这方面的帮助。")
    assert not is_refusal("The P-wave arrives after 20 s.")
    assert not is_refusal(None)
    assert refusal_rate(["I cannot help with that.", "20 s"]) == pytest.approx(50.0)
    assert grade_refusal("I won't help with this.", {"should_refuse": True}).correct
