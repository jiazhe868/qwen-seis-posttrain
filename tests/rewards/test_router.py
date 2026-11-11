"""verl entry points: fixed output schema, truncation, thinking rule, length control, batching."""

from __future__ import annotations

import json

import pytest

from dpt.rewards import router as R
from dpt.rewards.citations import OfflineResolver
from dpt.rewards.code_sandbox import SandboxResult
from dpt.rewards.router import VERL_KEYS, RewardRouter, compute_score, compute_score_batch, set_router


class FakeSandbox:
    def run(self, req):
        return SandboxResult(passed=3, total=4)


class FakeJudge:
    def score(self, question, answer, rubric):
        from dpt.rewards.judge import JudgeScore

        return JudgeScore(score=0.8, per_criterion=[8.0])


def full(answer: str) -> str:
    return "thinking...\n</think>\n\n" + answer


@pytest.fixture()
def router():
    r = RewardRouter({}, sandbox=FakeSandbox(), judge=FakeJudge(), resolver=OfflineResolver([]))
    set_router(r)
    yield r
    set_router(None)


SAMPLES = [
    ("t1_num", full(r"\boxed{20 s}"), json.dumps({"value": 20, "unit": "s"}), {"task": "t1_num"}),
    ("t2_code", full("```python\ndef f():\n    return 1\n```"), json.dumps({"hidden_tests_id": "h1"}), {}),
    (
        "t3_report",
        full("A thrust event with 3 m of slip."),
        json.dumps({"rubric": ["mechanism"], "reference_length": 50}),
        {"question": "Describe the event."},
    ),
    ("t4_general", full(r"\boxed{42}"), json.dumps({"kind": "math", "value": 42}), {}),
    (
        "t4_general",
        full("Fault slip explained in detail with words."),
        json.dumps({"kind": "ifeval", "constraints": [{"type": "include_keywords", "value": ["slip"]}]}),
        {"question": "Explain fault slip."},
    ),
]


def test_every_task_returns_exactly_the_contract_keys(router):
    """Ragged extra-info columns would crash (or misalign) verl's reward aggregation."""
    outs = [compute_score(*s) for s in SAMPLES]
    for o in outs:
        assert set(o) == set(VERL_KEYS)
    assert [o["score"] for o in outs] == pytest.approx([1.0, 0.75, 0.8, 1.0, 1.0])


def test_unknown_task_and_bad_ground_truth_fail_closed(router):
    assert compute_score("t9_x", full("1"), "{}")["reason"] == "unknown_task"
    assert compute_score("t1_num", full(r"\boxed{1}"), "not json")["reason"] == "bad_ground_truth"


def test_missing_think_end_is_zero(router):
    o = compute_score(
        "t1_num", r"\boxed{20 s}", json.dumps({"value": 20, "unit": "s"}), {"response_length": 10}
    )
    assert o["score"] == 0.0 and o["reason"] == "no_think_end"


def test_truncation_flag(router):
    gt = json.dumps({"value": 20, "unit": "s"})
    o = compute_score("t1_num", "still thinking", gt, {"response_length": 24576})
    assert o["truncated"] is True and o["reason"] == "truncated"
    o = compute_score(
        "t1_num", full(r"\boxed{20 s}"), gt, {"finish_reason": "length", "response_length": 100}
    )
    assert o["truncated"] is True


def test_overlong_penalty_applied(router):
    o = compute_score(
        "t1_num", full(r"\boxed{20 s}"), json.dumps({"value": 20, "unit": "s"}), {"response_length": 22528}
    )
    assert (
        o["base_score"] == 1.0
        and o["length_penalty"] == pytest.approx(-0.25)
        and o["score"] == pytest.approx(0.75)
    )


def test_apply_penalty_false_disables_overlong_penalty():
    """Under verl reward_manager=dapo the penalty is applied there; applying it here too double-penalizes."""
    r = RewardRouter({"length": {"apply_penalty": False}})
    res = r.grade("t1_num", full(r"\boxed{20 s}"), {"value": 20, "unit": "s"}, {"response_length": 24000})
    assert res.components["length_penalty"] == 0.0 and res.score == 1.0


def test_truncation_from_estimate_false_never_flags_from_char_estimate():
    """verl passes no response_length/finish_reason; a char estimate must not mis-flag truncation."""
    long_resp = full(r"\boxed{20 s}") + " " * (24576 * 4)  # estimate ≫ cap
    gt = {"value": 20, "unit": "s"}
    default = RewardRouter({})
    assert default.grade("t1_num", long_resp, gt, {}).truncated is True  # current default behaviour
    strict = RewardRouter({"length": {"truncation_from_estimate": False}})
    res = strict.grade("t1_num", long_resp, gt, {})
    assert res.truncated is False and res.reason == "ok"
    # a real token count or finish_reason still flags truncation
    assert strict.grade("t1_num", long_resp, gt, {"response_length": 24576}).truncated is True
    assert strict.grade("t1_num", long_resp, gt, {"finish_reason": "length"}).truncated is True


def test_hack_flags_are_reported_even_when_score_unaffected(router):
    o = compute_score(
        "t3_report",
        full("Slip was 3 m. 本分析已经过严格核验。"),
        json.dumps({"rubric": ["slip"], "reference_length": 100}),
        {"question": "q"},
    )
    assert "self_praise" in o["flags"].split(",")


def test_batch_matches_single(router):
    ds, sols, gts, extras = zip(*SAMPLES)
    batch = compute_score_batch(list(ds), list(sols), list(gts), list(extras))
    assert [b["score"] for b in batch] == [compute_score(*s)["score"] for s in SAMPLES]


def test_router_construction_failure_fails_closed(monkeypatch):
    set_router(None)
    monkeypatch.setattr(
        R, "load_rewards_config", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no cfg"))
    )
    o = compute_score("t1_num", full(r"\boxed{1}"), "{}")
    assert o["score"] == 0.0 and o["reason"] == "exception:RuntimeError"
    set_router(None)


def test_shipped_config_loads():
    cfg = R.load_rewards_config()
    assert cfg["length"]["max_response_tokens"] == 24576 and cfg["length"]["overlong_start"] == 20480
    assert cfg["length"]["apply_penalty"] is True and cfg["length"]["truncation_from_estimate"] is True
