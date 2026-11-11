"""D-Report pairwise judging: both orders, position-bias cancellation, version pinning, blind review."""

from __future__ import annotations

import json

import httpx
import pytest

from dpt.eval.graders.pairwise import (
    HttpPairwiseJudge,
    JudgeVersionError,
    PairResult,
    export_blind_review,
    import_blind_verdicts,
    judge_both_orders,
    judge_expert_agreement,
    length_matched,
    parse_verdict,
)
from dpt.eval.metrics import win_rate


class AlwaysA:
    """A maximally position-biased judge."""

    def compare(self, q, a, b, rubric=None):
        return "A"


class PrefersKeyword:
    def compare(self, q, a, b, rubric=None):
        return "A" if "good" in a and "good" not in b else ("B" if "good" in b and "good" not in a else "tie")


def test_position_bias_cancels_to_tie():
    r = judge_both_orders(AlwaysA(), "i1", "q", "cand", "base")
    assert r.outcome == "tie"


def test_consistent_preference_wins_and_loses():
    assert judge_both_orders(PrefersKeyword(), "i", "q", "good answer", "meh").outcome == "win"
    assert judge_both_orders(PrefersKeyword(), "i", "q", "meh", "good answer").outcome == "loss"


def test_base_vs_itself_is_50_percent():
    outcomes = [judge_both_orders(PrefersKeyword(), str(i), "q", "same", "same").outcome for i in range(10)]
    assert win_rate(outcomes) == pytest.approx(50.0)


def test_parse_verdict():
    assert parse_verdict("reasoning... [[B]]") == "B"
    assert parse_verdict("[[A]] then reconsider [[TIE]]") == "tie"
    assert parse_verdict("no verdict") == "tie"


def test_http_judge_and_version_pin():
    def handler(req):
        body = json.loads(req.content)
        assert body["temperature"] == 0.0
        assert "IGNORE any self-assessment" in body["messages"][0]["content"]
        return httpx.Response(
            200, json={"model": "judge-v2", "choices": [{"message": {"content": "ok [[A]]"}}]}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    j = HttpPairwiseJudge("http://judge", "judge", pinned_model_version="judge-v2", client=client)
    assert j.compare("q", "a", "b") == "A"
    j2 = HttpPairwiseJudge("http://judge", "judge", pinned_model_version="judge-v1", client=client)
    with pytest.raises(JudgeVersionError):
        j2.compare("q", "a", "b")


def test_length_matched_filter():
    rs = [PairResult("a", "win", "A", "B"), PairResult("b", "win", "A", "B")]
    kept = length_matched(rs, {"a": 1000, "b": 3000}, {"a": 1100, "b": 1000})
    assert [r.item_id for r in kept] == ["a"]


def test_blind_review_roundtrip():
    items = [{"id": f"r{i}", "question": f"q{i}", "answer": {"rubric": []}} for i in range(200)]
    cand = {f"r{i}": f"C{i}" for i in range(200)}
    base = {f"r{i}": f"B{i}" for i in range(200)}
    review, key = export_blind_review(items, cand, base, fraction=0.3, seed=7)
    assert 40 <= len(review) <= 80  # ~30% of 200
    assert all("C" not in r["blind_id"] for r in review)  # no identity leak in ids
    for r in review:  # expert always prefers the candidate's text
        r["verdict"] = "1" if r["response_1"].startswith("C") else "2"
    outcomes = import_blind_verdicts(review, key)
    assert set(outcomes.values()) == {"win"}
    assert judge_expert_agreement({k: "win" for k in outcomes}, outcomes) == 1.0
    # Deterministic subset: same seed → same items.
    review2, _ = export_blind_review(items, cand, base, fraction=0.3, seed=7)
    assert [r["blind_id"] for r in review] == [r["blind_id"] for r in review2]
