"""D-Report with two judges: hosted reasoning-model requests, verdict cache, gate selection, regrade."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from dpt.common.io import write_jsonl
from dpt.eval.datasets import EvalItem
from dpt.eval.graders.pairwise import CachedJudge, HttpPairwiseJudge, JudgeUnavailable
from dpt.eval.harness import (
    Sample,
    SamplingSpec,
    SetSpec,
    build_pairwise_judge,
    run_suite,
    score_pairwise,
)

SPEC = SetSpec("d_report", "", "pairwise", 1)


class Prefers:
    """Judge that prefers the answer containing ``word`` (position-independent)."""

    def __init__(self, word: str):
        self.word = word

    def compare(self, q, a, b, rubric=None):
        if self.word in a and self.word not in b:
            return "A"
        if self.word in b and self.word not in a:
            return "B"
        return "tie"


def _items(n=6):
    return [EvalItem(f"r{i}", "d_report", f"q{i}", {"rubric": ["covers X"]}) for i in range(n)]


def _gens(items, texts):
    return {it.id: [Sample(f"think</think>{t}", "stop", 100)] for it, t in zip(items, texts)}


# --------------------------------------------------------------------------- HTTP judge


def test_reasoning_model_request_shape_and_truncation():
    seen = []

    def handler(req):
        body = json.loads(req.content)
        seen.append(body)
        fin = "length" if "truncate" in body["messages"][1]["content"] else "stop"
        content = "" if fin == "length" else "fine [[B]]"
        return httpx.Response(200, json={"model": "gpt-6-luna-2026-09-22",
                                         "choices": [{"message": {"content": content}, "finish_reason": fin}]})

    j = HttpPairwiseJudge("https://api.example.com/v1", "gpt-6-luna", temperature=None,
                          max_tokens=32768, max_tokens_field="max_completion_tokens",
                          extra_body={"reasoning_effort": "xhigh", "unused": None},
                          client=httpx.Client(transport=httpx.MockTransport(handler)))
    call = j.call("q", "a", "b")
    assert (call.verdict, call.parsed, call.served_model) == ("B", True, "gpt-6-luna-2026-09-22")
    body = seen[0]
    assert "temperature" not in body and "max_tokens" not in body and "unused" not in body
    assert body["max_completion_tokens"] == 32768 and body["reasoning_effort"] == "xhigh"
    cut = j.call("q", "truncate me", "b")
    assert (cut.verdict, cut.parsed) == ("tie", False)      # counted as a tie, flagged as unparsed


def test_rate_limit_is_retried():
    n = {"calls": 0}

    def handler(req):
        n["calls"] += 1
        if n["calls"] < 3:
            return httpx.Response(429, json={"error": "slow down"})
        return httpx.Response(200, json={"model": "m", "choices": [{"message": {"content": "[[A]]"}}]})

    j = HttpPairwiseJudge("http://j", "m", retry_delay_s=0.0,
                          client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert j.compare("q", "a", "b") == "A" and n["calls"] == 3


# --------------------------------------------------------------------------- cache


def test_cache_replays_and_refuses_unknown_pairs(tmp_path):
    path = tmp_path / "judgments.second.jsonl"
    live = CachedJudge(Prefers("good"), path, model="luna")
    assert live.compare("q", "good", "bad") == "A"
    assert live.compare("q", "good", "bad") == "A"
    assert live.stats["calls"] == 2 and live.stats["cached"] == 1
    replay = CachedJudge(None, path)                          # no endpoint here: cache only
    assert replay.compare("q", "good", "bad") == "A"
    with pytest.raises(JudgeUnavailable):
        replay.compare("q", "bad", "good")                    # order is part of the key
    other = CachedJudge(None, path, model="another-judge")    # another model's verdicts are not reused
    with pytest.raises(JudgeUnavailable):
        other.compare("q", "good", "bad")


# --------------------------------------------------------------------------- gate selection


def test_min_gate_reads_the_lower_judge_and_reports_agreement():
    items = _items()
    cand = _gens(items, ["good alpha", "good alpha", "good", "good", "meh", "meh"])
    base = _gens(items, ["meh"] * 6)
    res = score_pairwise(SPEC, items, cand, base, Prefers("good"), second_judge=Prefers("alpha"),
                         gate_judge="min")
    assert res["by_judge"]["primary"]["score"] == pytest.approx(100 * (4 + 0.5 * 2) / 6)   # 4 wins, 2 ties
    assert res["by_judge"]["second"]["score"] == pytest.approx(100 * (2 + 0.5 * 4) / 6)    # 2 wins, 4 ties
    assert res["score_judge"] == "second"
    assert res["score"] == res["by_judge"]["second"]["score"]
    assert res["judge_agreement"] == pytest.approx(4 / 6)     # r0, r1 win for both; r4, r5 tie for both
    assert res["per_item"][2]["by_judge"] == {"primary": "win", "second": "tie"}
    prim = score_pairwise(SPEC, items, cand, base, Prefers("good"), second_judge=Prefers("alpha"))
    assert prim["score_judge"] == "primary"


def test_min_gate_without_second_judge_fails_closed(tmp_path):
    items = _items(3)
    cand, base = _gens(items, ["good"] * 3), _gens(items, ["meh"] * 3)
    res = score_pairwise(SPEC, items, cand, base, Prefers("good"), gate_judge="min")
    assert res["score"] is None and res["status"] == "pending"
    assert res["by_judge"]["primary"]["score"] == 100.0       # still reported
    miss = CachedJudge(None, tmp_path / "none.jsonl")
    res = score_pairwise(SPEC, items, cand, base, Prefers("good"), second_judge=miss, gate_judge="min")
    assert res["score"] is None and "JudgeUnavailable" in res["by_judge"]["second"]["note"]
    with pytest.raises(ValueError):
        score_pairwise(SPEC, items, cand, base, Prefers("good"), gate_judge="mean")


def test_too_many_unparsed_verdicts_leave_no_score(tmp_path):
    def handler(req):
        return httpx.Response(200, json={"model": "m", "choices": [
            {"message": {"content": "thinking..."}, "finish_reason": "length"}]})

    http = HttpPairwiseJudge("http://j", "m", client=httpx.Client(transport=httpx.MockTransport(handler)))
    items = _items(4)
    res = score_pairwise(SPEC, items, _gens(items, ["good"] * 4), _gens(items, ["meh"] * 4),
                         CachedJudge(http, tmp_path / "j.jsonl", model="m"))
    assert res["score"] is None
    assert res["by_judge"]["primary"]["status"] == "judge_unparsed"
    assert res["by_judge"]["primary"]["judge_calls"]["unparsed_rate"] == 1.0


# --------------------------------------------------------------------------- building judges


def test_hosted_judge_skipped_without_key_or_network(monkeypatch):
    cfg = {"endpoint": "https://api.example.com", "model": "luna", "api_key_env": "DPT_TEST_KEY",
           "require_api_key": True}
    monkeypatch.delenv("DPT_TEST_KEY", raising=False)
    assert build_pairwise_judge(cfg) is None
    monkeypatch.setenv("DPT_TEST_KEY", "sk-test")

    def unreachable(*a, **k):
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(httpx, "get", unreachable)
    assert build_pairwise_judge({**cfg, "probe": True}) is None
    j = build_pairwise_judge({**cfg, "temperature": None, "max_tokens_field": "max_completion_tokens",
                              "extra_body": {"reasoning_effort": "xhigh"}})
    assert j is not None and j.temperature is None and j.extra_body == {"reasoning_effort": "xhigh"}
    assert build_pairwise_judge({"endpoint": None, "model": None}) is None


# --------------------------------------------------------------------------- end to end


def test_second_judge_pass_on_another_machine_via_cache(tmp_path):
    """The cluster run with only the primary judge, then a regrade where only the second is reachable."""
    items = [{"id": f"r{i}", "question": f"q{i}", "answer": {"rubric": ["covers X"]}} for i in range(4)]
    write_jsonl(tmp_path / "d_report.jsonl", items)
    cfg = {"sets": {"d_report": {"path": str(tmp_path / "d_report.jsonl"), "grader": "pairwise",
                                 "samples": 1}},
           "general": {}, "bootstrap": {"n_boot": 100, "seed": 0}, "max_concurrency": 4,
           "pairwise": {"gate_judge": "min"}}
    ref = tmp_path / "base"
    base_sample = {"text": "t</think>meh", "finish_reason": "stop", "n_tokens": 100}
    write_jsonl(ref / "d_report" / "generations.jsonl",
                [{"item_id": it["id"], "samples": [base_sample]} for it in items])
    texts = {"q0": "good alpha", "q1": "good", "q2": "good alpha", "q3": "meh"}

    async def gen(messages, spec, n, seed):
        return [Sample(f"t</think>{texts[messages[-1]['content']]}", "stop", 100)] * n

    def run(primary, second, regrade=False):
        return asyncio.run(run_suite(cfg, ["d_report"], generate=None if regrade else gen, model_name="sft",
                                     out_dir=tmp_path / "sft", spec=SamplingSpec(effort="medium", max_tokens=64),
                                     mode="completions", judge=primary, second_judge=second,
                                     reference_run=ref, regrade_only=regrade,
                                     verify_manifests=False))["sets"]["d_report"]

    first = run(Prefers("good"), None)                        # compute nodes: no route to the API
    assert first["score"] is None and first["by_judge"]["primary"]["score"] == pytest.approx(87.5)
    later = run(None, Prefers("alpha"), regrade=True)        # elsewhere: Gemma verdicts from the cache
    assert later["by_judge"]["primary"]["score"] == pytest.approx(87.5)
    assert later["by_judge"]["primary"]["judge_calls"]["cached"] == 8
    assert later["by_judge"]["second"]["score"] == pytest.approx(75.0)  # wins q0, q2; ties q1, q3
    assert later["score"] == pytest.approx(75.0) and later["score_judge"] == "second"


def test_report_keeps_both_judges_even_when_pending(tmp_path):
    from dpt.common.io import write_json
    from dpt.eval.report import build_summary, render_markdown

    res = {"model_name": "sft", "sets": {"d_report": {
        "score": None, "status": "pending", "gate_judge": "min", "judge_agreement": None,
        "by_judge": {"primary": {"score": 62.0}, "second": {"score": None, "status": "pending"}}}}}
    write_json(tmp_path / "sft" / "results.json", res)
    summary = build_summary({"sft": tmp_path / "sft"})
    row = summary["rows"]["sft"]
    assert "d_report" not in row                               # no gated number → the gate fails closed
    assert row["d_report_by_judge"]["scores"] == {"primary": 62.0, "second": None}
    md = render_markdown(summary)
    assert "sft: primary 62.0%, second pending" in md and "(min)" in md


def test_update_replaces_only_the_regraded_sets():
    from dpt.eval.harness import merge_into_existing

    old = {"model_name": "sft", "config_hash": "c0",
           "sets": {"d_num": {"score": 40.0}, "d_report": {"score": None, "status": "pending"}},
           "general": {"ifeval": {"score": 80.0}}, "data_hashes": {"d_num": "h1", "d_report": "h2"}}
    new = {"config_hash": "c1", "sets": {"d_report": {"score": 61.0}}, "general": {},
           "data_hashes": {"d_report": "h2"}}
    m = merge_into_existing(old, new)
    assert m["sets"]["d_num"] == {"score": 40.0} and m["general"]["ifeval"] == {"score": 80.0}
    assert m["sets"]["d_report"]["score"] == 61.0
    assert m["config_hash"] == "c0" and m["updates"] == [{"sets": ["d_report"], "config_hash": "c1"}]
    assert old["sets"]["d_report"]["score"] is None                       # input not mutated
