"""End-to-end harness tests against a mocked OpenAI-compatible server (httpx.MockTransport)."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path

import httpx
import pytest

from dpt.common.io import write_jsonl
from dpt.eval.datasets import EvalItem
from dpt.eval.harness import (
    GenerationCache,
    OpenAICompatBackend,
    Sample,
    SamplingSpec,
    build_messages,
    generate_set,
    grade_sample,
    merge_max_tokens,
    overall_mean_output_tokens,
    run_suite,
)
from dpt.eval.graders.numeric import grade_numeric

ITEMS = [
    {
        "id": "n1",
        "question": "Q-correct",
        "answer": {"value": 20, "unit": "s", "rel_tol": 0.01, "abs_tol": 0},
    },
    {"id": "n2", "question": "Q-wrong", "answer": {"value": 20, "unit": "s", "rel_tol": 0.01, "abs_tol": 0}},
    {
        "id": "n3",
        "question": "Q-truncated",
        "answer": {"value": 20, "unit": "s", "rel_tol": 0.01, "abs_tol": 0},
    },
]


def _server(calls: list):
    """Fake vLLM /v1/completions: behavior keyed on the rendered prompt."""

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        calls.append(body)
        p = body["prompt"]
        if "Q-correct" in p:
            text, fin = "t = d/v = 20\n</think>\n\n\\boxed{20 s}", "stop"
        elif "Q-wrong" in p:
            text, fin = "hmm\n</think>\n\n\\boxed{25 s}", "stop"
        else:
            text, fin = "thinking forever about 20 s " * 5, "length"
        return httpx.Response(
            200,
            json={
                "choices": [{"text": text, "finish_reason": fin} for _ in range(body["n"])],
                "usage": {"completion_tokens": 100 * body["n"]},
            },
        )

    return httpx.MockTransport(handler)


def _cfg(tmp: Path) -> dict:
    write_jsonl(tmp / "d_num.jsonl", ITEMS)
    return {
        "sets": {
            "d_num": {"path": str(tmp / "d_num.jsonl"), "grader": "numeric", "metric": "avg@k", "samples": 4}
        },
        "general": {},
        "bootstrap": {"n_boot": 200, "seed": 0},
        "max_concurrency": 4,
    }


def _backend(calls):
    return OpenAICompatBackend(
        "http://vllm",
        "m",
        mode="completions",
        renderer=lambda msgs, effort: f"<{effort}>" + msgs[-1]["content"],
        transport=_server(calls),
    )


def test_run_suite_scores_truncation_and_resume(tmp_path):
    cfg, calls = _cfg(tmp_path), []
    out = tmp_path / "run"

    async def go():
        be = _backend(calls)
        try:
            return await run_suite(
                cfg,
                ["d_num"],
                generate=be.generate,
                model_name="m",
                out_dir=out,
                spec=SamplingSpec(effort="medium", max_tokens=64),
                mode="completions",
                verify_manifests=False,
            )
        finally:
            await be.aclose()

    res = asyncio.run(go())["sets"]["d_num"]
    assert res["metric"] == "avg@4"
    assert res["score"] == pytest.approx(100 / 3)  # 1 of 3 items always right
    assert res["truncation_rate"] == pytest.approx(1 / 3)
    assert res["reasons"]["no_think_end"] == 4  # truncated item fails closed, never graded raw
    assert res["reasons"]["wrong_value"] == 4
    assert all(
        c["temperature"] == 1.0 and c["top_p"] == 1.0 and c["prompt"].startswith("<medium>") for c in calls
    )
    n_first = len(calls)
    asyncio.run(go())  # resume: everything cached → no new requests
    assert len(calls) == n_first


def test_per_item_seeds_are_deterministic_and_distinct(tmp_path):
    cfg, calls = _cfg(tmp_path), []

    async def go(out):
        be = _backend(calls)
        try:
            await run_suite(
                cfg,
                ["d_num"],
                generate=be.generate,
                model_name="m",
                out_dir=out,
                spec=SamplingSpec(max_tokens=64, seed=3),
                mode="completions",
                verify_manifests=False,
            )
        finally:
            await be.aclose()

    asyncio.run(go(tmp_path / "a"))
    asyncio.run(go(tmp_path / "b"))
    seeds_a = sorted(c["seed"] for c in calls[:3])
    seeds_b = sorted(c["seed"] for c in calls[3:])
    assert seeds_a == seeds_b and len(set(seeds_a)) == 3


def test_regrade_only_uses_cache_and_requires_it(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(RuntimeError):
        asyncio.run(
            run_suite(
                cfg,
                ["d_num"],
                generate=None,
                model_name="m",
                out_dir=tmp_path / "empty",
                spec=SamplingSpec(),
                mode="completions",
                regrade_only=True,
                verify_manifests=False,
            )
        )


def test_cache_ignores_different_spec(tmp_path):
    p = tmp_path / "g.jsonl"
    c1 = GenerationCache(p, "spec-a")
    asyncio.run(c1.put("x", [Sample("t", "stop", 1)]))
    assert "x" in GenerationCache(p, "spec-a").data
    assert "x" not in GenerationCache(p, "spec-b").data


def test_in_flight_requests_and_samples_are_bounded(tmp_path):
    # max_concurrency requests at most, and 16 × max_concurrency samples: pass@64 must not send every item at once
    items = [EvalItem(f"i{k}", "d_num", "q", {}) for k in range(40)]

    def peak(n: int) -> int:
        live, most = [0], [0]

        async def gen(messages, spec, n_, seed):
            live[0] += 1
            most[0] = max(most[0], live[0])
            await asyncio.sleep(0.001)
            live[0] -= 1
            return [Sample("t", "stop", 1)] * n_

        cache = GenerationCache(tmp_path / f"g{n}.jsonl", f"spec-{n}")
        asyncio.run(generate_set(gen, items, SamplingSpec(n=n), cache, max_concurrency=8))
        return most[0]

    assert [peak(n) for n in (1, 8, 32, 256)] == [8, 8, 4, 1]


def test_grade_sample_rules():
    it = EvalItem("i", "d_num", "q", {"value": 20, "unit": "s", "rel_tol": 0.01, "abs_tol": 0})
    g = lambda a, gold: grade_numeric(a, gold)  # noqa: E731
    assert grade_sample(Sample("x\n</think>\n\n20 s", "stop", 5), it, g).correct
    assert grade_sample(Sample("x 20 s", "length", 5), it, g).reason == "no_think_end"
    assert grade_sample(Sample("x</think>   ", "stop", 5), it, g).reason == "empty_answer"


def test_chat_mode_reconstructs_marker_only_when_finished():
    def handler(req):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"reasoning_content": "r", "content": "20 s"}, "finish_reason": "stop"},
                    {"message": {"reasoning_content": "r still", "content": ""}, "finish_reason": "length"},
                ],
                "usage": {"completion_tokens": 10},
            },
        )

    be = OpenAICompatBackend("http://x", "m", mode="chat", transport=httpx.MockTransport(handler))
    out = asyncio.run(be.generate([{"role": "user", "content": "q"}], SamplingSpec(), 2, 0))
    assert "</think>" in out[0].text and "</think>" not in out[1].text
    asyncio.run(be.aclose())


def test_server_errors_retry_then_raise():
    n = {"c": 0}

    def handler(req):
        n["c"] += 1
        return httpx.Response(503)

    be = OpenAICompatBackend(
        "http://x",
        "m",
        mode="completions",
        renderer=lambda m, e: "p",
        transport=httpx.MockTransport(handler),
        max_retries=1,
    )
    import dpt.eval.harness as H

    orig = H.asyncio.sleep
    H.asyncio.sleep = lambda s: orig(0)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(be.generate([{"role": "user", "content": "q"}], SamplingSpec(), 1, 0))
    finally:
        H.asyncio.sleep = orig
    assert n["c"] == 2


def test_merge_max_tokens_keeps_higher_and_records_both():
    a = {"sets": {"d_num": {"score": 55.3}}, "general": {}}
    b = {"sets": {"d_num": {"score": 57.0}}, "general": {}}
    m = merge_max_tokens(a, b, 32768, 65536)
    assert m["sets"]["d_num"]["score"] == 57.0
    assert m["sets"]["d_num"]["score_by_max_tokens"] == {"32768": 55.3, "65536": 57.0}
    assert m["sets"]["d_num"]["selected_max_tokens"] == 65536


def test_overall_mean_tokens_is_sample_weighted():
    r = {
        "sets": {
            "d_num": {"mean_output_tokens": 1000, "n_items": 10, "n_samples": 8},
            "d_code": {"mean_output_tokens": 4000, "n_items": 10, "n_samples": 2},
        }
    }
    assert overall_mean_output_tokens(r) == pytest.approx((1000 * 80 + 4000 * 20) / 100)


def test_build_messages_uses_shared_prompt_builder(monkeypatch):
    stub = types.ModuleType("dpt.data.prompts")
    stub.build_messages = lambda task, q, extra=None: [
        {"role": "system", "content": f"S-{task}-{extra}"},
        {"role": "user", "content": q},
    ]
    monkeypatch.setitem(sys.modules, "dpt.data.prompts", stub)
    it = EvalItem("i", "d_code", "write f", {"entry_point": "f"}, task="t2_code")
    msgs = build_messages(it)
    assert msgs[0]["content"] == "S-t2_code-{'entry_point': 'f'}"
    assert build_messages(EvalItem("j", "gpqa", "q?", {}))[0] == {"role": "user", "content": "q?"}


def test_pairwise_self_reference_is_50(tmp_path):
    from dpt.eval.harness import SetSpec, score_pairwise

    items = [EvalItem(f"r{i}", "d_report", "q", {}) for i in range(4)]
    gens = {it.id: [Sample("x</think>report", "stop", 10)] for it in items}
    res = score_pairwise(SetSpec("d_report", "", "pairwise", 1), items, gens, None, None, self_reference=True)
    assert res["score"] == 50.0
    pending = score_pairwise(SetSpec("d_report", "", "pairwise", 1), items, gens, None, None)
    assert pending["status"] == "pending"
