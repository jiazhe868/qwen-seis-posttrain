"""Backends: OpenAI-compatible client (mocked transport) and deterministic fakes."""

from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from dpt.data.backends import (
    CachedGenerator,
    FakeEmbedder,
    FakeGenerator,
    FakeParaphraseJudge,
    JinjaCharTokenizer,
    LLMParaphraseJudge,
    OpenAICompatClient,
    VLLMGenerator,
    fake_grade,
)
from tests.fakes import TEMPLATE_PATH


def _client(handler):
    return OpenAICompatClient("http://srv/v1", "m", transport=httpx.MockTransport(handler), max_retries=1)


def test_chat_rejoins_parsed_reasoning_and_sends_effort():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, json={"choices": [
            {"index": 1, "message": {"content": "B"}},
            {"index": 0, "message": {"content": "A", "reasoning_content": "think"}}]})

    out = _client(handler).chat([{"role": "user", "content": "q"}], n=2, effort="medium")
    assert out == ["think\n</think>\n\nA", "B"]
    assert seen["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "medium"}


def test_completions_path_renders_prompt_locally():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, json={"choices": [{"index": 0, "text": "r\n</think>\n\nans"}]})

    gen = VLLMGenerator(_client(handler), tokenizer=JinjaCharTokenizer(TEMPLATE_PATH))
    out = gen([{"role": "user", "content": "q"}], n=1, effort="low", max_tokens=7, seed=3)
    assert out == ["r\n</think>\n\nans"]
    assert seen["prompt"].endswith("<|im_start|>assistant\n<think>\n")
    assert "Reasoning effort is set to low." in seen["prompt"] and seen["max_tokens"] == 7 and seen["seed"] == 3


def test_retry_on_server_error_then_success():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [3.0, 4.0]}]})

    import dpt.data.backends as B
    orig = B.time.sleep
    B.time.sleep = lambda s: None
    try:
        emb = _client(handler).embed(["x"])
    finally:
        B.time.sleep = orig
    assert calls["n"] == 2 and np.allclose(emb, [[0.6, 0.8]])


def test_url_file_waits_for_the_server_to_reopen_elsewhere(tmp_path, monkeypatch):
    import dpt.data.backends as B
    ep = tmp_path / "judge.url"
    ep.write_text("http://node01-001:8000\n")
    hosts, naps = [], []

    def nap(s):                              # while it waits, the window closes and reopens on another node
        naps.append(s)
        if len(naps) == 1:
            ep.unlink()
        elif len(naps) == 2:
            ep.write_text("http://node01-003:8000\n")

    monkeypatch.setattr(B.time, "sleep", nap)

    def handler(req):
        hosts.append(req.url.host)
        if req.url.host == "node01-001":
            raise httpx.ConnectError("connection refused", request=req)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [3.0, 4.0]}]})

    c = OpenAICompatClient("http://node01-001:8000/v1", "m", transport=httpx.MockTransport(handler), max_retries=0,
                           url_file=str(ep), wait_s=3600)
    assert np.allclose(c.embed(["x"]), [[0.6, 0.8]])
    assert hosts == ["node01-001", "node01-001", "node01-003"]       # no file between windows: the last address
    assert naps == [OpenAICompatClient.poll_s] * 2
    plain = OpenAICompatClient("http://node01-001:8000/v1", "m", transport=httpx.MockTransport(handler),
                               max_retries=1)                         # without url_file: the usual retries
    with pytest.raises(httpx.ConnectError):
        plain.embed(["x"])


def test_llm_judge_parses_final_answer_only():
    def handler(req):
        return httpx.Response(200, json={"choices": [{"index": 0, "message": {
            "content": "SAME", "reasoning_content": "they look DIFFERENT at first"}}]})

    assert LLMParaphraseJudge(_client(handler)).same_problem("a", "b")


def test_fakes_are_deterministic():
    g = FakeGenerator()
    msgs = [{"role": "user", "content": "q1"}]
    assert g(msgs, n=8, seed=1) == g(msgs, n=8, seed=1)
    outs = g(msgs, n=8, seed=1)
    assert all(fake_grade(None, o) in (0.0, 1.0) for o in outs)
    e = FakeEmbedder()
    a, b = e.embed(["wave speed 6 km/s", "wave speed 9 km/s"])
    assert np.isclose(a @ b, 1.0)  # numbers are masked
    assert FakeParaphraseJudge().same_problem("speed of 6 km/s wave", "speed of 7 km/s wave")


def test_local_embedder_pools_and_normalizes():
    import types

    import numpy as np
    import torch

    from dpt.data.backends import LocalEmbedder

    class Tok:
        def __call__(self, texts, **kw):
            ids = [[len(w) for w in t.split()] for t in texts]
            n = max(map(len, ids))
            pad = [[0] * (n - len(r)) + r for r in ids]                     # left padding
            mask = [[0] * (n - len(r)) + [1] * len(r) for r in ids]
            return types.SimpleNamespace(**{"input_ids": torch.tensor(pad), "attention_mask": torch.tensor(mask)},
                                         to=lambda dev: {"input_ids": torch.tensor(pad),
                                                         "attention_mask": torch.tensor(mask)})

    class Model(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            h = torch.stack([input_ids.float(), torch.ones_like(input_ids).float()], dim=-1)
            return types.SimpleNamespace(last_hidden_state=h)

    last = LocalEmbedder("x", pooling="last", device="cpu", model=Model(), tokenizer=Tok())
    v = last.embed(["a bb", "ccc"])
    assert v.shape == (2, 2) and np.allclose(np.linalg.norm(v, axis=1), 1.0)
    assert np.allclose(v[0], np.array([2, 1]) / np.sqrt(5)) and np.allclose(v[1], np.array([3, 1]) / np.sqrt(10))
    mean = LocalEmbedder("x", pooling="mean", device="cpu", model=Model(), tokenizer=Tok())
    assert np.allclose(mean.embed(["a bb"])[0], np.array([1.5, 1]) / np.linalg.norm([1.5, 1]))


def test_cached_generator_reuses_replies_across_runs(tmp_path):
    calls = []

    def gen(messages, **kw):
        calls.append((messages[0]["content"], kw))
        return [f"{messages[0]['content']}:{kw.get('seed')}:{i}" for i in range(kw.get("n", 1))]

    path = tmp_path / "cache" / "verify_judge.jsonl"
    msgs = [{"role": "user", "content": "q1"}]
    first = CachedGenerator(gen, path, "verify:judge:judge")
    a = first(msgs, n=2, seed=3)
    assert first(msgs, n=2, seed=3) == a and len(calls) == 1 and first.reused == 1
    first(msgs, n=2, seed=4)                       # another request: sampled
    assert len(calls) == 2
    with path.open("a") as f:                     # a torn last line from a killed run
        f.write('{"label": "verify:judge:judge", "key": "abc", "ou')
    again = CachedGenerator(gen, path, "verify:judge:judge")   # the next run reads the file
    assert len(again) == 2 and again(msgs, n=2, seed=3) == a and len(calls) == 2
    other = CachedGenerator(gen, path, "verify:teacher:teacher")   # another model: nothing reused
    other(msgs, n=2, seed=3)
    assert len(calls) == 3
