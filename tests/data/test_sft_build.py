"""SFT mixture: admission, quality, rendering with explicit effort, 25% replay share, buckets."""

from __future__ import annotations

import pytest

from dpt.data.schema import SFTSample
from dpt.data.sft_build import (
    SFTBuildConfig,
    admission_reason,
    bucket_for,
    build_sft,
    padding_stats,
    select_self_distill,
)
from tests.fakes import CharTokenizer

TOK = CharTokenizer()
BASE = "Qwen/Qwen3.8-27B"


def _s(subset, user="question?", think="reasoning", ans="answer", effort="medium", **meta):
    return SFTSample.from_dict({"subset": subset, "reasoning_effort": effort, "meta": meta, "messages": [
        {"role": "user", "content": user}, {"role": "assistant", "reasoning_content": think, "content": ans}]})


def test_admission_rules():
    cfg = SFTBuildConfig()
    assert admission_reason(_s("domain_reasoning", verified=True), cfg) is None
    assert admission_reason(_s("domain_reasoning"), cfg) == "unverified_answer"
    assert admission_reason(_s("tool_traj", exec_ok=True, hidden_tests_passed=False), cfg)
    assert admission_reason(_s("self_distill", generated_by="other-model"), cfg) == "not_generated_by_base"
    assert admission_reason(_s("report"), cfg) == "missing_rubric"


def test_bucket_for():
    assert bucket_for(100, [4096, 8192]) == 4096 and bucket_for(5000, [4096, 8192]) == 8192
    assert bucket_for(9000, [4096, 8192]) is None


def test_padding_stats_bucketing_beats_shuffled_batches():
    lengths = [500, 520, 3900, 4000, 7000, 7100, 15000, 15500] * 10
    st = padding_stats(lengths, [4096, 8192, 16384, 32768], micro_batch_size=4, pad_to_multiple_of=128)
    assert st["bucketed_overhead"] < 0.08 < st["unbucketed_overhead"]
    assert sum(b["n"] for b in st["per_bucket"].values()) == len(lengths)


def test_self_distill_share_in_tokens():
    replay = [_s("self_distill", user=f"q{i}", generated_by=BASE) for i in range(100)]
    for r in replay:
        r.meta["n_tokens"] = 10
    chosen, toks = select_self_distill(300, replay, 0.25, seed=1)
    assert toks / (300 + toks) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        select_self_distill(300, replay, 1.0, seed=1)


def test_build_end_to_end_with_real_template():
    dom = [_s("domain_reasoning", user=f"domain q{i}", think="t" * 50, verified=True) for i in range(6)]
    replay = [_s("self_distill", user=f"general q{i}", think="g" * 50, generated_by=BASE) for i in range(20)]
    bad = [_s("domain_reasoning", user="loop", think=" ".join(["the same words repeat here"] * 30), verified=True),
           _s("domain_reasoning", user="unverified")]
    res = build_sft(dom + replay + bad, TOK, SFTBuildConfig(buckets=[4096], max_length=4096))
    subsets = {s.subset for s in res.samples}
    assert subsets == {"domain_reasoning", "self_distill"}
    assert abs(res.mixture["self_distill_token_share"] - 0.25) < 0.05
    reasons = {d["reason"].split(":")[0] for d in res.dropped}
    assert {"unverified_answer", "quality", "mixture"} <= reasons
    s0 = res.samples[0]
    assert s0.meta["bucket"] == 4096 and 0 < s0.meta["n_loss_tokens"] < s0.meta["n_tokens"]
    assert [r["stage"] for r in res.stage_stats] == ["loaded", "admission", "quality", "decontam",
                                                     "render_tokenize", "mixture"]


def test_overlong_samples_dropped_not_truncated():
    s = _s("domain_reasoning", think="x" * 5000, verified=True)
    res = build_sft([s], TOK, SFTBuildConfig(max_length=1000, buckets=[1024]))
    assert not res.samples and res.dropped[0]["reason"] == "too_long"


def test_effort_rendering_failure_is_dropped():
    class XhighDefaultTok(CharTokenizer):
        def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False, **kw):
            kw.pop("reasoning_effort", None)  # simulate the incident: effort field not rendered
            return super().apply_chat_template(messages, tools, tokenize, add_generation_prompt, **kw)

    s = _s("domain_reasoning", think="a careful derivation of the travel time", verified=True)
    res = build_sft([s], XhighDefaultTok(), SFTBuildConfig())
    assert not res.samples and res.dropped[0]["reason"].startswith("render_effort")
