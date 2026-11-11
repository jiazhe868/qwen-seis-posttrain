"""Per-task panels, gradient share, hack-rate tracker, divergence detector, audits and sinks."""

from __future__ import annotations

import json

import pytest

from dpt.common.io import read_jsonl
from dpt.monitoring.audit import AuditSampler, score_spotcheck
from dpt.monitoring.divergence import ProxyGoldDivergence
from dpt.monitoring.hack_rate import HackRateTracker
from dpt.monitoring.panels import gradient_share, memory_panel, per_task_panels, zero_adv_fraction
from dpt.monitoring.sinks import JsonlSink, MultiSink


def test_per_task_panels_split_everything_by_task():
    tasks = ["t1_num"] * 4 + ["t3_report"] * 4
    rewards = [1, 0, 1, 0, 0.6, 0.6, 0.6, 0.6]
    lens = [1000, 3000, 1000, 3000, 8000, 8000, 8000, 8000]
    correct = [True, False, True, False, True, True, True, True]
    groups = ["a", "a", "b", "b", "c", "c", "c", "c"]
    p = per_task_panels(
        tasks,
        rewards,
        lens,
        correct=correct,
        truncated=[0, 1, 0, 0, 0, 0, 0, 0],
        entropies=[1.0] * 8,
        group_ids=groups,
    )
    assert p["task/t1_num/reward_mean"] == 0.5
    assert p["task/t1_num/len_correct"] == 1000 and p["task/t1_num/len_incorrect"] == 3000
    assert p["task/t1_num/truncation_rate"] == 0.25
    assert p["task/t1_num/zero_adv_frac"] == 0.0 and p["task/t3_report/zero_adv_frac"] == 1.0
    assert "task/t3_report/len_incorrect" not in p


def test_zero_adv_fraction():
    assert zero_adv_fraction([1, 1, 0, 1], ["g1", "g1", "g2", "g2"]) == 0.5


def test_gradient_share_reproduces_token_dominance():
    """Report: T1 = 50% of prompts but only 27% of tokens under token-level loss."""
    tasks = ["t1_num"] * 50 + ["t2_code"] * 25 + ["t4_general"] * 25
    toks = [2000] * 50 + [6400] * 25 + [4600] * 25
    g = gradient_share(tasks, toks)
    assert g["task/t1_num/prompt_share"] == 0.5
    assert g["task/t1_num/token_share"] == pytest.approx(100_000 / 375_000)
    # Per-task normalization w_k/B_k rebalances the effective share.
    w = {"t1_num": 1 / 2000, "t2_code": 1 / 6400, "t4_general": 1 / 4600}
    g2 = gradient_share(tasks, toks, loss_weights=w)
    assert g2["task/t1_num/token_share"] == pytest.approx(0.5)


def test_gradient_share_counts_prompts_by_group():
    g = gradient_share(["t1_num"] * 8 + ["t2_code"] * 8, [1] * 16, group_ids=["a"] * 8 + ["b"] * 8)
    assert g["task/t1_num/prompt_share"] == 0.5


def test_memory_panel():
    assert memory_panel([60, 61, 77, 62]) == {
        "memory/peak_gb_max_rank": 77.0,
        "memory/peak_gb_argmax_rank": 2.0,
    }


def test_hack_rate_tracker_detects_exploding_category():
    """Step-310 hack: self-praise rose from 0.4% to 17%."""
    t = HackRateTracker(window_steps=5, growth_factor=2.0, min_rate=0.005, min_count=5)
    for s in range(5):
        t.update(s, [["self_praise"]] * 1 + [[]] * 249 + [["bare_except"]] * 2)
    for s in range(5, 10):
        m = t.update(s, [["self_praise"]] * 40 + [[]] * 210 + [["bare_except"]] * 2)
    assert m["hack/self_praise/rate"] == pytest.approx(40 / 252)
    grow = t.growing()
    assert [g.category for g in grow] == ["self_praise"]
    t2 = HackRateTracker(window_steps=5)
    t2.load_state_dict(json.loads(json.dumps(t.state_dict())))
    assert [g.category for g in t2.growing()] == ["self_praise"]


def test_divergence_detector_resets_when_gold_recovers():
    d = ProxyGoldDivergence(consecutive=3)
    seq = [(0.5, 0.4), (0.6, 0.4), (0.7, 0.45), (0.8, 0.44), (0.9, 0.43)]
    fired = [d.observe(i, p, g) for i, (p, g) in enumerate(seq)]
    assert fired == [False, False, False, False, False]  # gold rose at eval 3 → streak reset
    assert d.observe(5, 1.0, 0.42) is True


def test_audit_sampler(tmp_path):
    recs = [
        {"uid": f"u{i}", "task": "t1_num", "prompt": "p", "response": "r", "reward": float(i % 2), "step": 40}
        for i in range(300)
    ]
    a = AuditSampler(tmp_path, every=20, seed=1)
    assert not a.maybe_export(19, recs)
    paths = a.maybe_export(40, recs)
    beh, spot = [list(read_jsonl(p)) for p in paths]
    assert len(beh) == 100 and sum(r["reward"] == 1.0 for r in beh) == 50
    assert len(spot) == 100 and all(r["human_label"] is None for r in spot)
    assert [r["uid"] for r in read_jsonl(a.behavior_audit(40, recs))] == [
        r["uid"] for r in beh
    ]  # deterministic


def test_score_spotcheck_fp_fn():
    rows = (
        [{"reward": 1.0, "human_label": "correct"}] * 97
        + [{"reward": 1.0, "human_label": "incorrect"}] * 3
        + [{"reward": 0.0, "human_label": "incorrect"}] * 96
        + [{"reward": 0.0, "human_label": "correct"}] * 4
    )
    s = score_spotcheck(rows)
    assert s["fp_rate"] == pytest.approx(0.03) and s["fn_rate"] == pytest.approx(0.04)


def test_jsonl_sink_and_multisink_isolation(tmp_path):
    class Broken:
        def log(self, step, metrics):
            raise RuntimeError("dashboard down")

        def close(self):
            pass

    js = JsonlSink(tmp_path / "m.jsonl")
    ms = MultiSink([Broken(), js])
    ms.log(3, {"a": 1.0, "nan": float("nan")})
    ms.close()
    rows = list(read_jsonl(tmp_path / "m.jsonl"))
    assert rows == [{"step": 3, "a": 1.0, "nan": "nan"}]
