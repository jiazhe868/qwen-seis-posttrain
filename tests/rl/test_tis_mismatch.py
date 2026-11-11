"""Tests for TIS weights, sequence masking, mismatch slicing and the parity probe (Incidents 1 & 3)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dpt.rl.mismatch import fixed_prompt_parity, mismatch_metrics
from dpt.rl.tis import apply_sequence_mask, sequence_mismatch_mask, tis_weights


def test_tis_weights_are_truncated_at_cap_and_zero_on_padding():
    train = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    roll = torch.tensor([[0.0, -0.5, -3.0, 1.0]])      # ratios 1, e^0.5, e^3, e^-1
    mask = torch.tensor([[1, 1, 1, 0]])
    w, m = tis_weights(train, roll, mask, cap=2.0)
    torch.testing.assert_close(w, torch.tensor([[1.0, float(np.exp(0.5)), 2.0, 0.0]]))
    assert m["tis/truncated_frac"] == pytest.approx(1 / 3)
    assert not w.requires_grad


def test_tis_rejects_bad_cap():
    with pytest.raises(ValueError):
        tis_weights(torch.zeros(1, 1), torch.zeros(1, 1), torch.ones(1, 1), cap=0)


def test_sequence_mask_drops_only_mismatched_sequences():
    train = torch.zeros(3, 5)
    roll = torch.zeros(3, 5)
    roll[1] -= 0.1                                    # whole sequence off by 0.1 nats/token
    roll[2, 0] = 1e-4                                 # tiny noise
    keep, stats = sequence_mismatch_mask(train, roll, torch.ones(3, 5), threshold=1e-2,
                                         statistic="abs_log_ppl_diff")
    assert keep.tolist() == [True, False, True]
    assert stats["tis/seq_masked_frac"] == pytest.approx(1 / 3)
    new_mask = apply_sequence_mask(torch.ones(3, 5), keep)
    assert new_mask[1].sum() == 0 and new_mask[0].sum() == 5


def test_late_position_drift_is_visible_only_in_position_slices():
    """Incident 1: BF16 recurrent-state error compounds with position; the global mean hides it."""
    B, T = 4, 16384
    torch.manual_seed(0)
    train = torch.randn(B, T) * 0.1 - 1.0
    roll = train.clone()
    roll[:, 12288:] += 0.02 * torch.randn(B, T - 12288).abs()   # drift only past 12K
    m = mismatch_metrics(train, roll, torch.ones(B, T))
    for early in ("0k-4k", "4k-8k", "8k-12k"):
        assert m[f"mismatch/pos/{early}/abs_log_ratio"] == 0.0
    # The global average dilutes the late drift by the fraction of affected tokens (1/4 here);
    # with realistic length distributions (most responses < 12K) the dilution is far stronger.
    late = m["mismatch/pos/12k+/abs_log_ratio"]
    assert late == pytest.approx(4 * m["mismatch/abs_log_ratio"], rel=1e-5) and late > 0.01
    assert m["mismatch/kl_k3"] >= 0


def test_task_slices_isolate_the_affected_task():
    """Incident 3 showed up only on T2 samples (stale prefix cache for the tool preamble)."""
    train = torch.full((4, 100), -1.0)
    roll = train.clone()
    roll[2:] += 0.014
    m = mismatch_metrics(train, roll, torch.ones(4, 100), task_ids=["t1_num", "t1_num", "t2_code", "t2_code"])
    assert m["mismatch/task/t1_num/log_ppl_abs_diff"] == 0.0
    assert m["mismatch/task/t2_code/log_ppl_abs_diff"] == pytest.approx(0.014, rel=1e-4)
    assert m["mismatch/log_ppl_abs_diff"] == pytest.approx(0.007, rel=1e-4)


def test_length_buckets_present():
    mask = torch.zeros(2, 20000)
    mask[0, :3000] = 1
    mask[1, :17000] = 1
    m = mismatch_metrics(torch.zeros(2, 20000), torch.zeros(2, 20000), mask)
    assert "mismatch/len/0k-4k/log_ppl_abs_diff" in m and "mismatch/len/16k+/log_ppl_abs_diff" in m


def test_parity_probe_passes_identical_and_blocks_on_stale_cache():
    ref = [np.full(50, -1.0) for _ in range(16)]
    assert fixed_prompt_parity(ref, [r.copy() for r in ref]).passed
    bad = [r.copy() for r in ref]
    bad[5] = bad[5] + 1.4e-2                          # the Incident-3 jump size
    res = fixed_prompt_parity(ref, bad)
    assert not res.passed and res.failing_prompts == [5]
    assert "FAIL" in res.summary()
    with pytest.raises(ValueError):
        fixed_prompt_parity(ref[:1], [np.zeros(3)])


def test_fresh_mask_separates_partial_rollout_samples():
    from dpt.rl.mismatch import fresh_sample_mask

    tags = [{"min_global_steps": 4, "max_global_steps": 4},      # fresh: all tokens from version 4
            {"min_global_steps": 3, "max_global_steps": 4},      # started on v3, resumed on v4
            {"min_global_steps": 3, "max_global_steps": 3},      # fully stale
            {"is_padding": True}, None]                          # untagged → fresh
    assert fresh_sample_mask(tags, version=4).tolist() == [True, False, False, True, True]


def test_drift_does_not_leak_into_engine_mismatch():
    """Stale samples carry the old policy's log-probs; only fresh ones may feed the Incident 1 alert."""
    import torch

    from dpt.rl.mismatch import fresh_sample_mask, mismatch_metrics

    train = torch.full((4, 8), -1.0)
    roll = train.clone()
    roll[2:] += 0.05                                     # policy moved by 0.05 nats/token since v3
    mask = torch.ones(4, 8)
    fresh = torch.as_tensor(fresh_sample_mask([{"min_global_steps": 4}] * 2 + [{"min_global_steps": 3}] * 2, 4))
    eng = mismatch_metrics(train[fresh], roll[fresh], mask[fresh])
    drift = mismatch_metrics(train[~fresh], roll[~fresh], mask[~fresh], prefix="drift")
    assert eng["mismatch/log_ppl_abs_diff"] == 0.0
    assert abs(drift["drift/log_ppl_abs_diff"] - 0.05) < 1e-6


def test_engine_mismatch_needs_enough_fresh_evidence():
    """Smoke-run false pause: 4 short fresh samples gave 8.8e-3; the alert must not see such a step."""
    import numpy as np
    import torch

    from dpt.rl.mismatch import fresh_stale_metrics

    B, T = 64, 512
    train = torch.full((B, T), -1.0)
    roll = train + 0.02 * torch.randn(B, T)
    mask = torch.zeros(B, T)
    mask[:, :T] = 1
    mask[:4, 20:] = 0                                    # the 4 fresh ones are short (20 tokens)
    few = np.zeros(B, dtype=bool)
    few[:4] = True
    m = fresh_stale_metrics(train, roll, mask, few, task_ids=["t4_general"] * B,
                            min_fresh_sequences=8, min_fresh_tokens=4096)
    assert m["mismatch/insufficient_fresh"] == 1.0 and "mismatch/log_ppl_abs_diff" not in m
    assert "drift/log_ppl_abs_diff" in m                 # stale samples are still reported

    many = np.zeros(B, dtype=bool)
    many[4:40] = True                                    # 36 full-length fresh samples
    m = fresh_stale_metrics(train, roll, mask, many, task_ids=["t1_num"] * B,
                            min_fresh_sequences=8, min_fresh_tokens=4096)
    assert m["mismatch/insufficient_fresh"] == 0.0 and m["mismatch/log_ppl_abs_diff"] < 5e-3
