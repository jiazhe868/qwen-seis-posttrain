"""Tests for token-budget micro-batching and the activation-memory pre-flight (Incident 4)."""

from __future__ import annotations

import numpy as np
import pytest

from dpt.rl.microbatch import (
    DEFAULT_BUCKETS,
    ActivationMemoryModel,
    check_sequence_caps,
    make_count_microbatches,
    make_token_budget_microbatches,
    padded_length,
    padding_overhead,
    preflight,
)


def test_padded_length():
    assert padded_length(31000, buckets=DEFAULT_BUCKETS) == 32768
    assert padded_length(129, pad_multiple=128) == 256
    with pytest.raises(ValueError):
        padded_length(40000, buckets=DEFAULT_BUCKETS)


def test_token_budget_never_exceeded_and_covers_all_samples():
    rng = np.random.default_rng(0)
    lengths = rng.lognormal(mean=8.3, sigma=0.7, size=1024).clip(200, 32000).astype(int)
    mbs = make_token_budget_microbatches(lengths, 32768)
    assert all(m.padded_tokens <= 32768 for m in mbs)
    assert sorted(i for m in mbs for i in m.indices) == list(range(1024))


def test_sorting_cuts_padding_versus_unsorted_count_batches():
    rng = np.random.default_rng(1)
    lengths = rng.lognormal(mean=8.3, sigma=0.7, size=1024).clip(200, 32000).astype(int)
    budget = padding_overhead(make_token_budget_microbatches(lengths, 32768))
    count = padding_overhead(make_count_microbatches(lengths, 4))
    assert budget < 0.06 < count


def test_memory_model_matches_report_reference_point():
    mm = ActivationMemoryModel()
    assert mm.activation_gb([32768, 32768]) == pytest.approx(14.0, rel=1e-6)
    assert mm.peak_gb([32768, 32768]) == pytest.approx(61.6, abs=0.05)   # report: ≈ 61.6 GB


def test_preflight_rejects_incident4_plan_and_accepts_token_budget_plan():
    """Incident 4: a 31K trajectory padded to the 32K bucket inside a count-based micro-batch."""
    lengths = [31000, 5200, 4100, 6000, 3000, 2500, 7000, 4000]
    mm = ActivationMemoryModel()
    count_plan = make_count_microbatches(lengths, 4, buckets=DEFAULT_BUCKETS)
    rep = preflight(count_plan, mm, limit_gb=76.0)
    assert not rep.ok and "token budget" in rep.messages[0]
    budget_plan = make_token_budget_microbatches(lengths, 32768, buckets=DEFAULT_BUCKETS)
    assert preflight(budget_plan, mm, limit_gb=76.0).ok


def test_single_sample_over_budget_raises():
    with pytest.raises(ValueError):
        make_token_budget_microbatches([40000], 32768)


def test_sequence_cap():
    assert check_sequence_caps([6000, 1000], [27000, 2000], max_total=32768).tolist() == [0]


def test_max_seqs_limit():
    mbs = make_token_budget_microbatches([100] * 10, 10**6, max_seqs=3)
    assert [len(m.indices) for m in mbs] == [3, 3, 3, 1]
