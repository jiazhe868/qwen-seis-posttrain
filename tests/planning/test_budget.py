"""Every hardware number quoted in the report, recomputed from the architecture (report §推理显存的单位成本, §显存规划)."""

from __future__ import annotations

import pytest

from dpt.planning.budget import (
    GB,
    Arch,
    Cluster,
    Layout,
    comm_seconds,
    deltanet_state_bytes_per_seq,
    full_logits_bytes,
    kv_bytes_per_token,
    rollout_memory,
    seq_inference_bytes,
    sft_hours,
    train_memory_per_gpu,
    weights_bytes,
)

A, C, L = Arch(), Cluster(), Layout()


def test_weights_54gb():
    assert weights_bytes(A) / GB == pytest.approx(54.0)


def test_kv_64kb_per_token():
    assert kv_bytes_per_token(A) == 64 * 1024


def test_deltanet_state_154mb_fp32_78mb_bf16():
    assert deltanet_state_bytes_per_seq(A, 4) / 1e6 == pytest.approx(154, abs=1.5)
    assert deltanet_state_bytes_per_seq(A, 2) / 1e6 == pytest.approx(78, abs=1.5)


def test_16k_response_about_1p2gb():
    assert seq_inference_bytes(A, 16384) / GB == pytest.approx(1.2, abs=0.05)


def test_full_logits_16gb_trap():
    assert full_logits_bytes(A, 16384) / GB == pytest.approx(16.3, abs=0.1)


def test_training_memory_table():
    m = train_memory_per_gpu(A, L)
    assert m.params_bf16 / GB == pytest.approx(6.75)
    assert m.grads_fp32 / GB == pytest.approx(13.5)
    assert m.optimizer / GB == pytest.approx(20.25)
    assert m.total / GB == pytest.approx(61.5, abs=0.2)
    assert m.total / GB < 72, "must stay under the 72 GB per-rank alert line"


def test_rollout_concurrency_about_100_per_replica():
    r = rollout_memory(A, C, L)
    assert r.weights_per_gpu / GB == pytest.approx(27.0)
    assert r.kv_state_pool_per_replica / GB == pytest.approx(54, abs=2)
    assert r.per_seq_bytes / GB == pytest.approx(0.54, abs=0.02)
    assert 90 <= r.concurrent_per_replica <= 110
    assert 720 <= r.concurrent_total <= 880


def test_sft_22h_350_gpu_hours():
    s = sft_hours(A, C)
    assert s["flops"] == pytest.approx(1.36e20, rel=0.02)
    assert s["hours"] == pytest.approx(21.5, abs=1.0)
    assert s["gpu_hours"] == pytest.approx(345, abs=20)


def test_offload_volume_and_time():
    cm = comm_seconds(A, C, L)
    assert cm["offload_bytes_per_node_gb"] == pytest.approx(324, abs=1)
    assert cm["offload_one_way_s"] == pytest.approx(5.0, abs=0.5)
    assert 0.3 <= cm["dp_grad_sync_s"] <= 2.0
