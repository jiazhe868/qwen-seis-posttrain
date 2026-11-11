"""CPU tests for the gated-delta-rule reference, including the Incident 1 mechanism."""

from __future__ import annotations

import torch

from dpt.parity.deltanet_reference import error_by_position, gated_delta_rule_reference, random_inputs


def test_single_step_matches_closed_form():
    x = random_inputs(T=1, H=1, K=4, V=3, seed=1, dtype=torch.float64)
    o, h = gated_delta_rule_reference(**x, scale=1.0)
    k = x["k"][0, 0, 0] / x["k"][0, 0, 0].norm()
    q = x["q"][0, 0, 0] / x["q"][0, 0, 0].norm()
    v, b = x["v"][0, 0, 0], x["beta"][0, 0, 0]
    h_expected = torch.outer(k, b * v)          # from zero state, decay is irrelevant
    assert torch.allclose(h[0, 0], h_expected, atol=1e-10)
    assert torch.allclose(o[0, 0, 0], h_expected.T @ q, atol=1e-10)


def test_delta_rule_retrieves_stored_value():
    """With β=1 and no decay, writing (k, v) then querying k returns v exactly (delta-rule property)."""
    K = V = 8
    k = torch.zeros(1, 1, 1, K, dtype=torch.float64)
    k[..., 0] = 1
    v = torch.arange(V, dtype=torch.float64).view(1, 1, 1, V)
    g = torch.zeros(1, 1, 1, dtype=torch.float64)
    beta = torch.ones(1, 1, 1, dtype=torch.float64)
    o, _ = gated_delta_rule_reference(k, k, v, g, beta, scale=1.0)
    assert torch.allclose(o[0, 0, 0], v[0, 0, 0])


def test_bf16_state_cache_error_is_large_and_persistent_incident1():
    """Re-rounding the state to BF16 every step gives a per-token error ~1e4-1e5x an FP32 state's, and
    it never washes out: late positions are at least as wrong as early ones. In the full model this
    persistent error passes through 48 stacked GDN layers and the attention layers that read their
    outputs, and the train/rollout log-ppl gap grows with position (Incident 1: visible after ~12K tokens)."""
    x = random_inputs(T=1536, H=2, K=16, V=16, decay_min=0.998, seed=3, dtype=torch.float64)
    o_ref, _ = gated_delta_rule_reference(**x)
    o_fp32, _ = gated_delta_rule_reference(**x, state_dtype=torch.float32)
    o_bf16, _ = gated_delta_rule_reference(**x, state_dtype=torch.bfloat16)
    e32 = error_by_position(o_fp32, o_ref)
    e16 = error_by_position(o_bf16, o_ref)
    assert min(e16) > 100 * max(e32)
    assert e16[-1] >= 0.9 * e16[0], f"BF16-state error should persist along the sequence, got {e16}"
