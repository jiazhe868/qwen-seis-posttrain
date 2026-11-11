"""Exact (slow) reference for the gated delta rule, the recurrence inside Gated DeltaNet layers.

Purpose
-------
1. **Kernel parity on A100 (SM80).** Public validation of the GDN Triton kernels
   (flash-linear-attention) was done on H100/B200. The report's first work
   item is to verify them on A100. ``deltanet_kernel_check.py`` compares
   the kernels against this FP64 recurrence.
2. **Explaining Incident 1.** In the first long run, rollout kept the recurrent
   state in BF16 while training accumulated in FP32. Re-rounding the state to
   BF16 after every decode step adds an error that compounds with sequence
   length. This made the train/rollout log-ppl gap grow after ~12K tokens and
   eventually broke the run. ``state_dtype=torch.bfloat16`` reproduces that
   mechanism on CPU (see ``tests/parity/test_deltanet_reference.py``).

Recurrence (FLA convention, per head; state ``h`` has shape [K, V])::

    h'   = exp(g_t) · h_{t-1}                       # gated decay, g_t ≤ 0
    u_t  = β_t · (v_t − h'ᵀ k_t)                    # delta-rule correction
    h_t  = h' + k_t u_tᵀ                            # rank-1 update
    o_t  = h_tᵀ (scale · q_t)

with ``q``/``k`` L2-normalized (``use_qk_l2norm_in_kernel=True`` in Qwen3.5/3.8).
Written as ``S = hᵀ``, this is the Gated DeltaNet paper's
``S_t = S_{t-1} α_t (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ`` with ``α_t = exp(g_t)``.
"""

from __future__ import annotations

import math

import torch


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


@torch.no_grad()
def gated_delta_rule_reference(
    q: torch.Tensor,            # [B, T, H, K]
    k: torch.Tensor,            # [B, T, H, K]
    v: torch.Tensor,            # [B, T, H, V]
    g: torch.Tensor,            # [B, T, H]   log-decay (≤ 0)
    beta: torch.Tensor,         # [B, T, H]   in (0, 1)
    *,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,   # [B, H, K, V]
    compute_dtype: torch.dtype = torch.float64,
    state_dtype: torch.dtype | None = None,
    l2norm_qk: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-by-token recurrence. Returns ``(o [B,T,H,V], final_state [B,H,K,V])``.

    ``compute_dtype`` is the arithmetic precision of each step. ``state_dtype``,
    if given, is the precision the state is *stored* in between steps (the cache
    dtype of an inference engine). Setting it to bf16 emulates a BF16
    recurrent-state cache.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    scale = scale if scale is not None else 1.0 / math.sqrt(K)
    q, k, v, g, beta = (x.to(compute_dtype) for x in (q, k, v, g, beta))
    if l2norm_qk:
        q, k = l2norm(q), l2norm(k)
    h = (
        initial_state.to(compute_dtype).clone()
        if initial_state is not None
        else torch.zeros(B, H, K, V, dtype=compute_dtype, device=q.device)
    )
    o = torch.empty(B, T, H, V, dtype=compute_dtype, device=q.device)
    for t in range(T):
        h = h * torch.exp(g[:, t])[..., None, None]                       # decay
        kt, vt, bt = k[:, t], v[:, t], beta[:, t]
        pred = torch.einsum("bhkv,bhk->bhv", h, kt)                       # h'ᵀ k
        u = bt[..., None] * (vt - pred)                                    # β (v − h'ᵀk)
        h = h + torch.einsum("bhk,bhv->bhkv", kt, u)                      # rank-1 update
        if state_dtype is not None:
            h = h.to(state_dtype).to(compute_dtype)                        # cache round-trip
        o[:, t] = torch.einsum("bhkv,bhk->bhv", h, q[:, t] * scale)
    return o, h


def random_inputs(
    B: int = 1, T: int = 1024, H: int = 2, K: int = 32, V: int = 32, *,
    decay_min: float = 0.995, seed: int = 0, device: str = "cpu", dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Inputs with a realistic regime: slow decay (long memory) and moderate β."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(B, T, H, K, generator=gen)
    k = torch.randn(B, T, H, K, generator=gen)
    v = torch.randn(B, T, H, V, generator=gen)
    alpha = decay_min + (1 - decay_min) * torch.rand(B, T, H, generator=gen)
    g = torch.log(alpha)
    beta = torch.sigmoid(torch.randn(B, T, H, generator=gen))
    return {n: x.to(device=device, dtype=dtype) for n, x in dict(q=q, k=k, v=v, g=g, beta=beta).items()}


def error_by_position(o_test: torch.Tensor, o_ref: torch.Tensor, n_buckets: int = 4) -> list[float]:
    """Mean |error| in ``n_buckets`` equal position slices. Drift shows up in the late buckets first."""
    err = (o_test.double() - o_ref.double()).abs().mean(dim=(0, 2, 3))   # [T]
    return [float(chunk.mean()) for chunk in err.chunk(n_buckets)]
