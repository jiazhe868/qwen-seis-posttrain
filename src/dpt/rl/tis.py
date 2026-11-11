"""Truncated importance sampling (TIS) and sequence-level mismatch masking.

Why (Incident 1, ablation A4)
-----------------------------
The sampler (vLLM) and the trainer (Megatron) compute different probabilities
for the same token. Kernels, accumulation order and, here, the precision of
the DeltaNet recurrent state all differ. PPO's ratio uses the trainer's
``old_logp``, but the tokens were sampled from the *rollout* distribution, so
the update is silently off-policy. Partial rollout makes this worse: tokens
generated before a suspension come from an even older policy.

TIS reweights each token by the truncated likelihood ratio

    w_t = min( π_train_old(y_t) / π_rollout(y_t),  cap ),   cap = 2.0

``π_train_old`` is the trainer's recomputed log-prob at the start of the step.
``π_rollout`` is the log-prob the sampler reported. The cap bounds the
variance: a handful of tokens with ratios of 10–100 would otherwise dominate
the gradient. Without TIS that is what happened at step 180 of run 1, when
entropy exploded and outputs turned into multilingual garbage within 6 steps.
With TIS on (cap 2.0) the 250-step 27B run stayed stable.

Sequence-level masking is a second line of defense. If a whole sequence's
mean |log-ratio| is above a threshold, its tokens are not merely downweighted.
Their logprobs are untrustworthy (state corruption, stale prefix cache), so
the sequence is dropped from the loss (Incident 1 fix, "整条 mask 掉").
"""

from __future__ import annotations

import numpy as np
import torch


def tis_weights(
    train_old_logp: torch.Tensor,
    rollout_logp: torch.Tensor,
    mask: torch.Tensor,
    *,
    cap: float = 2.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Token-level truncated IS weights ``min(exp(old_train - rollout), cap)``; zero on padding.

    Only the upper side is truncated, as in TIS. Tokens the trainer finds
    *less* likely than the sampler did get weights < 1, which is safe because
    it only shrinks their contribution.
    """
    if cap <= 0:
        raise ValueError("cap must be positive")
    m = mask.to(torch.float32)
    log_ratio = (train_old_logp.float() - rollout_logp.float()).clamp(-20.0, 20.0)
    raw = torch.exp(log_ratio)
    w = torch.clamp(raw, max=cap) * m
    n = m.sum().clamp_min(1.0)
    metrics = {
        "tis/weight_mean": float((w.sum() / n)),
        "tis/raw_ratio_max": float((raw * m).max()) if m.any() else 0.0,
        "tis/truncated_frac": float(((raw > cap).float() * m).sum() / n),
    }
    return w.detach(), metrics


def sequence_mismatch_mask(
    train_old_logp: torch.Tensor,
    rollout_logp: torch.Tensor,
    mask: torch.Tensor,
    *,
    threshold: float,
    statistic: str = "mean_abs_log_ratio",
) -> tuple[np.ndarray, dict[str, float]]:
    """Return a boolean ``keep`` mask per sequence and stats.

    statistic:
      * ``mean_abs_log_ratio``: mean over tokens of ``|old_train - rollout|``
        (robust; catches symmetric noise)
      * ``abs_log_ppl_diff``: ``|mean(rollout) - mean(old_train)|``, the
        per-sequence log-ppl gap that the dashboard alert uses
    """
    m = mask.to(torch.float32)
    tok = m.sum(-1).clamp_min(1.0)
    diff = (rollout_logp.float() - train_old_logp.float()) * m
    if statistic == "mean_abs_log_ratio":
        stat = diff.abs().sum(-1) / tok
    elif statistic == "abs_log_ppl_diff":
        stat = (diff.sum(-1) / tok).abs()
    else:
        raise ValueError(f"unknown statistic {statistic!r}")
    has_tokens = m.sum(-1) > 0
    keep = ((stat <= threshold) | ~has_tokens).cpu().numpy()
    stats = {
        "tis/seq_masked_frac": float(1.0 - keep.mean()) if keep.size else 0.0,
        "tis/seq_stat_max": float(stat[has_tokens].max()) if has_tokens.any() else 0.0,
    }
    return keep, stats


def apply_sequence_mask(response_mask: torch.Tensor, keep: np.ndarray) -> torch.Tensor:
    """Zero out the loss mask of dropped sequences (they keep their slot in the batch)."""
    k = torch.as_tensor(keep, dtype=response_mask.dtype, device=response_mask.device)
    return response_mask * k.unsqueeze(-1)
