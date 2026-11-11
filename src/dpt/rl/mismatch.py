"""Training-vs-rollout mismatch metrics and the post-weight-sync parity probe.

The single most valuable signal of the project (report §长时间 RL 崩溃) is the
log-ppl gap between the trainer and the sampler on the *same* tokens. It
started rising 60 steps before the run-1 collapse, while reward looked fine
until the last 6 steps. Thresholds (configs/rl/monitoring.yaml):
normal < 5e-4, alert > 2e-3, auto-pause > 5e-3.

Definitions, for a sequence with response tokens ``y_1..y_T``:

    log_ppl_train   = -(1/T) Σ_t log π_train(y_t)
    log_ppl_rollout = -(1/T) Σ_t log π_rollout(y_t)
    log_ppl_diff    = log_ppl_train - log_ppl_rollout = mean_t(rollout - train)

The headline metric ``log_ppl_abs_diff`` is the mean over sequences of
``|log_ppl_diff|``. Taking the absolute value per sequence stops opposite-sign
errors on different sequences from cancelling.

KL estimators (samples from the rollout policy q, trainer p, x = log q − log p):
    k1 = x,  k2 = x²/2,  k3 = exp(-x) − 1 + x  (unbiased, always ≥ 0)

Slicing matters. Incident 1's drift (BF16 DeltaNet state on the rollout side,
FP32 accumulation on the trainer side) grows with position because the
recurrent-state error compounds. It was obvious past token 12K and invisible
in the global average. Incident 3 appeared only on T2 samples (the
prefix-cached tool preamble). So every metric is also reported per task, per
position bucket and per response-length bucket.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

DEFAULT_POSITION_EDGES: tuple[int, ...] = (4096, 8192, 12288)
DEFAULT_LENGTH_EDGES: tuple[int, ...] = (4096, 8192, 16384)


def _bucket_names(edges: Sequence[int]) -> list[str]:
    names, lo = [], 0
    for e in edges:
        names.append(f"{lo // 1024}k-{e // 1024}k")
        lo = e
    names.append(f"{lo // 1024}k+")
    return names


def _core_metrics(train: torch.Tensor, rollout: torch.Tensor, m: torch.Tensor) -> dict[str, float]:
    tok = m.sum(-1)
    has = tok > 0
    if not has.any():
        return {}
    x = (rollout - train) * m                       # log q - log p on valid tokens
    seq_diff = x.sum(-1)[has] / tok[has]           # per-sequence log-ppl diff
    n = m.sum()
    return {
        "log_ppl_diff": float(seq_diff.mean()),
        "log_ppl_abs_diff": float(seq_diff.abs().mean()),
        "kl_k1": float(x.sum() / n),
        "kl_k2": float((0.5 * x**2 * m).sum() / n),
        "kl_k3": float(((torch.exp(-x) - 1 + x) * m).sum() / n),
        "abs_log_ratio": float((x.abs()).sum() / n),
    }


def fresh_sample_mask(tags: Sequence[object], version: int) -> np.ndarray:
    """True for samples generated entirely by policy ``version`` (the weights that computed old_log_probs).

    With partial rollout (verl ``colocate_async``) a sample may have been started by an older
    version and finished by a newer one; verl tags it with ``min_global_steps``/``max_global_steps``.
    Only fresh samples measure engine mismatch; for the others the rollout log-probs also contain
    the policy's own change since then. Untagged samples (no partial rollout) count as fresh.
    """
    def start(t: object) -> int:
        return int(t.get("min_global_steps", version)) if isinstance(t, dict) else version
    return np.array([start(t) == version for t in tags], dtype=bool)


def mismatch_metrics(
    train_logp: torch.Tensor,
    rollout_logp: torch.Tensor,
    mask: torch.Tensor,
    *,
    task_ids: Sequence[str] | None = None,
    position_edges: Sequence[int] = DEFAULT_POSITION_EDGES,
    length_edges: Sequence[int] = DEFAULT_LENGTH_EDGES,
    prefix: str = "mismatch",
) -> dict[str, float]:
    """All mismatch metrics, globally and sliced. Keys look like ``mismatch/log_ppl_abs_diff``,
    ``mismatch/task/t2_code/log_ppl_abs_diff``, ``mismatch/pos/12k+/abs_log_ratio``,
    ``mismatch/len/16k+/log_ppl_abs_diff``.
    """
    train = train_logp.float()
    roll = rollout_logp.float()
    m = mask.float()
    out = {f"{prefix}/{k}": v for k, v in _core_metrics(train, roll, m).items()}

    if task_ids is not None:
        tids = np.asarray(task_ids)
        for t in sorted(set(task_ids)):
            sel = torch.as_tensor(tids == t)
            for k, v in _core_metrics(train[sel], roll[sel], m[sel]).items():
                out[f"{prefix}/task/{t}/{k}"] = v

    # Position buckets: token-level statistics restricted to response positions in [lo, hi).
    T = m.shape[-1]
    pos = torch.arange(T).unsqueeze(0).expand_as(m)
    lo = 0
    for name, hi in zip(_bucket_names(position_edges), list(position_edges) + [10**12]):
        pm = m * ((pos >= lo) & (pos < hi)).float()
        n = pm.sum()
        if n > 0:
            x = (roll - train) * pm
            out[f"{prefix}/pos/{name}/log_ratio"] = float(x.sum() / n)
            out[f"{prefix}/pos/{name}/abs_log_ratio"] = float(x.abs().sum() / n)
        lo = hi

    # Response-length buckets: sequence-level statistics for sequences whose length falls in a bucket.
    lengths = m.sum(-1)
    lo = 0
    for name, hi in zip(_bucket_names(length_edges), list(length_edges) + [10**12]):
        sel = (lengths > lo) & (lengths <= hi) if lo > 0 else (lengths > 0) & (lengths <= hi)
        if sel.any():
            for k, v in _core_metrics(train[sel], roll[sel], m[sel]).items():
                if k in ("log_ppl_abs_diff", "abs_log_ratio"):
                    out[f"{prefix}/len/{name}/{k}"] = v
        lo = hi
    return out


def fresh_stale_metrics(
    train_logp: torch.Tensor,
    rollout_logp: torch.Tensor,
    mask: torch.Tensor,
    fresh: np.ndarray,
    *,
    task_ids: Sequence[str],
    position_edges: Sequence[int] = DEFAULT_POSITION_EDGES,
    length_edges: Sequence[int] = DEFAULT_LENGTH_EDGES,
    min_fresh_sequences: int = 1,
    min_fresh_tokens: float = 0,
) -> dict[str, float]:
    """Engine mismatch on fresh samples (``mismatch/*``), policy drift on stale ones (``drift/*``).

    ``fresh`` comes from :func:`fresh_sample_mask`. ``mismatch/*`` is emitted only when the fresh
    samples carry enough evidence (``min_fresh_sequences`` and ``min_fresh_tokens``). Otherwise the
    alert skips the step and ``mismatch/insufficient_fresh`` = 1. With partial rollout the fresh
    samples of a step can be a handful of short responses, whose per-sequence means are noisy.
    """
    fresh = np.asarray(fresh, dtype=bool)
    n = len(fresh)
    n_fresh_tok = float(mask[torch.as_tensor(fresh)].sum()) if fresh.any() else 0.0
    enough = int(fresh.sum()) >= int(min_fresh_sequences) and n_fresh_tok >= float(min_fresh_tokens)
    out = {"mismatch/fresh_sample_frac": float(fresh.sum() / max(1, n)),
           "mismatch/insufficient_fresh": 0.0 if enough else 1.0}
    for sel, prefix in ((fresh, "mismatch"), (~fresh, "drift")):
        if not sel.any() or (prefix == "mismatch" and not enough):
            continue
        st = torch.as_tensor(sel)
        out.update(mismatch_metrics(train_logp[st], rollout_logp[st], mask[st],
                                    task_ids=[t for t, k in zip(task_ids, sel) if k],
                                    position_edges=position_edges, length_edges=length_edges, prefix=prefix))
    return out


# ----------------------------------------------------------------------------------------------
# Post-weight-sync parity probe (Incident 3)
# ----------------------------------------------------------------------------------------------
@dataclass
class ParityResult:
    passed: bool
    mean_abs_diff: float                       # mean over prompts of per-prompt mean |Δlogp|
    max_prompt_mean_abs_diff: float            # worst prompt
    max_token_abs_diff: float
    failing_prompts: list[int] = field(default_factory=list)
    details: list[float] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[parity {status}] mean|Δ|={self.mean_abs_diff:.2e} worst-prompt={self.max_prompt_mean_abs_diff:.2e} "
            f"max-token={self.max_token_abs_diff:.2e} failing={self.failing_prompts}"
        )


def fixed_prompt_parity(
    trainer_logps: Sequence[Sequence[float]],
    rollout_logps: Sequence[Sequence[float]],
    *,
    prompt_mean_tol: float = 2e-3,
    token_max_tol: float = 0.5,
) -> ParityResult:
    """Compare per-token logprobs of N fixed (prompt, response) pairs scored on both sides.

    Run right after every weight sync, before any rollout. The trainer scores
    16 fixed prompt+response pairs, and the freshly synced vLLM engine scores
    the same tokens (``prompt_logprobs``). The check fails if any prompt's
    mean |Δ| exceeds ``prompt_mean_tol`` (the dashboard alert level, 2e-3) or
    any single token differs by more than ``token_max_tol`` nats. A failure
    blocks the rollout. Likely causes: an un-flushed prefix cache holding
    stale KV / DeltaNet-state snapshots (Incident 3), a resharding bug in the
    TP4×PP2 → TP2 weight conversion, or a wrong dtype for the recurrent state.
    """
    if len(trainer_logps) != len(rollout_logps):
        raise ValueError("trainer and rollout must score the same number of prompts")
    per_prompt, failing, max_tok = [], [], 0.0
    for i, (a, b) in enumerate(zip(trainer_logps, rollout_logps)):
        a_, b_ = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        if a_.shape != b_.shape:
            raise ValueError(f"prompt {i}: length mismatch {a_.shape} vs {b_.shape} (tokenization drift?)")
        d = np.abs(a_ - b_)
        pm = float(d.mean()) if d.size else 0.0
        tm = float(d.max()) if d.size else 0.0
        per_prompt.append(pm)
        max_tok = max(max_tok, tm)
        if pm > prompt_mean_tol or tm > token_max_tol:
            failing.append(i)
    return ParityResult(
        passed=not failing,
        mean_abs_diff=float(np.mean(per_prompt)) if per_prompt else 0.0,
        max_prompt_mean_abs_diff=float(np.max(per_prompt)) if per_prompt else 0.0,
        max_token_abs_diff=max_tok,
        failing_prompts=failing,
        details=per_prompt,
    )
