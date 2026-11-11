"""Token-budget micro-batching, length bucketing, and an activation-memory pre-flight check.

Constraints from the model and stack
------------------------------------
* **No packed sequences.** Megatron's Gated DeltaNet path does not reset the
  recurrent state at packed-sequence boundaries, so sequences must be laid
  out padded (BSHD). Every padding token costs real compute and memory.
* **Length bucketing.** Sorting by length before batching puts similar
  lengths together. RL padding fell from 14% to 6% (§并行策略), and SFT
  padding stays under 8% with 4K/8K/16K/32K buckets.

Incident 4 (run 2, step 263): one PP-stage-2 rank OOM'd. A 31K-token T2
trajectory was padded to the 32K bucket, micro-batches were cut by sequence
*count*, so every sequence in that micro-batch was padded to 32K. PP keeps 2
micro-batches in flight, and memory was fragmented after many offload
cycles. The fix had three parts:

1. cut micro-batches by a **token budget** that counts padding
   (:func:`make_token_budget_microbatches`);
2. cap prompt + response at 32K (:func:`check_sequence_caps`);
3. enable expandable segments (``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``,
   set by the launch scripts).

The permanent guard is :func:`preflight`. Every micro-batch plan is checked
against :class:`ActivationMemoryModel` *before* it is sent to the GPUs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

DEFAULT_BUCKETS: tuple[int, ...] = (4096, 8192, 16384, 32768)
GB = 1e9


@dataclass
class MicroBatch:
    indices: list[int]
    padded_len: int
    real_tokens: int

    @property
    def padded_tokens(self) -> int:
        return len(self.indices) * self.padded_len


def padded_length(n: int, *, buckets: Sequence[int] | None = None, pad_multiple: int = 128) -> int:
    """Length a sequence of ``n`` tokens occupies after padding.

    With ``buckets``: the smallest bucket ≥ n (raises if n exceeds the largest).
    Otherwise: ``n`` rounded up to ``pad_multiple`` (TP/SP need divisibility).
    """
    if n <= 0:
        raise ValueError("sequence length must be positive")
    if buckets:
        for b in sorted(buckets):
            if n <= b:
                return int(b)
        raise ValueError(f"sequence of {n} tokens exceeds the largest bucket {max(buckets)}")
    return int(-(-n // pad_multiple) * pad_multiple)


def make_token_budget_microbatches(
    lengths: Sequence[int],
    max_tokens: int,
    *,
    buckets: Sequence[int] | None = None,
    pad_multiple: int = 128,
    max_seqs: int | None = None,
) -> list[MicroBatch]:
    """Split sequences into micro-batches whose *padded* size never exceeds ``max_tokens``.

    Sequences are sorted by length (descending) and filled greedily, so each
    micro-batch holds similar lengths. In a sorted run, the first sequence of
    a micro-batch is its longest, so the padded cost of adding another
    sequence is known exactly.
    """
    order = sorted(range(len(lengths)), key=lambda i: (-int(lengths[i]), i))
    mbs: list[MicroBatch] = []
    cur: list[int] = []
    cur_pad = 0
    for i in order:
        p = padded_length(int(lengths[i]), buckets=buckets, pad_multiple=pad_multiple)
        if p > max_tokens:
            raise ValueError(f"sample {i} ({lengths[i]} tokens, padded {p}) exceeds the micro-batch budget")
        # cur_pad is the padded length of the micro-batch's first (= longest) sequence.
        if cur and ((len(cur) + 1) * cur_pad > max_tokens or (max_seqs is not None and len(cur) >= max_seqs)):
            mbs.append(MicroBatch(cur, cur_pad, int(sum(lengths[j] for j in cur))))
            cur = []
        if not cur:
            cur_pad = p
        cur.append(i)
    if cur:
        mbs.append(MicroBatch(cur, cur_pad, int(sum(lengths[j] for j in cur))))
    return mbs


def make_count_microbatches(
    lengths: Sequence[int],
    micro_batch_size: int,
    *,
    buckets: Sequence[int] | None = None,
    pad_multiple: int = 128,
    sort: bool = False,
) -> list[MicroBatch]:
    """Fixed sequence-count micro-batches, the pre-Incident-4 scheme. Kept for comparison and tests."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i]) if sort else list(range(len(lengths)))
    out = []
    for k in range(0, len(order), micro_batch_size):
        idx = order[k : k + micro_batch_size]
        pad = max(padded_length(int(lengths[i]), buckets=buckets, pad_multiple=pad_multiple) for i in idx)
        out.append(MicroBatch(idx, pad, int(sum(lengths[i] for i in idx))))
    return out


def padding_overhead(mbs: Sequence[MicroBatch]) -> float:
    """``1 - real/padded`` over a plan."""
    real = sum(m.real_tokens for m in mbs)
    padded = sum(m.padded_tokens for m in mbs)
    return 1.0 - real / padded if padded else 0.0


def check_sequence_caps(
    prompt_lengths: Sequence[int], response_lengths: Sequence[int], *, max_total: int = 32768
) -> np.ndarray:
    """Indices of samples whose prompt + response exceeds the 32K cap (they must not be trained)."""
    tot = np.asarray(prompt_lengths) + np.asarray(response_lengths)
    return np.flatnonzero(tot > max_total)


# ----------------------------------------------------------------------------------------------
# Activation memory model
# ----------------------------------------------------------------------------------------------
@dataclass
class ActivationMemoryModel:
    """Per-GPU activation memory of a full-recompute Megatron step, as a function of micro-batch size.

    With ``recompute_granularity=full`` a PP stage keeps only each layer's
    *input* per in-flight micro-batch (``tokens × hidden × 2 B``, divided by
    TP when sequence parallelism is on). During backward it rematerializes
    one layer at a time. The working set of that layer is

    * attention/MLP activations ≈ ``34 · s · h / TP`` bytes (Korthikanti et al.,
      "Reducing Activation Recomputation in Large Transformer Models", eq. 2,
      with SP),
    * SwiGLU intermediates ``3 · s · ffn · 2 B / TP``,
    * DeltaNet chunk states kept for backward: ``(s/64) · (V_heads/TP) ·
      d_k · d_v · 4 B`` (FP32).

    The formula is then scaled by ``calibration`` so that the report's
    reference point (one 32K micro-batch, 2 in flight, ≈ 14 GB) is matched.
    The calibration absorbs allocator overhead and kernels' temporary
    buffers. Re-fit it from the per-rank peak-memory logs whenever the stack
    changes.
    """

    hidden_size: int = 5120
    ffn_hidden_size: int = 17408
    num_layers: int = 64
    tp: int = 4
    pp: int = 2
    sequence_parallel: bool = True
    gdn_v_heads: int = 48
    gdn_head_dim: int = 128
    gdn_chunk: int = 64
    act_bytes: int = 2
    working_coeff: float = 34.0
    in_flight: int | None = None                      # default: pp (1F1B warm-up depth on stage 0)
    reference_point: tuple[int, float] = (32768, 14.0)
    calibration: float | None = None
    static_gb: float = 40.6     # params 6.8 + fp32 grads 13.5 + sharded optimizer 20.3 (report table)
    other_gb: float = 7.0       # chunked CE/logprob 2 + NCCL buffers, CUDA graphs, fragmentation 5
    _cal: float = field(init=False, default=1.0)

    def __post_init__(self) -> None:
        if self.in_flight is None:
            self.in_flight = self.pp
        if self.calibration is not None:
            self._cal = self.calibration
        else:
            tok, gb = self.reference_point
            raw = self._raw_gb([tok] * int(self.in_flight))
            self._cal = gb / raw

    @property
    def layers_per_stage(self) -> int:
        return -(-self.num_layers // self.pp)

    def _stored_bytes(self, tokens: int) -> float:
        div = self.tp if self.sequence_parallel else 1
        return self.layers_per_stage * tokens * self.hidden_size * self.act_bytes / div

    def _working_bytes(self, tokens: int) -> float:
        attn_mlp = self.working_coeff * tokens * self.hidden_size / self.tp
        swiglu = 3 * tokens * self.ffn_hidden_size * self.act_bytes / self.tp
        gdn = (tokens / self.gdn_chunk) * (self.gdn_v_heads / self.tp) * self.gdn_head_dim**2 * 4
        return attn_mlp + swiglu + gdn

    def _raw_gb(self, mb_tokens: Sequence[int]) -> float:
        if not mb_tokens:
            return 0.0
        k = int(self.in_flight or 1)
        worst = sorted(mb_tokens, reverse=True)[:k]
        return (sum(self._stored_bytes(t) for t in worst) + self._working_bytes(max(mb_tokens))) / GB

    def activation_gb(self, mb_tokens: Sequence[int]) -> float:
        """Worst-case activation memory for a plan (the ``in_flight`` largest micro-batches resident)."""
        return self._cal * self._raw_gb(mb_tokens)

    def peak_gb(self, mb_tokens: Sequence[int]) -> float:
        return self.static_gb + self.other_gb + self.activation_gb(mb_tokens)


@dataclass
class PreflightReport:
    ok: bool
    peak_gb: float
    limit_gb: float
    padding_overhead: float
    max_microbatch_tokens: int
    messages: list[str] = field(default_factory=list)


def preflight(
    plan: Sequence[MicroBatch],
    model: ActivationMemoryModel,
    *,
    limit_gb: float = 76.0,
) -> PreflightReport:
    """Reject a micro-batch plan whose estimated peak exceeds ``limit_gb``.

    The 76 GB default is the dashboard's hard limit for the per-rank memory
    peak; the alert fires at 72 GB.
    """
    toks = [m.padded_tokens for m in plan]
    peak = model.peak_gb(toks)
    msgs = []
    if peak > limit_gb:
        msgs.append(
            f"estimated peak {peak:.1f} GB > limit {limit_gb:.1f} GB "
            f"(largest micro-batch {max(toks)} padded tokens); cut micro-batches by token budget"
        )
    return PreflightReport(
        ok=not msgs, peak_gb=peak, limit_gb=limit_gb, padding_overhead=padding_overhead(plan),
        max_microbatch_tokens=max(toks) if toks else 0, messages=msgs,
    )
