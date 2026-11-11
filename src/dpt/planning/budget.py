"""Back-of-the-envelope planner: memory, communication and compute budgets from first principles.

Why this exists
---------------
The report makes several quantitative claims that drive the engineering
choices: KV per token, per-sequence DeltaNet state, the 16 GB full-logits
trap, per-GPU training memory, rollout concurrency, offload volume, and SFT
wall-clock. This module recomputes each one from the model architecture and
the hardware profile. When the model, the context limit or the cluster
changes, rerun it and read off the new numbers instead of guessing.
``tests/planning/test_budget.py`` checks every figure against the report.

Run:  python -m dpt.planning.budget                       # cluster profile, markdown tables
      python -m dpt.planning.budget --profile local_4xa100_40g
      python -m dpt.planning.budget --profile slurm_3xa100_40g --model-line qwen35_9b   # 9B

All sizes are in decimal GB (1e9 bytes), matching the report's convention.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

GB = 1e9


# ----------------------------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Arch:
    """Qwen3.8-27B text backbone (``config.json`` → ``text_config``)."""

    params: float = 27.0e9
    num_layers: int = 64
    num_full_attn_layers: int = 16
    num_linear_attn_layers: int = 48
    hidden: int = 5120
    ffn: int = 17408
    attn_kv_heads: int = 4
    attn_head_dim: int = 256
    gdn_v_heads: int = 48
    gdn_qk_heads: int = 16
    gdn_head_dim: int = 128
    conv_kernel: int = 4
    vocab: int = 248_320

    @classmethod
    def from_project(cls, arch_cfg: dict) -> "Arch":
        return cls(
            params=float(arch_cfg["params_billion"]) * 1e9,
            num_layers=arch_cfg["num_layers"],
            num_full_attn_layers=arch_cfg["num_full_attn_layers"],
            num_linear_attn_layers=arch_cfg["num_linear_attn_layers"],
            hidden=arch_cfg["hidden_size"],
            ffn=arch_cfg["ffn_hidden_size"],
            attn_kv_heads=arch_cfg["attn_kv_heads"],
            attn_head_dim=arch_cfg["attn_head_dim"],
            gdn_v_heads=arch_cfg["gdn_v_heads"],
            gdn_qk_heads=arch_cfg["gdn_qk_heads"],
            gdn_head_dim=arch_cfg["gdn_head_dim"],
            vocab=arch_cfg["vocab_size"],
        )


@dataclass(frozen=True)
class Cluster:
    nnodes: int = 2
    gpus_per_node: int = 8
    gpu_mem_gb: float = 80.0
    bf16_tflops: float = 312.0
    ib_gbps: float = 100.0            # GB/s per node
    # Effective host<->device rate during offload. PCIe Gen4 x16 is ~32 GB/s peak, but two GPUs
    # share a PCIe switch and pageable/pinned staging adds overhead; ~8 GB/s per GPU is what the
    # report's "~5 s per direction for ~40 GB" implies.
    offload_gbps_per_gpu: float = 8.0

    @property
    def n_gpus(self) -> int:
        return self.nnodes * self.gpus_per_node


@dataclass(frozen=True)
class Layout:
    tp: int = 4
    pp: int = 2
    dp: int = 2
    rollout_tp: int = 2
    rollout_gpu_mem_util: float = 0.7


# ----------------------------------------------------------------------------------------------
# Inference-side unit costs
# ----------------------------------------------------------------------------------------------
def weights_bytes(a: Arch, bytes_per_param: int = 2) -> float:
    return a.params * bytes_per_param


def kv_bytes_per_token(a: Arch, bytes_per_el: int = 2) -> float:
    """Only the 16 gated-attention layers keep a KV cache; DeltaNet layers keep fixed-size state."""
    return a.num_full_attn_layers * a.attn_kv_heads * a.attn_head_dim * 2 * bytes_per_el


def deltanet_state_bytes_per_seq(a: Arch, state_bytes: int = 4, conv_bytes: int = 2) -> float:
    """Recurrent state (V-heads × d × d per layer) plus the short-conv state (kernel-1 taps).

    The recurrent part dominates: 48 layers × 48 heads × 128 × 128. vLLM/SGLang keep it in FP32 by
    default (``mamba_ssm_dtype: float32`` in the model config). That is also the precision
    the training side accumulates in, which is why we never downcast it (Incident 1).
    """
    recurrent = a.num_linear_attn_layers * a.gdn_v_heads * a.gdn_head_dim * a.gdn_head_dim * state_bytes
    conv_dim = 2 * a.gdn_qk_heads * a.gdn_head_dim + a.gdn_v_heads * a.gdn_head_dim
    conv = a.num_linear_attn_layers * conv_dim * (a.conv_kernel - 1) * conv_bytes
    return recurrent + conv


def seq_inference_bytes(a: Arch, n_tokens: int, state_bytes: int = 4) -> float:
    return n_tokens * kv_bytes_per_token(a) + deltanet_state_bytes_per_seq(a, state_bytes)


def full_logits_bytes(a: Arch, n_tokens: int, bytes_per_el: int = 4) -> float:
    """Materializing FP32 logits for one sequence. At 16K tokens this is ~16 GB, hence chunked CE."""
    return a.vocab * n_tokens * bytes_per_el


# ----------------------------------------------------------------------------------------------
# Training-side memory (colocated, per GPU)
# ----------------------------------------------------------------------------------------------
@dataclass
class TrainMemory:
    params_bf16: float
    grads_fp32: float
    optimizer: float
    activations: float
    chunked_ce: float
    misc: float
    rows: dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return self.params_bf16 + self.grads_fp32 + self.optimizer + self.activations + self.chunked_ce + self.misc


def train_memory_per_gpu(
    a: Arch,
    lay: Layout,
    *,
    activations_gb: float | None = None,
    chunked_ce_gb: float = 2.0,
    misc_gb: float = 5.0,           # NCCL buffers, CUDA graphs, fragmentation
) -> TrainMemory:
    """Megatron with distributed optimizer, average pipeline stage.

    * Parameters and gradients are sharded by TP×PP (model parallel).
    * Optimizer state (FP32 master + Adam m + v = 12 bytes/param) is additionally
      sharded across DP by the distributed optimizer → TP×PP×DP = all 16 GPUs.
    * Activations default to :func:`activation_gb` for this layout.
    """
    mp = lay.tp * lay.pp
    act = activation_gb(a, lay) if activations_gb is None else activations_gb
    return TrainMemory(
        params_bf16=a.params * 2 / mp,
        grads_fp32=a.params * 4 / mp,
        optimizer=a.params * 12 / (mp * lay.dp),
        activations=act * GB,
        chunked_ce=chunked_ce_gb * GB,
        misc=misc_gb * GB,
    )


# Measured on the 27B at TP4 × PP2: full recompute, micro-batch 1 × 32K, 2 PP micro-batches in flight.
_ACT_CALIB_GB = 14.0
_ACT_CALIB_LAYERS_PER_TP_RANK = 8.0      # 64 layers / PP2 / TP4
_ACT_CALIB_HIDDEN = 5120


def activation_gb(a: Arch, lay: Layout) -> float:
    """Activation memory per GPU, scaled from the 27B measurement.

    With full recompute, the stored activations are each layer's input (``seq × hidden``) for the
    layers of this stage, split by TP (sequence parallel), plus one layer's recompute working set.
    Both scale with layers-per-stage / TP and with the hidden size, so the measurement transfers.
    """
    layers_per_rank = a.num_layers / lay.pp / lay.tp
    return _ACT_CALIB_GB * layers_per_rank / _ACT_CALIB_LAYERS_PER_TP_RANK * a.hidden / _ACT_CALIB_HIDDEN


def first_stage_memory_per_gpu(a: Arch, lay: Layout, *, tied_embeddings: bool = False) -> TrainMemory:
    """The heaviest pipeline stage: the first one holds the embedding (the last one the LM head).

    At 248K vocabulary the embedding is ``vocab × hidden`` parameters (1.0B on the 9B, 1.3B on the
    27B), several layers' worth. With deep pipelines it decides whether a layout fits.
    """
    emb = a.vocab * a.hidden
    per_layer = (a.params - emb * (1 if tied_embeddings else 2)) / a.num_layers
    stage = (per_layer * a.num_layers / lay.pp + emb) / lay.tp
    avg = train_memory_per_gpu(a, lay)
    return TrainMemory(
        params_bf16=stage * 2,
        grads_fp32=stage * 4,
        optimizer=stage * 12 / lay.dp,
        activations=avg.activations,
        chunked_ce=avg.chunked_ce,
        misc=avg.misc,
    )


def offloaded_state_per_gpu(a: Arch, lay: Layout) -> float:
    m = train_memory_per_gpu(a, lay)
    return m.params_bf16 + m.grads_fp32 + m.optimizer


# ----------------------------------------------------------------------------------------------
# Rollout-side memory and concurrency
# ----------------------------------------------------------------------------------------------
@dataclass
class RolloutMemory:
    weights_per_gpu: float
    kv_state_pool_per_gpu: float
    kv_state_pool_per_replica: float
    per_seq_bytes: float
    concurrent_per_replica: int
    concurrent_total: int


def rollout_memory(
    a: Arch,
    c: Cluster,
    lay: Layout,
    *,
    avg_response_tokens: int = 6144,
    runtime_overhead_gb: float = 2.0,   # vLLM activations / sampler workspace inside the util budget
) -> RolloutMemory:
    weights_per_gpu = weights_bytes(a) / lay.rollout_tp
    budget = c.gpu_mem_gb * GB * lay.rollout_gpu_mem_util
    pool = max(0.0, budget - weights_per_gpu - runtime_overhead_gb * GB)
    per_replica = pool * lay.rollout_tp
    per_seq = seq_inference_bytes(a, avg_response_tokens)
    conc = int(per_replica // per_seq)
    replicas = c.n_gpus // lay.rollout_tp
    return RolloutMemory(weights_per_gpu, pool, per_replica, per_seq, conc, conc * replicas)


# ----------------------------------------------------------------------------------------------
# Compute and communication
# ----------------------------------------------------------------------------------------------
def sft_hours(
    a: Arch,
    c: Cluster,
    *,
    tokens: float = 420e6,
    epochs: int = 2,
    mfu: float = 0.38,
    padding_overhead: float = 0.08,
) -> dict[str, float]:
    """Model FLOPs = 6·N·D (forward + backward).

    Full recompute adds hardware FLOPs but not model FLOPs, so the MFU figure absorbs it.
    """
    flops = 6 * a.params * tokens * epochs
    secs = flops / (c.n_gpus * c.bf16_tflops * 1e12 * mfu) * (1 + padding_overhead)
    return {"flops": flops, "hours": secs / 3600, "gpu_hours": secs / 3600 * c.n_gpus}


def comm_seconds(a: Arch, c: Cluster, lay: Layout) -> dict[str, float]:
    per_gpu_state = offloaded_state_per_gpu(a, lay)
    # DP reduce-scatter (BF16 grads) + param all-gather; each node moves ~one model copy over IB.
    dp_bytes_per_node = a.params * 2
    return {
        "dp_grad_sync_s": 2 * dp_bytes_per_node / (c.ib_gbps * GB) / lay.dp,
        "offload_one_way_s": per_gpu_state / (c.offload_gbps_per_gpu * GB),
        "offload_bytes_per_node_gb": per_gpu_state * c.gpus_per_node / GB,
    }


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------
def _fmt(x: float) -> str:
    return f"{x / GB:.2f} GB"


def report(a: Arch, c: Cluster, lay: Layout) -> str:
    lines: list[str] = []
    add = lines.append
    add("## Inference unit costs")
    add(f"- KV per token ({a.num_full_attn_layers} attn layers, BF16): {kv_bytes_per_token(a) / 1024:.0f} KiB")
    add(f"- DeltaNet state per sequence: FP32 {deltanet_state_bytes_per_seq(a, 4) / 1e6:.0f} MB, "
        f"BF16 {deltanet_state_bytes_per_seq(a, 2) / 1e6:.0f} MB")
    add(f"- One 16K-token response: {_fmt(seq_inference_bytes(a, 16384))} (KV + FP32 state)")
    add(f"- Full FP32 logits for 16K tokens: {_fmt(full_logits_bytes(a, 16384))} → must use chunked/fused CE")
    m = train_memory_per_gpu(a, lay)
    add("\n## Training memory per GPU (colocated, TP{}×PP{}×DP{})".format(lay.tp, lay.pp, lay.dp))
    for k in ("params_bf16", "grads_fp32", "optimizer", "activations", "chunked_ce", "misc"):
        add(f"- {k}: {_fmt(getattr(m, k))}")
    verdict = "fits" if m.total < c.gpu_mem_gb * GB * 0.9 else "DOES NOT FIT"
    add(f"- **total: {_fmt(m.total)}** of {c.gpu_mem_gb:.0f} GB → {verdict}")
    f = first_stage_memory_per_gpu(a, lay)
    fverdict = "fits" if f.total < c.gpu_mem_gb * GB * 0.9 else "DOES NOT FIT"
    add(f"- heaviest stage (embedding / LM head): **{_fmt(f.total)}** → {fverdict}")
    r = rollout_memory(a, c, lay)
    add(f"\n## Rollout (vLLM TP{lay.rollout_tp}, util {lay.rollout_gpu_mem_util})")
    add(f"- weights per GPU: {_fmt(r.weights_per_gpu)}; KV+state pool per GPU: {_fmt(r.kv_state_pool_per_gpu)}")
    add(f"- per 6K-token sequence: {_fmt(r.per_seq_bytes)} → ~{r.concurrent_per_replica} concurrent/replica, "
        f"~{r.concurrent_total} total")
    if r.concurrent_per_replica == 0:
        add("- rollout replica does not fit: raise rollout TP (the local box serves 27B with TP4 for eval)")
    s = sft_hours(a, c)
    add(f"\n## SFT compute\n- {s['flops']:.2e} FLOPs → {s['hours']:.1f} h on {c.n_gpus} GPUs "
        f"({s['gpu_hours']:.0f} GPU·h)")
    cm = comm_seconds(a, c, lay)
    add(f"\n## Communication\n- DP grad sync ≈ {cm['dp_grad_sync_s']:.1f} s/step; offload "
        f"{cm['offload_bytes_per_node_gb']:.0f} GB/node, ≈ {cm['offload_one_way_s']:.1f} s one way")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--profile", default="cluster_2x8xa100_80g")
    p.add_argument("--model-line", default=None, help="configs/model/<line>.yaml (default: $DPT_MODEL_LINE or 27B)")
    args = p.parse_args()
    from dpt.common.config import load_config, to_container

    cfg = to_container(load_config(profile=args.profile, model_line=args.model_line))
    hw = cfg["hardware"]
    a = Arch.from_project(cfg["project"]["model"]["arch"])
    c = Cluster(nnodes=hw["nnodes"], gpus_per_node=hw["gpus_per_node"], gpu_mem_gb=hw["gpu_mem_gb"],
                bf16_tflops=hw["gpu_bf16_tflops"], ib_gbps=hw["ib_gbps"] or 1.0)
    print(f"# {cfg['project']['model']['base']} on {hw['profile']} "
          f"({c.nnodes} × {c.gpus_per_node} × {c.gpu_mem_gb:.0f} GB)\n")
    t, r = hw["train"], hw["rollout"]
    lay = Layout(tp=t["tp"], pp=t["pp"], dp=t["dp"], rollout_tp=r["tp"],
                 rollout_gpu_mem_util=r["gpu_memory_utilization"])
    print(report(a, c, lay))


if __name__ == "__main__":
    main()
