# Systems design: memory, parallelism, the colocated RL step

This document explains *why the run is laid out the way it is*. Every number
below is recomputed by `python -m dpt.planning.budget` from the model config
and the hardware profile, and `tests/planning/test_budget.py` checks it against
the report. When the model or the cluster changes, rerun the planner and do not
copy numbers by hand.

---

## 1. Two constraints decide almost everything

**A100 has no FP8.** Training and inference both run in BF16. FP8 rollout,
the usual way to halve rollout memory and double decode bandwidth on H100,
is not available here. Rollout must fit 54 GB of BF16 weights plus KV cache
and GDN state.

**Qwen3.8-27B is a hybrid model.** It is not a plain dense transformer. From its
`config.json` (`model_type: qwen3_5`, same code path as Qwen3.5):

| fact | value | consequence |
|---|---|---|
| layers | 64 = 48 Gated DeltaNet + 16 gated attention (every 4th) | only 16 layers keep a KV cache; 48 keep a fixed-size recurrent state per sequence |
| attention heads | 24 Q, **4 KV**, head_dim 256 | training TP ≤ 4, otherwise KV heads are replicated |
| GDN heads | 48 V, 16 QK, head_dim 128 | per layer, per sequence: a 48×128×128 state matrix |
| hidden / FFN | 5120 / 17408 | GEMMs are still a healthy size at TP4 |
| vocab | 248,320 (confirmed) | full logits must never be materialized |
| `mamba_ssm_dtype` | float32 | engines keep GDN state in FP32 by default; never downcast it (Incident 1) |
| MTP | 1 layer | a free draft model for speculative decoding, *if it is kept trained* (Incident 5) |
| context | 262K native | the project caps at 32K |

## 2. Unit costs (`dpt.planning.budget`)

| quantity | value | why it matters |
|---|---|---|
| BF16 weights | 54 GB | inference needs ≥ TP2 on 80 GB cards (TP4 on the 40 GB workstation) |
| KV per token | 16 × 4 × 256 × 2 × 2 B = **64 KiB** | 4–8× less than a dense 27B: long rollouts are cheap in KV |
| GDN state per sequence | **154 MB FP32** / 78 MB BF16 | a fixed per-sequence cost, independent of length |
| one 16K response | ≈ 1.2 GB (1.07 KV + 0.15 state) | sets rollout concurrency |
| full FP32 logits, 16K tokens | **16.3 GB** | one long sample would OOM a naive loss → vocab-parallel/chunked logprobs |

## 3. Training layout: TP4 × PP2 × DP2, sequence parallel, CP 1

| choice | reason |
|---|---|
| TP = 4 | the KV-head limit; TP8 would replicate KV heads and waste memory/bandwidth |
| PP = 2 | TP×PP = 8 = one node. Each node holds one full model replica, so **only DP gradient traffic crosses InfiniBand** (≈54 GB per node per step, ~1–2 s at 100 GB/s) |
| DP = 2 | the two nodes; the distributed optimizer shards Adam state 16-way |
| SP on | activation memory of norms/dropout split across TP ranks |
| CP = 1 | context parallelism is not validated for GDN on this stack |
| BSHD padding, no packing | Megatron's GDN path cannot reset the recurrent state at packed-sequence boundaries. One sample per micro-batch (`ppo_micro_batch_size_per_gpu: 1`, `pad_bshd_to_minibatch_max: false`) means padding is only TP/SP alignment |
| full recompute | activations of a 32K micro-batch with 2 PP micro-batches in flight ≈ 14 GB |
| logprob / entropy | vocab-parallel over the TP shards (each rank holds 62K of the 248K vocab) + `entropy_from_logits_with_chunking` (2K-token chunks) |

**Deviation from the report's appendix.** The report enables a fused, chunked
FP32 cross-entropy via `VERL_MEGATRON_MEM_EFFICIENT_CE=1`. That variable does
not exist in verl 0.9.1. Its fused linear-CE kernel (`model.use_fused_kernels`)
is auto-disabled whenever MTP or BSHD padding is on, and both are required here.
The vocab-parallel path materializes a `[32K × 62K]` logits shard per TP rank
(≈4 GB BF16, ≈8 GB if upcast) instead of the report's ~2 GB. The plan below
still fits under the 72 GB alert line, but `memory/peak_gb_max_rank` is the
metric to watch in the first steps.

### Per-GPU memory, colocated (80 GB card)

| item | training phase | rollout phase |
|---|---|---|
| BF16 params (54 GB / 8) | 6.75 GB | 0 (offloaded) |
| FP32 grads | 13.5 GB | 0 (offloaded) |
| distributed optimizer, 12 B/param / 16 | 20.25 GB | 0 (offloaded) |
| activations (1 × 32K, PP 2 in flight, full recompute) | ~14 GB | – |
| logprob / entropy workspace | ~2 GB (report) – ~6 GB (verl 0.9.1, see above) | – |
| vLLM weights (TP2) | – | 27 GB |
| KV + GDN state (`gpu_memory_utilization` 0.7) | – | ~27 GB |
| NCCL, CUDA graphs, fragmentation | ~5 GB | ~4 GB |
| **total** | **≈ 61.5 GB** (planner) | **≈ 58 GB** |

The memory pre-flight in `dpt.launch.rl` evaluates the worst case (two maximal
32K sequences in flight) with `dpt.rl.microbatch.ActivationMemoryModel` and
refuses to launch above 76 GB. This guard came from Incident 4: a 31K
trajectory padded to a 32K bucket with 2 PP micro-batches in flight, after
several offload cycles had fragmented memory. The other guards from that
incident are `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and a hard
prompt + response ≤ 32K cap.

## 4. Rollout layout and the colocated step

**vLLM TP2 × 8 replicas.** Decode is memory-bandwidth bound. More, smaller
replicas means more concurrent sequences per byte of weight traffic. TP1
cannot hold 54 GB plus KV. Per replica ≈ 54 GB for KV + state; at a 6K-token
average response that is ≈ 0.55 GB per sequence → ~100 concurrent per replica,
~800 total (the planner gives 97 / 776).

**One RL step (verl v1 `colocate_async` + our hooks):**

```
 ┌─ rollout ────────────────────────────────────────────────────────────────┐
 │ vLLM awake on all 16 GPUs (training state offloaded to CPU, ~40 GB/GPU)   │
 │ 128 prompts × 8 samples; MTP speculative decoding (acceptance ≈ 0.7)      │
 │ rewards computed as trajectories finish (sandbox / judge are external)    │
 │ batch complete → abort the long tail (resumed next step = partial rollout)│
 └──────────────────────────────────────────────────────────────────────────┘
        │ replay buffer: drop zero-advantage groups (≈35% of tokens)
        ▼
 ┌─ train ──────────────────────────────────────────────────────────────────┐
 │ vLLM asleep; onload training state (~5 s)                                 │
 │ old-logprob recompute (Megatron) → TIS weights, mismatch metrics          │
 │ task-aware advantages → 2 mini-batch updates (token-sum loss)             │
 └──────────────────────────────────────────────────────────────────────────┘
        │ weight sync inside each node (CUDA IPC, TP4×PP2 → TP2, ~10 s);
        │ vLLM resets its prefix cache on wake-up (Incident 3)
        ▼ next step
```

**Partial rollout in verl.** Without it the step waits for the slowest 3% of
responses. 90% of tokens arrive in the first 4 of 9 minutes. verl's
`colocate_async` aborts unfinished requests once enough finished groups exist
and resumes them after the weight sync. The resumed sample is part old-policy
tokens, part new-policy tokens. Its stored per-token rollout logprobs make
TIS correct for that. `max_off_policy_threshold: 2` lets a group span at most two
policy versions (verl counts versions spanned, so this is "staleness ≤ 1").

## 5. Step-time budget (report, per step: 128 × 8 = 1,024 responses, mean 5.5K tokens)

| phase | baseline | optimized | what changed |
|---|---|---|---|
| rollout | 9.0 min | 4.6 | partial rollout; MTP speculative decoding |
| reward tail | 0.5 | 0.3 | warm sandboxes; timeouts counted separately |
| old-logprob | 3.2 | 2.1 | zero-advantage groups dropped *before* recompute |
| train fwd/bwd | 13.8 | 8.6 | the same drop (−35% tokens); length-aware micro-batches |
| offload/onload/sync | 0.5 | 0.5 | – |
| **total** | **27.0** | **16.1** | |

**Why not one node rolling out and one node training (ablation A12)?** On A100
the 27B run is *training*-bound. Halving the training GPUs cost more than
overlapping generation saved: 31 vs 27 min/step. Whether async helps depends on
the workload and the hardware. Measure it; do not assume it.

## 6. Communication

| traffic | when | volume | link | time |
|---|---|---|---|---|
| TP all-reduce / SP reduce-scatter | every layer | ~activations | NVLink | ~10% of train time, partly overlapped |
| PP activations | per micro-batch | ~84 MB | NVLink | negligible |
| DP grad reduce-scatter + param all-gather | per optimizer step | ~54 GB / node | IB | 1–2 s |
| weight resharding + sync to vLLM | per RL step | 54 GB / node | CUDA IPC (in-node) | ~10 s |
| offload / onload | twice per step | ~40 GB / GPU | PCIe | ~5 s each |

## 7. The 4×A100-40GB workstation

| works | does not |
|---|---|
| all CPU tests, data pipeline, graders, planner, launch dry runs | 27B training of any kind: training state alone is ~430 GB before activations |
| 27B *inference* with vLLM TP4 (13.5 GB weights/GPU, ~20 GB KV) → baseline eval, difficulty calibration, self-distillation data | 27B rollout at TP1/TP2 |
| GDN kernel parity and vLLM-vs-HF logprob parity on SM80 | |
| full RL/SFT smoke on Qwen3.5-0.8B (same architecture): TP1 × PP2 × DP2, vLLM TP1 × 4 | |

Environment caveat: the driver (560.35) supports CUDA ≤ 12.6, while the pinned
torch/vLLM wheels target newer CUDA. A100 supports CUDA forward compatibility,
so install `cuda-compat` or use the cluster image. `python -m dpt.launch.check_env`
detects this.
