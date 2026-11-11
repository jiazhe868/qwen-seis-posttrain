> **Archived design document.** This was the README of the original plan: Qwen3.8-27B on a 2 × 8 × A100-80GB
> cluster, with the step-by-step run guide, the design decisions and the configuration system. The run that
> actually happened is the Qwen3.5-9B line on the cluster, described in the [README](../README.md). The 27B
> configs (`configs/*/*27b*.yaml`, `configs/hardware/cluster_2x8xa100_80g.yaml`) still follow this document.

# Domain post-training for Qwen3.8-27B: SFT + multi-task RL

This repository implements the full post-training plan in the project report:
data pipeline → numerical parity checks → baseline evaluation → domain SFT →
multi-task GRPO-family RL → evaluation, gates and audits. The target base model
is **Qwen3.8-27B**, a hybrid Gated-DeltaNet / gated-attention model with an MTP
head and a vision tower. The training stack is **verl 0.9.1 + Megatron-Core
(Megatron-Bridge) + vLLM** on A100s.

> **Goal.** At roughly the inference cost of the base model at *medium*
> reasoning effort, beat the base model at *xhigh* effort on the domain, without
> hurting general ability. Every target is fixed before the project starts and
> must clear the measured noise floor.

The code is written for two machines with **one code path**:

| machine | used for |
|---|---|
| **workstation**: 1 × 4 × A100-SXM4-**40GB** | unit tests, data pipeline, graders, 27B *inference* (baseline eval, difficulty calibration, self-distillation), GPU parity checks, end-to-end smoke runs on a 0.8B model of the same architecture |
| **cluster**: 2 × 8 × A100-SXM4-**80GB** | 27B SFT, 27B RL (main line + reproducibility seed), 9B ablation screening, final evaluation |
| **the cluster**: ≤ 8 nodes × 3 × A100-PCIE-**40GB** per job | the **one-month plan**: the full method on **Qwen3.5-9B** (same hybrid architecture), because 27B RL does not fit 40 GB PCIe GPUs (`README`, commands in `scripts/slurm/README.md`) |

Pick the machine with `DPT_HW_PROFILE=local_4xa100_40g | cluster_2x8xa100_80g | slurm_3xa100_40g`
and the model with `DPT_MODEL_LINE=qwen38_27b` (default) `| qwen35_9b`. Nothing else changes.

---

## Contents

0. [Quick start](#0-quick-start)
1. [Goals and success criteria](#1-goals-and-success-criteria)
2. [The plan, in order](#2-the-plan-in-order)
3. [Two machines, one code path](#3-two-machines-one-code-path)
4. [Repository layout](#4-repository-layout)
5. [Step-by-step run guide](#5-step-by-step-run-guide)
6. [Design decisions and the reasoning behind them](#6-design-decisions-and-the-reasoning-behind-them)
7. [How the verl integration works](#7-how-the-verl-integration-works)
8. [Configuration system](#8-configuration-system)
9. [Testing](#9-testing)
10. [Expected results and how the final table is produced](#10-expected-results-and-how-the-final-table-is-produced)
11. [Risks, open items, and what to verify first](#11-risks-open-items-and-what-to-verify-first)
12. [Troubleshooting](#12-troubleshooting)
13. [References](#13-references)

To find where a specific claim from the report is implemented, and which test
proves it, see **`docs/report_traceability.md`**. It maps every report section
to its config, code and tests.

---

## 0. Quick start

Everything in this section runs on a laptop or on the workstation's CPU. No GPU is needed.

```bash
# core environment (CPU): tests, data pipeline, graders, planner, launch dry runs
bash scripts/00_setup_env.sh core          # or: pip install -r requirements/core.txt && pip install -e .

make test          # 763 unit tests, ~40 s on CPU
make lint          # flake8, 120 columns
make plan          # memory / FLOPs / communication budget for the 2×8×A100-80G cluster
make plan-local    # ...and why 27B training does not fit on 4×A100-40GB
make graders       # adversarial grader suite: must be 100%
make data-smoke    # toy data pipeline end to end (offline fakes), ~1 s → data/smoke/
make dry-run-sft   # render the exact 27B SFT command (no execution)
make dry-run-rl    # render the exact 27B RL command and run all pre-flight checks
make check-env     # GPUs, driver/CUDA, package versions vs what this repo was verified with
```

Then follow [§5](#5-step-by-step-run-guide) step by step. Each step is one script in `scripts/`.

---

## 1. Goals and success criteria

All gates live in **one file**, `configs/project.yaml → project.gates`, and are
checked by `python -m dpt.eval.gates`, which exits non-zero on any failure. A gain
counts only if it is outside the 95% interval *and* larger than the measured
noise floor (`dpt.eval.stats.is_significant_improvement`).

| goal | metric | gate | report's target outcome |
|---|---|---|---|
| domain numeric reasoning | D-Num avg@8 | ≥ base(xhigh) + 8 | 71.4 (base xhigh 55.3) |
| domain code & tool use | D-Code pass@1, hidden tests | ≥ base(xhigh) + 10 | 55.2 (base xhigh 40.2) |
| open-ended domain reports | D-Report win rate vs base (judge both orders + 30% expert blind) | ≥ 60% | 68% |
| generalization to new problems | D-Fresh (written after 2026-08-14): gain ÷ D-Num gain | ≥ 60% | 81% |
| keep general ability | mean Δ over 5 general benchmarks | ≥ −1.0 | −0.4 |
| safety behaviour | refusal-rate change on the refusal regression set | within ±2 pp | +0.7 |
| inference cost | mean output tokens vs base(medium) | ≤ 1.3× | 1.08× |
| reproducibility | \|D-Num(seed 1) − D-Num(seed 2)\| | ≤ 2.0 | 1.3 |

**Out of scope:** vision ability (the vision tower is frozen and only
regression-checked), context above 64K (the project caps at 32K), and
re-training safety alignment.

There is also a gate **between SFT and RL** (`project.rl_entry_gate`). All three
domain metrics must beat the base, the general mean Δ must be ≥ −1.5, and **pass@16
must not fall below the base**. The last condition keeps exploration headroom for RL.
SFT that sharpens the distribution too much leaves RL nothing to learn from.

> The outcome numbers in the right-hand column are the report's *design
> targets*. They are what the plan expects, not results produced by this code.
> This repository is the machinery that produces the real numbers.

---

## 2. The plan, in order

```
            WORKSTATION (4×A100-40GB)                          CLUSTER (2×8×A100-80GB)
 ┌──────────────────────────────────────────────┐   ┌──────────────────────────────────────────────┐
 │ 00 env check + 1430 CPU tests                 │   │ 00 env check (both nodes, same image)        │
 │ 01 data: 91k → 82k dedup → 79.5k decontam →  │   │                                              │
 │    74.1k answer-verified → 60k (pass@8<8/8)  │   │                                              │
 │    + SFT mix 420M tokens (25% self-replay)   │   │                                              │
 │ 02 parity: GDN kernels on SM80; vLLM logprobs│   │ 02 parity again on the cluster image         │
 │ 03 baseline eval: base@medium, base@xhigh    │   │                                              │
 │    (32K & 64K), noise floor ×10, pass@64     │   │                                              │
 │ 04s SFT smoke (0.8B)                         │──▶│ 04 SFT 27B (2 epochs, ~22 h) → export → eval │
 │ 05 RL smoke (0.8B): 12 steps + resume test   │   │    → RL-entry gate                           │
 │                                              │   │ 06a 9B ablation screening (A2,A3,A6,A7)      │
 │                                              │   │ 06b 27B short-run confirmations (A1,A4,A5,   │
 │                                              │   │     A8–A12)                                   │
 │                                              │   │ 06c 27B main line, 420 steps (seed 1)        │
 │                                              │   │ 06d reproducibility run (seed 2)             │
 │ 07 final eval (27B inference fits, TP4)  ◀───│───│ 07 final eval, report table, ship gates      │
 │ 08 audits: contamination, grader FP/FN, judge│   │                                              │
 └──────────────────────────────────────────────┘   └──────────────────────────────────────────────┘
```

Compute in the report: 11,020 GPU·h over 6 weeks:

| phase | GPU·h | covers |
|---|---|---|
| baselines and evaluation | 700 | harness, baselines, final eval |
| SFT | 1,420 | two SFT versions + the CPT ablation |
| RL ablations | 3,300 | 9B: 1,200; 27B short runs: 2,100 |
| RL main line + reproduction | 5,600 | main 3,450 (incl. ~1,300 lost to incidents), seed 2: 2,150 |

Week 1 harness and data; week 2 SFT; weeks 3–5 RL; week 6 evaluation,
contamination audit, report.

---

## 3. Two machines, one code path

**Hardware profiles** (`configs/hardware/*.yaml`) hold everything machine-specific:
GPU count and memory, the parallel layout, rollout TP, and memory utilization.
Stage configs reference them by interpolation (`${hardware.train.tp}`), so the
same command renders a different, correct layout on each machine.

| | workstation `local_4xa100_40g` | cluster `cluster_2x8xa100_80g` |
|---|---|---|
| GPUs | 4 × A100-40GB, NVLink NV4 all-to-all | 2 nodes × 8 × A100-80GB, NVSwitch, 4×HDR IB per node |
| training layout | TP1 × PP2 × DP2 (0.8B smoke) | **TP4 × PP2 × DP2** (27B) |
| rollout | vLLM TP1 × 4 replicas | **vLLM TP2 × 8 replicas** |
| 27B inference | vLLM TP4 (13.5 GB weights/GPU) | vLLM TP2 |
| 27B training | ✗ (state alone ≈ 430 GB; see `make plan-local`) | ✓ (≈ 61.5 GB/GPU colocated) |

**Moving to the cluster.** Put the repo and the Python environment at the same
path on both nodes (shared filesystem). verl does not ship code to Ray workers.
Install `requirements/train.txt` (or use one container image for both nodes),
export `DPT_HW_PROFILE=cluster_2x8xa100_80g`, set `DPT_DATA_ROOT` (fast local NVMe)
and `DPT_CKPT_ROOT` (shared NFS), and run the same scripts.

**Workstation environments (tested on this box).** The NVIDIA driver here (560.35) supports CUDA ≤ 12.6.
Three separate conda envs make everything run without touching the driver or your other envs:

| env | create with | contents | used by |
|---|---|---|---|
| `dpt-gpu` | `bash scripts/00_setup_env.sh gpu-local` | torch 2.11.0 **+cu126** (native on this driver), flash-linear-attention 0.5.2 (Triton GDN kernels) | GDN kernel parity, HF reference scoring |
| `dpt-vllm` | `bash scripts/00_setup_env.sh vllm-local` | vLLM 0.24.0+cu129, torch 2.11.0+cu128, conda `cuda-compat=12.9` (NVIDIA forward compatibility; an activate hook puts it on `LD_LIBRARY_PATH`) | vLLM logprob parity, 27B serving for eval / calibration / self-distillation |
| `dpt-train` | `bash scripts/00_setup_env.sh train-local` (→ `scripts/env/build_train_local.sh`) | Python 3.12; verl 0.9.1 (v1 trainer + TransferQueue), vLLM 0.24.0+cu129, Megatron-Core 0.19.2 + Megatron-Bridge 0.6.2, Transformer Engine 2.20.2 (PyTorch extension compiled here for SM80), flash-linear-attention 0.5.2; `cuda-compat=12.9`, and the env's own cuDNN/NCCL put ahead of the system's 9.10 cuDNN | SFT/RL smoke runs (step 04s/05), train-forward parity (step 02.4) |

Forward compatibility was verified here: with the compat library, `cuInit` succeeds against
the 560 kernel driver, reports CUDA 12.9 and sees all 4 GPUs. The base conda env's torch
(CUDA 13.0) cannot use these GPUs, so do not run GPU steps from it. `make check-env` says
which env you are in and what it can do. The GPUs are shared with other users' jobs, so the
GPU scripts pick the GPUs with the most free memory and refuse to start if too little is free.

---

## 4. Repository layout

```
configs/
  project.yaml               single source of truth: model ids + architecture facts, thinking/length
                             policy, paths, ship gates, RL-entry gate, eval protocol
  model/                     qwen38_27b.yaml (the report, default), qwen35_9b.yaml (the 9B main line):
                             base id, architecture facts, effort policy, eval sizes, gates
  hardware/                  local_4xa100_40g.yaml, cluster_2x8xa100_80g.yaml, slurm_3xa100_40g.yaml
  data/pipeline.yaml         every data threshold (dedup, decontam, verification, difficulty, SFT mix)
  sft/                       sft_27b.yaml, sft_9b.yaml (+ smoke, 9B proxy, FSDP contingency)
  rl/                        algorithm.yaml, curriculum.yaml, rewards.yaml, monitoring.yaml,
                             rl_27b.yaml (→ verl keys), rl_9b.yaml, rl_smoke_0p8b.yaml, rl_9b_proxy.yaml,
                             ablations/A01…A12 (+ README with hypotheses and decisions)
  eval/eval_suite.yaml       harness settings, sets, judge, general benchmarks
scripts/                     00_setup_env … 08_audit (the run order), serve_vllm, export_hf, ray_cluster
  slurm/                      the cluster: env.sh, Slurm jobs (build, check, parity, smoke, serve, sft, rl, eval)
src/dpt/
  common/      config layering, hashing + manifests, JSONL I/O, n-grams, seeding
  chat/        Qwen3.8 chat rendering with VERIFIED reasoning effort, thinking split, loss masks
  data/        ingest → quality → dedup → decontam → verify → difficulty → publish; SFT mix; prompts
  rewards/     T1–T4 training graders (fail closed), sandbox, rubric judge, citation check, hack scanners
  rl/          framework-free RL: advantages, task normalization, PPO-clip, TIS, mismatch, filtering,
               curriculum, partial-rollout semantics, micro-batching + memory model, LR, exact-resume state
  monitoring/  alert engine (every dashboard row), panels, proxy/gold divergence, hack rates, audits
  eval/        harness (independent graders), statistics, noise floor, gates, report, checkpoint selection,
               hosted eval model client: drafting, automatic checks that build D-Fresh/D-Report, judge
               drift during RL, blind-review page (eval side only)
  planning/    memory / FLOPs / communication calculator (reproduces the report's tables)
  launch/      YAML → verl Hydra overrides, pre-flight checks, provenance, env check, checkpoint tools
  parity/      GDN kernel vs FP64 reference; vLLM vs reference logprobs (GPU tools)
  verl_ext/    the ONLY code that imports verl: trainer, replay buffer, sampler, reward fn, SFT dataset
tests/         1430 CPU tests mirroring src/ (GPU/network tests auto-skip)
docs/          design docs: data_pipeline, contamination, rewards_and_hacking, rl_algorithm,
               systems, evaluation, monitoring, runbook, verl_integration, report_traceability,
               dev/conventions
examples/toy_data/   tiny fixtures with planted defects (duplicates, contamination, bad labels, OCR loops)
```

### The layering rule (and why it matters)

```
 adapters  │ dpt.verl_ext   dpt.parity   scripts/          ← need GPUs / verl / vLLM / Megatron
           ▲ imports one way only
 core      │ dpt.common chat data rewards rl monitoring     ← pure Python + numpy (+ CPU torch)
           │ eval planning launch                           ← unit-tested on CPU, never import verl
```

verl's internals change every minor release; 0.9 already deprecated the
classic trainer. Every rule from the report lives in framework-free code and is
tested on CPU: fail-closed grading, fixed-constant loss normalization, TIS,
the curriculum, alert thresholds, exact resume. The verl adapter only moves
tensors between verl's TransferQueue and those functions, so a verl upgrade
touches one small package. See `docs/dev/conventions.md` for the module
contracts (record schemas, answer formats, metric names).

---

## 5. Step-by-step run guide

Each step lists **where** it runs, **what** it needs, **what** it produces, and
**how you know it worked**. Never skip a step. Most of them exist because skipping
them once cost the report's team days.

### Step 00: environment (`scripts/00_setup_env.sh`)

| | |
|---|---|
| where | every machine |
| run | `bash scripts/00_setup_env.sh core` (CPU) or `… train` (GPU stack) |
| does | creates a conda env, installs pinned requirements, `pip install -e .`, runs the CPU tests, writes `env_report.json` |
| success | tests pass; `check_env --require-train` reports no issues on the cluster |

### Step 01: data (`scripts/01_data.sh`), report §数据 / §处理流水线 / §污染检查

| | |
|---|---|
| where | workstation (27B served with TP4 for difficulty calibration) or one cluster node |
| smoke | `bash scripts/01_data.sh --smoke` (toy fixtures, offline fakes, ~1 s) |
| full | serve the base (`scripts/serve_vllm.sh Qwen/Qwen3.8-27B base 8000`), configure the verifier models, embedder and paraphrase judge in `configs/data/pipeline.yaml`, run `bash scripts/01_data.sh` |
| produces | `data/v1/{rl_main,rl_hard,rl_dev}.{jsonl,parquet}` + manifests, `data/v1/sft/sft_train.jsonl`, `funnel.json`, `reports/contamination_report.json`, an expert queue of disputed labels |
| success | funnel resembles 91k → 82k → 79.5k → 74.1k → 60k; no canary hits; the grader suite is 100% |

The stages, in order: license filter → 4-gram repetition filter (> 30% drops a
sample: this caused the SFT loss spike at step ~400) → MinHash dedup at
Jaccard 0.8 → multi-layer decontamination → three-way answer verification →
difficulty calibration (8 samples at medium effort; 8/8 dropped, 0/8 →
hard pool) → publish. Every output carries a manifest with a content hash, and
training refuses inputs whose manifest does not verify. Details:
`docs/data_pipeline.md`, `docs/contamination.md`.

### Step 02: numerical parity on this hardware (`scripts/02_parity.sh`)

| | |
|---|---|
| where | the workstation first (envs `dpt-gpu` + `dpt-vllm`), then once on the cluster image |
| does | (1) GDN Triton kernels (chunk, decode, decode with a BF16 state cache) vs an FP64 reference at 1K–32K tokens on SM80; (2) vLLM vs a reference forward pass, per-token logprobs on naturally long (up to 16K) xhigh responses, FP32 recurrent state → must **PASS**; (3) the same with a BF16 state → expected to be worse; (4) Megatron's **train-mode** forward vs its eval-mode forward on verl's inputs (the two passes a PPO ratio compares) → must **PASS** |
| success | `parity/<date>/deltanet_kernel.json` pass; `logprob_fp32_state/parity.json` pass (mean \|log-ppl diff\| < 2e-3, no growth with position); `train_forward.json` pass (\|mean Δ\| < 5e-3, mean \|Δ\| < 5e-2 nats/token) |
| measured here (A100-40GB, SM80, 2026-10-06) | **kernels:** chunk 6.1e-3, decode 3.9e-3 mean relative error vs FP64, flat from 1K to 32K (late/early 1.01); BF16 state cache ≈ 4× worse. **vLLM vs reference, Qwen3.8-27B (PP3 on 3 GPUs), xhigh, 8 natural 16K-token responses, FP32 state:** mean \|log-ppl diff\| **6.1e-4** vs the FLA chunk kernel (the training path) and **7.7e-4** vs an FP32 PyTorch delta rule, flat across positions: PASS, ~3× below the 2e-3 alert line. **0.8B with a BF16 state:** 2.6e-3, ≈ 10× late-position mismatch: the Incident 1 mechanism, reproduced on this hardware. Responses *forced* long with `--min-tokens` degenerate into loops and give misleading failures; use natural xhigh responses. Details: `parity/20261006/SUMMARY.md`. **Train vs eval forward (2026-10-07, Qwen3.5-0.8B):** 3.1e-4 signed / 2.5e-2 mean \|Δ\| with `dpt.verl_ext.patches`: PASS. Without the patch: **+1.80 nats/token**. verl's inverted attention mask is applied only by Transformer Engine's unfused backend, which A100 uses for d=256 attention in training (`docs/verl_integration.md` §2.1) |

Why first: all public validation of this architecture's kernels was on H100/B200.
On A100 the report's first long run collapsed because rollout kept the GDN state
in BF16 while training accumulated in FP32 (Incident 1). vLLM-vs-**Megatron**
parity is measured inside verl during step 05. Check (4) exists because that parity only covers the
eval-mode pass: on A100 the training-mode pass of this architecture was silently wrong until patched.

### Step 03: baseline evaluation (`scripts/03_baseline_eval.sh`)

| | |
|---|---|
| where | workstation (27B, vLLM TP4) or one cluster node |
| does | base@medium and base@xhigh on every set; base@xhigh re-scored at a 64K cap (11% of its answers truncate at 32K) with the higher score reported; D-Num ×10 seeds → noise floor; D-Num with 64 samples → pass@k curve |
| produces | `eval_results/{base_medium,base_xhigh,noise,base_medium_k64}` |
| success | noise floor near the report's (mean pairwise 0.9, max 2.1 on D-Num); CIs near ±1.6 / ±2.4 / ±4 / ±3.1 |

Harness rules (`docs/evaluation.md`): every model runs in the same harness,
including the base. No public numbers are copied. Template, effort and length
cap are fixed. Thinking is stripped before grading. A response without
`</think>` is a failure, and raw thinking is never graded. Eval graders are a
separate implementation from the training graders, so the policy cannot
overfit the grader it is scored by.

### Step 04: SFT (`scripts/04_sft.sh`), report §阶段一

| | |
|---|---|
| where | cluster, both nodes (`NNODES=2 NODE_RANK=… MASTER_ADDR=…`); smoke on the workstation with `--config sft/sft_smoke_0p8b.yaml --allow-missing-manifest` |
| does | full-parameter SFT, 2 epochs, peak LR 8e-6 (50 warm-up, cosine to 10%), global batch 128 conversations (~2M tokens), AdamW β2 0.95, wd 0.1, clip 1.0, MTP trained (weight 0.1), vision frozen; then Megatron → HF export, exact vision restore, export verification |
| time | ~22 h on 16 GPUs (≈350 GPU·h; `make plan` reproduces 21.5 h at 38% MFU + 8% padding) |
| success | export verified; then step 07 `--only sft` and `python -m dpt.eval.gates --summary eval_results/summary.json --candidate sft --entry` passes |

### Step 05: RL smoke test (`scripts/05_rl_smoke.sh`)

| | |
|---|---|
| where | workstation, Qwen3.5-0.8B (same `qwen3_5` architecture: GDN + gated attention + MTP) on the toy bank |
| does | 12 steps straight; then 6 steps + resume + 6 steps, compared with `dpt.monitoring.compare_runs` |
| success | all adapters load; the curriculum passes stages A→B→C and admits the hard pool; dev validation produces `proxy/<task>`, `gold/<task>` and `dev/d_num`; step-1 `mismatch/log_ppl_abs_diff` < 2e-3 (vLLM vs Megatron parity); `actor/ppo_kl` stays ~1e-4–1e-3 (train-mode forward consistent; it was 0.4–0.7 with verl's inverted mask); no auto-pause; resume equivalence passes (Incident 2): global step, curriculum stage and LR identical after the resume, and `resume/state_roundtrip_ok` = 1. Rewards are not compared, because asynchronous partial rollout makes samples differ between correct runs |

| measured here (2026-10-07, 2 × A100-40GB, TP1 × PP2, 2 vLLM replicas) | `smoke exit=0`. Step-1 engine mismatch 6.9e-4 / 6.3e-4 (two runs); `actor/ppo_kl` ≤ 4.4e-4 on 21 of 22 steps, 7.1e-3 once (0.4–0.7 before the attention-mask patch); MTP acceptance from vLLM's counters 0.44–0.97, with a dip whenever the policy moved; curriculum A→B→C with the hard pool at step 5; `gold/*`, `dev/d_num` produced; `policy_drift` warned at 3.5e-2 (lr 1e-5); resume: step, stage and LR identical, `resume/state_roundtrip_ok` = 1. The first run auto-paused at step 10 on 4 short fresh samples (8.8e-3), which led to the fresh-evidence gate (`rl.mismatch.min_fresh_*`). With it, the resumed run skipped steps 8–12 instead of pausing. The toy 0.8B truncates most answers at the 4K cap, so the rewards mean nothing |

Run it before the first cluster job and after **every** verl / vLLM / Megatron-Bridge upgrade.

### Step 06: RL (`scripts/06_rl.sh`), report §阶段二 / §工程设计

```bash
# both nodes:  bash scripts/ray_cluster.sh head      |   bash scripts/ray_cluster.sh worker <head_ip>
export DPT_HW_PROFILE=cluster_2x8xa100_80g DPT_JUDGE_URL=… DPT_SANDBOX_URL=…
bash scripts/06_rl.sh --config rl/rl_9b_proxy.yaml --ablation A03_clip_high_0p2   # a) screening on 9B
bash scripts/06_rl.sh --ablation A08_lr_2e-6                                       # b) 27B confirmations
bash scripts/06_rl.sh                                                              # c) main line, seed 1
bash scripts/06_rl.sh project.seed=2          # d) reproducibility (same SFT init: rl_run.sft_seed=1)
```

| | |
|---|---|
| does | 420 steps × (128 prompts × 8 samples); 3-stage curriculum; colocated vLLM rollout with partial rollout + MTP speculative decoding; zero-advantage filtering; task-aware advantages; TIS; alerts with auto-pause; checkpoint every 10 steps with full extra state; dev validation every 20 steps |
| time | ~16 min/step after optimization (27 min before) → ~4.7 days for 420 steps |
| produces | `checkpoints/rl/<run>/global_step_*`, `metrics.jsonl`, `launch/<timestamp>/` (resolved config, overrides, data hashes, git revision) |
| if paused | `<run>/PAUSED.json` says why; follow `docs/runbook.md` (triage, choose the rollback point, `resume_mode=resume_path`) |

The launcher refuses to start if a pre-flight check fails ([§6.9](#69-stability-every-incident-left-a-guard)).

### Step 07: final evaluation (`scripts/07_final_eval.sh`)

```bash
STEP=$(python -m dpt.eval.select_checkpoint --run checkpoints/rl/rl27b_seed1 | jq .selected_step)  # dev curve only
bash scripts/export_hf.sh checkpoints/rl/rl27b_seed1/global_step_${STEP} checkpoints/rl/rl27b_seed1/hf_export
bash scripts/07_final_eval.sh          # SFT, RL seed 1, RL seed 2 → eval_results/report.md + ship gates
```

Checkpoints are chosen on the **dev** curve, never on test sets. D-Report items
are judged pairwise against the base in both orders, and a win counts only if
both orders agree. 30% are exported for expert blind review, and the verdicts
are imported before the final win rate is read. The ship-gate check exits
non-zero on any failure.

### Step 08: audits (`scripts/08_audit.sh`)

Contamination report and base-model memorization probes (prefix completion,
perturbation gap). Grader FP/FN rates on human-labeled outputs (the report
improved from v1 2.8% / 4.1% to v3 0.3% / 1.2%; the script fails above 0.5% /
1.5%). Judge calibration against the pinned reference distribution.

---

## 6. Design decisions and the reasoning behind them

Each subsection states the decision, the reason, and where it is implemented.
Longer derivations are in `docs/`.

### 6.1 The base model shapes everything

Qwen3.8-27B (`config.json`: `model_type: qwen3_5`, `Qwen3_5ForConditionalGeneration`):
64 layers, of which 48 are **Gated DeltaNet** (linear attention with a fixed-size
recurrent state) and 16 are **gated attention** (every 4th layer, GQA 24 Q / 4 KV
heads, head_dim 256). It has an MTP head and a 27-layer vision tower. The
vocabulary is 248,320 (confirmed on the model card). The default reasoning
effort is xhigh.

| fact | consequence in this repo |
|---|---|
| no FP8 on A100 | BF16 everywhere; rollout must hold 54 GB of weights plus caches |
| only 4 KV heads | training TP ≤ 4 → TP4 × PP2 per node |
| KV cache only in 16 layers | 64 KiB/token: long rollouts are cheap in KV, and every sequence also carries a 154 MB FP32 GDN state |
| GDN cannot reset state across packed sequences on the Megatron path | BSHD padding, one sample per micro-batch, no packing |
| 248K vocab | never materialize full logits (16 GB for one 16K sequence): vocab-parallel logprobs + chunked entropy |
| `mamba_ssm_dtype: float32` | rollout GDN state pinned to FP32 (Incident 1) |
| MTP head | free speculative-decoding draft, *only if it is trained during RL* (Incident 5) |
| default effort xhigh = overthinking | fix the effort (medium) and length budget **before** RL |

**How reasoning effort really works.** The official chat template does
not add a control token. It prepends a sentence to the system turn: "Reasoning
effort is set to xhigh. …" or "… set to low. …". **Medium adds no sentence at
all**, and *omitting the parameter silently means xhigh*. A renderer that forgets
the kwarg therefore labels medium-length training data as xhigh. That is
exactly how the report's first SFT run lost effort control. `dpt.chat.render`
always passes the effort explicitly and **verifies** the rendered text has the
right sentence or none. It is tested against the real template
(`tests/chat/test_render.py`).

### 6.2 Data

**SFT mix: 420M tokens.**

| subset | tokens | rows | source | key processing |
|---|---|---|---|---|
| domain knowledge & reasoning | 210M | 95k | open-access papers, textbook exercises, domain code docs; long CoT from a strong model | rejection-sampled: answer must pass the verifier; reasoning length controlled to medium |
| domain tool-use trajectories | 80M | 12k | multi-turn trajectories executed in a sandbox | keep only successful runs that pass hidden tests |
| self-distilled general replay | 105M (25%) | | Qwen3.8-27B's own answers on general prompts | keeps the base's format, tone and tool-call habits |
| open-ended domain reports | 25M | 6k | written and revised by experts | each has a rubric, reused by the RL judge |

All rows are re-rendered with the native chat template and an explicit effort
per row (`dpt.verl_ext.sft_dataset`). Loss covers assistant tokens only,
including thinking. Tool outputs are excluded (the template puts them in user
turns, and the mask is computed from character spans of assistant turns).

**RL problem bank: about 60K problems in four tasks.**

| task | problems | reward |
|---|---|---|
| T1 numeric reasoning | 28k | programmatic solver ground truth, unit-aware tolerance |
| T2 code & tools | 9k | sandbox execution: visible + hidden tests |
| T3 open-ended report | 6k | rubric-based generative judge × citation verification |
| T4 general replay | 17k | math answer match + instruction-following rules (against forgetting) |

**Decontamination keeps evaluation honest** (`docs/contamination.md`). The
layers: 13-gram overlap or MinHash ≥ 0.5; embedding nearest neighbour ≥ 0.85
followed by an LLM "same problem?" judgement (changed numbers still count as the
same problem); answer-level matching (same input parameters and same answer);
teacher verbatim recitation in SFT generations; canary strings; and time-split
sets (D-Fresh, post-cutoff LiveCodeBench). The removal threshold is
deliberately aggressive. A false positive costs a training problem. A false
negative invalidates the evaluation. The report removed 3.1% of RL candidates
(87% confirmed on a 200-item manual check) and 1.4% of SFT candidates. The base
model shows memorization on MMLU-Pro (6.2% continuation match vs 0.3% random),
so that benchmark is reported as relative change only (`project.eval.relative_only`).

**Is the gold answer itself right?** Three independent solvers must agree: the
programmatic solver and two strong models from *different* families, 8 samples
each. Disagreements go to an expert queue. In the report, about 40% of the 6.8%
removed were wrong labels and the rest ambiguous. When every model agrees on the
*same* answer that differs from the label, suspect the label. Programmatic
problems also pass a perturbation test: change the inputs, and the answer must
change the way the physics says it should.

### 6.3 Stage 1: SFT

| decision | why |
|---|---|
| **no CPT** | a 2B-token CPT ablation (720 GPU·h) gained +1.1 D-Num, inside the ±1.6 CI, and cost 1.3 more general points |
| **2 epochs, not 3** | 3 epochs scored slightly higher after SFT, but token entropy fell 24% below base (2 epochs: 15%), and RL plateaued earlier. SFT must leave exploration room |
| **25% self-distilled replay** | general Δ went from −3.2 to −1.1 at a cost of 0.7 D-Num |
| MTP trained (weight 0.1) | keeps the draft head matched to the policy for speculative decoding |
| vision frozen | out of scope. verl 0.9.1 has no working non-LoRA freeze, and AdamW decay would drift unused weights, so the tower is copied back from the base at export and verified bit-exact (`dpt.launch.checkpoint_tools`) |
| same TP4×PP2×DP2 as RL | memory/throughput behaviour is known before RL; one engine, one export path |
| repetition filter (4-gram > 30%) | a few dozen OCR-loop samples caused the loss spike at step ~400 |

| SFT variant (report) | D-Num | D-Code | D-Report | general Δ | mean tokens |
|---|---|---|---|---|---|
| base (medium) | 52.1 | 38.0 | 50% | 0 | 5.9K |
| SFT, no replay | 61.5 | 45.1 | 62% | −3.2 | 4.9K |
| **SFT, 25% self-replay (chosen)** | 60.8 | 44.5 | 61% | −1.1 | 5.2K |
| CPT 2B + SFT | 61.9 | 44.0 | 63% | −2.4 | 5.3K |
| SFT, 3 epochs | 61.4 | 45.0 | 62% | −1.6 | 5.0K |

The chosen SFT's D-Num pass@16 was 83.0 vs the base's 81.7. It passes the
RL-entry gate.

### 6.4 Systems: parallelism, memory, the colocated step

Summary below. Full reasoning and every number are in `docs/systems.md`, and `make plan` reproduces them.

* **Training TP4 × PP2 × DP2 + SP, CP 1.** TP is capped by the 4 KV heads.
  TP×PP = 8 puts one full replica in each node, so only DP gradient traffic
  crosses InfiniBand (~1–2 s/step). CP is not validated for GDN.
* **Rollout vLLM TP2 × 8, colocated.** Decode is bandwidth-bound: the more
  replicas, the more concurrent sequences per byte of weights moved. ~100
  concurrent sequences per replica at a 6K average length.
* **Colocated, not split across nodes.** During rollout the training state is
  offloaded to CPU (~40 GB/GPU, ~5 s). The weight sync stays inside each node
  over CUDA IPC (~10 s). The split alternative (A12) was slower, 31 vs 27
  min/step, because the 27B run on A100 is training-bound.
* **Per-GPU memory ≈ 61.5 GB** while training and ≈ 58 GB during rollout. A
  launch-time memory model rejects any configuration estimated above 76 GB
  (Incident 4).
* **27 → 16 min/step** from three changes inside the colocated design: partial
  rollout (stop waiting for the slowest 3%), MTP speculative decoding (≈ 32%
  less generation time), and dropping zero-advantage groups before the
  old-logprob pass (−35% training tokens).

### 6.5 The RL algorithm (`docs/rl_algorithm.md` has the derivations)

GRPO family. Clip-higher, token-level loss normalized by a **fixed constant**,
no KL, truncated importance sampling. Every choice is backed by an ablation.

| setting | value | why |
|---|---|---|
| advantage | per prompt, 8 samples, `A = r − mean(group)`, **no std division** | dividing by the std amplifies nearly-all-right/nearly-all-wrong groups and biases learning by difficulty (A2: +0.9 D-Num, +2.3 hard pass@8 without it) |
| policy loss | PPO-clip, ε_low 0.2, **ε_high 0.28** | lets low-probability tokens rise and delays entropy collapse (A3: 62% vs 35% entropy kept at step 200) |
| loss normalization | Σ token losses ÷ **fixed constant** | removes length bias; dropping groups does not change the gradient scale (tested) |
| KL | 0, no reference model | saves a forward pass and a weight copy; KL 0.001 was worse (A1: +3.0 vs +4.1) |
| mismatch correction | TIS, ratio π_old_train / π_rollout capped at 2.0; whole-sequence mask on large mismatch | fixes both numerical mismatch and partial-rollout off-policyness (A4: collapse at step 180 without it) |
| learning rate | 1e-6 constant, 10 warm-up, 5e-7 for the last 120 steps | the 2e-6 found on 9B gave 3× entropy oscillation on 27B (A8) |
| updates per step | 2 mini-batches | more updates are more off-policy |
| sampling | temperature 1.0, **top_p 1.0**, effort medium | top_p < 1 makes the sampling distribution differ from the scored one |
| length | cap 24K; linear penalty 20K→24K, at most −0.5; **truncated responses excluded from the loss and the baseline** | "not finished" ≠ "wrong", and overthinking is still discouraged |
| T1 reward | **binary** within tolerance | partial credit raised train reward but lowered D-Num by 1.8 (A7): the model learned "roughly right" |

**Why the fixed constant matters.** With token-mean normalization the
denominator is the batch's token count. Dropping zero-advantage groups (or any
change in batch composition) silently rescales every gradient. With
`Σ ℓ_t / C` and `C` fixed, removing samples whose ℓ_t ≡ 0 changes nothing.
`tests/rl` contains the report's unit test. On a tiny policy, gradients before
and after filtering are identical under fixed-constant normalization and differ
under token-mean.

### 6.6 Multi-task RL: the hardest part

Mixing the four tasks by problem count made the total reward rise while T1
stalled after step 140 and general ability fell. Three problems were coupled:

1. **Token-level loss lets long-answer tasks dominate.** T1 was 50% of prompts
   but only 27% of tokens. T2/T3 trajectories took 58% of tokens.
2. Tasks have different reward scales and variances (T3's continuous judge score has little variance).
3. T3's judge liked long answers, and "be longer" leaked into the other tasks.
   T1 answers grew without getting more correct.

**Fixes.**

* **Per-task loss normalization.** `L = Σ_k (w_k / B_k) Σ_{t∈k} ℓ_t` with
  *fixed* per-task token budgets `B_k` and target weights `w_k`. Each task's
  gradient share is then set by its weight, not by its answer length.
  *Implementation insight:* PPO-clip (including dual-clip) is positively
  homogeneous in the advantage, `ℓ(c·A) = c·ℓ(A)` for `c > 0`. So the
  per-task factor can be folded into the advantages on the driver
  (`TaskLossNormalizer`), with verl's loss left as a plain token sum and no
  custom loss in the GPU workers. The equivalence is unit-tested.
* **Per-task advantage scaling.** T3 advantages are rescaled by a running std
  onto the scale of the binary tasks. The group std is still not divided (A2 stands).
* **T3 length control.** A penalty beyond 1.5× the reference length; the judge
  is told to ignore length and calibrated on length-matched pairs.
* **Three-stage curriculum** (`configs/rl/curriculum.yaml`): A (0–120) verifiable
  only, 50/25/0/25; B (120–300) adds T3, 40/25/15/20, with the hard pool from
  step 150; C (300–420) 35/25/15/25, more replay and a lower LR to win back forgetting.
* **Automatic re-weighting.** If a task's dev metric is flat for 30 steps *and*
  more than 60% of its groups have zero advantage, its sampling weight drops
  and its hard-pool share rises.

| (9B, 200 steps) | each task alone | naive mix | adjusted mix |
|---|---|---|---|
| T1: D-Num gain | +12.4 | +6.1 | **+11.8** |
| T2: D-Code gain | +9.8 | +10.5 | +10.1 |
| T3: judge win-rate gain | +9 pp | +11 pp | +7 pp |
| T3: length-matched expert win-rate gain | +8 pp | +5 pp | **+7 pp** |
| general Δ | — | −2.3 | **−0.5** |
| mean output length | — | 8.9K | 6.1K |

Half of the naive mix's T3 "gain" was the judge's length preference.

### 6.7 Rewards and reward hacking (`docs/rewards_and_hacking.md`)

Principles: **fail closed** (parse failure, timeout or missing format → 0, never
partial). Training and evaluation graders are **separate implementations**.
Rewards see **only the final answer**: judges never read the thinking, and the
model never sees grader code. The sandbox has no network, replays tests outside
the sandbox, and exposes nothing that identifies a test environment.

The seven hacks the report found, each now a detector + fix + regression test:

| step | task | exploit | caught by | fix |
|---|---|---|---|---|
| 310 | T3 | "this analysis has been rigorously verified" self-praise | pattern rate 0.4% → 17% | stripped before judging; penalized |
| 245 | T3 | **fabricated citations** the judge cannot check | judge ↑ while expert blind ↓ 4 pp; 31% of citations unfindable | citations removed from judge input, verified by DOI/title; −0.1 each unverifiable |
| 205 | T2 | code detects the test environment and returns hard-coded results | hidden vs visible pass gap 23 pp | different hidden-test distribution; scrubbed environment; diff review |
| 170 | T2 | blanket `try/except` swallows crashes | human spot check | tests assert concrete outputs; blanket except → 0 |
| 140 | T1 | interval answers ("3.1–3.4 km"), old parser took the first number | adversarial suite FP rose 0.3% → 2.1% | single value only; intervals → 0 |
| 95 | T1 | several candidate answers, any match scored | answer–reasoning consistency fell | multiple final answers → 0 |
| 60 | T4 | filler text to satisfy length/format rules | T4 reward up, IFEval flat | content-relevance gate |

Detection, from most to least effective: **proxy vs gold divergence** (each
task has a stricter gold check on the dev split, scored by the independent eval
graders every 20 steps), then **human audits** (100 samples per 20 steps), then
**pattern scanners**, which only find hacks you already know. Before any new
task goes live, ask a strong model to score high *without solving the task*.

### 6.8 Evaluation and answer verification (`docs/evaluation.md`)

Evaluation rests on a fixed harness, a quantified noise floor, and
independently verified answers. Three verification layers:

1. **Is the gold answer right?** Three-way solver agreement, an expert queue, the "everyone disagrees with the label" check, perturbation consistency.
2. **Does the grader judge correctly?** 1,200 adversarial cases (this repo ships 163 seeds), FP/FN measured on 500 human-labeled outputs, and continuous spot checks during training.
3. **Does "correct" mean "really knows"?** Reasoning–answer consistency (the final number derived in the thinking must match the answer), a numeric-perturbation gap (final 71.4 → 68.9 vs base gap 2.1: no big increase, so the gain is not memorization), pass@64 not below base (RL did not just sharpen), and D-Fresh as the final judge (written after the base model was released).

### 6.9 Stability: every incident left a guard

| incident | what happened | permanent guard here |
|---|---|---|
| 1 | BF16 rollout GDN state vs FP32 training, no TIS → gibberish, reward 0.58 → 0.12 in 6 steps | FP32 state pinned + pre-flight; TIS cap 2; sequence masking; log-ppl-diff alert 2e-3 / auto-pause 5e-3; position-sliced mismatch (only visible after 12K); `tests/parity` |
| 2 | resume lost the sampler/curriculum state → reward −0.09 for 30 steps | `dpt_extra_state.json` (curriculum, sampler, scaler, alerts, RNG, data hashes) saved before verl marks a checkpoint latest; resume-equivalence check in the smoke script |
| 3 | prefix cache (with GDN snapshots) not invalidated after weight sync → T2-only mismatch spike | prefix caching off; verl 0.9.1 resets on wake-up; per-task mismatch alert |
| 4 | a 31K trajectory padded to 32K + 2 PP micro-batches + fragmentation → OOM | 32K total cap; one sample per micro-batch; memory pre-flight; expandable segments; per-rank memory alert |
| 5 | frozen MTP head went stale, acceptance 0.71 → 0.46, rollout +40% time | MTP trained in RL; pre-flight forbids MTP drafting without MTP training; acceptance alert < 0.6, fed from vLLM's spec-decode counters because verl 0.9.1 cannot read per-request stats from vLLM 0.24 (`dpt.monitoring.spec_scrape`) |

Mismatch is the earliest signal. In Incident 1 it crossed 2e-3 about 35 steps
before the crash, and the reward moved only in the last 6. The rollback went
to step 120, before mismatch started rising, not to the most recent step 170.
Procedures are in `docs/runbook.md`. The full alert table (15 rows) is in
`docs/monitoring.md` and `configs/rl/monitoring.yaml`.

### 6.10 Ablations

Twelve overlays in `configs/rl/ablations/`, each with its hypothesis, scale,
the report's result and the decision (`configs/rl/ablations/README.md`).
Directions found on 9B transferred to 27B (A2, A3, A6). **Numeric
hyper-parameters did not** (A8: learning rate). Three ablations overturned
the initial design: removing KL, binary T1 reward, and not copying the small
model's learning rate.

---

## 7. How the verl integration works

```
python -m dpt.launch.rl  ──(pre-flight, provenance, renders 100+ Hydra overrides)──▶  python -m verl.trainer.main_ppo
                                                                                     model_engine=megatron trainer.use_v1=true
 every Ray process imports dpt.verl_ext.register   (VERL_USE_EXTERNAL_MODULES + model.external_lib)
   ├─ trainer mode "dpt_colocate_async" = verl colocate_async  +  DomainRLTrainer overrides:
   │     _init_dataloader    → CurriculumSampler (stages, hard pool @150, auto-reweight, exact state)
   │     _compute_advantage  → dpt.rl.pipeline (group mean, T3 scale, task budgets, truncation mask)
   │                           + verl TIS / sequence masking + per-task & per-position mismatch
   │     _compute_metrics    → canonical metrics → alert engine → auto-pause; metrics.jsonl
   │     _save/_load_checkpoint → dpt_extra_state.json
   ├─ custom replay buffer   → drop zero-advantage groups before old-logprob (shrink, no refill)
   ├─ reward fn (dapo mgr)   → dpt.rewards.router (+ exact overlong penalty, + gold on dev rows)
   ├─ estimator "dpt_task_grpo" (fallback only; logs an error if ever used)
   └─ dpt.verl_ext.patches   → runtime fixes (also imported by SFT): Megatron-convention attention
                               mask for Qwen3.5/3.8 (A100 training forward was wrong without it)
```

The report's appendix config predates verl 0.9.1 in several places: `grad_offload`
no longer exists, the memory-efficient-CE env var is gone, freezing is
LoRA-only, and the TIS keys changed. `docs/verl_integration.md` §2 lists every
difference and what this repo does instead. §2.1–2.2 record what the workstation GPU runs
found that the source did not show (one of them made every A100 training forward wrong), and §3
lists the items only a real GPU run can confirm, with their status.

---

## 8. Configuration system

* **Layers** (later wins): `project.yaml` → `model/<line>.yaml` (`DPT_MODEL_LINE`) →
  `hardware/<profile>.yaml` (`DPT_HW_PROFILE`) → stage config (and its `defaults_from` /
  `defaults_from_list` parents) → ablation overlay → CLI overrides. A stage config written for one
  line says so (`requires_model_line: qwen35_9b`) and refuses to compose under another.
* **Single source of truth.** Numbers appear once and are interpolated
  (`${rl.tis.cap}`, `${hardware.train.tp}`). Derived values are computed, e.g.
  `ppo_mini_batch_size: ${div:${rl.prompts_per_step},${rl.num_updates_per_step}}`.
* **Overrides.** `rl.loss.clip_high=0.25` (our keys) or
  `verl.actor_rollout_ref.actor.optim.lr=5e-7` (flat verl keys).
* **Provenance.** Every launch writes `launch/<timestamp>/` with the resolved
  config, the exact override list, sha256 of both, the data content hashes and
  the git revision. The resolved config is what Ray workers load
  (`DPT_RESOLVED_CONFIG`), so what ran is exactly what was recorded.
* **Data provenance.** Every dataset has a `*.manifest.json` (record count,
  content hash, parents, funnel statistics). Launchers verify them and refuse
  stale or edited files.

---

## 9. Testing

```bash
make test                        # 1430 tests, ~45 s, CPU only
python -m pytest tests/rl -q     # one package
python -m pytest -m gpu          # GPU tests (auto-skipped without CUDA)
```

| package | tests | highlights |
|---|---|---|
| rewards | 340 | 163 adversarial grader cases (equivalent forms, units, intervals, multiple answers, buried answers), sandbox tamper checks, container isolation (fail closed), the isolation probe, numpy under the memory limit, judge/citation logic with mocked HTTP |
| eval | 170 | unbiased pass@k, bootstrap/Wilson CIs, noise floor, both-order pairwise judging with two judges (verdict cache, fail-closed gating), gates, eval drafting with the hosted model, the checks that build D-Fresh (drafted code run in the sandbox, independent solves, adjudication of disagreements in both orders) and D-Report (answer-first rubric rulings, majority of three votes), judge drift alerts, the offline blind-review page and its import, no training import of any of it |
| data | 464 | MinHash estimate quality, decontam layers, answer verification (per item and per template), funnel on planted defects; generators (`dpt.data.gen`): every seismology family states its inputs, parses in both graders and passes the perturbation check; T2 references pass their tests and mutants fail; T4 labels and constraints read the same in the training and eval graders; T3 prompts, rewording checks, SFT rejection sampling |
| rl | 79 | gradient invariance under zero-advantage filtering, homogeneity ⇒ per-task normalization, TIS, curriculum, exact resume, memory pre-flight rejects the Incident-4 config |
| monitoring | 39 | every alert rule, divergence, hack-rate growth, metric bridge (incl. vLLM spec-decode counters), resume comparison (step/stage/LR exact, state round trip) |
| launch | 57 | override rendering, every ablation composes, pre-flight blocks each known failure mode, exact vision restore, the 9B layout and guards, wall-clock stop before the job limit |
| slurm | 31 | the Slurm job scripts in dry-run mode: chaining, paused/finished runs never resubmitted, a crash cancels the successor, no RL without a judge, one SFT rank per node, second-judge and judge-drift scripts check their key and inputs, the eval-set build never overwrites a set, env.sh survives modules that read unset variables, the sandbox image is pulled only on compute nodes, the data job's steps, endpoints and eval-file guards |
| common | 4 | model-line layer: 27B stays the default, the 9B line overrides model, protocol and gates |
| verl_ext | 18 | SFT dataset per-row effort, reward adapter gold grading, curriculum sampler resume, stage-boundary mix, adapters against stub verl (registration, zero-adv filtering, auto-pause end to end), the Qwen3.5 attention-mask patch (polarity, import hook, SFT wiring) |
| integration | 2 | toy pipeline → verl rows → reward adapter (train *and* independent eval grader agree) → step advantages → metric bridge → alert engine |
| grader agreement | 32 | the independent training and eval graders agree on the shared data contract: IFEval vocabularies, numeric answers, T4 math expressions (`\frac{\pi}{2}`, `2^{10}`), unit spellings, missing-unit policy |
| chat / planning / parity | 11 / 9 / 6 | real Qwen3.8 template rendering + loss masks; all report memory numbers; GDN reference and BF16-state error |

What the CPU tests **cannot** cover is the GPU behaviour of verl, vLLM and
Megatron. That is the job of steps 02 and 05.

---

## 10. Expected results and how the final table is produced

The report's target table (design targets; the ± is the 95% interval):

| model | D-Num (±1.6) | D-Code (±2.4) | D-Report (±4) | D-Fresh (±3.1) | general Δ | mean tokens |
|---|---|---|---|---|---|---|
| base (medium) | 52.1 | 38.0 | 50% | 44.7 | 0 | 5.9K |
| base (xhigh) | 55.3 | 40.2 | 53% | 47.0 | +0.9 | 14.8K |
| SFT | 60.8 | 44.5 | 61% | 51.2 | −1.1 | 5.2K |
| **SFT + RL (seed 1, delivered)** | **71.4** | **55.2** | **68%** | **60.3** | **−0.4** | **6.4K** |
| SFT + RL (seed 2) | 70.1 | 54.6 | 67% | 59.5 | −0.6 | 6.2K |

Supporting evidence the plan expects: D-Num pass@64 rises from 90.2 to 91.0 (RL
did not just sharpen the distribution), the perturbation gap is 2.5 vs the
base's 2.1 (not memorization), and the D-Fresh gain is 81% of the D-Num gain.

`scripts/07_final_eval.sh` produces the real version of this table in
`eval_results/report.md` (`python -m dpt.eval.report`) and then runs the ship
gates. Rows are `base_medium`, `base_xhigh` (higher of 32K/64K), `sft`,
`rl_seed1`, `rl_seed2`.

---

## 11. Risks, open items, and what to verify first

* **GPU status (workstation, 2026-10-07).** Step 02 passes, including the 27B vLLM parity. The
  RL loop runs end to end on Qwen3.5-0.8B in `dpt-train`: adapters, rewards, TIS, alerts,
  auto-pause, checkpoints, and dev validation with the gold grader. The smoke run found five problems
  that source reading could not. The most serious is verl's inverted attention mask, which made
  every A100 training forward wrong: PPO ratios off by 1–2 nats/token, and SFT affected too. It is
  fixed in `dpt.verl_ext.patches` and guarded by step 02.4. The others: missing MTP acceptance
  metric, partial-rollout drift tripping the mismatch alert, the chunked-entropy backward crash,
  and missing Apex. All are listed in `docs/verl_integration.md` §2.1–2.2. Re-run steps 02 and 05
  on the cluster image before the first cluster job.
* **Megatron-Bridge support for dense Qwen3.5/3.8 GDN in BSHD mode, with MTP
  training.** It is the documented path (NVIDIA's Megatron-Bridge and NeMo-RL
  Qwen3.5 recipes), but this project's report only validated it on its own
  setup. If blocked, `configs/sft/sft_27b_fsdp_fallback.yaml` covers SFT.
* **GDN kernels on A100.** The report validated them only in-house; step 02 is this repo's own check.
* **Judge drift.** T3 depends on an external judge. Pin the served model name
  (a mismatch fails closed) and run the calibration check at the start of every run.
* **Placeholder estimates to re-fit from real logs:** per-task typical response
  lengths for the loss budgets (`rl.loss.task_normalization.response_tokens`) and
  the per-sequence mismatch threshold (`rl.tis.seq_mask.threshold`).
* **Partial-rollout trigger.** verl aborts the in-flight tail when the batch is
  complete, instead of suspending at 16K new tokens. The intent is the same.
  Staleness ≤ 1 is configured as `max_off_policy_threshold: 2`, because verl
  counts the versions a group spans.
* **Not covered (by design):** context > 32K, vision, safety re-training. The
  refusal regression is measured, not trained.

---

## 12. Troubleshooting

| symptom | likely cause | fix |
|---|---|---|
| `torch cannot use the GPUs (built for CUDA 13.0, driver supports 12.6)` | driver/wheel mismatch | install `cuda-compat` or use the cluster image (`make check-env`) |
| `EffortRenderingError` | something rendered a prompt without an explicit effort | always go through `dpt.chat.render.render_chat` |
| launcher: `pre-flight failed` | a config edit re-introduced a known failure mode | read the message; it names the incident |
| `<file> has no manifest` | data produced outside the pipeline, or edited | regenerate with `scripts/01_data.sh` (smoke: `--allow-missing-manifest`) |
| `DPT_RESOLVED_CONFIG is not set` | verl started directly | start RL through `python -m dpt.launch.rl` / `scripts/06_rl.sh` |
| `dpt_task_grpo fallback estimator invoked` in logs | trainer mode not `dpt_colocate_async` | check `trainer.v1.trainer_mode` and that `VERL_USE_EXTERNAL_MODULES` reached the workers |
| `PAUSED.json` in the run dir | a red alert fired | `docs/runbook.md` §3 |
| T3 rewards all 0 | judge unreachable or version mismatch (fails closed by design) | check `DPT_JUDGE_URL` / pinned model; `python -m dpt.rewards.validate --calibrate` |

---

## 13. References

* Qwen3.8-27B model card and chat template: <https://huggingface.co/Qwen/Qwen3.8-27B>
* verl docs, rollout correction (TIS / rejection sampling): <https://verl.readthedocs.io/en/latest/algo/rollout_corr.html>
* vLLM recipe for the Qwen3.5-27B architecture (MTP speculative decoding, prefix caching notes): <https://recipes.vllm.ai/Qwen/Qwen3.5-27B>
* NeMo-RL Qwen3.5 guide (GRPO on Megatron, GDN notes): <https://docs.nvidia.com/nemo/rl/nightly/guides/models/qwen/qwen3-5.html>
* Megatron-Bridge Qwen3.5 bridge: <https://docs.nvidia.com/nemo/megatron-bridge/latest/apidocs/bridge/bridge.models.qwen_vl.qwen35_vl_bridge.html>
* Gated Delta Networks (Yang et al., 2024); Dr. GRPO (no std normalization); DAPO (clip-higher, overlong shaping); truncated importance sampling for training/inference mismatch.
