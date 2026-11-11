# Traceability: report section → config → code → test

Use this table to answer "where is X from the report implemented, and what
proves it works?". Paths are relative to the repo root.

| report section / claim | config | code | test / check |
|---|---|---|---|
| **目标与成功标准**: 8 ship gates, fixed up front | `configs/project.yaml → project.gates` | `dpt/eval/gates.py` | `tests/eval/test_gates_report.py`; `scripts/07_final_eval.sh` |
| RL-entry gate (domain > base, general ≥ −1.5, pass@16 ≥ base) | `project.rl_entry_gate` | `dpt/eval/gates.py --entry` | `tests/eval/test_gates_report.py` |
| 基座: 64 layers = 48 GDN + 16 attention, 4 KV heads, 248K vocab, MTP | `project.model.arch` | `dpt/planning/budget.py` | `tests/planning/test_budget.py` |
| reasoning effort must be fixed before RL; template default is xhigh | `project.thinking.*` | `dpt/chat/render.py` | `tests/chat/test_render.py` (real template) |
| KV 64 KiB/token, GDN state 154/78 MB, 16 GB full logits | — | `dpt/planning/budget.py` | `tests/planning/test_budget.py` |
| **SFT 语料**: 4 subsets, 25% self-distilled replay, native template | `configs/data/pipeline.yaml → sft` | `dpt/data/sft_build.py`, `dpt/verl_ext/sft_dataset.py` | `tests/data/test_sft_build.py`, `tests/verl_ext/test_sft_dataset.py` |
| assistant-only loss incl. thinking; tool outputs excluded | — | `dpt/chat/render.py::tokenize_with_assistant_mask` | `tests/chat/test_render.py` |
| **RL 题库**: T1–T4, ~60K problems | `configs/data/pipeline.yaml` | `dpt/data/*` | `tests/data/test_pipeline.py` (funnel on planted defects) |
| 处理流水线 1: layout-aware OCR, license filter | `data_pipeline.ingest` | `dpt/data/ingest.py` | `tests/data/test_ingest.py` |
| 2: MinHash 5-gram, Jaccard 0.8 | `data_pipeline.dedup` | `dpt/data/dedup.py` | `tests/data/test_dedup.py` |
| 3: decontamination (3.1%) | `data_pipeline.decontam` | `dpt/data/decontam.py` | `tests/data/test_decontam.py` |
| 4: 3-way answer verification (6.8%), expert queue, perturbation consistency | `data_pipeline.verify` | `dpt/data/answer_verify.py`, `dpt/data/solvers.py` | `tests/data/test_answer_verify.py` |
| 5: difficulty, pass@8 at medium, 8/8 dropped, 0/8 → hard pool | `data_pipeline.difficulty` | `dpt/data/difficulty.py` | `tests/data/test_difficulty.py` |
| 6: manifests + content hashes, logged by training | — | `dpt/common/manifest.py`, `dpt/launch/common.py` | `tests/common`, `tests/launch` |
| **阶段一**: SFT 2 epochs, LR 8e-6, batch 128, MTP 0.1, vision frozen | `configs/sft/sft_27b.yaml` | `dpt/launch/sft.py`, `dpt/launch/checkpoint_tools.py` | `make dry-run-sft`; `tests/launch/test_checkpoint_tools.py` |
| loss spike from OCR loops → 4-gram repetition > 30% filter | `data_pipeline.quality` | `dpt/data/quality.py`, `dpt/common/text.py` | `tests/data/test_quality.py` |
| effort control lost after SFT → explicit, verified effort per row | — | `dpt/chat/render.py`, `dpt/verl_ext/sft_dataset.py` | `tests/chat`, `tests/verl_ext/test_sft_dataset.py` |
| **工程设计**: verl + Megatron (mbridge) + vLLM, colocated | `configs/rl/rl_27b.yaml → verl.*` | `dpt/verl_ext/*`, `dpt/launch/rl.py` | `tests/launch/test_rl_launch.py`, `tests/verl_ext/*`; `scripts/05_rl_smoke.sh` |
| parity test first on A100 (vLLM vs Megatron logprobs, GDN Triton on SM80) | — | `dpt/parity/*` | `tests/parity/*`; `scripts/02_parity.sh` |
| 并行: TP4×PP2×DP2 + SP, CP 1; vLLM TP2×8 | `configs/hardware/cluster_2x8xa100_80g.yaml` | rendered by `dpt/launch/rl.py` | `tests/launch/test_rl_launch.py` |
| BSHD padding, no packing (GDN) | `verl.actor_rollout_ref.model.use_remove_padding=false` | — | dry run |
| chunked/vocab-parallel logprob | `entropy_from_logits_with_chunking` | — | see `docs/systems.md` §3 (deviation) |
| 显存规划 ≈ 61.6 GB train / 58 GB rollout | — | `dpt/planning/budget.py`, `dpt/rl/microbatch.py` | `tests/planning`, `tests/rl/test_microbatch.py` |
| 单步 27 → 16 min: partial rollout, MTP spec decoding, zero-adv filtering | `trainer.v1.trainer_mode`, `model.mtp.*`, custom replay buffer | `dpt/verl_ext/replay_buffer.py`, `dpt/rl/filtering.py` | `tests/rl/test_filtering.py`, `tests/verl_ext/test_adapters_with_stubs.py` |
| filtering requires fixed-constant normalization (unit test) | `loss_agg_mode: token-sum` | `dpt/rl/loss.py`, `dpt/rl/advantage.py` | `tests/rl/test_loss.py` (gradient invariance) |
| 分节点异步 slower → colocated | ablation A12 | — | `configs/rl/ablations/A12_separate_async.yaml` |
| **算法配置**: group mean, no std | `rl.advantage` | `dpt/rl/advantage.py`, `dpt/rl/pipeline.py` | `tests/rl/test_advantage.py`, `tests/rl/test_pipeline.py` |
| PPO-clip 0.2 / 0.28 | `rl.loss` | `dpt/rl/loss.py` (reference) | `tests/rl/test_loss.py` |
| KL 0, no reference model | `rl.loss.kl_coef`, `use_kl_loss=false` | — | pre-flight warning |
| TIS cap 2.0 + sequence masking | `rl.tis` → `algorithm.rollout_correction.*` | `dpt/rl/tis.py` (reference) | `tests/rl/test_tis_mismatch.py`; pre-flight |
| LR 1e-6, 10 warm-up, 5e-7 last 120 steps | `rl.lr_schedule` → Megatron WSD | `dpt/rl/schedule.py` | `tests/rl/test_state.py` (schedule) |
| 2 updates per step; T=1.0, top_p 1.0, medium | `rl.*` | — | pre-flight (top_p) |
| length: cap 24K, 20K→24K penalty ≤ 0.5, truncated excluded from loss | `rl.length`, DAPO overlong buffer | `dpt/rl/pipeline.py`, `dpt/rewards/length.py` | `tests/rl/test_pipeline.py`, `tests/rewards/test_length.py` |
| **各任务 reward** T1/T2/T3/T4 and hard rules | `configs/rl/rewards.yaml` | `dpt/rewards/*` | `tests/rewards/*` (330 tests) |
| **三段课程** 0–120 / 120–300 / 300–420, hard pool at 150 | `configs/rl/curriculum.yaml` | `dpt/rl/curriculum.py`, `dpt/verl_ext/sampler.py` | `tests/rl/test_curriculum.py`, `tests/verl_ext/test_sampler.py` |
| **评测协议**: avg@8, pass@1×4, pairwise both orders, 30% expert blind, CIs | `configs/eval/eval_suite.yaml`, `project.eval` | `dpt/eval/*` | `tests/eval/*` |
| harness rules: same harness for all, base@xhigh 64K, strip thinking, no `</think>` = fail | — | `dpt/eval/harness.py`, `dpt/chat/thinking.py` | `tests/eval/test_harness.py` |
| noise floor: 10 repeats (0.9 mean / 2.1 max) | — | `dpt/eval/stats.py`, `dpt/eval/noise_floor.py` | `tests/eval/test_metrics_stats.py`; `scripts/03_baseline_eval.sh` |
| 第一层 gold answer verification | — | `dpt/data/answer_verify.py` | `tests/data/test_answer_verify.py` |
| 第二层 grader validation: adversarial set, FP/FN vs human labels | — | `dpt/rewards/validate.py`, `tests/rewards/adversarial_cases.jsonl` | `make graders`; `scripts/08_audit.sh` |
| 第三层: reasoning–answer consistency, perturbation gap, pass@k, D-Fresh | — | `dpt/eval/consistency.py`, `dpt/eval/metrics.py` | `tests/eval/test_consistency.py` |
| **消融** A1–A12 | `configs/rl/ablations/*` | `dpt/launch/rl.py --ablation` | `tests/launch/test_rl_launch.py::test_each_ablation_composes` |
| **事故 1**: BF16 rollout state + no TIS | FP32 state, TIS | `dpt/launch/rl.py` pre-flight; `dpt/rl/mismatch.py` | `tests/parity/test_deltanet_reference.py`, pre-flight tests |
| **事故 2**: resume lost sampler/curriculum state | — | `dpt/rl/state.py`, `dpt/verl_ext/trainer.py` | `tests/rl/test_state.py`, `tests/monitoring/test_compare_runs.py`; smoke resume test |
| **事故 3**: prefix cache not invalidated | prefix caching off | verl resets on wake-up; per-task mismatch | alert rules |
| **事故 4**: OOM from padding + fragmentation | 32K cap, expandable segments | `dpt/rl/microbatch.py` memory model | `tests/rl/test_microbatch.py`; pre-flight |
| **事故 5**: frozen MTP head → acceptance drop | `model.mtp.enable_train` | pre-flight; bridge `spec/mtp_acceptance_rate` | `tests/launch/test_rl_launch.py`, `tests/monitoring/test_bridge.py` |
| **排查方法** (5 steps) and rollback rule | — | — | `docs/runbook.md` §3 |
| **多任务**: per-task normalization, T3 scaling, length control, delayed T3, auto-reweighting | `rl.loss.task_normalization`, `rl.advantage`, `curriculum.auto_reweight` | `dpt/rl/advantage.py`, `dpt/rl/curriculum.py` | `tests/rl/test_advantage.py` (homogeneity ≡ normalization), `tests/rl/test_curriculum.py` |
| per-task panels; gradient-share decomposition | `configs/rl/monitoring.yaml` | `dpt/rl/pipeline.py`, `dpt/monitoring/panels.py` | `tests/monitoring/test_panels_and_trackers.py` |
| **Reward hacking** 7 cases, detection system | `rewards.*`, monitoring rules | `dpt/rewards/hack_patterns.py`, `dpt/monitoring/divergence.py`, `hack_rate.py`, `audit.py`, gold on dev in `dpt/verl_ext/reward_fn.py` | `tests/rewards/test_hack_patterns.py`, `tests/monitoring/*`, `tests/verl_ext/test_reward_fn.py` |
| **污染检查** 7 methods, results table, canaries | `data_pipeline.decontam` | `dpt/data/decontam.py` | `tests/data/test_decontam.py`; `scripts/08_audit.sh` |
| **默认监控面板** (all rows) | `configs/rl/monitoring.yaml` | `dpt/monitoring/alerts.py`, `dpt/monitoring/bridge.py` | `tests/monitoring/test_alerts.py` |
| 算力与时间 (11,020 GPU·h) | — | `dpt/planning/budget.py` (SFT hours) | README §2 |
