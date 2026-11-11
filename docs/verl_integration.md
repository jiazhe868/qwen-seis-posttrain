# verl integration (pinned: verl 0.9.1)

Everything that touches verl lives in `src/dpt/verl_ext/` and is plugged in
through documented configuration keys. verl itself is never forked or patched.
The adapters were written against the **verl 0.9.1 source** (wheel of 2026-09-20).
The file references below are into that wheel. CPU tests cover the adapters
with stub modules (`tests/verl_ext/test_adapters_with_stubs.py`). The real
behaviour is validated by `scripts/05_rl_smoke.sh` on GPUs. Run that before
the first cluster job and after every upgrade of verl, vLLM, Megatron-Bridge
or TransferQueue.

verl 0.9.1 wheel pins: `vllm==0.24.0`, `torch==2.11.0`, `transformers==5.9.0`.
The v1 trainer needs `TransferQueue>=0.1.10` (extra `verl-core`). The wheel does
**not** pin Megatron-Bridge. This repo uses `megatron-bridge==0.6.2` with
`megatron-core==0.19.2` (Python ≥ 3.12) and Transformer Engine 2.20.2. Version
0.5.2, named in earlier notes, was never published. The exact workstation build,
with every workaround, is `scripts/env/build_train_local.sh` (env `dpt-train`).

## 1. Extension points

| what | how it is plugged in | verl source (0.9.1) | our code |
|---|---|---|---|
| trainer mode `dpt_colocate_async` | `trainer.use_v1=true`, `trainer.v1.trainer_mode=dpt_colocate_async`; registry `register_trainer` | `trainer/ppo/v1/trainer_base.py:1996-2023`, base class `trainer_colocate_async.py:19` | `verl_ext/trainer.py` |
| registration in every process | env `VERL_USE_EXTERNAL_MODULES=dpt.verl_ext.register` (imported at `import verl`, `verl/__init__.py:49-52`) via `+ray_kwargs.ray_init.runtime_env.env_vars.*`; second route `actor_rollout_ref.model.external_lib` (`workers/config/model.py:149`) | | `verl_ext/register.py` |
| zero-advantage filtering (shrink, no refill) | `trainer.v1.sampler.custom_sampler.{path,name}` | `trainer/ppo/v1/replay_buffer.py:139-185, 422` | `verl_ext/replay_buffer.py` |
| task-aware advantages | override `_compute_advantage` (the registered estimator does not receive task ids) | `trainer_base.py:1743-1802` | `trainer.py` → `dpt.rl.pipeline` |
| curriculum sampler | override `_init_dataloader` (`create_rl_sampler` has no hook) | `trainer_base.py:708-760`, `trainer/ppo/utils.py:140-169` | `verl_ext/sampler.py` |
| per-step metrics, alerts, auto-pause | override `_compute_metrics`; stop by setting `total_training_steps = global_steps` | `trainer_base.py:1868`, loop at `:467` | `trainer.py` |
| extra checkpoint state | override `_save_checkpoint` (write *before* `latest_checkpointed_iteration.txt`) / `_load_checkpoint` | `trainer_base.py:850, 948-1030` | `trainer.py`, `dpt.rl.state` |
| reward function | `reward.custom_reward_function.{path,name}`; manager `reward.reward_manager.name=dapo` | `experimental/reward_loop/reward_manager/dapo.py` | `verl_ext/reward_fn.py` → `dpt.rewards.router` |
| overlong penalty | `+reward.reward_kwargs.{max_resp_len, overlong_buffer_cfg}` | `dapo.py:108-117` | config only |
| TIS + sequence masking | `algorithm.rollout_correction.{rollout_is: token, rollout_is_threshold: 2.0, rollout_rs: seq_mean_k3, rollout_rs_threshold}` | `trainer/ppo/rollout_corr_helper.py` | config only |
| SFT dataset (per-row effort) | `data.custom_cls.{path,name}` in `verl.trainer.sft_trainer` | `trainer/sft_trainer.py:468-484` | `verl_ext/sft_dataset.py` |

## 2. Where we deliberately differ from the report's appendix config

| report appendix | verl 0.9.1 reality | what this repo does |
|---|---|---|
| `strategy: megatron` | the CLI selects `model_engine=megatron`; `ppo_megatron_trainer.yaml` is deprecated | `verl.model_engine: megatron` |
| `grad_offload: true` | no such key; grads are offloaded with params | `param_offload` + `optimizer_offload` |
| `megatron.use_remove_padding: false` | overridden by `actor_rollout_ref.model.use_remove_padding` | set on the model key |
| `loss_agg_mode: seq-mean-token-sum-norm` | divides by the *configured* mini-batch sequence count × `loss_scale_factor`, which is also constant | `token-sum`, with per-task constants folded into the advantages (`dpt.rl.advantage.TaskLossNormalizer`) so per-task normalization is possible |
| `VERL_MEGATRON_MEM_EFFICIENT_CE=1` | does not exist; fused linear-CE is auto-disabled with MTP or BSHD | vocab-parallel logprobs + `entropy_from_logits_with_chunking` (docs/systems.md §3) |
| TIS key "varies by version" | `algorithm.rollout_correction.rollout_is=token`, `rollout_is_threshold=2.0` | as stated |
| LR 1e-6, drop to 5e-7 for the last 120 steps | the Megatron optimizer scheduler steps once per trainer step; `WSD` = warm-up → stable → decay | `lr_decay_style: WSD`, `lr_wsd_decay_steps: 120`, `min_lr: 5e-7`, linear ramp (not a step) |
| MTP trained / used | `actor_rollout_ref.model.mtp.{enable, enable_train, enable_rollout, mtp_loss_scaling_factor}` | all on, weight 0.1 |
| freeze vision tower | `freeze_vision_tower` exists but is never read; freezing only on the LoRA path | exact freeze on export: `dpt.launch.checkpoint_tools restore-frozen` + `verify` |
| prefix cache must be cleared after weight sync | the vLLM server resets it on `wake_up` / `resume_kv_cache` (`vllm_async_server.py` ~L841-892) | kept off anyway; per-task mismatch alert armed |
| DAPO dynamic sampling | `algorithm.filter_groups` *refills* to a full batch | custom replay buffer *shrinks* the batch; verl pads to `lcm(dp, mini×n)` with 128-token masked samples |
| partial rollout (suspend > 16K) | `colocate_async` aborts the in-flight tail when the batch is complete, resumes after sync; groups spanning more than `trainer.v1.sampler.max_off_policy_threshold` versions are evicted (`replay_buffer.py:528`: `step − submitted + 1 > threshold`) | native mechanism; threshold = `max_staleness + 1` = 2 |
| resume restores the LR scheduler | `optim.use_checkpoint_opt_param_scheduler` defaults to **false** (scheduler re-initialized) | set **true** for RL and SFT |
| auto-pause checkpoint | `_compute_metrics` runs after `on_step_end` woke vLLM | `_pause` aborts + sleeps the replicas before saving |
| `mamba_ssm_cache_dtype` FP32 | the model config says `mamba_ssm_dtype: float32`; `engine_kwargs.vllm.*` is passed verbatim to `vllm serve` | pinned explicitly with `+actor_rollout_ref.rollout.engine_kwargs.vllm.mamba_ssm_cache_dtype=float32` |

### 2.1 Critical: inverted attention mask in A100 training forwards (fixed in `dpt.verl_ext.patches`)

**Symptom.** Before any optimizer step, the actor-update forward gave log-probs on the sampled
tokens 1.0–1.8 nats/token *lower* than the old-log-prob forward on the same weights.
`actor/ppo_kl` was 0.35–0.50 with a single mini-batch, and 30–47% of tokens differed by more
than 1 nat. vLLM-vs-Megatron parity looked perfect (8e-4) because it only checks the
old-log-prob (eval-mode) pass.

**Cause.** For `Qwen3_5ForConditionalGeneration`, verl 0.9.1 takes its vision-language BSHD path.
`build_vlm_attn_mask_bshd` builds an HF-style *keep* mask (`[B, S]`, True = token).
Megatron-Bridge's `Qwen3VLModel` hands it unchanged to Megatron attention, where True means
*masked*. Transformer Engine then chooses its backend by mode. Its debug log on the workstation:

| pass | `is_training` | backend (head_dim 256, SM80) | effect of the keep mask |
|---|---|---|---|
| old log-probs / eval | False | FusedAttention (cuDNN), causal | none, so the result is correct |
| actor update / SFT | True | UnfusedDotProductAttention (no fused d=256 backward on SM80) | applied: every real token is masked in the full-attention layers |

GPUs with a fused d=256 backward (e.g. H100) take the fused path in training too, which is
presumably why this went unnoticed upstream. The production cluster is A100, so the 27B RL run
*and* SFT would have trained on a corrupted forward.

**Fix.** `dpt.verl_ext.patches` converts the mask to Megatron's convention (`[B, 1, 1, S]`, True =
masked) right where verl builds it. An import hook applies it in every process: RL through
`dpt.verl_ext.register`, SFT through `VERL_USE_EXTERNAL_MODULES` set by `dpt.launch.sft`.
Standalone Megatron-Bridge, Qwen3.5-0.8B, verl's inputs, train vs eval mode:

| mask | mean(old − new) | mean \|Δ\| | max \|Δ\| |
|---|---|---|---|
| verl's keep mask (unpatched) | **+1.80** | 2.11 | 12.7 |
| converted (patched) | 3.1e-4 | 2.5e-2 | 0.41 (BF16 kernel noise; same as with no mask) |

**Guard.** `python -m dpt.parity.train_forward_parity` (step 4 of `scripts/02_parity.sh`) checks
the production path and also reports whether the upstream bug is still present, so the patch
can be dropped once verl or Megatron-Bridge fix it. For in-loop diagnosis,
`DPT_DEBUG_PPO=<file.jsonl>` (exported before `ray start`) logs per-micro-batch ratio
statistics (`dpt.verl_ext.debug_ppo`). Bisection that led here: MTP off (E2), temperature
(logged 1.0 on both passes), weights (identical parameter fingerprints), PP2 vs PP1, then the
standalone reproduction.

### 2.2 Other findings from the smoke run

Found on the workstation smoke run (2026-10-07). None of these are visible from
the source alone:

| issue in verl 0.9.1 / its stack | symptom | what this repo does |
|---|---|---|
| `calculate_entropy=true` with `entropy_from_logits_with_chunking=true`: the chunked vocab-parallel entropy's custom backward edits views of the shared logits **in place** | backward fails: "one of the variables needed for gradient computation has been modified by an inplace operation" (logits `[1, 2048, 248320]`) | `calculate_entropy: false`; the launcher refuses the combination. The loss has no entropy bonus. The per-task entropy panels use our estimate, the mean −log π of the sampled tokens (`trainer.py`, `task/<task>/entropy`) |
| `actor/entropy` under `loss_agg_mode: token-sum` | when enabled, it is aggregated with the loss's mode, i.e. a **sum** over tokens, not a per-token entropy | not used for monitoring (see above) |
| MTP acceptance: verl reads `RequestOutput.metrics.request_spec_decode_stats`, which vLLM 0.24.0 does not have | vLLM drafts with MTP (`--speculative_config {"method": "mtp"}` is passed), but `rollout/spec_accept_rate` never appears, and the Incident 5 alert would be blind | `rollout.disable_log_stats: false`; the trainer reads vLLM's Prometheus counters on every replica's `/metrics` once per step (`dpt.monitoring.spec_scrape`) → `rollout/spec_accept_rate_counters` → `spec/mtp_acceptance_rate`. Checked against a live vLLM 0.24 server: 0.70 on Qwen3.5-0.8B. The launcher warns if stats logging is off |
| `rollout_corr/log_ppl_abs_diff` under partial rollout | a sample started by the previous policy version keeps that version's log-probs, so the metric is engine mismatch **plus** policy drift. On the smoke run (lr 1e-5) it reached 4.4e-2 at step 4 and the 5e-3 auto-pause fired, although step 1 (no stale samples) measured 8e-4 | `mismatch/*` (the Incident 1/3 alerts) is computed on fresh samples only, using verl's `min_global_steps` tag; stale samples go to `drift/*` (warn at 1e-2). The same mixing happens in the 27B run (staleness ≤ 1); its size there is not yet measured |
| `gradient_accumulation_fusion` (Megatron default **on**) needs Apex's `fused_weight_gradient_mlp_cuda` | import error at model build without Apex `--cuda_ext` | a hardware-profile setting: off on the workstation, on in the cluster profile (Apex in the image) |
| Hydra struct configs | adding a key that is not in verl's schema fails with "Key '…' is not in struct" | such keys carry a `+` prefix in `configs/rl/*.yaml` (`+data.apply_chat_template_kwargs.*`, `+…override_transformer_config.*`). The launcher runs `python -m <entry> … --cfg job` before launching, so these errors surface in seconds (`--no-hydra-check` skips it) |
| Ray `runtime_env.env_vars` must be strings | "value 1 is int" | the override renderer quotes number-, bool- and null-looking strings (`dpt.launch.overrides`) |
| `steps_per_epoch = len(dataloader)` | `ZeroDivisionError` when the curriculum sampler yields fewer rows than one batch (toy bank) | `DomainRLTrainer` sets it from the sampler length |
| Transformer Engine requires cuDNN ≥ 9.12 | the workstation's system cuDNN 9.10 (`/usr/lib/x86_64-linux-gnu`) is found first | the `dpt-train` activate hook puts the pip `nvidia/{cudnn,nccl}/lib` dirs first on `LD_LIBRARY_PATH` and sets `CUDNN_PATH` |

## 3. Items to verify on the smoke run (not provable from source alone)

Status after the workstation runs of 2026-10-07 (Qwen3.5-0.8B, env `dpt-train`, 2 × A100-40GB:
train TP1 × PP2, 2 vLLM replicas):

| # | item | status |
|---|---|---|
| 1 | Megatron-Bridge converts `Qwen3_5ForConditionalGeneration` (dense GDN) in BSHD; the MTP loss trains | **verified**: `Qwen35VLBridge` loads 383 tensors; `actor/mtp_losses/mtp_1_loss` ≈ 2.6–2.9 |
| 2 | vLLM takes `mamba_ssm_cache_dtype=float32` through `engine_kwargs`; MTP speculative decoding reports acceptance | **verified / fixed**: `--mamba_ssm_cache_dtype float32` and `--speculative_config {"method": "mtp", ...}` reach `vllm serve`. verl's acceptance metric needs a vLLM field that 0.24 lacks, so it is read from vLLM's counters instead (§2.2; 0.70 measured on a live server) |
| 3 | the replay buffer receives batches whose `rm_scores` are filled | **verified**: per-task zero-advantage statistics are computed from them every step |
| 4 | `batch.tags` are dicts (`is_padding`, `min/max_global_steps`) | **verified**; the version tags now also drive the fresh/stale mismatch split (§2.2) |
| 5 | validation metric names; `proxy/*`, `gold/*`, `dev/d_num` appear | **verified**: `proxy/*`, `gold/*` and `dev/d_num` at the step-6 validation. `gold/*` needs dev rows (`split="dev"`); the toy pipeline writes them (`publish.dev_copy_if_empty`) |
| 6 | auto-pause ends the loop after the current step | **verified**: the `logppl_diff` rule fired at step 4, saved `global_step_4` and `PAUSED.json`, and the run exited cleanly |
| 7 | step-1 `rollout_corr/log_ppl_abs_diff` < 2e-3 (vLLM vs Megatron eval-mode forward) | **verified**: 7.4e-4, 8.1e-4, 8.3e-4, 7.6e-4 in four runs |
| 8 | resume equivalence (`compare_runs`, 6 + 6 vs 12 steps) | **verified**: global step, stage and LR identical after the resume, `resume/state_roundtrip_ok` = 1 (steps 7–10 compared; the reference run had paused at 10). The criteria are deterministic state only, because asynchronous partial rollout makes rewards differ between correct runs |
| 9 | *(added)* the train-mode forward equals the eval-mode forward (PPO ratio = 1 before the first update) | **failed, then fixed**: 1.0–1.8 nats/token apart on A100 (§2.1). Guarded by `dpt.parity.train_forward_parity` |

## 4. Upgrade procedure

1. Read the new release's changes to `trainer/ppo/v1/*`, `rollout_corr_helper.py`,
   `experimental/reward_loop/*` and `trainer/config/*.yaml`.
2. Diff the method signatures we override (`_compute_advantage`, `_compute_metrics`,
   `_init_dataloader`, `_save_checkpoint`, `_load_checkpoint`, `ReplayBufferAsync.sample`).
3. Re-render the launch command (`DPT_DRY_RUN=1 bash scripts/06_rl.sh`) and check every
   key still exists in the new `_generated_ppo_megatron_trainer.yaml`.
4. Update `VERIFIED` in `dpt/launch/check_env.py`, then run `scripts/02_parity.sh` (step 4 says
   whether the attention-mask patch is still needed: `upstream_bug_present`) and
   `scripts/05_rl_smoke.sh` (all items above) before any production run.
