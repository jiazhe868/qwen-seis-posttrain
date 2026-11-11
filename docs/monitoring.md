# Monitoring and triage

> The report's top lesson: whether an RL project succeeds depends mainly on its monitoring and
> evaluation infrastructure, not on the choice of algorithm. All five long-run incidents were
> numerical or systems problems. The most useful signal, the train↔rollout log-ppl gap, started
> rising 60 steps before the crash, while reward looked abnormal only in the last 6 steps.

Code lives in `src/dpt/monitoring/` and the rules in `configs/rl/monitoring.yaml`. Metric names are
the contract in `src/dpt/monitoring/keys.py`: the trainer adapter emits them and the rules consume them.

The trainer, once per step:

```python
panels = per_task_panels(task_ids, rewards, lengths, correct=..., truncated=..., entropies=...,
                         clip_fracs=..., group_ids=uids)
panels |= gradient_share(task_ids, token_counts, group_ids=uids, loss_weights=task_norm)
panels |= hack_tracker.update(step, flags_per_sample)
alerts = alert_manager.evaluate(step, {**verl_metrics, **panels, **mismatch, **parity})
sink.log(step, {**verl_metrics, **panels, **alert_manager.step_metrics()})
if alert_manager.rollout_blocked(): ...   # do not generate; re-run the parity probe / clear caches
if alert_manager.should_pause():   ...    # save checkpoint + frozen batch, stop, page a human
audit.maybe_export(step, records)
```

`AlertManager.state_dict()` is saved with every checkpoint. Resuming without it would reset streaks
and windows, which is the kind of silent behavior change behind incident 2.

---

## 1. The panel (every row is a rule)

| Metric (key) | Normal | Alert (action) | Risk | What to do |
|---|---|---|---|---|
| train↔rollout log-ppl gap `mismatch/log_ppl_abs_diff` (fresh samples only, see below) | < 5e-4 | > 2e-3 **warn**, > 5e-3 **pause** | numerical mismatch, failed weight sync | Check the rollout recurrent-state dtype (must be FP32), the TIS settings, and whether a weight sync just happened. Slice by position (`…_pos_gt_12k`), task and rank. Roll back to the last checkpoint *before* the gap started rising (incident 1 rolled back to step 120, not 170). |
| post-sync fixed-prompt probe `parity/post_sync_pass` | pass | fail → **block_rollout** | prefix cache not invalidated, resharding error | Clear the vLLM prefix cache (including DeltaNet state snapshots) and re-sync weights. If it still fails, compare per-layer weights between Megatron and vLLM. |
| policy drift `drift/log_ppl_abs_diff` (stale partial-rollout samples) | ≈ per-step policy change | > 1e-2 **warn** | updates too large: LR too high, or updates driven only by the MTP aux loss / Adam momentum on steps without signal | Check `actor/lr`, the fraction of trained tokens, `rollout_corr/rollout_is_eff_sample_size`. This is *not* an engine problem; do not roll back for it alone. |
| per-task entropy `task/{task}/entropy` | slow decline | −30% or +50% within 20 steps (**warn**) | entropy collapse / explosion | Collapse: check clip-high, LR, and whether one task's easy problems dominate. Explosion: check the mismatch metrics first; incident 1 began as an entropy spike. |
| length by correctness `task/{task}/len_incorrect` vs `len_correct` | stable | wrong-answer length grows ≥ 10 pts faster over 20 steps (**warn**) | length bias, repetition loops | Check the per-task loss normalization, the T3 length penalty, and samples near the cap for loops. |
| truncation `task/{task}/truncation_rate` | < 3% | > 5% (**warn**) | overthinking | Check the effort rendering and the 20K→24K penalty. Truncated samples are excluded from the loss, so a rising rate also wastes rollout compute. |
| zero-advantage groups `task/{task}/zero_adv_frac` | 20%–45% | > 60% for 30 consecutive steps (**warn**) | problems too easy or too hard | The curriculum's automatic reweighting should already respond: lower the task's sampling weight and raise its hard-pool share. |
| clip fraction `actor/pg_clipfrac` | 1%–5% of tokens | > 10% (**warn**) | updates too off-policy | Fewer mini-batch updates per step, lower LR, check partial-rollout staleness. |
| grad norm `actor/grad_norm` | stable | > 5× moving mean (**warn**) | anomalous samples, numerics | Dump the step's batch; look for OCR-garbage-like repetitive samples (the SFT step-400 spike) or extreme IS ratios. |
| proxy vs gold `proxy/{task}` vs `gold/{task}` | move together | proxy up, gold flat/down 3 evals in a row (**warn**) | reward hacking | Pull 100 high-reward samples and look. Typical fixes: strip the exploited pattern before the judge, add a rule, change the test distribution. |
| hack-pattern rates `hack/*/rate` | flat | ≥ 2× window over window (**warn**) | active exploitation of a known pattern | Same as above. A rule only finds known hacks, so keep the audits going. |
| sandbox infrastructure errors `sandbox/t2_infra_error` (dev passes) | 0 | > 2% (**warn**), > 20% (**pause**) | the T2 sandbox cannot run code; it fails closed, so the code scores 0 and the rewards look merely low | Check that `apptainer` is on the PATH of every node's Ray workers (env.sh falls back to the module's install), run `python -m dpt.rewards.sandbox_check` on a worker node, and check the hidden-tests directory. Campaign 3532660 (10-29): 79% of the dev T2 samples, on the 7 nodes where Ray starts over ssh. |
| MTP acceptance `spec/mtp_acceptance_rate` | > 0.65 | < 0.6 (**warn**) | stale draft head | Make sure the MTP head trains during RL (aux loss 0.1). Incident 5: 0.71 → 0.46, rollout time 4.6 → 6.4 min. |
| per-rank peak memory `memory/peak_gb_max_rank` | < 72 GB | > 76 GB (**warn**; 36 GB on the 40 GB workstation) | OOM | Check `memory/peak_gb_argmax_rank`. Incident 4 was one PP-stage-2 rank holding a 31K trajectory padded to the 32K bucket, with 2 micro-batches in flight. Split micro-batches by token budget. |
| dev D-Num `dev/d_num` (with CI) | rising | 3 consecutive drops whose total exceeds the CI (**warn**) | overfitting | Stop selecting later checkpoints; consider the stage-C LR drop or more replay. |
| general replay reward `task/t4_general/reward_mean` | flat | > 2 points below its early baseline (**warn**) | forgetting | Raise the T4 share (stage C uses 25%); check the general-benchmark dev slice. |

The thresholds come straight from the report. On the 40 GB workstation, the memory threshold is
derived from the hardware profile (`gpu_mem_gb − 4`).

---

### 1.1 Where each metric comes from in the verl run

`DomainRLTrainer` emits or derives every metric the rules read
(`dpt/monitoring/bridge.py` maps verl names onto the canonical keys):

| rule | metric source in the verl run |
|---|---|
| logppl_diff, logppl_diff_tail | `dpt.rl.mismatch.mismatch_metrics` on old-logprob vs rollout logprob, per task / position bucket (tail = \|`mismatch/pos/12k+/log_ratio`\|), **on fresh samples only**: those generated entirely by the current weights (`dpt.rl.mismatch.fresh_sample_mask` on verl's `min_global_steps` tag). A partial-rollout sample started by the previous version carries that version's log-probs, so its gap also contains the policy's own change. On the workstation smoke run (lr 1e-5) the mixed metric reached 4.4e-2 by step 4 and tripped the 5e-3 pause, although step 1 (no stale samples yet) measured 8e-4 and TIS weights were drifting (IS mean 0.986, ESS 0.94): a drift signature, not an engine fault. verl's own `rollout_corr/log_ppl_abs_diff` mixes both kinds of sample; do not alert on it. `mismatch/fresh_sample_frac` shows how many samples the metric saw. The fresh samples of a step are its shortest responses, and per-sequence means of short responses are noisy. So `mismatch/*` is emitted only with ≥ `rl.mismatch.min_fresh_sequences` (8) fresh samples and ≥ `min_fresh_tokens` (4096) fresh tokens. Otherwise `mismatch/insufficient_fresh` = 1 and the alert skips the step. On the smoke run, 4 short fresh samples gave 8.8e-3 and a false auto-pause, while 28 on the previous step gave 2.8e-4. |
| policy_drift | the same metrics on stale samples, `drift/*` |
| grad_norm_spike, clip_frac | verl `actor/grad_norm`, `actor/pg_clipfrac` |
| memory_peak | verl `perf/max_memory_reserved_gb` (averaged over workers by verl, so the true max rank can be higher; keep the 4 GB margin) |
| mtp_acceptance | verl `rollout/spec_accept_rate` when the backend reports per-request stats; with vLLM 0.24 (which does not) the trainer's reading of vLLM's Prometheus counters, `rollout/spec_accept_rate_counters` (Δ accepted / Δ drafted over all replicas per step, `dpt.monitoring.spec_scrape`; needs `rollout.disable_log_stats: false`) |
| entropy_change | per-task mean of −log π_old over sampled tokens (unbiased entropy estimate at T=1) |
| wrong_length_growth, truncation | `dpt.rl.pipeline` per-task panels |
| zero_adv | replay-buffer filter statistics (`filter/<task>/zero_adv_group_frac`) |
| sandbox_infra | dev validation every 20 steps: `val-aux/t2_code/t2_infra_error/mean@n`, the share of T2 samples whose sandbox run ended in `infra_error` (tests unavailable, container unavailable, or a harness exception). Training steps do not report it. |
| proxy_gold, dev_d_num | dev validation every 20 steps: `score` (training grader) vs `gold_score` (independent eval grader), from `val-aux/<task>/<var>/mean@n` (verl files only the core var `acc` under `val-core/`) |
| hack_growth | per-sample reward `flags` → `HackRateTracker` |
| general_replay | `task/t4_general/reward_mean` |
| post_sync_parity | **not produced in-loop.** verl 0.9.1 resets vLLM's prefix cache on wake-up, and running a trainer-side logprob probe while vLLM is awake risks OOM. Run `dpt.parity.logprob_parity` after every (re)start instead. The per-task mismatch rules catch the Incident-3 signature (one task's mismatch jumps) in-loop. |

## 2. Triage method (report §排查方法)

1. **Classify first.** Is it a numerical/systems problem (mismatch, grad norm, memory, throughput)
   or a learning-dynamics problem (entropy, length, reward distribution)? **Look at the mismatch
   metrics before reward.** The panel's rule order follows this.
2. **Slice.** By task, token position, response length and rank. Incident 1 was clearly visible
   only on tokens past position 12K; the overall mean barely moved. That is why
   `mismatch/log_ppl_abs_diff_pos_gt_12k` exists.
3. **Freeze the scene.** Save the rollout batch from before and after the event. Offline, recompute
   logprobs on the training side, replay on the inference side under different settings, and
   bisect one variable at a time. The JSONL sink (`metrics.jsonl` next to the checkpoint) is the
   offline source of truth.
4. **Single-variable controls.** Change one setting at a time and rerun 10–20 steps from the
   rollback point.
5. **A fix is not done until it leaves a guard.** Every incident gets an alert, a unit test or a
   pre-flight check, and the runbook records it.

---

## 3. Incident → guard mapping

| # | Incident | Guard in this repo |
|---|---|---|
| 1 | BF16 recurrent state on the rollout side vs FP32 accumulation on the training side; no TIS. Entropy spike, garbage output, reward 0.58 → 0.12 within 6 steps. | `logppl_diff` (warn 2e-3 / pause 5e-3), `logppl_diff_tail` (positions > 12K), FP32 `mamba_ssm_cache_dtype` in the rollout config, TIS cap 2.0 plus a per-sequence mismatch mask (`dpt.rl`). Test: `tests/monitoring/test_alerts.py::test_incident1_logppl_warns_then_pauses`. |
| 2 | Resumed without sampler/curriculum/RNG/scheduler state: reward −0.09 and 30 steps to recover. | Every stateful component exposes `state_dict()`, including `AlertManager`, `HackRateTracker`, `ProxyGoldDivergence` and audit seeding. Test: `test_state_dict_roundtrip_gives_identical_alerts`. Resume-equivalence test lives in `dpt.rl`. |
| 3 | Prefix cache (including DeltaNet state snapshots) not invalidated after weight sync: log-ppl gap 5e-4 → 1.4e-2, only on T2. | `post_sync_parity` → `block_rollout`. The trainer clears the prefix cache after every sync and probes 16 fixed prompts. Test: `test_incident3_parity_failure_blocks_rollout`. |
| 4 | OOM on one PP-stage-2 rank (31K trajectory → 32K bucket, 2 micro-batches in flight, fragmentation after repeated offloads). | `memory_peak` rule with argmax rank; token-budget micro-batching and the 32K total cap live in `dpt.rl`; expandable segments are enabled in the launch scripts. |
| 5 | MTP head frozen during RL; acceptance 0.71 → 0.46; rollout 4.6 → 6.4 min with no error. | `mtp_acceptance` < 0.6 warn; the MTP head trains during RL (aux loss 0.1). Test: `test_incident5_mtp_acceptance`. |

Reward-hacking cases (steps 60–310) map to `proxy_gold`, `hack_growth`, `AuditSampler` and the
fail-closed graders. See docs/evaluation.md §4 for the grader side.

---

## 4. Human audits

Every 20 steps, `AuditSampler.maybe_export(step, records)` writes:

* `behavior_stepNNNNNN.jsonl`: the 50 highest-reward and 50 lowest-reward samples, to find unknown hacks.
* `spotcheck_stepNNNNNN.jsonl`: 50 reward=1 and 50 reward=0 samples, to measure grader FP/FN.

Reviewers fill in `human_label` (`correct` | `incorrect` | `hack:<category>`). Then run
`score_spotcheck(load_labeled(path))` to get the grader's FP/FN rates. Sampling is seeded by
`(seed, step)`, so a resumed run exports the same samples.
