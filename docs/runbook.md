# Run book: launching, watching, pausing, rolling back

> In the report, all five incidents in the long RL run were systems or
> numerical failures, not algorithm failures. They cost ~1,300 GPU·h. The
> earliest warning was always the **train/rollout log-ppl difference**. It rose
> 60 steps before the crash, while reward only moved in the last 6. Look at
> mismatch before reward.

## 1. Before launch (every time)

`dpt.launch.rl` runs these automatically and refuses to start on failure:

- [ ] data manifests verify (no file changed since the pipeline wrote it)
- [ ] prompt + response ≤ 32K; memory pre-flight under 76 GB
- [ ] rollout GDN state FP32; TIS on; top_p = 1; loss `token-sum`; MTP head trained if MTP drafting is on
- [ ] the eval sets are not readable by the training user

Do these yourself:

- [ ] `scripts/02_parity.sh` passed on the current software versions, including step 4 (train-mode
      vs eval-mode forward: otherwise PPO ratios are noise)
- [ ] `scripts/05_rl_smoke.sh` passed on the current software versions
- [ ] judge endpoint pinned and calibrated (`python -m dpt.rewards.validate --calibrate`)
- [ ] the grader adversarial suite is at 100% (`python -m dpt.rewards.validate`)
- [ ] sandbox cluster warm; `DPT_JUDGE_URL` / `DPT_SANDBOX_URL` exported
- [ ] seed set (`project.seed=1` for the main line, `2` for the reproducibility run)

## 2. While it runs

The dashboard (`docs/monitoring.md` §1) is the alert table. Each row's rule
lives in `configs/rl/monitoring.yaml`. All rows are evaluated every step on
canonical metric names (`dpt.monitoring.keys`). The step log is also written
to `<run_dir>/metrics.jsonl`.

Check daily:

- per-task panels, not the total reward. Multi-task failures are invisible in the sum
- `mismatch/log_ppl_abs_diff` and its ≥12K slice
- proxy vs gold on dev (`proxy/<task>`, `gold/<task>`)
- hack-pattern rates (`hack/<category>/rate`)
- the audit queue (100 samples every 20 steps) and the grader spot-check (50 reward=1 + 50 reward=0)
- on the cluster, judge drift (`bash scripts/slurm/judge_drift.sh`, login node): GPT-6 Luna re-scores the T3
  dev responses the reward judge scored. Exit status 3 = the reward judge pulls away from the
  independent one; handle it like a `proxy_gold` alert (§3 triage), but the run is not paused for you

## 3. Auto-pause

At a red alert (e.g. log-ppl diff > 5e-3), `DomainRLTrainer` saves a
checkpoint, writes `<run_dir>/PAUSED.json` with the reason, and exits cleanly.

**Triage (report §排查方法), in this order:**

1. **Layer.** Is this a numerical/systems problem (mismatch, grad norm,
   memory, throughput) or a learning-dynamics problem (entropy, length,
   reward distribution)? Look at mismatch first.
2. **Slice.** Split by task, token position, response length and rank.
   Incident 1 was only visible past position 12K; Incident 3 only on T2.
3. **Freeze the scene.** Keep the rollout batch from before and after the
   event. Recompute logprobs offline with the training side, and replay the
   inference side with different settings (`dpt.parity.logprob_parity` with
   `--ssm-dtype`, prefix caching on/off, etc.). Bisect one variable at a time.
   For the *training* side, export `DPT_DEBUG_PPO=<file.jsonl>` before `ray start` and run one
   step with `rl.num_updates_per_step=1`. Every micro-batch then logs train-time vs old log-probs
   (`dpt.verl_ext.debug_ppo`). Before the first update their gap must be at kernel-noise level
   (mean |Δ| ≈ 1e-2). This is how the inverted attention mask (`docs/verl_integration.md` §2.1) was found.
4. **Single-variable reruns.** From the rollback point, change one setting
   and run 10–20 steps.
5. **Leave a guard.** Every fix adds an alert, a unit test, or a pre-flight
   check. A fix without a guard is not finished.

**Choosing the rollback point.** Roll back to the last checkpoint *before the
mismatch metric started to rise*, not the last checkpoint before the crash.
Later checkpoints may already contain contaminated updates. In Incident 1
the run went back to step 120, not to the most recent step 170.

**Rolling back in verl.** Point the run at an older checkpoint and resume:

```bash
bash scripts/06_rl.sh verl.trainer.resume_mode=resume_path \
    verl.trainer.resume_from_path=<run_dir>/global_step_120
```

`dpt_extra_state.json` in that folder restores the curriculum position, the
hard-pool state, the advantage scaler, the alert and hack-rate history and the
RNG. The LR scheduler is restored from the Megatron checkpoint
(`use_checkpoint_opt_param_scheduler: true`). The resumed run therefore sees
exactly what it would have seen (Incident 2).

## 4. Incident playbooks (newest first in the report)

| # | symptom | first look | root cause (report) | fix | guard in this repo |
|---|---|---|---|---|---|
| 5 | rollout time creeps 4.6 → 6.4 min, no errors | `spec/mtp_acceptance_rate` | MTP head frozen in RL; policy drifted; acceptance 0.71 → 0.46 | train MTP in RL (aux weight 0.1) | pre-flight: MTP drafting requires `mtp.enable_train`; alert < 0.6 |
| 4 | one PP-stage rank OOMs | `memory/peak_gb_max_rank` per rank | 31K T2 trajectory padded to 32K, 2 PP micro-batches in flight, fragmentation after offloads | token-budget micro-batches, 32K total cap, expandable segments | pre-flight memory model; cap check; `PYTORCH_CUDA_ALLOC_CONF`; alert > 76 GB |
| 3 | log-ppl diff jumps 5e-4 → 1.4e-2, only on T2 | `mismatch/task/t2_code/*` | prefix cache (with GDN state snapshots) not invalidated after weight sync | clear prefix cache after every sync | prefix caching off; verl resets on `wake_up`; per-task mismatch alert |
| 2 | after resume, reward −0.09, 30 steps to recover | `curriculum/stage`, LR | sampler / curriculum / RNG / scheduler not checkpointed | save everything | `dpt_extra_state.json`; `compare_runs` resume-equivalence test in the smoke script |
| 1 | entropy spikes, multilingual gibberish, reward 0.58 → 0.12 in 6 steps | `mismatch/pos/12k+/*` | rollout GDN state BF16 vs training FP32 accumulation; no TIS | FP32 rollout state; TIS cap 2.0; mask high-mismatch sequences | pre-flight FP32 + TIS; alert 2e-3 / auto-pause 5e-3; `tests/parity` |

## 5. Reward-hack playbook

1. **Signal.** Proxy rises while gold is flat or falling for 3 consecutive dev
   evaluations, or a hack-pattern rate doubles window over window.
2. **Read samples.** Open the audit queue for the task and read 20 high-reward
   samples. Most hacks are obvious to a human within minutes.
3. **Fix the grader, not the policy.** Strip, verify or penalize the
   exploited pattern (see `docs/rewards_and_hacking.md` §3 for the seven
   known cases). Add the pattern to `hack_patterns.py` and a case to the
   adversarial suite.
4. **Decide the rollback.** If the hack had already moved gold, roll back to
   before the divergence started.
5. **Before the next new task:** run adversarial probing first. Ask a strong
   model to score high *without* solving the task. At least 3 of the report's 7
   hacks could have been found this way before training.

## 6. Cost accounting

Log the GPU·h of every restart in the run notes (time from the last good
checkpoint to the pause, × 16). The report's main line used 3,450 GPU·h, of
which ~1,300 were incident losses. Most of that was preventable with the
mismatch alert that is now armed.
