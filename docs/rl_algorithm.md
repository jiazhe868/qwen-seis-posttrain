# RL algorithm: what gradient the policy gets, and why

This document explains every algorithmic choice in stage 2 (RL) with its
math, the evidence behind it (ablation ids A1–A12, incidents 1–5 from the
project report), and the code that implements it. The code is in
`src/dpt/rl/` (framework-free, unit-tested on CPU). The values are in
`configs/rl/algorithm.yaml` and `configs/rl/curriculum.yaml`.

Contents

1. [One RL step, end to end](#1-one-rl-step-end-to-end)
2. [Advantage: group mean, no std division](#2-advantage-group-mean-no-std-division)
3. [Policy loss: PPO-clip with clip-higher](#3-policy-loss-ppo-clip-with-clip-higher)
4. [Loss normalization by a fixed constant](#4-loss-normalization-by-a-fixed-constant)
5. [Per-task normalization and advantage scaling](#5-multi-task-per-task-normalization-and-advantage-scaling)
6. [Training–inference mismatch: TIS and sequence masking](#6-traininginference-mismatch-tis-and-sequence-masking)
7. [No KL term, no reference model](#7-no-kl-term-no-reference-model)
8. [Length control](#8-length-control)
9. [Partial rollout](#9-partial-rollout)
10. [Curriculum](#10-curriculum)
11. [Learning-rate schedule](#11-learning-rate-schedule)
12. [Micro-batching and memory pre-flight](#12-micro-batching-and-memory-pre-flight)
13. [State and exact resume](#13-state-and-exact-resume)
14. [Which conclusions transfer from 9B to 27B](#14-which-conclusions-transfer-from-9b-to-27b)

---

## 1. One RL step, end to end

```
 curriculum.sample(step)            128 prompts, task mix of the current stage (+ hard pool after 150)
        │
        ▼
 vLLM rollout (TP2 × 8 replicas)    8 samples/prompt, T=1.0, top_p=1.0, effort=medium,
        │                           ≤16K new tokens this step (partial rollout), ≤24K total,
        │                           stores per-token rollout log-probs + policy version
        ▼
 rewards (dpt.rewards)              fail-closed graders; overlong penalty 20K→24K; truncation flag
        │
        ▼
 group-mean advantage               A_i = r_i − mean_G(r)  (truncated samples excluded from the mean)
        │
        ▼
 drop zero-advantage groups         ~35% of tokens never reach the GPUs again
        │
        ▼
 old-logprob recompute (Megatron)   π_old_train on the remaining tokens
        │
        ├─► mismatch metrics        log-ppl gap per task / position / length → alerts, auto-pause
        ├─► TIS weights             w = min(π_old_train / π_rollout, 2.0); mask mismatched sequences
        ▼
 per-task scaling                   T3 running-std gain × per-task loss multiplier w_k·C/B_k
        │
        ▼
 2 mini-batch updates               PPO-clip (0.2 / 0.28, dual-clip 3), Σ token loss / C
        │
        ▼
 weight sync TP4×PP2 → TP2          + flush prefix cache + fixed-prompt parity probe (blocks rollout on fail)
        │
        ▼
 curriculum.observe(step, stats)    zero-adv fraction, dev metric → auto-reweighting
```

The order matters in two places. Zero-advantage groups are dropped
*before* the old-logprob pass, so the pass runs on fewer tokens. And
mismatch is measured right after it, on the exact tokens that will be
trained.

## 2. Advantage: group mean, no std division

For prompt `x` with samples `y_1..y_G` (G = 8) and rewards `r_i`:

$$A_i = r_i - \frac{1}{G}\sum_{j=1}^{G} r_j \qquad\text{(we do **not** divide by } \operatorname{std}(r)\text{)}$$

**Why no std division (A2).** For binary rewards with `k` of `G` correct,
the group std is `sqrt(p(1-p))` with `p = k/G`. Dividing by it multiplies
advantages by `1/sqrt(p(1-p))`:

| k/8 | std | multiplier |
|-----|-----|-----------|
| 1/8 or 7/8 | 0.33 | 3.0× |
| 4/8 | 0.50 | 2.0× |

So the *least informative* groups (nearly solved, or nearly impossible
where one success may be a lucky guess) get the *largest* per-token push.
That is a difficulty bias and adds variance. Without the division, the
gradient magnitude of a group is proportional to how much its rewards
actually vary. 9B results: +0.9 D-Num, +2.3 hard-problem pass@8; direction
and size held on 27B.

**Invalid samples.** Responses truncated at the 24K cap are excluded from
the group mean and get `A = 0` (`valid` mask). Counting them as reward 0
would raise the advantage of every finished sample in the group. That
silently treats "did not finish" as "wrong", which §长度控制 forbids.

Code: `dpt.rl.advantage.group_mean_advantage`.

## 3. Policy loss: PPO-clip with clip-higher

Per token with ratio `ρ = π_θ(y_t)/π_old(y_t)`:

$$\ell_t = \max\big(-A\rho,\; -A\,\mathrm{clip}(\rho, 1-\varepsilon_\text{low}, 1+\varepsilon_\text{high})\big), \qquad
\ell_t \leftarrow \min(\ell_t, -cA)\ \text{if } A<0 \ \ (c=3)$$

with `ε_low = 0.2`, `ε_high = 0.28`.

**Why clip-higher (A3).** For `A > 0` the clip stops the gradient once
`ρ > 1 + ε_high`. A token at probability 0.9 can rise only to 0.9·1.2 >
1, so it is effectively unconstrained. A token at 0.01 can rise only to
0.012 per update. A symmetric clip therefore throttles exactly the
low-probability tokens that carry exploration, and entropy collapses.
Raising only the upper bound loosens this for `A > 0` and keeps the
conservative lower bound for `A < 0`. On 9B at step 200, entropy was 62%
of its initial value with 0.28 versus 35% with 0.2, and D-Num was 1.4
higher.

**Dual-clip.** For `A < 0` and a very large ratio, the clipped surrogate
is unbounded (`−Aρ` grows with `ρ`). The dual-clip caps it at `−cA`.

**Two update passes per step.** 128 prompts are split into 2 mini-batches
of 64. More passes reuse the same samples more times, making the update
more off-policy, for little gain.

Code: `dpt.rl.loss.ppo_clip_token_loss` (the reference). Production uses
verl's `vanilla` loss with the same parameters, and the adapter test
checks they agree.

## 4. Loss normalization by a fixed constant

How the token losses `ℓ_t` are reduced to one scalar decides two things:
how much each sample weighs, and whether the gradient depends on what
else is in the batch.

| mode | formula | problem |
|------|---------|---------|
| token-mean | `Σℓ / N_tokens(batch)` | the divisor depends on the batch; long answers dominate |
| seq-mean-token-mean | `mean_i (Σ_t ℓ_it / T_i)` | a token of a short answer weighs more than one of a long answer: length bias |
| **fixed-constant** | `Σℓ / C`, `C` fixed | none of the above |

**Why it matters for zero-advantage filtering (A10).** A zero-advantage
sample has `ℓ_t = 0·(…)` with zero gradient. Removing it changes nothing
in the numerator, but under token-mean it changes `N_tokens`, and so
rescales every other token's gradient. With ~35% of tokens dropped, the
effective learning rate would rise by ~1.5× in a way that depends on
batch difficulty. With a fixed `C`, the full and filtered batches give
*bitwise-identical* gradients. This is the pre-launch unit test
`tests/rl/test_loss.py::test_zero_adv_filter_keeps_gradient_under_fixed_constant`.
The same test checks that token-mean does *not* have the property. The
27B run with filtering matched the unfiltered learning curve within the
CI while saving 4.4 min/step.

## 5. Multi-task: per-task normalization and advantage scaling

### 5.1 The problem

With a token-level loss, a task's share of the gradient equals its share
of **tokens**, not of prompts. In the naive mix T1 had 50% of the prompts
and 27% of the tokens. The long T2 trajectories and T3 reports took 58%.
T1 stalled after step 140 (+6.1 vs +12.4 when trained alone), and removing
T3 brought it back to +11.0.

### 5.2 Per-task normalization

We want each task's gradient share set by a chosen weight `w_k`:

$$L = \sum_k \frac{w_k}{B_k}\sum_{t\in\text{task }k}\ell_t$$

where `B_k` is a **fixed token budget** for task `k` per optimizer update:

$$B_k = \frac{\text{(nominal prompts of task } k \text{ per step)} \times G \times L_k}{\text{updates per step}}$$

`L_k` is the configured typical response length of task `k`
(`rl.loss.task_normalization.response_tokens`). When the realized tokens
equal the budget, task `k` gets exactly share `w_k`
(`test_multipliers_restore_target_gradient_shares`). `B_k` is a design
constant. It changes only at curriculum-stage boundaries and never with
the batch, so the fixed-constant property of §4 still holds.

### 5.3 Implementing it as advantage scaling (the homogeneity identity)

The clipped token loss is **positively homogeneous** in the advantage: for
every `c > 0`,

$$\ell(cA) = \max(-cA\rho, -cA\,\mathrm{clip}(\rho)) = c\max(-A\rho, -A\,\mathrm{clip}(\rho)) = c\,\ell(A)$$

because `max` commutes with positive scaling. The dual-clip branch is
selected by `sign(A)`, which scaling preserves, and `min(cℓ, −c·cA) =
c·min(ℓ, −cA)`. TIS weights multiply the loss and are unaffected. So
with any aggregation `Σℓ/C`:

$$\frac{1}{C}\sum_k\sum_{t\in k}\ell\big(m_kA_t\big) = \sum_k\frac{m_k}{C}\sum_{t\in k}\ell(A_t),\qquad m_k = \frac{w_kC}{B_k}\ \Rightarrow\ = L$$

**Consequence:** per-task normalization needs no custom loss inside the
trainer workers. The driver multiplies each sample's advantage by
`m_k` (`TaskLossNormalizer.apply`). Tested in
`test_per_task_normalization_equals_advantage_prescaling`, values and
gradients.

`C` is whatever fixed constant the framework's aggregation divides by: 1
for a token sum, and `global_batch_size × loss_scale_factor` for verl's
`seq-mean-token-sum-norm` (both configured constants). The adapter
passes it in.

### 5.4 Per-task advantage scaling (T3)

Rubric scores are continuous and concentrated in a narrow band, so T3
advantages are ~10× smaller than binary-reward advantages. A clearly
better T3 report would barely move the policy. For tasks in
`scale_tasks` we rescale by a slow per-task statistic:

$$A'_i = A_i\cdot\frac{\sigma_\text{ref}}{\max(\hat\sigma_k, \sigma_\text{min})}, \qquad \text{gain} \le 4$$

`σ̂_k` is an EMA of the task's advantage std over samples in
non-degenerate groups. `σ_ref` is the pooled running std of the
verifiable tasks, or a fixed `scale_target_std`. This is **not** the
per-group std division rejected in A2: it is one slowly-varying number
per task, so it cannot favor easy or hard groups within a task. The
gain cap stops a noisy early estimate from blowing up the scale.

### 5.5 What changed in the numbers

9B, 200 steps; direction confirmed on 27B:

| | separate | naive mix | adjusted mix |
|---|---|---|---|
| D-Num gain | +12.4 | +6.1 | +11.8 |
| D-Report judge win-rate gain | +9 | +11 | +7 |
| length-matched expert win-rate gain | +8 | +5 | +7 |
| general Δ | — | −2.3 | −0.5 |
| mean output length | — | 8.9K | 6.1K |

The naive mix's higher *judge* score was half judge length preference.
The length-matched expert metric is the one that counts.

## 6. Training–inference mismatch: TIS and sequence masking

vLLM and Megatron give different probabilities for the same token. The
causes are kernels, accumulation order, and, decisively here, the
precision of the Gated DeltaNet recurrent state. Run 1 used a BF16 state
on the rollout side and FP32 accumulation in training. The error
compounded with sequence length, and at step 180 entropy exploded within 6
steps (reward 0.58 → 0.12).

**TIS.** Tokens were sampled from `π_rollout`, but PPO's ratio uses
`π_old_train`. Importance-weight each token by

$$w_t = \min\!\Big(\frac{\pi_\text{old,train}(y_t)}{\pi_\text{rollout}(y_t)},\ 2.0\Big)$$

Truncating at 2 bounds the variance. Without the cap, a few tokens with
ratios of 10–100 dominate the gradient, which is the run-1 failure mode.
Only the upper side is truncated: weights below 1 only shrink a token's
contribution. The same weight also corrects the staleness of
partial-rollout tokens (§9). With TIS on, the 250-step 27B run was stable
(A4).

**Sequence masking.** If a sequence's `|mean(rollout) − mean(train)|`
exceeds `1e-2`, five times the batch alert level, its logprobs are
untrustworthy as a whole. Typical causes are a corrupted state or a stale
cache. The sequence is removed from the loss instead of reweighted.

**Monitoring is what saves runs.** The log-ppl gap began rising 60 steps
before the collapse, while reward looked normal until 6 steps before.
Thresholds: normal < 5e-4, alert > 2e-3, auto-pause > 5e-3. Metrics are
sliced by task, by position bucket (Incident 1 was visible only past
12K) and by length bucket (`dpt.rl.mismatch.mismatch_metrics`). After
every weight sync, 16 fixed prompts are scored on both sides
(`fixed_prompt_parity`). A failure blocks the rollout. That guard
catches an un-flushed prefix cache (Incident 3) and resharding bugs.

## 7. No KL term, no reference model

A1 (27B, 100 steps): KL coefficient 0 gave +4.1 versus +3.0 for 0.001,
with similar entropy decline. Dropping the reference model also saves a
forward pass per step and one copy of the weights in memory. Stability
comes instead from clip-higher, TIS and sequence masking. Drift is
watched through the general-replay task (T4), the general-benchmark
evals and the dashboards.

## 8. Length control

* Hard cap 24K response tokens. Prompt + response ≤ 32K (Incident 4).
* Overlong penalty: linear from 20K to 24K, reaching −0.5 at the cap.
  This discourages overthinking without penalizing long answers that are
  genuinely needed.
* Truncated responses are excluded from the loss and from the group
  baseline (§2). "Did not finish" is not "wrong", and training on cut-off
  traces teaches the model to stop mid-thought.
* Rollouts use effort = medium (A5). xhigh gave +1.2 at 2.6× tokens per
  step; the project's goal is xhigh-level quality at medium cost.

## 9. Partial rollout

Decode time is set by the slowest sequence. 90% of a step's tokens arrive
in the first 4 minutes, and the remaining 5 minutes wait for the longest
3%. Each step therefore generates at most 16K *new* tokens per sample.
Unfinished samples are suspended with their tokens, rollout log-probs and
the policy version of each token. They continue next step, up to 24K.

* **Groups, not samples.** The baseline needs all 8 samples, so a group
  with any unfinished member waits as a whole
  (`PartialRolloutBuffer`).
* **Off-policy correction.** A resumed sample's first tokens came from
  version `v`, the rest from `v+1`. TIS uses the stored rollout log-probs
  of each token, so the ratio `π_old_train/π_rollout` corrects both the
  version gap and numerical mismatch.
* **Staleness bound.** With 16K chunks and a 24K cap, a sample needs at
  most 2 steps, so staleness ≤ 1. Groups older than `max_staleness` are
  evicted, not trained.

Generation time fell from 9.0 to 4.6 min together with MTP speculative
decoding (acceptance ≈ 0.7).

**How this maps onto verl 0.9.1.** verl's `colocate_async` trainer mode has
partial rollout built in. The trigger differs slightly from the report's
"suspend after 16K new tokens". verl stops collecting once enough *finished*
groups for the batch exist. It then aborts the in-flight requests, which by
construction are the long tail, and resumes them after the weight sync with
the tokens generated so far (`FullyAsyncLLMServerClient`). Staleness is
bounded by `trainer.v1.sampler.max_off_policy_threshold`. verl counts the
policy versions a group *spans* (`step_now − step_submitted + 1 > threshold`
evicts), so `max_staleness = 1` maps to a threshold of **2**. A threshold of 1
would evict every resumed partial rollout, silently biasing training against
long answers. The token-level rollout log-probs travel with the
trajectory, so TIS covers mixed-version samples as described above. The
framework-free `PartialRolloutBuffer` stays as the reference implementation
of the semantics, with tests. It is used directly by backends without a
native mechanism. See docs/systems.md §4.

## 10. Curriculum

| stage | steps | T1/T2/T3/T4 | reason |
|---|---|---|---|
| A | 0–120 | 50/25/0/25 | verifiable rewards first; confirm stability |
| B | 120–300 | 40/25/15/20 | T3 judge enters late (no judge-pleasing first); hard pool at 150 |
| C | 300–420 | 35/25/15/25 | more replay, lower LR to win back forgetting |

* Per-step counts use largest-remainder rounding, so they sum to exactly
  128 and are deterministic.
* **Hard pool.** Problems the base model solved 0/8 at medium effort give
  zero signal early, because all-wrong groups have zero advantage. They
  join at step 150 with a 15% share per task.
* **Auto-reweighting.** If a task's dev metric has not improved for 30
  steps *and* its mean zero-advantage fraction is above 60%, its sampling
  weight is multiplied by 0.8 (floor 0.4) and its hard share raised by
  0.10 (cap 0.5), with a 30-step cooldown. Refinement: the hard share is
  raised only if the degenerate groups are mostly *all-solved*
  (saturation). If they are mostly all-failed, more hard problems would
  only add zero-signal groups.
* **Determinism.** Pool order in epoch `e` is a permutation seeded by
  `derive_seed(seed, task, pool, e)`, so a sampler's state is `(epoch,
  cursor)`. `state_dict()` also stores data and config fingerprints.
  Resuming against different data or a different curriculum fails
  instead of silently continuing.

## 11. Learning-rate schedule

Warmup over 10 steps to `1e-6`, constant, then `5e-7` for the last 120
steps. A8: the 2e-6 tuned on 9B made 27B entropy oscillate with 3× the
amplitude, and D-Num stalled after step 60. LR did not transfer across
scale.

In verl the schedule is Megatron's `WSD` scheduler (`configs/rl/rl_27b.yaml`).
Warm-up and the stable phase are as above. The last 120 steps ramp
*linearly* from 1e-6 to 5e-7 instead of dropping in one step.
`RLLRSchedule` is the framework-free reference (step drop) used in tests.

## 12. Micro-batching and memory pre-flight

* Gated DeltaNet in Megatron does not support packed sequences, so the
  layout is padded BSHD. Sorting by length before batching cuts padding
  (RL: 14% → 6%).
* Micro-batches are cut by a **token budget that counts padding** (32K),
  not by sequence count. Incident 4 was a 31K trajectory padded to the
  32K bucket inside a count-based micro-batch, with 2 micro-batches in
  flight on the PP stage and a fragmented allocator.
* `ActivationMemoryModel` estimates full-recompute activation memory:
  stored layer inputs for the in-flight micro-batches plus the working set
  of one rematerialized layer, including FP32 DeltaNet chunk states. It is
  calibrated to the report's reference point (one 32K micro-batch, 2 in
  flight ≈ 14 GB; total ≈ 61.6 GB). `preflight` rejects any plan whose
  estimated peak exceeds 76 GB (dashboard alert at 72 GB). Re-fit the
  calibration from the per-rank peak logs whenever the stack changes.

## 13. State and exact resume

Incident 2: resuming at step 200 without sampler, curriculum, hard-pool,
RNG and scheduler state restarted stage A's mix and re-ran warmup. Reward
dropped 0.09 and took 30 steps to recover. Now:

* `ExtraStateManager` registers every stateful component (curriculum,
  partial-rollout buffer, advantage scaler, alert manager, LR schedule)
  and writes them with the RNG snapshot and data hashes to
  `dpt_extra_state.json` in the checkpoint directory. Loading fails if a
  registered component is missing (or the data hash changed) unless
  explicitly allowed.
* `resume_equivalence` is the guard. Running N steps continuously must
  match running k steps, then save, rebuild, load and finish, within
  tolerance. The toy version is a unit test with a deterministic generator,
  and it also shows that dropping the curriculum from the checkpoint breaks
  equivalence. The real version (12 steps, smoke model, `scripts/05_rl_smoke.sh`)
  is a pre-launch check. There the rollouts are *not* reproducible across runs
  (asynchronous vLLM scheduling; partial rollout aborts the tail whenever the
  batch completes), so `dpt.monitoring.compare_runs` requires only the
  step-determined quantities to match exactly (global step, curriculum stage, LR),
  plus `resume/state_roundtrip_ok`: every component re-serializes to exactly the
  state it loaded. Reward and loss curves are reported, not required.

## 14. Which conclusions transfer from 9B to 27B

| conclusion | transfers? |
|---|---|
| no std division (A2), clip-higher 0.28 (A3), 8 samples/group (A6) | direction and magnitude |
| binary T1 reward beats partial credit (A7) | direction; gap shrinks 1.8 → 0.9 |
| learning rate (A8) | **no**: re-tune on 27B |

Small-model results are directional. Every *numeric* hyperparameter is
re-checked on 27B short runs (80–250 steps) before the main run.
