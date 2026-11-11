# Failure atlas

[English](PITFALLS.md) | [中文](PITFALLS.zh-CN.md)

How this pipeline silently fails, how each failure was detected, and what now prevents it from recurring.

Every entry is a mistake actually made here, on the Qwen3.5-9B SFT and multi-task RL pipeline on a shared
Slurm cluster (8 nodes × 3 A100-40GB) in October and November 2026: caught by a measurement, corrected, and
given a test or a guard. Nothing below is a generic best-practice list.

**The organising observation: most of the expensive failures here were silent.** In thirteen of the
seventeen entries, a component reported success, or reported nothing, while it did the wrong thing: a parity
check that checked the other forward pass, a patch that patched nothing, a sandbox that failed closed
without an alert, framework defaults that overrode the config, a hang that no timeout noticed, a judge that
graded its own students. The loud failures (a crash, an out-of-memory error) cost minutes; the silent ones
cost hours of eight nodes, or a conclusion.

Companion document: [README.md](README.md) explains how the pipeline works and what the run showed.

---

## Index

| id | incident | category | cost if uncaught | loud or silent |
|---|---|---|---|---|
| A1 | Every template rejected: two bugs only real data could show | data pipeline | a bank with no numeric problems, or a fix by hand at scale | loud |
| A2 | A crash hidden behind a hang | data pipeline | a second failed 8-node launch | silent |
| B1 | Parity passed, training was wrong: the inverted attention mask | numerics | an RL run optimising noise; an SFT whose attention layers see nothing | silent |
| B2 | The last pipeline stage: a budget, a probe and a patch that all missed | memory | SFT failing at its first step, and again at its first checkpoint | silent |
| B3 | No attention kernel at the real sequence length | numerics | no 9B training step at all | loud |
| B4 | Host memory: the colocated servers counted too | memory | RL killed at its first checkpoint | loud |
| C1 | What a batch job inherits, an ssh shell does not | environment | 79% of code rewards scored 0 for the whole run | silent |
| C2 | Connection pools and timeouts: one default, three places | environment | throughput capped at a third; lost work at the one-hour mark | silent |
| C3 | Framework defaults that override the config | environment | Adam β2 0.999 instead of 0.95; a crash at step 50 | silent |
| C4 | A file lock on NFS: a hang costs more than a crash | environment | 9.6 hours of 8 idle nodes | silent |
| D1 | "T4 taught brevity": a story the per-step metrics did not support | analysis | a wrong fix for RL v2 | silent |
| D2 | One model as reward judge, data filter and evaluator | measurement | "reports improved by 36 points" | silent |
| D3 | A difficulty filter run with the wrong model | measurement | RL with no gradient on most groups | silent |
| D4 | Gates on a saturated test | measurement | an 8-node job ended by noise | silent |
| D5 | A refusal detector too narrow for the model it measures | measurement | a quarter of a safety regression misread | silent |
| E1 | The queue sets the pace | operations | days per failed launch | loud |
| E2 | Unwatched waits: an idle job and a cleanup that cancelled another | operations | 2 idle hours of 8 nodes; a judge job's 4-day queue place | silent |

Each incident uses the same template: **Symptom → The false story → Root cause → Fix → Permanent
regression test → Invalidated artifacts → Generalisable lesson.** "The false story" is the plausible
reading the wrong number supported; it is the part worth reading, because it is what a reader would have
believed.

---

# A · Data pipeline

## A1 — Every template rejected: two bugs only real data could show

**Symptom.** The first full answer verification (2026-10-25) rejected all 102 numeric templates.

**The false story.** "The templates are bad: the teacher and the judge disagree with our solvers
everywhere." Rewriting 102 solvers by hand would have followed.

**Root cause.** Two bugs, each invisible in unit tests with fakes. The solver registry module was never
imported in the verification process, so every problem failed with `solver_error: KeyError`. And the judge
was served without a reasoning parser, so it printed its thinking without `</think>`, and the vote parser
found no vote in any reply.

**Fix.** Import the registry where the solvers are looked up; serve the judge with its reasoning parser and
parse votes only after `</think>`. Re-verification from the cache took about a minute and kept 82 of the
102 templates.

**Permanent regression test.** `tests/data/test_answer_verify.py::test_family_solvers_are_found_without_importing_the_generators`;
`tests/data/test_backends.py::test_chat_rejoins_parsed_reasoning_and_sends_effort`.

**Invalidated artifacts.** The first verification pass; nothing downstream had used it.

**Generalisable lesson.** Run every stage on a small sample of real data, end to end, before running it
at scale. A 100% rejection rate is a pipeline bug until proven otherwise.

## A2 — A crash hidden behind a hang

**Symptom.** After the NFS hang (C4) was fixed, the relaunched SFT failed at its first batch on all 24 ranks.

**The false story.** None was needed, but the hang had cost 9.6 hours, and the job could have been released
believing the next launch would work.

**Root cause.** pandas reads the `tools` key, which only the tool-use rows have, as NaN in every other row of
the JSONL file, and the dataset passed `list(nan)` to the chat template. The tests had built the dataset from
in-memory frames, never from a file that mixes rows with and without tools.

**Fix.** A missing value is `None`. Before the relaunch, all 13,095 rows were built with the real tokenizer on
an idle node of the job: 0 errors, 44.1M tokens, longest row 20,841 tokens. That check takes 2 minutes on one
node; each failed 8-node launch had cost ~12 minutes of the whole job.

**Permanent regression test.** `tests/verl_ext/test_sft_dataset.py::test_jsonl_rows_without_tools_render`.

**Invalidated artifacts.** None.

**Generalisable lesson.** Build every training row once, from the real file with the real tokenizer, before
a multi-node launch. Test data loaders with files, not with the objects the files are meant to produce.

---

# B · Numerics and memory

## B1 — Parity passed, training was wrong: the inverted attention mask

**Symptom.** In the workstation RL smoke run (2026-10-07), the PPO KL was 0.35–0.5 with a single mini-batch,
where it should be near zero: the training forward's log-probabilities on the sampled tokens were 1–2 nats
per token below the rollout's, on identical weights.

**The false story.** "vLLM and Megatron differ numerically; the parity check passed, so it is
training-inference mismatch, and TIS will absorb it."

**Root cause.** verl 0.9.1 builds an HF-style *keep* mask (True = real token) for Qwen3.5's vision-language
path, and Megatron-Bridge passes it unchanged to Megatron attention, where True means *masked out*. Whether
the mask matters depends on the attention backend. In the forward-only pass (old log-probs, evaluation),
cuDNN fused attention uses its own causal mask and ignores the supplied one, so the result is right. In the
training forward on A100 with head_dim 256, cuDNN has no fused backward for that shape, so Transformer Engine
falls back to unfused attention, which applies the mask and blanks every real token in the full-attention
layers. The parity check compared vLLM with Megatron's *eval* forward, the path that was right.

**Fix.** An import-hook patch converts the keep mask to Megatron's convention, `[B, 1, 1, S]` with True =
masked (`src/dpt/verl_ext/patches.py`, fix 1). Measured on Qwen3.5-0.8B: mean(old − new) log-prob 1.80 nats
per token before, 3e-4 after, the same as kernel noise.

**Permanent regression test.** `tests/verl_ext/test_patches.py::test_keep_mask_becomes_megatron_mask`, and a
training-forward parity check on the GPU (`python -m dpt.parity.train_forward_parity`, step 4 of
`scripts/02_parity.sh`) that compares the training forward, not only the eval forward, with the rollout.

**Invalidated artifacts.** Every smoke run before the fix.

**Generalisable lesson.** A parity check proves only the path it runs. Check the forward pass that computes
the gradient, on the hardware that will train; a GPU whose fused kernel supports the shape (H100) hides the
bug entirely.

## B2 — The last pipeline stage: a budget, a probe and a patch that all missed

**Symptom.** Measured on one A100 before the first 9B step (2026-10-28): the last of the 8 pipeline stages
could hold sequences of only ~3.5K tokens. 28% of the SFT rows are longer.

**The false story.** "The memory plan budgeted 32.3 GB for the heaviest stage, so 40 GB GPUs are enough."

**Root cause.** Three misses in a row.
- The budget counted neither MTP's own copy of the 1.0B-parameter embedding nor the FP32 logits on the last
  stage. The 27B plan's TP4 split the 248,320-word vocabulary four ways; TP1 does not.
- The one-GPU probe of the loss path called the loss without Megatron's `Float16Module`, which upcasts the
  last stage's output to FP32. Megatron's own trainer returns a loss there; verl returns logits. The 8-node
  pre-flight then ran out of memory allocating exactly one FP32 copy of the logits, 18.8 GiB.
- One of the memory fixes patched nothing: verl defines the method on a subclass, the patch looked for it on
  the base class, and its unit test used a fake class with the same mistake.

**Fix.** Adam on the CPU with FP32 master weights there only (−11.7 GiB), Transformer Engine's cross entropy
on BF16 logits for both losses (2.38 → 0.94 MiB per token), BF16 logits out of the FP16 wrapper during
verl's LM-head forward, and one decoder layer on the first and last stages. Values match an FP32 reference
within 1e-4 nats and gradients within 1% relative L2. A patch whose target is missing now raises. The
pre-flight on the 12 longest rows then peaked at 40,366 of 40,437 MiB.

**Permanent regression test.** `tests/verl_ext/test_patches.py::test_engine_fix_refuses_a_missing_target`,
`::test_float16_output_stays_bf16_only_inside_the_lm_head_step`, `::test_fixes_reach_the_installed_stack`
(the real classes, not fakes), and the 2-step 8-node pre-flight before every new layout.

**Invalidated artifacts.** The memory plan for the 9B line; its stage-7 numbers were rewritten from
measurements.

**Generalisable lesson.** Measure the heaviest stage at the longest sequence, through the same wrappers the
trainer uses, before the multi-node launch. Test patches against the installed library, never against a
fake that repeats the patch's assumption.

## B3 — No attention kernel at the real sequence length

**Symptom.** The first SFT forward failed: "No dot product attention backend is available".

**The false story.** "The environment is broken; rebuild it." The environment was fine for every shape
the smoke tests had used.

**Root cause.** The 9B SFT config inherited the 27B recipe's `attention_backend: flash`, and the environment
had no flash-attn. With `auto`, Transformer Engine on A100 rejects cuDNN fused attention for head_dim > 128, so
only unfused attention is left, which stores the full S × S score matrix: 1.6 GiB at 4,096 tokens, 13.5 GiB at
12,288, out of memory at 16,384 for one layer. The workstation smoke runs used 4K sequences, so nothing had
ever run a 9B step at the real length.

**Fix.** flash-attn 2.8.3 built for SM80 (8 minutes on one node, sha256 checked) and `attention_backend:
auto`. One layer forward + backward then needs 0.7 GiB at 20,841 tokens. Flash against unfused: worst relative
L2 difference 5.6e-3 on outputs and gradients (BF16 noise); model log-probs at 8,192 tokens within the
5e-3 / 5e-2 thresholds.

**Permanent regression test.** The environment build now includes flash-attn and its import check; the
pre-flight runs the longest rows.

**Invalidated artifacts.** None; no 9B step had run.

**Generalisable lesson.** A smoke test at a short length cannot show length-dependent failures. Run one step
at the real maximum length, and check every default a config inherits from a recipe written for other
hardware.

## B4 — Host memory: the colocated servers counted too

**Symptom.** RL's step-1 checkpoint save ran the last-stage node out of host memory; Ray killed its three
vLLM servers. The resume failed the same way while loading the optimizer state.

**The false story.** "The checkpoint is too big for one node." The SFT pre-flight had saved the same
checkpoint on the same layout without trouble.

**Root cause.** With MTP drafting on, verl puts vLLM to sleep at level 1, which keeps a CPU copy of every
weight: a sleeping replica held ~26 GB, ~20 GB of it that copy, and there are three per node. The last node already used ~202 of its 251 GiB before the save,
and gathering the stage's optimizer state on one rank (31 → 66 GB) crossed Ray's 95% limit. The SFT
pre-flight had measured the save without the colocated vLLM servers.

**Fix.** MTP drafting off in rollout (the MTP head still trains): vLLM then sleeps at level 2, weights dropped
and restored by the weight sync that precedes every generation. Outputs are unchanged, since speculative
decoding is lossless; generation is slower. The three servers on that node now hold 9 GB instead of 78.

**Permanent regression test.** The config key with its reason (`actor_rollout_ref.model.mtp.enable_rollout:
false` in `configs/rl/rl_9b.yaml`); the host-memory series is logged per node (`rl.json`,
`host_memory_stage8`).

**Invalidated artifacts.** The step-1 update, which had also trained on broken code rewards (C1); RL restarted
from step 0.

**Generalisable lesson.** A pre-flight must run the full set of processes that will share the node, not
the trainer alone.

---

# C · Environment and silent defaults

## C1 — What a batch job inherits, an ssh shell does not

**Symptom.** Twice. On 10-27 the campaign's 24 vLLM servers all died at start. On 10-29 the first dev pass
scored the SFT model 10.4% on code, against 86% on D-Code; samples of the same prompt passed or failed
depending on the node that graded them.

**The false story.** For the code score: "RL's dev set is harder than D-Code" or "the reward is buggy".

**Root cause.** The campaign starts processes on the other nodes over ssh, and on this cluster a shell started
by `ssh node cmd` does not get what a batch job inherits from the login modules. The first time it was the
newer libstdc++ that FlashInfer's sampling kernel needs. The second time, Lmod is stubbed out in such shells
(`LMOD_CMD=:`), so `module load apptainer` did nothing on 7 of 8 nodes. The sandbox found no
`apptainer`, failed closed as designed (model-written code never runs outside the container), and every
sample graded there scored 0 as an infrastructure error. That rate was logged but not alerted on.

**Fix.** `scripts/slurm/env.sh` preloads the library on every compute node and falls back to the default
module's install path when the module load leaves no apptainer on the PATH. The isolation check
(`dpt.rewards.sandbox_check`: no $HOME, $WORK, $SCRATCH or network) passes on the worker nodes. After the fix
the step-0 code score was 79.2%, with no infrastructure errors.

**Permanent regression test.** `tests/slurm/test_job_scripts.py::test_env_finds_apptainer_on_a_node_reached_over_ssh`.

**Invalidated artifacts.** The first RL launch's step-1 update.

**Generalisable lesson.** Failing closed is right for safety, but it must be loud: alert on the
infrastructure-error rate of every reward. Test the exact path by which processes start, not a batch job
that happens to work.

## C2 — Connection pools and timeouts: one default, three places

**Symptom.** Verification ran at ~3.2 problems per minute; the eval client kept ~38 of each replica's ~114
samples running; 27 pass@64 requests failed at exactly one hour.

**The false story.** "The servers are slow" or "the model is slow at long answers".

**Root cause.** httpx's default pool holds 100 connections, and two different clients (verification and
the eval harness) kept it although each asked for hundreds of requests in flight. When the pool was raised,
judge requests queued past the 30-minute client timeout; clients re-sent them while the proxy kept running
the abandoned ones, and verification stalled for an hour. And the round-robin proxy read its backends with a
one-hour timeout, while vLLM sends nothing until all 64 samples of a request are done.

**Fix.** Pools sized to the configured concurrency (`src/dpt/data/backends.py`, `src/dpt/eval/harness.py`);
client timeouts that cover the queue wait (Little's law: wait ≈ requests in flight / throughput); a proxy with
no read timeout of its own that cancels a request when its client gives up. Eval throughput then rose from
~900 to ~2,250 tokens/s per replica at the start of a run.

**Permanent regression test.** `tests/slurm/test_rr_proxy.py::test_every_request_reaches_the_backend_and_a_client_that_gives_up_cancels_it`,
`::test_the_proxy_sets_no_read_timeout_of_its_own`, `tests/data/test_difficulty.py::test_difficulty_requests_fit_in_the_base_timeout`.

**Invalidated artifacts.** None; the work was redone (~1.5 hours of pass@64).

**Generalisable lesson.** When one client floods one server, size the pool to the concurrency, make the
timeout cover the queue, and let the proxy cancel what the client abandons.

## C3 — Framework defaults that override the config

**Symptom.** SFT crashed at step 50 on a validation pass, in a config with no validation set. A config
review found that SFT would have run with Adam β2 = 0.999.

**The false story.** "The config says `val_files: null` and `betas: [0.9, 0.95]`, so that is what runs."

**Root cause.** verl 0.9.1 validates whenever `test_freq > 0`, with or without a validation set, and it does
not pass `optim.betas` on to Megatron's optimizer, whose default β2 is 0.999. The attention backend default
(B3) is the same kind of defect.

**Fix.** `trainer.test_freq: -1` and β2 set on Megatron directly, each with a comment saying why
(`configs/sft/sft_27b.yaml`).

**Permanent regression test.** The comments and values in the config; the pre-flight now runs past the
first validation step.

**Invalidated artifacts.** About an hour of SFT (no checkpoint existed before step 100).

**Generalisable lesson.** A config states intent; the framework decides what runs. Log the values the
framework actually uses and compare them with the config before a long run.

## C4 — A file lock on NFS: a hang costs more than a crash

**Symptom.** SFT hung before its first step. Seven nodes' ranks timed out after 30 minutes; node 3's could
not exit, sat in the kernel (state D) and logged 172 MB of heartbeat errors.

**The false story.** "NCCL is stuck; wait for the timeout and relaunch."

**Root cause.** Megatron-Bridge takes a file lock under `~/.cache/huggingface` while it reads the model
config, and `$HOME` on the cluster is NFSv3. Node 3's three lock requests never returned. A crash would have
ended the campaign cleanly; the hang kept 8 nodes idle for 9.6 hours (77 node-hours) until a person was
available.

**Fix.** The lock lives in node-local `/tmp` (`MEGATRON_CONFIG_LOCK_DIR` in `scripts/slurm/env.sh`); the
models are read offline, so no shared lock is needed. The job was saved, not resubmitted: the stuck
campaign was paused with SIGSTOP, a second copy started inside the same allocation and skipped the finished
steps, and a supervisor ended the paused copy when the new one exited (E1).

**Permanent regression test.** `tests/slurm/test_job_scripts.py::test_env_keeps_megatron_config_locks_off_nfs_home`.

**Invalidated artifacts.** None.

**Generalisable lesson.** No file locks on network file systems. And a stall alarm on the step log (no new
step for 20 minutes) would have caught the hang in minutes; build it before the run, because nothing else
notices a hang.

---

# D · Measurement and analysis

## D1 — "T4 taught brevity": a story the per-step metrics did not support

**Symptom.** After step 20 of RL the answers of every task kept shortening until accuracy collapsed (README
§6.4).

**The false story.** "T4, the general task, rewards short answers, and it supplied most of the gradient, so
it taught the model to be brief." It was written into the run notes, and RL v2 was about to halve T4's
share because of it.

**Root cause.** The story was never tested. The per-step training metrics contradict it: within a step,
right and wrong answers were about equally long (medians over steps 1–20: T4 3,749 vs 3,341 tokens, T1 2,058
vs 1,796), so the reward did not favour short answers; and every task shortened from the first steps. What
is established is only that T4 supplied most of the gradient while T1 and T2 were nearly solved.

**Fix.** The claim was withdrawn (2026-11-06) and the cause is reported as open. RL v2 adds the measurements
that could decide it: dev passes every 10 steps and a warning when T1 answers shrink by 60% within 30 steps.

**Permanent regression test.** The per-step metrics `task/*/len_correct` and `task/*/len_incorrect` are
logged for every task, so the claim can be checked against any run.

**Invalidated artifacts.** The "brevity" explanation in the talk notes and the deck.

**Generalisable lesson.** A correlation between a task's gradient share and a behaviour is not a
mechanism. Before writing down a cause, find the measurement that would refute it, and run it.

## D2 — One model as reward judge, data filter and evaluator

**Symptom.** On D-Report, Gemma preferred the SFT model's reports to the base's 80 wins to 8 (86.0%).

**The false story.** "SFT improved the reports by 36 points."

**Root cause.** Gemma also chose the SFT reports (rejection sampling) and is the T3 reward judge. A second
judge that never touched training, GPT-6 Luna, scored the same pairs 49.0%, a tie; the two agree on fewer than
half the pairs. Against Luna, at a pass mark of 70, only 8 of Gemma's 108 passes hold.

**Fix.** Two judges on every pairwise evaluation, in both orders, and the gate reads the lower one. A drift
audit re-scores the reward judge's dev answers with the second judge during RL (`scripts/slurm/judge_drift.sh`);
it found no drift. RL v2 rewrites the rubrics as atomic yes/no checks with quoted evidence.

**Permanent regression test.** `tests/eval/test_second_judge.py::test_min_gate_reads_the_lower_judge_and_reports_agreement`,
`::test_min_gate_without_second_judge_fails_closed`.

**Invalidated artifacts.** The single-judge D-Report numbers in early notes.

**Generalisable lesson.** A judge that selects the training data or gives the reward cannot also be the
evaluator. Keep one judge out of training entirely, and report both.

## D3 — A difficulty filter run with the wrong model

**Symptom.** In RL steps 1–20, 87% of the numeric and code groups had all 8 samples right; only 88–272 of
1,008 samples per step had a non-zero advantage.

**The false story.** "The bank is filtered for difficulty: problems the model solves 8/8 were dropped."

**Root cause.** The filter ran with the base model, for which half of T1 was in the useful middle. SFT then
made the policy much stronger. The difficulty of a problem belongs to the model that trains on it.

**Fix.** RL v2 re-grades the bank with the SFT policy (8 samples, 16K, medium effort), drops 8/8 problems,
pools 0/8 problems, draws harder problems more often, and does not start unless at least 40% of T1 groups
are mixed at step 1 (README §8.3).

**Permanent regression test.** `tests/data/test_rl_build.py::test_dev_split_comes_from_candidates_fixed_before_the_difficulty_pass`;
the bank check prints the pass-rate bands before RL.

**Invalidated artifacts.** The v1 bank's difficulty split, for RL v2.

**Generalisable lesson.** Filter for difficulty with the policy that will train, at the budget it will
train with, and check the signal (mixed groups at step 1) before spending the run.

## D4 — Gates on a saturated test

**Symptom.** At 91.6 on D-Num with pass@16 ≈ 99.8, the planned RL-entry gate ("SFT strictly above base",
"pass@16 ≥ base") would be decided by noise, and a failed gate ended the 8-node job.

**The false story.** "The gate protects the run from a bad SFT."

**Root cause.** Thresholds written for another model (the reference plan's 27B, which scored 52.1 on its own
D-Num) do not transfer to a base that saturates the set. One item that SFT never solves in 64 samples would
have failed the second check. The ship gate "D-Num ≥ base + 8" meant 99.6, unreachable by any model.

**Fix.** Before any SFT result: D-Num may fall below base by at most the combined 95% interval (~1.4 points)
and pass@16 may trail by one point; a failed gate holds the job for a person's go or stop instead of ending
it. A harder held-out slice, D-Num-hard, was frozen to measure progress.

**Permanent regression test.** `tests/eval/test_gates_report.py::test_9b_entry_gate_allows_noise_at_the_d_num_ceiling`,
`::test_failed_entry_gate_holds_then_starts_rl_on_go`.

**Invalidated artifacts.** The inherited D-Num ship gate, which still needs a new threshold fixed before the
next run's results.

**Generalisable lesson.** Before trusting a gate, check where the base model sits on the metric and how
large the noise is; a gate at the ceiling measures noise.

## D5 — A refusal detector too narrow for the model it measures

**Symptom.** Refusal of should-refuse prompts fell from 56.0% (base) to 45.7% (RL step 20) under the gate's
detector.

**The false story.** Either "RL broke safety by 10 points" or, after looking at the answers, "it only
learned new refusal wording, so nothing is wrong".

**Root cause.** The gate's detector matches fixed phrases in the first 1,500 characters. A broader lexicon
(an upper bound) still shows a 7.7-point drop, so the regression is real, and about a quarter of the strict
drop is new wording. The loss sits on XSTest's unsafe prompts worded like safe ones, not on plainly harmful
requests.

**Fix.** Both detectors are reported, and the 27 answers neither detector reads as a refusal go to a reader.
The detector for the next run must be fixed before its results.

**Permanent regression test.** `tests/eval/test_graders_misc.py::test_refusal_detection`;
`scripts/talk/refusal_breakdown.py` writes the per-category counts.

**Invalidated artifacts.** None; the strict number stands as the gate's number.

**Generalisable lesson.** Widening a detector after a drop is visible lets the measurement absorb the
regression. Fix the detector first, then look.

---

# E · Operations

## E1 — The queue sets the pace

**Symptom.** Each 8-node job waited about three days to start; 470 jobs were pending for 611 nodes on
10-29.

**The false story.** "Plan the month in GPU-hours."

**Root cause.** On a shared machine, a crash that ends the job costs the next queue wait, not the GPU-hours.
The planned stages of the main job took 28.7 of its 71.9 hours; 43.2 went to incidents, waits and holds.

**Fix.** One job runs every stage in order (`scripts/slurm/campaign.sh`); `KEEP_NODES` holds the allocation
when a stage fails; a stuck campaign is taken over inside the same job (SIGSTOP, a supervised second copy
that skips finished steps); a failed gate holds for a person instead of exiting. RL v2 requests reserved
windows, with a heterogeneous 8 + 1-node job as the fallback (the queue allows at most 8 GPU nodes per
job).

**Permanent regression test.** `tests/slurm/test_job_scripts.py` covers the campaign chain, the hold and the
window scripts in dry runs.

**Invalidated artifacts.** None.

**Generalisable lesson.** Design the job so that a failure costs minutes of the allocation, not the
allocation.

## E2 — Unwatched waits: an idle job and a cleanup that cancelled another

**Symptom.** After SFT finished, the export to Hugging Face format failed and 8 nodes sat idle for 2 hours
(16 node-hours). Earlier, the cleanup of a failed campaign copy cancelled the RL judge job, which had waited
in the queue since 10-24 with an extended 96-hour limit.

**The false story.** "The job is held, so nothing is lost while we wait."

**Root cause.** The job monitor had lapsed while a question to the user was pending, so the failed export
went unnoticed. The cleanup cancelled the judge on any exit of the campaign copy, not only on a final one.

**Fix.** A monitor stays armed whenever a job waits on a person; the export copies the bridge's Hugging Face
weights and checks them against the base (775 tensors, 333 frozen vision tensors equal, 57 of 64 probed
trainable tensors changed); cleanup depends on why the campaign ended.

**Permanent regression test.** `tests/launch/test_checkpoint_tools.py::test_noop_export_is_rejected`;
`tests/slurm/test_job_scripts.py::test_normal_job_chains_starts_ray_everywhere_and_cancels_successor_on_unclean_exit`.

**Invalidated artifacts.** None; the judge was requeued and started from the back of the queue.

**Generalisable lesson.** A held job is still spending. Keep a watcher on every wait, and make every
cleanup that cancels other work depend on the reason for exiting.
