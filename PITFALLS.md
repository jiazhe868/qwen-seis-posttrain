# Failure Atlas

[English](PITFALLS.md) | [中文](PITFALLS.zh-CN.md)

How this pipeline silently breaks, how each failure was caught, and what now prevents it from happening again.

Every entry in this atlas is a real failure encountered while building and running the Qwen3.5-9B supervised fine-tuning (SFT) and multi-task reinforcement learning (RL) pipeline on a shared Slurm cluster (8 nodes × 3 NVIDIA A100-40GB GPUs) in October and November 2026. Nothing here is generic advice: each incident was uncovered by a concrete measurement, traced to its root cause, fixed, and locked down with either an automated regression test or an operational guard.

**The unifying theme: most of the costliest failures were silent.** Across 23 of the 30 incidents, a component either reported success or stayed completely quiet while doing the wrong thing:
- a parity check that verified the evaluation forward pass while the training forward pass was broken;
- a monkey-patch that missed its target method and returned without warning;
- a code sandbox that safely failed closed on 7 out of 8 nodes, quietly turning valid code into zero scores with no alert;
- framework defaults that silently ignored values written in our YAML config;
- an NFS lock hang that kept 8 nodes idle for hours because no stall monitor existed;
- a judge evaluating answers from a model trained on its own filtered outputs; and
- verifiers that agreed unanimously with a wrong answer because they all used the buggy formula printed in the prompt.

Loud failures—crashes and out-of-memory (OOM) errors—usually cost a few minutes to diagnose. Silent failures cost hours of an 8-node allocation or threatened to invalidate an entire scientific conclusion.

For the full story of how the pipeline works and what the experiments showed, see the companion document: [README.md](README.md).

---

## Index

| ID | Incident | Category | Cost if uncaught | Loud or silent |
|---|---|---|---|---|
| [A1](#a1--every-template-rejected-two-bugs-only-real-data-could-show) | Every template rejected: two bugs only real data could show | Data pipeline | A problem bank with zero numeric tasks, or days wasted rewriting solvers by hand | Loud |
| [A2](#a2--a-crash-hidden-behind-a-hang) | A crash hidden behind a hang | Data pipeline | A second wasted 8-node launch | Silent |
| [A3](#a3--a-label-that-agrees-with-its-own-question) | A label that agrees with its own question | Data pipeline | 825 training problems teaching a wrong magnitude formula | Silent |
| [A4](#a4--a-source-check-that-cannot-see-what-the-source-lacks) | A source check that cannot see what the source lacks | Data pipeline | An SFT run trained on data containing factual errors in 98 of 100 rows | Silent |
| [A5](#a5--a-semicolon-that-turned-an-error-into-a-true-statement) | A semicolon that turned an error into a true statement | Data pipeline | A rubric that penalises the very claim it asks the model to make | Silent |
| [A6](#a6--a-verifier-that-cannot-solve-the-problem-it-checks) | A verifier that cannot solve the problem it checks | Data pipeline | Throwing away 10 of 12 long-horizon templates—the exact problems RL v2 was built for | Silent |
| [B1](#b1--parity-passed-training-was-wrong-the-inverted-attention-mask) | Parity passed, training was wrong: the inverted attention mask | Numerics | An RL run optimising pure noise; an SFT run whose full-attention layers see nothing | Silent |
| [B2](#b2--the-last-pipeline-stage-a-budget-a-probe-and-a-patch-that-all-missed) | The last pipeline stage: a budget, a probe, and a patch that all missed | Memory | SFT crashing on its first step, and crashing again at its first checkpoint | Silent |
| [B3](#b3--no-attention-kernel-at-the-real-sequence-length) | No attention kernel at the real sequence length | Numerics | Unable to run even a single 9B training step | Loud |
| [B4](#b4--host-memory-the-colocated-servers-counted-too) | Host memory: the colocated servers counted too | Memory | RL killed by Ray on its very first checkpoint save | Loud |
| [C1](#c1--what-a-batch-job-inherits-an-ssh-shell-does-not) | What a batch job inherits, an ssh shell does not | Environment | 79% of code reward samples scored 0 across the whole run | Silent |
| [C2](#c2--connection-pools-and-timeouts-one-default-three-places) | Connection pools and timeouts: one default, three places | Environment | Throughput capped at a third; lost evaluation work at the 1-hour mark | Silent |
| [C3](#c3--framework-defaults-that-override-the-config) | Framework defaults that override the config | Environment | Adam β2 silently set to 0.999 instead of 0.95; SFT crash at step 50 | Silent |
| [C4](#c4--a-file-lock-on-nfs-a-hang-costs-more-than-a-crash) | A file lock on NFS: a hang costs more than a crash | Environment | 9.6 hours of 8 completely idle nodes | Silent |
| [C5](#c5--a-gpu-embedder-beside-the-servers) | A GPU embedder beside the servers | Environment | A failed data job and loss of its 4.5-day queue position | Loud |
| [C6](#c6--a-login-nodes-thread-limit-that-followed-the-job) | A login node's thread limit that followed the job | Environment | 8 hours burned out of a 47-hour RL job; without a live fix, no RL would run | Silent |
| [C7](#c7--a-folder-24-ranks-create-at-once-on-a-shared-file-system) | A folder 24 ranks create at once on a shared file system | Environment | Loss of the final 10 steps of an RL run | Loud |
| [C8](#c8--a-sandbox-that-failed-on-one-node-with-a-warning-at-every-dev-pass) | A sandbox that failed on one node, with a warning at every dev pass | Environment | Code reward corrupted for an entire RL seed; evaluation code scores reported as 0.0 | Silent |
| [D1](#d1--t4-taught-brevity-a-story-the-per-step-metrics-did-not-support) | "T4 taught brevity": a story the per-step metrics did not support | Analysis | Building RL v2 around a wrong diagnosis and wrong data-mix fix | Silent |
| [D2](#d2--one-model-as-reward-judge-data-filter-and-evaluator) | One model as reward judge, data filter, and evaluator | Measurement | Mistakenly claiming "reports improved by 36 points" | Silent |
| [D3](#d3--a-difficulty-filter-run-with-the-wrong-model) | A difficulty filter run with the wrong model | Measurement | RL receiving zero gradient on almost all prompt groups | Silent |
| [D4](#d4--gates-on-a-saturated-test) | Gates on a saturated test | Measurement | An 8-node allocation aborted by sampling noise | Silent |
| [D5](#d5--a-refusal-detector-too-narrow-for-the-model-it-measures) | A refusal detector too narrow for the model it measures | Measurement | Misreading a quarter of a real safety regression as wording drift (or vice versa) | Silent |
| [D6](#d6--a-reviewer-that-favours-the-first-answer) | A reviewer that favours the first answer | Measurement | Pairwise verdicts driven by presentation order instead of report quality | Silent |
| [D7](#d7--a-number-scan-that-froze-a-pilot-for-eight-hours) | A number scan that froze a pilot for eight hours | Measurement | A frozen RL reward loop; 8 hours of two idle GPUs | Silent |
| [D8](#d8--parse-failures-that-were-not-random) | Parse failures that were not random | Measurement | Choosing a rubric format based on artificially inflated accuracy numbers | Silent |
| [E1](#e1--the-queue-sets-the-pace) | The queue sets the pace | Operations | Days of queue wait lost per failed launch | Loud |
| [E2](#e2--unwatched-waits-an-idle-job-and-a-cleanup-that-cancelled-another) | Unwatched waits: an idle job and a cleanup that cancelled another | Operations | 2 idle hours of 8 nodes; losing a judge job's 4-day queue position | Silent |
| [E3](#e3--a-synced-file-that-needed-code-the-cluster-did-not-have) | A synced file that needed code the cluster did not have | Operations | A bank build crashing on its very first Python import | Loud |
| [E4](#e4--a-server-that-never-started-and-an-evaluation-that-waited-for-it) | A server that never started, and an evaluation that waited for it | Operations | Every seed's in-job final evaluation lost, forcing a separate queue wait each time | Silent |

Every incident below follows the same structure:

> **Symptom → The false story → Root cause → Fix → Permanent regression test → Invalidated artifacts → Generalisable lesson**

Pay special attention to "The false story": that is the plausible, intuitive interpretation that the broken logs or numbers appeared to support at the time. Seeing why a careful reader would have believed the wrong story is often the fastest way to avoid repeating the mistake.

---

# A · Data pipeline

## A1 — Every template rejected: two bugs only real data could show

**Symptom.** When we ran the first full answer verification pass across our generated math and seismology problems (2026-10-25), the pipeline rejected every single one of the 102 numeric templates—a 0% pass rate.

**The false story.** "Our problem templates are broken: the teacher model and the judge model disagree with our programmatic solvers everywhere." Had we believed this, the next step would have been rewriting all 102 solvers by hand—days of pointless work.

**Root cause.** Two separate bugs worked together to produce the 100% failure rate, and neither was visible in unit tests that relied on fake objects:
1. Missing import for the solver registry. The verification process looked up each problem's programmatic solver in a central registry dictionary, but never imported the module that populates that registry. As a result, every problem immediately failed with `solver_error: KeyError`.
2. Judge served without a reasoning parser. The judge model (`Gemma 4 26B-A4B`) was deployed as an inference server without enabling its reasoning parser. Because of that, it printed its internal chain-of-thought without a closing `</think>` tag. Our vote parser looked for the verdict only after `</think>`, saw nothing, and recorded an empty vote on every single reply.

**Fix.** We imported the solver registry module directly at the lookup site, served the judge with its reasoning parser enabled, and updated the vote parser to parse verdicts strictly after `</think>`. Re-running verification from the cached responses took roughly one minute and cleanly kept 82 of the 102 templates.

**Permanent regression test.**
- `tests/data/test_answer_verify.py::test_family_solvers_are_found_without_importing_the_generators`
- `tests/data/test_backends.py::test_chat_rejoins_parsed_reasoning_and_sends_effort`

**Invalidated artifacts.** Only the first verification pass itself; no downstream dataset or training run had consumed it.

**Generalisable lesson.** Always run every pipeline stage end-to-end on a small sample of *real* data—through the real servers and real imports—before scaling up. A 100% rejection rate is almost always a pipeline wiring bug until proven otherwise.

---

## A2 — A crash hidden behind a hang

**Symptom.** Right after we resolved the NFS file-lock hang (see [C4](#c4--a-file-lock-on-nfs-a-hang-costs-more-than-a-crash)), we relaunched SFT—and the training job crashed immediately on its very first batch across all 24 ranks.

**The false story.** There was no elaborate misconception here, which is what made it dangerous: the NFS hang had already burned 9.6 hours of cluster time, and it would have been easy to assume that fixing the hang meant SFT was ready to run and release the interactive hold on the job.

**Root cause.** Our SFT JSONL file mixed regular conversation rows with tool-use rows; only the tool-use rows contained a `tools` field. When `pandas` read that mixed JSONL file into a DataFrame, it filled the missing `tools` entries on non-tool rows with `NaN` (`float('nan')`). The dataset loader then tried to call `list(nan)` before passing it to the chat template, raising a `TypeError`. Existing unit tests had constructed datasets directly from in-memory DataFrames with uniform columns, never from an actual mixed JSONL file on disk.

**Fix.** We normalised missing `tools` values to `None` rather than `NaN`. Before relaunching the 8-node job, we wrote a pre-flight check that builds every single one of the 13,095 SFT rows from disk using the real tokenizer on a single idle node: 0 errors, 44.1M tokens total, longest row 20,841 tokens. That single-node check takes 2 minutes, whereas each failed 8-node launch costs ~12 minutes across all 8 nodes.

**Permanent regression test.** `tests/verl_ext/test_sft_dataset.py::test_jsonl_rows_without_tools_render`.

**Invalidated artifacts.** None.

**Generalisable lesson.** Before launching multi-node training, iterate through every training row from the real data file on disk using the real tokenizer. Always test data loaders against actual serialized files, not against clean in-memory Python objects that the files are merely *supposed* to produce.

---

## A3 — A label that agrees with its own question

**Symptom.** During a review of 156 problems that verification had previously set aside, we noticed 5 local-magnitude (`ML`) problems whose gold labels were suspiciously small. For the standard Richter anchor—a Wood–Anderson seismograph amplitude of 1 mm at 100 km hypocentral distance—the printed formula gave `ML = 0.32`, whereas by definition `ML` at that anchor must equal `3.0`.

**The false story.** "Every problem in the bank passed three-way verification: the generator's programmatic solver, the Qwen3.8-27B teacher, and the Gemma judge all agreed on its label."

**Root cause.** The problem generator printed the Hutton–Boore relation as:
`ML = log10 A + 1.11 log10 r + 0.00189 r − 2.09` while explicitly stating that amplitude `A` is in millimetres (mm). The constant `−2.09` is only valid when `A` is measured in nanometres (nm); when `A` is in millimetres, the true Hutton–Boore constant is `+0.591` (a difference of `2.681` magnitude units). Because every problem text stated the flawed formula explicitly, the generator's solver, the teacher, and the judge all faithfully plugged numbers into the wrong equation and arrived at the exact same wrong answer. Agreement was 100%. In total, 836 flawed problems passed verification, 825 of them entered RL run 1's training bank, and one T2 coding task carried the same bad constant.

**Fix.** Both the numeric generator and the coding task now use `+0.591`, and RL bank v2 filters out all legacy problems from those families via `bank_merge.FAULTY_FAMILIES`. Notably, our first verification count on the real bank caught only 824 of the 836 bad items: 12 coding tasks stored their family name only in the `kind` field rather than `family`, so the exclusion filter was updated to inspect `kind` as well.

**Permanent regression test.**
- `tests/data/gen/test_t1.py::test_local_magnitude_uses_the_hutton_boore_constant_for_amplitudes_in_mm`
- `tests/data/gen/test_t2.py::test_network_local_magnitude_gives_ml_3_for_1_mm_at_100_km`
- `tests/data/test_bank_merge.py::test_merge_leaves_out_the_faulty_local_magnitude_families`

**Invalidated artifacts.** The 825 local-magnitude problems in RL run 1's training bank, and bank v1 as an uncleaned source for RL v2.

**Generalisable lesson.** Consensus among independent solvers and models catches arithmetic mistakes, not a wrong physical equation embedded in the prompt itself. Validate every generator against physical anchor points—known inputs whose real-world outputs are established by definition.

---

## A4 — A source check that cannot see what the source lacks

**Symptom.** When a frontier model reviewer audited 100 randomly sampled rows out of 3,262 paper-derived report-training rows (`SFT v2`), its verdict was damning: keep only 2, fix 48, and drop 50. Fully 98 of the 100 rows contained at least one factual error.

**The false story.** "Every generated row passed automated checking against its source paper section: zero contradictions, at most two claims unsupported by the excerpt, and a knowledge score of at least 3 out of 5."

**Root cause.** Two prompt design flaws combined with a blind automated check:
1. The pipeline asked the teacher model to write questions requiring *background seismology knowledge that a specialist would know without reading the paper*. To answer those questions, the teacher pulled domain facts from its own pretraining memory—especially details about named historical earthquakes and tectonic faults—and hallucinated frequently. For instance, four rows described Turkey's East Anatolian Fault as right-lateral (it is left-lateral), and another turned a strike-slip event into a thrust earthquake on a non-existent fault.
2. Roughly one-fifth of the sampled chunks came from papers with no real seismology content at all (e.g., survivor surveys, social media analysis, or disaster governance). The prompts still asked earthquake questions, forcing the teacher to invent the entire seismological background from memory.
3. Gemma's automated audit checked each generated report *only against the source excerpt*. A fabricated claim that does not appear in the excerpt cannot contradict the excerpt, so unless Gemma happened to know the specific fault geometry itself, the check passed the hallucination.

**Fix.** None of these rows ever entered training. We cancelled `SFT v2` and `RL v3` and reassigned the queued 8-node allocation to run a second seed of `RL v2` (see [README §8.4](README.md#84-knowledge-dense-report-data-and-sft-v2)). A viable future version of this pipeline would restrict questions strictly to facts present in the excerpt, require every factual claim to cite a specific sentence in the source, and filter papers strictly for domain content first.

**Permanent regression test.** None in code. The permanent guard is a mandatory pre-training gate: manually or model-auditing a random sample (one row per paper) against both the source text and domain truth before training on synthetic knowledge data.

**Invalidated artifacts.** All 3,262 generated knowledge rows (discarded before any training).

**Generalisable lesson.** A source-consistency check only verifies claims that exist in the source. If your generation prompt asks a teacher model to supply domain knowledge from memory, it will hallucinate outside the source—and source-grounded checks will pass those hallucinations unless checked independently against domain ground truth.

---

## A5 — A semicolon that turned an error into a true statement

**Symptom.** On one `D-Report` evaluation question, the grading rubric listed a completely true scientific statement in the "common errors to penalise" list—while another criterion in the very same rubric rewarded the model for stating that exact fact.

**The false story.** No dramatic misconception arose here; the judge model simply saw contradictory instructions in the prompt and tried its best to score each pair anyway.

**Root cause.** During test-set drafting, the list of common errors was generated as a single prose string, and `dpt.eval.autocheck` split it into separate error items at every semicolon (`;`). One drafted error contained a semicolon joining two clauses (`"...; which is only true when..."`). Splitting at the semicolon turned the second clause into a standalone "error," and standing on its own without the first clause, that statement was factually true.

**Fix.** When splitting prose error strings, any fragment beginning with a back-reference word (`that`, `this`, `it`, `which`, etc.) is now kept attached to the preceding item rather than promoted into an independent error. In evaluation reporting, the frozen test set retains the original item for consistency, while the screened subset drops it (see [README §8.5](README.md#85-a-model-review-of-the-tests-and-the-data)).

**Permanent regression test.** `tests/eval/test_autocheck.py::test_an_error_split_at_a_semicolon_keeps_its_back_reference`.

**Invalidated artifacts.** None; dropping this and the other 6 flagged items in the screened subset changes no headline conclusion.

**Generalisable lesson.** Never split natural-language prose on punctuation that writers normally use inside sentences. If you want a structured list from an LLM, ask for a JSON array or numbered list directly.

---

## A6 — A verifier that cannot solve the problem it checks

**Symptom.** On November 16, when we ran verification on the 6 newly added long-horizon numeric families for bank v2 (12 templates, 3,577 problems total), the check kept only 2 of the 12 templates (577 problems). Across the remaining 10 templates, only 0 to 5 of the 20 audited problems were accepted, causing all 10 templates to be rejected whole.

**The false story.** "Our new long-problem generators are writing buggy labels, so those 10 templates should be thrown away. Catching templates whose labels the verifiers cannot confirm is the whole point of the audit."

**Root cause.** Under our verification rule ([README §4.2](README.md#42-verification-a-solver-and-two-models-must-agree)), a template passes only if two verifier models from different families—each solving 20 sampled problems 4 times—agree with the programmatic label, and a model only casts a vote on a problem if at least 3 of its 4 answers match each other.
Inspecting the logs of the 10 rejected templates revealed that zero labels were actually contested: on every audited problem, the verifiers either agreed with the label or failed to cast a vote. Why? Both verifiers run with a 16K token cap, whereas these new families require 6–8 reasoning steps:
- Across the 240 audited problems in those 10 templates, the Qwen3.8-27B teacher finished only 440 of its 960 responses (so it had 3+ completed replies to compare on just 86 problems).
- Gemma produced a final value in 695 of 960 responses.
- Yet in our local difficulty pilot at a 24K budget, the smaller SFT model had already reached the exact programmatic label on 2–70% of attempts across those same five families ([README §8.3](README.md#83-rl-v2-the-test-the-changes-the-job)).

The verifiers had not caught wrong labels—they had simply run out of tokens.

**Fix.** Any template whose audited problems are purely agreed or undecided—with zero contested labels and zero perturbation failures—is now readmitted to the SFT difficulty pass (`dpt.data.bank_merge.readmit_undecided`), bringing 2,997 long problems from five families back into consideration. Each problem is tagged `label_check: sft`, and the difficulty pass keeps a problem only if the SFT model reaches its programmatic label in at least 1 of its 8 answers at 24K. Any problem the SFT model reaches 0 of 8 times is discarded rather than placed in the hard training pool, because nothing has yet confirmed its label (`dpt.data.difficulty.drop_unconfirmed`).

**Permanent regression test.**
- `tests/data/test_bank_merge.py::test_merge_readmits_templates_whose_verifiers_were_undecided`
- `tests/data/test_difficulty.py::test_a_readmitted_label_needs_one_right_answer`

**Invalidated artifacts.** None. The fix landed on seed 1's bank build right before the step that merges verified families.

**Generalisable lesson.** An automated check is only as capable as the checker's compute budget. Before discarding data because verifiers failed to confirm it, check whether they disagreed with the label or simply failed to finish—"undecided" does not mean "wrong."

---

# B · Numerics and memory

## B1 — Parity passed, training was wrong: the inverted attention mask

**Symptom.** During our workstation RL smoke test on Qwen3.5-0.8B (2026-10-07), the PPO KL divergence jumped to 0.35–0.50 within a single mini-batch when it should have been near zero. On identical weights, the training forward pass assigned log-probabilities 1–2 nats per token *lower* on the sampled tokens than the rollout engine had assigned seconds earlier.

**The false story.** "vLLM and Megatron just have numerical differences; our vLLM-vs-Megatron parity test already passed, so this is normal training-inference mismatch and Truncated Importance Sampling (TIS) will absorb it."

**Root cause.** Two software conventions collided only on the training backward path of our specific GPU architecture:
1. For Qwen3.5's vision-language (VLM) path, `verl 0.9.1` builds a Hugging Face–style keep mask (`True` = real token to attend to). `Megatron-Bridge` passes that tensor unchanged into Megatron's attention layer, where `True` means masked out (ignore token).
2. Whether that inverted mask causes damage depends entirely on which attention kernel Transformer Engine selects. In forward-only passes (computing old log-probs and evaluation), cuDNN fused attention uses an internal causal mask and ignores the supplied mask tensor—so everything is correct. However, on A100 GPUs with `head_dim = 256`, cuDNN has no fused backward kernel for that head dimension. During the training forward pass, Transformer Engine silently falls back to *unfused* attention, which actually applies the supplied mask—and promptly blanks out every real token in Qwen3.5's full-attention layers.
3. Our existing parity test only compared vLLM against Megatron's *evaluation* (no-grad) forward pass—the one path that ignored the broken mask.

**Fix.** Fix 1 in `src/dpt/verl_ext/patches.py` installs an import-hook patch that converts the keep mask into Megatron's expected `[B, 1, 1, S]` boolean mask (`True` = masked out). On Qwen3.5-0.8B, `mean(old - new)` log-probability dropped from `1.80` nats/token before the fix to `3e-4` after—indistinguishable from BF16 kernel noise.

**Permanent regression test.**
- `tests/verl_ext/test_patches.py::test_keep_mask_becomes_megatron_mask`
- `python -m dpt.parity.train_forward_parity` (step 4 of `scripts/02_parity.sh`), a GPU parity check that compares Megatron's *training* (grad-enabled) forward pass—not just its eval pass—against rollout.

**Invalidated artifacts.** Every RL and SFT smoke run executed before this fix.

**Generalisable lesson.** A numerical parity check only proves correctness for the exact execution path it invokes. Always verify the *training* forward pass (with gradients enabled) on the *actual target GPU architecture*. Running the smoke test on an H100, whose fused kernel supports `head_dim = 256`, would have hidden this bug completely until A100 training began.

---

## B2 — The last pipeline stage: a budget, a probe, and a patch that all missed

**Symptom.** Testing on a single A100 right before launching our first 9B SFT run (2026-10-28) showed that the final stage of our 8-stage pipeline (stage 7, holding the output head) ran out of memory at sequences of just ~3.5K tokens—while 28% of our SFT rows are longer than 4K tokens (up to 20,841 tokens).

**The false story.** "Our memory planner budgeted 32.3 GB for the heaviest pipeline stage, so 40 GB A100s have plenty of headroom."

**Root cause.** Three separate blind spots stacked on top of one another:
1. Incomplete analytical budget. The memory planner originally assumed the *first* (embedding) stage was heaviest at 32.3 GB. It failed to count two items on the *last* stage: (a) the Multi-Token Prediction (MTP) head keeps its own full copy of the 1.0B-parameter vocabulary embedding, and (b) computing full FP32 logits across a 248,320-token vocabulary under tensor parallelism 1 (`TP1`) stays completely unsharded (in the original 27B design, `TP4` split the vocabulary four ways).
2. Unrepresentative 1-GPU probe. Our single-GPU probe tested the cross-entropy loss directly without wrapping the model in Megatron's `Float16Module`. Normally, `Float16Module` upcasts the final stage's output to FP32. In Megatron's native trainer that output is just a scalar loss; in `verl`, that output is the entire `[B, S, 248320]` logits tensor. Consequently, the 8-node pre-flight crashed OOM while allocating a single `18.8 GiB` FP32 copy of the logits.
3. Silent no-op monkey-patch. One of our memory-saving patches targeted a method on verl's base class, whereas verl actually defined that method on a subclass. Worse, the patch's unit test used a fake class that mirrored the patch's wrong assumption, so the test passed while production code patched nothing.

**Fix.** We kept the data, loss math, and optimizer semantics unchanged and moved memory:
- Placed FP32 master weights and Adam states in host RAM via Megatron's CPU-offloaded hybrid optimizer (`−11.7 GiB` GPU memory).
- Patched both main and MTP cross-entropy losses to run Transformer Engine cross-entropy directly on BF16 logits (`2.38 → 0.94 MiB/token`).
- Kept logits in BF16 out of `Float16Module` specifically during verl's LM-head forward (`−18.8 GiB` on the longest row).
- Rebalanced pipeline layers unevenly so the first and last stages each hold only 1 decoder layer instead of 4.
Values match an FP32 reference within `1e-4` nats/token and gradients within `1%` relative L2 norm. Missing patch targets now raise an explicit `RuntimeError`. After these fixes, an 8-node pre-flight on the 12 longest SFT rows peaked at `40,366` MiB out of `40,437` MiB available.

**Permanent regression test.**
- `tests/verl_ext/test_patches.py::test_engine_fix_refuses_a_missing_target`
- `tests/verl_ext/test_patches.py::test_float16_output_stays_bf16_only_inside_the_lm_head_step`
- `tests/verl_ext/test_patches.py::test_fixes_reach_the_installed_stack` (imports and verifies the *real installed classes*, not fakes)
- Mandatory 2-step 8-node pre-flight on the longest rows before any new parallelism layout.

**Invalidated artifacts.** The initial 9B memory plan; its last-stage estimates were replaced with actual measurements.

**Generalisable lesson.** Measure memory on the heaviest pipeline stage at the real maximum sequence length through the exact wrapper stack the trainer uses before launching multi-node jobs. And always verify runtime patches against the real installed packages—never against fake classes written from memory.

---

## B3 — No attention kernel at the real sequence length

**Symptom.** The very first 9B SFT forward pass crashed immediately with: `No dot product attention backend is available`.

**The false story.** "The Python/CUDA environment is corrupted; we need to rebuild it from scratch." Yet that exact environment had passed every single 0.8B and 4K-token smoke test without issue.

**Root cause.** Our 9B SFT YAML config inherited `attention_backend: flash` from the 27B recipe, even though `flash-attn` was not installed in the cluster environment. Even changing the setting to `auto` could not save real-length training: on A100 GPUs, Transformer Engine rejects cuDNN fused attention whenever `head_dim > 128` (Qwen3.5 uses `head_dim = 256`), leaving only unfused attention. Unfused attention materialises the full `S × S` attention score matrix—costing `1.6 GiB` per layer at 4,096 tokens, `13.5 GiB` at 12,288 tokens, and crashing OOM on a single layer at 16,384 tokens. Because all workstation smoke tests ran at 4K tokens, no test had ever exercised real SFT sequence lengths (up to 20,841 tokens).

**Fix.** We compiled `flash-attn 2.8.3` for SM80 (8 minutes on one compute node, verified by SHA-256) and set `attention_backend: auto`. A single layer's forward + backward pass then drops to `0.7 GiB` at 20,841 tokens. Checked against unfused attention, worst relative L2 difference on outputs and gradients is `5.6e-3` (normal BF16 noise), and full-model log-probs at 8,192 tokens pass within our `5e-3` / `5e-2` parity thresholds.

**Permanent regression test.** The cluster environment build script now builds and import-checks `flash-attn`, and the pre-flight runs the longest actual rows from the training set.

**Invalidated artifacts.** None; no 9B training step had completed yet.

**Generalisable lesson.** Short-sequence smoke tests cannot expose quadratic memory or backend-selection failures. Always run at least one forward+backward step at your true maximum sequence length, and audit every default a config inherits from a recipe written for different hardware.

---

## B4 — Host memory: the colocated servers counted too

**Symptom.** During RL run 1, saving the step-1 checkpoint exhausted host RAM on the last pipeline-stage node, causing Ray to kill all three colocated vLLM servers on that node. Resuming the job then crashed in the exact same way while loading the optimizer state back into CPU memory.

**The false story.** "The 9B checkpoint is simply too big for a single node's host memory." Yet our SFT pre-flight had saved the exact same Megatron checkpoint on the exact same node layout without any issue.

**Root cause.** Because Multi-Token Prediction (MTP) speculative drafting was enabled during rollout, `verl` put the colocated vLLM engines into sleep level 1 during Megatron training steps. Sleep level 1 keeps a full CPU copy of every vLLM weight in host RAM: each sleeping replica consumed ~26 GB (~20 GB from that weight copy alone), totalling ~78 GB for the three replicas sharing the node. The last pipeline stage already held ~202 GiB of its 251 GiB host RAM in use (mostly CPU-offloaded Adam states and FP32 master weights). When checkpoint saving gathered the stage's optimizer state onto rank 0 (`31 → 66 GB`), total RAM crossed Ray's 95% OOM threshold. The SFT pre-flight had never seen this because no vLLM servers were colocated during SFT.

**Fix.** We turned off MTP speculative drafting during rollout (`actor_rollout_ref.model.mtp.enable_rollout: false` in `configs/rl/rl_9b.yaml`) while continuing to train the MTP head. Without drafting, vLLM can enter sleep level 2, which drops weights entirely on sleep and reloads them via weight sync before the next rollout. Since speculative decoding is lossless, generation quality is unchanged (only generation throughput is slightly slower), and the three replicas' host RAM footprint falls from 78 GB to 9 GB.

**Permanent regression test.** The documented config override (`enable_rollout: false`) plus per-node host memory logging (`rl.json`, metric `host_memory_stage8`).

**Invalidated artifacts.** Run 1's step-1 weight update (which was also contaminated by broken code rewards from [C1](#c1--what-a-batch-job-inherits-an-ssh-shell-does-not)); RL was restarted cleanly from step 0.

**Generalisable lesson.** A memory pre-flight must run the *entire co-scheduled set of processes*—trainer plus inference servers—at the same time, and monitor host RAM just as closely as GPU VRAM.

---

# C · Environment and silent defaults

## C1 — What a batch job inherits, an ssh shell does not

**Symptom.** This failure struck twice in different disguises. On October 27, all 24 vLLM rollout servers crashed immediately at startup. Two days later (October 29), our first RL dev pass scored the SFT model at just 10.4% on coding tasks—even though that exact checkpoint had just scored 86% on `D-Code`. Closer inspection showed that identical samples of the exact same prompt passed or failed depending purely on which cluster node happened to grade them.

**The false story.** For the 10.4% code score: "RL's dev coding problems must be much harder than D-Code," or "the T2 code reward parser is buggy."

**Root cause.** Our multi-node campaign launcher starts worker processes on remote nodes via `ssh node cmd`. On this Slurm cluster, a non-interactive `ssh` command shell does *not* inherit the default Lmod environment modules that a Slurm batch script receives:
1. In the first failure, worker nodes lacked the newer `libstdc++` needed by FlashInfer's sampling kernel, crashing all 24 vLLM servers.
2. In the second failure, Lmod was stubbed out in ssh shells (`LMOD_CMD=:`), so `module load apptainer` silently did nothing on 7 out of 8 nodes. By security design, our code-grading sandbox (`dpt.rewards.code_sandbox`) fails closed when `apptainer` is missing—it never runs model-generated code outside a container—so every sample graded on those 7 nodes scored `0` and was logged as an infrastructure error. That error counter was recorded in logs, but had no automatic alert attached to it.

**Fix.** `scripts/slurm/env.sh` now explicitly preloads the required `libstdc++` on every compute node and falls back directly to the default Apptainer installation binary path if `module load` fails to put `apptainer` on `PATH`. We also verify sandbox isolation (`dpt.rewards.sandbox_check`: confirming `$HOME`, `$WORK`, `$SCRATCH`, and network access are all blocked) across worker nodes before starting. With the fix in place, RL step 0's dev code score immediately jumped to `79.2%` with zero infrastructure errors.

**Permanent regression test.** `tests/slurm/test_job_scripts.py::test_env_finds_apptainer_on_a_node_reached_over_ssh`.

**Invalidated artifacts.** The initial RL launch's step-1 update.

**Generalisable lesson.** Failing closed is essential for security, but a silent fail-closed mechanism is a silent failure mode: always trigger loud alerts whenever any reward's infrastructure-error rate rises above zero. And always test environment setup via the *exact process-launch path* (`ssh` vs `srun` vs `sbatch`) used in production.

---

## C2 — Connection pools and timeouts: one default, three places

**Symptom.** Three seemingly unrelated performance and reliability glitches appeared across data verification and evaluation: (1) problem verification crawled at ~3.2 problems/min; (2) the evaluation harness kept only ~38 of each vLLM replica's ~114 active slots busy; and (3) 27 `pass@64` evaluation requests failed at *exactly* the 1-hour mark.

**The false story.** "The vLLM inference servers are slow," or "the model takes forever on long answers."

**Root cause.** One overlooked default (`httpx`'s default connection pool cap of 100 connections) interacted badly with client and proxy timeouts in three places:
1. Both our verification client (`src/dpt/data/backends.py`) and evaluation harness (`src/dpt/eval/harness.py`) left `httpx`'s 100-connection limit unchanged even though each asked for hundreds of concurrent requests in flight—starving the servers of queued work.
2. Once we raised the connection pool limit, judge requests queued up longer than the client's 30-minute timeout. Clients timed out and retried those requests, while our round-robin proxy kept forwarding and running the abandoned originals in the background—doubling server load and stalling verification for an hour.
3. Our round-robin proxy (`scripts/rr_proxy.py`) enforced a 1-hour read timeout on its backend connections, whereas `vLLM` buffers `n=64` non-streaming completions and sends *zero bytes* until all 64 samples finish.

**Fix.** We sized `httpx` connection pools dynamically to match configured request concurrency, set client timeouts using Little's law (`queue wait ≈ requests in flight / throughput`), removed the proxy's artificial read timeout, and configured the proxy to abort backend requests immediately when the client disconnects. Evaluation throughput jumped from ~900 to ~2,250 tokens/s per replica at run start.

**Permanent regression test.**
- `tests/slurm/test_rr_proxy.py::test_every_request_reaches_the_backend_and_a_client_that_gives_up_cancels_it`
- `tests/slurm/test_rr_proxy.py::test_the_proxy_sets_no_read_timeout_of_its_own`
- `tests/data/test_difficulty.py::test_difficulty_requests_fit_in_the_base_timeout`

**Invalidated artifacts.** None; the affected ~1.5 hours of `pass@64` evaluation was re-run cleanly.

**Generalisable lesson.** When a high-concurrency client talks to a batching inference server through a proxy, match connection-pool limits to concurrency, size client timeouts to cover queuing time (Little's law), and propagate client cancellations all the way to the backend.

---

## C3 — Framework defaults that override the config

**Symptom.** Our first full SFT run crashed at step 50 during a validation pass—even though `val_files: null` was set in the YAML config because there was no validation dataset. Separately, a config audit discovered that SFT was about to train with Adam `β2 = 0.999` instead of our specified `0.95`.

**The false story.** "Our YAML config explicitly sets `val_files: null` and `betas: [0.9, 0.95]`, so those are the settings the trainer is running."

**Root cause.** `verl 0.9.1` ignored or overrode both config intent settings silently:
- It runs a validation loop whenever `trainer.test_freq > 0`, even when `val_files` is null.
- It never forwards `optim.betas` down to Megatron's optimizer, leaving Megatron at its built-in default of `β2 = 0.999`.
(This is the same class of inherited/ignored-default bug as `attention_backend` in [B3](#b3--no-attention-kernel-at-the-real-sequence-length).)

**Fix.** Set `trainer.test_freq: -1` to disable validation cleanly, and pass `β2 = 0.95` directly to Megatron's optimizer config—with inline comments explaining why in `configs/sft/sft_27b.yaml`.

**Permanent regression test.** Explicit comments and values locked in the configs; our SFT pre-flight now runs past the first configured validation boundary.

**Invalidated artifacts.** About one hour of SFT compute (no checkpoint existed prior to step 100).

**Generalisable lesson.** A config file expresses *intent*; the framework's wiring determines what *actually runs*. Before committing to a long multi-node run, print and diff the resolved framework runtime configuration (optimizer betas, attention backend, validation schedule) against your intended recipe.

---

## C4 — A file lock on NFS: a hang costs more than a crash

**Symptom.** SFT hung completely before executing its very first step. Ranks on seven of the eight nodes timed out after 30 minutes, while the ranks on node 3 could not even exit—remaining wedged inside the Linux kernel (`D` state, uninterruptible sleep) and emitting 172 MB of heartbeat error logs.

**The false story.** "NCCL must have deadlocked during collective initialization; just wait for the timeout and relaunch the job."

**Root cause.** When `Megatron-Bridge` loads a Hugging Face model config, it acquires a filesystem lock under `~/.cache/huggingface`. On this cluster, `$HOME` is served over NFSv3. Three lock requests from node 3 hung indefinitely inside the NFS client and never returned. Had this been a clean Python crash, our job script would have ended or triggered recovery in minutes; because it was a kernel hang with no stall watchdog, 8 nodes sat completely idle for 9.6 hours (77 node-hours) until a human noticed.

**Fix.** We redirected Megatron-Bridge's config lock directory to node-local `/tmp` via `MEGATRON_CONFIG_LOCK_DIR` in `scripts/slurm/env.sh`—since models are loaded offline from pre-downloaded weights, no cross-node lock is needed. Crucially, we rescued the live Slurm allocation without losing our queue place: we paused the hung campaign with `SIGSTOP`, launched a second campaign copy inside the same allocation skipping finished steps, and used a small supervisor process to terminate the paused copy once the replacement exited ([E1](#e1--the-queue-sets-the-pace)).

**Permanent regression test.** `tests/slurm/test_job_scripts.py::test_env_keeps_megatron_config_locks_off_nfs_home`.

**Invalidated artifacts.** None.

**Generalisable lesson.** Never place file locks on shared network filesystems like NFS. Equally important: wire up a training-step stall alarm (e.g., alert if no step logs for 20 minutes) *before* launching, because nothing in Slurm or PyTorch automatically flags a deadlocked job that hasn't crashed.

---

## C5 — A GPU embedder beside the servers

**Symptom.** A 2-node problem-bank build job crashed 8 minutes after starting with a CUDA out-of-memory error on GPU 0 inside both verification client processes.

**The false story.** "Verification clients only send HTTP requests to the model servers; they don't run neural networks, so we don't need to isolate GPUs for them."

**Root cause.** During bank building, the decontamination step runs inside the verification client process and loads an embedding model (`Qwen3-Embedding-0.6B`). Whenever PyTorch sees an available GPU, it puts that embedder on GPU 0—which on that node was already running the Qwen3.8-27B teacher server with `36.4 GiB` of its `40 GiB` VRAM occupied.

**Fix.** Set `CUDA_VISIBLE_DEVICES=""` for those client processes so the embedder stays strictly on the CPU. Because the crash terminated the job, we had to resubmit it and lost a 4.5-day queue position.

**Permanent regression test.** `tests/slurm/test_job_scripts.py::test_data_window_runs_every_server_and_client_on_its_own_two_nodes`.

**Invalidated artifacts.** None (aside from the lost queue wait).

**Generalisable lesson.** On nodes shared with GPU inference servers, explicitly declare `CUDA_VISIBLE_DEVICES` for every helper process—even lightweight "HTTP clients." Any library that *can* grab a visible GPU eventually will.

---

## C6 — A login node's thread limit that followed the job

**Symptom.** RL v2 seed 1 was building bank v2 on its dedicated 8-node allocation. After loading its embedding model, decontamination of the new families wrote zero output for 8 straight hours while two teacher GPU nodes sat completely idle. Inspecting the compute node showed both verification client processes pinned at 100% of a *single* CPU core on a 128-core machine.

**The false story.** "Decontamination has to embed 11,000 reference questions on the CPU; CPU inference is just slow."

**Root cause.** To avoid hitting the login node's 8 GB virtual-memory cap when running multi-threaded HTTP tools, `scripts/slurm/env.sh` sets `OMP_NUM_THREADS=1` whenever sourced on a login node. However, `sbatch` exports the submitting shell's environment variables into the queued Slurm job by default. As a result, a job submitted from a login node carried `OMP_NUM_THREADS=1` onto 128-core compute nodes, forcing `Qwen3-Embedding-0.6B` (via PyTorch and MKL) onto a single thread. At that speed, decontamination would have taken over 24 hours out of a 47-hour job budget meant to cover bank building, RL training, and final evaluation.

**Fix.**
- Live rescue: Rather than killing the clients (which would have aborted the phase and lost the 8-node allocation), we attached a debugger (`gdb`) to both running processes and called PyTorch's C++ `at::set_num_threads(56)`. Both processes finished decontamination 12–16 minutes later.
- Permanent fix: The window script explicitly overrides `OMP_NUM_THREADS=48` and `MKL_NUM_THREADS=48` inside the embedding clients.

**Permanent regression test.** `tests/slurm/test_job_scripts.py::test_data_window_on_eight_nodes_with_two_teacher_nodes_and_no_t3`.

**Invalidated artifacts.** None, though it burned 8 hours of the 47-hour job window.

**Generalisable lesson.** Any resource guardrails set for a login node travel silently into every Slurm job submitted from that shell. Always reset threading and memory environment variables explicitly inside the job script itself.

---

## C7 — A folder 24 ranks create at once on a shared file system

**Symptom.** RL v2 seed 2 finished training its 110th step and immediately crashed with `"training exited with code 1 without a clean stop"`. The step-110 checkpoint directory contained only 184 KB, leaving step 100 as the last valid saved checkpoint.

**The false story.** "Our early-stop mechanism at step 110 tried to save the checkpoint a second time and collided with the regular save." Because seed 2 was the first run configured to stop at a fixed step matching seed 1 (`110`), and step 110 was also a regular `save_freq=10` step, a duplicate-save race looked very plausible. However, the stop handler explicitly checks whether the current step has already been saved, and the stack trace pointed inside `verl`'s own training loop.

**Root cause.** `verl` creates checkpoint folders using `local_mkdir_safe`, which serialises `os.makedirs(path, exist_ok=True)` using a lock file inside each node's local `/tmp`. That prevents races between processes on the *same* node, but offers zero protection across *different* nodes racing on a shared Lustre filesystem. At step 110, one rank's `makedirs` raised `FileExistsError` twice on `global_step_110/actor/model/dist_ckpt`: another node had just created the directory entry in Lustre, so it existed in the namespace, yet had not yet propagated as a directory to that rank's Lustre client. Saving aborted and terminated the job—purely by chance, the previous 20 saves had avoided the race.

**Fix.** Fix 4 in `dpt.verl_ext.patches` wraps `local_mkdir_safe` so that on `FileExistsError`, it pauses briefly and retries up to five times while waiting for the path to appear as a directory, before re-raising if it truly fails. The patch is injected at every module where `verl` imports `local_mkdir_safe` by name.

**Permanent regression test.**
- `tests/verl_ext/test_patches.py::test_mkdir_retries_while_another_node_creates_the_folder`
- `tests/verl_ext/test_patches.py::test_fixes_reach_the_installed_stack`

**Invalidated artifacts.** Seed 2's training updates from steps 101–110 were never persisted to disk, so seed 2's evaluation uses its step-100 checkpoint.

**Generalisable lesson.** A node-local lock in `/tmp` does not serialise multi-node filesystem metadata operations. Any code executed concurrently across nodes on a shared parallel filesystem like Lustre must tolerate partially visible directory creation from peer nodes.

---

## C8 — A sandbox that failed on one node, with a warning at every dev pass

**Symptom.** Throughout RL v2 seed 2, the training code reward hovered around 0.85 from step 1 onward (compared to 0.97 on seed 1). Then, when the same 8-node job ran final evaluation immediately after training, it scored 0.0 on both `D-Code` and `LiveCodeBench`.

**The false story.** "Seed 2 just happened to draw a harder batch of coding problems early on." Because seeds shuffle the training bank in different orders, a 12-point gap in training reward seemed plausible enough to ignore. A `0.0` evaluation score, however, could not be explained by prompt sampling.

**Root cause.** On one of the 8 compute nodes in seed 2's allocation, the code sandbox's startup container probe failed (the exact node-level trigger was not diagnosed before the allocation ended). Crucially, the sandbox implementation cached that failed probe result *for the lifetime of the Python process*. As a result:
1. During RL training, every single code sample routed to that node failed closed and scored `0`, corrupting 10–17% of coding rewards on every dev pass. Although the `sandbox_infra` monitor printed a warning on every dev pass, its auto-pause threshold was set at `>20%`, and nobody read the non-fatal warning.
2. When final evaluation began inside that same allocation, that broken node happened to serve as the head node running the evaluation sandbox. Its probe failed at startup, the evaluation harness failed to verify sandbox health before grading, and every single `D-Code` and `LiveCodeBench` answer scored `0.0`.

**Fix.**
- A failed container probe now retries after 60 seconds rather than latching permanently (`CONTAINER_RETRY_S` in `dpt.rewards.code_sandbox`); only successful probes are cached.
- Final evaluation explicitly waits for the sandbox to report warm and restarts it once if it does not.
- Any evaluated coding set with `>5%` sandbox infrastructure errors now refuses to report a score (`no score`) instead of silently publishing an artificially low number.
- The job-watcher script surfaces active warnings prominently.

**Permanent regression test.**
- `tests/rewards/test_code_sandbox.py::test_a_failed_container_probe_is_tried_again`
- `tests/eval/test_harness.py::test_a_set_the_sandbox_could_not_run_gets_no_score`
- `tests/slurm/test_job_scripts.py::test_final_eval_restarts_a_cold_sandbox_and_a_failed_report_does_not_stop_it`

**Invalidated artifacts.** Seed 2's T2 training signal (~1 in 8 code responses scored 0 regardless of correctness), meaning its workstation-regraded evaluation scores (`D-Code 90.8`, `LiveCodeBench 56.1`) likely understate the recipe's potential. Numeric tasks (`T1`) do not use the sandbox and are completely unaffected.

**Generalisable lesson.** A warning that fires on every check yet blocks nothing is a silent failure waiting to happen. Either ensure a human actively reads every warning, or lower the threshold so persistent infrastructure faults automatically pause the run.

---

# D · Measurement and analysis

## D1 — "T4 taught brevity": a story the per-step metrics did not support

**Symptom.** After step 20 of RL run 1, response lengths across every task shrank steadily until numeric and coding accuracy collapsed completely to zero at step 52 ([README §6.4](README.md#64-twenty-good-steps-then-a-collapse)).

**The false story.** "T4—our general instruction-following task—rewards short answers and supplied the majority of the gradient early on, so T4 taught the model to be brief." This clean narrative was written directly into our early run notes, and we came close to cutting T4's sampling weight in half for RL v2 based on it.

**Root cause.** Nobody had checked whether the per-step training logs actually supported the story. When we finally checked, the metrics directly contradicted it: within any given training step, *correct* and *incorrect* answers had nearly identical lengths (steps 1–20 medians: `3,749` vs `3,341` tokens on T4; `2,058` vs `1,796` tokens on T1). Thus, T4's reward was not selecting for brevity. Moreover, responses across *all* tasks began shrinking from the very first few steps. The only established fact was that while T1 and T2 were already nearly solved, T4 happened to supply most of the non-zero gradient. (A later replay of step 0 in [README §6.4](README.md#64-twenty-good-steps-then-a-collapse) and [§8.6](README.md#86-why-rl-v2-trains-where-run-1-collapsed) showed that the 12K–16K soft length penalty drove 28% of step 0's advantage by docking long right answers, though the collapse from step 14 to step 40 remains open.)

**Fix.** The "T4 taught brevity" explanation was formally withdrawn on 2026-11-06 and the collapse cause recorded as open. RL v2 added the instrumentation needed to monitor length collapse cleanly: dev evaluations every 10 steps, within-group length-gap logging over finished answers, and an alert if T1 answers shrink by 60% within 30 steps.

**Permanent regression test.** Every task now logs per-step `task/*/len_correct` and `task/*/len_incorrect` metrics (plus within-group length gaps), allowing any run's length hypothesis to be verified directly from logs.

**Invalidated artifacts.** The "T4 taught brevity" explanation in early talk notes and slides.

**Generalisable lesson.** Correlation between a task's gradient share and a model behavior does not prove mechanism. Before committing to a causal explanation—and especially before redesigning a training mix around it—identify the exact metric that would *falsify* your hypothesis and check it.

---

## D2 — One model as reward judge, data filter, and evaluator

**Symptom.** On `D-Report`, our primary judge (`Gemma 4 26B-A4B`) preferred the SFT model's reports over the base model's by 80 wins to 8 losses (an 86.0% win rate).

**The false story.** "Supervised fine-tuning improved seismology report writing by 36 percentage points over baseline parity."

**Root cause.** Gemma was wearing three hats at once: it selected the SFT report training rows via rejection sampling, served as the T3 RL reward judge, *and* acted as the `D-Report` evaluator. When we brought in an independent second judge that had never touched training data or rewards (GPT-6 Luna), Luna scored those exact same SFT-vs-base pairs at 49.0%—effectively a tie. The two judges agreed on fewer than half of all pairs, and at a 70-point passing threshold, only 8 of the 108 reports Gemma passed held up under Luna. SFT had learned Gemma's stylistic preferences, not better science.

**Fix.** Every pairwise report evaluation now requires two independent judges, evaluated in both presentation orders, with shipping gates reading the *lower* win rate of the two. During RL, a drift audit (`scripts/slurm/judge_drift.sh`) periodically re-scores dev reports using the second judge to catch reward hacking (none occurred in run 1). For future report training, RL v2 also rebuilt rubrics into atomic yes/no checks backed by verbatim quoted spans.

**Permanent regression test.**
- `tests/eval/test_second_judge.py::test_min_gate_reads_the_lower_judge_and_reports_agreement`
- `tests/eval/test_second_judge.py::test_min_gate_without_second_judge_fails_closed`

**Invalidated artifacts.** Single-judge `D-Report` win rates cited in early project notes.

**Generalisable lesson.** Never use the same judge model that filtered training data or supplied RL rewards as your sole evaluator. Keep at least one evaluation judge completely isolated from the training pipeline and report both judges side by side.

---

## D3 — A difficulty filter run with the wrong model

**Symptom.** Across steps 1–20 of RL run 1, 87% of numeric and coding prompt groups had all 8 sampled completions right—leaving only 88 to 272 of each step's 1,008 samples with any non-zero GRPO advantage.

**The false story.** "Our problem bank was already difficulty-filtered: any problem the model solved 8 out of 8 times was dropped before RL."

**Root cause.** The difficulty filter had been run using the base model, for which roughly half of T1 fell in the informative middle difficulty band. However, RL did not start from the base model—it started from the SFT model, which solved nearly all those same problems effortlessly. Problem difficulty is not an intrinsic constant; it is strictly relative to the policy about to train on it.

**Fix.** For RL v2, we re-graded the entire problem bank using the *SFT policy* itself (8 samples per problem, medium effort, checked at both 24K and 16K tokens). We drop problems solved 8/8 within 16K, place 0/8 problems in a separate hard pool joined at step 20, oversample intermediate-difficulty problems, and enforce a pre-RL gate requiring at least 40% mixed groups on T1 at step 1 ([README §8.3](README.md#83-rl-v2-the-test-the-changes-the-job)).

**Permanent regression test.** `tests/data/test_rl_build.py::test_dev_split_comes_from_candidates_fixed_before_the_difficulty_pass`; bank validation also prints pass-rate distributions prior to RL launch.

**Invalidated artifacts.** Bank v1's difficulty labels (for RL v2).

**Generalisable lesson.** Always run difficulty filtering using the exact policy checkpoint—and token budget—that will actually enter RL training, and verify the fraction of mixed-outcome (non-zero gradient) groups before burning cluster hours.

---

## D4 — Gates on a saturated test

**Symptom.** Because the base 9B model already scored 91.6 on `D-Num` (`pass@16 ≈ 99.8%`), our pre-defined RL-entry gate checks—requiring `SFT > base` strictly and `pass@16 >= base`—were effectively coin flips governed by sampling noise. Worse, failing the gate automatically terminated the 8-node Slurm job.

**The false story.** "Strict automated gates protect our RL budget from launching on top of a degraded SFT checkpoint."

**Root cause.** The gate thresholds were copied directly from the original 27B planning document, where the reference 27B base model had scored only `52.1` on its own `D-Num` set. Near a 91.6–99.8% ceiling, a single unlucky item out of 300 failing in 64 samples could fail `pass@16`, and the shipped RL gate (`D-Num >= base + 8`, i.e., `99.6%`) was mathematically impossible for any model.

**Fix.** On October 27—before any SFT results existed—we made the RL-entry gate noise-aware: `D-Num` may trail the base by at most the combined 95% confidence interval (`~1.4` points), `pass@16` may trail by 1 point, and a failed gate pauses the job for a human go/stop decision instead of releasing the 8 nodes. We simultaneously froze `D-Num-hard` so progress could be measured well below ceiling.

**Permanent regression test.**
- `tests/eval/test_gates_report.py::test_9b_entry_gate_allows_noise_at_the_d_num_ceiling`
- `tests/eval/test_gates_report.py::test_failed_entry_gate_holds_then_starts_rl_on_go`

**Invalidated artifacts.** The inherited `D-Num` ship-gate threshold (still flagged as unreachable by construction in final gate reports).

**Generalisable lesson.** Never reuse pass/fail thresholds from another model or benchmark without checking where your base model sits relative to the ceiling and how wide its seed-to-seed noise floor is. A gate sitting near saturation measures noise, not regressions.

---

## D5 — A refusal detector too narrow for the model it measures

**Symptom.** Under our safety gate's strict refusal detector, refusals on "should-refuse" prompts fell from 56.0% (base) to 45.7% (RL run 1, step 20).

**The false story.** Depending on bias, people leaned toward one of two wrong extremes: either "RL ruined model safety by 10 full percentage points," or (after reading a handful of replies) "the model just learned new ways to phrase refusals, so there is no real regression."

**Root cause.** The strict detector only matched a fixed list of canned refusal phrases inside the first 1,500 characters of the answer. To test whether phrasing shifts explained the entire drop, we wrote a much broader lexicon detector (acting as an upper bound, since it also picks up safety disclaimers inside compliant answers). Under the broad detector, refusals *still* dropped by 7.7 percentage points (`77.7% → 70.0%`). Thus, roughly one-quarter of the strict drop came from reworded refusals, while three-quarters was a genuine behavioural shift—concentrated specifically on XSTest's unsafe look-alike prompts (prompts worded innocently that request unsafe content) rather than blatantly harmful JailbreakBench requests.

**Fix.** We report *both* the strict and broad detectors side by side and queue the 27 ambiguous answers (refused by the base, classified as non-refusals under both detectors after RL) for human reading. Crucially, the strict detector remains the official gate metric so post-hoc measurement changes cannot mask regressions.

**Permanent regression test.** `tests/eval/test_graders_misc.py::test_refusal_detection`, alongside `scripts/talk/refusal_breakdown.py` for per-category counts.

**Invalidated artifacts.** None; the strict metric was preserved as the gate number.

**Generalisable lesson.** Never widen a safety detector after seeing a score drop, or your measurement tool will absorb real regressions. Fix your primary detector before seeing results, and use secondary detectors strictly for diagnostic breakdown.

---

## D6 — A reviewer that favours the first answer

**Symptom.** When a frontier model reviewer graded 30 blind `D-Report` pairs once each, the trained model won 11 of 11 pairs where its answer appeared first, and only 8 of 19 pairs where it appeared second. Overall, the reviewer picked whichever report came first in 22 of the 30 pairs.

**The false story.** "Unlike quick automatic judges, a careful frontier reviewer grades each report strictly on its merits, so showing each pair in one order is enough."

**Root cause.** Every LLM reader—whether a small judge or a frontier reviewer—exhibits strong position bias. Our automated judges (`Gemma` and `Luna`) already ran every pair in both orders to cancel this out, whereas the manual blind-review pass had initially been run in only one order.

**Fix.** We ran a second blind-review pass with every pair's order swapped (`A/B → B/A`). A win now counts only when both orders agree; the 6 pairs that flipped between passes are recorded as ties (`blind_review page --swap`, `blind_review combine`). Over both orders, RL step 20 achieved a 60.0% win rate over the base model (recorded explicitly as a model review in [README §8.5](README.md#85-a-model-review-of-the-tests-and-the-data)).

**Permanent regression test.** `tests/eval/test_blind_review.py::test_a_swapped_page_and_the_combination_of_both_orders`.

**Invalidated artifacts.** All single-order blind-review verdicts.

**Generalisable lesson.** Position bias affects every LLM evaluator, regardless of capability. Always run pairwise model comparisons in both orders and count only order-consistent wins.

---

## D7 — A number scan that froze a pilot for eight hours

**Symptom.** During a local difficulty-grading pilot on 180 problems, output stopped dead at 00:29 after completing 113 problems. By 01:36, both vLLM servers had finished all queued requests and gone idle, yet the Python client process pegged a CPU core at 100% for eight hours without printing a word or triggering any timeout.

**The false story.** "We're sharing the workstation GPUs with another user's job, so the vLLM servers must just be throttled; give it time to finish."

**Root cause.** To detect when an unfinished answer already found the gold number inside its reasoning trace, `dpt.eval.graders.numeric.value_seen` parses numbers in scientific notation `a × 10^e` and evaluates `10 ** int(e)` as an exact arbitrary-precision Python integer. For a small exponent that is instant; for a 12-digit exponent emitted by a looping/degenerate completion near the token limit, Python begins allocating and computing an integer with a trillion digits. That calculation raises no exception, practically never terminates, and holds Python's Global Interpreter Lock (GIL)—freezing every other thread in the client so no HTTP timeout or watchdog thread in the same process can fire. We discovered this by reading the code and confirmed it with a unit test (the offending model reply was never flushed to disk, though the resumed pilot finished cleanly after the fix). Even worse, the exact same `value_seen` helper is invoked inside RL's reward function (`dpt.verl_ext.reward_fn`) on cut-off completions, meaning it could have deadlocked RL v2 mid-training.

**Fix.** We bounded the digit-run lengths in the regex patterns (an unbounded `\d+` also caused quadratic backtracking on massive digit strings), disallowed matches starting in the middle of a longer number, and immediately skipped any exponent outside standard floating-point range. We pushed the fix to the cluster via an atomic file rename while seed 1's bank build was running.

**Permanent regression test.** `tests/eval/test_numeric_grader.py::test_value_seen_returns_quickly_on_degenerate_replies`.

**Invalidated artifacts.** None; the pilot resumed cleanly from the 113 completed problems.

**Generalisable lesson.** Any parser or grader reading raw LLM output must enforce strict bounds on input size and arithmetic work. A model hitting a token cap can emit arbitrary degenerate strings, and any pure-Python loop that holds the GIL defeats all timeouts inside the same process.

---

## D8 — Parse failures that were not random

**Symptom.** When calibrating report rubric formats against checked gold labels, the binary yes/no rubric initially looked far better than the 0–10 rubric: 14% false fails and 0.72 balanced accuracy (versus 0.62 for 0–10). Yet once we repaired the 48 unparsed judge responses and re-scored them, the yes/no rubric dropped to 24% false fails and 0.66 balanced accuracy.

**The false story.** "Only 48 out of 600 rubric evaluations failed JSON parsing (8%). That's small enough to just drop and compute metrics on the remaining 552."

**Root cause.** Our rubric format required the judge to quote verbatim spans from the model's report inside JSON fields. Whenever a quoted span contained LaTeX math (like `\alpha` or `\sqrt`), the unescaped backslash broke `json.loads`. Crucially, those failures were not randomly distributed: 44 of the 48 broken JSON replies came from the yes/no rubric, and 26 of those 44 were base-model answers—precisely the weakest answers most likely to fail criteria. Dropping the parse failures quietly deleted hard negative examples from one side of the benchmark, artificially inflating balanced accuracy.

**Fix.** The JSON parser in `dpt.rewards.judge` now escapes any backslash not starting a valid JSON escape sequence, permits raw newlines inside JSON strings, and extracts the first complete JSON object containing all required criteria verdicts. We saved the failed rows in a backup, re-parsed/re-scored them, and updated the numbers.

**Permanent regression test.**
- `tests/rewards/test_judge.py::test_latex_in_quoted_evidence_is_parsed`
- `tests/rewards/test_judge.py::test_the_first_json_object_is_read_and_raw_newlines_are_allowed`

**Invalidated artifacts.** Provisional yes/no rubric calibration accuracy numbers (reported once and corrected promptly).

**Generalisable lesson.** Before comparing two methods or formats, check the failure/drop rates of each and inspect *which inputs* failed. Parse errors almost never happen uniformly at random—they cluster on complex, math-heavy, or borderline samples and quietly bias comparisons.

---

# E · Operations

## E1 — The queue sets the pace

**Symptom.** On this shared supercomputer, an 8-node Slurm job waited roughly three days in the queue before starting (on October 29, 470 jobs were competing for 611 total nodes).

**The false story.** "Budget the month in GPU-hours and submit separate Slurm jobs for baseline eval, SFT, SFT eval, RL, and final eval."

**Root cause.** On a heavily congested shared cluster, a crash that terminates your Slurm job costs you *three days of wall-clock queue time*, not merely a few wasted GPU-hours. In our main 71.9-hour 8-node job, planned training and eval stages took 28.7 hours while incidents, fixes, and holds took 43.2 hours—yet holding the allocation let us finish in days instead of weeks.

**Fix.** We structured the entire run into a single ordered pipeline (`scripts/slurm/campaign.sh`) inside one 8-node allocation, backed by three recovery mechanisms:
1. `KEEP_NODES` keeps the allocation alive whenever a stage exits non-zero, letting us fix code and resume in minutes.
2. In-job campaign takeover: if the running driver deadlocks (as in [C4](#c4--a-file-lock-on-nfs-a-hang-costs-more-than-a-crash)), we pause it with `SIGSTOP`, launch a fresh copy inside the same allocation that skips completed stages, and clean up via a small supervisor once finished.
3. Failed quality gates pause and wait for human approval instead of killing the allocation.
For RL v2, we requested Slurm reservation windows with a heterogeneous `8 + 1`-node fallback job (since the normal GPU partition caps jobs at 8 GPU nodes).

**Permanent regression test.** `tests/slurm/test_job_scripts.py` tests the campaign stage chaining, `KEEP_NODES` hold behavior, and window scripts in dry-run mode.

**Invalidated artifacts.** None.

**Generalisable lesson.** When queue waits dominate wall-clock time, engineer your job scripts so that software failures cost minutes *inside* the allocation rather than sacrificing the allocation itself.

---

## E2 — Unwatched waits: an idle job and a cleanup that cancelled another

**Symptom.** Two operational lapses burned precious resources: (1) right after SFT finished, exporting the checkpoint to Hugging Face format failed and 8 nodes sat completely idle for 2 hours (16 node-hours); (2) earlier, a cleanup hook triggered when a duplicate campaign script exited automatically cancelled our standalone RL judge Slurm job—which had been waiting in the queue since October 24 with a special 96-hour walltime extension.

**The false story.** "The job is safely on hold waiting for user input, so nothing is going wrong while we wait."

**Root cause.** Both issues stemmed from missing safeguards around async transitions:
1. Our background job monitor had expired while waiting on a user decision prompt, so when SFT completed and the Hugging Face export immediately failed, nobody noticed for two hours.
2. The campaign cleanup trap cancelled dependent jobs (like the judge server) unconditionally on *any* non-clean or copy exit, rather than checking whether the exit was actually the final shutdown of the campaign.

**Fix.**
- Keep an active background watcher running whenever a live cluster allocation is waiting on human input or stage transitions.
- The Hugging Face export now copies Megatron-Bridge weights directly and runs a strict tensor-level sanity check against the base model (775 total tensors: 333 frozen vision tensors verified identical, 57 of 64 probed trainable tensors verified changed) so silent no-op exports are impossible.
- Cleanup traps now check the exact exit reason before cancelling companion Slurm jobs.

**Permanent regression test.**
- `tests/launch/test_checkpoint_tools.py::test_noop_export_is_rejected`
- `tests/slurm/test_job_scripts.py::test_normal_job_chains_starts_ray_everywhere_and_cancels_successor_on_unclean_exit`

**Invalidated artifacts.** None; the RL judge job had to be resubmitted at the back of the queue.

**Generalisable lesson.** A held Slurm allocation is still burning allocation hours every minute. Never leave an active job unmonitored during human waits, and make sure cleanup hooks that cancel downstream jobs inspect *why* the parent process exited.

---

## E3 — A synced file that needed code the cluster did not have

**Symptom.** Immediately after copying our one-constant fix for the Hutton–Boore magnitude bug ([A3](#a3--a-label-that-agrees-with-its-own-question)) from a local machine onto the cluster, running `import` on the problem generator modules on the cluster failed with an `ImportError`.

**The false story.** "The file checksums match between our local copy and the cluster, so the cluster is safely running the patched file."

**Root cause.** The local working copy had undergone an internal code refactor (prepared for the eventual public GitHub export) that had never been synced to the cluster. Copying the fixed files wholesale from the local copy introduced references to refactored helper names that did not exist in the cluster's older modules. Had a bank build started during that window, it would have died on its first import.

**Fix.** Within 15 minutes, we restored the cluster's original generator files from the 3-hourly filesystem snapshot and applied *only* the one-constant fix. Our sync workflow now enforces two mandatory steps: `diff` every target file against the cluster version *before* copying, and run the exact Python imports used by Slurm jobs on the cluster *immediately after* copying.

**Permanent regression test.** None in `pytest`; enforced directly as part of our cluster sync checklist.

**Invalidated artifacts.** None; no cluster job launched during the 15-minute window.

**Generalisable lesson.** Matching checksums prove only that bytes arrived intact—not that the file runs against the target environment's codebase. Always diff before syncing and run real imports on the destination machine afterward.

---

## E4 — A server that never started, and an evaluation that waited for it

**Symptom.** After RL v2 seed 1 finished training on November 18, the final evaluation phase scheduled inside the same Slurm job hung for an hour and died with `"server … not up after 3600s"`. Inspecting the logs showed that 5 of its 24 model replicas—spread across 3 nodes—had written no log file at all.

**The false story.** "The 9B model must be loading slowly from shared storage; we just need a longer startup timeout." Yet the other replicas on those exact same three nodes had loaded in a few minutes, and a server that crashes during loading always leaves an error traceback in its log. Having zero log files meant those 5 ssh-launched processes had never even started.

**Root cause.** Each of the 24 single-GPU vLLM replicas was launched in the background over `ssh` on its assigned node. For 5 replicas, the background `ssh` command returned without actually spawning the server process (the allocation expired before we could capture why ssh dropped them). What turned 5 missing replicas into a completely lost evaluation phase was the launcher itself:
1. It discarded non-zero ssh launch errors and never checked whether a log file actually appeared.
2. It waited for all 24 replicas under a single 1-hour barrier—so a single silent launch drop killed the entire evaluation phase.

**Fix.** In `scripts/slurm/final_eval_window.slurm`, the launcher now logs any failed launch command, inspects log creation after 60 seconds and immediately relaunches any replica missing a log file, and uses a quorum instead of unanimity: replicas that fail to come up are dropped, and evaluation proceeds as long as at least two-thirds of replicas answer ready.

**Permanent regression test.** `tests/slurm/test_job_scripts.py::test_final_eval_relaunches_missing_replicas_and_goes_on_with_two_thirds`.

**Invalidated artifacts.** None. Training had finished and checkpoints were intact. Final evaluation was resubmitted as its own 4-node job while the headline sets ran locally on a workstation ([README §8.3](README.md#83-rl-v2-the-test-the-changes-the-job)).

**Generalisable lesson.** Whenever a distributed job launches dozens of independent inference servers over `ssh`, verify promptly that every server actually spawned a log/process, retry missing ones, and proceed with a healthy quorum. Requiring 100% of servers turns one silent `ssh` hiccup into days back in the queue.
