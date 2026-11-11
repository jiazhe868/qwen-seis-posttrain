# Data pipeline: RL problem bank and SFT mixture

Code: `src/dpt/data/` · Config: `configs/data/pipeline.yaml` · Tests: `tests/data/`
Related: [contamination.md](contamination.md) (decontamination in depth), `docs/dev/conventions.md` §4 (record schemas).

This document explains **what** each stage does, **why** it is built that way, which
**thresholds** it uses and how they were chosen, how it **fails**, and **how to run it**
on the 4×A100-40GB workstation versus the 2×8×A100-80GB cluster.

---

## 0. The shape of the problem

The report's data has three parts. They are kept strictly apart:

| part | size | produced by |
|---|---|---|
| SFT corpus | ~420M tokens (25% self-distilled general replay) | `sft` stage |
| RL problem bank | ~60k problems, 4 tasks (+7k hard pool) | `ingest` → … → `publish` |
| Eval sets | 1,250 items (D-Num 600, D-Code 300, D-Report 200, D-Fresh 150) | *not here*: stored separately, read only by the decontam stage |

The RL bank funnel from the report, which the pipeline reproduces stage by stage:

```
ingest      91.0k   parse (layout-aware OCR output), license allow-list, schema
dedup       82.0k   MinHash char-5-gram, Jaccard ≥ 0.8, within + across banks
decontam    79.5k   −3.1%: 13-gram / MinHash ≥ 0.5 / embedding+LLM / answer-level
verify      74.1k   −6.8%: solver + 2 model families × 8 must agree with the label
difficulty  60.0k   drop 8/8 at medium; 7.0k problems at 0/8 go to the hard pool
```

Every stage writes `stages/NN_<stage>.jsonl` plus a manifest, `removed/<stage>.jsonl`
(uid + machine-readable reason for every removal) and one row of `funnel.json`. The
published files carry the **whole funnel** in their manifest's `stage_stats`. "Why
are there 60k problems and not 61k?" can always be answered from files, never
from memory.

## 1. Running it

### 1.1 Offline toy run (laptop, no GPU, about 1 s)

```bash
python -m dpt.data.pipeline --offline \
    --stages ingest,quality,dedup,decontam,verify,difficulty,publish,sft,probe \
    data_pipeline.output_dir=/tmp/dpt_toy
```

`--offline` replaces every model call with the deterministic fakes in
`dpt/data/backends.py`. The toy inputs in `examples/toy_data/` contain one planted
defect per stage, so the toy funnel shows each stage doing its job:

```
ingest      25 → 24   license: 1          (a "proprietary" source)
quality     24 → 23   repetition_4gram: 1 (an OCR loop)
dedup       23 → 22   near_duplicate: 1   (one word changed + punctuation)
decontam    22 → 18   ngram_minhash: 2 (verbatim eval copy, public-benchmark copy),
                      semantic: 1 (eval item reworded with a new number),
                      answer_level: 1 (eval parameters + answer, fully reworded)
verify      18 → 16   suspect_label: 1 (label 140 km, physics says 100 km), no_solver: 1
difficulty  16 → 12   all_correct: 4 (fake policy; the split is meaningless)
```

`tests/data/test_pipeline.py` asserts exactly this funnel, so a regression in any
stage fails CI. The offline numbers say nothing about model quality.

### 1.2 Real run

1. **Serve the base model** for difficulty calibration. On the workstation
   (4×A100-40GB) the 27B model fits for inference with TP4: about 13.5 GB of
   weights per GPU, leaving about 20 GB per GPU for KV cache and DeltaNet state.
   ```bash
   vllm serve Qwen/Qwen3.8-27B --tensor-parallel-size 4 --max-model-len 32768 \
        --gpu-memory-utilization 0.88 --port 8000
   ```
   Do **not** enable prefix caching here unless you also understand Incident 3:
   for this hybrid model the cached prefix includes DeltaNet state snapshots.
2. **Point the other backends** in `configs/data/pipeline.yaml` at real services:
   `verify.families` (two strong models from *different* families),
   `backends.embed` and `backends.judge`.
3. **Run as the eval-authorized account** for the `decontam` and `sft` stages,
   which read eval sets (see [contamination.md §5](contamination.md)). The other
   stages can run as the training account.
4. Run everything, or resume from a stage:
   ```bash
   python -m dpt.data.pipeline --config data/pipeline.yaml
   python -m dpt.data.pipeline --config data/pipeline.yaml --stages verify,difficulty,publish
   ```

Where to run: the workstation is enough for the whole data pipeline. The
expensive part is difficulty calibration: about 74k problems × 8 samples × about
6k tokens, roughly 3.5B generated tokens. A single TP4 replica of the 27B on
A100-40GB decodes this in days, not hours. To shorten it, run calibration on the
cluster *before* SFT starts (16 GPUs as 8 TP2 replicas, the same layout as RL
rollouts) or shard the bank across both machines. The pipeline itself does not
care where the vLLM servers run.

### 1.3 Output layout

```
<output_dir>/                         default: data/<version>/
  resolved_config.json                exact config + its hash
  funnel.json                         one row per stage (in, out, reasons, extras)
  stages/NN_<stage>.jsonl(+manifest)  stage outputs; each manifest names its parent's hash
  removed/<stage>.jsonl               every removal with a reason (no eval text)
  expert_queue.jsonl                  verification disagreements, suspect labels first
  rl_main.{jsonl,parquet}(+manifest)  verl rows, main pool
  rl_hard.{jsonl,parquet}(+manifest)  verl rows, hard pool (curriculum adds it after step 150)
  reports/contamination_report.json   merged decontam / canary / probe summary
  sft/sft_train.jsonl(+manifest)      final SFT mixture with n_tokens / bucket per sample
  sft/sft_report.json                 SFT funnel, token mixture, padding statistics
  secure_audit/semantic_audit.jsonl   human-audit sample (contains eval text → move to secure storage)
```

Parquet requires `pyarrow` (`pip install -e .[data]`). Without it only JSONL is
written and a warning says so. verl reads parquet.

## 2. Stages

### 2.1 `ingest`: parse, license, schema

* **What.** Reads candidate problems produced by the layout-aware OCR/extraction
  service (`PreParsedJSONLParser`). It canonicalizes LaTeX delimiters
  (`\(..\)` → `$..$`), applies a **license allow-list**, validates the schema
  (`dpt.data.schema.Problem`), assigns stable ids and drops exact duplicates.
* **Why no OCR inside the pipeline.** OCR is a GPU model with its own failure
  modes. Running it inline would make a CPU pipeline non-deterministic and
  slow to re-run. Run OCR once, then ingest its JSONL. `ExternalOCRParser`
  exists only to raise a helpful error.
* **Why an allow-list, not a block-list.** A missing or unknown license means
  drop. Fail closed: a problem we were not allowed to train on cannot be
  removed from a released model.
* **Stable ids.** `uid = "p-" + sha256(normalize_text(question))[:16]`.
  Re-ingesting a source reproduces ids, and trivially identical questions
  collapse before MinHash runs.
* **Failure modes.** OCR that garbles formulas still passes schema checks. The
  quality stage and answer verification are what catch such problems later. If
  the license field is filled inconsistently upstream, many good problems get
  dropped: always read `reasons.license` in the funnel.

### 2.2 `quality`: repetition and garbage

* **Rule.** Drop if > 30% of 4-gram occurrences repeat an earlier 4-gram. Also
  drop texts that are too short or too long, or have > 1% U+FFFD (undecodable
  bytes).
* **Why.** In the report, an SFT loss spike near step 400 came from a few dozen
  OCR-failure samples that repeated lines. The same filter also runs on SFT
  assistant turns.
* **Why 0.30.** Normal prose and long CoT score 0.02–0.15. Code scores higher,
  but stays well under 0.3. OCR loops score 0.6–0.95. The threshold sits in a
  wide empty gap, so its exact value hardly matters. Re-check the histogram
  whenever a new source is added.
* **Short texts** (fewer than about 50 words) skip the repetition test. The
  ratio is noise at that length.

### 2.3 `dedup`: MinHash LSH, Jaccard ≥ 0.8

* **Shingles.** Character 5-grams of the normalized text with spaces removed.
  They are robust to OCR word-splitting and LaTeX spacing.
* **Hashing.** blake2b-32 per shingle, then Dietzfelbinger's multiply-add-shift
  hash `((a·x + b) mod 2^64) >> 32` for each of 128 permutations. This is exact
  in numpy uint64, so it is bit-identical across machines. A unit test
  (`test_estimates_are_unbiased_across_many_pairs`) guards estimate quality.
  An earlier draft used `(a·x+b) mod p` with small `a`. That is nearly monotone
  in `x`, its "permutations" were correlated, and pairs with true Jaccard 0.79
  came out at 0.31.
* **LSH bands** are chosen by minimizing weighted FP/FN area, with FN weight
  0.98, because every candidate pair is then verified with the **exact**
  Jaccard of the hashed shingle sets. A false positive costs one set
  intersection. A false negative is a duplicate that survives. The result is
  17 bands × 7 rows at 0.8, about 98% recall at the threshold. Equal weights
  (9 × 13) missed about 30% of pairs at J = 0.84.
* **Within and across banks.** Cross-bank dedup runs first, against
  `inputs.reference_banks`, which have priority. Within-bank clusters are
  transitive (union-find). One representative is kept per cluster: the first
  in input order, so order the inputs by source quality.
* **Why 0.8 here but 0.5 in decontam.** The costs are asymmetric. A missed
  duplicate slightly over-weights one problem. A missed contamination
  invalidates an eval number.

### 2.4 `decontam`: keep eval content out

The full treatment is in [contamination.md](contamination.md). In short:
13-gram overlap or MinHash ≥ 0.5, then embedding cosine ≥ 0.85 confirmed by an
LLM judge (a reworded problem with changed numbers counts as the same problem),
then an answer-level match for numeric problems. Canary strings are scanned
first. Outputs on the training side contain **only ids**.

### 2.5 `verify`: is the label right?

For numeric problems (T1 and T4-math):

1. **Programmatic solver** (`meta.generator` → `dpt.data.solvers`) recomputes the answer.
2. **Two model families × 8 samples.** A family votes only if ≥ 5/8 of its
   samples agree (`min_family_agreement: 0.625`). One lucky or unlucky sample
   cannot swing the outcome.
3. **All sources must match the label** within the problem's tolerance, after
   unit conversion. Otherwise the problem is rejected **and queued for an
   expert**, never silently relabeled. In the report, about 40% of rejections
   were real label errors and about 60% were ambiguous wording.
4. **"Everyone against the label."** If both families and the solver agree on a
   value that is not the label, the label is the prime suspect. Such items get
   `reason=suspect_label` and `priority=high` and go to the top of
   `expert_queue.jsonl`.
5. **Perturbation consistency** for generated problems. Each input is nudged by
   ±1%:
   * if `scaling` exponents are declared (e.g. travel time ∝ distance¹ · velocity⁻¹),
     the measured elasticity `ln(f(p(1±δ))/f(p)) / ln(1±δ)` must equal the exponent.
     Exponents are compared rather than ratios, because at δ = 1% a ratio check
     cannot tell `x` from `x²` (1.0100 vs 1.0201);
   * otherwise the answer must change, and the forward and backward slopes must
     agree within 10%. If the answer does not change, or a kink shows up, the
     template or solver has a bug;
   * parameters listed as `invariant` (e.g. a station name) must not change the answer.

`require_solver: {t1_num: true}`: every T1 problem must have a solver, as in the
report. A T1 without one is a pipeline error (`no_solver`) and goes to the queue.

Non-numeric tasks return `not_applicable` here. T2 hidden tests are validated on
the sandbox cluster: the reference solution must pass and trivial or mutant
solutions must fail. T3 rubrics are reviewed by experts. T4-ifeval constraints
are executable rules.

The answer parser here (`answer_verify.last_boxed` / `parse_quantity` and a small SI
table) is **deliberately independent** of the training graders in `dpt.rewards`.
Verifying labels with the parser that will later grade the policy would hide bugs
in both places.

### 2.6 `difficulty`: drop what GRPO cannot learn from

* Sample the **base** model 8× per problem at **medium** effort, temperature
  1.0, top-p 1.0, with the same prompts (`dpt.data.prompts`) and the **training
  grader** (`dpt.rewards.router.compute_score`). These are the RL rollout
  settings. With any other settings, "8/8" would not mean "zero advantage
  during RL".
* **8/8 → drop.** The GRPO advantage `r − mean(r_group)` is 0 for every
  sample. Such a prompt costs a rollout and a forward/backward pass and yields
  no gradient.
* **0/8 → hard pool** (`meta.pool = "hard"`). It is also zero-advantage today,
  but becomes learnable once the policy improves. The curriculum adds it after
  RL step 150.
* T3 rewards are continuous. "Correct" means score ≥ 0.9, so only items that
  are uniformly near-perfect are dropped.
* `meta.pass_at_8_medium` is stored on every kept problem. The curriculum and
  the analysis notebooks use it.

### 2.7 `publish`: verl rows

`rl_build.problem_to_verl_row` produces:

```json
{"data_source": "t1_num", "ability": "t1_num",
 "prompt": [{"role": "system", ...}, {"role": "user", ...}],
 "reward_model": {"style": "rule", "ground_truth": "{\"value\": 20.0, ...}"},
 "extra_info": {"uid", "task", "pool", "effort", "reference_length", "pass_at_8_medium",
                "prompt_version", "index"}}
```

* `ground_truth` is a JSON **string**, which keeps the parquet schema uniform across tasks.
* T2's ground truth carries only `hidden_tests_id`. Hidden tests never enter
  the training data or the policy's context.
* `extra_info` has identical keys in every row, so parquet schema inference is stable.
* A final canary scan runs over the published rows. A hit aborts the publish.

## 3. Prompts: the shared contract

`dpt.data.prompts.build_messages(task, question, extra=...)` is used by RL export
**and** the eval harness, so training and evaluation prompts are byte-identical.

| task | final-answer format required by the system prompt |
|---|---|
| t1_num, t4-math | exactly one `\boxed{<number> <unit>}`, no ranges, no multiple candidates (reward hacks at steps 95/140) |
| t2_code | complete solution in the **last** ```python block |
| t3_report | ends with a `References` section, one source per line with DOI or full title |
| t4-ifeval | no system prompt: the instruction alone, as IF benchmarks are scored |

`PROMPT_VERSION` is recorded in every row. Changing prompt text requires bumping it.

## 4. SFT mixture (`sft` stage)

Order: admission → quality → decontam (including teacher recitation) →
render/tokenize → mixture → buckets.

* **Admission rules** are applied as funnel stages, not as schema errors, so
  the counts are visible:
  `domain_reasoning` needs `meta.verified` (the answer passed the verifier),
  `tool_traj` needs `exec_ok` **and** `hidden_tests_passed`,
  `self_distill` needs `generated_by == <base model>`,
  `report` needs a `rubric`.
* **Rendering.** Every sample is rendered with the model's own chat template
  and an **explicit** `reasoning_effort`, then verified by
  `dpt.chat.render.check_effort_rendering`. Qwen3.8's template inserts the
  *xhigh* sentence when the kwarg is missing and *no* sentence for medium. The
  report's "model stopped respecting effort levels" incident came from exactly
  this. Samples that fail the check are dropped with reason `render_effort`.
* **Loss mask.** Assistant turns only, including `<think>` and `<|im_end|>`.
  Tool outputs are rendered in user turns and are therefore excluded.
* **No truncation.** Samples longer than 32K are dropped. A truncated sample
  teaches the model to stop mid-thought.
* **25% replay, by tokens.** Self-distilled samples are subsampled (seeded, no
  upsampling) so that replay / total ≈ 0.25 in tokens. Tokens are the unit
  because the loss is per token. In the report's ablation, SFT without replay
  cost −3.2 on general benchmarks and 25% replay cost −1.1.
* **Buckets 4K/8K/16K/32K.** Megatron's DeltaNet path has no packed-sequence
  support, so micro-batches are padded. `padding_stats` reports the padding
  overhead with bucketing (sorted within bucket) against a shuffled stream,
  with a < 8% target. The toy run warns because character-level toy samples
  are tiny next to the 128-token pad multiple. That warning is expected
  offline.

## 5. Extending

| to add… | do this |
|---|---|
| a new raw source | make the OCR service emit candidate JSONL; add the path to `inputs.rl_candidates`; check `license` values |
| a new parser | implement `parse(path) -> Iterable[dict]`, register in `ingest.PARSERS` |
| a programmatic problem family | `@register_solver("name")` in `dpt/data/solvers.py`; emit `meta.generator` with `params` (+ `scaling` if the physics is a power law) |
| an eval set / protected benchmark | add to `inputs.eval_sets` / `inputs.public_benchmarks`; embed a canary from `make_canary("<set>/<version>")` |
| a threshold change | edit `configs/data/pipeline.yaml` **and bump `data_pipeline.version`** |

## 6. Checklist before a real run

- [ ] `inputs.eval_sets` point at the secure copies; the decontam account can read them, the training account cannot.
- [ ] Canary strings in the config match those embedded in the eval files.
- [ ] Difficulty calibration uses the same effort, temperature, top-p, max tokens and grader as RL rollouts.
- [ ] Verification families are two *different* model families, neither of which is the base.
- [ ] After `decontam`, a human audits `secure_audit/semantic_audit.jsonl` (≥ 200 items). Record the confirmation rate (report: 87%).
- [ ] After `verify`, experts clear `expert_queue.jsonl`, high priority first. Fixed items re-enter through `ingest`.
- [ ] `pyarrow` is installed so that `rl_*.parquet` exist.
- [ ] Training configs reference the published files' **content hashes** from their manifests.
