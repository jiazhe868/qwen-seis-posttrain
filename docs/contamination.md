# Contamination control

Code: `src/dpt/data/decontam.py` (+ the `decontam`, `sft`, `probe` and `publish`
stages of `src/dpt/data/pipeline.py`) · Config: `data_pipeline.decontam` / `.probe` in
`configs/data/pipeline.yaml` · Tests: `tests/data/test_decontam.py`, `tests/data/test_pipeline.py`

An eval number is only worth something if the model has not seen the eval items.
This document covers what we check, why the thresholds are set where they are,
how eval data is isolated, and what to do when something is found.

---

## 1. Threat model: three ways eval content reaches a model

| direction | example | where it is checked |
|---|---|---|
| **Training data ↔ eval** | an RL candidate is a reworded D-Num item; an SFT prompt copies a public benchmark | layers 1–3 on RL candidates and SFT prompts |
| **Teacher leakage** | the strong model that wrote SFT CoT memorized an eval solution and recites it while answering a *different* prompt | layer 4 (teacher recitation) on SFT assistant turns |
| **Base-model memorization** | Qwen3.8-27B already saw MMLU-Pro in pre-training | memorization probes; affected benchmarks are reported as deltas only |

Two structural defenses sit underneath the checks:

* **Time split.** D-Fresh (150 items) was written by experts *after* the base
  release (2026-08-14), and only LiveCodeBench problems published after
  2026-08 are used. No training data, teacher or base model can contain them.
  D-Fresh is the final arbiter. The report's success criterion asks that the
  D-Fresh gain be ≥ 60% of the D-Num gain, and it came out at 81%.
* **Isolation.** Eval sets live in separately-permissioned storage
  (`project.paths.eval_secure_root`). Training processes run as an account
  that cannot read it (§5).

## 2. The layers

Layers run in order, and each removal is attributed to the **first** layer that
fires. That is how the report's table splits 1,190 (n-gram/MinHash) from 1,350
(embedding+LLM). Cheap layers run first, and embeddings are computed only for
the survivors.

### Layer 0: canaries (always first)

Each internal eval file embeds a unique string, `make_canary("<set>/<version>")`,
which looks like `DPT-CANARY-<uuid5>`. Every nested string of every candidate and
SFT record is scanned. A hit means an eval *file* leaked into a data source. The
item is removed, the run logs an error, and the leak path must be found. The
`publish` and `sft` stages re-scan their outputs and **abort** on any hit. The
same scan should run periodically over model outputs, because a model that
emits a canary has memorized eval files. Report result: 0 hits.

### Layer 1: `ngram_minhash`. Exact and near-exact overlap

* **13-gram overlap.** Any shared 13-gram of normalized word tokens (NFKC,
  lowercased, punctuation stripped; CJK characters are single tokens). Thirteen
  words of shared text is almost never coincidental, and this is the
  standard length used in LLM contamination studies. Texts shorter than 13
  tokens fall back to an exact match on the whole normalized sequence.
* **Boilerplate is not counted** (`ngram_max_df`, default 10). A 13-gram that
  occurs in more than 10 of the texts being checked is template or instruction
  text: a generator's formula sentence, IFEval's constraint wording, the
  code-task scaffold. A leaked eval item shows up in a few texts, not in
  hundreds. The first real bank run (2026-10-24) counted such n-grams and removed
  13.2% of the candidates: 85% of the T2 code tasks, every problem of five T1
  seismic-moment families, and 589 instruction prompts on IFEval's constraint
  phrases. With the rule it removed 3.3%: 1,399 by n-gram/MinHash and 441 by
  the semantic layer. The same rule applies to SFT prompts and to the
  teacher-recitation layer.
* **MinHash Jaccard ≥ 0.5** on character 5-grams (exact Jaccard verified after
  LSH). This catches light edits that break every 13-gram, for example a
  changed number in a short problem. With "3.5e19" changed to "4.1e19" the
  Jaccard is 0.79, but no 13-gram survives because every window spans the
  number.

### Layer 2: `semantic`. Paraphrases

* Embedding nearest neighbours (top-5) with cosine ≥ 0.85, then an **LLM judge**
  answers "same problem?". The judge prompt (`backends.LLMParaphraseJudge`)
  states that rewording, translation, or changing numbers/names/units while
  keeping the solution method still counts as the **same** problem.
* Neighbours the judge rejects are counted (`semantic_judge_rejected`). A
  sudden jump in that count usually means the embedding model changed.
* **Human audit.** A seeded sample of up to 200 semantic removals is written
  to `audit_dir` with both texts. An expert labels each one. Report result:
  87% confirmed. That is the layer's precision. Record it in
  `contamination_report.json → rl.semantic_audit.confirmed_rate`.

### Layer 3: `answer_level`. Same numbers, same result

For numeric problems, a candidate is contaminated if an eval item's input
parameters (a multiset of ≥ 2 values, compared at 4 significant digits) **all
appear** in the candidate and the answers match. Parameters come from
`meta.params` / `meta.generator.params` when declared, otherwise from the
numbers in the text. A problem can be reworded beyond any text similarity,
but if its inputs and its answer travel together it is the same item. Inclusion
(eval ⊆ candidate) rather than equality tolerates extra incidental numbers in
the candidate text, such as a station number.

### Layer 4: `teacher_verbatim` (SFT only)

The assistant text of an SFT sample (reasoning + answer) is checked for 13-grams
of eval **questions and reference solutions** that do **not** occur in the
sample's own prompt. Such n-grams mean the teacher recited eval content
unprompted. The removed record carries the sample's `source`, so the prompt
source can be investigated. The report found 46 such samples and traced
their prompt origins.

## 3. Why these thresholds: asymmetric costs

| error | cost |
|---|---|
| false positive (a clean item removed) | one fewer training problem out of tens of thousands |
| false negative (a contaminated item kept) | the eval conclusion is invalid; every number built on it is suspect |

All thresholds are therefore set for **recall**:

* MinHash uses 0.5 in decontam, against 0.8 in dedup.
* LSH banding is recall-weighted (FN weight 0.98, ≈98% recall at the threshold),
  and candidates are verified with exact Jaccard.
* The semantic layer accepts a precision of about 87% (13% collateral removals).

Reported scale: 3.1% of RL candidates and 1.4% of SFT candidates removed. If a
new source suddenly loses > 10%, suspect the *source* (for example a scraped
copy of a benchmark), not the thresholds.

## 4. Base-model memorization probes (`probe` stage)

`prefix_completion_probe`: give the base model the first half of each
benchmark item through the **raw** completion endpoint (no chat template,
greedy). A "match" means ≥ 80% of the next 20 normalized words agree
position by position. The **chance baseline** pairs each generation with a
*different* item's continuation, using a seeded derangement, which measures
accidental agreement without extra generations. A z-test flags significance
(`z > 3` and ≥ 1 pp above baseline).

Report results and their consequences:

| benchmark | match | baseline | consequence |
|---|---|---|---|
| MMLU-Pro subset | 6.2% | 0.3% | memorized: report **before/after deltas only**, never absolute scores (`project.eval.relative_only`) |
| GPQA-Diamond | 0.4% | ≈ baseline | no evidence of memorization: used normally |

`perturbation_gap` covers a second angle. It compares accuracy on original vs
numerically perturbed items with a paired bootstrap CI. A model that memorized
answers falls off on perturbed numbers. Report: final model 71.4 → 68.9 (gap
2.5) against base gap 2.1. The gap did not widen, so the RL gains are not
memorization. The function lives here so that the eval harness can apply it to the
200-item perturbed D-Num subset.

## 5. Access control and what leaves the secure side

* The `decontam` and `sft` stages read eval sets and therefore run as the
  **eval-authorized account**. Training jobs (SFT/RL) run as an account
  without read access to `eval_secure_root`.
* Training-side outputs (`removed/decontam.jsonl`, `stages/*`,
  `reports/contamination_*.json`, `sft/dropped.jsonl`) contain **only candidate
  uids, the ids of matched eval items (`"d_num/e-num-001"`), layer names and
  scores**, never eval text. A test asserts this.
* The semantic **audit sample** necessarily contains eval text. It is written
  to `decontam.audit_dir`, which must point at secure storage in production
  (the default under `output_dir` exists only for the offline toy run).

## 6. Reading `reports/contamination_report.json`

```json
{
  "rl":  {"scanned": 82000, "ngram_minhash": 1190, "semantic": 1350, "answer_level": 0,
          "semantic_judge_rejected": 412, "total_removed": 2540, "rate": 0.031,
          "semantic_audit": {"sampled": 200, "path": "...", "confirmed_rate": 0.87},
          "canary": {"scanned": 82000, "hits": 0, "details": []}},
  "sft": {"scanned": 98000, "prompt_side": 1324, "teacher_verbatim": 46, "total_removed": 1370,
          "rate": 0.014, "teacher_sources": ["..."], "canary": {"hits": 0, ...}},
  "memorization_probes": {"mmlu_pro_2k": {"match_rate": 0.062, "baseline_rate": 0.003, "z": ..., "significant": true}}
}
```

(The numbers shown are the report's. The structure is what the pipeline writes.
`answer_level` is attributed only after the first two layers, so it is often small.)

## 7. When something is found

1. **Canary hit.** Stop. Find which source file contains the eval file content,
   remove the source, and re-run from `ingest`. Scan model outputs from any
   checkpoint trained on that data.
2. **A source with an unusually high removal rate.** Inspect `removed/decontam.jsonl`
   grouped by `source.origin`. Scraped Q&A sites often mirror benchmarks.
   Prefer dropping the whole source.
3. **Teacher recitation.** Remove the samples (automatic), then check which
   *prompts* triggered it. They are usually near-copies of eval questions that
   slipped through prompt-side checks. Add them to the protected set.
4. **Eval set changed** (new items, new version). Bump the eval set version and
   its canary, and re-run `decontam` and `sft` over the **current** training
   data before comparing any numbers.

## 8. Known limitations

* The embedding layer is only as good as the embedding model. Cross-lingual
  paraphrases (e.g. an English eval item translated into Chinese training
  data) need a multilingual embedder.
* Answer-level matching skips problems whose parameters are not explicit
  numbers (symbolic answers, plots).
* Probes measure verbatim memorization. They cannot detect a model that learned
  the *solution* of a benchmark item without its wording. D-Fresh and the
  perturbation gap cover that case.
