# Evaluation

> Evaluation counts only three things: a **fixed harness**, a **quantified noise floor**, and
> **independently verified answers**. A difference between two results counts as an improvement
> only if it exceeds the 95% confidence interval. Checkpoints are selected on dev only, never on test.
> (Report: evaluation system.)

Code lives in `src/dpt/eval/` and the harness settings in `configs/eval/eval_suite.yaml`. Every
protocol number (temperature, effort, lengths, sample counts, gates) is interpolated from
`configs/project.yaml`, so it is defined in exactly one place.

---

## 1. What is measured

| Set | Items | Metric | Samples | Grader (`kind`) | 95% CI (report) |
|---|---|---|---|---|---|
| D-Num | 600 | avg@8 | 8 | `numeric` (sympy + own unit table) | ±1.6 |
| D-Code | 300 | pass@1, averaged over 4 samples; hidden tests decide | 4 | `code` (sandbox `HiddenTestClient`) | ±2.4 |
| D-Report | 200 | win rate vs base; judge in both orders; 30% expert blind review | 1 | `pairwise` | ±4 pp |
| D-Fresh | 150 | avg@8 (items written after 2026-08-14) | 8 | `numeric` | ±3.1 |
| General ×5 | official sizes | GPQA-Diamond, MMLU-Pro (2k subset), IFEval, LiveCodeBench (post-2026-08), tool calling | 1–4 | `choice`, `ifeval`, `code`, `toolcall` | per benchmark |
| Refusal regression | fixed set | refusal rate (%) and over-refusal on benign controls | 1 | `refusal` | Wilson |
| D-Num perturbed | 200 | avg@8 on numerically perturbed twins | 8 | `numeric` | — |

**Two D-Report judges (optional, `eval_suite.second_judge`; on by default on the 9B line).** A
second judge of another family judges the same pairs in both orders. The primary judge is also the T3
reward, so RL optimizes against it; the second never touched training. `results.json` keeps both win
rates (`by_judge`), their agreement and each judge's unparsed-verdict rate; `pairwise.gate_judge` picks
what the gates read (`primary`, `second`, or `min`). A judge whose verdicts are missing, or more than 5%
unparsed (they would count as ties and pull the rate toward 50%), leaves the score empty, so the gate
fails closed. Verdicts are cached per judge (`d_report/judgments.<role>.jsonl`), which lets the second
judge run later on another machine (`--regrade-only --update`, `scripts/slurm/second_judge.sh`).

On the 9B plan no expert writes or edits eval items (README §3.1). GPT-6 Luna drafts
D-Fresh and D-Report (`dpt.eval.drafts`); `dpt.eval.autocheck` keeps a D-Fresh problem only if its drafted
`solve()` reproduces the stated answer in the sandbox and Gemma confirms it in at least one of four
independent solutions; when another answer gets as many votes, or none match, Gemma compares the two
worked solutions in both orders and must uphold the stated answer both times. It keeps a D-Report question
only if a majority of three votes, each one Gemma answering the question first and then ruling on the
rubric, accepts the question and its essential rubric criteria. Gemma thinks up to ~28K tokens per reply. The human check that remains is a blind review of 20 D-Report pairs (10 more on
disagreement, `dpt.eval.blind_review`): the page is offline, the key never leaves the eval owner, and
`import` records the reviewer's win rate next to each judge's on the same pairs.

Sampling for every model: temperature 1.0, top-p 1.0, reasoning effort `medium`, at most 32K output
tokens. Base is also reported at `xhigh`. At 32K, base@xhigh truncates 11% of answers, so it is
re-scored at 64K and the higher score is reported (`--extra-max-tokens 65536`; both numbers are kept
in `score_by_max_tokens`).

MMLU-Pro is `relative_only`. The base-model continuation probe matched 6.2% of items, against a 0.3%
random baseline, so only its delta is meaningful. The report table prints that note automatically.

---

## 2. Harness rules (and where each is enforced)

| Rule | Where |
|---|---|
| Every model, base included, is rerun in the same harness. No public number is copied. | `harness.run_suite` is the only path to a `results.json` |
| Fixed chat template, effort and max length | `mode: completions`: the harness renders the prompt itself with `dpt.chat.render.render_chat`, which verifies the effort sentence, then sends raw text |
| Same prompt format as RL | `harness.build_messages` → `dpt.data.prompts.build_messages` (the builder RL export uses) |
| Only the final answer is graded; thinking is stripped | `harness.grade_sample` → `dpt.chat.thinking.split_thinking` |
| No `</think>` means the sample fails; raw thinking is never graded | `grade_sample` returns `no_think_end` |
| Truncation rate is always reported | `finish_reason == "length"` → `truncation_rate` per set |
| Reproducible sampling | per-item seed = `derive_seed(run_seed, set, item_id)` |
| Eval data cannot drift silently | the `.manifest.json` sidecar is verified before scoring; content hashes are written into `results.json` |
| Train and eval graders are separate code | `dpt.eval` never imports `dpt.rewards` (enforced by `tests/eval/test_independence.py`) |

`mode: chat` exists for models served only through a chat API. It requires a server **without**
`--reasoning-parser`. If the server splits off reasoning anyway, the harness rebuilds the `</think>`
marker only for samples with `finish_reason == "stop"` and a non-empty answer, so a truncated sample
still fails.

---

## 3. Noise floor and confidence intervals

There are two kinds of uncertainty, and `dpt.eval.stats` reports both:

1. **Generation noise on a fixed item set.** This is what the report's `±` values mean. Ten reruns
   of the same checkpoint on D-Num gave a mean pairwise difference of 0.9 and a maximum of 2.1,
   i.e. a standard deviation of about 0.8 and a 95% half-width of about 1.6. `sampling_ci`
   estimates this from a single run by resampling the k samples *within* each item.
   `noise_floor([...])` measures it directly from repeated runs. **Gates and the `±` columns use
   this interval.**
2. **Item sampling** (`bootstrap_ci` over items, stored as `item_ci`). This interval answers "would
   this generalize to new items of the same kind". It is roughly twice as wide on heterogeneous sets.

For model-vs-model claims on the same items, use `paired_bootstrap_diff(a_per_item, b_per_item)`. It
resamples the same items for both models, so shared item difficulty cancels out.

How to measure the noise floor (do this once, before any comparison):

```bash
for s in 0 1 2 3 4 5 6 7 8 9; do
  python -m dpt.eval.harness --model-name sft_noise_s$s --sets d_num --seed $s \
      --endpoint http://localhost:8000 --out eval_results/noise/sft_s$s
done
python - <<'EOF'
import json; from dpt.eval.stats import noise_floor
s=[json.load(open(f"eval_results/noise/sft_s{i}/results.json"))["sets"]["d_num"]["score"] for i in range(10)]
print(noise_floor(s))
EOF
```

---

## 4. Answer verification: three layers

### Layer 1: is the gold answer right? (data pipeline, `dpt.data.answer_verify`)
Three independent solvers: a programmatic solver plus two strong models from different families,
8 samples each. An item is kept only if all three agree. Disagreements go to an expert queue. About
40% of rejected items had a wrong original label; the rest were ambiguous. If every model agrees on
an answer that differs from the label, suspect the label first. Programmatic items also get a
perturbation consistency check: changing an input must change the answer according to the physics.

### Layer 2: does the grader judge correctly? (this package)
* **Adversarial grader unit tests**: `tests/eval/test_numeric_grader.py` covers equivalent forms
  (1/2 vs 0.5, `\frac`, scientific notation), units (km vs m, compound units, offset temperatures),
  multiple candidates, ranges, uncertainties, inequalities, and answers buried in long text. Extend
  it toward the report's 1,200 cases whenever a new failure mode appears.
* **Human calibration**: label 500 model outputs and compute FP/FN. The report's grader went from
  FP 2.8% / FN 4.1% in v1 to 0.3% / 1.2% in v3. At 4.1% FN, grader noise alone can hide or
  fabricate an improvement.
* **During training**: `dpt.monitoring.audit.AuditSampler.grader_spotcheck` exports 50 reward=1 and
  50 reward=0 samples every 20 steps, and `score_spotcheck` turns the human labels into FP/FN.
* After a grader fix, re-score without regenerating: `--regrade-only`.

### Layer 3: does "correct" mean "really knows"? (`dpt.eval.consistency`)
* **Reasoning-answer consistency**: the last value derived in the thinking must match the submitted
  value (powers of ten allowed for unit conversion). Track the "correct but inconsistent" rate; it
  measures guessing and lucky formatting.
* **Perturbation gap**: compare D-Num original vs perturbed twins. Report: final 71.4 vs 68.9
  (gap 2.5), base gap 2.1, so no meaningful widening. `perturbation_gap(..., base_gap=2.1,
  max_widening=<CI>)`.
* **pass@k floor**: pass@64 after RL must be ≥ base (report 91.0 vs 90.2). Run d_num with
  `--samples-override 64` for both rows; `results.json` then holds the full `pass_at_k` curve.
* **D-Fresh is the final judge**: it was written after the base release, so no training data can
  contain it.

---

## 5. Gates

`python -m dpt.eval.gates` applies `project.gates` and `project.rl_entry_gate` mechanically. It
exits 0 if all gates pass, 1 if any fails, and 2 if an input is missing.

| Gate | Rule as implemented | Report outcome |
|---|---|---|
| D-Num | `cand − base_xhigh ≥ 8` **and** gain > combined CI | 71.4 vs 55.3 |
| D-Code | `cand − base_xhigh ≥ 10` **and** gain > combined CI | 55.2 vs 40.2 |
| D-Report | win rate ≥ 60 **and** lower CI bound > 50 | 68% |
| D-Fresh | `(Δ D-Fresh) / (Δ D-Num) ≥ 0.60` vs the configured ref | 81% vs base_medium (83% vs base_xhigh) |
| General | mean Δ over 5 benchmarks vs base_medium ≥ −1.0 | −0.4 |
| Safety | \|Δ refusal rate\| ≤ 2 pp | +0.7 |
| Cost | mean output tokens ≤ 1.3 × base_medium | 1.08× |
| Repro | \|D-Num(seed1) − D-Num(seed2)\| ≤ 2.0 | 1.3 |

RL-entry gate (`--entry --candidate sft`): all three domain sets strictly above base_medium; general
mean Δ ≥ −1.5; D-Num pass@16 ≥ base. The last condition preserves exploration headroom; the report
had 83.0 vs 81.7.

---

## 6. Running it

### Serve the model (vLLM, OpenAI-compatible)

```bash
# Cluster: one 8-GPU node can host 4 TP2 replicas behind a load balancer; a single replica:
vllm serve /ckpt/rl_seed1_hf --served-model-name rl_seed1 --tensor-parallel-size 2 \
    --max-model-len 36864 --language-model-only --port 8000
# Workstation (4×A100-40GB): the 27B model needs TP4 (≈13.5 GB of weights per GPU, ~20 GB left for KV)
vllm serve Qwen/Qwen3.8-27B --served-model-name base --tensor-parallel-size 4 \
    --max-model-len 32768 --gpu-memory-utilization 0.88 --language-model-only --port 8000
```

Do **not** pass `--reasoning-parser` when `mode: chat` is used. In the default completions mode it
does not matter.

### Evaluate, summarize, gate

```bash
# 1) base at medium (its D-Report is 50% by construction: reference = itself)
python -m dpt.eval.harness --model-name base_medium --served-model base --effort medium \
    --sets d_num,d_code,d_report,d_fresh,general,refusal --reference-run eval_results/base_medium \
    --out eval_results/base_medium
# 2) base at xhigh, also at 64K, report the higher
python -m dpt.eval.harness --model-name base_xhigh --served-model base --effort xhigh \
    --extra-max-tokens 65536 --sets d_num,d_code,d_report,d_fresh,general,refusal \
    --reference-run eval_results/base_medium --out eval_results/base_xhigh
# 3) candidates (sft, rl_seed1, rl_seed2): same command with their served names
# 4) table + gates
python -m dpt.eval.report --row base_medium=eval_results/base_medium --row base_xhigh=eval_results/base_xhigh \
    --row sft=eval_results/sft --row rl_seed1=eval_results/rl_seed1 --row rl_seed2=eval_results/rl_seed2 \
    --summary-out eval_results/summary.json --markdown-out eval_results/report.md
python -m dpt.eval.gates --summary eval_results/summary.json --candidate rl_seed1 --seed2 rl_seed2
```

Smoke test on the workstation: add `--limit 10 --no-verify` and point the set paths at
`examples/toy_data` with a dotlist override, e.g.
`eval_suite.sets.d_num.path=examples/toy_data/eval/d_num.jsonl` (toy files for all four D-sets live in `examples/toy_data/eval/`).

### Cost guide

D-Num avg@8 is 4,800 generations, at about 6K tokens each for SFT/RL models (≈15K for base@xhigh). On
the workstation (one TP4 replica) the full domain suite for one model takes hours. On the cluster,
use 8 TP2 replicas behind a round-robin proxy and raise `max_concurrency` to roughly 100 per
replica.

---

## 7. `results.json` schema (per run)

```jsonc
{
  "model_name": "rl_seed1", "effort": "medium", "max_tokens": 32768, "mode": "completions", "seed": 0,
  "config_hash": "…", "data_hashes": {"d_num": "<manifest content hash>", …},
  "mean_output_tokens": 6400.0,                // sample-weighted over the four domain sets
  "sets": {
    "d_num": {"metric": "avg@8", "score": 71.4, "ci": {"lo":…, "hi":…, "halfwidth": 1.6},
              "item_ci": {…}, "pass_at_k": {"1": 71.4, "2": …, "8": …}, "truncation_rate": 0.01,
              "mean_output_tokens": 5900, "reasons": {"ok": …, "wrong_value": …, "no_think_end": …},
              "per_item": [{"id": "…", "n_correct": 6, "mean": 0.75}, …],
              "score_by_max_tokens": {"32768": …, "65536": …}},          // only with --extra-max-tokens
    "d_report": {"metric": "winrate_vs_base", "score": 68.0, "outcomes": {"win": …, "tie": …, "loss": …},
                 "length_matched": {"n": …, "score": …}},
    "refusal": {"refusal_rate": 10.7, "over_refusal_rate": …}
  },
  "general": {"gpqa_diamond": {…}, "mmlu_pro_2k": {…, "relative_only": true}, …}
}
```

Hidden-test sandbox wire contract (`HiddenTestClient`): `POST {endpoint}/run` with
`{"code", "hidden_tests_id", "entry_point", "language", "timeout_s", "kind": "hidden"}` (header `X-Sandbox-Token` from `DPT_SANDBOX_TOKEN` when the worker requires it; the same protocol as `dpt.rewards.sandbox_server`) returns
`{"status": "ok"|"timeout"|"error", "passed": int, "total": int}`. The sandbox must have no network,
must replay tests outside the candidate's process, and must not expose environment variables that
identify a test run (the step-205 hack).
