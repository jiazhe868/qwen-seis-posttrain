# Code conventions and module contracts

This file is the contract between modules. If you change an interface listed
here, update this file in the same commit.

## 1. Layering rule (the most important rule in this repo)

```
            ┌──────────────────────────────────────────────────────────┐
  adapters  │ dpt.verl_ext      dpt.parity      scripts/*.sh            │  need GPUs / verl / vLLM / Megatron
            └───────────────▲──────────────────────────────────────────┘
                            │ imports (one direction only)
            ┌───────────────┴──────────────────────────────────────────┐
  core      │ dpt.common  dpt.chat  dpt.data  dpt.rewards  dpt.rl       │  pure Python + numpy (+ torch on CPU)
            │ dpt.monitoring  dpt.eval  dpt.planning  dpt.launch        │  unit-tested on CPU, no GPU, no verl
            └──────────────────────────────────────────────────────────┘
```

* **Core modules never import verl, vLLM, Megatron, Ray or CUDA-only code.**
  Optional heavy imports (pandas, pyarrow, transformers) happen *inside* the
  function that needs them, so `import dpt.x` always succeeds on a laptop.
* Adapters are thin: they translate between framework objects (verl
  `DataProto`, vLLM outputs) and the plain arrays/dicts the core functions take.
  All the logic that matters for correctness lives in core and is tested there.
* Why: verl's internal APIs change every minor release (0.9 already deprecated
  the classic trainer). Keeping the algorithm in framework-free code means a
  verl upgrade touches one adapter package, and every rule from the project
  report (fail-closed grading, fixed-constant loss normalization, TIS, the
  curriculum, alert thresholds) stays covered by fast tests.

## 2. Style

* Python ≥ 3.10, type hints on all public functions, `from __future__ import annotations`.
* Every module starts with a docstring that says **what** it does, **why** it
  exists (cite the report section / incident / ablation id, e.g. "Incident 3",
  "Ablation A7"), and **how** it is used.
* Comments explain *why*, not *what*. Numbers that come from the report are
  never hard-coded in logic; they live in `configs/` and are passed in.
* Pure functions where possible. Stateful components (curriculum sampler,
  running statistics, partial-rollout buffer, alert manager) expose
  `state_dict() -> dict` / `load_state_dict(dict)` with JSON-serializable
  content, so checkpoint/resume is exact (Incident 2).
* Fail closed: any grader/parsing error returns score 0 with a machine-readable
  `reason`, never raises into the training loop and never returns partial credit.
* Logging via `dpt.common.logging.get_logger(__name__)`; no bare `print` in library code.
* CLI entry points use `argparse` and live in `main()` functions guarded by
  `if __name__ == "__main__":`, runnable as `python -m dpt.<pkg>.<module>`.

## 3. Identifiers shared across modules

| Name | Values | Defined in |
|---|---|---|
| task id | `t1_num`, `t2_code`, `t3_report`, `t4_general` | `dpt.common.types.TASKS` |
| reasoning effort | `low`, `medium`, `xhigh` | `dpt.common.types.EFFORTS` |
| eval subset | `d_num`, `d_code`, `d_report`, `d_fresh` | `dpt.common.types.EVAL_SETS` |
| thinking markers | `<think>`, `</think>` | `dpt.chat.thinking` |

## 4. Record schemas (JSONL, one object per line)

### 4.1 RL problem (`dpt.data.schema.Problem`)

```json
{
  "uid": "p-3f2a…",                 // sha256 of normalized question, first 16 hex
  "task": "t1_num",
  "question": "…",                 // user-visible prompt body
  "answer": {…},                   // task-specific ground truth, see below
  "source": {"origin": "…", "license": "CC-BY-4.0", "doc_id": "…", "created_at": "2026-05-01"},
  "meta": {"pass_at_8_medium": 0.25, "pool": "main" | "hard", "flags": [],
           "params": {…},            // input parameters of a numeric problem (answer-level decontam)
           "generator": {"name", "params", "scaling", "invariant"},   // programmatic problems
           "verified": {…}}          // 3-way verification record
}
```

Task-specific `answer` payloads:

| task | `answer` fields |
|---|---|
| `t1_num` | `value: float`, `unit: str` (may be ""), `rel_tol: float`, `abs_tol: float` |
| `t2_code` | `entry_point: str`, `visible_tests: str`, `hidden_tests_id: str` (hidden tests live only on the sandbox cluster), `language: "python"` |
| `t3_report` | `rubric: list[{criterion, weight}]`, `reference_length: int` (tokens) |
| `t4_general` | `kind: "math" \| "ifeval"`, plus `value`/`unit` for math or `constraints: list[dict]` for ifeval |

### 4.2 SFT sample (`dpt.data.schema.SFTSample`)

```json
{"uid": "…", "subset": "domain_reasoning|tool_traj|self_distill|report",
 "messages": [{"role": "system|user|assistant|tool", "content": "…", "reasoning_content": "…"}],
 "reasoning_effort": "medium", "tools": null, "source": {…}}
```

### 4.3 Reward result (`dpt.rewards.base.RewardResult`)

```python
RewardResult(score: float, task: str, reason: str = "ok",
             correct: bool | None = None,
             components: dict[str, float] = {}, flags: list[str] = [],
             truncated: bool = False)
```
`to_verl_dict()` returns `{"score": score, "acc": float(correct), "reason": …, "truncated": …, **components}`.

### 4.4 Final-answer formats (shared by prompts, training graders and eval graders)

| task | the model must end its post-`</think>` answer with |
|---|---|
| `t1_num`, `t4_general` (math) | exactly one `\boxed{<number> <unit>}`; several distinct boxes, ranges or ± → 0 |
| `t2_code` | the full solution in the **last** ```` ```python ```` block |
| `t3_report` | the report, with a trailing `References` section (citations verified separately) |
| `t4_general` (ifeval) | free text satisfying the stated constraints |

`dpt.data.prompts.build_messages` is the only place that writes these instructions;
RL export and the eval harness both call it.

### 4.5 verl row (`dpt.data.rl_build`)

`data_source` = task id, `prompt` = messages, `reward_model.ground_truth` = JSON string of
`Problem.answer`, `extra_info` = `{uid, task, pool, split, effort, reference_length,
pass_at_8_medium, prompt_version, index, question}` (identical keys in every row).

### 4.6 Metric names

`dpt.monitoring.keys` is the contract between the trainer and the alert rules;
`dpt.monitoring.bridge.to_canonical` maps verl / `dpt.rl` names onto it.

## 5. Tests

* Mirror the package path: `src/dpt/rl/advantage.py` → `tests/rl/test_advantage.py`.
* CPU-only, < 1 s each unless marked `slow`. GPU tests are marked `gpu` and
  skip automatically. Network tests are marked `network` and are skipped unless
  selected with `-m network`.
* Every incident in the runbook has at least one regression test whose
  docstring names the incident.
