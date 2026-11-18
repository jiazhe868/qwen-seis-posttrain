"""Run 1 against RL v2: what each run learned from, and what happened to its answers (2026-11-17).

Run 1 collapsed after step 20; RL v2 changed the bank, the rule for cut-off answers, the learning rate, the task mix
and the budget at once, and has no control run. No ablation fits in the time left, so this script puts the two runs
side by side on the evidence they already logged, and writes the numbers README §8.6 quotes:

- where the gradient came from: each task's share of it, and the share of numeric groups that carry any;
- the length of right numeric answers;
- both on one axis: the sum of the learning rates of all steps so far. With Adam, a step moves the weights by about
  its learning rate, so this sum says how far a run has pushed the policy. RL v2 runs at half run 1's rate, so its
  step 2n has gone about as far as run 1's step n;
- what happened to unfinished answers: run 1 on D-Num-hard at 32K (SFT → step 20), RL v2 on its hard dev split at
  24K (its step-0 dev pass is the SFT model).

    python scripts/talk/rl_compare.py        # reads docs/talk/data/{rl,rl_v2_seed1,numhard_64k,...}.json

Writes docs/talk/data/rl_runs_compared.json (counts and rates only).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "docs" / "talk" / "data"
TASKS = ("t1_num", "t2_code", "t3_report", "t4_general")
COLLAPSE_STEP_RUN1 = 40                  # dev 97.3 at step 20, 78.9 at step 40 (README §6.4)


def per_step(metrics: list[dict]) -> list[dict]:
    """The series both runs logged, with the running sum of learning rates in units of 1e-6."""
    rows, lr_sum = [], 0.0
    for r in sorted(metrics, key=lambda r: r["step"]):
        lr_sum += 1e6 * float(r.get("actor/lr") or 0.0)
        zero_adv = r.get("task/t1_num/zero_adv_frac")
        rows.append({
            "step": r["step"], "lr_sum": round(lr_sum, 2),
            **{f"{t}_grad_share": r.get(f"task/{t}/grad_share") for t in TASKS},
            "t1_groups_with_gradient": None if zero_adv is None else 1.0 - zero_adv,
            "t1_len_right": r.get("task/t1_num/len_correct"), "t1_acc": r.get("task/t1_num/acc"),
            "t1_truncation": r.get("task/t1_num/truncation_rate"), "len_mean": r.get("response_length/mean"),
            "t1_len_gap_finished": r.get("task/t1_num/len_gap_within_finished")})
    return rows


def median_over(rows: list[dict], key: str, first: int, last: int) -> float | None:
    values = [r[key] for r in rows if first <= r["step"] <= last and isinstance(r.get(key), (int, float))]
    return round(statistics.median(values), 4) if values else None


def windows(rows: list[dict]) -> dict:
    """Medians over steps 1-20 and 21-40, then the rest; a run longer than 60 steps also gets its last 20 alone,
    which is where RL v2 had as few numeric groups with gradient as run 1 had at the start."""
    last = max(r["step"] for r in rows)
    spans = [(1, 20), (21, min(40, last))]
    if last > 60:
        spans += [(41, last - 20), (last - 19, last)]
    elif last > 40:
        spans += [(41, last)]
    keys = [f"{t}_grad_share" for t in TASKS] + ["t1_groups_with_gradient", "t1_len_right", "t1_acc", "t1_truncation"]
    return {f"{a}-{b}": {k: median_over(rows, k, a, b) for k in keys} for a, b in spans}


def at_lr_sum(rows: list[dict], target: float, key: str, span: float = 4.0) -> float | None:
    """The median of ``key`` over the steps whose learning-rate sum is within ``span`` of ``target``."""
    values = [r[key] for r in rows if abs(r["lr_sum"] - target) <= span and isinstance(r.get(key), (int, float))]
    return round(statistics.median(values), 3 if key != "t1_len_right" else 0) if values else None


def main() -> None:
    run1 = json.loads((DATA / "rl.json").read_text())
    v2 = json.loads((DATA / "rl_v2_seed1.json").read_text())
    rows1, rows2 = per_step(run1["metrics"]), per_step(v2["metrics"])
    collapse_lr_sum = next(r["lr_sum"] for r in rows1 if r["step"] == COLLAPSE_STEP_RUN1)
    targets = [10, 20, 40, 60, 80, 100]
    replay = json.loads((DATA / "run1_step0_replay_1116.json").read_text())
    first_step = replay["families"]["all"]["by_budget"]["16384"]["left out + penalty"]
    gaps = [r["t1_len_gap_finished"] for r in rows2 if isinstance(r.get("t1_len_gap_finished"), (int, float))]
    numhard = json.loads((DATA / "numhard_64k.json").read_text())["rows"]
    sft32, rl32 = numhard["sft"]["at_32k_pct"], numhard["rl_seed1"]["at_32k_pct"]
    dev = v2.get("dev_by_set") or {}
    steps = sorted(dev, key=int)
    hard0, hard_last = (dev[steps[0]]["hard_dev"], dev[steps[-1]]["hard_dev"]) if steps else ({}, {})

    def change(before: dict, after: dict, key: str, scale: float = 1.0) -> float:
        return round(scale * (after[key] - before[key]), 1)

    out = {
        "about": "Run 1 (rl9b_seed1, docs/talk/data/rl.json) against RL v2 seed 1 (rl_v2_seed1.json): per-step "
                 "series with the running sum of learning rates (units of 1e-6), medians "
                 "over step windows, values at equal learning-rate sums, and what happened to unfinished answers. "
                 "Counts and rates only (scripts/talk/rl_compare.py).",
        "collected": {"run1": run1.get("collected_at"), "rl_v2_seed1": v2.get("collected_at"),
                      "rl_v2_seed1_steps": len(rows2)},
        "run1_collapse": {"step": COLLAPSE_STEP_RUN1, "lr_sum": collapse_lr_sum},
        "windows": {"run1": windows(rows1), "rl_v2_seed1": windows(rows2)},
        "at_lr_sum": {str(t): {name: {k: at_lr_sum(rows, t, k)
                                      for k in ("t1_len_right", "t1_acc", "t1_groups_with_gradient")}
                               for name, rows in (("run1", rows1), ("rl_v2_seed1", rows2))} for t in targets},
        "run1_first_step": {"length_only_groups": first_step["length_only"],
                            "length_only_advantage_share": first_step["length_only_advantage_share"],
                            "right_and_wrong_groups": first_step["right_and_wrong"]},
        "rl_v2_len_gap_finished": {"steps": len(gaps), "positive": sum(g > 0 for g in gaps),
                                   "median": round(statistics.median(gaps)) if gaps else None},
        "unfinished_answers": {
            "run1_d_num_hard_32k": {"from": "sft", "to": "rl_seed1 step 20",
                                    "unfinished": round(rl32["unfinished"] - sft32["unfinished"], 1),
                                    "right": round(rl32["right"] - sft32["right"], 1),
                                    "wrong_value": round(rl32["wrong_value"] - sft32["wrong_value"], 1)},
            "rl_v2_hard_dev_24k": {} if not steps else {
                "from": f"step {steps[0]} (the SFT model)", "to": f"step {steps[-1]}",
                "unfinished": change(hard0, hard_last, "unfinished", 100),
                "right": change(hard0, hard_last, "right", 100),
                "wrong_value": change(hard0, hard_last, "wrong_value", 100)}},
        "per_step": {"run1": rows1, "rl_v2_seed1": rows2},
    }
    (DATA / "rl_runs_compared.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_step"}, indent=1))


if __name__ == "__main__":
    main()
