"""How well do the RL rewards agree with the evaluation's verdicts? Aggregates only, never the text of a prompt, an
answer or a rubric.

    source scripts/slurm/env.sh; python scripts/talk/reward_vs_eval.py          # the cluster's login node, a few minutes

There are no human gold labels in this project (`dpt.rewards.validate --labeled` needs
data/grader_human_labels.jsonl, which was never built). So this script measures the next best thing, the training
reward against an independent reference on the same answers:

- T1 numeric: the training reward (`dpt.rewards`, base score, length penalty excluded as in `dpt.rewards.validate`)
  against the evaluation grader (`dpt.eval.graders.numeric`, a separate implementation, `missing_unit: strict`) on
  every D-Num sample of the base, SFT and RL step-20 runs. The reference answers are solver-computed, so the eval
  verdict is the closest thing to gold. Positive = right.
- T3 reports: the reward judge (Gemma's rubric score) against GPT-6 Luna's score of the same RL dev answers
  (`dpt.eval.judge_drift` cache), split at a score of 70. Luna never touched training, but it is a judge, not gold.

Writes `$WORK/dpt/talk/reward_vs_eval.json`.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

from dpt.chat.thinking import split_thinking
from dpt.common.io import read_jsonl
from dpt.eval.graders.numeric import grade_numeric
from dpt.rewards.router import RewardRouter, load_rewards_config

WORK = Path(os.environ.get("WORK", "."))
SCRATCH = Path(os.environ.get("SCRATCH", "."))
RUNS = ("base_medium", "sft", "rl_seed1")


def eval_verdict(text: str, gold: dict[str, Any]) -> tuple[bool, str]:
    """The harness's grade_sample for D-Num: thinking stripped, fail closed, then the numeric grader (strict units)."""
    sp = split_thinking(text)
    if not sp.has_end:
        return False, "no_think_end"
    if not sp.answer:
        return False, "empty_answer"
    try:
        g = grade_numeric(sp.answer, gold, missing_unit="strict")
    except Exception:
        return False, "grader_exception"
    return bool(g.correct), str(g.reason)


def t1_confusion(eval_root: Path, items: dict[str, dict[str, Any]], router: RewardRouter) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for run in RUNS:
        cm = collections.Counter()
        why: collections.Counter[str] = collections.Counter()
        for r in read_jsonl(eval_root / run / "d_num" / "generations.jsonl"):
            it = items.get(r["item_id"])
            if it is None:
                continue
            gt = json.dumps(it["answer"])
            for s in r["samples"]:
                ev, ev_reason = eval_verdict(s["text"], it["answer"])
                res = router.grade("t1_num", s["text"], gt, {"task": "t1_num", "response_length": 0})
                rw = float(res.components.get("base_score", res.score)) >= 0.5
                key = ("T" if rw == ev else "F") + ("P" if rw else "N")
                cm[key] += 1
                if rw != ev:
                    why[f"{key}: eval {ev_reason} / reward {res.reason}"] += 1
        n = sum(cm.values())
        tp, fp, tn, fn = (cm[k] for k in ("TP", "FP", "TN", "FN"))
        out[run] = {"n": n, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
                    "agreement": (tp + tn) / n if n else None,
                    "fp_share": fp / n if n else None, "fn_share": fn / n if n else None,
                    "fpr": fp / (fp + tn) if fp + tn else None, "fnr": fn / (fn + tp) if fn + tp else None,
                    "disagreements": dict(why.most_common(12))}
    return out


def t3_confusion(run: Path, dev: Path, threshold: float) -> dict[str, Any]:
    from dpt.eval.judge_drift import DriftConfig, _sha, load_dev_questions, select_rows, step_files

    cache = {r["key"]: r for r in read_jsonl(run / "judge_drift" / "luna_scores.jsonl")}
    model = next(iter(cache.values()))["model"]
    q = load_dev_questions(dev)
    cm = collections.Counter()
    for _, path in step_files(run):
        for r in select_rows(list(read_jsonl(path)), q, DriftConfig().n_prompts, DriftConfig().min_responses):
            rubric = (json.loads(r["gt"]) or {}).get("rubric") or []
            rec = cache.get(_sha(model, r["question"], r["answer"], rubric))
            if rec is None or rec.get("score") is None:
                continue
            g, l = 100 * r["gemma"] >= threshold, 100 * rec["score"] >= threshold
            cm[("T" if g == l else "F") + ("P" if g else "N")] += 1
    n = sum(cm.values())
    return {"threshold": threshold, "n": n, **{k: cm[k] for k in ("TP", "FP", "TN", "FN")},
            "agreement": (cm["TP"] + cm["TN"]) / n if n else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", type=Path, default=WORK / "dpt" / "eval_results")
    ap.add_argument("--d-num", type=Path, default=WORK / "dpt" / "eval_secure" / "d_num.jsonl")
    ap.add_argument("--run", type=Path, default=SCRATCH / "dpt" / "checkpoints" / "rl" / "rl9b_seed1")
    ap.add_argument("--dev", type=Path, default=WORK / "dpt" / "data" / "v1" / "rl_dev.jsonl")
    ap.add_argument("--out", type=Path, default=WORK / "dpt" / "talk" / "reward_vs_eval.json")
    a = ap.parse_args()

    items = {r["id"]: r for r in read_jsonl(a.d_num)}
    router = RewardRouter(load_rewards_config())
    data = {"collected_at": dt.datetime.now().isoformat(timespec="minutes"),
            "t1_reward_vs_eval_grader": t1_confusion(a.eval_root, items, router),
            "t3_gemma_vs_luna": [t3_confusion(a.run, a.dev, th) for th in (50.0, 70.0, 90.0)]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(data, indent=1))
    for run, c in data["t1_reward_vs_eval_grader"].items():
        print(f"T1 {run:12} n={c['n']}  TP {c['TP']}  FP {c['FP']}  TN {c['TN']}  FN {c['FN']}  agree {100 * c['agreement']:.2f}%  "
              f"FPR {100 * (c['fpr'] or 0):.1f}%  FNR {100 * (c['fnr'] or 0):.2f}%")
        for k, v in c["disagreements"].items():
            print(f"      {v:4}  {k}")
    for c in data["t3_gemma_vs_luna"]:
        print(f"T3 Gemma vs Luna at {c['threshold']:.0f}: n={c['n']}  TP {c['TP']}  FP {c['FP']}  TN {c['TN']}  FN {c['FN']}  "
              f"agree {100 * (c['agreement'] or 0):.0f}%")
    print(a.out)


if __name__ == "__main__":
    main()
