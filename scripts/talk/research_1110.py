"""Numbers for the reward / rubric / scaling recap of 2026-11-10 (docs/reports/reward_rubric_scaling_1110_en.md).
Aggregates only, never the text of a prompt, an answer or a rubric.

    source scripts/slurm/env.sh; python scripts/talk/research_1110.py      # the cluster's login node, a few minutes

- **Majority voting (maj@k) next to pass@k** on D-Num, from the 64-sample runs of the base, SFT and RL step-20
  models (`<eval_root>/<run>_k64/d_num/generations.jsonl`). Every sample is graded as the harness grades it
  (thinking stripped, fail closed, `grade_numeric` with strict units). A sample whose answer parses to a value in
  the gold's dimension votes with that value in SI units. Votes within the item's tolerance of each other are one
  answer, and the most common answer wins; ties count as the average of the tied answers. Samples without such a
  value abstain, and a chunk where every sample abstains is wrong. For each k the 64 samples are cut into 64/k
  disjoint chunks, and the accuracy is averaged over chunks and items. pass@k is the unbiased estimator from the same
  grades.
- **How atomic the training rubrics and checks are**, per task, from the verified v1 bank
  (`$DATA/v1/stages/04_verify.jsonl`):
  - T3: criteria per rubric, words per criterion, the share joined by "and" / "or" / ";", the share that contains a
    number;
  - T4: constraints per prompt and the kinds;
  - T1: tolerances and units.

Writes `$WORK/dpt/talk/research_1110.json`.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
import os
import re
import statistics
from pathlib import Path
from typing import Any

from dpt.chat.thinking import split_thinking
from dpt.common.io import read_jsonl
from dpt.eval.graders.numeric import grade_numeric

WORK = Path(os.environ.get("WORK", "."))
RUNS = {"base_medium": "base_medium_k64", "sft": "sft_k64", "rl_seed1": "rl_seed1_k64"}
KS = (1, 2, 4, 8, 16, 32, 64)
JOIN = re.compile(r"\b(?:and|or)\b|;", re.I)


# ---------------------------------------------------------------------------------------------- D-Num votes
def vote(text: str, gold: dict[str, Any]) -> tuple[float, float, bool] | None:
    """(value in SI, the item's tolerance in SI, graded right) for a sample with a usable answer, else None."""
    sp = split_thinking(text)
    if not sp.has_end or not sp.answer:
        return None
    try:
        g = grade_numeric(sp.answer, gold, missing_unit="strict")
    except Exception:
        return None
    d = g.details or {}
    if "pred_si" not in d:
        return None
    return float(d["pred_si"]), float(d["tol"]), bool(g.correct)


def graded(text: str, gold: dict[str, Any]) -> bool:
    v = vote(text, gold)
    return bool(v and v[2])


def majority_right(chunk: list[tuple[float, float, bool] | None]) -> float:
    """1 if the most common answer of the chunk is right, 0 if wrong, the average over tied answers."""
    votes = sorted((v for v in chunk if v is not None), key=lambda v: v[0])
    if not votes:
        return 0.0
    tol = votes[0][1]
    clusters: list[list[tuple[float, float, bool]]] = []
    for v in votes:
        if clusters and abs(v[0] - clusters[-1][-1][0]) <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    top = max(len(c) for c in clusters)
    tied = [c for c in clusters if len(c) == top]
    return sum(1.0 for c in tied if c[len(c) // 2][2]) / len(tied)


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def voting(eval_root: Path, items: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for model, run in RUNS.items():
        maj = {k: [] for k in KS}
        pk = {k: [] for k in KS}
        abstain = n_samples = 0
        margins = []
        for r in read_jsonl(eval_root / run / "d_num" / "generations.jsonl"):
            it = items.get(r["item_id"])
            if it is None:
                continue
            votes = [vote(s["text"], it["answer"]) for s in r["samples"]]
            n = len(votes)
            n_samples += n
            abstain += sum(v is None for v in votes)
            c = sum(1 for v in votes if v and v[2])
            for k in KS:
                if k > n:
                    continue
                chunks = [votes[i:i + k] for i in range(0, n - k + 1, k)]
                maj[k].append(statistics.fmean(majority_right(ch) for ch in chunks))
                pk[k].append(pass_at_k(n, c, k))
            margins.append(c / n)
        out[model] = {
            "run": run, "items": len(maj[1]), "samples": n_samples, "abstain_share": abstain / max(1, n_samples),
            "maj_at_k": {str(k): 100 * statistics.fmean(v) for k, v in maj.items() if v},
            "pass_at_k": {str(k): 100 * statistics.fmean(v) for k, v in pk.items() if v},
            "items_by_share_right": dict(sorted(collections.Counter(
                min(9, int(10 * m)) for m in margins).items())),       # decile of each item's sample accuracy
        }
    return out


# ---------------------------------------------------------------------------------------------- the bank
def bank(path: Path) -> dict[str, Any]:
    t3_n, t3_words, t3_joined, t3_joins, t3_number, t4_n = [], [], [], [], [], []
    t4_kinds: collections.Counter[str] = collections.Counter()
    t4_types: collections.Counter[str] = collections.Counter()
    t1_rel: collections.Counter[str] = collections.Counter()
    t1_unit = t1_total = 0
    tasks: collections.Counter[str] = collections.Counter()
    for r in read_jsonl(path):
        t = r["task"]
        tasks[t] += 1
        a = r.get("answer") or {}
        if t == "t3_report":
            rub = a.get("rubric") or []
            t3_n.append(len(rub))
            for c in rub:
                text = str(c.get("criterion", ""))
                t3_words.append(len(text.split()))
                j = len(JOIN.findall(text))
                t3_joins.append(j)
                t3_joined.append(j > 0)
                t3_number.append(bool(re.search(r"\d", text)))
        elif t == "t4_general":
            cons = a.get("constraints") or []
            t4_n.append(len(cons))
            t4_kinds[str(a.get("kind"))] += 1
            for c in cons:
                t4_types[str(c.get("type") or c.get("id") or c.get("kind") or "?").split(":")[0]] += 1
        elif t == "t1_num":
            t1_total += 1
            t1_unit += bool(str(a.get("unit") or "").strip())
            t1_rel[f"{float(a.get('rel_tol', 0.01)):g}"] += 1
    edges = [0, 10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 10_000]
    hist = [sum(lo <= w < hi for w in t3_words) for lo, hi in zip(edges, edges[1:])]
    return {
        "tasks": dict(tasks),
        "t3": {"rubrics": len(t3_n), "criteria": len(t3_words),
               "criteria_per_rubric": dict(sorted(collections.Counter(t3_n).items())),
               "words_median": statistics.median(t3_words) if t3_words else None,
               "words_p10": statistics.quantiles(t3_words, n=10)[0] if len(t3_words) > 10 else None,
               "words_p90": statistics.quantiles(t3_words, n=10)[-1] if len(t3_words) > 10 else None,
               "words_hist_edges": edges, "words_hist": hist,
               "joined_share": statistics.fmean(t3_joined) if t3_joined else None,
               "joins_per_criterion": dict(sorted(collections.Counter(min(6, j) for j in t3_joins).items())),
               "number_share": statistics.fmean(t3_number) if t3_number else None},
        "t4": {"prompts": len(t4_n), "constraints_per_prompt": dict(sorted(collections.Counter(t4_n).items())),
               "kinds": dict(t4_kinds.most_common()), "constraint_types": dict(t4_types.most_common(25))},
        "t1": {"problems": t1_total, "unit_share": t1_unit / max(1, t1_total), "rel_tol": dict(t1_rel.most_common())},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", type=Path, default=WORK / "dpt" / "eval_results")
    ap.add_argument("--d-num", type=Path, default=WORK / "dpt" / "eval_secure" / "d_num.jsonl")
    ap.add_argument("--bank", type=Path, default=WORK / "dpt" / "data" / "v1" / "stages" / "04_verify.jsonl")
    ap.add_argument("--out", type=Path, default=WORK / "dpt" / "talk" / "research_1110.json")
    a = ap.parse_args()
    items = {r["id"]: r for r in read_jsonl(a.d_num)}
    data = {"collected_at": dt.datetime.now().strftime("%Y-%m-%dT%H:%M"),
            "d_num_voting": voting(a.eval_root, items), "bank_v1": bank(a.bank)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(data, indent=1))
    for m, v in data["d_num_voting"].items():
        print(m, "maj@k", {k: round(x, 1) for k, x in v["maj_at_k"].items()},
              "pass@k", {k: round(x, 1) for k, x in v["pass_at_k"].items()}, f"abstain {v['abstain_share']:.3f}")
    t3 = data["bank_v1"]["t3"]
    print("T3", t3["rubrics"], "rubrics", t3["criteria"], "criteria; median words", t3["words_median"],
          f"joined {t3['joined_share']:.2f} number {t3['number_share']:.2f}")
    print("T4", data["bank_v1"]["t4"]["constraints_per_prompt"], data["bank_v1"]["t4"]["kinds"])
    print(a.out)


if __name__ == "__main__":
    main()
