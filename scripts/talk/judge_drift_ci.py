"""The judge-drift audit with confidence intervals, for the talk and the reports: aggregates only, never the text of
a prompt, an answer or a rubric.

    source scripts/slurm/env.sh; python scripts/talk/judge_drift_ci.py      # the cluster's login node, after judge_drift.sh

`scripts/slurm/judge_drift.sh` (dpt.eval.judge_drift) has GPT-6 Luna re-score the T3 dev answers that the Gemma
reward judge scored at each RL validation step, and caches Luna's scores in `<run>/judge_drift/`. Its summary gives
each step's mean over responses. This script rebuilds the same response selection, joins both judges' scores from
the dumps and the cache (no API call), and adds what 12 dev prompts need before anyone reads a trend into them:

- prompt-level means (each prompt weighted equally, so a prompt with an extra sample does not count more);
- 95% intervals from resampling prompts (B = 10,000, seed 0), for each step and for the change between steps,
  paired by prompt;
- how far the judges agree on the same answers (Pearson, Spearman) and how often each gives 90 or more.

Writes `$WORK/dpt/talk/judge_drift.json`; copy it to docs/talk/data/ (scripts/talk/deck_figures.py draws it).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import statistics as st
from pathlib import Path
from typing import Any, Callable

from dpt.common.io import read_jsonl
from dpt.eval.judge_drift import DriftConfig, _sha, load_dev_questions, select_rows, step_files

WORK = Path(os.environ.get("WORK", "."))
SCRATCH = Path(os.environ.get("SCRATCH", "."))
Pair = tuple[float, float]                          # (Gemma, Luna), 0-100


def joined(run: Path, dev: Path, cfg: DriftConfig) -> dict[int, dict[str, list[Pair]]]:
    """step -> prompt -> both judges' scores of the responses judge_drift selected (Luna unparsed: left out)."""
    cache = {r["key"]: r for r in read_jsonl(run / "judge_drift" / "luna_scores.jsonl")}
    models = {r["model"] for r in cache.values()}
    if len(models) != 1:
        raise SystemExit(f"expected one Luna model in the cache, found {sorted(models)}")
    model = models.pop()
    questions = load_dev_questions(dev)
    out: dict[int, dict[str, list[Pair]]] = {}
    for step, path in step_files(run):
        d: dict[str, list[Pair]] = {}
        for r in select_rows(list(read_jsonl(path)), questions, cfg.n_prompts, cfg.min_responses):
            rubric = (json.loads(r["gt"]) or {}).get("rubric") or []
            rec = cache.get(_sha(model, r["question"], r["answer"], rubric))
            if rec is not None and rec.get("score") is not None:
                d.setdefault(r["gt"], []).append((100.0 * r["gemma"], 100.0 * rec["score"]))
        if d:
            out[step] = d
    return out


def ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=xs.__getitem__)
    r, i = [0.0] * len(xs), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def pearson(xs: list[float], ys: list[float]) -> float | None:
    mx, my = st.fmean(xs), st.fmean(ys)
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy) if sx and sy else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=SCRATCH / "dpt" / "checkpoints" / "rl" / "rl9b_seed1")
    ap.add_argument("--dev", type=Path, default=WORK / "dpt" / "data" / "v1" / "rl_dev.jsonl")
    ap.add_argument("--out", type=Path, default=WORK / "dpt" / "talk" / "judge_drift.json")
    ap.add_argument("--boot", type=int, default=10_000)
    a = ap.parse_args()

    per = joined(a.run, a.dev, DriftConfig())
    steps = sorted(per)
    prompts = sorted(set.intersection(*(set(per[s]) for s in steps)))
    rng = random.Random(0)
    boots = [[rng.choice(prompts) for _ in prompts] for _ in range(a.boot)]
    fields: dict[str, Callable[[Pair], float]] = {"gemma": lambda p: p[0], "luna": lambda p: p[1],
                                                  "gap": lambda p: p[0] - p[1]}

    def mean(sample: list[str], step: int, f: Callable[[Pair], float]) -> float:
        return st.fmean(st.fmean(f(p) for p in per[step][g]) for g in sample)

    def interval(fn: Callable[[list[str]], float]) -> dict[str, float]:
        xs = sorted(fn(b) for b in boots)
        return {"point": fn(prompts), "lo": xs[int(0.025 * a.boot)], "hi": xs[int(0.975 * a.boot) - 1]}

    rows: list[dict[str, Any]] = []
    for s in steps:
        pairs = [p for g in prompts for p in per[s][g]]
        gs, ls = [p[0] for p in pairs], [p[1] for p in pairs]
        rows.append({"step": s, "prompts": len(prompts), "responses": len(pairs),
                     **{k: interval(lambda b, s=s, f=f: mean(b, s, f)) for k, f in fields.items()},
                     "pearson": pearson(gs, ls), "spearman": pearson(ranks(gs), ranks(ls)),
                     "gemma_ge90": sum(x >= 90 for x in gs), "luna_ge90": sum(x >= 90 for x in ls)})
    changes = [{"from": x, "to": y, **{k: interval(lambda b, x=x, y=y, f=f: mean(b, y, f) - mean(b, x, f))
                                       for k, f in fields.items()}}
               for i, x in enumerate(steps) for y in steps[i + 1:]]
    summary = json.loads((a.run / "judge_drift" / "summary.json").read_text())
    data = {"collected_at": dt.datetime.now().isoformat(timespec="minutes"), "run": str(a.run),
            "method": f"prompt-level means (each of {len(prompts)} T3 dev prompts weighted equally); 95% intervals "
                      f"from resampling prompts (B={a.boot}, seed 0), paired by prompt across steps",
            "alert": summary.get("alert"), "reasons": summary.get("reasons"),
            "served_models": summary.get("served_models"), "steps": rows, "changes": changes}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(data, indent=1))
    for r in rows:
        print(f"step {r['step']:>3}: Gemma {r['gemma']['point']:5.1f}  Luna {r['luna']['point']:5.1f}  "
              f"gap {r['gap']['point']:5.1f} [{r['gap']['lo']:.1f}, {r['gap']['hi']:.1f}]  n {r['responses']}")
    for c in changes:
        print(f"{c['from']:>3} -> {c['to']:>3}: " + "  ".join(
            f"{k} {c[k]['point']:+.1f} [{c[k]['lo']:+.1f}, {c[k]['hi']:+.1f}]" for k in fields))
    print(f"{a.out}: alert {data['alert']}")


if __name__ == "__main__":
    main()
