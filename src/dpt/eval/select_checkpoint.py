"""Pick the RL checkpoint to ship from the *dev* curve, never from test sets (report §评测体系).

    python -m dpt.eval.select_checkpoint --run checkpoints/rl/rl27b_seed1 [--metric dev/d_num]

Reads ``<run>/metrics.jsonl`` (written by the trainer every step) and the
``global_step_*`` directories that exist. It returns the saved step with the
best dev metric. Ties within the noise band (``--tie-band``, default = the
D-Num noise-floor max of 2.1 / 100 when the metric is a fraction) go to the
*later* step, which has seen more data and trained longer at the final LR.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dpt.common.io import read_jsonl


def select(run_dir: str | Path, metric: str = "dev/d_num", tie_band: float = 2.1) -> dict:
    run = Path(run_dir)
    saved = sorted(int(p.name.split("_")[-1]) for p in run.glob("global_step_*") if p.is_dir())
    curve = {int(r["step"]): float(r[metric]) for r in read_jsonl(run / "metrics.jsonl") if metric in r}
    cands = [(s, curve[s]) for s in saved if s in curve]
    if not cands:
        raise SystemExit(f"no saved step of {run} has {metric}; align save_freq with the dev frequency")
    best = max(v for _, v in cands)
    within = [s for s, v in cands if best - v <= tie_band]
    step = max(within)
    return {"run": str(run), "metric": metric, "selected_step": step, "value": curve[step],
            "best_value": best, "candidates": dict(cands)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--metric", default="dev/d_num")
    ap.add_argument("--tie-band", type=float, default=2.1)
    a = ap.parse_args()
    print(json.dumps(select(a.run, a.metric, a.tie_band), indent=2))


if __name__ == "__main__":
    main()
