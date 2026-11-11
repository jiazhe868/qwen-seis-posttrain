"""Resume-equivalence check (Incident 2): a resumed run must continue exactly where it stopped.

    python -m dpt.monitoring.compare_runs A/metrics.jsonl B/metrics.jsonl --steps 7-12

Run A trains N steps straight. Run B trains k steps, stops, resumes and trains the rest. In the
report the resumed run silently restarted the sampler and curriculum, which cost 30 steps of
degraded reward. Two families of signals separate "state restored" from "state lost":

* **Step-determined quantities must be identical in A and B**: the global step, the curriculum
  stage, and the learning rate. A curriculum restart or an LR warm-up restart shows up here at
  once. These are the pass criteria.
* **The resumed run must have applied all of its saved state**: ``resume/state_roundtrip_ok`` is
  logged at the first step after the resume (``dpt.rl.state.ExtraStateManager.roundtrip_mismatches``).
  A component that silently ignores part of its checkpoint fails here. Also a pass criterion when
  the metric is present.

Rewards, losses and curriculum mix weights are *reported but not required to match*. Rollouts
are not reproducible across runs: vLLM schedules requests asynchronously, and partial rollout aborts
the long tail at the moment the batch completes. So two correct runs sample different responses,
see different zero-advantage fractions, and the curriculum's auto-reweight can react differently.
With a deterministic generator, pass ``--deterministic`` to make those curves pass criteria too
(relative tolerance ``--rtol``).
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from dpt.common.io import read_jsonl

EXACT_KEYS = ("training/global_step", "curriculum/stage", "actor/lr")
CURVE_KEYS = ("actor/pg_loss", "task/t1_num/reward_mean", "curriculum/mix/t1_num", "actor/grad_norm")
ROUNDTRIP_KEY = "resume/state_roundtrip_ok"


def compare(a_path: str, b_path: str, steps: range, *, rtol: float = 0.05, deterministic: bool = False,
            exact_keys=EXACT_KEYS, curve_keys=CURVE_KEYS) -> dict:
    a = {int(r["step"]): r for r in read_jsonl(a_path)}
    b = {int(r["step"]): r for r in read_jsonl(b_path)}
    common = [s for s in steps if s in a and s in b]
    report: dict = {"steps": [steps.start, steps.stop - 1], "compared_steps": common, "keys": {},
                    "informational": {}, "pass": bool(common)}
    for k in (*exact_keys, *curve_keys):
        pairs = [(a[s][k], b[s][k]) for s in common if a[s].get(k) is not None and b[s].get(k) is not None]
        if not pairs:
            continue
        x, y = np.array(pairs, dtype=float).T
        diff = float(np.max(np.abs(x - y)))
        if k in exact_keys:
            ok = bool(np.allclose(x, y, rtol=1e-9, atol=1e-12))
        else:
            ok = bool(np.all(np.abs(x - y) <= rtol * np.maximum(np.abs(x), 1e-3)))
        entry = {"ok": ok, "max_abs_diff": diff}
        if k in exact_keys or deterministic:
            report["keys"][k] = entry
            report["pass"] &= ok
        else:
            report["informational"][k] = entry
    rt = [b[s][ROUNDTRIP_KEY] for s in sorted(b) if b[s].get(ROUNDTRIP_KEY) is not None]
    if rt:
        ok = all(v >= 0.5 for v in rt)
        report["keys"][ROUNDTRIP_KEY] = {"ok": ok, "values": rt}
        report["pass"] &= ok
    else:
        report["informational"][ROUNDTRIP_KEY] = "not logged by run B (code older than the round-trip check)"
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--steps", default="11-20")
    ap.add_argument("--rtol", type=float, default=0.05)
    ap.add_argument("--deterministic", action="store_true",
                    help="also require loss/reward/mix curves to match (only with a reproducible generator)")
    x = ap.parse_args()
    lo, hi = (int(v) for v in x.steps.split("-"))
    rep = compare(x.a, x.b, range(lo, hi + 1), rtol=x.rtol, deterministic=x.deterministic)
    print(json.dumps(rep, indent=2))
    raise SystemExit(0 if rep["pass"] else 1)


if __name__ == "__main__":
    main()
