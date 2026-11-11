"""Periodic human audits of RL samples (report §检测体系-3, §第二层-3).

Two exports, both every ``every`` steps (default 20):

* **Behavior audit**: 100 samples, 50 from the highest-reward and 50 from
  the lowest-reward responses of the step. Reviewers look for behavior a rule
  would miss. The step-170 hack, a broad ``try/except`` that swallows
  failures, was found this way.
* **Grader spot check**: 50 samples with reward == 1 and 50 with
  reward == 0. Reviewers mark each as truly right or wrong, and
  :func:`score_spotcheck` turns the labels into grader false-positive and
  false-negative rates. Grader v1 had FP 2.8% and FN 4.1%; after iteration,
  v3 has FP 0.3% and FN 1.2%.

Files are JSONL with an empty ``human_label`` field that reviewers fill in.
Sampling is seeded by ``(seed, step)``, so a resumed run exports the same
samples.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterable, Sequence

from dpt.common.io import read_jsonl, write_jsonl
from dpt.common.seeding import derive_seed

_KEEP = ("uid", "task", "prompt", "response", "reward", "reason", "flags", "step")


def _slim(r: dict[str, Any], step: int) -> dict[str, Any]:
    out = {k: r.get(k) for k in _KEEP if k in r}
    out["step"] = step
    out["human_label"] = None  # reviewer fills: "correct" | "incorrect" | "hack:<category>"
    out["reviewer_note"] = ""
    return out


class AuditSampler:
    def __init__(
        self,
        out_dir: str | Path,
        *,
        every: int = 20,
        n_high: int = 50,
        n_low: int = 50,
        n_spot_pos: int = 50,
        n_spot_neg: int = 50,
        seed: int = 0,
    ):
        self.out_dir = Path(out_dir)
        self.every, self.n_high, self.n_low = every, n_high, n_low
        self.n_spot_pos, self.n_spot_neg, self.seed = n_spot_pos, n_spot_neg, seed

    def due(self, step: int) -> bool:
        return step > 0 and step % self.every == 0

    def behavior_audit(self, step: int, records: Sequence[dict[str, Any]]) -> Path:
        """Top-``n_high`` and bottom-``n_low`` by reward (ties broken by seeded shuffle)."""
        rng = random.Random(derive_seed(self.seed, "audit", step))
        order = list(range(len(records)))
        rng.shuffle(order)
        order.sort(key=lambda i: float(records[i]["reward"]))  # stable: shuffled ties stay shuffled
        low = order[: self.n_low]
        high = order[max(len(low), len(order) - self.n_high) :]  # never double-count a sample
        chosen = [records[i] for i in low + high]
        path = self.out_dir / f"behavior_step{step:06d}.jsonl"
        write_jsonl(path, [_slim(r, step) for r in chosen])
        return path

    def grader_spotcheck(self, step: int, records: Sequence[dict[str, Any]]) -> Path:
        """``n_spot_pos`` samples with reward==1 and ``n_spot_neg`` with reward==0."""
        rng = random.Random(derive_seed(self.seed, "spot", step))
        pos = [r for r in records if float(r["reward"]) == 1.0]
        neg = [r for r in records if float(r["reward"]) == 0.0]
        chosen = rng.sample(pos, min(len(pos), self.n_spot_pos)) + rng.sample(
            neg, min(len(neg), self.n_spot_neg)
        )
        path = self.out_dir / f"spotcheck_step{step:06d}.jsonl"
        write_jsonl(path, [_slim(r, step) for r in chosen])
        return path

    def maybe_export(self, step: int, records: Sequence[dict[str, Any]]) -> list[Path]:
        if not self.due(step):
            return []
        return [self.behavior_audit(step, records), self.grader_spotcheck(step, records)]


def score_spotcheck(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    """Grader FP/FN rates from labeled spot-check rows.

    FP rate = P(grader said 1 | human says incorrect) is reported as
    the fraction of reward==1 rows that humans marked incorrect. FN rate is
    the fraction of reward==0 rows that humans marked correct. These are the
    quantities the report tracks for grader v1 to v3.
    """
    pos = neg = fp = fn = 0
    for r in rows:
        lab = r.get("human_label")
        if lab is None:
            continue
        human_ok = str(lab) == "correct"
        if float(r["reward"]) == 1.0:
            pos += 1
            fp += not human_ok
        elif float(r["reward"]) == 0.0:
            neg += 1
            fn += human_ok
    return {"n_pos": pos, "n_neg": neg, "fp_rate": fp / max(1, pos), "fn_rate": fn / max(1, neg)}


def load_labeled(path: str | Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path))
