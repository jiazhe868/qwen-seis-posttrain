"""Grader validation CLI (report §评测体系 第二层: "grader 判得对不对").

Three checks, run before every RL launch and after any change to ``dpt.rewards``:

1. **Adversarial suite** (``--cases``): hand-built cases with known correct
   scores (equivalent forms, units, multiple candidates, intervals, scientific
   notation, answers buried in prose, …). Every case must pass, and any
   failure exits non-zero. Seeded by ``tests/rewards/build_adversarial_cases.py``.
2. **Human-label calibration** (``--labeled``): model outputs judged right or
   wrong by people. Reports the grader's false-positive and false-negative
   rates. The report's grader went from FP 2.8% / FN 4.1% (v1) to FP 0.3% /
   FN 1.2% (v3). A 4.1% false-negative rate is enough to hide a real
   improvement or to fake one, which is why this number gates RL.
3. **Judge calibration** (``--calibrate``): re-score the fixed T3 calibration
   set with the pinned judge and fail if its score distribution moved.

Rate definitions (both reported, because people quote both):

* ``fp_share = FP / N`` and ``fn_share = FN / N``: share of *all* samples.
  This is the convention used in the project report.
* ``fpr = FP / (FP + TN)`` and ``fnr = FN / (FN + TP)``: conditional rates.

Usage::

    python -m dpt.rewards.validate                                  # adversarial suite only
    python -m dpt.rewards.validate --labeled data/grader_human_labels.jsonl
    python -m dpt.rewards.validate --calibrate                      # needs the judge endpoint
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from dpt.common.config import REPO_ROOT
from dpt.common.io import read_jsonl
from dpt.rewards.router import RewardRouter, load_rewards_config

DEFAULT_CASES = REPO_ROOT / "tests" / "rewards" / "adversarial_cases.jsonl"


def _grade(router: RewardRouter, rec: dict[str, Any]) -> dict[str, Any]:
    extra = dict(rec.get("extra_info") or {})
    extra.setdefault("task", rec["task"])
    extra.setdefault("response_length", 0)  # validation measures correctness, not length penalties
    res = router.grade(rec["task"], rec["response"], rec["ground_truth"], extra)
    return {"base_score": res.components.get("base_score", res.score), "reason": res.reason}


def run_adversarial(cases: Iterable[dict[str, Any]], router: RewardRouter) -> dict[str, Any]:
    """Return per-category pass counts and the list of failures."""
    per_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    failures: list[dict[str, Any]] = []
    for c in cases:
        out = _grade(router, c)
        ok = abs(out["base_score"] - float(c["expected"])) < 1e-9
        per_cat[c["category"]][0] += int(ok)
        per_cat[c["category"]][1] += 1
        if not ok:
            failures.append({"id": c["id"], "expected": c["expected"], **out})
    total = sum(v[1] for v in per_cat.values())
    passed = sum(v[0] for v in per_cat.values())
    return {
        "total": total,
        "passed": passed,
        "per_category": {k: {"passed": v[0], "total": v[1]} for k, v in sorted(per_cat.items())},
        "failures": failures,
    }


def fp_fn_rates(
    records: Iterable[dict[str, Any]], router: RewardRouter, threshold: float = 0.5
) -> dict[str, Any]:
    """Compare grader decisions (``base_score >= threshold``) with ``human_label`` (0/1)."""
    tp = fp = tn = fn = 0
    examples: dict[str, list[str]] = {"fp": [], "fn": []}
    for r in records:
        pred = _grade(router, r)["base_score"] >= threshold
        gold = bool(int(r["human_label"]))
        if pred and gold:
            tp += 1
        elif pred and not gold:
            fp += 1
            examples["fp"].append(str(r.get("id", "?")))
        elif not pred and gold:
            fn += 1
            examples["fn"].append(str(r.get("id", "?")))
        else:
            tn += 1
    n = max(1, tp + fp + tn + fn)
    return {
        "n": tp + fp + tn + fn,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "fp_share": fp / n,
        "fn_share": fn / n,
        "fpr": fp / max(1, fp + tn),
        "fnr": fn / max(1, fn + tp),
        "examples": {k: v[:20] for k, v in examples.items()},
    }


def run_calibration(cfg: dict[str, Any]) -> dict[str, Any]:
    from dpt.rewards.judge import CalibrationItem, JudgeConfig, RubricJudge, calibrate

    t3 = cfg.get("t3_report", {})
    cal = t3.get("calibration", {})
    items = [
        CalibrationItem(**{k: r[k] for k in ("question", "answer", "rubric", "reference_score")})
        for r in read_jsonl(cal["path"])
    ]
    judge = RubricJudge(JudgeConfig.from_dict(t3.get("judge")))
    rep = calibrate(
        judge,
        items,
        max_mean_shift=float(cal.get("max_mean_shift", 0.05)),
        max_mae=float(cal.get("max_mae", 0.10)),
    )
    return rep.__dict__


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Validate training graders (adversarial suite, FP/FN, judge drift)."
    )
    ap.add_argument("--cases", default=str(DEFAULT_CASES))
    ap.add_argument("--labeled", default=None, help="JSONL with task/response/ground_truth/human_label")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--max-fp-share", type=float, default=None, help="fail if fp_share exceeds this")
    ap.add_argument("--max-fn-share", type=float, default=None, help="fail if fn_share exceeds this")
    ap.add_argument("--json", default=None, help="write the summary here")
    args = ap.parse_args(argv)

    cfg = load_rewards_config()
    router = RewardRouter(cfg)
    summary: dict[str, Any] = {}
    rc = 0

    if args.cases and Path(args.cases).exists():
        adv = run_adversarial(read_jsonl(args.cases), router)
        summary["adversarial"] = adv
        print(f"[adversarial] {adv['passed']}/{adv['total']} passed")
        for cat, v in adv["per_category"].items():
            print(f"  {cat:24s} {v['passed']:4d}/{v['total']:<4d}")
        for f in adv["failures"][:50]:
            print(f"  FAIL {f}")
        rc |= int(bool(adv["failures"]))

    if args.labeled:
        rates = fp_fn_rates(read_jsonl(args.labeled), router, args.threshold)
        summary["human_labels"] = rates
        print(
            f"[human labels] n={rates['n']}  FP share {rates['fp_share']:.2%}  FN share {rates['fn_share']:.2%}  "
            f"(FPR {rates['fpr']:.2%}, FNR {rates['fnr']:.2%})"
        )
        if args.max_fp_share is not None and rates["fp_share"] > args.max_fp_share:
            rc |= 1
        if args.max_fn_share is not None and rates["fn_share"] > args.max_fn_share:
            rc |= 1

    if args.calibrate:
        cal = run_calibration(cfg)
        summary["judge_calibration"] = cal
        print(
            f"[judge calibration] ok={cal['ok']} n={cal['n']} mean_shift={cal['mean_shift']:+.3f} mae={cal['mae']:.3f}"
        )
        rc |= int(not cal["ok"])

    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return rc


if __name__ == "__main__":
    sys.exit(main())
