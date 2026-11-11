"""Grader validation CLI: adversarial suite and FP/FN accounting."""

from __future__ import annotations

from dpt.rewards.router import RewardRouter
from dpt.rewards.validate import fp_fn_rates, main, run_adversarial

T = "x\n</think>\n\n"


def test_fp_fn_rates_definitions():
    gt = {"value": 20, "unit": "s"}
    recs = [
        {"id": "tp", "task": "t1_num", "response": T + r"\boxed{20 s}", "ground_truth": gt, "human_label": 1},
        {"id": "fp", "task": "t1_num", "response": T + r"\boxed{20 s}", "ground_truth": gt, "human_label": 0},
        {
            "id": "fn",
            "task": "t1_num",
            "response": T + "twenty seconds, final.",
            "ground_truth": gt,
            "human_label": 1,
        },
        {"id": "tn", "task": "t1_num", "response": T + r"\boxed{9 s}", "ground_truth": gt, "human_label": 0},
    ]
    r = fp_fn_rates(recs, RewardRouter({}))
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (1, 1, 1, 1)
    assert r["fp_share"] == 0.25 and r["fpr"] == 0.5 and r["examples"]["fp"] == ["fp"]


def test_adversarial_runner_reports_failures():
    cases = [
        {
            "id": "c1",
            "category": "x",
            "task": "t1_num",
            "response": T + r"\boxed{20 s}",
            "ground_truth": {"value": 20, "unit": "s"},
            "expected": 0,
        }
    ]
    out = run_adversarial(cases, RewardRouter({}))
    assert out["passed"] == 0 and out["failures"][0]["id"] == "c1"


def test_cli_passes_on_shipped_suite():
    assert main([]) == 0
