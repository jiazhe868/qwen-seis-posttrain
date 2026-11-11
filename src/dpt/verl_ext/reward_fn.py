"""verl reward entry point (``reward.custom_reward_function.{path,name}``).

The reward workers are Ray actors started by verl. On first use each one
builds a :class:`dpt.rewards.router.RewardRouter` from the ``rewards`` section
of the run's resolved config (``DPT_RESOLVED_CONFIG``). That config includes
any CLI or ablation overrides, so graders, judge endpoint and tolerances are
exactly what the launch record says.

Length handling: the run uses verl's ``dapo`` reward manager. It adds the exact,
token-counted overlong penalty (20K → 24K, down to −0.5) after this function
returns. The router's own penalty and char-estimated truncation flag are
therefore switched off in ``rl_27b.yaml`` (``rewards.length.apply_penalty=false``,
``rewards.length.truncation_from_estimate=false``). Truncation for the loss mask
is decided by the trainer from the response mask, not here.

Gold grading on the dev split (report §检测体系 1: proxy vs gold divergence)
--------------------------------------------------------------------------
For dev rows (``extra_info.split == "dev"``), validated every 20 steps, we also
compute ``gold_score`` with the **independent** eval graders in ``dpt.eval.graders``.
The training graders in ``dpt.rewards`` are a separate implementation, which
is the point. verl reports it as ``val-aux/<task>/gold_score/mean@n``, and
:mod:`dpt.monitoring.bridge` turns that into ``gold/<task>`` and ``dev/d_num``
for the divergence alert. The gold check per task:

* T1 / T4-math: sympy-based numeric grader with **strict** units (stricter than training)
* T4-ifeval:    independent IFEval-style constraint checker
* T2:           hidden-test pass rate (the sandbox is already the strictest signal)
* T3:           not computed in-loop → −1. Gold is a second judge outside training: expert blind review
                via the audit queue, or on the cluster GPT-6 Luna on the dumped dev responses
                (``dpt.eval.judge_drift``; it never feeds back into the trainer)

Training rows get ``gold_score = −1``; the bridge ignores negative values.

Output keys are identical for every sample (``dpt.rewards.router.VERL_KEYS`` +
``gold_score``). verl requires this, because it turns per-sample dicts into columns.
"""

from __future__ import annotations

import os
import threading
from typing import Any

import json

from dpt.chat.thinking import strip_thinking
from dpt.common.io import read_yaml
from dpt.common.logging import get_logger
from dpt.common.types import T1_NUM, T2_CODE, T4_GENERAL
from dpt.rewards import router as _router

log = get_logger(__name__)
_LOCK = threading.Lock()
_READY = False


def _ensure_router() -> None:
    global _READY
    if _READY:
        return
    with _LOCK:
        if _READY:
            return
        path = os.environ.get("DPT_RESOLVED_CONFIG")
        if path:
            cfg = read_yaml(path)
            _router.set_router(_router.RewardRouter(cfg["rewards"]))
            log.info("reward router built from %s", path)
        else:   # stand-alone use (e.g. offline re-scoring): fall back to configs/rl/rewards.yaml
            log.warning("DPT_RESOLVED_CONFIG not set; using configs/rl/rewards.yaml defaults")
        _READY = True


def compute_score(data_source: str, solution_str: str, ground_truth: Any,
                  extra_info: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
    _ensure_router()
    # verl may pass reward_router_address / reward_model_tokenizer; graders do not use them.
    kwargs.pop("reward_router_address", None)
    kwargs.pop("reward_model_tokenizer", None)
    out = _router.compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)
    out["gold_score"] = -1.0
    if (extra_info or {}).get("split") == "dev":
        try:
            out["gold_score"] = gold_score(str(data_source), solution_str, ground_truth, out)
        except Exception as e:  # gold is a monitor; never fail the reward because of it
            log.warning("gold grading failed (%s: %s)", type(e).__name__, e)
            out["gold_score"] = 0.0
    return out


def gold_score(task: str, solution_str: str, ground_truth: Any, proxy: dict[str, Any]) -> float:
    """Independent, stricter correctness check for dev rows (see module docstring)."""
    gt = json.loads(ground_truth) if isinstance(ground_truth, str) else dict(ground_truth)
    answer = strip_thinking(solution_str)
    if task == T2_CODE:
        return float(proxy.get("t2_hidden_pass", 0.0))
    if answer is None:
        return 0.0
    if task == T1_NUM or (task == T4_GENERAL and gt.get("kind") == "math"):
        from dpt.eval.graders.numeric import grade_numeric

        return float(grade_numeric(answer, gt, missing_unit="strict").correct)
    if task == T4_GENERAL and gt.get("kind") == "ifeval":
        from dpt.eval.graders.ifeval import grade_ifeval

        return float(grade_ifeval(answer, gt).score)
    return -1.0
