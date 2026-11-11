"""Canonical identifiers shared across the code base.

Keeping these in one place avoids the classic multi-task bug where one module
says ``"t1"`` and another says ``"numeric"`` and a per-task metric silently
collapses to zero.
"""

from __future__ import annotations

from typing import Literal

# --- RL / SFT task identifiers (report: "RL 题库" table) --------------------------------------
T1_NUM = "t1_num"          # domain numeric reasoning, programmatic solver ground truth
T2_CODE = "t2_code"        # domain code & tool use, sandbox execution + hidden tests
T3_REPORT = "t3_report"    # open-ended domain report, rubric judge + citation verification
T4_GENERAL = "t4_general"  # general replay (math + instruction following) against forgetting

TASKS: tuple[str, ...] = (T1_NUM, T2_CODE, T3_REPORT, T4_GENERAL)
TaskId = Literal["t1_num", "t2_code", "t3_report", "t4_general"]

# --- Reasoning effort levels exposed by the Qwen3.8 chat template ------------------------------
EFFORTS: tuple[str, ...] = ("low", "medium", "xhigh")
Effort = Literal["low", "medium", "xhigh"]

# --- Evaluation subsets (report: "评测集") ------------------------------------------------------
D_NUM, D_CODE, D_REPORT, D_FRESH = "d_num", "d_code", "d_report", "d_fresh"
EVAL_SETS: tuple[str, ...] = (D_NUM, D_CODE, D_REPORT, D_FRESH)

# --- SFT subsets (report: "SFT 语料") -----------------------------------------------------------
SFT_SUBSETS: tuple[str, ...] = ("domain_reasoning", "tool_traj", "self_distill", "report")


def check_task(task: str) -> str:
    """Return ``task`` unchanged if it is a known task id, else raise ``ValueError``."""
    if task not in TASKS:
        raise ValueError(f"unknown task id {task!r}; expected one of {TASKS}")
    return task


def check_effort(effort: str) -> str:
    """Return ``effort`` unchanged if it is a supported reasoning-effort level."""
    if effort not in EFFORTS:
        raise ValueError(f"unknown reasoning effort {effort!r}; expected one of {EFFORTS}")
    return effort
