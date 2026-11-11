"""Train and eval graders must be separate implementations (report §防范原则).

If the eval harness reused the training grader, a parser bug the policy
learned to exploit would inflate the eval number too. This test fails if
any module under ``dpt.eval`` imports ``dpt.rewards``, except two tools that
score no model for the eval (``ALLOWED``), and those two are never imported
by the harness or the graders.
"""

from __future__ import annotations

import ast
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[2] / "src" / "dpt" / "eval"

# Module → the reward modules it may import, and why that does not share grading with the eval.
ALLOWED = {
    # builds D-Fresh: runs the drafted solver code in the T2 sandbox after the T2 static scan (runtime and
    # safety, not grading; the answers are checked with the eval's own numeric grader)
    "autocheck.py": {"dpt.rewards.code_sandbox", "dpt.rewards.hack_patterns"},
    # measures the T3 reward judge itself: sends a second judge the reward judge's exact prompt and input
    "judge_drift.py": {"dpt.rewards.judge", "dpt.rewards.citations", "dpt.rewards.hack_patterns"},
}


def _imports(py: Path) -> list[str]:
    mods = []
    for node in ast.walk(ast.parse(py.read_text())):
        if isinstance(node, ast.Import):
            mods += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods += [node.module] + [f"{node.module}.{a.name}" for a in node.names]
    return mods


def test_eval_never_imports_training_rewards():
    offenders = []
    for py in EVAL_DIR.rglob("*.py"):
        allowed = ALLOWED.get(py.name, set()) if py.parent == EVAL_DIR else set()
        offenders += [
            f"{py.name}: {m}" for m in _imports(py)
            if (m == "dpt.rewards" or m.startswith("dpt.rewards.")) and not any(
                m == a or m.startswith(a + ".") for a in allowed)
        ]
    assert not offenders, offenders


def test_scoring_code_never_reaches_rewards_through_the_allowed_tools():
    tools = tuple(f"dpt.eval.{name[:-3]}" for name in ALLOWED)
    scoring = [EVAL_DIR / "harness.py", EVAL_DIR / "report.py", EVAL_DIR / "gates.py", *(EVAL_DIR / "graders").rglob("*.py")]
    offenders = [f"{py.name}: {m}" for py in scoring for m in _imports(py) if m.startswith(tools)]
    assert not offenders, offenders


SRC = EVAL_DIR.parent
HOSTED = ("dpt.eval.hosted", "dpt.eval.drafts", "dpt.eval.autocheck", "dpt.eval.judge_drift",
          "dpt.eval.blind_review")
TRAINING_PACKAGES = ("data", "rewards", "verl_ext", "rl", "monitoring", "planning")


def test_training_never_imports_the_hosted_model_client():
    """The hosted eval model's output must not reach training data (OpenAI terms; dpt.eval.hosted)."""
    offenders = []
    for pkg in TRAINING_PACKAGES:
        for py in (SRC / pkg).rglob("*.py"):
            for node in ast.walk(ast.parse(py.read_text())):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:   # also `from dpt.eval import hosted`
                    mods = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
                offenders += [f"{pkg}/{py.name}: {m}" for m in mods if m.startswith(HOSTED)]
    assert not offenders, offenders
