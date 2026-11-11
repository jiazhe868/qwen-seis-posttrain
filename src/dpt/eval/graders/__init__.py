"""Evaluation graders, written independently of the training graders.

Why a second implementation
---------------------------
Training rewards (:mod:`dpt.rewards`) and evaluation grading are deliberately
separate code (report §防范原则: "训练和评测的 grader 分开实现"). If both
used one parser, a parser bug the policy learns to exploit would also inflate
the eval score. A shared bug is invisible; two independent bugs disagree and
show up as a proxy/gold divergence. Nothing in this package imports
:mod:`dpt.rewards`, and ``tests/eval/test_independence.py`` enforces that.

Every grader returns a :class:`GradeResult` and fails closed. A parse failure,
missing answer, multiple answers, interval answer or runner error scores 0
with a machine-readable ``reason``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GradeResult:
    correct: bool
    score: float  # 1.0/0.0 for exact graders; fraction for partial (e.g. tests passed)
    reason: str = "ok"  # "ok" | "no_answer" | "multiple_answers" | "interval" | "unit_mismatch" | ...
    details: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def fail(reason: str, **details: Any) -> "GradeResult":
        return GradeResult(correct=False, score=0.0, reason=reason, details=dict(details))


__all__ = ["GradeResult"]
