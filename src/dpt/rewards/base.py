"""Reward result type and the fail-closed decorator.

Fail closed (report §防范原则): a parse failure, a timeout, a missing format
element or an unexpected exception gives a score of **0** and a
machine-readable ``reason``. It never gives a "middle" score, and it never
raises into the training loop. A grader that leaks partial credit on error
paths is an invitation to reward hacking. A grader that raises kills a
20-minute RL step.

The ``reason`` strings are a stable vocabulary (see :data:`KNOWN_REASONS`).
Monitoring plots their rates per task, so a sudden jump in, say,
``interval_answer`` shows up as a curve and not as a silent reward change.
"""

from __future__ import annotations

import functools
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from dpt.common.logging import get_logger

log = get_logger(__name__)

# Stable reason vocabulary. New reasons may be added; existing ones are never renamed
# (dashboards key on them).
KNOWN_REASONS = (
    "ok",  # graded normally (score may still be 0 = wrong answer)
    "wrong_value",  # parsed fine, value outside tolerance
    "partial_credit",  # ablation A7 arm only: outside tolerance but within max_rel_err
    "truncated",  # response hit the length cap; excluded from loss by the trainer
    "no_think_end",  # no </think> marker: never grade raw thinking
    "no_final_answer",  # no \boxed{} / "Final answer:" and answer text too long to be the answer
    "multi_answer",  # more than one distinct candidate final answer (step-95 hack)
    "interval_answer",  # range/interval/± instead of a single value (step-140 hack)
    "parse_fail",  # value could not be parsed
    "missing_unit",  # reference has a unit, answer has none
    "unit_parse_fail",  # unit string not understood
    "unit_dimension_mismatch",  # e.g. answered in seconds when metres were asked
    "no_code",  # T2 response contains no code block
    "tamper",  # T2 tamper detected (prefix of "tamper:<category>")
    "sandbox_error",  # sandbox infrastructure failure (fail closed)
    "timeout",  # grader / sandbox / judge timeout
    "judge_parse_fail",  # judge output not parseable
    "judge_error",  # judge HTTP failure
    "judge_version_mismatch",  # judge server returned a different model than pinned
    "constraint_failed",  # T4 instruction-following rule violated
    "irrelevant_content",  # T4 filler / off-topic content (step-60 hack)
    "unknown_task",
    "bad_ground_truth",
    "exception",  # prefix of "exception:<Type>"
)


@dataclass
class RewardResult:
    """Outcome of grading one response.

    ``score`` is the scalar the trainer uses. ``components`` holds named
    sub-scores for monitoring. ``flags`` lists hack-pattern categories found in
    the response, whether or not they changed the score. ``truncated=True``
    tells the trainer to drop this response from the loss (report §算法配置:
    "被截断的回答不进 loss"). Its score is irrelevant and is set to 0.
    """

    score: float
    task: str
    reason: str = "ok"
    correct: bool | None = None
    components: dict[str, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    truncated: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.score, (int, float)) or math.isnan(float(self.score)):
            # A NaN reward would poison the group-mean advantage of all 8 siblings.
            self.reason = f"invalid_score:{self.score!r}"
            self.score = 0.0
            self.correct = False
        self.score = float(self.score)

    @property
    def acc(self) -> float:
        """Binary accuracy for logging: ``correct`` when known, else the clipped score."""
        if self.correct is not None:
            return 1.0 if self.correct else 0.0
        return min(1.0, max(0.0, self.score))

    def to_verl_dict(self) -> dict[str, Any]:
        """Flat dict for verl's reward manager (``score`` is the reward; the rest becomes extra info)."""
        out: dict[str, Any] = {
            "score": self.score,
            "acc": self.acc,
            "reason": self.reason,
            "truncated": bool(self.truncated),
        }
        out.update({k: float(v) for k, v in self.components.items()})
        return out


def zero(task: str, reason: str, **kwargs: Any) -> RewardResult:
    """Shorthand for a fail-closed zero result."""
    return RewardResult(score=0.0, task=task, reason=reason, correct=False, **kwargs)


F = TypeVar("F", bound=Callable[..., RewardResult])


def fail_closed(task: str | None = None) -> Callable[[F], F]:
    """Decorator: any exception inside the grader becomes ``score=0, reason="exception:<Type>"``.

    ``TimeoutError`` maps to ``reason="timeout"``. The wrapped function's own
    task id is used when the decorator is given none, taken from a ``task``
    kwarg or else from the RewardResult default.
    """

    def deco(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> RewardResult:
            t0 = time.perf_counter()
            the_task = kwargs.get("task") or task or "unknown"
            try:
                res = fn(*args, **kwargs)
            except TimeoutError:
                return zero(the_task, "timeout")
            except Exception as e:  # noqa: BLE001 - fail closed is the whole point
                log.warning("grader %s failed closed: %s: %s", fn.__name__, type(e).__name__, e)
                return zero(the_task, f"exception:{type(e).__name__}")
            if not isinstance(res, RewardResult):
                return zero(the_task, "exception:BadReturnType")
            res.components.setdefault("grader_ms", (time.perf_counter() - t0) * 1000.0)
            return res

        return wrapper  # type: ignore[return-value]

    return deco
