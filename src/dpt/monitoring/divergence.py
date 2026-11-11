"""Proxy-vs-gold divergence: the most effective reward-hacking detector in the report.

Every task pairs its training reward (the *proxy*) with a stricter *gold*
check that is evaluated every 20 steps (report §检测体系-1):

==========  ==========================  ==============================================
task        proxy                        gold
==========  ==========================  ==============================================
T1 numeric  training grader on dev       independent eval grader (``dpt.eval.graders``)
T2 code     visible-test pass rate       hidden-test pass rate
T3 report   rubric judge score           expert blind win rate (length-matched)
T4 general  rule checks                  IFEval / independent math grader
==========  ==========================  ==============================================

The alert fires when the proxy keeps rising while gold stays flat or falls
for ``consecutive`` evaluations in a row (default 3). It caught the step-245
fabricated-citation hack: judge score up, expert win rate down 4 points.
It also caught the step-205 test-environment hack: the visible/hidden gap
widened to 23 points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProxyGoldDivergence:
    consecutive: int = 3
    min_proxy_rise: float = 0.0  # proxy must rise by more than this between evals
    gold_tolerance: float = 0.0  # gold "flat" if it rises by at most this
    history: list[tuple[int, float, float]] = field(default_factory=list)
    streak: int = 0

    def observe(self, step: int, proxy: float, gold: float) -> bool:
        """Record one evaluation; return True when the divergence streak reaches ``consecutive``."""
        if self.history:
            _, p0, g0 = self.history[-1]
            diverging = (proxy - p0) > self.min_proxy_rise and (gold - g0) <= self.gold_tolerance
            self.streak = self.streak + 1 if diverging else 0
        self.history.append((int(step), float(proxy), float(gold)))
        return self.streak >= self.consecutive

    def state_dict(self) -> dict[str, Any]:
        return {"history": [list(h) for h in self.history], "streak": self.streak}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.history = [(int(s), float(p), float(g)) for s, p, g in state["history"]]
        self.streak = int(state["streak"])
