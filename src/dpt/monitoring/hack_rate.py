"""Per-category hack-pattern rates and growth detection (report §检测体系-2).

Rule scanners in ``dpt.rewards.hack_patterns`` tag each response with flag
strings such as ``self_praise``, ``unverifiable_citation``,
``multi_answer``, ``interval_answer``, ``bare_except`` and ``env_probe``.
This tracker turns those flags into per-category rates over windows of
steps. It alerts when a category's rate in the current window is at least
``growth_factor`` times the previous window's rate. The report states the
rule as week-over-week doubling; here it is window-over-window, with the
window measured in RL steps.

A pattern that is being *actively exploited* grows exponentially. The
step-310 self-praise hack went from 0.4% to 17% of responses. Background
noise stays flat. ``min_rate`` and ``min_count`` keep a category from
alerting on 1 → 2 occurrences.

Rule scanning only finds hacks you already know about. That is why
proxy/gold divergence and human audits rank above it.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from dpt.monitoring import keys as K


@dataclass(frozen=True)
class HackGrowth:
    category: str
    prev_rate: float
    cur_rate: float


class HackRateTracker:
    def __init__(
        self, window_steps: int = 20, growth_factor: float = 2.0, min_rate: float = 0.005, min_count: int = 5
    ):
        self.window_steps, self.growth_factor = window_steps, growth_factor
        self.min_rate, self.min_count = min_rate, min_count
        # Per step: (n_samples, {category: count}). Two windows of history are enough.
        self.steps: deque[tuple[int, int, dict[str, int]]] = deque(maxlen=2 * window_steps)

    def update(self, step: int, flags_per_sample: Sequence[Iterable[str]]) -> dict[str, float]:
        """Record one step's flags. Returns ``hack/<category>/rate`` metrics for this step."""
        counts: dict[str, int] = defaultdict(int)
        for flags in flags_per_sample:
            for f in set(flags):
                counts[f] += 1
        n = max(1, len(flags_per_sample))
        self.steps.append((int(step), len(flags_per_sample), dict(counts)))
        return {K.HACK_RATE.format(category=c): v / n for c, v in counts.items()}

    def _window(self, recs: list[tuple[int, int, dict[str, int]]]) -> tuple[int, dict[str, int]]:
        n, cnt = 0, defaultdict(int)
        for _, ns, c in recs:
            n += ns
            for k, v in c.items():
                cnt[k] += v
        return n, cnt

    def growing(self) -> list[HackGrowth]:
        """Categories whose current-window rate is ≥ growth_factor × previous-window rate."""
        if len(self.steps) < 2 * self.window_steps:
            return []
        recs = list(self.steps)
        n_prev, c_prev = self._window(recs[: self.window_steps])
        n_cur, c_cur = self._window(recs[self.window_steps :])
        out = []
        for cat, cnt in c_cur.items():
            cur = cnt / max(1, n_cur)
            prev = c_prev.get(cat, 0) / max(1, n_prev)
            if (
                cnt >= self.min_count
                and cur >= self.min_rate
                and cur >= self.growth_factor * max(prev, 1e-12)
            ):
                out.append(HackGrowth(cat, prev, cur))
        return sorted(out, key=lambda h: -h.cur_rate)

    def state_dict(self) -> dict[str, Any]:
        return {"steps": [[s, n, c] for s, n, c in self.steps]}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.steps.clear()
        for s, n, c in state["steps"]:
            self.steps.append((int(s), int(n), {k: int(v) for k, v in c.items()}))
