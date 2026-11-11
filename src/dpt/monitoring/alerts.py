"""Alert rule engine for the default RL monitoring panel (report §默认监控面板).

Each row of the report's monitoring table becomes one rule in
``configs/rl/monitoring.yaml``. The trainer calls
:meth:`AlertManager.evaluate` once per step with that step's metric dict,
then acts on the result:

* ``warn``: log it and surface it on the dashboard.
* ``block_rollout``: do not start the next generation phase. It fires when
  the post-sync fixed-prompt logprob probe fails. Incident 3 was a prefix
  cache (including DeltaNet state snapshots) that was not invalidated after
  weight sync; generating from a stale cache poisons the batch.
* ``pause``: stop training and save state for a human to look at. It fires
  when the train↔rollout log-ppl gap exceeds 5e-3. In incident 1 the gap
  crossed 2e-3 about 35 steps before the collapse and reward stayed normal
  until the last 6 steps.

Rule kinds
----------
``threshold``           value above/below levels → highest severity reached
``boolean``             value is falsy (< 0.5) → action
``relative_change``     drop > ``drop_frac`` or rise > ``rise_frac`` vs window extremes
``persistent``          above a level for ``consecutive`` steps in a row
``spike``               value > ``factor`` × mean of the previous ``window`` values
``length_divergence``   incorrect-answer length grows faster than correct-answer length
``consecutive_drop``    an eval metric falls ``consecutive`` times and by more than its CI overall
``baseline_drop``       recent mean falls more than ``max_drop`` (× ``scale``) below an early baseline
``best_drop``           an eval metric falls more than ``max_drop`` (× ``scale``) below its best so far (2026-11-07)
``divergence``          proxy rises while gold is flat/down for ``consecutive`` evals
``rate_growth``         a hack-pattern rate grows ``factor``× window over window

Metric names may contain ``{task}`` (expanded over the configured tasks) or
shell-style ``*`` wildcards (matched against keys present in the step). All
state (histories, streaks) is captured by ``state_dict()`` so alerts behave
identically across checkpoint/resume (incident 2).
"""

from __future__ import annotations

import fnmatch
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from dpt.common.logging import get_logger
from dpt.common.types import TASKS
from dpt.monitoring.divergence import ProxyGoldDivergence

log = get_logger(__name__)

SEVERITY_ORDER = {"warn": 0, "block_rollout": 1, "pause": 2}


@dataclass(frozen=True)
class Alert:
    step: int
    rule: str
    severity: str  # "warn" | "block_rollout" | "pause"
    metric: str
    value: float
    threshold: float | None
    message: str
    risk: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _expand(pattern: str, metrics: Mapping[str, Any], tasks: Sequence[str]) -> list[str]:
    names = [pattern.format(task=t) for t in tasks] if "{task}" in pattern else [pattern]
    out: list[str] = []
    for n in names:
        if any(ch in n for ch in "*?["):
            out += sorted(k for k in metrics if fnmatch.fnmatchcase(k, n))
        else:
            out.append(n)
    return out


def _num(metrics: Mapping[str, Any], key: str) -> float | None:
    v = metrics.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class Rule:
    """Base class. Subclasses implement :meth:`check` for one concrete metric key."""

    kind = "base"

    def __init__(self, name: str, *, metric: str = "", action: str = "warn", risk: str = "", **params: Any):
        if action not in SEVERITY_ORDER:
            raise ValueError(f"rule {name}: unknown action {action!r}")
        self.name, self.metric, self.action, self.risk, self.params = name, metric, action, risk, params
        self.hist: dict[str, deque[tuple[int, float]]] = defaultdict(lambda: deque(maxlen=self._maxlen()))

    def _maxlen(self) -> int:
        return int(self.params.get("window", 20)) * 2 + 2

    def evaluate(self, step: int, metrics: Mapping[str, Any], tasks: Sequence[str]) -> list[Alert]:
        alerts: list[Alert] = []
        for key in _expand(self.metric, metrics, tasks):
            v = _num(metrics, key)
            if v is None:
                continue
            a = self.check(step, key, v, metrics)
            self.hist[key].append((step, v))
            if a is not None:
                alerts.append(a)
        return alerts

    def check(self, step: int, key: str, value: float, metrics: Mapping[str, Any]) -> Alert | None:
        raise NotImplementedError

    def _alert(
        self, step: int, key: str, value: float, thr: float | None, msg: str, severity: str | None = None
    ) -> Alert:
        return Alert(step, self.name, severity or self.action, key, value, thr, msg, self.risk)

    def state_dict(self) -> dict[str, Any]:
        return {"hist": {k: [list(x) for x in v] for k, v in self.hist.items()}}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.hist.clear()
        for k, v in state.get("hist", {}).items():
            self.hist[k].extend((int(s), float(x)) for s, x in v)


class ThresholdRule(Rule):
    kind = "threshold"

    def check(self, step, key, value, metrics):
        best: tuple[str, float, str] | None = None
        for direction in ("above", "below"):
            for sev, thr in (self.params.get(direction) or {}).items():
                hit = value > thr if direction == "above" else value < thr
                if hit and (best is None or SEVERITY_ORDER[sev] > SEVERITY_ORDER[best[0]]):
                    best = (sev, float(thr), direction)
        if best is None:
            return None
        return self._alert(
            step, key, value, best[1], f"{key}={value:.4g} is {best[2]} {best[1]:.4g}", severity=best[0]
        )


class BooleanRule(Rule):
    kind = "boolean"

    def check(self, step, key, value, metrics):
        if value < 0.5:
            return self._alert(step, key, value, None, f"{key} failed")
        return None


class RelativeChangeRule(Rule):
    kind = "relative_change"

    def check(self, step, key, value, metrics):
        window = int(self.params.get("window", 20))
        recent = [v for s, v in self.hist[key] if s >= step - window]
        if len(recent) < int(self.params.get("min_history", max(2, window // 2))):
            return None
        hi, lo = max(recent), min(recent)
        drop_frac, rise_frac = self.params.get("drop_frac"), self.params.get("rise_frac")
        if drop_frac is not None and hi > 0 and (hi - value) / hi > drop_frac:
            return self._alert(
                step,
                key,
                value,
                drop_frac,
                f"{key} fell {100 * (hi - value) / hi:.0f}% within {window} steps (max {hi:.4g})",
            )
        if rise_frac is not None and lo > 0 and (value - lo) / lo > rise_frac:
            return self._alert(
                step,
                key,
                value,
                rise_frac,
                f"{key} rose {100 * (value - lo) / lo:.0f}% within {window} steps (min {lo:.4g})",
            )
        return None


class PersistentRule(Rule):
    kind = "persistent"

    def __init__(self, name: str, **kw: Any):
        super().__init__(name, **kw)
        self.streak: dict[str, int] = defaultdict(int)

    def check(self, step, key, value, metrics):
        thr, n = float(self.params["above"]), int(self.params["consecutive"])
        self.streak[key] = self.streak[key] + 1 if value > thr else 0
        if self.streak[key] == n:  # fire once when the streak is reached; re-arms after it breaks
            return self._alert(step, key, value, thr, f"{key} > {thr} for {n} consecutive steps")
        return None

    def state_dict(self):
        return {**super().state_dict(), "streak": dict(self.streak)}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.streak = defaultdict(int, {k: int(v) for k, v in state.get("streak", {}).items()})


class SpikeRule(Rule):
    kind = "spike"

    def check(self, step, key, value, metrics):
        window, factor = int(self.params.get("window", 20)), float(self.params.get("factor", 5.0))
        prev = [v for _, v in list(self.hist[key])[-window:]]
        if len(prev) < int(self.params.get("min_history", 5)):
            return None
        mean = sum(prev) / len(prev)
        if mean > 0 and value > factor * mean:
            return self._alert(
                step, key, value, factor * mean, f"{key}={value:.3g} > {factor}× moving mean {mean:.3g}"
            )
        return None


class LengthDivergenceRule(Rule):
    """Wrong answers getting longer faster than right ones signals length bias or repetition loops."""

    kind = "length_divergence"

    def evaluate(self, step, metrics, tasks):
        alerts = []
        inc_t, cor_t = self.params["incorrect"], self.params["correct"]
        window, margin = int(self.params.get("window", 20)), float(self.params.get("margin", 0.10))
        for task in tasks if "{task}" in inc_t else [None]:
            ki = inc_t.format(task=task) if task else inc_t
            kc = cor_t.format(task=task) if task else cor_t
            vi, vc = _num(metrics, ki), _num(metrics, kc)
            if vi is None or vc is None:
                continue
            hi, hc = self.hist[ki], self.hist[kc]
            old_i = next((v for s, v in hi if s >= step - window), None)
            old_c = next((v for s, v in hc if s >= step - window), None)
            if old_i and old_c and len(hi) >= window // 2:
                gi, gc = vi / old_i, vc / old_c
                if gi > 1.0 and gi - gc > margin:
                    alerts.append(
                        self._alert(
                            step,
                            ki,
                            vi,
                            margin,
                            f"incorrect-length growth ×{gi:.2f} vs correct " f"×{gc:.2f} over {window} steps",
                        )
                    )
            hi.append((step, vi))
            hc.append((step, vc))
        return alerts

    def check(self, step, key, value, metrics):  # pragma: no cover - evaluate() is overridden
        return None


class ConsecutiveDropRule(Rule):
    kind = "consecutive_drop"

    def __init__(self, name: str, **kw: Any):
        super().__init__(name, **kw)
        self.streak: dict[str, int] = defaultdict(int)
        self.streak_start: dict[str, float] = {}

    def check(self, step, key, value, metrics):
        n = int(self.params.get("consecutive", 3))
        ci = _num(metrics, self.params.get("ci_metric", "")) or float(self.params.get("default_ci", 0.0))
        if self.hist[key]:
            prev = self.hist[key][-1][1]
            if value < prev:
                if self.streak[key] == 0:
                    self.streak_start[key] = prev
                self.streak[key] += 1
            else:
                self.streak[key] = 0
        if self.streak[key] >= n and self.streak_start[key] - value > ci:
            self.streak[key] = 0  # re-arm
            return self._alert(
                step,
                key,
                value,
                ci,
                f"{key} fell {n}× in a row, total " f"{self.streak_start[key] - value:.2f} > CI {ci:.2f}",
            )
        return None

    def state_dict(self):
        return {**super().state_dict(), "streak": dict(self.streak), "streak_start": dict(self.streak_start)}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.streak = defaultdict(int, {k: int(v) for k, v in state.get("streak", {}).items()})
        self.streak_start = {k: float(v) for k, v in state.get("streak_start", {}).items()}


class BaselineDropRule(Rule):
    kind = "baseline_drop"

    def _maxlen(self) -> int:
        return 10_000

    def check(self, step, key, value, metrics):
        nb, w = int(self.params.get("baseline_steps", 10)), int(self.params.get("window", 10))
        scale, max_drop = float(self.params.get("scale", 1.0)), float(self.params["max_drop"])
        vals = [v for _, v in self.hist[key]] + [value]
        if len(vals) < nb + w:
            return None
        base = sum(vals[:nb]) / nb
        recent = sum(vals[-w:]) / w
        drop = (base - recent) * scale
        if drop > max_drop:
            return self._alert(
                step,
                key,
                recent,
                max_drop,
                f"{key} recent mean {recent:.4g} is {drop:.2f} points " f"below baseline {base:.4g}",
            )
        return None


class BestDropRule(Rule):
    """Pause when an eval metric falls more than ``max_drop`` below its best value so far (2026-11-07).

    Run 1 (rl9b_seed1) had dev D-Num 92.0 → 97.3 at step 20 → 78.9 at step 40 and was stopped only at step 56 by
    the train/rollout mismatch alarm; ``consecutive_drop`` needs three falls in a row. This fires on the first dev
    pass after the turn. ``min_evals`` values must have been seen before it can fire.
    """

    kind = "best_drop"

    def __init__(self, name: str, **kw: Any):
        super().__init__(name, **kw)
        self.best: dict[str, float] = {}

    def check(self, step, key, value, metrics):
        scale, max_drop = float(self.params.get("scale", 1.0)), float(self.params["max_drop"])
        seen = len(self.hist[key])
        best = self.best.get(key)
        self.best[key] = value if best is None else max(best, value)
        if best is None or seen < int(self.params.get("min_evals", 1)):
            return None
        drop = (best - value) * scale
        if drop > max_drop:
            return self._alert(step, key, value, max_drop,
                               f"{key} is {drop:.2f} points below its best {best:.4g} (limit {max_drop:g})")
        return None

    def state_dict(self):
        return {**super().state_dict(), "best": dict(self.best)}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.best = {k: float(v) for k, v in state.get("best", {}).items()}


class DivergenceRule(Rule):
    kind = "divergence"

    def __init__(self, name: str, **kw: Any):
        super().__init__(name, **kw)
        self.detectors: dict[str, ProxyGoldDivergence] = {}

    def evaluate(self, step, metrics, tasks):
        alerts = []
        pt, gt = self.params["proxy"], self.params["gold"]
        for task in tasks if "{task}" in pt else [None]:
            kp, kg = (pt.format(task=task), gt.format(task=task)) if task else (pt, gt)
            p, g = _num(metrics, kp), _num(metrics, kg)
            if p is None or g is None:
                continue
            det = self.detectors.setdefault(
                kp,
                ProxyGoldDivergence(
                    consecutive=int(self.params.get("consecutive", 3)),
                    min_proxy_rise=float(self.params.get("min_proxy_rise", 0.0)),
                    gold_tolerance=float(self.params.get("gold_tolerance", 0.0)),
                ),
            )
            if det.observe(step, p, g):
                alerts.append(
                    self._alert(
                        step,
                        kp,
                        p,
                        None,
                        f"{kp} rising while {kg} flat/down for "
                        f"{det.streak} consecutive evals: suspect reward hacking",
                    )
                )
        return alerts

    def check(self, step, key, value, metrics):  # pragma: no cover
        return None

    def state_dict(self):
        return {"detectors": {k: d.state_dict() for k, d in self.detectors.items()}}

    def load_state_dict(self, state):
        self.detectors = {}
        for k, s in state.get("detectors", {}).items():
            d = ProxyGoldDivergence(consecutive=int(self.params.get("consecutive", 3)))
            d.load_state_dict(s)
            self.detectors[k] = d


class RateGrowthRule(Rule):
    """Window-over-window growth of hack-pattern rates; absent categories count as rate 0 that step."""

    kind = "rate_growth"

    def evaluate(self, step, metrics, tasks):
        window, factor = int(self.params.get("window", 20)), float(self.params.get("factor", 2.0))
        min_rate = float(self.params.get("min_rate", 0.005))
        present = set(_expand(self.metric, metrics, tasks))
        alerts = []
        for key in present | set(self.hist):
            v = _num(metrics, key) if key in present else 0.0
            h = self.hist[key]
            h.append((step, v if v is not None else 0.0))
            if len(h) < 2 * window:
                continue
            vals = [x for _, x in h]
            prev, cur = sum(vals[-2 * window : -window]) / window, sum(vals[-window:]) / window
            if cur >= min_rate and cur >= factor * max(prev, 1e-12):
                alerts.append(
                    self._alert(
                        step,
                        key,
                        cur,
                        factor * prev,
                        f"{key} window mean {cur:.3%} vs previous {prev:.3%} (≥{factor}×): "
                        "pattern may be actively exploited",
                    )
                )
        return alerts

    def check(self, step, key, value, metrics):  # pragma: no cover
        return None


RULE_KINDS: dict[str, type[Rule]] = {
    cls.kind: cls
    for cls in (
        ThresholdRule,
        BooleanRule,
        RelativeChangeRule,
        PersistentRule,
        SpikeRule,
        LengthDivergenceRule,
        ConsecutiveDropRule,
        BaselineDropRule,
        BestDropRule,
        DivergenceRule,
        RateGrowthRule,
    )
}


def build_rule(spec: Mapping[str, Any]) -> Rule:
    spec = dict(spec)
    kind = spec.pop("kind")
    name = spec.pop("name")
    if kind not in RULE_KINDS:
        raise ValueError(f"unknown rule kind {kind!r}; known: {sorted(RULE_KINDS)}")
    return RULE_KINDS[kind](name, **spec)


class AlertManager:
    """Evaluates all rules each step; tracks whether training must pause or rollout is blocked."""

    def __init__(self, rules: Iterable[Rule], *, tasks: Sequence[str] = TASKS, cooldown_steps: int = 10):
        self.rules = list(rules)
        names = [r.name for r in self.rules]
        if len(set(names)) != len(names):
            raise ValueError("duplicate rule names")
        self.tasks, self.cooldown = tuple(tasks), cooldown_steps
        self.last_alerts: list[Alert] = []
        self.history: list[Alert] = []
        self._last_fired: dict[str, int] = {}

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "AlertManager":
        """``cfg`` is the ``monitoring`` section of configs/rl/monitoring.yaml (a plain dict). ``extra_rules`` are
        appended to ``rules``: a config that inherits the panel adds rules there without restating the list."""
        return cls(
            [build_rule(r) for r in list(cfg["rules"]) + list(cfg.get("extra_rules") or [])],
            tasks=cfg.get("tasks", TASKS),
            cooldown_steps=int(cfg.get("cooldown_steps", 10)),
        )

    def evaluate(self, step: int, metrics: Mapping[str, Any]) -> list[Alert]:
        fired: list[Alert] = []
        for rule in self.rules:
            for a in rule.evaluate(step, metrics, self.tasks):
                key = f"{a.rule}|{a.metric}|{a.severity}"
                # Warnings are rate-limited; pause / block_rollout are never suppressed.
                if a.severity == "warn" and step - self._last_fired.get(key, -(10**9)) < self.cooldown:
                    continue
                self._last_fired[key] = step
                fired.append(a)
                (log.error if a.severity != "warn" else log.warning)(
                    "[%s] %s: %s", a.severity, a.rule, a.message
                )
        self.last_alerts = fired
        self.history.extend(fired)
        return fired

    def should_pause(self) -> bool:
        return any(a.severity == "pause" for a in self.last_alerts)

    def rollout_blocked(self) -> bool:
        return any(a.severity == "block_rollout" for a in self.last_alerts)

    def step_metrics(self) -> dict[str, float]:
        """Counts for logging alongside training metrics."""
        out = {f"alerts/n_{s}": 0.0 for s in SEVERITY_ORDER}
        for a in self.last_alerts:
            out[f"alerts/n_{a.severity}"] += 1
        return out

    def state_dict(self) -> dict[str, Any]:
        return {
            "rules": {r.name: r.state_dict() for r in self.rules},
            "last_fired": dict(self._last_fired),
            "history": [a.to_dict() for a in self.history[-1000:]],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for r in self.rules:
            if r.name in state.get("rules", {}):
                r.load_state_dict(state["rules"][r.name])
        self._last_fired = {k: int(v) for k, v in state.get("last_fired", {}).items()}
        self.history = [Alert(**a) for a in state.get("history", [])]
