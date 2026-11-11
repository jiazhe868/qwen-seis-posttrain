"""Three-stage multi-task curriculum with a hard-problem pool and automatic reweighting.

Report §三段课程 and §多任务 RL 难调:

| stage | steps   | T1 / T2 / T3 / T4 | why |
|-------|---------|-------------------|-----|
| A     | 0–120   | 50 / 25 /  0 / 25 | learn on reliable, verifiable rewards first and confirm the run is stable |
| B     | 120–300 | 40 / 25 / 15 / 20 | judge-scored T3 enters late (no judge-pleasing first); hard pool at 150 |
| C     | 300–420 | 35 / 25 / 15 / 25 | more general replay plus a lower LR to win back forgetting |

The mix was not designed up front. A naive proportional mix made T1 stall
after step 140 and general ability drop, because token-level loss let long
T2/T3 answers dominate and the T3 judge's length preference leaked into the
other tasks.

Auto-reweighting (§调整-6): if a task's dev metric has not improved for 30
steps *and* its zero-advantage group fraction is above 60%, its sampling
weight is reduced and its hard-pool share is raised. Design note: the hard
share is raised only when the degenerate groups are mostly *all-solved*
(the task is saturating). If they are mostly all-failed, more hard problems
would only add more zero-signal groups. When the solved/failed breakdown is
not supplied, we follow the report literally and raise it.

Determinism and exact resume (Incident 2). Each pool's order in epoch ``e``
is a permutation seeded by ``derive_seed(seed, task, pool, e)``. A sampler's
entire state is therefore ``(epoch, cursor)``, plus fingerprints that detect
a resume against different data. Checkpoints store the full
:meth:`Curriculum.state_dict`. The run-2 incident (missing sampler,
curriculum, hard-pool, RNG and scheduler state, so the resumed run restarted
stage A's mix and repeated warmup) cannot recur silently.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from dpt.common.hashing import sha256_obj
from dpt.common.logging import get_logger
from dpt.common.seeding import derive_seed
from dpt.common.types import TASKS
from dpt.rl.advantage import largest_remainder_counts

log = get_logger(__name__)

MAIN, HARD = "main", "hard"


# ----------------------------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Stage:
    name: str
    start: int          # inclusive
    end: int            # exclusive
    mix: Mapping[str, float]


@dataclass
class AutoReweightConfig:
    enabled: bool = True
    window: int = 30                    # "连续 30 步"
    zero_adv_threshold: float = 0.60    # "零优势组比例超过 60%"
    min_dev_improvement: float = 0.0    # dev must beat its best by more than this to count as improving
    weight_factor: float = 0.8          # multiplicative cut of the task's sampling weight per trigger
    min_weight_factor: float = 0.4      # never cut a task below 40% of its stage weight
    hard_share_increment: float = 0.10
    max_hard_share: float = 0.50
    cooldown: int = 30                  # steps between two triggers for the same task


@dataclass
class DifficultyWeights:
    """Draw harder main-pool problems more often (2026-11-07). A problem's pass rate is the RL start policy's
    share of right answers in the difficulty pass (``extra_info.pass_at_8_medium``). The first band whose upper
    bound is at least the rate gives the problem's copies per epoch of its pool; a problem without a rate (T2, T3:
    not calibrated) gets ``unknown``. Copies 0 leave a problem out."""

    enabled: bool = False
    bands: list[tuple[float, int]] = field(default_factory=lambda: [(0.375, 3), (0.625, 2), (1.0, 1)])
    unknown: int = 1

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "DifficultyWeights":
        d = dict(d or {})
        bands = [(float(b[0]), int(b[1])) for b in d.get("bands", cls().bands)]
        if any(c < 0 for _, c in bands) or bands != sorted(bands, key=lambda b: b[0]):
            raise ValueError(f"difficulty bands must be sorted by upper bound with copies >= 0: {bands}")
        unknown = {k for k in d} - {"enabled", "bands", "unknown"}
        if unknown:
            raise ValueError(f"unknown difficulty_weights keys {sorted(unknown)}")
        return cls(enabled=bool(d.get("enabled", False)), bands=bands, unknown=int(d.get("unknown", 1)))

    def copies(self, rate: float | None) -> int:
        if rate is None:
            return self.unknown
        for upper, c in self.bands:
            if rate <= upper + 1e-9:
                return c
        return self.bands[-1][1]


@dataclass
class CurriculumConfig:
    stages: list[Stage]
    prompts_per_step: int = 128
    hard_pool_join_step: int = 150
    hard_share: float = 0.15            # initial share of a task's prompts drawn from its hard pool
    seed: int = 1
    auto_reweight: AutoReweightConfig = field(default_factory=AutoReweightConfig)
    difficulty_weights: DifficultyWeights = field(default_factory=DifficultyWeights)
    # A hard pool smaller than this has its share scaled by size / hard_pool_min_size, so a few problems are not
    # drawn a hundred times each (2026-10-27: 26 hard problems would have been 9.5% of all prompts). 0 = off.
    hard_pool_min_size: int = 0

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CurriculumConfig":
        stages = [Stage(name=s["name"], start=int(s["start"]), end=int(s["end"]), mix=dict(s["mix"]))
                  for s in d["stages"]]
        ar = AutoReweightConfig(**dict(d.get("auto_reweight", {})))
        return cls(
            stages=stages,
            prompts_per_step=int(d.get("prompts_per_step", 128)),
            hard_pool_join_step=int(d.get("hard_pool_join_step", 150)),
            hard_share=float(d.get("hard_share", 0.15)),
            seed=int(d.get("seed", 1)),
            auto_reweight=ar,
            difficulty_weights=DifficultyWeights.from_dict(d.get("difficulty_weights")),
            hard_pool_min_size=int(d.get("hard_pool_min_size", 0)),
        )

    def validate(self) -> None:
        prev_end = 0
        for s in self.stages:
            if s.start != prev_end:
                raise ValueError(f"stage {s.name} starts at {s.start}, expected {prev_end} (no gaps/overlaps)")
            if s.end <= s.start:
                raise ValueError(f"stage {s.name} is empty")
            unknown = set(s.mix) - set(TASKS)
            if unknown:
                raise ValueError(f"stage {s.name}: unknown tasks {unknown}")
            if abs(sum(s.mix.values()) - 1.0) > 1e-6:
                raise ValueError(f"stage {s.name}: mix sums to {sum(s.mix.values())}, expected 1")
            prev_end = s.end

    @property
    def total_steps(self) -> int:
        return self.stages[-1].end


# ----------------------------------------------------------------------------------------------
# Pool sampler
# ----------------------------------------------------------------------------------------------
class PoolSampler:
    """Epoch-wise shuffled sampling without replacement; state is ``(epoch, cursor)``."""

    def __init__(self, items: Sequence[str], seed: int, key: str) -> None:
        self.items = list(items)
        self.seed = seed
        self.key = key
        self.epoch = 0
        self.cursor = 0
        self.fingerprint = sha256_obj(self.items)[:16]
        self._perm_epoch = -1
        self._perm: np.ndarray | None = None

    def _permutation(self) -> np.ndarray:
        if self._perm_epoch != self.epoch:
            rng = np.random.default_rng(derive_seed(self.seed, self.key, self.epoch))
            self._perm = rng.permutation(len(self.items))
            self._perm_epoch = self.epoch
        assert self._perm is not None
        return self._perm

    def draw(self, k: int) -> list[str]:
        if k and not self.items:
            raise ValueError(f"pool {self.key} is empty but {k} items were requested")
        out: list[str] = []
        while len(out) < k:
            perm = self._permutation()
            take = min(k - len(out), len(self.items) - self.cursor)
            out.extend(self.items[i] for i in perm[self.cursor : self.cursor + take])
            self.cursor += take
            if self.cursor >= len(self.items):
                self.epoch += 1
                self.cursor = 0
        return out

    def state_dict(self) -> dict[str, Any]:
        return {"epoch": self.epoch, "cursor": self.cursor, "n": len(self.items), "fingerprint": self.fingerprint}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state["fingerprint"] != self.fingerprint:
            raise ValueError(
                f"pool {self.key}: checkpoint fingerprint {state['fingerprint']} != data {self.fingerprint}. "
                "Resuming against a different problem bank must be an explicit decision."
            )
        self.epoch, self.cursor = int(state["epoch"]), int(state["cursor"])


# ----------------------------------------------------------------------------------------------
# Curriculum
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Draw:
    task: str
    pool: str
    uid: str


class Curriculum:
    """Decides which problems each RL step sees.

    Usage per step ``s`` (0-indexed)::

        draws = cur.sample(s)                      # exactly once per step
        ...rollout, reward, train...
        cur.observe(s, {"t1_num": {"zero_adv_frac": 0.31, "dev_metric": 63.2}, ...})
    """

    def __init__(self, config: CurriculumConfig, pools: Mapping[str, Mapping[str, Sequence[str]]],
                 rates: Mapping[str, float | None] | None = None) -> None:
        config.validate()
        self.cfg = config
        self.samplers: dict[str, dict[str, PoolSampler]] = {}
        dw = config.difficulty_weights
        for t in TASKS:
            tp = pools.get(t, {})
            main = list(tp.get(MAIN, []))
            if dw.enabled:              # a problem appears copies(rate) times in each epoch of its pool
                main = [u for u in main for _ in range(dw.copies((rates or {}).get(u)))]
            self.samplers[t] = {MAIN: PoolSampler(main, config.seed, f"{t}/{MAIN}"),
                                HARD: PoolSampler(tp.get(HARD, []), config.seed, f"{t}/{HARD}")}
        ar = config.auto_reweight
        self.weight_factor = {t: 1.0 for t in TASKS}
        self.hard_share_extra = {t: 0.0 for t in TASKS}
        self.zero_adv_hist: dict[str, deque] = {t: deque(maxlen=ar.window) for t in TASKS}
        self.best_dev: dict[str, float | None] = {t: None for t in TASKS}
        self.best_dev_step: dict[str, int | None] = {t: None for t in TASKS}
        self.last_trigger_step: dict[str, int | None] = {t: None for t in TASKS}
        self.last_sampled_step = -1
        self.events: list[dict[str, Any]] = []

    # -- queries --------------------------------------------------------------------------
    def stage_at(self, step: int) -> Stage:
        for s in self.cfg.stages:
            if s.start <= step < s.end:
                return s
        return self.cfg.stages[-1]  # past the end: stay in the last stage

    def effective_mix(self, step: int) -> dict[str, float]:
        stage = self.stage_at(step)
        raw = {t: float(stage.mix.get(t, 0.0)) * self.weight_factor[t] for t in TASKS}
        s = sum(raw.values())
        return {t: v / s for t, v in raw.items()}

    def hard_share(self, task: str, step: int) -> float:
        n_hard = len(self.samplers[task][HARD].items)
        if step < self.cfg.hard_pool_join_step or not n_hard:
            return 0.0
        share = min(self.cfg.hard_share + self.hard_share_extra[task], 1.0)
        if self.cfg.hard_pool_min_size > 0:
            share *= min(1.0, n_hard / self.cfg.hard_pool_min_size)
        return share

    # -- sampling -------------------------------------------------------------------------
    def sample(self, step: int) -> list[Draw]:
        if step <= self.last_sampled_step:
            raise RuntimeError(
                f"sample({step}) after step {self.last_sampled_step} was already sampled. "
                "Calling it twice per step would desynchronize resume."
            )
        counts = largest_remainder_counts(self.effective_mix(step), self.cfg.prompts_per_step)
        draws: list[Draw] = []
        for t in TASKS:
            c = counts[t]
            if c == 0:
                continue
            n_hard = int(np.floor(c * self.hard_share(t, step) + 0.5))
            n_main = c - n_hard
            draws += [Draw(t, HARD, u) for u in self.samplers[t][HARD].draw(n_hard)]
            draws += [Draw(t, MAIN, u) for u in self.samplers[t][MAIN].draw(n_main)]
        self.last_sampled_step = step
        return draws

    # -- feedback / auto-reweighting ------------------------------------------------------
    def observe(self, step: int, task_stats: Mapping[str, Mapping[str, float]]) -> list[dict[str, Any]]:
        """Feed per-task statistics for ``step``; return auto-reweighting events (possibly empty).

        Recognized keys per task: ``zero_adv_frac`` (every step), ``dev_metric``
        (on eval steps; higher is better), ``all_solved_frac`` (optional,
        fraction of groups that were all-correct).
        """
        ar = self.cfg.auto_reweight
        stage = self.stage_at(step)
        new_events: list[dict[str, Any]] = []
        for t, st in task_stats.items():
            if t not in TASKS:
                raise ValueError(f"unknown task {t!r}")
            if "zero_adv_frac" in st:
                self.zero_adv_hist[t].append(float(st["zero_adv_frac"]))
            if st.get("dev_metric") is not None:
                v = float(st["dev_metric"])
                best = self.best_dev[t]
                if best is None or v > best + ar.min_dev_improvement:
                    self.best_dev[t], self.best_dev_step[t] = v, step
            if self.best_dev_step[t] is None:
                self.best_dev_step[t] = step  # start the "no improvement" clock at first observation

            if not ar.enabled or float(stage.mix.get(t, 0.0)) <= 0.0:
                continue
            hist = self.zero_adv_hist[t]
            stalled = step - int(self.best_dev_step[t]) >= ar.window
            saturated = len(hist) >= ar.window and float(np.mean(hist)) > ar.zero_adv_threshold
            last = self.last_trigger_step[t]
            cooled = last is None or step - last >= ar.cooldown
            if not (stalled and saturated and cooled):
                continue
            old_w, old_h = self.weight_factor[t], self.hard_share_extra[t]
            self.weight_factor[t] = max(old_w * ar.weight_factor, ar.min_weight_factor)
            solved = st.get("all_solved_frac")
            zero = float(np.mean(hist))
            mostly_solved = solved is None or (zero > 0 and float(solved) / zero >= 0.5)
            if mostly_solved:
                cap = max(ar.max_hard_share - self.cfg.hard_share, 0.0)
                self.hard_share_extra[t] = min(old_h + ar.hard_share_increment, cap)
            self.last_trigger_step[t] = step
            ev = {
                "step": step, "task": t, "zero_adv_mean": zero,
                "weight_factor": [old_w, self.weight_factor[t]],
                "hard_share_extra": [old_h, self.hard_share_extra[t]],
                "best_dev": self.best_dev[t], "best_dev_step": self.best_dev_step[t],
            }
            log.warning("curriculum auto-reweight: %s", ev)
            new_events.append(ev)
        self.events.extend(new_events)
        return new_events

    # -- checkpointing --------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "config_fingerprint": sha256_obj(
                {"stages": [vars(s) | {"mix": dict(s.mix)} for s in self.cfg.stages],
                 "prompts_per_step": self.cfg.prompts_per_step,
                 "hard_pool_join_step": self.cfg.hard_pool_join_step,
                 "hard_share": self.cfg.hard_share, "seed": self.cfg.seed, **self._v2_fingerprint()}
            )[:16],
            "stage_at_last_sample": self.stage_at(max(self.last_sampled_step, 0)).name,
            "last_sampled_step": self.last_sampled_step,
            "samplers": {t: {p: s.state_dict() for p, s in ps.items()} for t, ps in self.samplers.items()},
            "weight_factor": dict(self.weight_factor),
            "hard_share_extra": dict(self.hard_share_extra),
            "zero_adv_hist": {t: list(h) for t, h in self.zero_adv_hist.items()},
            "best_dev": dict(self.best_dev),
            "best_dev_step": dict(self.best_dev_step),
            "last_trigger_step": dict(self.last_trigger_step),
            "events": list(self.events),
        }

    def _v2_fingerprint(self) -> dict[str, Any]:
        """The 2026-11-07 settings, only when they are on: fingerprints of earlier runs stay as they were."""
        out: dict[str, Any] = {}
        dw = self.cfg.difficulty_weights
        if dw.enabled:
            out["difficulty_weights"] = {"bands": [list(b) for b in dw.bands], "unknown": dw.unknown}
        if self.cfg.hard_pool_min_size:
            out["hard_pool_min_size"] = self.cfg.hard_pool_min_size
        return out

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        mine = self.state_dict()["config_fingerprint"]
        if state["config_fingerprint"] != mine:
            raise ValueError(
                f"curriculum config changed since checkpoint ({state['config_fingerprint']} != {mine})"
            )
        for t, ps in state["samplers"].items():
            for p, s in ps.items():
                self.samplers[t][p].load_state_dict(s)
        self.last_sampled_step = int(state["last_sampled_step"])
        self.weight_factor = {t: float(v) for t, v in state["weight_factor"].items()}
        self.hard_share_extra = {t: float(v) for t, v in state["hard_share_extra"].items()}
        w = self.cfg.auto_reweight.window
        self.zero_adv_hist = {t: deque(h, maxlen=w) for t, h in state["zero_adv_hist"].items()}
        self.best_dev = dict(state["best_dev"])
        self.best_dev_step = dict(state["best_dev_step"])
        self.last_trigger_step = dict(state["last_trigger_step"])
        self.events = list(state["events"])
        expected_stage = self.stage_at(max(self.last_sampled_step, 0)).name
        if state.get("stage_at_last_sample") != expected_stage:
            raise ValueError(f"stage pointer mismatch: {state.get('stage_at_last_sample')} vs {expected_stage}")
