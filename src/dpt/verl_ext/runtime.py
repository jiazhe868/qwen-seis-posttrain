"""Per-process runtime context shared by the verl adapter pieces (trainer, replay buffer, sampler).

verl instantiates our classes by import path, deep inside its own code. We
cannot pass them constructor arguments. Instead the launcher
(``dpt.launch.rl``) writes the fully resolved dpt config to the run directory
and exports its path as ``DPT_RESOLVED_CONFIG`` to every Ray process. This
module loads that file once per process and owns the stateful components that
must survive checkpoint/resume:

* ``scaler``      TaskAdvantageScaler (running per-task std; T3 scaling)
* ``normalizer``  TaskLossNormalizer (fixed per-task token budgets)
* ``curriculum``  set by the sampler once the dataset is known
* ``alerts``      the monitoring alert engine

Everything here is importable without verl (helpers operate on torch/numpy),
so it is unit-tested on CPU.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dpt.common.io import read_yaml
from dpt.common.logging import get_logger
from dpt.rl.advantage import TaskAdvantageScaler, TaskLossNormalizer

log = get_logger(__name__)

ENV_RESOLVED_CONFIG = "DPT_RESOLVED_CONFIG"


@dataclass
class Runtime:
    cfg: dict[str, Any]
    scaler: TaskAdvantageScaler
    normalizer: TaskLossNormalizer | None
    curriculum: Any = None                     # dpt.rl.curriculum.Curriculum, set by the sampler
    alerts: Any = None                         # dpt.monitoring alert engine (built lazily)
    last_filter_stats: dict[str, float] = field(default_factory=dict)
    pause_requested: bool = False
    pause_reason: str = ""

    def nominal_mix(self, step: int) -> dict[str, float]:
        """The stage's nominal task mix: it defines the fixed per-task budgets (never the effective mix)."""
        if self.curriculum is not None:
            return dict(self.curriculum.stage_at(step).mix)
        for st in self.cfg["curriculum"]["stages"]:
            if st["start"] <= step < st["end"]:
                return dict(st["mix"])
        return dict(self.cfg["curriculum"]["stages"][-1]["mix"])

    def mix_for_batch(self, step: int, present_tasks: set[str]) -> dict[str, float]:
        """Nominal mix to use for a batch, robust to stage boundaries.

        The sampler runs a few prompts ahead of the trainer (async warm-up/refill). Right after a
        stage switch, a batch trained at step 119 can therefore contain T3 prompts drawn for
        stage B, while stage A gives T3 weight 0. Use the newest sampled stage's mix whenever the
        current one would give a present task zero weight. That is a one-step budget change,
        whereas the alternative would crash the step.
        """
        mix = self.nominal_mix(step)
        if all(mix.get(t, 0.0) > 0 for t in present_tasks):
            return mix
        if self.curriculum is not None and self.curriculum.last_sampled_step >= 0:
            newer = dict(self.curriculum.stage_at(self.curriculum.last_sampled_step).mix)
            if all(newer.get(t, 0.0) > 0 for t in present_tasks):
                return newer
        log.warning("tasks %s have zero weight in the nominal mix at step %d; using a uniform fallback",
                    sorted(t for t in present_tasks if mix.get(t, 0.0) <= 0), step)
        out = dict(mix)
        for t in present_tasks:
            if out.get(t, 0.0) <= 0:
                out[t] = 1.0 / max(1, len(present_tasks))
        return out


def build_runtime(cfg: dict[str, Any]) -> Runtime:
    rl = cfg["rl"]
    adv = rl["advantage"]
    scaler = TaskAdvantageScaler(
        scaled_tasks=tuple(adv.get("scale_tasks") or ()),
        momentum=float(adv.get("scale_momentum", 0.9)),
        min_std=float(adv.get("scale_min_std", 0.05)),
        target_std=adv.get("scale_target_std"),
        max_gain=float(adv.get("scale_max_gain", 4.0)),
    )
    tn = rl["loss"]["task_normalization"]
    normalizer = None
    if tn.get("enabled", True):
        normalizer = TaskLossNormalizer(
            response_tokens=dict(tn["response_tokens"]),
            group_size=int(rl["group_size"]),
            prompts_per_step=int(rl["prompts_per_step"]),
            num_updates=int(rl["num_updates_per_step"]),
            constant=1.0,                       # verl loss_agg_mode=token-sum divides by nothing
            weights=tn.get("weights"),
        )
    rt = Runtime(cfg=cfg, scaler=scaler, normalizer=normalizer)
    rt.alerts = _build_alerts(cfg)
    return rt


def _build_alerts(cfg: dict[str, Any]) -> Any:
    try:
        from dpt.monitoring.alerts import AlertManager

        return AlertManager.from_config(cfg.get("monitoring", {}))
    except Exception as e:  # monitoring must never block training start-up; log loudly instead
        log.error("alert engine unavailable (%s); training will run WITHOUT auto-pause", e)
        return None


_RUNTIME: Runtime | None = None


def get_runtime() -> Runtime:
    """Load (once per process) the resolved config named by ``DPT_RESOLVED_CONFIG``."""
    global _RUNTIME
    if _RUNTIME is None:
        path = os.environ.get(ENV_RESOLVED_CONFIG)
        if not path:
            raise RuntimeError(
                f"{ENV_RESOLVED_CONFIG} is not set; launch RL through `python -m dpt.launch.rl` "
                "so every Ray process receives the resolved dpt config."
            )
        _RUNTIME = build_runtime(read_yaml(path))
        log.info("dpt runtime loaded from %s", path)
    return _RUNTIME


def set_runtime(rt: Runtime | None) -> None:
    """Testing hook."""
    global _RUNTIME
    _RUNTIME = rt


# ---------------------------------------------------------------------------------------------
# Small conversion helpers for TransferQueue / TensorDict payloads
# ---------------------------------------------------------------------------------------------
def row_sums(x: Any) -> np.ndarray:
    """Per-row sums of a padded tensor, a nested (jagged) tensor, or a list of 1-D tensors/arrays."""
    import torch

    if isinstance(x, torch.Tensor):
        if getattr(x, "is_nested", False):
            return np.array([float(t.float().sum()) for t in x.unbind()], dtype=np.float64)
        return x.float().reshape(x.shape[0], -1).sum(-1).cpu().numpy().astype(np.float64)
    return np.array([float(np.asarray(t, dtype=np.float64).sum()) for t in x], dtype=np.float64)


def to_str_list(x: Any) -> list[str]:
    """Strings from a NonTensorStack / numpy object array / list (``None`` → "")."""
    if hasattr(x, "tolist"):
        x = x.tolist()
    return ["" if v is None else str(v) for v in x]
