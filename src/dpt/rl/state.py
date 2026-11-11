"""Extra trainer state: everything a resume needs beyond model weights and optimizer.

Incident 2 (run 2, step 212): resuming from the step-200 checkpoint dropped
reward by 0.09, and it took 30 steps to recover. The checkpoint had weights
and optimizer state but not the sampler position, the curriculum stage
pointer, the hard pool, the RNG or the LR scheduler. The run restarted stage
A's mix from the first problems and re-ran warmup.

This module makes the omission impossible to miss:

* :class:`ExtraStateManager` holds a *registry* of named stateful components
  (curriculum, partial-rollout buffer, advantage scaler, alert manager,
  scheduler, …). Saving writes all of them plus the RNG snapshot and the
  data hashes to ``<ckpt_dir>/dpt_extra_state.json``. Loading **fails** if a
  registered component is missing from the file or the file has an unknown
  one, unless the caller explicitly allows it.
* :func:`resume_equivalence` is the permanent guard. Run ``N`` steps
  continuously, or ``k`` steps + save + fresh process + load + ``N-k`` steps;
  the loss curves must agree within tolerance. The real-trainer version (20
  steps on the smoke model) runs as a pre-launch check. The toy version is a
  CPU unit test.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from dpt.common.io import read_json, write_json
from dpt.common.logging import get_logger
from dpt.common.seeding import capture_rng_state, restore_rng_state

log = get_logger(__name__)

EXTRA_STATE_FILE = "dpt_extra_state.json"
STATE_VERSION = 1


class Stateful(Protocol):
    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


class ExtraStateManager:
    """Registry + (de)serializer for every non-model piece of trainer state."""

    def __init__(self) -> None:
        self._components: dict[str, Stateful] = {}

    def register(self, name: str, component: Stateful) -> None:
        if name in self._components:
            raise ValueError(f"component {name!r} already registered")
        for attr in ("state_dict", "load_state_dict"):
            if not callable(getattr(component, attr, None)):
                raise TypeError(f"component {name!r} lacks {attr}()")
        self._components[name] = component

    @property
    def names(self) -> list[str]:
        return sorted(self._components)

    def snapshot(self, step: int, *, data_hashes: Mapping[str, str] | None = None,
                 extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "step": int(step),
            "components": {n: c.state_dict() for n, c in self._components.items()},
            "rng": capture_rng_state(),
            "data_hashes": dict(data_hashes or {}),
            "extra": dict(extra or {}),
        }

    def save(self, ckpt_dir: str | os.PathLike, step: int, **kw: Any) -> Path:
        path = Path(ckpt_dir) / EXTRA_STATE_FILE
        write_json(path, self.snapshot(step, **kw))
        log.info("saved extra trainer state (%s) for step %d to %s", ", ".join(self.names), step, path)
        return path

    def restore(self, snap: Mapping[str, Any], *, allow_missing: bool = False,
                expected_data_hashes: Mapping[str, str] | None = None) -> int:
        if snap.get("version") != STATE_VERSION:
            raise ValueError(f"extra-state version {snap.get('version')} != {STATE_VERSION}")
        saved = set(snap["components"])
        mine = set(self._components)
        missing, unknown = mine - saved, saved - mine
        if (missing or unknown) and not allow_missing:
            raise ValueError(
                f"extra-state mismatch: missing in checkpoint {sorted(missing)}, unknown {sorted(unknown)}. "
                "A resume without all components silently changes training (Incident 2)."
            )
        if expected_data_hashes:
            for k, h in expected_data_hashes.items():
                if snap["data_hashes"].get(k) not in (None, h):
                    raise ValueError(f"data {k!r} changed since checkpoint: {snap['data_hashes'][k]} != {h}")
        for n in sorted(mine & saved):
            self._components[n].load_state_dict(snap["components"][n])
        restore_rng_state(snap["rng"])
        self.last_roundtrip_mismatches = self.roundtrip_mismatches(snap)
        if self.last_roundtrip_mismatches:
            log.error("extra state did not round-trip for %s: these components did not apply all of the saved "
                      "state, so the resumed run differs from the original (Incident 2)",
                      self.last_roundtrip_mismatches)
        return int(snap["step"])

    def roundtrip_mismatches(self, snap: Mapping[str, Any]) -> list[str]:
        """Components whose live state differs from ``snap`` (call right after :meth:`restore`).

        A component that ignores part of its saved state passes ``load_state_dict`` silently but
        fails here. The comparison is on JSON-normalized values (tuples vs lists, int keys vs str).
        """
        def norm(x: Any) -> Any:
            return json.loads(json.dumps(x, sort_keys=True, default=str))
        return [n for n in sorted(set(self._components) & set(snap["components"]))
                if norm(self._components[n].state_dict()) != norm(snap["components"][n])]

    def load(self, ckpt_dir: str | os.PathLike, **kw: Any) -> int:
        """Restore every component from ``ckpt_dir``; returns the checkpointed step."""
        return self.restore(read_json(Path(ckpt_dir) / EXTRA_STATE_FILE), **kw)


class DictState:
    """Wrap an opaque dict (e.g. the alert manager's state) as a :class:`Stateful`."""

    def __init__(self, initial: Mapping[str, Any] | None = None) -> None:
        self.value: dict[str, Any] = dict(initial or {})

    def state_dict(self) -> dict[str, Any]:
        return dict(self.value)

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.value = dict(state)


# ----------------------------------------------------------------------------------------------
# Resume-equivalence harness
# ----------------------------------------------------------------------------------------------
class ResumableRun(Protocol):
    """Anything that can take a training step and checkpoint itself."""

    def step(self, step: int) -> float: ...           # returns the loss of that step

    def save(self, ckpt_dir: str, step: int) -> None: ...

    def load(self, ckpt_dir: str) -> int: ...          # returns the checkpointed step


@dataclass
class ResumeReport:
    passed: bool
    max_abs_diff: float
    continuous: list[float] = field(default_factory=list)
    resumed: list[float] = field(default_factory=list)


def resume_equivalence(
    make_run: Callable[[], ResumableRun],
    total_steps: int = 20,
    split_step: int = 10,
    *,
    atol: float = 1e-6,
    workdir: str | None = None,
) -> ResumeReport:
    """Compare ``total_steps`` continuous steps against ``split_step`` + save/load + the rest.

    ``make_run`` must build a *fresh* run (new process state as far as the run
    is concerned). The second half of the resumed run is executed by a
    different object that only sees the checkpoint.
    """
    run_a = make_run()
    curve_a = [run_a.step(s) for s in range(total_steps)]

    tmp_ctx = tempfile.TemporaryDirectory() if workdir is None else None
    ckpt = workdir or tmp_ctx.name  # type: ignore[union-attr]
    try:
        run_b = make_run()
        curve_b = [run_b.step(s) for s in range(split_step)]
        run_b.save(ckpt, split_step - 1)
        del run_b
        run_c = make_run()
        last = run_c.load(ckpt)
        curve_b += [run_c.step(s) for s in range(last + 1, total_steps)]
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()
    diff = float(np.max(np.abs(np.asarray(curve_a) - np.asarray(curve_b)))) if curve_a else 0.0
    return ResumeReport(passed=diff <= atol, max_abs_diff=diff, continuous=curve_a, resumed=curve_b)
