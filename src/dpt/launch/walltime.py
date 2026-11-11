"""Stop a training run cleanly before the batch scheduler's wall-clock limit.

Jobs on the cluster end after at most 48 h. The 9B RL run (420 steps) needs more than one job, so each
job must stop at a step boundary *with a fresh checkpoint* and the next job must resume exactly
there (``trainer.resume_mode=auto`` + the extra state of :mod:`dpt.rl.state`; the resume test in
``scripts/05_rl_smoke.sh`` checks that this is lossless). Being killed by Slurm instead would lose
everything since the last periodic save (up to ``save_freq`` steps) and could leave a half-written
checkpoint.

The job script exports the deadline::

    export DPT_DEADLINE=$(python -m dpt.launch.walltime --slurm-deadline --margin-s 900)

and :class:`WallClock` decides after every step whether one more step and a checkpoint still fit:

    stop  iff  now + safety × (slowest of the last few steps) + checkpoint time  >  deadline

The trainer then saves, writes ``WALLTIME.json`` and ends the fit loop (``DomainRLTrainer._stop``).
``scripts/slurm/rl.slurm`` resubmits the job unless the run finished or an alert paused it.

No verl import: unit-tested on CPU (``tests/launch/test_walltime.py``).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

DEADLINE_ENV = "DPT_DEADLINE"            # unix seconds; unset = no limit
MARGIN_ENV = "DPT_DEADLINE_MARGIN_S"     # set by the job script: re-read the limit from Slurm (see WallClock)
STOP_AT_ENV = "DPT_STOP_AT_STEP"         # save and stop once this step is done (HOLD.json); unset = never


def parse_slurm_duration(text: str) -> int:
    """Seconds in a Slurm duration: ``MM``, ``MM:SS``, ``HH:MM:SS``, ``D-HH``, ``D-HH:MM``, ``D-HH:MM:SS``."""
    text = text.strip()
    m = re.fullmatch(r"(?:(\d+)-)?(\d+)(?::(\d+))?(?::(\d+))?", text)
    if not m or text in ("", "UNLIMITED", "INVALID"):
        raise ValueError(f"not a Slurm duration: {text!r}")
    days, a, b, c = m.groups()
    days = int(days or 0)
    parts = [int(x) for x in (a, b, c) if x is not None]
    if days:                               # D-HH[:MM[:SS]]
        h, mi, s = (parts + [0, 0])[:3]
    elif len(parts) == 3:                  # HH:MM:SS
        h, mi, s = parts
    elif len(parts) == 2:                  # MM:SS
        h, (mi, s) = 0, parts
    else:                                  # MM
        h, mi, s = 0, parts[0], 0
    return ((days * 24 + h) * 60 + mi) * 60 + s


def slurm_time_left(job_id: str | None = None, *, runner=subprocess.run) -> int:
    """Seconds left in the current Slurm job (``squeue -h -j $SLURM_JOB_ID -o %L``)."""
    job_id = job_id or os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("not inside a Slurm job (SLURM_JOB_ID unset)")
    out = runner(["squeue", "-h", "-j", str(job_id), "-o", "%L"], capture_output=True, text=True, check=True)
    return parse_slurm_duration(out.stdout.strip().splitlines()[0])


def deadline_from_env(env=os.environ) -> float | None:
    raw = env.get(DEADLINE_ENV)
    return float(raw) if raw else None


@dataclass
class WallClock:
    """Per-step decision: is there time for one more step plus a checkpoint before the deadline?"""

    deadline: float | None
    safety: float = 1.5                 # steps vary (partial rollout, dev validation every 20 steps)
    save_estimate_s: float = 900.0      # until a save has been timed; a 9B save with optimizer ≈ 135 GB
    window: int = 3
    # Re-read the deadline (e.g. from Slurm) at most every refresh_every_s: site staff can extend a running
    # job's time limit, and a deadline fixed at start would stop the run at the old limit anyway.
    refresh: Callable[[float], float | None] | None = None
    refresh_every_s: float = 1800.0
    _refreshed: float | None = field(default=None, repr=False)
    _steps: deque = field(default_factory=lambda: deque(maxlen=3), repr=False)
    _last: float | None = field(default=None, repr=False)
    _save_s: float | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._steps = deque(maxlen=self.window)

    @classmethod
    def from_env(cls, env=os.environ, **kw) -> "WallClock":
        """The deadline the job script exported; inside Slurm with a margin set, refreshed from ``squeue``."""
        margin, job = env.get(MARGIN_ENV), env.get("SLURM_JOB_ID")
        if margin and job and "refresh" not in kw:
            kw["refresh"] = lambda now: now + slurm_time_left(job) - float(margin)
        return cls(deadline_from_env(env), **kw)

    def _maybe_refresh(self, now: float) -> None:
        if self.refresh is None or (self._refreshed is not None and now - self._refreshed < self.refresh_every_s):
            return
        self._refreshed = now
        try:
            new = self.refresh(now)
        except Exception:                 # squeue unreachable for a moment: keep the deadline we have
            return
        if new is not None:
            self.deadline = float(new)

    def observe_step(self, now: float) -> None:
        """Call once per finished step. The first call only starts the clock (it includes start-up)."""
        if self._last is not None:
            self._steps.append(max(0.0, now - self._last))
        self._last = now

    def observe_save(self, seconds: float) -> None:
        self._save_s = max(seconds, self._save_s or 0.0)

    def needed_s(self) -> float | None:
        if not self._steps:
            return None
        save = 1.2 * self._save_s if self._save_s is not None else self.save_estimate_s
        return self.safety * max(self._steps) + save

    def should_stop(self, now: float) -> tuple[bool, str]:
        self._maybe_refresh(now)
        if self.deadline is None:
            return False, ""
        left = self.deadline - now
        need = self.needed_s()
        if need is None:                  # no full step timed yet: stop only if already past the line
            return (left <= self.save_estimate_s,
                    f"{left / 60:.0f} min left before the job's wall-clock limit, no step timed yet")
        if left < need:
            return True, (f"{left / 60:.0f} min left before the job's wall-clock limit; the next step and "
                          f"a checkpoint need ~{need / 60:.0f} min (slowest recent step "
                          f"{max(self._steps) / 60:.1f} min)")
        return False, ""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--slurm-deadline", action="store_true",
                   help="print the unix time at which training must be stopped (job end − margin)")
    p.add_argument("--margin-s", type=int, default=900,
                   help="reserved for teardown after training stops (Ray shutdown, resubmission)")
    a = p.parse_args()
    if not a.slurm_deadline:
        p.error("nothing to do (use --slurm-deadline)")
    import time

    print(int(time.time() + slurm_time_left() - a.margin_s))


if __name__ == "__main__":
    main()
