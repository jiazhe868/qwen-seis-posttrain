"""RL learning-rate schedule (report §算法配置 学习率, ablation A8).

    steps [0, warmup)                 linear warmup to base_lr
    steps [warmup, total - final)     base_lr = 1e-6 (constant)
    steps [total - final, total)      final_lr = 5e-7 (stage C, "收尾")

Why so low and so flat: the 2e-6 tuned on 9B made 27B entropy oscillate 3×
harder, and D-Num stalled after step 60 (A8). LR is one of the conclusions that
did *not* transfer from the small model. The drop in the last 120 steps
pairs with stage C's higher general-replay share to win back forgetting.

The schedule is a pure function of the step, plus a ``state_dict`` that
records its parameters, so a resumed run cannot silently re-run warmup
(Incident 2: "warmup 也重新跑了一遍").
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass
class RLLRSchedule:
    base_lr: float = 1e-6
    warmup_steps: int = 10
    total_steps: int = 420
    final_lr: float = 5e-7
    final_phase_steps: int = 120
    decay: str = "step"           # "step" (as in the report) or "linear" ramp over the final phase
    last_step: int = -1

    def lr(self, step: int) -> float:
        if step < 0:
            raise ValueError("step must be >= 0")
        if step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps
        start_final = self.total_steps - self.final_phase_steps
        if step < start_final:
            return self.base_lr
        if self.decay == "step":
            return self.final_lr
        if self.decay == "linear":
            frac = min((step - start_final) / max(self.final_phase_steps, 1), 1.0)
            return self.base_lr + (self.final_lr - self.base_lr) * frac
        raise ValueError(f"unknown decay {self.decay!r}")

    def step(self, step: int) -> float:
        """Record that ``step`` ran and return its LR (``last_step`` is what gets checkpointed)."""
        self.last_step = step
        return self.lr(step)

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for k, v in state.items():
            setattr(self, k, v)
