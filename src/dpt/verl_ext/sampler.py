"""Curriculum-driven sampler for verl's RL dataloader (report §三段课程, §多任务 调整-5/6, Incident 2).

verl 0.9.1 builds the train dataloader with ``create_rl_sampler`` (seeded random
or sequential only) and has no curriculum hook. ``DomainRLTrainer._init_dataloader``
swaps in this sampler. It turns :class:`dpt.rl.curriculum.Curriculum` draws
(task, pool, uid) into dataset row indices, ``prompts_per_step`` at a time:

    step s  →  curriculum.sample(s)  →  128 draws  →  128 row indices (yielded one by one)

In verl's async modes the dataloader batch size is forced to 1 and prompts are
pulled as rollout slots free up, so a step boundary here is "every
``prompts_per_step`` prompts drawn". It runs ahead of the trainer's step by at
most the async warm-up depth. Stage switches and hard-pool admission are
therefore exact up to a few prompts. That is immaterial at a 120-step stage
length, but it is the reason this sampler counts its own steps.

Resume exactness: ``state_dict`` captures the step counter, the undrained rest
of the current step and the full curriculum state (pool permutations, cursors,
reweighting history). The trainer stores it in ``dpt_extra_state.json`` with
every checkpoint. Run the RL dataloader with ``num_workers=0`` (set in
``rl_27b.yaml``) so no prefetched indices are lost at a checkpoint boundary.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterator, Mapping

from torch.utils.data import Sampler

from dpt.common.logging import get_logger
from dpt.rl.curriculum import Curriculum, CurriculumConfig

log = get_logger(__name__)


def _column(dataset: Any, name: str) -> list[Any]:
    df = getattr(dataset, "dataframe", dataset)
    col = df[name]
    return col.tolist() if hasattr(col, "tolist") else list(col)


def build_pools(dataset: Any) -> tuple[dict[str, dict[str, list[str]]], dict[str, int]]:
    """Read ``data_source`` (task) and ``extra_info.{uid,pool}`` for every row of verl's RLHFDataset."""
    pools, index, _ = build_pools_and_rates(dataset)
    return pools, index


def build_pools_and_rates(dataset: Any) -> tuple[dict[str, dict[str, list[str]]], dict[str, int],
                                                 dict[str, float | None]]:
    """:func:`build_pools` plus each uid's pass rate (``extra_info.pass_at_8_medium``; None when absent), for the
    curriculum's difficulty weights (2026-11-07)."""
    tasks = _column(dataset, "data_source")
    infos = _column(dataset, "extra_info")
    pools: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"main": [], "hard": []})
    index: dict[str, int] = {}
    rates: dict[str, float | None] = {}
    for i, (t, info) in enumerate(zip(tasks, infos)):
        info = dict(info or {})
        uid = str(info.get("uid", f"row-{i}"))
        if uid in index:
            log.warning("duplicate uid %s in RL data (rows %d and %d); keeping the first", uid, index[uid], i)
            continue
        index[uid] = i
        pools[str(t)][str(info.get("pool", "main"))].append(uid)
        r = info.get("pass_at_8_medium")
        rates[uid] = float(r) if isinstance(r, (int, float)) and r == r else None
    return {k: dict(v) for k, v in pools.items()}, index, rates


class CurriculumSampler(Sampler[int]):
    def __init__(self, dataset: Any, curriculum_cfg: Mapping[str, Any], total_steps: int) -> None:
        pools, self.index, rates = build_pools_and_rates(dataset)
        self.curriculum = Curriculum(CurriculumConfig.from_dict(dict(curriculum_cfg)), pools, rates)
        self.prompts_per_step = int(curriculum_cfg["prompts_per_step"])
        self.total_steps = int(total_steps)
        self.next_step = 0
        self.pending: list[int] = []
        sizes = {t: {p: len(u) for p, u in v.items()} for t, v in pools.items()}
        log.info("CurriculumSampler: pools %s", sizes)

    def __len__(self) -> int:
        # Large enough for the whole run; verl bounds training by total_training_steps.
        return (self.total_steps + 8) * self.prompts_per_step

    def _refill(self) -> None:
        draws = self.curriculum.sample(self.next_step)
        self.pending = [self.index[d.uid] for d in draws]
        self.next_step += 1

    def __iter__(self) -> Iterator[int]:
        # Continue from the saved position; a new epoch does not restart the curriculum.
        for _ in range(len(self)):
            if not self.pending:
                self._refill()
            yield self.pending.pop(0)

    # -- Incident 2: exact resume ----------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {"next_step": self.next_step, "pending": list(self.pending),
                "curriculum": self.curriculum.state_dict()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.next_step = int(state["next_step"])
        self.pending = [int(i) for i in state["pending"]]
        self.curriculum.load_state_dict(state["curriculum"])
