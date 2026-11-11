"""Map verl / dpt.rl metric names onto the canonical keys in :mod:`dpt.monitoring.keys`.

The alert rules and panels only know canonical names. verl names things its
own way (``timing_s/gen``, ``rollout/spec_accept_rate``), and some canonical
values are derived (``dev/d_num`` from the dev gold accuracy). This single
function is the translation, called by the trainer once per step before the
alert engine runs. Keeping it here, not inside the trainer, means it is
unit-tested on CPU and easy to update when verl renames a metric.

Unknown/missing source keys are simply skipped. A rule whose metric is absent
does not fire, so the README's "which alerts are live" table must be kept honest.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from dpt.common.types import TASKS
from dpt.monitoring import keys as K

# verl files the "core" variable of each data source (acc, else reward) under val-core/ and every
# other reward_extra_info key (our score, gold_score, ...) under val-aux/ (v1 trainer_base.py ~1358).
_VAL_RE = re.compile(r"^val-(?:core|aux)/(?P<task>[^/]+)/(?P<var>[^/]+)/mean@\d+$")


def to_canonical(metrics: Mapping[str, Any], *, tail_bucket: str = "12k+") -> dict[str, float]:
    """Return canonical aliases derived from ``metrics`` (the input dict is not modified)."""
    out: dict[str, float] = {}

    def put(k: str, v: Any) -> None:
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            pass

    # numerics / systems
    tail = metrics.get(f"mismatch/pos/{tail_bucket}/log_ratio")
    if tail is not None:
        put(K.LOG_PPL_DIFF_TAIL, abs(float(tail)))       # |log-ppl diff| restricted to positions ≥ 12K
    # verl's per-request metric when the backend reports it; else the trainer's reading of vLLM's
    # Prometheus counters (vLLM 0.24 has no per-request stats, see dpt.monitoring.spec_scrape).
    for src in ("rollout/spec_accept_rate", "rollout/spec_accept_rate_counters"):
        if src in metrics:
            put(K.MTP_ACCEPTANCE, metrics[src])
            break
    # verl's actor metrics arrive renamed with an "actor/" prefix. The value is reduced over ranks
    # (a mean), so it understates the worst rank; the 72/76 GB thresholds keep a margin for that.
    for src in ("actor/perf/max_memory_reserved_gb", "perf/max_memory_reserved_gb",
                "actor/perf/max_memory_allocated_gb", "perf/max_memory_allocated_gb"):
        if src in metrics:                               # reserved ≥ allocated: the conservative one wins
            put(K.MEMORY_PEAK_MAX_RANK, metrics[src])
            break
    if "timing_s/gen" in metrics:
        put(K.ROLLOUT_TIME, metrics["timing_s/gen"])

    # per-task: zero-advantage group fraction from the replay-buffer filter
    for t in TASKS:
        src = f"filter/{t}/zero_adv_group_frac"
        if src in metrics:
            put(K.task_key(K.TASK_ZERO_ADV, t), metrics[src])

    # validation (dev) → proxy / gold / dev D-Num
    for k, v in metrics.items():
        m = _VAL_RE.match(k)
        if not m:
            continue
        task, var = m.group("task"), m.group("var")
        if var == "score":
            put(K.PROXY.format(task=task), v)
        elif var == "gold_score" and float(v) >= 0:      # −1 = "no in-loop gold for this task" (T3)
            put(K.GOLD.format(task=task), v)
        elif var == "t2_infra_error" and task == "t2_code":
            # the sandbox fails closed: an ungraded sample scores 0, so this rate is silent in the rewards
            put(K.SANDBOX_INFRA_ERROR, v)
    gold_t1 = out.get(K.GOLD.format(task="t1_num"))
    if gold_t1 is not None:
        put(K.DEV_D_NUM, 100.0 * gold_t1)                # dev D-Num on the independent (gold) grader
    return out
