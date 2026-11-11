"""Metric sinks keyed by step: an always-on JSONL file plus optional wandb.

The JSONL sink is the source of truth for offline triage (report §排查方法:
"冻结现场"). Every step's metrics and alerts land in an append-only file next
to the checkpoint, so an incident can be replayed without a dashboard.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from dpt.common.logging import get_logger

log = get_logger(__name__)


class MetricsSink(Protocol):
    def log(self, step: int, metrics: Mapping[str, Any]) -> None: ...

    def close(self) -> None: ...


def _clean(v: Any) -> Any:
    if isinstance(v, float) and not math.isfinite(v):
        return str(v)
    if hasattr(v, "item"):
        try:
            return _clean(v.item())
        except Exception:
            return str(v)
    return v


class JsonlSink:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")

    def log(self, step: int, metrics: Mapping[str, Any]) -> None:
        rec = {"step": int(step), **{k: _clean(v) for k, v in metrics.items()}}
        self._f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


class WandbSink:
    """Thin wandb wrapper; construction fails loudly if wandb is requested but not installed."""

    def __init__(self, project: str, run_name: str, config: Mapping[str, Any] | None = None, **kwargs: Any):
        import wandb  # optional dependency

        self._wandb = wandb
        self._run = wandb.init(project=project, name=run_name, config=dict(config or {}), **kwargs)

    def log(self, step: int, metrics: Mapping[str, Any]) -> None:
        self._wandb.log({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, step=int(step))

    def close(self) -> None:
        self._run.finish()


class MultiSink:
    def __init__(self, sinks: Sequence[MetricsSink]):
        self.sinks = list(sinks)

    def log(self, step: int, metrics: Mapping[str, Any]) -> None:
        for s in self.sinks:
            try:
                s.log(step, metrics)
            except Exception as e:  # a dashboard outage must never stop training
                log.warning("sink %s failed: %s", type(s).__name__, e)

    def close(self) -> None:
        for s in self.sinks:
            s.close()
