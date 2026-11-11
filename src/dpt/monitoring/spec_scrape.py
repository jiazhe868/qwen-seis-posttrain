"""MTP draft acceptance from vLLM's Prometheus counters (fallback for ``rollout/spec_accept_rate``).

Why this exists: verl 0.9.1 computes ``rollout/spec_accept_rate`` from a per-request field,
``RequestOutput.metrics.request_spec_decode_stats``. vLLM 0.24.0, the version verl 0.9.1 pins, does
not have that field. verl then skips the metric, so the Incident 5 alert (``mtp_acceptance`` < 0.6)
would never fire. Found by the workstation smoke run: vLLM was started with
``--speculative_config {"method": "mtp", ...}``, but the metric never appeared.

vLLM does count drafted and accepted tokens per engine, as Prometheus counters on each rollout
server's ``/metrics`` endpoint. That requires stats logging, so set
``actor_rollout_ref.rollout.disable_log_stats: false`` (rl_27b.yaml does). The trainer polls every
replica once per step. The acceptance rate over the step is

    Δ accepted tokens / Δ draft tokens       (summed over replicas and engines)

which is the per-draft-token acceptance the report quotes (0.71 healthy, 0.46 in Incident 5).
Counters are cumulative and survive vLLM sleep/wake, so the step's delta is exact. They are not
cleared by a prefix-cache reset either. A replica that does not answer is skipped for that step.
"""

from __future__ import annotations

import re
import urllib.request
from typing import Iterable

from dpt.common.logging import get_logger

log = get_logger(__name__)

ACCEPTED = "vllm:spec_decode_num_accepted_tokens_total"
DRAFTED = "vllm:spec_decode_num_draft_tokens_total"
_SAMPLE = re.compile(r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+(?P<value>\S+)")


def parse_counters(text: str, names: Iterable[str] = (ACCEPTED, DRAFTED)) -> dict[str, float]:
    """Sum each named counter over all of its label sets in a Prometheus text exposition."""
    want = set(names)
    out = dict.fromkeys(want, 0.0)
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE.match(line)
        if m and m.group("name") in want:
            try:
                out[m.group("name")] += float(m.group("value"))
            except ValueError:
                pass
    return out


class SpecDecodeScraper:
    """Polls ``http://<addr>/metrics`` on every rollout replica; returns the acceptance since last poll."""

    def __init__(self, addresses: Iterable[str], *, timeout_s: float = 2.0):
        self.addresses = list(addresses)
        self.timeout_s = timeout_s
        self._prev: dict[str, dict[str, float]] = {}
        self._warned: set[str] = set()

    def _fetch(self, addr: str) -> dict[str, float] | None:
        try:
            with urllib.request.urlopen(f"http://{addr}/metrics", timeout=self.timeout_s) as r:
                return parse_counters(r.read().decode("utf-8", "replace"))
        except Exception as e:  # monitoring must never break the step
            if addr not in self._warned:
                log.warning("cannot read vLLM metrics from %s (%s); acceptance skips it", addr, e)
                self._warned.add(addr)
            return None

    def poll(self) -> float | None:
        """Acceptance rate over all replicas since the previous poll, or None if nothing was drafted.

        The first poll only records the baseline (it would otherwise include the counters' whole
        history, e.g. validation generations before step 1).
        """
        accepted = drafted = 0.0
        for addr in self.addresses:
            now = self._fetch(addr)
            if now is None:
                continue
            prev = self._prev.get(addr)
            self._prev[addr] = now
            if prev is None:
                continue
            accepted += max(0.0, now[ACCEPTED] - prev[ACCEPTED])
            drafted += max(0.0, now[DRAFTED] - prev[DRAFTED])
        return accepted / drafted if drafted > 0 else None
