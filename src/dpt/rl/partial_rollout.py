"""Partial rollout: suspend very long generations and continue them in the next step.

Why (report §单步时间预算)
--------------------------
In the baseline, 90% of a step's tokens were generated in the first 4
minutes, and the next 5 minutes went to the longest 3% of responses. Decode
time is set by the slowest sequence while most of the GPUs sit idle.
Partial rollout caps each step's generation at ``chunk_tokens`` (16K) new
tokens per sample. A sample that has not finished is suspended with its
tokens and log-probs and continues next step from where it stopped, up to
the hard cap of 24K total. Generation time fell from 9.0 to 4.6 min
(together with MTP speculative decoding).

The price is off-policy tokens. A resumed sample's first 16K tokens came from
policy version ``v`` and the rest from ``v+1``. The PPO ratio uses the
trainer's current ``old_logp``, so those early tokens are off-policy. That
is why every token's *rollout* log-prob is stored next to the version that
produced it. TIS (``dpt.rl.tis``) then corrects old-version and numerical
mismatch together, and the version segments let the monitor report the
off-policy token fraction and staleness.

Groups, not samples
-------------------
GRPO needs all 8 samples of a prompt to form the baseline. A group with any
unfinished member is held back as a whole: finished members wait in the
buffer, and the group trains in the step where its last member finishes.
Groups whose oldest token exceeds ``max_staleness`` policy versions are
evicted rather than trained on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping


@dataclass
class SampleState:
    sample_id: str
    response_ids: list[int] = field(default_factory=list)
    rollout_logprobs: list[float] = field(default_factory=list)
    segments: list[list[int]] = field(default_factory=list)   # [[policy_version, n_tokens], ...]
    finished: bool = False       # hit EOS, or hit the hard cap
    truncated: bool = False      # hit the hard cap without EOS (excluded from loss downstream)

    def append(self, ids: list[int], logprobs: list[float], version: int) -> None:
        if len(ids) != len(logprobs):
            raise ValueError("ids and logprobs must have the same length")
        if not ids:
            return
        self.response_ids.extend(ids)
        self.rollout_logprobs.extend(logprobs)
        if self.segments and self.segments[-1][0] == version:
            self.segments[-1][1] += len(ids)
        else:
            self.segments.append([version, len(ids)])

    @property
    def oldest_version(self) -> int | None:
        return min(v for v, _ in self.segments) if self.segments else None

    def token_versions(self) -> list[int]:
        """Policy version that produced each response token."""
        out: list[int] = []
        for v, n in self.segments:
            out.extend([v] * n)
        return out


@dataclass
class PendingGroup:
    group_id: str
    uid: str
    task: str
    prompt_ids: list[int]
    samples: list[SampleState]
    first_step: int

    @property
    def ready(self) -> bool:
        return all(s.finished for s in self.samples)

    def staleness(self, current_version: int) -> int:
        olds = [s.oldest_version for s in self.samples if s.oldest_version is not None]
        return current_version - min(olds) if olds else 0


@dataclass(frozen=True)
class ResumeRequest:
    group_id: str
    sample_id: str
    input_ids: list[int]          # prompt + response so far; the engine continues from here
    max_new_tokens: int


class PartialRolloutBuffer:
    """Holds groups with unfinished samples across RL steps.

    Per-step protocol (driver side)::

        # 1. continue suspended samples first (they are oldest), then new prompts
        reqs = buf.resume_requests()
        ...generate reqs with max_tokens=req.max_new_tokens...
        buf.record_continuation(group_id, sample_id, new_ids, new_logps, version, hit_eos)
        # 2. fresh groups from this step's prompts
        for g in fresh_groups: buf.admit(g)          # returns nothing; ready groups go to the ready list
        # 3. evict groups that got too stale, then take what is trainable
        stale = buf.evict_stale(version)
        batch = buf.pop_ready()
    """

    def __init__(self, chunk_tokens: int = 16384, max_response: int = 24576, max_staleness: int = 1) -> None:
        if chunk_tokens <= 0 or max_response < chunk_tokens:
            raise ValueError("need 0 < chunk_tokens <= max_response")
        self.chunk_tokens = chunk_tokens
        self.max_response = max_response
        self.max_staleness = max_staleness
        self.pending: dict[str, PendingGroup] = {}
        self.ready: list[PendingGroup] = []
        self.evicted_total = 0

    # -- building samples from one generation call ----------------------------------------
    def finalize_chunk(self, sample: SampleState, hit_eos: bool) -> None:
        """Set ``finished`` / ``truncated`` after a generation chunk was appended."""
        if hit_eos:
            sample.finished, sample.truncated = True, False
        elif len(sample.response_ids) >= self.max_response:
            sample.finished, sample.truncated = True, True
        else:
            sample.finished = False  # suspended: continues next step

    def admit(self, group: PendingGroup) -> None:
        if group.group_id in self.pending:
            raise ValueError(f"group {group.group_id} already pending")
        if group.ready:
            self.ready.append(group)
        else:
            self.pending[group.group_id] = group

    def resume_requests(self) -> list[ResumeRequest]:
        reqs = []
        for g in self.pending.values():
            for s in g.samples:
                if not s.finished:
                    budget = min(self.chunk_tokens, self.max_response - len(s.response_ids))
                    reqs.append(ResumeRequest(g.group_id, s.sample_id, g.prompt_ids + s.response_ids, budget))
        return reqs

    def record_continuation(
        self, group_id: str, sample_id: str, new_ids: list[int], new_logprobs: list[float],
        version: int, hit_eos: bool,
    ) -> None:
        g = self.pending[group_id]
        s = next(x for x in g.samples if x.sample_id == sample_id)
        if s.finished:
            raise ValueError(f"sample {sample_id} already finished")
        s.append(new_ids, new_logprobs, version)
        self.finalize_chunk(s, hit_eos)
        if g.ready:
            self.ready.append(self.pending.pop(group_id))

    def evict_stale(self, current_version: int) -> list[PendingGroup]:
        out = [g for g in self.pending.values() if g.staleness(current_version) > self.max_staleness]
        for g in out:
            del self.pending[g.group_id]
        self.evicted_total += len(out)
        return out

    def pop_ready(self) -> list[PendingGroup]:
        out, self.ready = self.ready, []
        return out

    # -- diagnostics ----------------------------------------------------------------------
    @staticmethod
    def off_policy_token_fraction(groups: Iterable[PendingGroup], current_version: int) -> float:
        """Fraction of response tokens produced by a policy older than ``current_version``."""
        old = tot = 0
        for g in groups:
            for s in g.samples:
                for v, n in s.segments:
                    tot += n
                    old += n if v < current_version else 0
        return old / tot if tot else 0.0

    def stats(self) -> dict[str, float]:
        n_unfinished = sum(1 for g in self.pending.values() for s in g.samples if not s.finished)
        return {
            "partial/pending_groups": float(len(self.pending)),
            "partial/suspended_samples": float(n_unfinished),
            "partial/evicted_total": float(self.evicted_total),
        }

    # -- checkpointing --------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {
            "chunk_tokens": self.chunk_tokens,
            "max_response": self.max_response,
            "max_staleness": self.max_staleness,
            "pending": [asdict(g) for g in self.pending.values()],
            "ready": [asdict(g) for g in self.ready],
            "evicted_total": self.evicted_total,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.chunk_tokens = int(state["chunk_tokens"])
        self.max_response = int(state["max_response"])
        self.max_staleness = int(state["max_staleness"])

        def _g(d: Mapping[str, Any]) -> PendingGroup:
            samples = [SampleState(**s) for s in d["samples"]]
            return PendingGroup(**{**d, "samples": samples})

        self.pending = {d["group_id"]: _g(d) for d in state["pending"]}
        self.ready = [_g(d) for d in state["ready"]]
        self.evicted_total = int(state["evicted_total"])
