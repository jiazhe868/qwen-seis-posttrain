"""Replay buffer that drops zero-advantage groups *before* old-logprob recomputation.

Plugged in via ``trainer.v1.sampler.custom_sampler.{path,name}``.

What and why (report §单步时间预算; ablation A10)
-------------------------------------------------
A group whose 8 samples all got the same reward has advantage 0 for every
token. It contributes nothing to the policy gradient, but it still costs an
old-logprob forward and a full forward/backward. Dropping these groups early
removed ~35% of training tokens and saved 4.4 min/step, with a learning curve
inside the CI of the unfiltered run. The precondition is that the loss is
normalized by a *fixed constant*, not by the current batch's token count.
We guarantee that: ``loss_agg_mode=token-sum`` plus fixed per-task budgets
folded into advantages (``dpt.rl.pipeline``). Otherwise filtering would
silently rescale the gradient.

Why a custom buffer instead of verl's ``algorithm.filter_groups``: verl's
built-in filter is DAPO "dynamic sampling". It *refills* the batch with fresh
prompts until it is full again, which spends more generation to keep the batch
size constant. The report instead *shrinks* the batch. verl then pads it back
to a multiple of ``lcm(dp, mini_batch×n)`` with tiny synthetic 128-token
samples whose loss mask is 0. Those pads cost almost nothing, and the token
savings are kept.

Degenerate-group detection ignores truncated samples (they are excluded from
loss and baseline anyway). A group whose only "disagreement" is a truncation
therefore also counts as zero-advantage.

This buffer also computes the per-task **panel metrics** (reward, accuracy,
truncation, lengths) on the full batch *before* filtering. Measured after
filtering, they would hide forgetting (all-wrong groups vanish) and
under-report truncation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import transfer_queue as tq
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.v1.replay_buffer import ReplayBufferAsync

from dpt.common.logging import get_logger
from dpt.rl.filtering import filter_zero_advantage
from dpt.rl.pipeline import panel_metrics
from dpt.verl_ext.runtime import get_runtime, row_sums, to_str_list

log = get_logger(__name__)


class ZeroAdvantageFilteringReplayBuffer(ReplayBufferAsync):
    """``ReplayBufferAsync`` (colocate_async mode) + panel metrics + zero-advantage group removal."""

    #: never shrink below this many groups; with fewer, keep the batch unfiltered (and warn)
    min_groups_to_keep: int = 4

    def sample(self, global_steps: int, partition_id: str, batch_size: int) -> tuple[KVBatchMeta, dict]:
        batch, metrics = super().sample(global_steps, partition_id, batch_size)
        if partition_id != "train":
            return batch, metrics
        rt = get_runtime()
        try:
            view = self._read(batch, rt)
        except Exception:  # monitoring/optimization only: never take the run down because of it
            log.exception("could not read batch for panels/filtering at step %d", global_steps)
            metrics["filter/error"] = 1.0
            return batch, metrics
        # Panel metrics on the FULL batch, before filtering removes all-wrong/all-right groups.
        metrics.update(panel_metrics(view["scores"], view["correct"], view["tasks"], view["groups"],
                                     view["lengths"], view["truncated"]))
        if not rt.cfg["rl"]["filtering"]["drop_zero_advantage_groups"]:
            return batch, metrics
        try:
            return self._filter(batch, metrics, rt, view)
        except Exception:
            log.exception("zero-advantage filtering failed at step %d; using the unfiltered batch", global_steps)
            metrics["filter/error"] = 1.0
            return batch, metrics

    @staticmethod
    def _read(batch: KVBatchMeta, rt: Any) -> dict[str, Any]:
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id,
                               select_fields=["rm_scores", "response_mask"])
        meta = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=["data_source"])
        scores = row_sums(data["rm_scores"])
        lengths = row_sums(data["response_mask"]).astype(np.int64)
        # The grader's own verdict ("acc"): the final score includes the overlong penalty, so
        # `score >= 1` would count correct-but-long answers as wrong.
        correct = scores >= 1.0
        try:
            ex = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=["extra_fields"])
            vals = ex["extra_fields"]
            vals = vals.tolist() if hasattr(vals, "tolist") else list(vals)
            accs = [(v.get("reward_extra_info", {}) if isinstance(v, dict) else {}).get("acc") for v in vals]
            if all(a is not None for a in accs):
                correct = np.asarray([float(a) >= 0.5 for a in accs])
        except Exception as e:
            log.debug("extra_fields unavailable, using score >= 1 as correctness: %s", e)
        return {
            "scores": scores, "lengths": lengths, "correct": correct,
            "tasks": to_str_list(meta["data_source"]),
            "groups": [k.split("_")[0] for k in batch.keys],       # keys are "{uid}_{session}_{idx}"
            "truncated": lengths >= int(rt.cfg["rl"]["sampling"]["max_response"]),
        }

    def _filter(self, batch: KVBatchMeta, metrics: dict, rt: Any, view: dict[str, Any]) -> tuple[KVBatchMeta, dict]:
        group_ids = view["groups"]
        res = filter_zero_advantage(view["scores"], group_ids, task_ids=view["tasks"],
                                    response_lengths=view["lengths"], valid=~view["truncated"],
                                    eps=float(rt.cfg["rl"]["advantage"]["zero_adv_eps"]))
        metrics.update(res.stats)
        rt.last_filter_stats = dict(res.stats)                   # read by the trainer for the curriculum

        kept_groups = {group_ids[i] for i in res.keep_idx}
        if len(kept_groups) < self.min_groups_to_keep:
            log.warning("only %d non-degenerate groups; keeping the unfiltered batch", len(kept_groups))
            metrics["filter/skipped_too_few"] = 1.0
            return batch, metrics

        drop = {int(i) for i in res.drop_idx}
        keep_keys = [k for i, k in enumerate(batch.keys) if i not in drop]
        keep_tags = [t for i, t in enumerate(batch.tags) if i not in drop]
        drop_keys = [k for i, k in enumerate(batch.keys) if i in drop]
        if drop_keys:
            # Prompt-level keys were already cleared by _materialize_batch; the dropped trajectories
            # are no longer part of the batch, so the trainer's end-of-step clear will not see them.
            tq.kv_clear(partition_id=batch.partition_id, keys=drop_keys)
        metrics["filter/dropped_samples"] = float(len(drop_keys))
        metrics["filter/kept_samples"] = float(len(keep_keys))
        return KVBatchMeta(partition_id=batch.partition_id, keys=keep_keys, tags=keep_tags), metrics


__all__ = ["ZeroAdvantageFilteringReplayBuffer"]
