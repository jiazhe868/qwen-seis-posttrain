"""Registration side effects for verl. Imported in *every* process.

Two import routes are configured by ``dpt.launch.rl`` because verl loads
different extension points in different processes (see docs/verl_integration.md):

* ``VERL_USE_EXTERNAL_MODULES=dpt.verl_ext.register`` (Ray runtime env): imported at
  ``import verl`` in the driver, the TaskRunner and every worker. This is
  required for the trainer registry, which is read in the TaskRunner before
  any model config exists.
* ``actor_rollout_ref.model.external_lib=dpt.verl_ext.register``: imported when
  model configs are built, in the TaskRunner and in each actor worker. It is a
  second route to the same registrations.

Both routes are idempotent: the registries map names to the same objects.
"""

from __future__ import annotations

import os

import numpy as np
import torch

# Runtime fixes for the pinned stack, installed before anything else touches verl
# (inverted Qwen3.5 attention mask in A100 training forwards; see dpt.verl_ext.patches).
import dpt.verl_ext.patches  # noqa: F401,I001
from verl.trainer.ppo import core_algos

from dpt.common.logging import get_logger
from dpt.rl.advantage import group_mean_advantage

log = get_logger(__name__)
ESTIMATOR = "dpt_task_grpo"


if ESTIMATOR not in core_algos.ADV_ESTIMATOR_REGISTRY:

    @core_algos.register_adv_est(ESTIMATOR)
    def dpt_task_grpo(token_level_rewards: torch.Tensor, response_mask: torch.Tensor,
                      index: np.ndarray, config=None, **_: object):
        """Fallback estimator: group mean, no std (A2), no task terms.

        DomainRLTrainer overrides ``_compute_advantage`` and never calls this. It
        exists so verl's config validation sees a registered name. If it ever
        runs, the task-aware path was bypassed, so we say so loudly.
        """
        log.error("dpt_task_grpo fallback estimator invoked: task-aware advantages are NOT active")
        scores = token_level_rewards.sum(-1).cpu().numpy()
        adv = torch.as_tensor(group_mean_advantage(scores, list(index)), dtype=torch.float32,
                              device=token_level_rewards.device)
        out = adv[:, None] * response_mask
        return out, out


# Opt-in PPO-ratio diagnostics (DPT_DEBUG_PPO=<jsonl>, see dpt.verl_ext.debug_ppo).
if os.environ.get("DPT_DEBUG_PPO"):
    import dpt.verl_ext.debug_ppo  # noqa: F401

# The trainer registry lives in verl.trainer.ppo.v1 (needs TransferQueue). Import lazily-but-here so
# the registration happens in the TaskRunner; worker processes without TQ just skip it.
try:
    import dpt.verl_ext.trainer  # noqa: F401  (registers "dpt_colocate_async")
except ModuleNotFoundError as e:
    if e.name == "transfer_queue":   # a process without the v1 stack (it cannot host the trainer anyway)
        log.debug("trainer registration skipped in this process: %s", e)
    else:                            # anything else (e.g. a circular import) must be visible
        log.error("dpt trainer registration FAILED (%s); trainer_mode=dpt_colocate_async will be unknown", e)
        raise
except ImportError as e:
    log.error("dpt trainer registration FAILED (%s); trainer_mode=dpt_colocate_async will be unknown", e)
    raise
