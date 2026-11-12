"""Canonical metric names, the contract between the trainer adapter and the alert rules.

The RL trainer (``dpt.verl_ext``) logs these keys every step. Panels and
rules in ``configs/rl/monitoring.yaml`` refer to them by name, with ``{task}``
expanded over :data:`dpt.common.types.TASKS`. Renaming a key means updating
this file, the trainer, and the config in the same commit.

Keys owned by verl itself (``actor/pg_clipfrac``, ``actor/grad_norm``) are
consumed unchanged.
"""

from __future__ import annotations

# --- numerics / systems (incidents 1, 3, 4, 5) -------------------------------------------------
LOG_PPL_DIFF = "mismatch/log_ppl_abs_diff"  # |mean log-ppl(train) - mean log-ppl(rollout)|
LOG_PPL_DIFF_TAIL = (
    "mismatch/log_ppl_abs_diff_pos_gt_12k"  # same, tokens at positions > 12K (incident 1 slice)
)
POST_SYNC_PARITY = "parity/post_sync_pass"  # 1.0 pass / 0.0 fail on 16 fixed prompts after weight sync
POST_SYNC_MAX_DIFF = "parity/post_sync_max_abs_logprob_diff"
MTP_ACCEPTANCE = "spec/mtp_acceptance_rate"
MEMORY_PEAK_MAX_RANK = "memory/peak_gb_max_rank"
MEMORY_PEAK_ARGMAX_RANK = "memory/peak_gb_argmax_rank"
GRAD_NORM = "actor/grad_norm"  # verl
CLIP_FRAC = "actor/pg_clipfrac"  # verl
ROLLOUT_TIME = "timing/gen_s"
SANDBOX_INFRA_ERROR = "sandbox/t2_infra_error"  # dev T2 samples the sandbox could not grade (scored 0; incident 23)

# --- per-task learning dynamics (format with task=...) -----------------------------------------
TASK_REWARD = "task/{task}/reward_mean"
TASK_ACC = "task/{task}/acc"
TASK_ENTROPY = "task/{task}/entropy"
TASK_LEN = "task/{task}/len_mean"
TASK_LEN_CORRECT = "task/{task}/len_correct"
TASK_LEN_INCORRECT = "task/{task}/len_incorrect"
TASK_TRUNC = "task/{task}/truncation_rate"
TASK_ZERO_ADV = "task/{task}/zero_adv_frac"
TASK_CLIP = "task/{task}/clip_frac"
TASK_PROMPT_SHARE = "task/{task}/prompt_share"
TASK_TOKEN_SHARE = "task/{task}/token_share"
TASK_COUNT = "task/{task}/n_samples"

# --- reward-hacking detection ------------------------------------------------------------------
PROXY = "proxy/{task}"  # training reward on the dev slice
GOLD = "gold/{task}"  # stricter check: hidden tests / expert blind / independent grader
HACK_RATE = "hack/{category}/rate"

# --- evaluation during training ----------------------------------------------------------------
DEV_D_NUM = "dev/d_num"
DEV_D_NUM_CI = "dev/d_num_ci_halfwidth"
DEV_T1_WRONG_VALUE = "dev/t1_wrong_value"  # % of T1 dev answers that finished with a wrong value (gold grader)
DEV_T1_UNFINISHED = "dev/t1_unfinished"  # % of T1 dev answers whose thinking never ended
GENERAL_REPLAY_REWARD = TASK_REWARD.format(task="t4_general")


def task_key(template: str, task: str) -> str:
    return template.format(task=task)
