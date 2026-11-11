"""Framework-free RL algorithm pieces (no verl / vLLM / Megatron / Ray imports).

Everything that decides *what gradient the policy receives* lives here and is
unit-tested on CPU. The verl adapter (``dpt.verl_ext``) only moves arrays
between verl's ``DataProto`` and these functions.

Module map (report §阶段二 RL 设计 unless noted):

    advantage       group-mean advantage (no std division, ablation A2), per-task running-std
                    scaling (§多任务 RL 难调 调整-2), per-task loss-share multipliers (调整-1)
    loss            reference PPO-clip with clip-higher (A3), dual-clip, TIS weighting, and the
                    aggregation modes (fixed-constant normalization, A10)
    tis             truncated importance sampling weights + sequence-level mismatch masking (Incident 1)
    mismatch        training-vs-rollout log-ppl gap, KL estimators, slices by task / position /
                    length, and the post-weight-sync fixed-prompt parity check (Incident 3)
    filtering       zero-advantage group removal, truncated-response masking
    curriculum      3-stage task mix, hard pool from step 150, auto-reweighting, exact state (Incident 2)
    partial_rollout group-level buffer for responses suspended past 16K tokens (§单步时间预算)
    microbatch      token-budget micro-batching, length bucketing, activation-memory pre-flight (Incident 4)
    schedule        RL learning-rate schedule (1e-6, 10-step warmup, 5e-7 for the last 120 steps)
    state           bundle + save/load of every stateful component, resume-equivalence harness
"""
