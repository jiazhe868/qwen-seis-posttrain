"""Adapter layer: the only package in this repo that imports verl / TransferQueue.

Modules (each is plugged into verl 0.9.1 through a documented config key):

    register       imported in every process via VERL_USE_EXTERNAL_MODULES and model.external_lib;
                   registers the trainer mode `dpt_colocate_async` and estimator `dpt_task_grpo`
    trainer        DomainRLTrainer (colocate_async + curriculum, task-aware advantages, alerts,
                   extra checkpoint state)
    replay_buffer  zero-advantage group removal before old-logprob recompute
    sampler        curriculum sampler (stages, hard pool, auto-reweighting; exact resume)
    reward_fn      verl reward entry point → dpt.rewards router (built from the resolved config)
    sft_dataset    SFT dataset rendering each row with its own reasoning effort
    runtime        per-process context loaded from DPT_RESOLVED_CONFIG (no verl import)

Importing this package itself has no side effects. Import the submodules explicitly.
"""
