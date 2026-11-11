"""Every RL config and ablation composes, renders, and passes (or deliberately waives) pre-flight."""

from __future__ import annotations

from pathlib import Path

import pytest

from dpt.common.config import CONFIG_ROOT
from dpt.launch.common import compose_stage
from dpt.launch.rl import PreflightError, build, preflight

ABLATIONS = sorted(p.stem for p in (CONFIG_ROOT / "rl" / "ablations").glob("A*.yaml"))


def _build(tmp_path, config, profile, ablation=None, extra=()):
    import os

    os.environ["DPT_HW_PROFILE"] = profile
    try:
        return build(config, [f"rl_run.output_dir={tmp_path}", *extra], ablation=ablation,
                     require_manifest=False, strict=True)
    finally:
        os.environ.pop("DPT_HW_PROFILE", None)


def test_main_config_renders_on_cluster(tmp_path):
    run = _build(tmp_path, "rl/rl_27b.yaml", "cluster_2x8xa100_80g")
    ov = dict(o.split("=", 1) for o in run.overrides if "=" in o)
    assert ov["trainer.v1.trainer_mode"] == "dpt_colocate_async"
    assert ov["actor_rollout_ref.actor.megatron.tensor_model_parallel_size"] == "4"
    assert ov["actor_rollout_ref.actor.ppo_mini_batch_size"] == "64"
    assert ov["+actor_rollout_ref.rollout.engine_kwargs.vllm.mamba_ssm_cache_dtype"] == "float32"
    assert "len:4096" in ov["+reward.reward_kwargs.overlong_buffer_cfg"]
    # struct keys in verl's schema take new sub-keys only via "+" (found by the smoke run)
    assert ov["+data.apply_chat_template_kwargs.reasoning_effort"] == "medium"
    assert (Path(run.env["DPT_RESOLVED_CONFIG"])).exists()


@pytest.mark.parametrize("ablation", ABLATIONS)
def test_each_ablation_composes(tmp_path, ablation):
    cfg = compose_stage("rl/rl_27b.yaml", overlays=[f"rl/ablations/{ablation}.yaml"],
                        profile="cluster_2x8xa100_80g")
    base = Path(cfg.ablation.base_config).name
    run = _build(tmp_path, f"rl/{base}", "cluster_2x8xa100_80g", ablation=ablation)
    assert run.name.endswith(ablation)


def test_smoke_config_renders_locally(tmp_path):
    run = _build(tmp_path, "rl/rl_smoke_0p8b.yaml", "local_4xa100_40g")
    ov = dict(o.split("=", 1) for o in run.overrides if "=" in o)
    assert ov["actor_rollout_ref.rollout.tensor_model_parallel_size"] == "1"
    assert ov["data.train_batch_size"] == "16"


@pytest.mark.parametrize("override,needle", [
    ("verl.actor_rollout_ref.rollout.top_p=0.9", "top_p"),
    ("verl.algorithm.rollout_correction.rollout_is=null", "TIS"),
    ("verl.actor_rollout_ref.actor.loss_agg_mode=token-mean", "token-sum"),
    ("verl.actor_rollout_ref.model.mtp.enable_train=false", "MTP"),
    ("rl_run.max_prompt_length=9000", "Incident 4"),
    ("verl.actor_rollout_ref.actor.calculate_entropy=true", "in-place"),
])
def test_preflight_blocks_known_failure_modes(override, needle):
    cfg = compose_stage("rl/rl_27b.yaml", [override], profile="cluster_2x8xa100_80g")
    with pytest.raises(PreflightError, match=needle):
        preflight(cfg)


def test_flat_cli_override_hits_literal_verl_key():
    cfg = compose_stage("rl/rl_27b.yaml", ["verl.actor_rollout_ref.actor.optim.lr=2e-6"],
                        profile="cluster_2x8xa100_80g")
    assert cfg.verl["actor_rollout_ref.actor.optim.lr"] == 2e-6
    assert "actor_rollout_ref" not in cfg.verl
