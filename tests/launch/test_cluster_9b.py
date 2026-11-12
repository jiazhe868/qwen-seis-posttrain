"""The 9B main line: configs compose, render the planned layout, and keep every guard."""

from __future__ import annotations

import pytest

from dpt.common.config import load_config, to_container
from dpt.launch.common import compose_stage
from dpt.launch.rl import PreflightError, build, preflight
from dpt.launch.sft import build as build_sft
from dpt.planning.budget import Arch, Cluster, Layout, first_stage_memory_per_gpu, rollout_memory, train_memory_per_gpu

cluster_environment_variables = {"DPT_HW_PROFILE": "slurm_3xa100_40g", "DPT_MODEL_LINE": "qwen35_9b"}


@pytest.fixture()
def cluster_environment(monkeypatch, tmp_path):
    for k, v in cluster_environment_variables.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("WORK", str(tmp_path / "work"))
    monkeypatch.setenv("SCRATCH", str(tmp_path / "scratch"))
    monkeypatch.setenv("DPT_SANDBOX_SIF", str(tmp_path / "sandbox.sif"))
    return tmp_path


def _ov(run):
    return dict(o.split("=", 1) for o in run.overrides if "=" in o)


def test_rl_9b_renders_the_cluster_layout(cluster_environment):
    ov = _ov(build("rl/rl_9b.yaml", [f"rl_run.output_dir={cluster_environment}/rl"], require_manifest=False))
    assert ov["trainer.nnodes"] == "8" and ov["trainer.n_gpus_per_node"] == "3"
    assert ov["actor_rollout_ref.actor.megatron.tensor_model_parallel_size"] == "1"
    assert ov["actor_rollout_ref.actor.megatron.pipeline_model_parallel_size"] == "8"
    assert ov["actor_rollout_ref.rollout.tensor_model_parallel_size"] == "1"
    assert ov["actor_rollout_ref.actor.optim.lr"] == "2e-06"                  # the 9B-tuned LR (A8)
    assert ov["data.train_batch_size"] == "126"                               # 128 does not divide DP 3
    assert "rl_main.parquet" in ov["data.train_files"] and "rl_hard" not in ov["data.train_files"]   # 26 problems
    assert ov["actor_rollout_ref.model.path"].endswith("/sft/sft9b_seed1/hf_export")
    assert ov["trainer.experiment_name"] == "rl9b_seed1"
    # every report guard is inherited from rl_27b.yaml
    assert ov["+actor_rollout_ref.rollout.engine_kwargs.vllm.mamba_ssm_cache_dtype"] == "float32"
    assert ov["algorithm.rollout_correction.rollout_is"] == "token"
    assert ov["actor_rollout_ref.actor.loss_agg_mode"] == "token-sum"


def test_rl_9b_resolves_the_sandbox_container_into_the_reward_config(cluster_environment):
    cfg = compose_stage("rl/rl_9b.yaml", [f"rl_run.output_dir={cluster_environment}/rl"])
    sb = to_container(cfg)["rewards"]["t2_code"]["sandbox"]
    assert sb["container_image"] == str(cluster_environment / "sandbox.sif") and sb["container_network"] == "none"


def test_cluster_launch_refuses_an_unisolated_sandbox(cluster_environment, monkeypatch):
    monkeypatch.delenv("DPT_SANDBOX_SIF")
    cfg = compose_stage("rl/rl_9b.yaml", [f"rl_run.output_dir={cluster_environment}/rl"])
    with pytest.raises(PreflightError, match="filesystem isolation"):
        preflight(cfg, strict=True)
    assert any("filesystem isolation" in w for w in preflight(cfg, strict=False))   # smoke: warning only


def test_9b_stage_configs_refuse_the_27b_line(cluster_environment, monkeypatch):
    monkeypatch.setenv("DPT_MODEL_LINE", "qwen38_27b")
    for stage in ("rl/rl_9b.yaml", "sft/sft_9b.yaml"):
        with pytest.raises(ValueError, match="export DPT_MODEL_LINE=qwen35_9b"):
            compose_stage(stage)


def test_sft_9b_batch_divides_the_data_parallel_size(cluster_environment):
    run = build_sft("sft/sft_9b.yaml", [f"sft.output_dir={cluster_environment}/sft"], require_manifest=False)
    ov = _ov(run)
    hw = to_container(load_config())["hardware"]
    assert int(ov["data.train_batch_size"]) % hw["train"]["dp"] == 0     # verl would silently drop the rest
    assert ov["model.path"] == "Qwen/Qwen3.5-9B"
    assert ov["engine.pipeline_model_parallel_size"] == "8"
    assert run.env["VERL_USE_EXTERNAL_MODULES"] == "dpt.verl_ext.patches"     # A100 mask fix (SM80)


@pytest.mark.parametrize("stage", ["rl/rl_9b.yaml", "rl/rl_9b_timing.yaml"])
def test_rl_9b_batches_divide_the_data_parallel_size(cluster_environment, stage):
    # verl.utils.config.validate_config: prompts × n % (DP × micro-batch) == 0, and verl/workers/engine_workers.py:
    # mini-batch × n % DP == 0. The timing run 3530168 died on the first after 7 min on 8 nodes.
    ov = _ov(build(stage, [f"rl_run.output_dir={cluster_environment}/rl"], require_manifest=False))
    mp = 1
    for k in ("tensor_model_parallel_size", "pipeline_model_parallel_size", "context_parallel_size"):
        mp *= int(ov[f"actor_rollout_ref.actor.megatron.{k}"])
    gpus = int(ov["trainer.nnodes"]) * int(ov["trainer.n_gpus_per_node"])
    dp, n = gpus // mp, int(ov["actor_rollout_ref.rollout.n"])
    assert gpus % mp == 0 and dp == 3
    assert int(ov["data.train_batch_size"]) * n % (dp * int(ov["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"])) == 0
    assert int(ov["actor_rollout_ref.actor.ppo_mini_batch_size"]) * n % dp == 0


def test_planned_layout_fits_40gb_with_margin(cluster_environment):
    c = to_container(load_config())
    hw, a = c["hardware"], Arch.from_project(c["project"]["model"]["arch"])
    lay = Layout(tp=hw["train"]["tp"], pp=hw["train"]["pp"], dp=hw["train"]["dp"],
                 rollout_tp=hw["rollout"]["tp"], rollout_gpu_mem_util=hw["rollout"]["gpu_memory_utilization"])
    assert lay.tp * lay.pp * lay.dp == hw["nnodes"] * hw["gpus_per_node"] == 24
    limit = 0.9 * hw["gpu_mem_gb"] * 1e9
    assert train_memory_per_gpu(a, lay).total < limit
    assert first_stage_memory_per_gpu(a, lay).total < limit                  # the embedding stage too
    r = rollout_memory(a, Cluster(nnodes=8, gpus_per_node=3, gpu_mem_gb=40), lay)
    assert r.concurrent_per_replica >= 32 and r.concurrent_total >= 768      # ≥ 3/4 of a 126 × 8 batch at once


def test_pp4_would_not_fit_the_embedding_stage(cluster_environment):
    a = Arch.from_project(to_container(load_config())["project"]["model"]["arch"])
    assert first_stage_memory_per_gpu(a, Layout(tp=1, pp=4, dp=6)).total > 0.9 * 40e9


def test_deadline_reaches_the_ray_workers_as_a_string(cluster_environment, monkeypatch):
    monkeypatch.setenv("DPT_DEADLINE", "1767225600.7")
    ov = _ov(build("rl/rl_9b.yaml", [f"rl_run.output_dir={cluster_environment}/rl"], require_manifest=False))
    # quoted so Hydra keeps it a string (Ray rejects non-string env values)
    assert ov["+ray_kwargs.ray_init.runtime_env.env_vars.DPT_DEADLINE"] == "'1767225600'"


def test_slurm_refresh_and_hold_step_reach_the_ray_workers(cluster_environment, monkeypatch):
    """Without the margin and the job id in the trainer's env, a walltime extension by the site never counted."""
    for k, v in {"DPT_DEADLINE": "1767225600", "DPT_DEADLINE_MARGIN_S": "1200", "SLURM_JOB_ID": "3532660",
                 "DPT_STOP_AT_STEP": "119"}.items():
        monkeypatch.setenv(k, v)
    ov = _ov(build("rl/rl_9b.yaml", [f"rl_run.output_dir={cluster_environment}/rl"], require_manifest=False))
    env = {k.rsplit(".", 1)[1]: v for k, v in ov.items() if k.startswith("+ray_kwargs.ray_init.runtime_env.env_vars.")}
    got = [env[k] for k in ("DPT_DEADLINE_MARGIN_S", "SLURM_JOB_ID", "DPT_STOP_AT_STEP")]
    assert got == ["'1200'", "'3532660'", "'119'"]


def test_describe_reports_progress_for_job_chaining(cluster_environment):
    import json

    from dpt.launch.rl import describe

    run = cluster_environment / "rl"
    d = describe("rl/rl_9b.yaml", [f"rl_run.output_dir={run}"])
    assert (d["latest_step"], d["total_steps"], d["done"], d["paused"]) == (0, 120, False, False)
    run.mkdir(parents=True, exist_ok=True)
    (run / "latest_checkpointed_iteration.txt").write_text("77\n")
    (run / "WALLTIME.json").write_text(json.dumps({"step": 77}))
    d = describe("rl/rl_9b.yaml", [f"rl_run.output_dir={run}"])
    assert (d["latest_step"], d["done"], d["walltime"], d["paused"]) == (77, False, True, False)
    (run / "PAUSED.json").write_text("{}")
    (run / "latest_checkpointed_iteration.txt").write_text("120")
    d = describe("rl/rl_9b.yaml", [f"rl_run.output_dir={run}"])
    assert d["done"] and d["paused"] and not d["held"]
    (run / "HOLD.json").write_text("{}")
    assert describe("rl/rl_9b.yaml", [f"rl_run.output_dir={run}"])["held"]


def test_rl_9b_v2_renders_bank_v2_weights_and_the_stop_rule(cluster_environment):
    """RL v2 (2026-11-07, changed 11-11): bank v2 with its hard pool, the dev split without T3, dev and checkpoints
    every 10 steps, no T3 in training, T4 at 15%, half of run 1's learning rate, and the two dev pause rules."""
    from dpt.launch.common import compose_stage

    ov = _ov(build("rl/rl_9b_v2.yaml", [f"rl_run.output_dir={cluster_environment}/rl"], require_manifest=False))
    assert "/v2/rl_main.parquet" in ov["data.train_files"] and "/v2/rl_hard.parquet" in ov["data.train_files"]
    assert "/v2/rl_dev_no_t3.parquet" in ov["data.val_files"]
    assert ov["trainer.test_freq"] == "10" and ov["trainer.save_freq"] == "10"
    assert ov["trainer.experiment_name"] == "rl9b_v2_seed1"
    assert ov["actor_rollout_ref.actor.optim.lr"] == "1e-06" and ov["actor_rollout_ref.actor.optim.min_lr"] == "5e-07"
    assert ov["data.train_batch_size"] == "126" and ov["data.max_response_length"] == "16384"
    cfg = compose_stage("rl/rl_9b_v2.yaml")
    assert cfg.curriculum.difficulty_weights.enabled and cfg.curriculum.hard_pool_join_step == 34
    for stage in cfg.curriculum.stages:
        assert dict(stage.mix) == {"t1_num": 0.60, "t2_code": 0.25, "t3_report": 0.00, "t4_general": 0.15}
    assert [r.name for r in cfg.monitoring.extra_rules] == ["dev_d_num_best_drop", "dev_t1_wrong_value_rise",
                                                            "t1_length_drop"]
    from dpt.monitoring.alerts import AlertManager
    from dpt.rl.curriculum import CurriculumConfig
    from omegaconf import OmegaConf

    CurriculumConfig.from_dict(OmegaConf.to_container(cfg.curriculum, resolve=True))
    mgr = AlertManager.from_config(OmegaConf.to_container(cfg.monitoring, resolve=True))
    assert any(r.kind == "best_drop" for r in mgr.rules)
