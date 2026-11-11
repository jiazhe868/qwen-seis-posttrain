"""Launch stage-2 RL (verl ``main_ppo``, v1 trainer, Megatron + vLLM colocated).

    python -m dpt.launch.rl --config rl/rl_27b.yaml [--ablation A08_lr_2e-6] [--dry-run] [key=value ...]

Run on the Ray head node after ``ray start`` on all nodes (``scripts/06_rl.sh``
does both). Before anything is launched, the pre-flight checks below must pass.
Each one encodes a lesson from the report's incident log, so a known failure
mode cannot be re-introduced by a config edit.

=========================  ========================================================
check                      why (report reference)
=========================  ========================================================
data manifests verified    results must be traceable to exact data (§数据-6)
prompt + response ≤ 32K    Incident 4: a 31K trajectory padded into a 32K bucket OOMed
memory pre-flight          Incident 4: estimate the worst micro-batch before launch
FP32 recurrent state       Incident 1: BF16 rollout state → train/rollout drift → collapse
prefix cache off / reset   Incident 3: stale prefix cache after weight sync
TIS on, KL off, top_p = 1  §算法配置 (A4, A1; top_p<1 breaks logprob consistency)
MTP head trained           Incident 5: frozen MTP head → acceptance 0.71 → 0.46
MTP acceptance observable  Incident 5: vLLM stats logging feeds spec/mtp_acceptance_rate
token-sum loss             A10: fixed-constant normalization keeps filtering exact
eval sets unreadable       §数据: the training user must not be able to read eval data
T2 sandbox isolated        Shared cluster: model-written code must not see $WORK/$SCRATCH (container)
=========================  ========================================================
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from dpt.common.config import to_container
from dpt.common.logging import get_logger
from dpt.launch.common import as_list, compose_stage, execute, new_launch_dir, prepare_run, python_exe
from dpt.launch.walltime import DEADLINE_ENV, MARGIN_ENV, STOP_AT_ENV

log = get_logger(__name__)


class PreflightError(RuntimeError):
    pass


def preflight(cfg: DictConfig, *, strict: bool = True) -> list[str]:
    """Return a list of warnings; raise PreflightError on any hard failure."""
    c = to_container(cfg)
    v, rl, hw = c["verl"], c["rl"], c["hardware"]
    errors: list[str] = []
    warnings: list[str] = []
    # An ablation that deliberately reverts a guarded setting declares it (e.g. A4 turns TIS off).
    waivers = set((c.get("ablation") or {}).get("preflight_waivers", []) or [])

    total = int(v["data.max_prompt_length"]) + int(v["data.max_response_length"])
    if total > int(c["project"]["thinking"]["max_context"]):
        errors.append(f"prompt+response = {total} > max_context (Incident 4)")
    if v.get("+actor_rollout_ref.rollout.engine_kwargs.vllm.mamba_ssm_cache_dtype") != "float32":
        errors.append("rollout GDN state must be float32 (Incident 1)")
    if v.get("actor_rollout_ref.rollout.enable_prefix_caching"):
        warnings.append("prefix caching ON: verl resets it on wake_up, but keep the per-task mismatch alert armed "
                        "(Incident 3)")
    if v.get("algorithm.rollout_correction.rollout_is") != "token" and "tis" not in waivers:
        errors.append("TIS must be enabled (algorithm.rollout_correction.rollout_is=token; A4)")
    if v.get("actor_rollout_ref.actor.use_kl_loss") or v.get("algorithm.use_kl_in_reward"):
        warnings.append("KL is enabled; the report's ablation A1 chose KL=0")
    if float(v.get("actor_rollout_ref.rollout.top_p", 1.0)) != 1.0:
        errors.append("rollout top_p must be 1.0 (sampling vs scoring distribution mismatch)")
    if v.get("actor_rollout_ref.model.mtp.enable_rollout") and not v.get("actor_rollout_ref.model.mtp.enable_train"):
        errors.append("MTP speculative decoding needs the MTP head trained in RL (Incident 5)")
    mtp_rollout = v.get("actor_rollout_ref.model.mtp.enable_rollout")
    if mtp_rollout and v.get("actor_rollout_ref.rollout.disable_log_stats", True):
        warnings.append("MTP drafting with rollout.disable_log_stats=true: vLLM keeps no spec-decode counters, so "
                        "spec/mtp_acceptance_rate is missing and the Incident 5 alert cannot fire")
    if v.get("actor_rollout_ref.actor.loss_agg_mode") != "token-sum" and "loss_agg" not in waivers:
        errors.append("loss_agg_mode must be token-sum (fixed-constant normalization lives in the advantages; A10)")
    if v.get("actor_rollout_ref.actor.calculate_entropy") and v.get(
            "actor_rollout_ref.actor.megatron.entropy_from_logits_with_chunking"):
        errors.append("calculate_entropy=true with entropy_from_logits_with_chunking crashes the backward in "
                      "verl 0.9.1 (in-place edits of shared logits views); keep calculate_entropy=false")
    if not v.get("actor_rollout_ref.rollout.calculate_log_probs"):
        errors.append("rollout.calculate_log_probs must be true (TIS + mismatch metrics)")

    # Memory pre-flight for the production layout (the activation model is calibrated on 27B).
    is_27b = "27B" in str(v.get("actor_rollout_ref.model.path", "")) + c["project"]["model"]["base"]
    if hw["gpu_mem_gb"] >= 80 and is_27b:
        try:
            from dpt.rl.microbatch import ActivationMemoryModel, make_count_microbatches, preflight as mem_preflight

            # Worst case: two maximal sequences in flight (PP=2), one per micro-batch.
            plan = make_count_microbatches([total, total], 1)
            rep = mem_preflight(plan, ActivationMemoryModel(), limit_gb=float(rl["microbatch"]["memory_limit_gb"]))
            log.info("memory pre-flight: peak %.1f GB (limit %.1f)", rep.peak_gb, rep.limit_gb)
            if not rep.ok:
                errors.extend(rep.messages)
        except PreflightError:
            raise
        except Exception as e:
            (errors if strict else warnings).append(f"memory pre-flight failed: {e}")

    # T2 runs tens of thousands of model-written programs. On a shared cluster account they must not be
    # able to touch the project's files ($HOME/$WORK/$SCRATCH): require the container sandbox there.
    sb = ((c.get("rewards") or {}).get("t2_code") or {}).get("sandbox") or {}
    if (hw["profile"].startswith("slurm_") and sb.get("mode", "local") == "local" and not sb.get("container_image")
            and "sandbox_container" not in waivers):
        (errors if strict else warnings).append(
            "T2 sandbox has no filesystem isolation on a shared cluster: model-written code would run as you "
            "and could delete files in $WORK/$SCRATCH. Build the image and export DPT_SANDBOX_SIF "
            "(scripts/slurm/README.md §sandbox, then scripts/slurm/check_sandbox.sh)")

    secure = c["project"]["paths"]["eval_secure_root"]
    if os.path.exists(secure) and os.access(secure, os.R_OK):
        msg = f"eval sets at {secure} are readable by this user; run training as a user without access"
        (errors if strict and hw["profile"].startswith("cluster") else warnings).append(msg)

    if waivers:
        warnings.append(f"ablation waives pre-flight checks: {sorted(waivers)}")
    for w in warnings:
        log.warning("pre-flight: %s", w)
    if errors:
        raise PreflightError("pre-flight failed:\n  - " + "\n  - ".join(errors))
    return warnings


def _compose(config: str, overrides: list[str], ablation: str | None) -> DictConfig:
    overlays = [f"rl/ablations/{ablation}.yaml"] if ablation else []
    cfg = compose_stage(config, overrides, overlays=overlays)
    if ablation:
        cfg.rl_run.run_name = f"{cfg.rl_run.run_name}_{ablation}"
        want = cfg.get("ablation", {}).get("base_config")
        if want and Path(want).name != Path(config).name:
            raise PreflightError(f"ablation {ablation} is defined against {want}, not {config}")
    return cfg


def describe(config: str, overrides: list[str], *, ablation: str | None = None) -> dict[str, Any]:
    """Where a run lives and how far it got, for job scripts deciding whether to (re)submit.

    ``done``: verl's last checkpoint is the final step. ``paused``: an alert stopped the run and a
    human must look (``PAUSED.json``); job scripts never resubmit then. ``walltime``: the previous
    job stopped at its wall-clock limit (``WALLTIME.json``); the next job resumes. ``held``: the run stopped
    at ``$DPT_STOP_AT_STEP`` (``HOLD.json``, e.g. before T3 needs the judge); relaunching resumes it.
    """
    cfg = _compose(config, overrides, ablation)
    run_dir = Path(cfg.rl_run.output_dir)
    latest = run_dir / "latest_checkpointed_iteration.txt"
    step = int(latest.read_text().strip()) if latest.exists() else 0
    total = int(cfg.verl["trainer.total_training_steps"])
    return {"run_name": str(cfg.rl_run.run_name), "run_dir": str(run_dir), "latest_step": step,
            "total_steps": total, "done": step >= total, "paused": (run_dir / "PAUSED.json").exists(),
            "walltime": (run_dir / "WALLTIME.json").exists(), "held": (run_dir / "HOLD.json").exists()}


def build(config: str, overrides: list[str], *, ablation: str | None = None, require_manifest: bool = True,
          strict: bool = True):
    cfg = _compose(config, overrides, ablation)
    preflight(cfg, strict=strict)

    run_dir = Path(cfg.rl_run.output_dir)
    launch_dir = new_launch_dir(run_dir)
    resolved = launch_dir / "resolved_config.yaml"
    hashes = launch_dir / "data_hashes.json"
    env_prefix = "+ray_kwargs.ray_init.runtime_env.env_vars."
    cfg.verl[env_prefix + "DPT_RESOLVED_CONFIG"] = str(resolved)
    cfg.verl[env_prefix + "DPT_DATA_HASHES"] = str(hashes)
    # The trainer runs in a Ray worker, which does not inherit this shell's environment: pass the
    # job's wall-clock deadline through Ray (dpt.launch.walltime; Ray env values must be strings).
    if os.environ.get(DEADLINE_ENV):
        cfg.verl[env_prefix + DEADLINE_ENV] = str(int(float(os.environ[DEADLINE_ENV])))
    # So are the margin and the job id the trainer needs to re-read the limit from Slurm (an extension by
    # site staff counts), and the hold step (stop before the judge is needed while its job is still queued).
    for key in (MARGIN_ENV, "SLURM_JOB_ID", STOP_AT_ENV):
        if os.environ.get(key):
            cfg.verl[env_prefix + key] = str(os.environ[key])

    argv = [python_exe(), "-m", "verl.trainer.main_ppo"]
    run_c: dict[str, Any] = to_container(cfg)["rl_run"]
    data = as_list(run_c["train_files"]) + as_list(run_c["dev_files"])
    env = {"DPT_RESOLVED_CONFIG": str(resolved), "DPT_DATA_HASHES": str(hashes),
           "VERL_USE_EXTERNAL_MODULES": "dpt.verl_ext.register",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    return prepare_run(name=cfg.rl_run.run_name, cfg=cfg, section="verl", base_argv=argv, run_dir=run_dir,
                       data_files=data, env=env, require_manifest=require_manifest, launch_dir=launch_dir)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="rl/rl_27b.yaml")
    p.add_argument("--ablation", default=None, help="name of a file in configs/rl/ablations/ (without .yaml)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-missing-manifest", action="store_true", help="smoke tests only")
    p.add_argument("--no-strict", action="store_true", help="downgrade some pre-flight errors to warnings (smoke)")
    p.add_argument("--no-hydra-check", action="store_true", help="skip composing verl's config before launch")
    p.add_argument("--describe", action="store_true",
                   help="print the run's directory and progress as JSON (done / paused / walltime) and exit")
    p.add_argument("overrides", nargs="*")
    a = p.parse_args()
    if a.describe:
        import json

        print(json.dumps(describe(a.config, a.overrides, ablation=a.ablation)))
        return
    run = build(a.config, a.overrides, ablation=a.ablation, require_manifest=not a.allow_missing_manifest,
                strict=not a.no_strict)
    raise SystemExit(execute(run, dry_run=a.dry_run or bool(os.environ.get("DPT_DRY_RUN")),
                             check_module=None if a.no_hydra_check else "verl.trainer.main_ppo"))


if __name__ == "__main__":
    main()
