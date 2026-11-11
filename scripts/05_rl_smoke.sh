#!/usr/bin/env bash
# STEP 05: end-to-end RL smoke test on the workstation (Qwen3.5-0.8B, toy bank) in env `dpt-train`.
#
#   bash scripts/05_rl_smoke.sh                  # picks 4 free GPUs, else 2 (the box is shared)
#   DPT_SMOKE_GPUS=2 bash scripts/05_rl_smoke.sh
#
# Must pass before any cluster RL time is spent, and again after every verl/vLLM upgrade:
#   1. 12 steps straight: all adapters load, curriculum A→B→C, hard pool admitted, filtering,
#      TIS, alerts, checkpoints, dev validation with the gold grader (gold/<task>, dev/d_num);
#      step-1 `rollout_corr/log_ppl_abs_diff` is the vLLM-vs-Megatron parity number (must be < 2e-3)
#   2. resume equivalence (Incident 2): 6 steps, stop, resume, 6 more; steps 7-12 must match run 1
#
# Layout: 4 GPUs → train TP1×PP2×DP2, 4 vLLM replicas; 2 GPUs → TP1×PP2×DP1, 2 replicas.
# Env: bash scripts/00_setup_env.sh train-local  (verl 0.9.1 + Megatron-Bridge + vLLM + TE, CUDA 12.x).
source "$(dirname "$0")/common.sh"
# One node: the workstation, or one cluster gpu-a100 node (scripts/slurm/smoke.slurm; 3 GPUs → 2 used).
[[ "${DPT_HW_PROFILE}" == local_4xa100_40g || "${DPT_HW_PROFILE}" == slurm_3xa100_40g ]] \
  || die "this step runs on one node: DPT_HW_PROFILE=local_4xa100_40g or slurm_3xa100_40g (current: ${DPT_HW_PROFILE})"
TRAIN_ENV="${DPT_TRAIN_ENV:-dpt-train}"
[[ -n "$(env_python "${TRAIN_ENV}")" ]] || die "env ${TRAIN_ENV} missing: bash scripts/00_setup_env.sh train-local"

NEED_MIB=24000      # vLLM util 0.5 of 40 GB (20 GB) + headroom; training state is offloaded during rollout
NGPU="${DPT_SMOKE_GPUS:-$([[ "$(count_free ${NEED_MIB})" -ge 4 ]] && echo 4 || echo 2)}"
GPUS="$(free_gpus "${NGPU}" "${NEED_MIB}")"
[[ "$(echo "${GPUS}" | tr ',' '\n' | grep -c .)" -eq "${NGPU}" ]] \
  || die "need ${NGPU} GPUs with ≥ ${NEED_MIB} MiB free; only '${GPUS}' qualify (other jobs are running)"
export CUDA_VISIBLE_DEVICES="${GPUS}"
# Whole layout spelled out, so the smoke is the same on every profile (the cluster profile's 9B layout is 8 × PP8).
LAYOUT=("hardware.nnodes=1" "hardware.gpus_per_node=${NGPU}" "hardware.train.tp=1" "hardware.train.pp=2"
        "hardware.train.dp=$((NGPU / 2))" "hardware.rollout.tp=1" "hardware.rollout.replicas=${NGPU}"
        "hardware.rollout.gpu_memory_utilization=0.5")
log "smoke on GPUs ${GPUS} (${NGPU} GPUs: TP1×PP2×DP$((NGPU / 2)), ${NGPU} vLLM replicas) in env ${TRAIN_ENV}"

# The smoke config reads ${project.paths.data_root}/smoke: <repo>/data on the workstation, $WORK/dpt/data on the cluster.
SMOKE_DATA="$(cfg project.paths.data_root)/smoke"
if [[ ! -f "${SMOKE_DATA}/rl_dev.parquet" ]]; then
  log "toy data → ${SMOKE_DATA} (parquet needs pyarrow → built inside ${TRAIN_ENV})"
  in_env "${TRAIN_ENV}" python -m dpt.data.pipeline --offline \
    --stages ingest,quality,dedup,decontam,verify,difficulty,publish,sft data_pipeline.output_dir="${SMOKE_DATA}" \
    data_pipeline.publish.dev_copy_if_empty=true
fi

in_env "${TRAIN_ENV}" ray stop --force >/dev/null 2>&1 || true
in_env "${TRAIN_ENV}" ray start --head --num-gpus "${NGPU}" --disable-usage-stats
trap 'in_env "${TRAIN_ENV}" ray stop --force >/dev/null 2>&1 || true' EXIT
SMOKE=(--config rl/rl_smoke_0p8b.yaml --allow-missing-manifest --no-strict "${LAYOUT[@]}")
OUT="$(cfg project.paths.ckpt_root)/rl/smoke"     # workstation: <repo>/checkpoints; the cluster: $SCRATCH
# Start clean: verl's resume_mode=auto would otherwise continue from a previous smoke's checkpoints.
rm -rf "${OUT}/continuous" "${OUT}/resumed"
# Must match rl.total_steps / save_freq in configs/rl/rl_smoke_0p8b.yaml.
TOTAL=12
HALF=6
log "run 1: ${TOTAL} continuous steps"
in_env "${TRAIN_ENV}" python -m dpt.launch.rl "${SMOKE[@]}" rl_run.output_dir="${OUT}/continuous"
log "run 2: ${HALF} steps, then resume to ${TOTAL} (resume equivalence)"
# Stop at step HALF by overriding ONLY verl's stopping point: rl.total_steps also sets the LR-decay
# horizon and the sampler length, and changing it would make the later steps differ by construction.
in_env "${TRAIN_ENV}" python -m dpt.launch.rl "${SMOKE[@]}" rl_run.output_dir="${OUT}/resumed" verl.trainer.total_training_steps=${HALF}
in_env "${TRAIN_ENV}" python -m dpt.launch.rl "${SMOKE[@]}" rl_run.output_dir="${OUT}/resumed"
log "resume equivalence (steps $((HALF + 1))-${TOTAL})"
in_env "${TRAIN_ENV}" python -m dpt.monitoring.compare_runs "${OUT}/continuous/metrics.jsonl" "${OUT}/resumed/metrics.jsonl" --steps "$((HALF + 1))-${TOTAL}"
