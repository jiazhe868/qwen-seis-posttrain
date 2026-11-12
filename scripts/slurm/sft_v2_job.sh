#!/bin/bash
# Job 2 as one 8-node job (2026-11-12, the user's decision), run by scripts/slurm/sft_v2_queue.slurm when that job starts.
# This file is read from the repository at that moment, so it can still change while the job waits in the queue.
#
#   1. SFT v2: scripts/slurm/sft.slurm sft/sft_9b_v2.yaml, SFT v1 continued for one epoch on the knowledge rows plus a
#      replay sample of the v1 mix ($DATA/sft_v2/sft_train.jsonl, built by the knowledge job). Node 0 exports it to
#      HF; the export is copied to $WORK/dpt/exports/sft9b_v2_seed1, since $SCRATCH files left unread for 10 days are
#      purged and RL v3 starts from it.
#   2. SFT v2's eval: every set, D-Num-hard and the three budgets for row sft_v2_seed1 (final_eval_window.slurm with
#      phases main,hard,budgets on the export), ~3 h. A failure here is logged and RL v3 still runs.
#   3. RL v3 and its final eval: rl_v2_job.sh with rl/rl_9b_v3.yaml and row rl_v3_seed1. That script checks (and if
#      needed repairs or builds) bank v2, runs the 24K test unless a cap is known, trains, and evaluates. RL v2's
#      cap is kept when RL v2 chose one: same model, layout and lengths, so the same memory.
#
# Logs: this job's output, $WORK/dpt/logs/finaleval_<job>/ (both evals), $WORK/dpt/logs/rlv2_<job>/ (RL v3's steps).
# Dry run: DPT_SLURM_DRY_RUN=1 DPT_SLURM_NODES="<8 nodes>" bash scripts/slurm/sft_v2_job.sh
#
# The whole script is one { … } block: bash reads all of it before the first step, so a copy synced while the
# job runs cannot change the running job's steps (scripts it calls are read when they start).
{
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/slurm/env.sh
source scripts/slurm/lib.sh
export HF_HUB_OFFLINE=1

mapfile -t NODES < <(job_nodes)
(( ${#NODES[@]} == 8 )) || tdie "job 2 needs 8 nodes, this job has ${#NODES[@]}"
read -r DATA CK TAG <<< "$(python - <<'PY'
from dpt.common.config import load_config
p = load_config().project
print(p.paths.data_root, p.paths.ckpt_root, p.model.tag)
PY
)"
MIX="${DATA}/sft_v2/sft_train.jsonl"
SFT_DIR="${CK}/sft/sft${TAG}_v2_seed1"; SFT_EXPORT="${WORK}/dpt/exports/sft${TAG}_v2_seed1"
# SFT v2's data comes from the knowledge job (data.slurm knowledge). If it is still being made, wait up to 8 h rather
# than give up a slot that took days in the queue (8 idle nodes cost 24 SU per hour).
waited=0
until [[ -f "${MIX}" ]] || dry; do
  (( waited < 8 * 3600 )) || tdie "no ${MIX} after 8 h: the knowledge job has not built SFT v2's data"
  (( waited % 3600 == 0 )) && tlog "waiting for ${MIX} (knowledge job: squeue -n dpt-knowledge)"
  sleep 600; waited=$(( waited + 600 ))
done
gpus_free() {    # gpus_free: wait up to 10 min until no process holds a GPU on this job's nodes
  dry && return 0
  local h i busy=()
  for i in $(seq 1 60); do
    busy=()
    for h in "${NODES[@]}"; do
      [[ -z "$(on_node "${h}" nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]] || busy+=("${h}")
    done
    (( ${#busy[@]} )) || return 0
    sleep 10
  done
  tlog "WARN GPUs still in use after 10 min on ${busy[*]}"
}

# ---- 1. SFT v2 ---------------------------------------------------------------------------------------------------
if [[ -f "${SFT_EXPORT}/config.json" ]]; then
  tlog "SFT v2 is exported already: ${SFT_EXPORT}"
else
  tlog "SFT v2 on ${NODES[*]} ($(wc -l < "${MIX}" 2>/dev/null || echo '?') rows)"
  env DPT_SLURM_NODES="${NODES[*]}" bash scripts/slurm/sft.slurm sft/sft_9b_v2.yaml || tdie "SFT v2 failed: see this log"
  dry || [[ -f "${SFT_DIR}/hf_export/config.json" ]] || tdie "SFT v2 left no HF export at ${SFT_DIR}/hf_export"
  run rm -rf "${SFT_EXPORT}.partial"
  run cp -r "${SFT_DIR}/hf_export" "${SFT_EXPORT}.partial"
  run mv "${SFT_EXPORT}.partial" "${SFT_EXPORT}"
fi
gpus_free

# ---- 2. SFT v2's eval ----------------------------------------------------------------------------------------------
tlog "SFT v2's eval (row sft_v2_seed1)"
env -u DPT_CANCEL_JOB DPT_SLURM_NODES="${NODES[*]}" DPT_FINAL_ROW=sft_v2_seed1 DPT_FINAL_EXPORT="${SFT_EXPORT}" \
  DPT_FINAL_PHASES=main,hard,budgets bash scripts/slurm/final_eval_window.slurm ||
  tlog "WARN SFT v2's eval failed (see its log); RL v3 next"
gpus_free

# ---- 3. RL v3 and its final eval ---------------------------------------------------------------------------------
V2_CAP="${CK}/rl/rl${TAG}_v2_seed1.cap.json"
if [[ -z "${DPT_RL_CAP:-}" && -f "${V2_CAP}" ]]; then
  export DPT_RL_CAP="$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["cap"])' "${V2_CAP}")"
  tlog "RL v3 keeps RL v2's cap: ${DPT_RL_CAP}"
fi
exec env DPT_RL_CONFIG=rl/rl_9b_v3.yaml DPT_FINAL_ROW=rl_v3_seed1 bash scripts/slurm/rl_v2_job.sh
}
