#!/bin/bash
#SBATCH -J dpt-sft
#SBATCH -p gpu-a100
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 24:00:00
#SBATCH -o dpt-sft-%j.out
# Stage-1 SFT of the 9B on 8 gpu-a100 nodes (TP1 × PP8 × DP3), then the HF export on node 0.
#
#   sbatch -A <allocation> scripts/slurm/sft.slurm [sft/sft_9b.yaml] [dotlist overrides...]
#
# ~2-3 h for the plan's ~120M tokens × 2 epochs (dpt.planning.budget: 1.5 h at the report's 38% MFU, 5.1 h
# for the report's 420M; PCIe A100s are power-capped); 24 SU per hour. If the job dies, resubmit the same command: verl resumes from the
# last save (trainer.save_freq). Then evaluate: sbatch scripts/slurm/eval.slurm sft
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/slurm/env.sh
source scripts/slurm/lib.sh
CONFIG="${1:-sft/sft_9b.yaml}"; shift || true
export HF_HUB_OFFLINE=1
HEAD="$(job_head)"
MASTER="$(getent hosts "${HEAD}" 2>/dev/null | awk '{print $1; exit}' || true)"; MASTER="${MASTER:-${HEAD}}"
N="$(job_nodes | wc -l)"
tlog "SFT ${CONFIG} on ${N} nodes, master ${HEAD} (${MASTER})"
PIDS=(); i=0
for h in $(job_nodes); do
  VARS=(NNODES="${N}" NODE_RANK="${i}" MASTER_ADDR="${MASTER}" MASTER_PORT=29500)
  if [[ ${i} -eq 0 ]]; then
    run env "${VARS[@]}" bash scripts/04_sft.sh --config "${CONFIG}" "$@" &      # node 0 also exports to HF
  else
    on_node "${h}" env "${VARS[@]}" bash scripts/04_sft.sh --config "${CONFIG}" "$@" &
  fi
  PIDS+=($!); i=$((i + 1))
done
RC=0
for p in "${PIDS[@]}"; do wait "${p}" || RC=1; done
[[ ${RC} -eq 0 ]] || tdie "SFT failed on at least one node (this log has every node's output)"
tlog "SFT done. Copy the export to \$WORK (SCRATCH is purged after 10 days) and evaluate it."
