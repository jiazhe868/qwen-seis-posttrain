#!/bin/bash
#SBATCH -J dpt-smoke
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 06:00:00
#SBATCH -o dpt-smoke-%j.out
# Step 05 on one cluster node: the 0.8B end-to-end RL smoke test + resume equivalence, on 2 of the
# node's 3 GPUs (TP1 × PP2 × DP1, 2 vLLM replicas). It proves the env, the adapters, the A100 mask fix,
# the container sandbox and checkpoint/resume on the cluster before any 8-node job. ~2 h, ≤ 18 SU.
#
#   sbatch -A <alloc> scripts/slurm/smoke.slurm
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/slurm/env.sh
source scripts/slurm/lib.sh
export HF_HUB_OFFLINE=1
DPT_SMOKE_GPUS=2 bash scripts/05_rl_smoke.sh
