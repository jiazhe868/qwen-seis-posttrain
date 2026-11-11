#!/bin/bash
#SBATCH -J dpt-build
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 03:00:00
#SBATCH -o dpt-build-%j.out
# Build the training env `dpt-train` on a cluster GPU node (1 A100, 1.5 SU/h; ~1-2 h, most of it
# compiling Transformer Engine's PyTorch extension for SM80). Same script as the workstation:
# scripts/env/build_train_local.sh, which decides on CUDA forward compatibility from this node's driver.
#
#   sbatch -A <alloc> scripts/slurm/build_env.slurm
#
# Needs outbound network from the compute node (pip). check_node.slurm reports whether it exists;
# if it does not, see scripts/slurm/README.md §2 (wheelhouse on a login node, then --no-index here).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/slurm/env.sh
source scripts/slurm/lib.sh
curl -s -o /dev/null --max-time 10 https://pypi.org/simple/pip/ || tdie "no network from this compute node (README §2)"
# nvcc for the TE build: the newest CUDA 12.x module ≥ 12.8 (torch 2.11 is built with 12.8).
if [[ -z "${CUDA_TOOLKIT:-}" ]] && command -v module >/dev/null 2>&1; then
  for v in 12.9 12.8; do module load "cuda/${v}" >/dev/null 2>&1 && break; done
  [[ -n "${CUDA_HOME:-}" ]] || tdie "no cuda/12.8+ module; set CUDA_TOOLKIT to a CUDA ≥ 12.8 toolkit (README §2)"
fi
export ENV_NAME="${DPT_ENV}" NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS:-80}" MAX_JOBS="${MAX_JOBS:-32}"
bash scripts/env/build_train_local.sh
tlog "dpt-train built; next: setup_login.sh models + sandbox (login node), then check_node.slurm"
