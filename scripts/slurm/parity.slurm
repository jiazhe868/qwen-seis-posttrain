#!/bin/bash
#SBATCH -J dpt-parity
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 04:00:00
#SBATCH -o dpt-parity-%j.out
# Step 02 on a cluster A100 node (SM80, same as the workstation; the PCIe parts have a different
# driver/CUDA stack, so the numerics are re-checked here). ~1-2 h, ≤ 12 SU.
#
#   sbatch -A <alloc> scripts/slurm/parity.slurm [model]      # default: the 9B base
#
# 1. GDN Triton kernels vs FP64 on SM80   2. vLLM vs HF logprobs, FP32 state (must PASS)
# 3. same with BF16 state (expected FAIL)  4. Megatron train-mode vs eval-mode forward (must PASS:
#    the A100 attention-mask bug; dpt.verl_ext.patches) — on the 9B itself, same head_dim 256.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/slurm/env.sh
source scripts/slurm/lib.sh
export HF_HUB_OFFLINE=1
MODEL="${1:-$(python -c 'from dpt.common.config import load_config; print(load_config().project.model.base)')}"
export DPT_TRAIN_PARITY_MODEL="${DPT_TRAIN_PARITY_MODEL:-${MODEL}}"
bash scripts/02_parity.sh "${MODEL}"
tlog "results in ${DPT_REPO}/parity/$(date +%Y%m%d)/ — commit them with a SUMMARY.md (see parity/20261007/)"
