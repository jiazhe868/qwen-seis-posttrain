#!/usr/bin/env bash
# STEP 00: create the Python environments and check the machine.
#
#   bash scripts/00_setup_env.sh core        # CPU only: tests, data pipeline, graders, planner, dry runs
#   bash scripts/00_setup_env.sh gpu-local   # workstation GPU env: torch 2.11 (CUDA 12.6) + Triton GDN kernels
#   bash scripts/00_setup_env.sh vllm-local  # workstation vLLM env: vLLM 0.24 (CUDA 12.9) via CUDA forward compat
#   bash scripts/00_setup_env.sh train-local # workstation training env: verl 0.9.1 + Megatron-Bridge + vLLM + TE
#   bash scripts/00_setup_env.sh train       # cluster nodes: the full verl + Megatron + vLLM stack
#
# Why two workstation envs. The workstation driver (560.35) supports CUDA ≤ 12.6.
#   * PyTorch publishes CUDA 12.6 builds of torch 2.11 (verl 0.9.1's pin), so torch and the
#     flash-linear-attention Triton kernels run natively → `gpu-local` (GDN kernel parity, HF scoring).
#   * vLLM publishes only CUDA ≥ 12.9 wheels. A100 is a datacenter GPU, so NVIDIA's CUDA
#     forward-compatibility library works on this driver: conda's `cuda-compat=12.9` provides a
#     user-space libcuda (575.x) that the 560 kernel driver accepts (cuInit → OK, API 12.9). `vllm-local`
#     installs it and an activate.d hook that puts it on LD_LIBRARY_PATH (no root, no driver upgrade).
# Each env is separate on purpose. Your base env and other envs are never modified.
source "$(dirname "$0")/common.sh"
KIND="${1:-core}"
# conda's activate hooks of some packages reference unset variables; relax `set -u` for this script.
set +u
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

case "${KIND}" in
  core|train)
    ENV_NAME="${ENV_NAME:-dpt-${KIND}}"
    conda env list | grep -q "^${ENV_NAME} " || conda create -y -n "${ENV_NAME}" python=3.11
    conda activate "${ENV_NAME}"
    pip install -U pip
    pip install -r "${DPT_REPO}/requirements/${KIND}.txt"
    ;;
  gpu-local)
    ENV_NAME="${ENV_NAME:-dpt-gpu}"
    conda env list | grep -q "^${ENV_NAME} " || conda create -y -n "${ENV_NAME}" python=3.11
    conda activate "${ENV_NAME}"
    pip install --index-url https://download.pytorch.org/whl/cu126 "torch==2.11.0"
    pip install "flash-linear-attention==0.5.2" accelerate -r "${DPT_REPO}/requirements/core.txt"
    ;;
  vllm-local)
    ENV_NAME="${ENV_NAME:-dpt-vllm}"
    conda env list | grep -q "^${ENV_NAME} " || conda create -y -n "${ENV_NAME}" -c conda-forge python=3.11 "cuda-compat=12.9"
    conda activate "${ENV_NAME}"
    mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d" "${CONDA_PREFIX}/etc/conda/deactivate.d"
    cat > "${CONDA_PREFIX}/etc/conda/activate.d/cuda_compat.sh" <<'HOOK'
# CUDA forward compatibility: newer user-space libcuda for the older kernel driver (dpt vllm-local env)
export DPT_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/cuda-compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
HOOK
    cat > "${CONDA_PREFIX}/etc/conda/deactivate.d/cuda_compat.sh" <<'HOOK'
export LD_LIBRARY_PATH="${DPT_OLD_LD_LIBRARY_PATH:-}"; unset DPT_OLD_LD_LIBRARY_PATH
HOOK
    conda deactivate; conda activate "${ENV_NAME}"          # apply the hook
    pip install --extra-index-url https://download.pytorch.org/whl/cu128 "torch==2.11.0"
    pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
      "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.24.0/vllm-0.24.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
    pip install accelerate omegaconf httpx regex pytest "flash-linear-attention==0.5.2"  # fast GDN for HF scoring
    ;;
  train-local)
    exec bash "${DPT_REPO}/scripts/env/build_train_local.sh" ;;
  *) die "unknown target ${KIND} (core | gpu-local | vllm-local | train-local | train)" ;;
esac
pip install -e "${DPT_REPO}"
log "environment report (${ENV_NAME})"
python -m dpt.launch.check_env $([[ "${KIND}" == train ]] && echo --require-train) --json "${DPT_REPO}/env_report_${ENV_NAME}.json" || true
if [[ "${KIND}" == core ]]; then (cd "${DPT_REPO}" && python -m pytest -q); fi
