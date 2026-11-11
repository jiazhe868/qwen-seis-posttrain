#!/usr/bin/env bash
# Build the local training env `dpt-train`: verl 0.9.1 + Megatron-Core/Bridge + vLLM + Transformer
# Engine, for the workstation (A100 SM80, NVIDIA driver 560 → CUDA ≤ 12.6 natively).
#
#   bash scripts/00_setup_env.sh train-local      (or run this file directly)
#   the cluster: sbatch -A <alloc> scripts/slurm/build_env.slurm   (same script on a compute node)
#
# Site settings (environment): CUDA_TOOLKIT (nvcc for the TE build; default $CUDA_HOME, else
# /usr/local/cuda-12.8), CUDA_COMPAT (auto | yes | no: CUDA 12.9 forward compatibility, needed only
# when the driver supports CUDA < 12.9; on a newer driver the compat libcuda would break cuInit),
# NVTE_CUDA_ARCHS (default 80 = A100), MAX_JOBS.
#
# Why it differs from verl's own `verl[megatron]` extra (which targets CUDA 13):
#   * CUDA 13 forward compatibility is REJECTED by driver 560 (cuInit → 803), CUDA 12.9 is accepted.
#     So: torch 2.11.0+cu128, vLLM 0.24.0+cu129, conda `cuda-compat=12.9` (activate hook sets
#     LD_LIBRARY_PATH), and Transformer Engine's PyTorch extension compiled here against the system
#     CUDA 12.8 toolkit (TE publishes no cu12 build of transformer_engine_torch).
#   * The verl 0.9.1 wheel does not pin megatron-bridge, and 0.5.2 (in this repo's first notes) was
#     never published. 0.6.2 provides every megatron.bridge API verl imports (checked) and was
#     released with megatron-core 0.19.2.
#   * Python 3.12 (megatron-core ≥ 0.17 dropped 3.11).
#   * megatron-core's [dev] extra is skipped. It source-builds several CUDA packages we do not use.
#     Qwen3.5/3.8 GDN layers need only flash-linear-attention; the model spec needs Transformer Engine.
#   * flash-attn 2.8.3 (TE accepts < 2.8.4) is compiled for NVTE_CUDA_ARCHS only: 8 min with 32 jobs on a
#     the cluster node. On A100 at head_dim 256, TE's cuDNN attention cannot train and its unfused fallback
#     ran out of memory above ~13K tokens (2026-10-28), so flash-attn is the only training backend there.
#     Configs set attention_backend=auto and TE picks it.
set -euo pipefail
ENV_NAME="${ENV_NAME:-dpt-train}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CUDA_TOOLKIT="${CUDA_TOOLKIT:-${CUDA_HOME:-/usr/local/cuda-12.8}}"
MAX_JOBS="${MAX_JOBS:-32}"
step() { printf '\n\033[1;34m[%s] == %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
[[ -x "${CUDA_TOOLKIT}/bin/nvcc" ]] || { echo "no nvcc at ${CUDA_TOOLKIT}/bin (set CUDA_TOOLKIT, e.g. module load cuda/12.8)"; exit 1; }
# CUDA forward compatibility: only when the driver's CUDA version is older than the 12.9 runtime vLLM needs.
CUDA_COMPAT="${CUDA_COMPAT:-auto}"
if [[ "${CUDA_COMPAT}" == auto ]]; then
  DRV="$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9.]*' | awk '{print $3}' || true)"
  [[ -n "${DRV}" ]] || { echo "nvidia-smi not available: build on a GPU node (CUDA_COMPAT=auto needs the driver)"; exit 1; }
  CUDA_COMPAT=$(awk -v v="${DRV}" 'BEGIN { split(v, a, "."); print (a[1] < 12 || (a[1] == 12 && a[2] < 9)) ? "yes" : "no" }')
  echo "driver supports CUDA ${DRV} → cuda-compat: ${CUDA_COMPAT}"
fi
COMPAT_PKG=(); [[ "${CUDA_COMPAT}" == yes ]] && COMPAT_PKG=("cuda-compat=12.9")

set +u   # conda hooks of other packages reference unset variables
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# Python 3.12: megatron-core ≥ 0.17 requires it (the other dpt envs can stay on 3.11).
step "1/8 conda env ${ENV_NAME} (python 3.12${COMPAT_PKG:+, ${COMPAT_PKG[*]}})"
if conda env list | grep -q "^${ENV_NAME} "; then
  PYV="$(conda run -n "${ENV_NAME}" python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  [[ "${PYV}" == "3.12" ]] || { echo "recreating ${ENV_NAME} (python ${PYV} → 3.12)"; conda env remove -y -q -n "${ENV_NAME}"; }
fi
conda env list | grep -q "^${ENV_NAME} " || conda create -y -q -n "${ENV_NAME}" -c conda-forge python=3.12 "${COMPAT_PKG[@]}"
# The hooks are written before the first activation: an old hook from an earlier build would otherwise run
# (and, under this script's `set -e`, a failing line in it would stop the build silently).
PREFIX="$(conda info --base)/envs/${ENV_NAME}"
mkdir -p "${PREFIX}/etc/conda/activate.d" "${PREFIX}/etc/conda/deactivate.d"
cat > "${PREFIX}/etc/conda/activate.d/cuda_compat.sh" <<'HOOK'
# dpt-train runtime library order (see scripts/env/build_train_local.sh):
#  1. CUDA forward-compat libcuda (driver 560 → CUDA 12.9 runtime)
#  2. the env's pip cuDNN / NCCL, ahead of the OLDER system copies in /usr/lib/x86_64-linux-gnu
#     (Transformer Engine loads cuDNN by name and refuses cuDNN < 9.12; the system one is 9.10)
export DPT_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
_dpt_libs=""
[ -d "${CONDA_PREFIX}/cuda-compat" ] && _dpt_libs="${CONDA_PREFIX}/cuda-compat"   # only if installed (old driver)
for _d in "${CONDA_PREFIX}"/lib/python3.*/site-packages/nvidia/cudnn/lib \
          "${CONDA_PREFIX}"/lib/python3.*/site-packages/nvidia/nccl/lib; do
  [ -d "${_d}" ] && _dpt_libs="${_dpt_libs:+${_dpt_libs}:}${_d}"
done
export LD_LIBRARY_PATH="${_dpt_libs}${LD_LIBRARY_PATH:+${_dpt_libs:+:}${LD_LIBRARY_PATH}}"
_c=$(ls -d "${CONDA_PREFIX}"/lib/python3.*/site-packages/nvidia/cudnn 2>/dev/null | head -1 || true)   # none yet: ok
[ -n "${_c}" ] && export CUDNN_PATH="${_c}"
unset _dpt_libs _d _c
HOOK
cat > "${PREFIX}/etc/conda/deactivate.d/cuda_compat.sh" <<'HOOK'
export LD_LIBRARY_PATH="${DPT_OLD_LD_LIBRARY_PATH:-}"; unset DPT_OLD_LD_LIBRARY_PATH CUDNN_PATH
HOOK
conda activate "${ENV_NAME}"
# setuptools: ≥ 80 for nvidia-modelopt, < 81 for vLLM 0.24 on Python 3.12
python -m pip install -q -U pip "setuptools>=80,<81" wheel ninja pybind11 packaging

PIN=(--extra-index-url https://download.pytorch.org/whl/cu128 "torch==2.11.0" "torchvision==0.26.0"
     "torchaudio==2.11.0" "transformers==5.9.0" "tensordict==0.10.0" "numpy>=2,<2.3"
     "setuptools>=80,<81" "fsspec[http]<=2026.7.0" "s3fs==2026.7.0")

step "2/8 torch 2.11.0+cu128 and vLLM 0.24.0+cu129"
pip install -q "${PIN[@]}"
pip install -q "${PIN[@]}" \
  "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.24.0/vllm-0.24.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"

step "3/8 verl 0.9.1 (+ TransferQueue, Ray) and mbridge"
pip install -q "${PIN[@]}" "verl[verl-core]==0.9.1" mbridge "TransferQueue>=0.1.10" cupy-cuda12x

step "4/8 Megatron-Core 0.19.2 + Megatron-Bridge 0.6.2 (curated deps, no [dev] extra)"
pip install -q "${PIN[@]}" "megatron-core==0.19.2" "flash-linear-attention==0.5.2" \
  "peft>=0.18.1" datasets accelerate einops omegaconf tensorboard rich wandb six regex pyyaml tqdm \
  "hydra-core>=1.3.4,<1.4" qwen-vl-utils timm open-clip-torch mistral-common sentencepiece tiktoken \
  "megatron-energon~=7.0" "nvidia-modelopt>=0.44"
# ^ both are imported unconditionally on paths verl uses: megatron.bridge.training.config pulls the
#   Energon data builders; megatron.bridge.training.checkpointing imports modelopt (found by the smoke run).
pip install -q --no-deps "megatron-bridge==0.6.2"

step "5/8 Transformer Engine 2.20.2 (cu12 core binary + PyTorch extension) and flash-attn 2.8.3, built for SM80"
SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
pip install -q "transformer_engine_cu12==2.20.2" "nvidia-cudnn-frontend>=1.29" onnx onnxscript nvdlfw-inspect pydantic
export CUDA_HOME="${CUDA_TOOLKIT}" PATH="${CUDA_TOOLKIT}/bin:${PATH}"
export CUDNN_PATH="${SITE}/nvidia/cudnn" NCCL_HOME="${SITE}/nvidia/nccl"
export NVTE_FRAMEWORK=pytorch NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS:-80}" MAX_JOBS
export CPLUS_INCLUDE_PATH="${CUDNN_PATH}/include:${NCCL_HOME}/include:${CPLUS_INCLUDE_PATH:-}"
pip install -v --no-build-isolation --no-deps "transformer_engine_torch==2.20.2" 2>&1 | tail -20
# TE's import guard checks for the `transformer-engine` meta-distribution (metadata only, no deps).
pip install -q --no-deps "transformer-engine==2.20.2"
# Version knot, resolved in favor of vLLM (it pins exactly; TE only needs these at build time / optionally):
#   TE metadata wants nvidia-cudnn-frontend>=1.29, which hard-requires cutlass-dsl>=4.6.2 and tvm-ffi>=0.1.11;
#   vLLM 0.24 pins cutlass-dsl==4.5.2 and tvm-ffi==0.1.9. TE compiled the cudnn-frontend headers into its
#   extension above; its CuTeDSL backend (the only user of cutlass-dsl/tvm-ffi) is optional and falls back
#   to TE's CUDA kernels. `pip check` therefore reports one expected conflict for transformer-engine-torch.
pip install -q --no-deps "nvidia-cudnn-frontend==1.28.0" "apache-tvm-ffi==0.1.9" \
  "nvidia-cutlass-dsl==4.5.2" "nvidia-cutlass-dsl-libs-base==4.5.2"
pip uninstall -y -q nvidia-cutlass-dsl-libs-cu12 nvidia-cutlass-dsl-libs-cu13 2>/dev/null || true
# flash-attn has no wheel for torch 2.11: build it, for the TE archs only, without touching torch.
FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS="${NVTE_CUDA_ARCHS}" NVCC_THREADS=2 \
  pip install -v --no-build-isolation --no-deps "flash-attn==2.8.3" 2>&1 | tail -5

step "6/8 this repo (editable)"
pip install -q --no-deps -e "${REPO}"

conda deactivate; conda activate "${ENV_NAME}"   # re-run the hook: pip nvidia lib dirs now exist
step "7/8 import checks"
python - <<'PY'
import importlib, torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "| GPUs visible:", torch.cuda.is_available(), torch.cuda.device_count())
for m in ["vllm", "verl", "transfer_queue", "megatron.core", "megatron.bridge", "transformer_engine.pytorch",
          "flash_attn", "fla", "tensordict", "ray", "transformers"]:
    mod = importlib.import_module(m)
    print(f"  ok  {m:28s} {getattr(mod, '__version__', '')}")
from megatron.bridge import AutoBridge                                         # noqa: F401
# every megatron.bridge import verl 0.9.1 makes (grep of the verl source)
from megatron.bridge.training.utils.train_utils import LinearForLastLayer, freeze_moe_router, make_value_model  # noqa
from megatron.bridge.training.utils.config_utils import create_ddp_config                                   # noqa
from megatron.bridge.training.checkpointing import save_fsdp_dtensor_checkpoint, apply_peft_adapter_filter_to_state_dict  # noqa
from megatron.bridge.peft.utils import create_peft_hook, load_peft_adapter_checkpoint, create_peft          # noqa
from megatron.bridge.models.gpt_provider import modelopt_transformer_layer_spec                            # noqa
from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping, AutoMapping              # noqa
from megatron.bridge.models.conversion.model_bridge import _megatron_local_name_to_global                  # noqa
from megatron.bridge.models.qwen_vl.qwen35_vl_provider import Qwen35VLModelProvider  # noqa: F401
from verl.trainer.ppo.v1.trainer_colocate_async import PPOTrainerColocateAsync  # noqa: F401
import dpt.verl_ext.register                                                    # noqa: F401
from verl.trainer.ppo.v1.trainer_base import TRAINER_REGISTRY
print("  dpt trainer registered:", "dpt_colocate_async" in TRAINER_REGISTRY)
PY

step "8/8 pip check (conflicts are reported, not fatal)"
pip check || true
echo "done: conda activate ${ENV_NAME}"
