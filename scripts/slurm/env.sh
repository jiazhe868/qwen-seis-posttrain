# shellcheck shell=bash
# Environment for every cluster job (and for your login shell):  source scripts/slurm/env.sh
#
# Everything site-specific lives here, so the job scripts and the repo's numbered scripts stay the
# same on every machine. Override any variable by exporting it before sourcing.
#
#   $WORK     (1 TB, not purged)   repo, conda, models (HF cache), data, eval sets, results, sandbox image
#   $SCRATCH  (no quota, purged after 10 days without access)   checkpoints only
# Do not "touch" files to dodge the $SCRATCH purge (site policy); copy what you keep to $WORK.

# --- project selection: the Qwen3.5-9B main line on the cluster's gpu-a100 nodes -------------------
export DPT_HW_PROFILE="${DPT_HW_PROFILE:-slurm_3xa100_40g}"
export DPT_MODEL_LINE="${DPT_MODEL_LINE:-qwen35_9b}"
export DPT_REPO="${DPT_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="${DPT_REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"

# --- Python: one conda env (Miniforge in $WORK; $HOME's 10 GB quota is too small) -----------------
export DPT_CONDA="${DPT_CONDA:-${WORK}/miniforge3}"
export DPT_ENV="${DPT_ENV:-dpt-train}"
# The workstation splits GPU work over three envs; on the cluster dpt-train has all of it
# (torch + flash-linear-attention + vLLM + verl/Megatron), so scripts/02_parity.sh uses it for every step.
export DPT_GPU_ENV="${DPT_GPU_ENV:-${DPT_ENV}}" DPT_VLLM_ENV="${DPT_VLLM_ENV:-${DPT_ENV}}"
export DPT_TRAIN_ENV="${DPT_TRAIN_ENV:-${DPT_ENV}}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${WORK}/conda_pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${WORK}/.cache/pip}"

# --- login nodes: HTTP clients with many threads (bank verification, eval-set checks) --------------
# 100 MB thread stacks under the 8 GB virtual-memory cap allow only ~48 threads per process; 8 MB stacks,
# two malloc arenas and single-threaded BLAS / tokenizers allow 160.
if [[ "$(hostname -s 2>/dev/null)" == login* ]]; then
  ulimit -s 8192 2>/dev/null || true
  export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
fi

# --- models, logs, caches -------------------------------------------------------------------------
export HF_HOME="${HF_HOME:-${WORK}/hf}"           # download on a login node: scripts/slurm/setup_login.sh
# `hf auth login` in a shell without this file stores the token under ~/.cache/huggingface; keep using it
# (gated datasets such as GPQA) even though the cache is in $WORK.
if [[ -z "${HF_TOKEN_PATH:-}" && ! -f "${HF_HOME}/token" && -f "${HOME}/.cache/huggingface/token" ]]; then
  export HF_TOKEN_PATH="${HOME}/.cache/huggingface/token"
fi
export WANDB_MODE="${WANDB_MODE:-offline}"        # compute nodes may have no outbound network; `wandb sync` later
export WANDB_DIR="${WANDB_DIR:-${WORK}/dpt/wandb}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_${USER}}"   # node-local: Lustre is slow for many small files
# Megatron-Bridge takes a file lock while it reads a model config, by default under ~/.cache/huggingface. $HOME
# is NFSv3 on the cluster, and on 2026-10-28 that lock hung one node's three SFT ranks in the kernel: the run never
# started. Node-local /tmp has no NFS lock (offline, so nothing is downloaded and the nodes need not agree).
export MEGATRON_CONFIG_LOCK_DIR="${MEGATRON_CONFIG_LOCK_DIR:-/tmp/megatron_locks_${USER}}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${WORK}/.cache/vllm}"     # torch.compile cache; not $HOME (10 GB quota)
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${WORK}/.cache}"
# nvcc for kernels compiled at run time (FlashInfer all-reduce, …). Only CUDA_HOME and PATH: the module's
# lib64 on LD_LIBRARY_PATH would shadow the env's own CUDA runtime libraries.
if [[ -z "${CUDA_HOME:-}" ]]; then
  for _cd in /opt/apps/cuda/12.8 /usr/local/cuda; do
    if [[ -x "${_cd}/bin/nvcc" ]]; then export CUDA_HOME="${_cd}" PATH="${PATH}:${_cd}/bin"; break; fi
  done
  unset _cd
fi
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_${USER}}"                  # short path: Ray's unix sockets max 107 chars
# Ray 2.59 authenticates a cluster with the token in ~/.ray/auth_token (mode 600; $HOME is shared by the nodes), so
# nobody else on the network can run work in it. Every Ray process must use the same mode: a token file left by a
# local Ray made the head require a token that the workers, in the default mode, did not send (2026-10-29).
# lib.sh's ray_up creates the token when it is missing.
export RAY_AUTH_MODE="${RAY_AUTH_MODE:-token}"
export DPT_ENDPOINTS="${DPT_ENDPOINTS:-${WORK}/dpt/endpoints}"       # serve.slurm writes <name>.url / <name>.model
export DPT_SERVE_TP="${DPT_SERVE_TP:-1}"          # scripts/serve_vllm.sh default: the 9B fits one GPU

# --- T2 code sandbox: model-written code runs in Apptainer, seeing only its own directory ----------
export DPT_SANDBOX_SIF="${DPT_SANDBOX_SIF:-${WORK}/dpt/sandbox/sandbox.sif}"
export DPT_SANDBOX_PYTHON="${DPT_SANDBOX_PYTHON:-${DPT_CONDA}/envs/dpt-sandbox/bin/python}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${WORK}/.apptainer_cache}"   # not $HOME (10 GB quota)

# --- second D-Report judge (GPT-6 Luna through the OpenAI API; eval side only) -----------------------
# Keep the key out of the repo and out of job scripts: one line in a file only you can read
#   mkdir -p ~/.config/dpt && (umask 077; printf '%s\n' 'sk-...' > ~/.config/dpt/openai_key)
if [[ -z "${OPENAI_API_KEY:-}" && -r "${HOME}/.config/dpt/openai_key" ]]; then
  OPENAI_API_KEY="$(head -1 "${HOME}/.config/dpt/openai_key")"; export OPENAI_API_KEY
fi

# --- modules and activation -------------------------------------------------------------------------
# Lmod modules and conda's (de)activate hooks reference unset variables (apptainer's bash completion
# reads BASH_COMPLETION_DEBUG): under `set -u` that kills the calling job script, even with `|| true`.
case $- in *u*) _dpt_u=1; set +u ;; *) _dpt_u=0 ;; esac
if command -v module >/dev/null 2>&1; then
  module load apptainer >/dev/null 2>&1 || true
fi
# A shell started over ssh (lib.sh on_node: the Ray workers) has Lmod stubbed out (LMOD_CMD=:), so the load above
# does nothing there. RL's reward workers on the 7 worker nodes then found no apptainer and the T2 sandbox failed
# closed: 79% of the dev T2 samples of campaign 3532660 scored 0 as infra errors (2026-10-29). Use the default
# module's install (the version a batch job loads), as the module would.
_dpt_appt="${DPT_APPTAINER_BIN:-/opt/apps/apptainer/1.1.8/bin}"
if [[ "$(hostname -s 2>/dev/null)" != login* ]] && ! command -v apptainer >/dev/null 2>&1 && [[ -x "${_dpt_appt}/apptainer" ]]; then
  export PATH="${_dpt_appt}:${PATH}"
fi
unset _dpt_appt
if [[ -f "${DPT_CONDA}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${DPT_CONDA}/etc/profile.d/conda.sh"
  conda activate "${DPT_ENV}" 2>/dev/null || echo "env.sh: conda env ${DPT_ENV} not built yet (sbatch scripts/slurm/build_env.slurm)" >&2
else
  echo "env.sh: no conda at ${DPT_CONDA} (install Miniforge there, see scripts/slurm/README.md §1)" >&2
fi
[[ ${_dpt_u} == 1 ]] && set -u
unset _dpt_u

# --- compute nodes: the env's own libstdc++ first ---------------------------------------------------
# FlashInfer's JIT kernels (vLLM's sampler) need GLIBCXX_3.4.26, which the system /lib64/libstdc++.so.6 lacks.
# A batch job inherits a newer one from the login shell's modules; a shell started over ssh (campaign.sh
# on_node: vLLM replicas, SFT workers, Ray) does not, and every vLLM server of campaign 3532660 died at
# startup (2026-10-27). Preloading this one library shadows nothing else.
if [[ "$(hostname -s 2>/dev/null)" != login* && -f "${CONDA_PREFIX:-}/lib/libstdc++.so.6" ]]; then
  case ":${LD_PRELOAD:-}:" in
    *":${CONDA_PREFIX}/lib/libstdc++.so.6:"*) ;;
    *) export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}" ;;
  esac
fi
