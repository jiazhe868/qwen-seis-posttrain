#!/usr/bin/env bash
# Shared helpers sourced by every numbered script. Never run directly.
set -euo pipefail

DPT_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DPT_REPO
export PYTHONPATH="${DPT_REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
# Hardware profile: local_4xa100_40g (workstation) | cluster_2x8xa100_80g (production).
export DPT_HW_PROFILE="${DPT_HW_PROFILE:-local_4xa100_40g}"
PY="${PYTHON:-python}"

log()  { printf '\033[1;34m[%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '\033[1;33m[%s] WARN %s\033[0m\n' "$(date +%H:%M:%S)" "$*" >&2; }
die()  { printf '\033[1;31m[%s] FAIL %s\033[0m\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }

require_profile() {   # require_profile cluster_2x8xa100_80g
  [[ "${DPT_HW_PROFILE}" == "$1" ]] || die "this step needs DPT_HW_PROFILE=$1 (current: ${DPT_HW_PROFILE})"
}

# Read a resolved config value: cfg project.model.base
cfg() {
  "${PY}" - "$1" <<'PYEOF'
import sys
from omegaconf import OmegaConf
from dpt.common.config import load_config
print(OmegaConf.select(load_config(), sys.argv[1]))
PYEOF
}

# Wait until an OpenAI-compatible server answers /v1/models.
wait_for_server() {   # wait_for_server http://localhost:8000 [timeout_s]
  local url="$1" t="${2:-1800}" i=0
  until curl -sf "${url}/v1/models" >/dev/null; do
    sleep 10; i=$((i + 10)); [[ $i -lt $t ]] || die "server ${url} not up after ${t}s"
  done
}

# Python interpreter of a conda env by name (empty if the env does not exist).
env_python() {   # env_python dpt-gpu
  local base; base="$(conda info --base 2>/dev/null)" || return 0
  local py="${base}/envs/$1/bin/python"
  [[ -x "${py}" ]] && echo "${py}"
}

# Run a command inside a conda env's environment, including its activate.d hooks
# (dpt-vllm puts the CUDA forward-compat libcuda on LD_LIBRARY_PATH there).
in_env() {       # in_env dpt-vllm python -m ...
  local env="$1"; shift
  # `set +u`: conda's (de)activate hooks of other packages (e.g. geotiff's) reference unset variables
  # and would abort the activation under this repo's strict `set -u`.
  # shellcheck disable=SC1091
  ( set +u; source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "${env}" && "$@" )
}

# Pick the N GPUs with the most free memory (the workstation is shared with other users' jobs).
pick_gpus() {    # pick_gpus 1  → "2"
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -nr | head -"${1:-1}" | cut -d, -f1 | sort -n | paste -sd, -
}

# Number of GPUs with at least MIN_MIB free; and the N freest GPUs having at least MIN_MIB free.
count_free() {   # count_free 26000
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk -v m="$1" '$1 >= m' | wc -l
}
free_gpus() {    # free_gpus 2 24000 → "0,2"
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F', ' -v m="$2" '$2 >= m' | sort -t, -k2 -nr | head -"$1" | cut -d, -f1 | sort -n | paste -sd, -
}

# Free memory (MiB) of the least-free GPU among a comma list.
min_free_mib() { # min_free_mib 0,1
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F', ' -v ids=",$1," 'index(ids, ","$1",") {print $2}' | sort -n | head -1
}
