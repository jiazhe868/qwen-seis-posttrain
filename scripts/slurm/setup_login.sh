#!/usr/bin/env bash
# Light, network-bound setup on a cluster LOGIN node (no compiling, no GPU work: site policy).
#
#   bash scripts/slurm/setup_login.sh conda     # Miniforge into $WORK/miniforge3 (~5 min)
#   bash scripts/slurm/setup_login.sh models    # HF downloads into $HF_HOME = $WORK/hf (~80 GB, ~1 h)
#   bash scripts/slurm/setup_login.sh sandbox   # T2 sandbox interpreter env (~5 min; check_node.slurm pulls the image)
#   bash scripts/slurm/setup_login.sh all
#
# Models: the 0.8B (smoke), the 9B base (the main line), the 27B teacher, and the judge (Gemma 4 26B-A4B:
# T3 reward, primary D-Report judge, second answer-verification family; plan §5), and the embedding model
# for decontamination (Qwen3-Embedding-0.6B, run in-process by the data job).
set -euo pipefail
source "$(dirname "$0")/env.sh"
WHAT="${1:?conda | models | sandbox | all}"
say() { printf '\033[1;34m== %s\033[0m\n' "$*"; }

do_conda() {
  if [[ -x "${DPT_CONDA}/bin/conda" ]]; then say "Miniforge already at ${DPT_CONDA}"; return; fi
  say "installing Miniforge into ${DPT_CONDA}"
  local inst="${WORK}/Miniforge3-Linux-x86_64.sh"
  curl -fsSL -o "${inst}" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash "${inst}" -b -p "${DPT_CONDA}"; rm -f "${inst}"
  "${DPT_CONDA}/bin/conda" config --set auto_activate_base false
  "${DPT_CONDA}/bin/pip" install -q "huggingface_hub[cli]>=0.30"
  say "done. Next: sbatch -A <alloc> scripts/slurm/build_env.slurm"
}

do_models() {
  local hf="${DPT_CONDA}/bin/hf"; [[ -x "${hf}" ]] || hf="$(command -v hf || command -v huggingface-cli)"
  local models="${DPT_MODELS:-Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-9B Qwen/Qwen3.8-27B RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic Qwen/Qwen3-Embedding-0.6B}"
  # Login nodes cap virtual memory at 8 GB (ulimit -v); the hf-xet downloader aborts past it
  # ("memory allocation ... failed"). Plain HTTP stays well under it. A rerun resumes from the cache.
  export HF_HUB_DISABLE_XET=1
  for m in ${models}; do
    say "download ${m} → ${HF_HOME}"
    "${hf}" download "${m}" --max-workers 4 --exclude "*.pth" --exclude "original/*"   # one pattern per flag
  done
  du -sh "${HF_HOME}" 2>/dev/null || true
}

do_sandbox() {
  # The interpreter that runs model-written code: a small, separate env (no torch, no project code),
  # bound read-only into the container. Add a package here only if T2 tasks need it.
  if [[ ! -x "${DPT_SANDBOX_PYTHON}" ]]; then
    say "sandbox interpreter env dpt-sandbox"
    "${DPT_CONDA}/bin/conda" create -y -q -n dpt-sandbox -c conda-forge python=3.12 numpy scipy pandas
  fi
  # The image (a minimal OS: the container contributes only glibc and /bin; the interpreter is bound in) is
  # pulled by check_node.slurm on a compute node: on login nodes `apptainer` is a stub that does nothing.
  say "sandbox interpreter ready. check_node.slurm pulls the image and verifies the isolation on a compute node"
}

case "${WHAT}" in
  conda) do_conda ;; models) do_models ;; sandbox) do_sandbox ;;
  all) do_conda; do_models; do_sandbox ;;
  *) echo "usage: $0 conda | models | sandbox | all"; exit 2 ;;
esac
