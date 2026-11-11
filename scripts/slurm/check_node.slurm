#!/bin/bash
#SBATCH -J dpt-check
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:30:00
#SBATCH -o dpt-check-%j.out
# Facts about the cluster's compute nodes that the plan depends on, in one 30-minute dev job (≤ 1.5 SU):
#
#   sbatch -A <alloc> scripts/slurm/check_node.slurm
#
#   driver / CUDA version      → whether the env needs cuda-compat (build_env.slurm decides automatically)
#   outbound network           → pip on compute nodes; Crossref for T3 citation checks; HF downloads;
#                                the OpenAI API (the second D-Report judge runs inside eval jobs only if reachable)
#   GPU topology               → confirms PCIe (no NVLink), so TP1 everywhere
#   ssh between job nodes      → needed by sft.slurm / ray_up (checked in the 2-node variant, see README)
#   check_env                  → package versions vs the verified ones (after build_env.slurm)
#   sandbox isolation          → pulls the sandbox image if missing, then dpt.rewards.sandbox_check
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/slurm/env.sh
source scripts/slurm/lib.sh
OUT="${WORK}/dpt/node_check_$(date +%Y%m%d_%H%M%S)"; mkdir -p "${OUT}"
section() { printf '\n==== %s\n' "$*"; }
{
  section "node"; hostname; nproc; free -g | head -2; df -h /tmp | tail -1
  section "driver"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  nvidia-smi | grep -o "CUDA Version: [0-9.]*"
  section "topology"; nvidia-smi topo -m
  section "network (pypi, huggingface, crossref, openai)"
  for u in https://pypi.org/simple/pip/ https://huggingface.co/api/models/Qwen/Qwen3.5-9B https://api.crossref.org/works?rows=0 https://api.openai.com/v1/models; do
    # Any HTTP answer means the route works (the OpenAI API answers 401 without a key).
    code="$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "${u}" || true)"
    if [[ "${code}" =~ ^[234] ]]; then echo "reachable  ${u}"; else echo "BLOCKED    ${u}"; fi
  done
  section "apptainer"; command -v apptainer && apptainer --version
  section "modules (cuda)"; module -t avail cuda 2>&1 | tail -n +1 | head -20
  if command -v python >/dev/null && python -c "import torch" 2>/dev/null; then
    section "check_env"; python -m dpt.launch.check_env --json "${OUT}/env_report.json" || true
  fi
  section "sandbox image"; sandbox_image && echo "image: ${DPT_SANDBOX_SIF}" || echo "could not pull the sandbox image"
  if [[ -f "${DPT_SANDBOX_SIF}" ]]; then
    section "sandbox isolation"; python -m dpt.rewards.sandbox_check --json "${OUT}/sandbox_check.json" \
      && echo "SANDBOX ISOLATED" || echo "SANDBOX NOT ISOLATED: do not start RL (see the failed probes above)"
  else
    section "sandbox isolation"; echo "skipped: no image at ${DPT_SANDBOX_SIF} (see the pull above)"
  fi
} 2>&1 | tee "${OUT}/report.txt"
echo "report: ${OUT}/report.txt"
