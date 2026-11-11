#!/bin/bash
#SBATCH -J dpt-serve
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 48:00:00
#SBATCH -o dpt-serve-%j.out
# Serve one model with vLLM on one node and publish its endpoint for other jobs.
#
#   sbatch -A <alloc> scripts/slurm/serve.slurm <name> <model> [tp] [max_model_len]
#
#   name   endpoint name other jobs look up: judge | teacher | base | verifier_b | embed ...
#   model  HF id (downloaded into $HF_HOME on a login node first) or a local path
#   tp     tensor parallel size per replica; the node's GPUs are filled with replicas (data parallel)
#
# Examples:
#   sbatch -A X -p gpu-h100 scripts/slurm/serve.slurm judge Qwen/Qwen3.8-27B 2       # 2 × H100 80 GB, 6 SU/h
#   sbatch -A X              scripts/slurm/serve.slurm teacher Qwen/Qwen3.8-27B 2    # TP2 on 2 of 3 A100s
#   sbatch -A X              scripts/slurm/serve.slurm base Qwen/Qwen3.5-9B 1        # TP1 × DP3, 3 SU/h
#
# Writes $DPT_ENDPOINTS/<name>.url (http://host:8000, no /v1) and <name>.model (the served name =
# <name>), and removes them when the job ends. Fixed vLLM flags (FP32 GDN state, no prefix cache,
# language model only) come from scripts/serve_vllm.sh.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/slurm/env.sh
source scripts/slurm/lib.sh
NAME="${1:?endpoint name}"; MODEL="${2:?model id or path}"; TP="${3:-1}"; MAXLEN="${4:-40960}"
export HF_HUB_OFFLINE=1                                 # models were downloaded on a login node
NGPU="$(nvidia-smi -L | wc -l)"
# DPT_SERVE_DP caps the replicas. vLLM data parallelism shards an MoE model's experts across the DP ranks,
# so the Gemma judge must not use it: DPT_SERVE_DP=1. DPT_SERVE_REPLICAS=K instead runs K independent servers
# (ports 8001…, TP GPUs each) behind the least-busy proxy scripts/rr_proxy.py on :8000, e.g. the judge on a
# 3-GPU gpu-a100-dev node: DPT_SERVE_DP=1 DPT_SERVE_REPLICAS=3.
DP="${DPT_SERVE_DP:-$((NGPU / TP))}"; REPLICAS="${DPT_SERVE_REPLICAS:-1}"
[[ ${DP} -ge 1 && ${REPLICAS} -ge 1 && $((TP * DP * REPLICAS)) -le ${NGPU} ]] ||
  tdie "TP${TP} × DP${DP} × ${REPLICAS} replicas needs more than the ${NGPU} GPUs of this node"
URL="http://$(hostname):8000"
mkdir -p "${DPT_ENDPOINTS}"
cleanup() {   # only remove the files if they still point at this job
  [[ "$(endpoint_url "${NAME}")" == "${URL}" ]] && rm -f "${DPT_ENDPOINTS}/${NAME}.url" "${DPT_ENDPOINTS}/${NAME}.model"
  kill $(jobs -p) 2>/dev/null || true
  return 0
}
trap cleanup EXIT
tlog "serving ${MODEL} as '${NAME}' at ${URL}: ${REPLICAS} × TP${TP} × DP${DP} on ${NGPU} GPUs, max_model_len ${MAXLEN}"
if (( REPLICAS == 1 )); then
  DPT_VLLM_DP="${DP}" CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((TP * DP - 1)))" GPU_MEM="${GPU_MEM:-0.9}" \
    bash scripts/serve_vllm.sh "${MODEL}" "${NAME}" 8000 "${MAXLEN}" "${TP}" &
  SERVER=$!
  wait_url "${URL}" 3600 "${SERVER}"        # fails at once if vLLM dies while loading
else
  BACKENDS=(); PIDS=()
  for ((r = 0; r < REPLICAS; r++)); do
    first=$((r * TP * DP))
    DPT_VLLM_DP="${DP}" CUDA_VISIBLE_DEVICES="$(seq -s, "${first}" $((first + TP * DP - 1)))" GPU_MEM="${GPU_MEM:-0.9}" \
      bash scripts/serve_vllm.sh "${MODEL}" "${NAME}" $((8001 + r)) "${MAXLEN}" "${TP}" &
    PIDS+=($!); BACKENDS+=(--backend "http://127.0.0.1:$((8001 + r))")
  done
  for ((r = 0; r < REPLICAS; r++)); do wait_url "http://$(hostname):$((8001 + r))" 3600 "${PIDS[r]}"; done
  python scripts/rr_proxy.py --port 8000 "${BACKENDS[@]}" &
  SERVER=$!
  wait_url "${URL}" 300 "${SERVER}"
fi
echo "${URL}" > "${DPT_ENDPOINTS}/${NAME}.url"; echo "${NAME}" > "${DPT_ENDPOINTS}/${NAME}.model"
tlog "endpoint published: ${DPT_ENDPOINTS}/${NAME}.url"
if (( REPLICAS == 1 )); then wait "${SERVER}"; else wait -n; fi   # with replicas: stop when any server or the proxy exits
