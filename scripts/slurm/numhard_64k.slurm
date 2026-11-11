#!/bin/bash
#SBATCH -J dpt-nh64k
#SBATCH -p gpu-a100-dev
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -t 02:00:00
#SBATCH -o dpt-nh64k-%j.out
# D-Num-hard at a 64K budget (2026-11-10, docs/reports/failure_types_and_tests_1110_en.md §7).
# - The answers that ran out at 32K are continued for another 32K tokens by the same model
#   (scripts/talk/numhard_continue.py). The answers that finished are unchanged, because the budget does not change
#   sampling.
# - The rows run in order: base_medium, sft, rl_seed1. Each row uses both nodes: one vLLM server per node, TP1 × DP3,
#   max_model_len 67,584, behind rr_proxy on :8000.
# - A row with nothing left to do is skipped. The client stops new requests 35 min before the job's limit, and
#   results are appended as they arrive, so a second job resumes:
#
#   J=$(sbatch --parsable -A PROJ scripts/slurm/numhard_64k.slurm | tail -n 1)
#   sbatch -A PROJ --dependency=afterany:$J scripts/slurm/numhard_64k.slurm
#
# When every row is complete, the job writes $WORK/dpt/talk/numhard_64k.json (counts).
# gpu-a100-dev allows 2 nodes, 2 h and one running job per user.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/slurm/env.sh
source scripts/slurm/lib.sh
export HF_HUB_OFFLINE=1
ROWS="${DPT_ROWS:-base_medium sft rl_seed1}"
MAXLEN=67584
mapfile -t NODES < <(job_nodes)
(( ${#NODES[@]} >= 1 )) || tdie "no nodes"
LOGS="${WORK}/dpt/logs/nh64k_${SLURM_JOB_ID:-local}"; PIDS="${LOGS}/pids"
mkdir -p "${LOGS}" "${PIDS}"
BASE="$(python -c 'from dpt.common.config import load_config; print(load_config().project.model.base)')"
declare -A MODEL=([base_medium]="${BASE}" [sft]="${WORK}/dpt/exports/sft9b_seed1"
                  [rl_seed1]="${WORK}/dpt/exports/rl9b_seed1_step20")

serve() {      # serve <row>: the row's model on every node, then the proxy on this node
  local row="$1" h urls=() args=()
  for h in "${NODES[@]}"; do
    on_node "${h}" bash -c "HF_HUB_OFFLINE=1 DPT_GDN_MODEL=1 DPT_VLLM_DP=3 GPU_MEM=0.9 setsid nohup bash \
scripts/serve_vllm.sh '${MODEL[${row}]}' '${row}' 8001 ${MAXLEN} 1 > '${LOGS}/${row}_${h}.log' 2>&1 < /dev/null \
& echo \$! > '${PIDS}/${h}_8001'"
    urls+=("http://${h}:8001")
  done
  for h in "${urls[@]}"; do wait_url "${h}" 2400 || tdie "${h} did not come up (${LOGS})"; done
  for h in "${urls[@]}"; do args+=(--backend "${h}"); done
  if dry; then run python scripts/rr_proxy.py --port 8000 "${args[@]}"; else
    setsid nohup python scripts/rr_proxy.py --port 8000 "${args[@]}" > "${LOGS}/proxy_${row}.log" 2>&1 < /dev/null &
    echo $! > "${PIDS}/${NODES[0]}_8000"
  fi
  wait_url "http://127.0.0.1:8000" 300 || tdie "proxy did not come up"
}
stop() {       # every server and proxy of this job, then wait (up to 3 min) for free GPUs
  local f host i busy
  for f in "${PIDS}"/*; do
    [[ -e "${f}" ]] || continue
    host="$(basename "${f}")"; host="${host%_*}"
    on_node "${host}" bash -c "kill -TERM -- -$(cat "${f}") 2>/dev/null; true" || true
    rm -f "${f}"
  done
  dry && return 0
  for i in $(seq 1 18); do
    busy=0
    for host in "${NODES[@]}"; do
      [[ -z "$(on_node "${host}" nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]] || busy=1
    done
    (( busy )) || return 0
    sleep 10
  done
  tlog "WARN GPUs still in use 3 min after stopping the servers"
}
trap 'stop || true' EXIT
left_s() { if dry; then echo 7200; else python -c 'from dpt.launch.walltime import slurm_time_left; print(slurm_time_left())'; fi; }

complete=1
for row in ${ROWS}; do
  if dry; then n=1; else n="$(python scripts/talk/numhard_continue.py todo --row "${row}")"; fi
  if (( n == 0 )); then tlog "${row}: complete"; continue; fi
  left="$(left_s)"
  if (( left < 3000 )); then tlog "${row}: ${n} left but only $((left / 60)) min in this job; the next job resumes"
    complete=0; continue
  fi
  tlog "${row}: ${n} continuations to do; $((left / 60)) min left in this job"
  serve "${row}"
  left="$(left_s)"
  run python scripts/talk/numhard_continue.py run --row "${row}" --endpoint http://127.0.0.1:8000 \
    --served-model "${row}" --concurrency 60 --stop-new-after $((left - 2100)) 2>&1 | tee -a "${LOGS}/client.log"
  stop
  if ! dry && (( $(python scripts/talk/numhard_continue.py todo --row "${row}") > 0 )); then complete=0; fi
done
if (( complete )); then
  run python scripts/talk/numhard_continue.py summary 2>&1 | tee -a "${LOGS}/summary.log"
else
  tlog "not complete yet: submit this job again (it resumes)"
fi
