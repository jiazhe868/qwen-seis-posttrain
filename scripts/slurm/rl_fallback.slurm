#!/bin/bash
#SBATCH -J dpt-rlfb
#SBATCH -p gpu-a100
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 24:00:00
#SBATCH -o dpt-rlfb-%j.out
# RL v2 without a reservation (2026-11-09/10, README §8.3). The judge, RL and the final eval run in ONE job
# from the normal queue, so the judge is up when RL starts and nothing waits on a second queue. This is window 2 if
# the site grants no reservation. Window 1 then runs as a plain 2-node job (data_window.slurm), and this job goes in after
# bank v2 has passed its check.
#
# The cluster allows at most 8 gpu-a100 nodes per job (QOS qa100; 12 per user), so this is a HETEROGENEOUS job. Its two
# components start together:
#   component 0: 8 gpu-a100 nodes, RL and then the final eval;
#   component 1: 1 node for the judge. gpu-a100 (default): Gemma ×3, then 3 more model replicas in the final
#                eval. gpu-a100-small: Gemma ×1, as run 1's judge.
#
#   scripts/slurm/submit_window.sh rl-queue       # checks the bank, then submits this with both components:
#   sbatch -A PROJ -p gpu-a100 -N 8 -t 24:00:00 : -A PROJ -p gpu-a100 -N 1 -t 24:00:00 scripts/slurm/rl_fallback.slurm
#
# What runs:
#   - the judge node: Gemma, one replica per GPU (:8101…) behind rr_proxy (:8100), published as the `judge` endpoint;
#   - the 8 RL nodes: scripts/slurm/rl.slurm --config rl/rl_9b_v2.yaml, run inside this job with no successor. The
#     trainer saves and stops DPT_EVAL_H hours (default 4) + 20 min before this job's limit;
#   - then the judge stops and scripts/slurm/final_eval_window.slurm runs: the judge and the sandbox on RL node 1, the
#     model ×3 on each other node (the judge node too if it has 3 GPUs). It runs whatever way RL ended (as
#     afterany): after a pause or a crash it evaluates the best dev checkpoint so far.
#
# Logs: this job's output (RL), $WORK/dpt/logs/rlfb_<job>/ (judge), $WORK/dpt/logs/finaleval_<job>/ (final eval).
# Dry run: DPT_SLURM_DRY_RUN=1 DPT_SLURM_NODES="<8 RL nodes> <judge node>"; DPT_JUDGE_NODE_GPUS (default 3 when dry).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/slurm/env.sh
source scripts/slurm/lib.sh
export HF_HUB_OFFLINE=1
EVAL_S=$(( ${DPT_EVAL_H:-4} * 3600 ))
if dry; then                                                    # dry runs: DPT_SLURM_NODES
  mapfile -t ALL < <(job_nodes)
  RL_NODES=("${ALL[@]:0:8}"); JN="${ALL[8]:-}"
else                                                            # the heterogeneous job's two components
  mapfile -t RL_NODES < <(het_nodes 0)
  JN="$(het_nodes 1 | head -n 1)"
fi
(( ${#RL_NODES[@]} == 8 )) && [[ -n "${JN}" ]] ||
  tdie "this job needs 8 RL nodes and a judge node, it has ${#RL_NODES[@]} and '${JN}' (submit_window.sh rl-queue)"
JURL="http://${JN}:8100"
JG="${DPT_JUDGE_NODE_GPUS:-}"
if [[ -z "${JG}" ]]; then
  if dry; then JG=3; else JG="$(on_node "${JN}" nvidia-smi -L | grep -c '^GPU' || true)"; fi
fi
(( JG >= 1 )) || tdie "no GPU on the judge node ${JN}"
if (( JG >= 3 )); then JG=3; EVAL_NODES=("${RL_NODES[@]}" "${JN}"); else EVAL_NODES=("${RL_NODES[@]}"); fi
LOGS="${WORK}/dpt/logs/rlfb_${SLURM_JOB_ID:-local}"; PIDS="${LOGS}/pids"
mkdir -p "${LOGS}" "${PIDS}" "${DPT_ENDPOINTS}"
read -r DATA JUDGE_ID <<< "$(python - <<'PY'
from dpt.common.config import load_config
p = load_config().project
print(p.paths.data_root, p.judges.primary.hf_id)
PY
)"

# ---- before any server: the bank and the time ------------------------------------------------------------------
CHECK="${DATA}/v2/bank_check.json"
[[ -f "${CHECK}" ]] || tdie "no ${CHECK}: build bank v2 first (data_window.slurm)"
python - "${CHECK}" <<'PY' || tdie "bank v2 did not pass its check: ${CHECK}"
import json, sys
problems = json.load(open(sys.argv[1])).get("problems", ["no problems field"])
for p in problems:
    print("bank check:", p, file=sys.stderr)
sys.exit(1 if problems else 0)
PY
if ! dry; then
  LEFT_S="$(python -c 'from dpt.launch.walltime import slurm_time_left; print(slurm_time_left())')"
  (( LEFT_S - EVAL_S >= 6 * 3600 )) || tdie "only $((LEFT_S / 60)) min in this job: RL would get less than 6 h"
fi

# ---- the judge -------------------------------------------------------------------------------------------------
wait_judge() {   # wait_judge <port>: until the judge node's server on <port> answers; stop at once if it exited
  local port="$1" i=0
  dry && return 0
  until curl -sf --max-time 10 "http://${JN}:${port}/v1/models" >/dev/null; do
    on_node "${JN}" bash -c "kill -0 $(cat "${PIDS}/${port}") 2>/dev/null" ||
      tdie "judge server :${port} on ${JN} exited while starting (${LOGS}/*_${JN}_${port}.log)"
    sleep 15; i=$((i + 15)); (( i < 3600 )) || tdie "judge server :${port} on ${JN} not up after 1 h"
  done
}
judge_up() {     # one Gemma replica per GPU of the judge node, behind one proxy
  local g backends=""
  for (( g = 0; g < JG; g++ )); do
    on_node "${JN}" bash -c "HF_HUB_OFFLINE=1 DPT_VLLM_DP=1 CUDA_VISIBLE_DEVICES=${g} GPU_MEM=0.9 setsid nohup bash \
scripts/serve_vllm.sh '${JUDGE_ID}' judge $((8101 + g)) 40960 1 > '${LOGS}/judge_${JN}_$((8101 + g)).log' 2>&1 \
< /dev/null & echo \$! > '${PIDS}/$((8101 + g))'"
  done
  for (( g = 0; g < JG; g++ )); do wait_judge $((8101 + g)); backends+=" --backend http://127.0.0.1:$((8101 + g))"; done
  on_node "${JN}" bash -c "setsid nohup python scripts/rr_proxy.py --port 8100${backends} \
> '${LOGS}/proxy_${JN}_8100.log' 2>&1 < /dev/null & echo \$! > '${PIDS}/8100'"
  wait_judge 8100
  dry || { echo "${JURL}" > "${DPT_ENDPOINTS}/judge.url"; echo judge > "${DPT_ENDPOINTS}/judge.model"; }
  tlog "judge: Gemma ×${JG} on ${JN}, published at ${JURL}"
}
judge_down() {   # unpublish, then stop the judge node's servers (TERM, then KILL) until its GPUs are free
  [[ "$(endpoint_url judge)" == "${JURL}" ]] && rm -f "${DPT_ENDPOINTS}/judge.url" "${DPT_ENDPOINTS}/judge.model"
  local files=() f sig i
  for f in "${PIDS}"/*; do [[ -e "${f}" ]] && files+=("${f}"); done
  (( ${#files[@]} )) || return 0
  for sig in TERM KILL; do
    for f in "${files[@]}"; do on_node "${JN}" bash -c "kill -${sig} -- -$(cat "${f}") 2>/dev/null; true" || true; done
    dry && break
    for i in $(seq 1 12); do
      [[ -z "$(on_node "${JN}" nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]] && break 2
      sleep 10
    done
  done
  rm -f "${files[@]}"
}
gpus_free() {    # gpus_free <host…>: wait up to 10 min until no process holds a GPU on these nodes
  dry && return 0
  local h i busy=()
  for i in $(seq 1 60); do
    busy=()
    for h in "$@"; do
      [[ -z "$(on_node "${h}" nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]] || busy+=("${h}")
    done
    (( ${#busy[@]} )) || return 0
    sleep 10
  done
  tlog "WARN GPUs still in use after 10 min on ${busy[*]}"
}
trap 'judge_down || true' EXIT

# ---- RL on the 8 RL nodes --------------------------------------------------------------------------------------
tlog "RL v2 on ${RL_NODES[*]}, the judge (${JG} GPU) on ${JN}; the final eval gets the last $((EVAL_S / 3600)) h"
judge_up
RLENV=(DPT_SLURM_NODES="${RL_NODES[*]}" DPT_CHAIN_LEFT=1 DPT_JUDGE_WAIT_S=600 DPT_RL_MARGIN_S=$((EVAL_S + 1200)))
dry && RLENV+=(DPT_JUDGE_WAIT_S=0 DPT_ALLOW_NO_JUDGE=1)   # a dry run publishes no endpoint file
RC=0
env "${RLENV[@]}" bash scripts/slurm/rl.slurm --config rl/rl_9b_v2.yaml || RC=$?
tlog "RL ended (exit ${RC}); final eval next"

# ---- the final eval ---------------------------------------------------------------------------------------------
judge_down
gpus_free "${EVAL_NODES[@]}"
env -u DPT_CANCEL_JOB DPT_SLURM_NODES="${EVAL_NODES[*]}" bash scripts/slurm/final_eval_window.slurm
tlog "done: RL exit ${RC}, final eval finished"
exit "${RC}"
