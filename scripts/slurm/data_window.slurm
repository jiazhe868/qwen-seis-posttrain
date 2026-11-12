#!/bin/bash
#SBATCH -J dpt-datawin
#SBATCH -p gpu-a100
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -t 24:00:00
#SBATCH -o dpt-datawin-%j.out
# Bank v2's data phase as ONE job (2026-11-09, README §8.3). Every server and client runs on the job's own
# nodes, so nothing waits for another queue (on 11-09 the separate serve and data jobs of 11-07 had waited ~2 days,
# and the FP8 teacher did not fit one A100). It runs as a 2-node job (window 1), or inside RL v2's 8-node job when
# bank v2 is still missing there (scripts/slurm/rl_v2_job.sh, 2026-11-11).
# The teacher is the FP8 checkpoint on two GPUs: ~14 GiB of weights per GPU. In BF16 it would hold ~25 GiB per GPU,
# close to the 28 GiB with which the FP8 copy ran out of memory on one A100 (jobs 3530657 and 3561787).
#
#   teacher nodes (the first DPT_DATA_TEACHER_NODES, default 1): the teacher, FP8 with TP2 on GPUs 0-1 (:8010),
#                 and Gemma on GPU 2 (:8110)
#   other nodes:  the SFT model, one replica per GPU (:8001-8003)
#   node 1 also runs the proxies: teacher (:8000), judge_v (:8100) and sft_audit (:8200)
#
# Phase A, in parallel:
#   - verify the hard and the long T1 families (data.slurm bank, ingest … verify);
#   - rewrite the T3 rubrics as atomic checks (t3_atomic.slurm), unless DPT_DATA_SKIP_T3=1 (RL v2 trains no T3);
#   - grade the v1 problems with the SFT model (bank v2, audit_all).
# Phase B, once the teacher work is done: the teacher nodes stop the teacher and Gemma and serve the SFT model on
#   :8001-8003 as well. The sft_audit proxy lists every node's replicas from the start. A replica that is not up
#   yet refuses connections and rr_proxy skips it, so the v1 grading speeds up as soon as these replicas join.
#   The verified hard + long problems are then graded on all replicas (bank v2_new).
# Phase C: merge both difficulty stages (with the atomic rubrics, unless skipped) into bank v2, publish, put back
#   run 1's dev split (bank_merge pin-dev: re-grading with the SFT model would otherwise change it), check.
#
#   sbatch -A PROJ --reservation=<name> scripts/slurm/data_window.slurm
#   scripts/slurm/submit_window.sh data-queue -t 48:00:00      # without a reservation
#
# Logs: $WORK/dpt/logs/datawin_<job>/ (DPT_DATA_LOGS). A phase that already finished is skipped on a rerun
# (data.slurm markers; t3_atomic resumes its map).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/slurm/env.sh
source scripts/slurm/lib.sh
export HF_HUB_OFFLINE=1

DATA="$(python -c 'from dpt.common.config import load_config; print(load_config().project.paths.data_root)')"
G="${DATA}/gen"; V2="${DATA}/v2"; NEW="${DATA}/v2_new"
SFT="${DPT_SFT_EXPORT:-${WORK}/dpt/exports/sft9b_seed1}"
TEACHER_MODEL="${DPT_TEACHER_MODEL:-Qwen/Qwen3.8-27B-FP8}"
JUDGE_MODEL="${DPT_JUDGE_MODEL_ID:-RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic}"
SKIP_T3="${DPT_DATA_SKIP_T3:-0}"
TEACHER_NODE_COUNT="${DPT_DATA_TEACHER_NODES:-1}"
mapfile -t NODES < <(job_nodes)
(( ${#NODES[@]} >= TEACHER_NODE_COUNT + 1 )) ||
  tdie "this job needs $((TEACHER_NODE_COUNT + 1)) nodes (${TEACHER_NODE_COUNT} for the teacher, 1 or more for the SFT model), it has ${#NODES[@]}"
TEACHER_NODES=("${NODES[@]:0:TEACHER_NODE_COUNT}"); SFT_NODES=("${NODES[@]:TEACHER_NODE_COUNT}")
N1="${NODES[0]}"
LOGS="${DPT_DATA_LOGS:-${WORK}/dpt/logs/datawin_${SLURM_JOB_ID:-local}}"; PIDS="${LOGS}/pids"
mkdir -p "${LOGS}" "${PIDS}" "${DPT_ENDPOINTS}"
export DPT_ENDPOINT_WAIT_S=600                     # the clients start once their servers answer
PROXIES=()                                          # process groups of node 1's proxies
# Problems in flight per SFT replica, for each grading client: one A100 generates ~1,300 tok/s, and at 64 per server
# the wait reached the old 30 min client timeout (configs/data/pipeline.yaml, 2026-10-25). While both clients run in
# phase B, a replica holds up to ~56; the client now waits up to 4 h (backends.base.timeout_s), so that only queues.
PER_REPLICA=32

bg() {   # bg <log> <command…>: run in the background with output in $LOGS/<log>; its pid in $BG
  local log="${LOGS}/$1"; shift
  if dry; then printf 'DRY:'; printf ' %q' "$@"; printf ' > %q\n' "${log}"; true & BG=$!; return; fi
  "$@" > "${log}" 2>&1 < /dev/null & BG=$!
}
serve_on() {   # serve_on <host> <served name> <model> <port> <tp> <gpus> [VAR=value…]: a vLLM server in its own group
  local host="$1" name="$2" model="$3" port="$4" tp="$5" gpus="$6"; shift 6
  on_node "${host}" bash -c "$* DPT_VLLM_DP=1 CUDA_VISIBLE_DEVICES=${gpus} GPU_MEM=0.9 setsid nohup bash \
scripts/serve_vllm.sh '${model}' ${name} ${port} 40960 ${tp} > '${LOGS}/${name}_${host}_${port}.log' 2>&1 \
< /dev/null & echo \$! > '${PIDS}/${host}_${port}'"
}
wait_server() {  # wait_server <host> <port>: until it answers; stop at once if its process exited
  local host="$1" port="$2" i=0
  dry && return 0
  until curl -sf --max-time 10 "http://${host}:${port}/v1/models" >/dev/null; do
    on_node "${host}" bash -c "kill -0 $(cat "${PIDS}/${host}_${port}") 2>/dev/null" ||
      tdie "server :${port} on ${host} exited before it came up (${LOGS}/*_${host}_${port}.log)"
    sleep 15; i=$((i + 15)); (( i < 3600 )) || tdie "server :${port} on ${host} not up after 1 h"
  done
}
stop_servers() { # stop_servers <host…>: TERM the process groups of these nodes' servers
  local h f
  for h in "$@"; do
    for f in "${PIDS}/${h}"_*; do
      [[ -e "${f}" ]] || continue
      on_node "${h}" bash -c "kill -TERM -- -$(cat "${f}") 2>/dev/null; true" || true
      rm -f "${f}"
    done
  done
}
gpus_free() {    # gpus_free <host…>: wait up to 5 min until no process holds a GPU on these nodes
  dry && return 0
  local h i
  for h in "$@"; do
    for (( i = 0; i < 30; i++ )); do
      [[ -z "$(on_node "${h}" nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]] && break
      sleep 10
    done
    (( i < 30 )) || tdie "GPUs still busy on ${h} 5 min after stopping its servers"
  done
}
proxy_here() {   # proxy_here <name> <port> <backend…>: rr_proxy on node 1, published as endpoint <name>
  local name="$1" port="$2" b args=(); shift 2
  for b in "$@"; do args+=(--backend "${b}"); done
  bg "proxy_${name}.log" setsid python scripts/rr_proxy.py --port "${port}" "${args[@]}"
  PROXIES+=("${BG}"); wait_url "http://${N1}:${port}" 300 "${BG}"
}
publish()   { dry && return 0; echo "$2" > "${DPT_ENDPOINTS}/$1.url"; echo "$3" > "${DPT_ENDPOINTS}/$1.model"; }
unpublish() { [[ "$(endpoint_url "$1")" == "http://${N1}:"* ]] && rm -f "${DPT_ENDPOINTS}/$1.url" "${DPT_ENDPOINTS}/$1.model"; true; }
cleanup() {
  local n p
  for n in teacher judge_v sft_audit; do unpublish "${n}"; done
  stop_servers "${NODES[@]}"
  for p in "${PROXIES[@]}"; do kill -TERM -- "-${p}" 2>/dev/null || true; done
  kill $(jobs -p) 2>/dev/null || true
}
trap cleanup EXIT
must() {   # must <pid> <what>: wait for a background step and stop the job if it failed
  wait "$1" || tdie "$2 failed: see ${LOGS}"
}
backends_of() {  # backends_of <port…> -- <host…>: http://host:port for every pair
  local ports=() h p
  while [[ "$1" != -- ]]; do ports+=("$1"); shift; done; shift
  for h in "$@"; do for p in "${ports[@]}"; do printf 'http://%s:%s\n' "${h}" "${p}"; done; done
}

# ---- servers --------------------------------------------------------------------------------------------------
tlog "teacher TP2 + Gemma on ${TEACHER_NODES[*]}; the SFT model ×3 on ${SFT_NODES[*]}"
for h in "${SFT_NODES[@]}"; do
  for g in 0 1 2; do serve_on "${h}" sft "${SFT}" $((8001 + g)) 1 "${g}" DPT_GDN_MODEL=1; done
done
for h in "${TEACHER_NODES[@]}"; do
  serve_on "${h}" teacher "${TEACHER_MODEL}" 8010 2 0,1
  serve_on "${h}" judge_v "${JUDGE_MODEL}" 8110 1 2 DPT_SERVE_DP=1
done
for h in "${TEACHER_NODES[@]}"; do wait_server "${h}" 8010; wait_server "${h}" 8110; done
mapfile -t TEACHER_URLS < <(backends_of 8010 -- "${TEACHER_NODES[@]}")
mapfile -t JUDGE_URLS < <(backends_of 8110 -- "${TEACHER_NODES[@]}")
proxy_here teacher 8000 "${TEACHER_URLS[@]}"; publish teacher "http://${N1}:8000" teacher
proxy_here judge_v 8100 "${JUDGE_URLS[@]}"; publish judge_v "http://${N1}:8100" judge_v
for h in "${SFT_NODES[@]}"; do for g in 0 1 2; do wait_server "${h}" $((8001 + g)); done; done
# every node's replicas, the teacher nodes' included: those refuse connections until phase B, and rr_proxy skips them
mapfile -t SFT_URLS < <(backends_of 8001 8002 8003 -- "${SFT_NODES[@]}" "${TEACHER_NODES[@]}")
proxy_here sft_audit 8200 "${SFT_URLS[@]}"; publish sft_audit "http://${N1}:8200" sft

# ---- phase A ----------------------------------------------------------------------------------------------------
tlog "phase A: hard + long verification$( [[ "${SKIP_T3}" == 1 ]] || echo ', atomic T3'), v1 difficulty"
bg hard.log env DPT_BANK_OUT="${DATA}/v1_hard" DPT_BANK_NAME=hard DPT_BANK_CANDIDATES="${G}/t1_hard_candidates.jsonl" \
  DPT_BANK_STAGES=ingest,quality,dedup,decontam,verify DPT_JUDGE_ENDPOINT=judge_v bash scripts/slurm/data.slurm bank
A_HARD=${BG}
bg long.log env DPT_BANK_OUT="${DATA}/v1_long" DPT_BANK_NAME=long DPT_BANK_CANDIDATES="${G}/t1_long_candidates.jsonl" \
  DPT_BANK_STAGES=ingest,quality,dedup,decontam,verify DPT_JUDGE_ENDPOINT=judge_v bash scripts/slurm/data.slurm bank
A_LONG=${BG}
if [[ "${SKIP_T3}" != 1 ]]; then
  bg t3_atomic.log env DPT_TEACHER_ENDPOINT=teacher DPT_JUDGE_ENDPOINT=judge_v bash scripts/slurm/t3_atomic.slurm
  A_T3=${BG}
fi
bg v2a.log env DPT_BANK_OUT="${V2}" DPT_BANK_NAME=v2a DPT_BANK_STAGES=difficulty DPT_DIFFICULTY_ENDPOINT=sft_audit \
  DPT_DIFFICULTY_MODE=audit_all DPT_DIFFICULTY_SKIP="[t3_report,t2_code]" \
  DPT_DIFFICULTY_CONCURRENCY=$(( PER_REPLICA * 3 * ${#SFT_NODES[@]} )) bash scripts/slurm/data.slurm bank
A_V1=${BG}
must "${A_HARD}" "hard verification"; must "${A_LONG}" "long verification"
[[ "${SKIP_T3}" == 1 ]] || must "${A_T3}" "atomic T3 rubrics"
tlog "phase A: teacher work done"

# ---- phase B ----------------------------------------------------------------------------------------------------
tlog "phase B: the teacher nodes switch to the SFT model; difficulty of the new problems on every replica"
unpublish teacher; unpublish judge_v
kill -TERM -- "-${PROXIES[0]}" "-${PROXIES[1]}" 2>/dev/null || true
stop_servers "${TEACHER_NODES[@]}"; gpus_free "${TEACHER_NODES[@]}"
for h in "${TEACHER_NODES[@]}"; do
  for g in 0 1 2; do serve_on "${h}" sft "${SFT}" $((8001 + g)) 1 "${g}" DPT_GDN_MODEL=1; done
done
for h in "${TEACHER_NODES[@]}"; do for g in 0 1 2; do wait_server "${h}" $((8001 + g)); done; done
run mkdir -p "${NEW}/stages"
run python -m dpt.data.bank_merge merge --name rl_bank.verify --out "${NEW}/stages/04_verify.jsonl" \
  "${DATA}/v1_hard/stages/04_verify.jsonl" "${DATA}/v1_long/stages/04_verify.jsonl"
bg v2new.log env DPT_BANK_OUT="${NEW}" DPT_BANK_NAME=v2new DPT_BANK_STAGES=difficulty DPT_DIFFICULTY_ENDPOINT=sft_audit \
  DPT_DIFFICULTY_MODE=audit_all DPT_DIFFICULTY_SKIP="[t3_report,t2_code]" \
  DPT_DIFFICULTY_CONCURRENCY=$(( PER_REPLICA * 3 * ${#NODES[@]} )) bash scripts/slurm/data.slurm bank
must "${BG}" "difficulty of the new problems"
must "${A_V1}" "difficulty of the v1 problems"

# ---- phase C ----------------------------------------------------------------------------------------------------
tlog "phase C: merge, publish, run 1's dev split, check"
T3_MAP=(--t3-map "${V2}/t3_atomic_map.jsonl"); CHECK_T3=()
[[ "${SKIP_T3}" != 1 ]] || { T3_MAP=(); CHECK_T3=(--t3-not-trained); }
run python -m dpt.data.bank_merge merge --name rl_bank.difficulty --out "${V2}/stages/05_difficulty.jsonl" \
  "${T3_MAP[@]}" "${V2}/stages/05_difficulty.jsonl" "${NEW}/stages/05_difficulty.jsonl"
run env DPT_BANK_OUT="${V2}" DPT_BANK_NAME=v2pub DPT_BANK_STAGES=publish bash scripts/slurm/data.slurm bank
run python -m dpt.data.bank_merge pin-dev --reference-dev "${DATA}/v1/rl_dev.jsonl" "${T3_MAP[@]}" "${V2}"
run python -m dpt.data.bank_merge check "${V2}" --reference-dev "${DATA}/v1/rl_dev.jsonl" "${CHECK_T3[@]}"
tlog "bank v2 ready: ${V2} (check: ${V2}/bank_check.json)"
