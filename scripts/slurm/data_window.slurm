#!/bin/bash
#SBATCH -J dpt-datawin
#SBATCH -p gpu-a100
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -t 24:00:00
#SBATCH -o dpt-datawin-%j.out
# Bank v2's data phase as ONE 2-node job, for a reservation (2026-11-09, README §8.3). Every server
# and client runs on the job's own nodes, so nothing waits for another queue (on 11-09 the separate serve and data
# jobs of 11-07 had waited ~2 days, and the FP8 teacher did not fit one A100).
# The teacher is the FP8 checkpoint on two GPUs: ~14 GiB of weights per GPU. In BF16 it would hold ~25 GiB per GPU,
# close to the 28 GiB with which the FP8 copy ran out of memory on one A100 (jobs 3530657 and 3561787).
#
#   node 1 (this one): the teacher, FP8 with TP2 on GPUs 0-1 (:8000, `teacher`), and Gemma on GPU 2 (:8100, `judge_v`)
#   node 2:            the SFT model, one replica per GPU, behind rr_proxy on node 1 (:8200, `sft_audit`)
#
# Phase A, in parallel:
#   - verify the hard and the long T1 families (data.slurm bank, ingest … verify);
#   - rewrite the T3 rubrics as atomic checks (t3_atomic.slurm);
#   - grade the v1 problems with the SFT model (bank v2, audit_all).
# Phase B, once the teacher work is done: node 1 stops the teacher and Gemma, serves three more SFT replicas
#   (:8300, `sft_new`), and grades the verified hard + long problems there (bank v2_new).
# Phase C: merge both difficulty stages with the atomic rubrics into bank v2, publish, check.
#
#   sbatch -A PROJ --reservation=<name> scripts/slurm/data_window.slurm
#
# Logs: $WORK/dpt/logs/datawin_<job>/. A phase that already finished is skipped on a rerun (data.slurm markers;
# t3_atomic resumes its map).
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
mapfile -t NODES < <(job_nodes)
(( ${#NODES[@]} >= 2 )) || tdie "this job needs 2 nodes, it has ${#NODES[@]}"
N1="${NODES[0]}"; N2="${NODES[1]}"
LOGS="${WORK}/dpt/logs/datawin_${SLURM_JOB_ID:-local}"
mkdir -p "${LOGS}" "${DPT_ENDPOINTS}"
export DPT_ENDPOINT_WAIT_S=600                     # the clients start once their servers answer
LOCAL=()                                            # process groups of this node's servers and proxies
STARTING=()                                         # port:pid of this node's servers not yet checked up

bg() {   # bg <log> <command…>: run in the background with output in $LOGS/<log>; its pid in $BG
  local log="${LOGS}/$1"; shift
  if dry; then printf 'DRY:'; printf ' %q' "$@"; printf ' > %q\n' "${log}"; true & BG=$!; return; fi
  "$@" > "${log}" 2>&1 < /dev/null & BG=$!
}
serve_here() {   # serve_here <name> <model> <port> <tp> <gpus> [VAR=value…]: one vLLM server in its own process group
  local name="$1" model="$2" port="$3" tp="$4" gpus="$5"; shift 5
  bg "${name}_${port}.log" env "$@" DPT_VLLM_DP=1 CUDA_VISIBLE_DEVICES="${gpus}" GPU_MEM=0.9 \
    setsid bash scripts/serve_vllm.sh "${model}" "${name}" "${port}" 40960 "${tp}"
  LOCAL+=("${BG}"); STARTING+=("${port}:${BG}")
}
up_here() {      # wait for the servers started by serve_here since the last call (they start in parallel)
  local s; for s in "${STARTING[@]}"; do wait_url "http://${N1}:${s%%:*}" 3600 "${s##*:}"; done; STARTING=()
}
proxy_here() {   # proxy_here <name> <port> <backend…>: rr_proxy on this node, published as endpoint <name>
  local name="$1" port="$2" b args=(); shift 2
  for b in "$@"; do args+=(--backend "${b}"); done
  bg "proxy_${name}.log" setsid python scripts/rr_proxy.py --port "${port}" "${args[@]}"
  LOCAL+=("${BG}"); wait_url "http://${N1}:${port}" 300 "${BG}"
  publish "${name}" "http://${N1}:${port}"
}
publish()   { dry && return 0; echo "$2" > "${DPT_ENDPOINTS}/$1.url"; echo "$1" > "${DPT_ENDPOINTS}/$1.model"; }
unpublish() { [[ "$(endpoint_url "$1")" == "http://${N1}:"* ]] && rm -f "${DPT_ENDPOINTS}/$1.url" "${DPT_ENDPOINTS}/$1.model"; true; }
stop_here() { local p; for p in "$@"; do kill -TERM -- "-${p}" 2>/dev/null || true; done; }
stop_n2()   { local f; for f in "${LOGS}"/n2_*.pid; do
                [[ -f "${f}" ]] && on_node "${N2}" bash -c "kill -TERM -- -$(cat "${f}") 2>/dev/null; true" || true
              done; }
cleanup() {
  local n
  for n in teacher judge_v sft_audit sft_new; do unpublish "${n}"; done
  stop_here "${LOCAL[@]}"; stop_n2
  kill $(jobs -p) 2>/dev/null || true
}
trap cleanup EXIT
gpus_free() {   # wait until this node's GPUs hold no process (after stopping servers), up to 5 min
  dry && return 0
  local i=0
  until [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; do
    sleep 10; i=$((i + 10)); (( i < 300 )) || tdie "GPUs still busy 5 min after stopping the teacher and Gemma"
  done
}
must() {   # must <pid> <what>: wait for a background step and stop the job if it failed
  wait "$1" || tdie "$2 failed: see ${LOGS}"
}

# ---- servers --------------------------------------------------------------------------------------------------
tlog "node 1 ${N1}: teacher TP2 + Gemma; node 2 ${N2}: the SFT model ×3"
for g in 0 1 2; do
  on_node "${N2}" bash -c "DPT_GDN_MODEL=1 DPT_VLLM_DP=1 CUDA_VISIBLE_DEVICES=${g} GPU_MEM=0.9 setsid nohup bash \
scripts/serve_vllm.sh '${SFT}' sft_audit $((8001 + g)) 40960 1 > '${LOGS}/sft_audit_${N2}_$((8001 + g)).log' 2>&1 \
< /dev/null & echo \$! > '${LOGS}/n2_$((8001 + g)).pid'"
done
serve_here teacher "${TEACHER_MODEL}" 8000 2 0,1
serve_here judge_v "${JUDGE_MODEL}" 8100 1 2 DPT_SERVE_DP=1
up_here
publish teacher "http://${N1}:8000"; publish judge_v "http://${N1}:8100"
for g in 0 1 2; do wait_url "http://${N2}:$((8001 + g))" 3600; done
proxy_here sft_audit 8200 "http://${N2}:8001" "http://${N2}:8002" "http://${N2}:8003"

# ---- phase A ----------------------------------------------------------------------------------------------------
tlog "phase A: hard + long verification, atomic T3, v1 difficulty"
bg hard.log env DPT_BANK_OUT="${DATA}/v1_hard" DPT_BANK_NAME=hard DPT_BANK_CANDIDATES="${G}/t1_hard_candidates.jsonl" \
  DPT_BANK_STAGES=ingest,quality,dedup,decontam,verify DPT_JUDGE_ENDPOINT=judge_v bash scripts/slurm/data.slurm bank
A_HARD=${BG}
bg long.log env DPT_BANK_OUT="${DATA}/v1_long" DPT_BANK_NAME=long DPT_BANK_CANDIDATES="${G}/t1_long_candidates.jsonl" \
  DPT_BANK_STAGES=ingest,quality,dedup,decontam,verify DPT_JUDGE_ENDPOINT=judge_v bash scripts/slurm/data.slurm bank
A_LONG=${BG}
bg t3_atomic.log env DPT_TEACHER_ENDPOINT=teacher DPT_JUDGE_ENDPOINT=judge_v bash scripts/slurm/t3_atomic.slurm
A_T3=${BG}
bg v2a.log env DPT_BANK_OUT="${V2}" DPT_BANK_NAME=v2a DPT_BANK_STAGES=difficulty DPT_DIFFICULTY_ENDPOINT=sft_audit \
  DPT_DIFFICULTY_MODE=audit_all DPT_DIFFICULTY_SKIP="[t3_report,t2_code]" DPT_DIFFICULTY_CONCURRENCY=96 \
  bash scripts/slurm/data.slurm bank
A_V1=${BG}
must "${A_HARD}" "hard verification"; must "${A_LONG}" "long verification"; must "${A_T3}" "atomic T3 rubrics"
tlog "phase A: teacher work done"

# ---- phase B ----------------------------------------------------------------------------------------------------
tlog "phase B: node 1 switches to the SFT model; difficulty of the new problems"
unpublish teacher; unpublish judge_v
stop_here "${LOCAL[@]:0:2}"; gpus_free
for g in 0 1 2; do serve_here sft_new "${SFT}" $((8301 + g)) 1 "${g}" DPT_GDN_MODEL=1; done
up_here
proxy_here sft_new 8300 "http://${N1}:8301" "http://${N1}:8302" "http://${N1}:8303"
run mkdir -p "${NEW}/stages"
run python -m dpt.data.bank_merge merge --name rl_bank.verify --out "${NEW}/stages/04_verify.jsonl" \
  "${DATA}/v1_hard/stages/04_verify.jsonl" "${DATA}/v1_long/stages/04_verify.jsonl"
bg v2new.log env DPT_BANK_OUT="${NEW}" DPT_BANK_NAME=v2new DPT_BANK_STAGES=difficulty DPT_DIFFICULTY_ENDPOINT=sft_new \
  DPT_DIFFICULTY_MODE=audit_all DPT_DIFFICULTY_SKIP="[t3_report,t2_code]" DPT_DIFFICULTY_CONCURRENCY=96 \
  bash scripts/slurm/data.slurm bank
must "${BG}" "difficulty of the new problems"
must "${A_V1}" "difficulty of the v1 problems"

# ---- phase C ----------------------------------------------------------------------------------------------------
tlog "phase C: merge, publish, check"
run python -m dpt.data.bank_merge merge --name rl_bank.difficulty --out "${V2}/stages/05_difficulty.jsonl" \
  --t3-map "${V2}/t3_atomic_map.jsonl" "${V2}/stages/05_difficulty.jsonl" "${NEW}/stages/05_difficulty.jsonl"
run env DPT_BANK_OUT="${V2}" DPT_BANK_NAME=v2pub DPT_BANK_STAGES=publish bash scripts/slurm/data.slurm bank
run python -m dpt.data.bank_merge check "${V2}" --reference-dev "${DATA}/v1/rl_dev.jsonl"
tlog "bank v2 ready: ${V2} (check: ${V2}/bank_check.json)"
