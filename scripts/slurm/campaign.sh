#!/bin/bash
# Steps of the single 8-node training job (scripts/slurm/campaign.slurm; plan §8, decided 2026-10-24).
# Each finished step leaves a marker in $WORK/dpt/campaign/state, so a rerun (a resubmission after a crash,
# or the next job) continues with the first unfinished step:
#
#   (bank)    only while the RL bank is unpublished: its difficulty pass and publish (data.slurm bank) on the
#             base 9B as 24 one-GPU replicas on all 8 nodes, ~2 h instead of ~1.5 days on a one-GPU base job.
#             Replies already received come from the pipeline cache. No marker: rl_data checks the bank files.
#   baseline  the base 9B as 21 one-GPU vLLM replicas on the 7 worker nodes behind scripts/rr_proxy.py, the
#             Gemma judge as 3 replicas on the head node, the T2 sandbox on the head node:
#             scripts/03_baseline_eval.sh (all sets, noise floor ×10, pass@64)
#   sft_data  wait up to DPT_CAMPAIGN_DATA_WAIT_S (4 h) for the SFT mix (data.slurm sft → gen/.done/sft_mix)
#   sft       scripts/04_sft.sh on all 8 nodes (TP1 × PP8 × DP3); the HF export is copied to $WORK
#   sft_eval  the export served like the base: scripts/07_final_eval.sh --only sft (report included), then
#             the RL-entry gate (dpt.eval.gates --entry). A failed gate ends the job: state/GATE_FAILED. On the
#             9B line it first holds up to 8 h for `go` or `stop` in campaign/GATE_DECISION (configs/model).
#   rl_data   the published RL bank (RL fails without it); if it is still missing, the bank pass runs once more
#   rl        dpt.launch.rl until the run is done or the walltime runs out (a successor rl.slurm is queued).
#             Its judge is the external `judge` endpoint (serve.slurm on gpu-a100-small, submitted with
#             --dependency=after:<this job>). If it is not up yet, RL runs stage A only (DPT_STOP_AT_STEP),
#             waits up to DPT_CAMPAIGN_JUDGE_WAIT_S (4 h) for the judge, then resumes. A watchdog asks the
#             trainer to hold (HOLD_REQUEST) when the judge stops answering or moves, e.g. at its walltime.
#
# Before this job starts, the difficulty pass runs from a login node on a separate one-GPU base job (its id in
# $WORK/dpt/campaign/base_job). The bank step writes campaign/bank_takeover and cancels that job; the runner then
# stops its client and answers with campaign/bank_released (heartbeat: campaign/bank_runner.alive), so only one
# client writes the cache. Remove bank_takeover to give the pass back to a login-node runner.
#
# Pending timing runs (dpt-rl jobs) are cancelled at the start: they would hold the nodes after this job.
# The RL judge job named in $WORK/dpt/campaign/judge_job is cancelled when this job ends without queuing an
# rl.slurm successor (RL done, or a failure such as the gate): it would otherwise idle to its own walltime.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/slurm/env.sh
source scripts/slurm/lib.sh
export HF_HUB_OFFLINE=1
C="${WORK}/dpt/campaign"; ST="${C}/state"; LOGS="${C}/logs/${SLURM_JOB_ID:-dry}"; PIDS="${C}/pids/${SLURM_JOB_ID:-dry}"
mkdir -p "${ST}" "${LOGS}" "${PIDS}"
read -r DATA SECURE EVAL_ROOT CKPT BASE_ID TAG JUDGE_ID T3_START RL_FILES <<< "$(python - <<'PY'
from dpt.common.config import load_config
from dpt.launch.common import compose_stage
p = load_config().project
c = compose_stage("rl/rl_9b.yaml")
t3 = min(int(s.start) for s in c.curriculum.stages if float(s.mix.t3_report) > 0)
print(p.paths.data_root, p.paths.eval_secure_root, p.paths.eval_root, p.paths.ckpt_root, p.model.base, p.model.tag,
      p.judges.primary.hf_id, t3, ",".join([*c.rl_run.train_files, *c.rl_run.dev_files]))
PY
)"
mapfile -t NODES < <(job_nodes)
HEAD="${NODES[0]}"; WORKERS=("${NODES[@]:1}")
SFT_EXPORT="${CKPT}/sft/sft${TAG}_seed1/hf_export"
RL_CONFIG=rl/rl_9b.yaml
tlog "campaign on ${#NODES[@]} nodes (head ${HEAD}); state ${ST}; logs ${LOGS}"

finished() { [[ -e "${ST}/$1" ]]; }
mark() { date > "${ST}/$1"; tlog "campaign: $1 done"; }

# ---- one-GPU vLLM replicas and the least-busy proxy in front of them ----------------------------------
start_replicas() {   # start_replicas <host> <model> <served name> <first port> [env]: one server per GPU
  local host="$1" model="$2" name="$3" port0="$4" extra="${5:-}" g
  for g in 0 1 2; do
    on_node "${host}" bash -c "${extra} DPT_VLLM_DP=1 CUDA_VISIBLE_DEVICES=${g} GPU_MEM=0.9 setsid nohup bash \
scripts/serve_vllm.sh '${model}' '${name}' $((port0 + g)) 40960 1 > '${LOGS}/${name}_${host}_$((port0 + g)).log' \
2>&1 < /dev/null & echo \$! > '${PIDS}/${host}_$((port0 + g))'"
  done
}
start_proxy() {      # start_proxy <port> <backend URL>...: on this (head) node
  local port="$1" b args=(); shift
  for b in "$@"; do args+=(--backend "${b}"); done
  if dry; then run python scripts/rr_proxy.py --port "${port}" "${args[@]}"; return 0; fi
  setsid nohup python scripts/rr_proxy.py --port "${port}" "${args[@]}" > "${LOGS}/proxy_${port}.log" 2>&1 < /dev/null &
  echo $! > "${PIDS}/${HEAD}_${port}"
}
wait_backends() {    # wait_backends <URL>...: every server answers (a model load takes ~5-15 min)
  local u
  for u in "$@"; do wait_url "${u}" 3600 || tdie "${u} did not come up (logs ${LOGS})"; done
}
stop_servers() {     # every server and proxy this job started, then wait until the GPUs are free
  local f host sig busy groups=()
  for f in "${PIDS}"/*; do
    [[ -e "${f}" ]] || continue
    host="$(basename "${f}")"; groups+=("${host%_*}:$(cat "${f}")"); rm -f "${f}"
  done
  for sig in TERM KILL; do     # a server that ignores SIGTERM for 3 min gets SIGKILL
    for f in "${groups[@]}"; do on_node "${f%%:*}" bash -c "kill -${sig} -- -${f##*:} 2>/dev/null; true" || true; done
    dry && return 0
    for _ in $(seq 1 18); do
      busy=0
      for host in "${NODES[@]}"; do
        [[ -z "$(on_node "${host}" nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]] || busy=1
      done
      (( busy )) || return 0
      sleep 10
    done
  done
  tlog "WARN GPUs still in use after stopping the servers"
}
serve_for_eval() {   # serve_for_eval <model> <served name>: judge on the head node, the model on the workers
  local h p urls=()
  start_replicas "${HEAD}" "${JUDGE_ID}" judge 8101
  # DPT_GDN_MODEL: a local export path does not name the Qwen hybrid (GDN) architecture (serve_vllm.sh)
  for h in "${WORKERS[@]}"; do start_replicas "${h}" "$1" "$2" 8001 DPT_GDN_MODEL=1; done
  for h in "${WORKERS[@]}"; do for p in 8001 8002 8003; do urls+=("http://${h}:${p}"); done; done
  wait_backends http://"${HEAD}":810{1,2,3} "${urls[@]}"
  start_proxy 8100 http://127.0.0.1:810{1,2,3}
  start_proxy 8000 "${urls[@]}"
  wait_backends "http://${HEAD}:8100" "http://${HEAD}:8000"
}
SANDBOX=""
start_sandbox() {    # D-Code / LiveCodeBench hidden tests stay on this node (as in eval.slurm)
  [[ -n "${SANDBOX}" ]] && return 0
  sandbox_image || tdie "no sandbox image (check_node.slurm pulls it)"
  run python -m dpt.rewards.sandbox_server --host 127.0.0.1 --port 8080 --hidden-tests-dir "${SECURE}/hidden_tests" \
    --max-concurrency 32 --python "${DPT_SANDBOX_PYTHON}" --container-image "${DPT_SANDBOX_SIF}" &
  SANDBOX=$!
}
eval_env() {         # harness settings for the in-job servers (eval_suite.yaml keys)
  export DPT_EVAL_ENDPOINT="http://127.0.0.1:8000"
  export DPT_EVAL_OVERRIDES="eval_suite.sandbox_endpoint=http://127.0.0.1:8080 \
eval_suite.judge.endpoint=http://127.0.0.1:8100 eval_suite.judge.model=judge eval_suite.judge.max_concurrency=24 \
eval_suite.max_concurrency=384"
}
JUDGE_JOB="$(cat "${C}/judge_job" 2>/dev/null || true)"
cleanup() {
  stop_servers || true
  [[ -n "${SANDBOX}" ]] && kill "${SANDBOX}" 2>/dev/null || true
  ray_down >/dev/null 2>&1 || true
  # The RL judge serves this job and its rl.slurm successor (NEXT) only.
  if [[ -n "${JUDGE_JOB}" && -z "${NEXT:-}" ]]; then
    tlog "releasing the RL judge job ${JUDGE_JOB}"; run scancel "${JUDGE_JOB}" || true
  fi
}
trap cleanup EXIT

# ---- 0: nothing else of ours may hold 8 nodes after this job ------------------------------------------
for j in $(squeue -h -u "${USER}" -t PENDING -n dpt-rl -o %i 2>/dev/null || true); do
  tlog "cancelling pending RL job ${j} (the campaign runs RL itself)"; run scancel "${j}"
done
[[ -e "${ST}/GATE_FAILED" ]] && tdie "the RL-entry gate failed earlier (${ST}/GATE_FAILED): a human decides what next"

# ---- 1: the RL bank, while it is unpublished --------------------------------------------------------------
IFS=, read -ra BANK <<< "${RL_FILES}"
bank_missing() { local f; for f in "${BANK[@]}"; do [[ -f "${f}" ]] || { echo "${f}"; return 0; }; done; return 1; }
take_over_bank() {   # stop the login-node runner of the pass and its base job: one client writes the cache
  local waited=0 job
  rm -f "${C}/bank_released"; date > "${C}/bank_takeover"
  job="$(cat "${C}/base_job" 2>/dev/null || true)"
  [[ -z "${job}" ]] || { tlog "releasing the base-model job ${job}"; run scancel "${job}" || true; }
  until [[ -e "${C}/bank_released" || -z "$(find "${C}/bank_runner.alive" -mmin -3 2>/dev/null)" ]]; do
    (( waited >= ${DPT_CAMPAIGN_RUNNER_WAIT_S:-600} )) && { tlog "WARN the login-node difficulty runner is still running"; return 1; }
    sleep 10; waited=$((waited + 10))
  done
}
bank_pass() {        # difficulty + publish on 24 one-GPU base replicas
  local h urls=() rc=0 ep="${C}/endpoints/${SLURM_JOB_ID:-dry}"
  take_over_bank || return 1
  tlog "RL bank: difficulty pass and publish on ${#NODES[@]} nodes × 3 GPUs"
  for h in "${NODES[@]}"; do start_replicas "${h}" "${BASE_ID}" base 8001 DPT_GDN_MODEL=1; urls+=(http://"${h}":800{1,2,3}); done
  wait_backends "${urls[@]}"
  start_proxy 8000 "${urls[@]}"
  wait_backends "http://${HEAD}:8000"
  mkdir -p "${ep}"; echo "http://127.0.0.1:8000" > "${ep}/base.url"; echo base > "${ep}/base.model"
  # 24 problems (192 replies) per GPU keep every server full; a request then waits ~40 min (timeout 4 h).
  run env DPT_ENDPOINTS="${ep}" DPT_BANK_STAGES=difficulty,publish DPT_DIFFICULTY_SKIP='[t3_report,t2_code]' \
    DPT_DIFFICULTY_CONCURRENCY=$((24 * ${#urls[@]})) bash scripts/slurm/data.slurm bank 2>&1 | tee "${LOGS}/bank.log" || rc=$?
  stop_servers
  return "${rc}"
}
if ! finished rl_data && bank_missing >/dev/null; then
  bank_pass || tlog "WARN no RL bank yet (${LOGS}/bank.log): the pass runs once more before RL"
fi

# ---- 2: baseline eval ------------------------------------------------------------------------------------
if ! finished baseline && [[ -f "${EVAL_ROOT}/base_medium/results.json" && -f "${EVAL_ROOT}/noise/noise_floor.json" \
      && -f "${EVAL_ROOT}/base_medium_k64/results.json" ]]; then
  tlog "baseline results exist already (${EVAL_ROOT}): not rerunning them"; mark baseline
fi
if ! finished baseline; then
  tlog "baseline eval: ${BASE_ID} on ${#WORKERS[@]} nodes × 3 GPUs, judge ${JUDGE_ID} on ${HEAD}"
  start_sandbox; eval_env; serve_for_eval "${BASE_ID}" base
  run bash scripts/03_baseline_eval.sh 2>&1 | tee "${LOGS}/baseline_eval.log"
  stop_servers
  mark baseline
fi

# ---- 3: the SFT data --------------------------------------------------------------------------------------
if ! finished sft_data; then
  waited=0
  until [[ -e "${DATA}/gen/.done/sft_mix" && -f "${DATA}/v1/sft/sft_train.jsonl" ]] || dry; do
    (( waited >= ${DPT_CAMPAIGN_DATA_WAIT_S:-14400} )) && tdie "no SFT mix after ${waited} s (data.slurm sft)"
    (( waited % 1800 == 0 )) && tlog "waiting for the SFT mix (${DATA}/gen/.done/sft_mix)"
    sleep 60; waited=$((waited + 60))
  done
  mark sft_data
fi

# ---- 4: SFT ----------------------------------------------------------------------------------------------
if ! finished sft; then
  MASTER="$(getent hosts "${HEAD}" 2>/dev/null | awk '{print $1; exit}' || true)"; MASTER="${MASTER:-${HEAD}}"
  tlog "SFT on ${#NODES[@]} nodes, master ${HEAD} (${MASTER})"
  pids=(); i=0
  for h in "${NODES[@]}"; do
    vars=(NNODES="${#NODES[@]}" NODE_RANK="${i}" MASTER_ADDR="${MASTER}" MASTER_PORT=29500)
    if (( i == 0 )); then   # node 0 also exports to HF
      run env "${vars[@]}" bash scripts/04_sft.sh --config sft/sft_9b.yaml > "${LOGS}/sft_node${i}.log" 2>&1 &
    else
      on_node "${h}" env "${vars[@]}" bash scripts/04_sft.sh --config sft/sft_9b.yaml > "${LOGS}/sft_node${i}.log" 2>&1 &
    fi
    pids+=($!); i=$((i + 1))
  done
  rc=0; for p in "${pids[@]}"; do wait "${p}" || rc=1; done
  [[ ${rc} -eq 0 ]] || tdie "SFT failed on at least one node (${LOGS}/sft_node*.log)"
  dry || [[ -f "${SFT_EXPORT}/config.json" ]] || tdie "no HF export at ${SFT_EXPORT}"
  run rsync -a "${SFT_EXPORT}/" "${WORK}/dpt/exports/sft${TAG}_seed1/"   # $SCRATCH is purged after 10 days
  mark sft
fi

# ---- 5: SFT eval and the RL-entry gate --------------------------------------------------------------------
if ! finished sft_eval; then
  tlog "SFT eval: ${SFT_EXPORT} on ${#WORKERS[@]} nodes × 3 GPUs"
  start_sandbox; eval_env; serve_for_eval "${SFT_EXPORT}" sft
  run bash scripts/07_final_eval.sh --only sft 2>&1 | tee "${LOGS}/sft_eval.log"
  stop_servers
  if ! run python -m dpt.eval.gates --summary "${EVAL_ROOT}/summary.json" --candidate sft --entry \
      2>&1 | tee "${LOGS}/entry_gate.log"; then
    date > "${ST}/GATE_FAILED"
    tdie "the RL-entry gate failed (${LOGS}/entry_gate.log, ${EVAL_ROOT}/report.md): not starting RL"
  fi
  mark sft_eval
fi

# ---- 6: the RL bank ----------------------------------------------------------------------------------------
if ! finished rl_data; then
  if bank_missing >/dev/null; then bank_pass || true; fi
  if m="$(bank_missing)"; then tdie "no RL bank (${m}) after its difficulty pass (${LOGS}/bank.log)"; fi
  mark rl_data
fi

# ---- 7: RL ---------------------------------------------------------------------------------------------------
describe() { python -m dpt.launch.rl --config "${RL_CONFIG}" --describe; }
judge_watchdog() {   # judge_watchdog <run dir> <judge URL>: request a hold once the judge is gone or has moved
  local run_dir="$1" url="$2" misses=0 now
  while sleep 60; do
    now="$(endpoint_url judge || true)"
    if [[ "${now}/v1" == "${url}" ]] && curl -sf --max-time 10 "${url}/models" >/dev/null; then misses=0; continue; fi
    misses=$((misses + 1))
    (( misses < 5 )) && continue
    echo "judge ${url} gone or moved (endpoint now: ${now:-none})" > "${run_dir}/HOLD_REQUEST"
    tlog "judge watchdog: hold requested (${url} gone or moved)"; return 0
  done
}
export DPT_DEADLINE_MARGIN_S=1200      # the trainer re-reads the limit from Slurm (an extension by the site counts)
until finished rl; do
  STATE="$(describe)"; RUN_DIR="$(json_field "${STATE}" run_dir)"; STEP="$(json_field "${STATE}" latest_step)"
  [[ "$(json_field "${STATE}" done)" == true ]] && { mark rl; break; }
  [[ "$(json_field "${STATE}" paused)" == true ]] && tdie "RL is PAUSED by an alert: ${RUN_DIR}/PAUSED.json (docs/runbook.md)"
  unset DPT_STOP_AT_STEP DPT_JUDGE_URL DPT_JUDGE_MODEL
  HOLD=$((T3_START - 2))                 # the last stage-A steps, with a step to spare
  if endpoint_up judge; then
    export DPT_JUDGE_URL="$(endpoint_url judge)/v1" DPT_JUDGE_MODEL="$(endpoint_model judge)"
    tlog "RL from step ${STEP}: judge ${DPT_JUDGE_MODEL} at ${DPT_JUDGE_URL}"
  elif (( STEP < HOLD )); then
    export DPT_STOP_AT_STEP="${HOLD}"
    tlog "RL from step ${STEP} without a judge (its job is still queued): stage A only, hold at step ${HOLD}"
  else
    tlog "RL at step ${STEP} needs the judge from step ${T3_START}: waiting for it"
    wait_endpoint judge "${DPT_CAMPAIGN_JUDGE_WAIT_S:-14400}" || tdie "no judge endpoint after the wait"
    continue
  fi
  run rm -f "${RUN_DIR}/WALLTIME.json" "${RUN_DIR}/HOLD.json" "${RUN_DIR}/HOLD_REQUEST"
  if dry; then export DPT_DEADLINE=$(( $(date +%s) + 172800 )); else
    export DPT_DEADLINE="$(python -m dpt.launch.walltime --slurm-deadline --margin-s 1200)"
  fi
  tlog "training must stop by $(date -d "@${DPT_DEADLINE}" '+%F %T')"
  ray_up 3
  WD=""
  if [[ -n "${DPT_JUDGE_URL:-}" ]] && ! dry; then judge_watchdog "${RUN_DIR}" "${DPT_JUDGE_URL}" & WD=$!; fi
  RC=0; run python -m dpt.launch.rl --config "${RL_CONFIG}" 2>&1 | tee -a "${LOGS}/rl.log" || RC=$?
  [[ -z "${WD}" ]] || kill "${WD}" 2>/dev/null || true
  ray_down
  dry && break
  STATE="$(describe)"
  if [[ "$(json_field "${STATE}" done)" == true ]]; then
    mark rl
  elif [[ "$(json_field "${STATE}" held)" == true ]]; then
    tlog "RL held at step $(json_field "${STATE}" latest_step); continuing when the judge is up"
  elif [[ "$(json_field "${STATE}" walltime)" == true ]]; then
    NEXT="$(sbatch --parsable -A "${SLURM_JOB_ACCOUNT}" --export="ALL,DPT_CHAIN_LEFT=4" scripts/slurm/rl.slurm \
      --config "${RL_CONFIG}" | job_id)"
    tlog "walltime reached at step $(json_field "${STATE}" latest_step): rl.slurm ${NEXT} continues the run"
    break
  else
    tdie "RL exited with code ${RC} without a clean stop (${LOGS}/rl.log, ${RUN_DIR})"
  fi
done
tlog "campaign finished: $(ls "${ST}" | tr '\n' ' ')"
