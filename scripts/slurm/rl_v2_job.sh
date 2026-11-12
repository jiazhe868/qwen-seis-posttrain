#!/bin/bash
# RL v2 as one 8-node job (2026-11-11), run by scripts/slurm/rl_v2_queue.slurm when that job starts. This file is
# read from the repository at that moment, so it can still change while the job waits in the queue.
#
#   1. bank v2. If $DATA/v2/bank_check.json is missing or lists problems, stop window 1 (dpt-datawin), which builds
#      the same bank. If window 1 had published it, put back run 1's dev split (bank_merge pin-dev) and check again.
#      Otherwise build it here with data_window.slurm on all 8 nodes: 2 teacher nodes, 6 nodes of the SFT model,
#      and no atomic T3 rubrics (RL v2 trains no T3). Stages window 1 already finished are skipped.
#   2. the dev split without T3 (dpt.data.bank_merge without-task): no judge runs in this job.
#   3. the 24K test. One RL step in a throwaway run, with every answer forced to the full 24K tokens (ignore_eos),
#      the MTP head frozen, and 6 prompts × 8 (8 sequences per data-parallel rank, so the first pipeline stage holds
#      as many in flight as in the real run). The last pipeline stage is the memory limit of this layout. With the
#      MTP head trained ~17K tokens fit there; freezing it removes its [tokens × 248,320] logits (~0.47 MiB per token
#      in BF16), so 24K should fit, but that was never measured (memory note of 2026-10-29). If the step finishes,
#      RL v2 trains at 24K with the MTP head frozen, else at 16K like run 1. Serving never drafts with the MTP head
#      (serve_vllm.sh: SPEC=0), so a frozen head changes no eval output. The choice is kept in
#      <ckpt>/rl/rl9b_v2_seed1.cap.json, so a later job resumes the run at the same cap.
#   4. RL v2: rl.slurm --config rl/rl_9b_v2.yaml inside this job (no successor). The trainer saves and stops
#      DPT_EVAL_H hours (default 5) + 20 min before the job's limit.
#   5. the final eval on the same 8 nodes (final_eval_window.slurm), whatever way RL ended.
#
# Logs: this job's output, and $WORK/dpt/logs/rlv2_<job>/ (bank build, 24K test).
# Dry run: DPT_SLURM_DRY_RUN=1 DPT_SLURM_NODES="<8 nodes>" bash scripts/slurm/rl_v2_job.sh
# DPT_RL_CAP=16k skips the 24K test; DPT_RL_CAP=24k skips it and trains at 24K (DPT_RL_V2_CAP is the older name).
# Job 2 runs RL v3 with this script (2026-11-12): DPT_RL_CONFIG=rl/rl_9b_v3.yaml DPT_FINAL_ROW=rl_v3_seed1. The run's
# directory, cap file and 24K test follow the config's run name.
#
# The whole script is one { … } block: bash reads all of it before the first step, so a copy synced while the
# job runs cannot change the running job's steps (scripts it calls are read when they start).
{
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/slurm/env.sh
source scripts/slurm/lib.sh
export HF_HUB_OFFLINE=1

mapfile -t NODES < <(job_nodes)
(( ${#NODES[@]} == 8 )) || tdie "RL v2 needs 8 nodes, this job has ${#NODES[@]}"
EVAL_S=$(( ${DPT_EVAL_H:-5} * 3600 ))
LOGS="${WORK}/dpt/logs/rlv2_${SLURM_JOB_ID:-local}"
mkdir -p "${LOGS}"
read -r DATA CK TAG <<< "$(python - <<'PY'
from dpt.common.config import load_config
p = load_config().project
print(p.paths.data_root, p.paths.ckpt_root, p.model.tag)
PY
)"
V2="${DATA}/v2"; CHECK="${V2}/bank_check.json"
RL_CONFIG="${DPT_RL_CONFIG:-rl/rl_9b_v2.yaml}"; FINAL_ROW="${DPT_FINAL_ROW:-rl_v2_seed1}"
RUN_DIR="$(json_field "$(python -m dpt.launch.rl --config "${RL_CONFIG}" --describe)" run_dir)"
RUN_NAME="$(basename "${RUN_DIR}")"; CAP_FILE="${RUN_DIR}.cap.json"
# 24K: the response cap, the length penalty from 20K (the same 4K ramp as at 16K), and the MTP head frozen
CAP_24K=(project.thinking.rl_max_response=24576 project.thinking.rl_overlong_start=20480
         verl.actor_rollout_ref.model.mtp.enable_train=false)

time_left_s() { if dry; then echo 172800; else python -c 'from dpt.launch.walltime import slurm_time_left; print(slurm_time_left())'; fi; }
bank_ready() {   # bank_ready: bank v2 exists and its check lists no problems
  [[ -f "${CHECK}" ]] && python - "${CHECK}" <<'PY'
import json, sys
problems = json.load(open(sys.argv[1])).get("problems", ["no problems field"])
for p in problems:
    print("bank check:", p, file=sys.stderr)
sys.exit(1 if problems else 0)
PY
}
gpus_free() {    # gpus_free: wait up to 10 min until no process holds a GPU on this job's nodes
  dry && return 0
  local h i busy=()
  for i in $(seq 1 60); do
    busy=()
    for h in "${NODES[@]}"; do
      [[ -z "$(on_node "${h}" nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]] || busy+=("${h}")
    done
    (( ${#busy[@]} )) || return 0
    sleep 10
  done
  tlog "WARN GPUs still in use after 10 min on ${busy[*]}"
}

# ---- 1. bank v2 --------------------------------------------------------------------------------------------------
tlog "RL v2 on ${NODES[*]} ($(( $(time_left_s) / 3600 )) h left in this job)"
if bank_ready; then
  tlog "bank v2 is ready (${CHECK})"
else
  mapfile -t WINDOW_1 < <(squeue -h -u "${USER}" -n dpt-datawin -o %i 2>/dev/null || true)
  if (( ${#WINDOW_1[@]} )); then
    tlog "stopping window 1 (${WINDOW_1[*]}): this job builds bank v2 itself"
    run scancel "${WINDOW_1[@]}"
    for i in $(seq 1 90); do
      dry || [[ -z "$(squeue -h -j "$(IFS=,; echo "${WINDOW_1[*]}")" -o %i 2>/dev/null)" ]] && break
      sleep 10
    done
  fi
  if [[ -e "${DATA}/gen/.done/v2pub_publish" ]]; then
    # A bank published by window 1's own copy of data_window.slurm, frozen at its submission before pin-dev existed:
    # its dev split was drawn after the SFT model's re-grading. Put back run 1's split and check again (CPU, ~1 min).
    PIN_T3=(); [[ -f "${V2}/t3_atomic_map.jsonl" ]] && PIN_T3=(--t3-map "${V2}/t3_atomic_map.jsonl")
    run python -m dpt.data.bank_merge pin-dev --reference-dev "${DATA}/v1/rl_dev.jsonl" "${PIN_T3[@]}" "${V2}"
    run python -m dpt.data.bank_merge check "${V2}" --reference-dev "${DATA}/v1/rl_dev.jsonl" --t3-not-trained || true
  fi
  # Two of our 8-node jobs (RL v2, job 2) could run at once with no bank yet: the second waits for the first's build
  # instead of writing the same files ($V2/BUILDING holds the builder's job id).
  BUILDER="$(cat "${V2}/BUILDING" 2>/dev/null || true)"
  running() { [[ -n "$1" ]] && squeue -h -j "$1" -t R -o %i 2>/dev/null | grep -q .; }
  if ! dry && [[ "${BUILDER}" != "${SLURM_JOB_ID:-local}" ]] && running "${BUILDER}"; then
    tlog "job ${BUILDER} is building bank v2: waiting for its check"
    waited=0
    until bank_ready || ! running "${BUILDER}"; do
      (( waited < 16 * 3600 )) || tdie "bank v2 is not ready after 16 h of job ${BUILDER}"
      sleep 600; waited=$(( waited + 600 ))
    done
  fi
  if bank_ready; then
    tlog "bank v2 is ready with run 1's dev split (${CHECK})"
  else
    tlog "building bank v2 on all 8 nodes (logs ${LOGS}/data)"
    dry || { mkdir -p "${V2}"; echo "${SLURM_JOB_ID:-local}" > "${V2}/BUILDING"; }
    env DPT_SLURM_NODES="${NODES[*]}" DPT_DATA_TEACHER_NODES=2 DPT_DATA_SKIP_T3=1 DPT_DATA_LOGS="${LOGS}/data" \
      bash scripts/slurm/data_window.slurm || tdie "the bank build failed: see ${LOGS}/data"
    dry || rm -f "${V2}/BUILDING"
    dry || bank_ready || tdie "bank v2 did not pass its check: ${CHECK}"
    gpus_free
  fi
fi

# ---- 2. the dev split without T3 -----------------------------------------------------------------------------------
run python -m dpt.data.bank_merge without-task --task t3_report --out "${V2}/rl_dev_no_t3" "${V2}/rl_dev.jsonl"

# ---- 3. the 24K test -------------------------------------------------------------------------------------------------
CAP="${DPT_RL_CAP:-${DPT_RL_V2_CAP:-}}"
if [[ -z "${CAP}" && -f "${CAP_FILE}" ]]; then
  CAP="$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["cap"])' "${CAP_FILE}")"
  tlog "the run already chose its cap: ${CAP} (${CAP_FILE})"
fi
if [[ -z "${CAP}" ]]; then
  TEST_RUN="${RUN_DIR}_memtest24k"
  TEST_ARGS=(--config "${RL_CONFIG}" "${CAP_24K[@]}" rl_run.run_name="${RUN_NAME}_memtest24k"
             rl.prompts_per_step=6 rl.total_steps=1 rl.lr_schedule.warmup_steps=0 rl.lr_schedule.final_phase_steps=1
             rl_run.save_freq=-1 rl_run.dev_freq=-1 verl.trainer.val_before_train=false
             verl.actor_rollout_ref.rollout.ignore_eos=true rl.filtering.drop_zero_advantage_groups=false
             "monitoring.rules=[]" "monitoring.extra_rules=[]")
  tlog "24K test: one step, every answer 24,576 tokens, MTP head frozen (${TEST_RUN})"
  run rm -rf "${TEST_RUN}"
  memtest() {   # the test's launcher and verl under it in one process group, stopped after 75 min: model loading
                # ~15 min, 2 answers of 24K per vLLM replica at ~80 tok/s ~5 min, the update ~5 min
    if dry; then run python -m dpt.launch.rl "${TEST_ARGS[@]}"; return; fi
    setsid python -m dpt.launch.rl "${TEST_ARGS[@]}" > "${LOGS}/memtest24k.log" 2>&1 < /dev/null &
    local pid=$! watchdog rc=0
    ( sleep 4500; kill -TERM -- "-${pid}" 2>/dev/null; sleep 60; kill -KILL -- "-${pid}" 2>/dev/null ) &
    watchdog=$!
    wait "${pid}" || rc=$?
    kill "${watchdog}" 2>/dev/null || true
    return "${rc}"
  }
  ray_up 3
  TEST_RC=0
  memtest || TEST_RC=$?
  ray_down
  CAP=16k
  if dry; then
    CAP=24k
  elif (( TEST_RC == 0 )) && python - "${TEST_RUN}/metrics.jsonl" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1])]
step = next(r for r in rows if r.get("step") == 1)
print("24K test: longest answer", step.get("response_length/max"), "tokens; memory reserved (mean over ranks)",
      step.get("actor/perf/max_memory_reserved_gb"), "GB", file=sys.stderr)
sys.exit(0 if float(step.get("response_length/max", 0)) >= 24000 else 1)
PY
  then
    CAP=24k
  fi
  tlog "24K test: exit ${TEST_RC}; RL v2 trains at ${CAP} (log ${LOGS}/memtest24k.log)"
  run mkdir -p "${CK}/rl"
  dry || printf '{"cap": "%s", "memtest_exit": %d, "job": "%s", "time": "%s"}\n' \
           "${CAP}" "${TEST_RC}" "${SLURM_JOB_ID:-local}" "$(date -Is)" > "${CAP_FILE}"
  gpus_free
fi
RL_ARGS=(--config "${RL_CONFIG}")
case "${CAP}" in
  24k) RL_ARGS+=("${CAP_24K[@]}") ;;
  16k) ;;
  *) tdie "unknown cap '${CAP}' (DPT_RL_V2_CAP: 16k or 24k)" ;;
esac

# ---- 4. RL v2 ---------------------------------------------------------------------------------------------------
LEFT_S="$(time_left_s)"
if (( LEFT_S - EVAL_S < 3 * 3600 )); then
  tdie "only $((LEFT_S / 60)) min left: RL would get less than 3 h. Bank v2 is ready; the next job starts RL from it"
fi
tlog "${RUN_NAME} at ${CAP}; the final eval gets the last $((EVAL_S / 3600)) h"
RC=0
env DPT_SLURM_NODES="${NODES[*]}" DPT_CHAIN_LEFT=1 DPT_ALLOW_NO_JUDGE=1 DPT_JUDGE_WAIT_S=0 \
  DPT_RL_MARGIN_S=$((EVAL_S + 1200)) bash scripts/slurm/rl.slurm "${RL_ARGS[@]}" || RC=$?
tlog "RL ended (exit ${RC}); final eval next"

# ---- 5. the final eval ---------------------------------------------------------------------------------------------
gpus_free
env -u DPT_CANCEL_JOB DPT_SLURM_NODES="${NODES[*]}" DPT_FINAL_ROW="${FINAL_ROW}" DPT_FINAL_RUN="${RUN_DIR}" \
  bash scripts/slurm/final_eval_window.slurm
tlog "done: RL exit ${RC}, final eval finished"
exit "${RC}"
}
