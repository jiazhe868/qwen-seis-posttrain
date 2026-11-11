#!/bin/bash
#SBATCH -J dpt-rl
#SBATCH -p gpu-a100
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 48:00:00
#SBATCH -o dpt-rl-%j.out
# Stage-2 RL of the 9B on 8 gpu-a100 nodes (24 × A100-40GB), resubmitting itself until the run ends.
#
#   cd $WORK/nanogpt_seis_qwen && sbatch -A <allocation> scripts/slurm/rl.slurm [--config rl/rl_9b.yaml] \
#        [--ablation A03_clip_high_0p2] [project.seed=2] [other dotlist overrides]
#
# Before submitting: the SFT export exists, the judge is being served (serve.slurm judge ...), and
# scripts/slurm/check_sandbox.sh passed. ~3 SU per node-hour → 24 SU per hour for this job.
#
# One job = one ≤ 48 h slice of the run:
#   1. stop at once if the run is finished, or PAUSED by an alert (a human must look: docs/runbook.md)
#   2. queue the successor (afterany: it starts when this job ends, for whatever reason)
#   3. start Ray on all nodes, export the wall-clock deadline, launch dpt.launch.rl; verl resumes from
#      the latest checkpoint (resume_mode=auto) together with the curriculum/sampler/RNG/alert state
#   4. the trainer saves and stops before the deadline (WALLTIME.json); the successor continues.
#      Any other exit (crash, OOM, pause, done) cancels the successor, so a broken run cannot burn
#      SUs in a resubmission loop. DPT_CHAIN_LEFT (default 6 jobs = 12 days) caps the chain.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/slurm/env.sh
source scripts/slurm/lib.sh

CONFIG=rl/rl_9b.yaml; ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in --config) CONFIG="$2"; shift 2 ;; *) ARGS+=("$1"); shift ;; esac
done
describe() { python -m dpt.launch.rl --config "${CONFIG}" --describe "${ARGS[@]}"; }

STATE="$(describe)"; RUN_DIR="$(json_field "${STATE}" run_dir)"
tlog "run ${RUN_DIR}: step $(json_field "${STATE}" latest_step)/$(json_field "${STATE}" total_steps)"
[[ "$(json_field "${STATE}" done)" == true ]] && { tlog "run finished; nothing to do"; exit 0; }
[[ "$(json_field "${STATE}" paused)" == true ]] && tdie "run is PAUSED by an alert: read ${RUN_DIR}/PAUSED.json and docs/runbook.md, fix, delete PAUSED.json, resubmit"

# T3 rewards fail closed (score 0) without a judge: never let a missing judge silently drop T3.
# DPT_JUDGE_WAIT_S (default 0): wait that long for the judge to answer first, e.g. while a judge job submitted with
# --dependency=after:<this job> starts its server (2026-11-07). The nodes idle meanwhile (24 SU/h): keep it short.
[[ "${DPT_JUDGE_WAIT_S:-0}" -le 0 ]] || wait_endpoint judge "${DPT_JUDGE_WAIT_S}" || true
if endpoint_up judge; then
  export DPT_JUDGE_URL="$(endpoint_url judge)/v1" DPT_JUDGE_MODEL="$(endpoint_model judge)"
  tlog "judge: ${DPT_JUDGE_MODEL} at ${DPT_JUDGE_URL}"
elif [[ "${DPT_ALLOW_NO_JUDGE:-0}" != 1 ]]; then
  tdie "no judge endpoint (${DPT_ENDPOINTS}/judge.url): sbatch scripts/slurm/serve.slurm judge <model> first, or DPT_ALLOW_NO_JUDGE=1"
fi

CHAIN_LEFT="${DPT_CHAIN_LEFT:-6}"
NEXT=""
if [[ "${CHAIN_LEFT}" -gt 1 ]]; then
  NEXT="$(run sbatch --parsable --dependency="afterany:${SLURM_JOB_ID}" -A "${SLURM_JOB_ACCOUNT}" \
            --export="ALL,DPT_CHAIN_LEFT=$((CHAIN_LEFT - 1))" "${DPT_REPO}/scripts/slurm/rl.slurm" \
            --config "${CONFIG}" "${ARGS[@]}" | job_id)"
  tlog "successor queued: ${NEXT} (chain budget left after it: $((CHAIN_LEFT - 2)))"
fi
cancel_next() { [[ -n "${NEXT}" ]] && run scancel "${NEXT}" || true; }

rm -f "${RUN_DIR}/WALLTIME.json"
mkdir -p "${RUN_DIR}"
ray_up 3
# DPT_RL_MARGIN_S (default 1200): the trainer saves and stops this long before the job's limit. rl_fallback.slurm
# adds the hours of the final eval, which runs in the same job after RL.
MARGIN_S="${DPT_RL_MARGIN_S:-1200}"
if dry; then export DPT_DEADLINE=$(( $(date +%s) + 172800 )); else
  export DPT_DEADLINE="$(python -m dpt.launch.walltime --slurm-deadline --margin-s "${MARGIN_S}")"
  export DPT_DEADLINE_MARGIN_S="${MARGIN_S}"   # the trainer re-reads the limit from Slurm (an extension counts)
fi
tlog "training must stop by $(date -d "@${DPT_DEADLINE}" '+%F %T') (${MARGIN_S} s before the job's limit)"
RC=0
run python -m dpt.launch.rl --config "${CONFIG}" "${ARGS[@]}" || RC=$?
ray_down

STATE="$(describe)"
if [[ "$(json_field "${STATE}" done)" == true ]]; then
  cancel_next; tlog "run finished at step $(json_field "${STATE}" latest_step)"
elif [[ "$(json_field "${STATE}" paused)" == true ]]; then
  cancel_next; tdie "auto-paused by an alert: ${RUN_DIR}/PAUSED.json (docs/runbook.md)"
elif [[ "$(json_field "${STATE}" walltime)" == true ]]; then
  tlog "stopped for the wall-clock limit at step $(json_field "${STATE}" latest_step); ${NEXT:-no successor (chain budget used up)} continues"
else
  cancel_next; tdie "training exited with code ${RC} without a clean stop: successor cancelled; see this log and ${RUN_DIR}"
fi
