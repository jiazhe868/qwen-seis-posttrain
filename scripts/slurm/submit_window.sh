#!/bin/bash
# Submit one window of RL v2 (2026-11-09, README §8.3). Run on a login node, from anywhere:
#
#   scripts/slurm/submit_window.sh data <reservation>   # window 1: data_window.slurm on 2 nodes
#   scripts/slurm/submit_window.sh rl <reservation>     # window 2: the judge (1 node) and RL v2 (8 nodes), then the
#                                                      #   final eval on the RL nodes once RL ends (afterany)
# Without a reservation (the normal gpu-a100 queue); options after the mode go to sbatch, e.g. -t 48:00:00:
#   scripts/slurm/submit_window.sh data-queue [sbatch options]   # window 1 as a plain 2-node job
#   scripts/slurm/submit_window.sh rl-queue                      # window 2 as ONE heterogeneous job: rl_fallback.slurm,
#     8 gpu-a100 nodes + 1 judge node (the cluster allows at most 8 gpu-a100 nodes per job). DPT_FB_TIME (default 24:00:00)
#     is both components' limit; DPT_JUDGE_PARTITION (default gpu-a100, or gpu-a100-small) the judge node's queue.
#
# A job in a reservation starts only if it would end before the reservation does. The limits are therefore counted
# back from the reservation's end, less 10 minutes:
#   - the judge runs to the end;
#   - RL ends DPT_EVAL_H hours (default 4) earlier, and its trainer saves 20 minutes before its own limit;
#   - the final eval gets those hours (October's in-job eval took about 2.5 h).
# `rl` and `rl-queue` refuse to submit until bank v2's check has passed: $DATA/v2/bank_check.json with no problems
# listed. Cancel the superseded queue jobs of 11-07 first; they write the same files (README §8.3).
# DPT_SLURM_DRY_RUN=1 prints the sbatch commands instead of running them.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/slurm/env.sh
source scripts/slurm/lib.sh
WHAT="${1:?data | rl | data-queue | rl-queue}"
ACCT="${DPT_ACCOUNT:-PROJ}"; EVAL_MIN=$(( ${DPT_EVAL_H:-4} * 60 ))
read -r DATA JUDGE_ID <<< "$(python - <<'PY'
from dpt.common.config import load_config
p = load_config().project
print(p.paths.data_root, p.judges.primary.hf_id)
PY
)"

DRY_ID=9000
submit() {   # submit <var> <sbatch args…>: queue one job; its id goes into <var>
  local var="$1" out; shift
  if dry; then run sbatch "$@"; DRY_ID=$((DRY_ID + 1)); printf -v "${var}" '%s' "${DRY_ID}"; return 0; fi
  out="$(sbatch --parsable "$@")" || tdie "sbatch failed: ${out}"
  printf -v "${var}" '%s' "$(job_id <<< "${out}")"
  [[ "${!var}" =~ ^[0-9]+$ ]] || tdie "no job id in the sbatch output: ${out}"
}
bank_ok() {  # bank v2 exists and its check lists no problems
  local check="${DATA}/v2/bank_check.json"
  [[ -f "${check}" ]] || tdie "no ${check}: window 1 (data) has not finished"
  python - "${check}" <<'PY' || tdie "bank v2 did not pass its check: ${check}"
import json, sys
problems = json.load(open(sys.argv[1])).get("problems", ["no problems field"])
for p in problems:
    print("bank check:", p, file=sys.stderr)
sys.exit(1 if problems else 0)
PY
}

case "${WHAT}" in
  data-queue)
    shift
    submit DATA_JOB -A "${ACCT}" "$@" scripts/slurm/data_window.slurm
    tlog "data window in the queue: job ${DATA_JOB} (logs \$WORK/dpt/logs/datawin_${DATA_JOB}/)"
    exit 0 ;;
  rl-queue)
    (( $# == 1 )) || tdie "rl-queue takes no sbatch options: set DPT_FB_TIME or DPT_JUDGE_PARTITION"
    bank_ok
    T="${DPT_FB_TIME:-24:00:00}"; JP="${DPT_JUDGE_PARTITION:-gpu-a100}"
    submit RLFB -A "${ACCT}" -p gpu-a100 -N 8 -t "${T}" : -A "${ACCT}" -p "${JP}" -N 1 -t "${T}" \
      scripts/slurm/rl_fallback.slurm
    tlog "RL window in the queue: job ${RLFB} (8 gpu-a100 nodes + 1 ${JP} judge node: judge, RL, final eval)"
    exit 0 ;;
  data|rl) ;;
  *) tdie "unknown window ${WHAT}: data | rl | data-queue | rl-queue" ;;
esac

RES="${2:?reservation name}"
INFO="$(scontrol show reservation "${RES}" -o 2>/dev/null)" && [[ "${INFO}" == *"ReservationName=${RES} "* ]] ||
  tdie "no reservation ${RES} (scontrol show reservation)"
field() { sed -n "s/.* $1=\([^ ]*\).*/\1/p" <<< " ${INFO}"; }
START="$(date -d "$(field StartTime)" +%s)"; END="$(date -d "$(field EndTime)" +%s)"; NODES="$(field NodeCnt)"
NOW="$(date +%s)"; BEGIN=$(( NOW > START ? NOW : START ))
LEFT=$(( (END - BEGIN) / 60 - 10 ))                  # minutes a job starting now (or at the start) may run
hm() { printf '%d:%02d:00' $(( $1 / 60 )) $(( $1 % 60 )); }
tlog "reservation ${RES}: ${NODES} nodes, $(date -d "@${START}" '+%m-%d %H:%M') to $(date -d "@${END}" '+%m-%d %H:%M'), $(hm "${LEFT}") usable"

if [[ "${WHAT}" == data ]]; then
  (( NODES >= 2 )) || tdie "window 1 needs 2 nodes, ${RES} has ${NODES}"
  (( LEFT >= 600 )) || tdie "only $(hm "${LEFT}") left in ${RES}; the data phase needs about 10 h or more"
  submit DATA_JOB -A "${ACCT}" --reservation="${RES}" -t "$(hm $(( LEFT < 1440 ? LEFT : 1440 )))" \
    scripts/slurm/data_window.slurm
  tlog "data window: job ${DATA_JOB} (logs \$WORK/dpt/logs/datawin_${DATA_JOB}/; result \$DATA/v2/bank_check.json)"
  exit 0
fi

(( NODES >= 9 )) || tdie "window 2 needs 9 nodes (judge 1 + RL 8), ${RES} has ${NODES}"
RL_MIN=$(( LEFT - EVAL_MIN ))
(( RL_MIN >= 360 )) || tdie "only $(hm "${LEFT}") left in ${RES}: RL would get $(hm "${RL_MIN}"), less than 6 h"
bank_ok
submit JUDGE -A "${ACCT}" --reservation="${RES}" -t "$(hm "${LEFT}")" \
  --export=ALL,DPT_SERVE_DP=1,DPT_SERVE_REPLICAS=3 scripts/slurm/serve.slurm judge "${JUDGE_ID}" 1
submit RL -A "${ACCT}" --reservation="${RES}" -t "$(hm "${RL_MIN}")" \
  --export=ALL,DPT_CHAIN_LEFT=1,DPT_JUDGE_WAIT_S=3600 scripts/slurm/rl.slurm --config rl/rl_9b_v2.yaml
submit EVAL -A "${ACCT}" --reservation="${RES}" -t "$(hm "${EVAL_MIN}")" --dependency="afterany:${RL}" \
  --export="ALL,DPT_CANCEL_JOB=${JUDGE}" scripts/slurm/final_eval_window.slurm
tlog "judge ${JUDGE} ($(hm "${LEFT}")), RL ${RL} ($(hm "${RL_MIN}")), final eval ${EVAL} after RL ($(hm "${EVAL_MIN}"))"
