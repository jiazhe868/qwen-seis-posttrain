#!/bin/bash
#SBATCH -J dpt-t3atomic
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 12:00:00
#SBATCH -o dpt-t3atomic-%j.out
# Rewrite the T3 report rubrics as atomic yes/no checks (dpt.data.gen.t3_atomic, 2026-11-07). The teacher writes
# the checks; Gemma (judge_v) checks that each criterion's checks cover it. CPU only: both models are serve.slurm
# endpoints. Resumable: a rerun keeps the rubrics already in the map.
#
#   sbatch -A <alloc> --dependency=after:<teacher job>:<judge_v job> scripts/slurm/t3_atomic.slurm
#
# DPT_T3_IN   problem rows to read (default $DATA/v1/stages/04_verify.jsonl: every verified T3 problem, dev included)
# DPT_T3_MAP  the map to write (default $DATA/v2/t3_atomic_map.jsonl; `apply` puts it into a stage or bank file)
# DPT_TEACHER_ENDPOINT (teacher), DPT_JUDGE_ENDPOINT (judge_v), DPT_ENDPOINT_WAIT_S (7200), DPT_T3_CONCURRENCY (32)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/slurm/env.sh
source scripts/slurm/lib.sh
export HF_HUB_OFFLINE=1                                  # the teacher's tokenizer comes from $HF_HOME

read -r DATA TEACHER_ID <<< "$(python - <<'PY'
from dpt.common.config import load_config
c = load_config().project
print(c.paths.data_root, c.model.teacher)
PY
)"
IN="${DPT_T3_IN:-${DATA}/v1/stages/04_verify.jsonl}"
MAP="${DPT_T3_MAP:-${DATA}/v2/t3_atomic_map.jsonl}"
TEACHER="${DPT_TEACHER_ENDPOINT:-teacher}"; JUDGE="${DPT_JUDGE_ENDPOINT:-judge_v}"
[[ -f "${IN}" ]] || tdie "no ${IN}"
mkdir -p "$(dirname "${MAP}")/cache"
for e in "${TEACHER}" "${JUDGE}"; do
  wait_endpoint "${e}" "${DPT_ENDPOINT_WAIT_S:-7200}" || tdie "the ${e} endpoint never answered"
done
tlog "t3_atomic: ${IN} → ${MAP} (teacher ${TEACHER}, coverage ${JUDGE})"
run python -m dpt.data.gen.t3_atomic convert --in "${IN}" --out "${MAP}" \
  --teacher-url "$(endpoint_url "${TEACHER}")/v1" --teacher-model "$(endpoint_model "${TEACHER}")" \
  --teacher-url-file "${DPT_ENDPOINTS}/${TEACHER}.url" --tokenizer "${TEACHER_ID}" \
  --judge-url "$(endpoint_url "${JUDGE}")/v1" --judge-model "$(endpoint_model "${JUDGE}")" \
  --judge-url-file "${DPT_ENDPOINTS}/${JUDGE}.url" \
  --cache-dir "$(dirname "${MAP}")/cache" --concurrency "${DPT_T3_CONCURRENCY:-32}"
tlog "t3_atomic: done ($(cat "${MAP}.summary.json" 2>/dev/null | tr -d '\n' | cut -c1-300))"
