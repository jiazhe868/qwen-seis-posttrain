#!/bin/bash
#SBATCH -J dpt-eval-sets
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 04:00:00
#SBATCH -o dpt-eval-sets-%j.out
# Build D-Fresh and D-Report from GPT-6 Luna's drafts with the automatic checks (dpt.eval.autocheck): the
# drafted solver code runs in the sandbox container, and the Gemma judge endpoint (serve.slurm judge ...)
# solves every D-Fresh problem 4 times, adjudicates the disagreements, and checks every D-Report rubric in 3
# answer-then-check votes, thinking up to ~28K tokens per reply. CPU only: the GPUs are the judge's. It uses
# the `normal` queue: with that much thinking it can run past the 2 h limit of `development`.
#
#   sbatch -A <alloc> scripts/slurm/eval_sets.slurm            # both sets (est. 1–2 h, ~9M judge tokens)
#   sbatch -A <alloc> scripts/slurm/eval_sets.slurm fresh      # or one: fresh | report
#
# Drafts: $DPT_DRAFTS (default $WORK/dpt/drafts) holds fresh_drafts.jsonl and d_report_drafts.csv, written by
# `python -m dpt.eval.drafts fresh|report` on a login node. Output: d_fresh.jsonl / d_report.jsonl with
# manifests in the secure eval dir, and d_fresh_check.csv / d_report_check.csv next to the drafts (one row per
# draft: status, reason, vote counts, adjudication). An existing set is never overwritten (it may be frozen): delete it first, or
# DPT_EVAL_SETS_FORCE=1 before Oct 20.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/slurm/env.sh
source scripts/slurm/lib.sh
WHICH="${1:-both}"
SECURE="$(python -c 'from dpt.common.config import load_config; print(load_config().project.paths.eval_secure_root)')"
DRAFTS="${DPT_DRAFTS:-${WORK}/dpt/drafts}"

case "${WHICH}" in
  fresh)  JOBS=("fresh fresh_drafts.jsonl d_fresh") ;;
  report) JOBS=("report d_report_drafts.csv d_report") ;;
  both)   JOBS=("fresh fresh_drafts.jsonl d_fresh" "report d_report_drafts.csv d_report") ;;
  *) tdie "usage: eval_sets.slurm [fresh | report | both]" ;;
esac
for job in "${JOBS[@]}"; do
  read -r cmd drafts set <<< "${job}"
  [[ -f "${DRAFTS}/${drafts}" ]] || tdie "no ${DRAFTS}/${drafts}: run \`python -m dpt.eval.drafts ${cmd} ...\` on a login node"
  if [[ -f "${SECURE}/${set}.jsonl" && "${DPT_EVAL_SETS_FORCE:-0}" != 1 ]]; then
    tdie "${SECURE}/${set}.jsonl exists (frozen?): delete it, or DPT_EVAL_SETS_FORCE=1 before the freeze"
  fi
done
# Queued behind the judge's serve job (--dependency=after:<job>)? DPT_ENDPOINT_WAIT_S lets it wait for the load.
wait_endpoint judge || tdie "no judge endpoint: sbatch -A <alloc> scripts/slurm/serve.slurm judge RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic 1 (-p gpu-a100-small)"
OV=("eval_suite.judge.endpoint=$(endpoint_url judge)" "eval_suite.judge.model=$(endpoint_model judge)")
mkdir -p "${SECURE}"
for job in "${JOBS[@]}"; do
  read -r cmd drafts set <<< "${job}"
  tlog "autocheck ${cmd}: ${DRAFTS}/${drafts} → ${SECURE}/${set}.jsonl"
  run python -m dpt.eval.autocheck "${cmd}" --drafts "${DRAFTS}/${drafts}" --out "${SECURE}/${set}.jsonl" \
    --log "${DRAFTS}/${set}_check.csv" "${OV[@]}"
done
