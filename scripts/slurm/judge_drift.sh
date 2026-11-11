#!/usr/bin/env bash
# Judge drift during RL: GPT-6 Luna re-scores the T3 dev responses that the Gemma reward judge scored at each
# validation step (every 20 steps), and alerts when Gemma's scores pull away from Luna's (dpt.eval.judge_drift).
#
#   bash scripts/slurm/judge_drift.sh [run_name]          # default rl9b_seed1; exit status 3 = alert
#
# Run it on a login node (only HTTP calls) whenever you look at the RL metrics, at least once a day; each run
# scores only the validation steps that are new. The trainer never reads its output: an alert is for you
# (docs/runbook.md §3, treat it like proxy_gold). Key: ~/.config/dpt/openai_key (env.sh).
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "${DPT_REPO}/scripts/common.sh"
RUN_NAME="${1:-rl9b_seed1}"
[[ -n "${OPENAI_API_KEY:-}" ]] || die "no OPENAI_API_KEY (put it in ~/.config/dpt/openai_key, see env.sh)"
RUN="$(cfg project.paths.ckpt_root)/rl/${RUN_NAME}"
[[ -d "${RUN}/val_generations" ]] || die "no ${RUN}/val_generations yet (the first dump is written at step 0)"
"${PY}" -m dpt.eval.judge_drift --run "${RUN}" --dev "${DPT_RL_DEV:-$(cfg project.paths.data_root)/v1/rl_dev.jsonl}"
