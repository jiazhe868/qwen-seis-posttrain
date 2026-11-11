#!/usr/bin/env bash
# STEP 07: final evaluation, report table and ship gates (report §最终设想结果, §目标与成功标准).
#
#   bash scripts/07_final_eval.sh               # sft, rl_seed1, rl_seed2 (base rows from step 03)
#   bash scripts/07_final_eval.sh --only sft    # e.g. right after SFT, for the RL-entry gate
#   bash scripts/07_final_eval.sh --report-only # rebuild report + gates from existing results
#                                                 (after scripts/slurm/second_judge.sh)
#
# Checkpoints are chosen on the DEV curve during RL, never on these test sets. This script only
# measures the chosen ones. D-Report is judged pairwise vs base (both orders); 30% of items are
# exported for expert blind review (eval_results/<row>/d_report/expert_review.jsonl). Import the
# verdicts before reading the final win rate.
#
# RL rows: pick the checkpoint on the dev curve and export it first, e.g.
#   STEP=$(python -m dpt.eval.select_checkpoint --run checkpoints/rl/rl27b_seed1 | jq .selected_step)
#   bash scripts/export_hf.sh checkpoints/rl/rl27b_seed1/global_step_${STEP} checkpoints/rl/rl27b_seed1/hf_export
source "$(dirname "$0")/common.sh"
R="$(cfg project.paths.eval_root)"; CK="$(cfg project.paths.ckpt_root)"
# harness appends /v1/... With DPT_EVAL_ENDPOINT set, the caller serves the --only row under its name
# (scripts/slurm/campaign.sh: one replica per GPU on several nodes behind scripts/rr_proxy.py).
EP="${DPT_EVAL_ENDPOINT:-http://localhost:8000}"
TAG="$(cfg project.model.tag)"                 # 27b | 9b (model line): run names sft<tag>_seed1, rl<tag>_seed1
SETS=d_num,d_code,d_report,d_fresh,general,refusal
# Extra harness settings, e.g. the sandbox and judge endpoints started by scripts/slurm/eval.slurm.
read -r -a OV <<< "${DPT_EVAL_OVERRIDES:-}"
declare -A ROWS=(
  [sft]="${CK}/sft/sft${TAG}_seed1/hf_export"
  [rl_seed1]="${CK}/rl/rl${TAG}_seed1/hf_export"
  [rl_seed2]="${CK}/rl/rl${TAG}_seed2/hf_export"
  [rl_v2_seed1]="${CK}/rl/rl${TAG}_v2_seed1/hf_export"     # RL v2 (2026-11-07, README §8.3)
)
ONLY="${2:-}"; [[ "${1:-}" == "--only" ]] || ONLY=""
[[ "${1:-}" == "--report-only" ]] && ONLY="-"     # no row matches: skip straight to the report
for row in sft rl_seed1 rl_seed2 rl_v2_seed1; do
  [[ -z "${ONLY}" || "${ONLY}" == "${row}" ]] || continue
  [[ -d "${ROWS[$row]}" ]] || { warn "skip ${row}: ${ROWS[$row]} missing"; continue; }
  SERVER=""
  if [[ -z "${DPT_EVAL_ENDPOINT:-}" ]]; then
    "${DPT_REPO}/scripts/serve_vllm.sh" "${ROWS[$row]}" "${row}" 8000 40960 & SERVER=$!   # 32K response + 8K prompt
    wait_for_server http://localhost:8000
  fi
  "${PY}" -m dpt.eval.harness --model-name "${row}" --served-model "${row}" --endpoint "${EP}" \
    --sets "${SETS}" --effort medium --reference-run "${R}/base_medium" --out "${R}/${row}" "${OV[@]}"
  "${PY}" -m dpt.eval.harness --model-name "${row}" --served-model "${row}" --endpoint "${EP}" \
    --sets d_num --effort medium --samples-override 64 --out "${R}/${row}_k64" "${OV[@]}"   # pass@64 ≥ base
  [[ -z "${SERVER}" ]] || { kill "${SERVER}"; wait "${SERVER}" 2>/dev/null || true; }
done
ARGS=(--row base_medium="${R}/base_medium")
[[ -f "${R}/base_xhigh/results.json" ]] && ARGS+=(--row base_xhigh="${R}/base_xhigh")   # 27B line only
for row in sft rl_seed1 rl_seed2 rl_v2_seed1; do [[ -f "${R}/${row}/results.json" ]] && ARGS+=(--row "${row}=${R}/${row}"); done
"${PY}" -m dpt.eval.report "${ARGS[@]}" --summary-out "${R}/summary.json" --markdown-out "${R}/report.md"
# DPT_FINAL_CANDIDATE: the row the ship gates judge (default rl_seed1; RL v2: rl_v2_seed1).
CAND="${DPT_FINAL_CANDIDATE:-rl_seed1}"
if [[ -f "${R}/${CAND}/results.json" ]]; then
  "${PY}" -m dpt.eval.gates --summary "${R}/summary.json" --candidate "${CAND}" --seed2 rl_seed2
fi
log "report → ${R}/report.md"
