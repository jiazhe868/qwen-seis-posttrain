#!/usr/bin/env bash
# STEP 04: stage-1 SFT (report §阶段一). Then the RL-entry gate.
#
#   cluster, 27B (run on EVERY node; ~22 h on 16 GPUs):
#     NNODES=2 NODE_RANK=<0|1> MASTER_ADDR=<node0> DPT_HW_PROFILE=cluster_2x8xa100_80g bash scripts/04_sft.sh
#   workstation smoke (0.8B, toy data from `01_data.sh --smoke`):
#     bash scripts/04_sft.sh --config sft/sft_smoke_0p8b.yaml --allow-missing-manifest
#
# After training (node 0): export the last checkpoint to HF, then evaluate it and apply the
# RL-entry gate (domain > base on all three, general Δ ≥ -1.5, pass@16 ≥ base).
source "$(dirname "$0")/common.sh"
CONFIG=sft/sft_27b.yaml; prev=""
for a in "$@"; do [[ "${prev}" == "--config" ]] && CONFIG="${a}"; prev="${a}"; done
"${PY}" -m dpt.launch.sft "$@"
[[ "${NODE_RANK:-0}" == "0" ]] || exit 0
read -r OUTDIR BASE_MODEL < <("${PY}" -c "import sys; from dpt.launch.common import compose_stage; \
c = compose_stage(sys.argv[1]); print(c.sft.output_dir, c.sft.base_model)" "${CONFIG}")
LAST="$(cat "${OUTDIR}/latest_checkpointed_iteration.txt")"
bash "${DPT_REPO}/scripts/export_hf.sh" "${OUTDIR}/global_step_${LAST}" "${OUTDIR}/hf_export" "${BASE_MODEL}"
log "SFT exported → ${OUTDIR}/hf_export. Next: evaluate it (07_final_eval.sh --only sft) and run the entry gate:"
log "  python -m dpt.eval.gates --summary eval_results/summary.json --candidate sft --entry"
