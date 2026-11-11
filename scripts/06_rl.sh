#!/usr/bin/env bash
# STEP 06: stage-2 RL on 2 × 8 A100-80G, colocated (report §阶段二, §工程设计). ~420 steps ≈ 4-5 days.
#
#   (both nodes up: scripts/ray_cluster.sh head / worker <ip>)
#   DPT_HW_PROFILE=cluster_2x8xa100_80g bash scripts/06_rl.sh                       # main line, seed 1
#   DPT_HW_PROFILE=cluster_2x8xa100_80g bash scripts/06_rl.sh project.seed=2        # reproducibility seed (starts from the same SFT checkpoint)
#   DPT_HW_PROFILE=cluster_2x8xa100_80g bash scripts/06_rl.sh --ablation A08_lr_2e-6
#   DPT_HW_PROFILE=cluster_2x8xa100_80g bash scripts/06_rl.sh --config rl/rl_9b_proxy.yaml --ablation A03_clip_high_0p2
#
# The launcher refuses to start if a pre-flight check fails (see dpt/launch/rl.py). If the alert
# engine auto-pauses the run it writes <run_dir>/PAUSED.json. Follow docs/runbook.md, fix, then
# rerun the same command: resume_mode=auto continues from the last checkpoint (or roll back further
# by deleting newer global_step_* dirs and editing latest_checkpointed_iteration.txt, see runbook).
source "$(dirname "$0")/common.sh"
require_profile cluster_2x8xa100_80g
ray status >/dev/null 2>&1 || die "no Ray cluster; start it with scripts/ray_cluster.sh on both nodes"
[[ -n "${DPT_JUDGE_URL:-}" ]] || warn "DPT_JUDGE_URL not set: T3 rewards will fail closed (score 0)"
[[ -n "${DPT_SANDBOX_URL:-}" ]] || warn "DPT_SANDBOX_URL not set: T2 uses the local subprocess sandbox"
CONFIG=rl/rl_27b.yaml; ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
exec "${PY}" -m dpt.launch.rl --config "${CONFIG}" "${ARGS[@]}"
