#!/usr/bin/env bash
# STEP 08: audits that must accompany any reported number (report §污染检查, §评测体系 第二层).
#
#   bash scripts/08_audit.sh [labeled_grader_samples.jsonl]
#
#   * contamination report (13-gram/MinHash, embedding+LLM, answer-level, teacher recitation,
#     canary scan) and base-model memorization probes (prefix completion, perturbation gap)
#   * grader validation: adversarial suite (must be 100%) + FP/FN rate on human-labeled outputs
#     (report: v1 2.8% / 4.1% → v3 0.3% / 1.2%); fails if FP > 0.5% or FN > 1.5%
#   * judge calibration: fixed sample set; score distribution must match the pinned reference
source "$(dirname "$0")/common.sh"
"${PY}" -m dpt.data.pipeline --stages probe,publish "$@" || warn "probe stage needs live base-model endpoints"
LABELED="${1:-}"
if [[ -n "${LABELED}" ]]; then
  "${PY}" -m dpt.rewards.validate --labeled "${LABELED}" --max-fp-share 0.005 --max-fn-share 0.015 \
    --json "${DPT_REPO}/eval_results/grader_validation.json"
else
  "${PY}" -m dpt.rewards.validate
fi
"${PY}" -m dpt.rewards.validate --calibrate || warn "judge calibration needs DPT_JUDGE_URL"
