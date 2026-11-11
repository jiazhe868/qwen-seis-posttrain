#!/usr/bin/env bash
# STEP 03: baseline evaluation of the untouched base model (report §评测协议, §harness 规则).
#
#   bash scripts/03_baseline_eval.sh
#   DPT_EVAL_OVERRIDES="eval_suite.sandbox_endpoint=http://h:8080 ..." adds harness settings to every run
#   the cluster: sbatch -A <alloc> scripts/slurm/eval.slurm base   (serves the model itself, sets DPT_EVAL_ENDPOINT)
#
# Runs locally (4×A100-40GB, vLLM TP4) or on one cluster node. Produces (base_xhigh only when the
# model line has an xhigh effort; the Qwen3.5-9B line has one thinking mode, so medium only):
#   eval_results/base_medium/         all sets, medium effort, 32K cap
#   eval_results/base_xhigh/          all sets, xhigh effort, 32K AND 64K caps (higher score reported;
#                                     base@xhigh truncates ~11% at 32K)
#   eval_results/noise/base_medium_s{0..9}  D-Num ×10 seeds → noise floor (report: mean 0.9, max 2.1)
#   eval_results/base_medium_k64/     D-Num with 64 samples/item → pass@k curve (pass@64 floor for RL)
# Every model, including the base, is re-run in the same harness; no public numbers are copied.
source "$(dirname "$0")/common.sh"
BASE="$(cfg project.model.base)"; R="$(cfg project.paths.eval_root)"
EP="${DPT_EVAL_ENDPOINT:-http://localhost:8000}"   # harness appends /v1/...
SETS=d_num,d_code,d_report,d_fresh,general,refusal
read -r -a OV <<< "${DPT_EVAL_OVERRIDES:-}"
# Efforts come from the model line: Qwen3.8 → medium + xhigh; Qwen3.5-9B → medium only (its template
# has a single thinking mode, so there is no base@xhigh row and no 64K re-score).
read -r -a EFFORTS <<< "$("${PY}" -c "from dpt.common.config import load_config; \
print(' '.join(load_config().project.thinking.eval_efforts))")"
EXTRA="$(cfg project.thinking.eval_base_xhigh_extra_max_response)"
[[ " ${EFFORTS[*]} " == *" xhigh "* && "${EXTRA}" != "None" ]] && MAXLEN=$((EXTRA + 8192)) || MAXLEN=$((32768 + 8192))

if [[ -z "${DPT_EVAL_ENDPOINT:-}" ]]; then   # scripts/slurm/eval.slurm starts its own server
  log "serving base with a $((MAXLEN / 1024))K context (longest response + 8K prompt)"
  "${DPT_REPO}/scripts/serve_vllm.sh" "${BASE}" base 8000 "${MAXLEN}" & SERVER=$!
  trap 'kill ${SERVER} 2>/dev/null || true' EXIT
  wait_for_server http://localhost:8000
fi

# D-Report is a win rate against base(medium); for the base itself it is 50% by construction.
"${PY}" -m dpt.eval.harness --model-name base_medium --served-model base --endpoint "${EP}" \
  --sets "${SETS}" --effort medium --reference-run "${R}/base_medium" --out "${R}/base_medium" "${OV[@]}"
if [[ " ${EFFORTS[*]} " == *" xhigh "* ]]; then
  "${PY}" -m dpt.eval.harness --model-name base_xhigh --served-model base --endpoint "${EP}" \
    --sets "${SETS}" --effort xhigh --extra-max-tokens "${EXTRA}" --reference-run "${R}/base_medium" \
    --out "${R}/base_xhigh" "${OV[@]}"
fi
for s in $(seq 0 9); do
  "${PY}" -m dpt.eval.harness --model-name base_medium --served-model base --endpoint "${EP}" \
    --sets d_num --effort medium --seed "${s}" --out "${R}/noise/base_medium_s${s}" "${OV[@]}"
done
"${PY}" -m dpt.eval.noise_floor --set d_num --out "${R}/noise/noise_floor.json" "${R}"/noise/base_medium_s*
"${PY}" -m dpt.eval.harness --model-name base_medium --served-model base --endpoint "${EP}" \
  --sets d_num --effort medium --samples-override 64 --out "${R}/base_medium_k64" "${OV[@]}"
log "baseline done → ${R}"
