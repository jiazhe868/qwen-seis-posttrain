#!/usr/bin/env bash
# STEP 01: build the RL problem bank and the SFT mix (report §数据, §处理流水线, §污染检查).
#
#   bash scripts/01_data.sh --smoke     # toy fixtures + deterministic fakes, ~1 s, writes data/smoke/
#   bash scripts/01_data.sh             # full run against live services (see below)
#
# Full run prerequisites (configs/data/pipeline.yaml → data_pipeline.backends):
#   * base model served for difficulty calibration (8 samples/problem at medium effort):
#       scripts/serve_vllm.sh Qwen/Qwen3.8-27B base 8000      (local box: TP4, ~1 day for 74K problems)
#   * two strong models of DIFFERENT families for 3-way answer verification
#   * an embedding server + an LLM paraphrase judge for semantic decontamination
#   * eval sets readable ONLY by the account running the decontam stage
# Output: data/<version>/{rl_main,rl_hard,rl_dev}.{jsonl,parquet} + manifests, sft/sft_train.jsonl,
#         funnel.json (91k → 82k → 79.5k → 74.1k → 60k in the report), reports/contamination_report.json
source "$(dirname "$0")/common.sh"
if [[ "${1:-}" == "--smoke" ]]; then
  shift
  SMOKE_DATA="$("${PY}" -c 'from dpt.common.config import load_config; print(load_config().project.paths.data_root)')/smoke"
  log "toy pipeline (offline fakes) → ${SMOKE_DATA}"
  "${PY}" -m dpt.data.pipeline --offline \
    --stages ingest,quality,dedup,decontam,verify,difficulty,publish,sft \
    data_pipeline.output_dir="${SMOKE_DATA}" \
    data_pipeline.publish.dev_copy_if_empty=true "$@"
else
  log "full data pipeline"
  "${PY}" -m dpt.data.pipeline "$@"
fi
log "grader self-test (adversarial suite must be 100%)"
"${PY}" -m dpt.rewards.validate
