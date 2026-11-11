#!/usr/bin/env bash
# Serve a checkpoint with vLLM for eval / calibration / self-distillation (OpenAI-compatible API).
#
#   scripts/serve_vllm.sh <model_path_or_id> <served_name> [port] [max_model_len] [tp]
#
# Fixed, non-negotiable flags (each one is an incident or a harness rule):
#   --mamba-ssm-cache-dtype float32   GDN recurrent state in FP32 (Incident 1)
#   --no-enable-prefix-caching        harness determinism; stale-cache class of bugs (Incident 3)
#   --language-model-only             vision tower is out of scope; frees its memory
#   --gdn-prefill-backend triton      GDN prefill on the FLA Triton kernels (the ones parity checks validate,
#                                     and A100's only option); on H100 vLLM would pick a FlashInfer kernel
#                                     that is JIT-compiled at start-up and needs nvcc
# The eval harness renders prompts itself (dpt.chat.render, explicit effort) and calls
# /v1/completions, so no server-side chat template or reasoning parser is involved.
source "$(dirname "$0")/common.sh"
MODEL="${1:?model path or HF id}"; NAME="${2:?served model name}"
PORT="${3:-8000}"; MAXLEN="${4:-32768}"
# 27B: TP4 on 40GB cards, TP2 on 80GB. The 9B line sets DPT_SERVE_TP=1 (scripts/slurm/env.sh).
TP="${5:-${DPT_SERVE_TP:-$([[ ${DPT_HW_PROFILE} == local_* ]] && echo 4 || echo 2)}}"
# Data parallelism inside one server (vLLM --data-parallel-size): e.g. the 9B on a cluster node,
# TP1 × DP3, behind ONE endpoint, so clients need not balance across replicas.
DP="${DPT_VLLM_DP:-1}"
GDN_ARGS=()   # only for the Qwen3.5/3.8 hybrid (Gated DeltaNet) models
[[ "${MODEL}" =~ Qwen3\.[58] || "${MODEL}" =~ qwen3[58] || -n "${DPT_GDN_MODEL:-}" ]] && GDN_ARGS=(--gdn-prefill-backend triton)
SPEC_ARGS=()
if [[ "${SPEC:-0}" == "1" ]]; then   # MTP speculative decoding (distribution-preserving; faster eval)
  SPEC_ARGS=(--speculative-config '{"method": "mtp", "num_speculative_tokens": 1}')
fi
# Use the workstation vLLM env (CUDA forward compat) when vllm is not on PATH.
RUN=(); if ! command -v vllm >/dev/null 2>&1 && [[ -n "$(env_python "${DPT_VLLM_ENV:-dpt-vllm}")" ]]; then
  RUN=(in_env "${DPT_VLLM_ENV:-dpt-vllm}")
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then export CUDA_VISIBLE_DEVICES="$(pick_gpus "$((TP * DP))")"; fi
log "serving ${MODEL} as ${NAME} on :${PORT} (TP${TP} × DP${DP} on GPUs ${CUDA_VISIBLE_DEVICES}, max_model_len ${MAXLEN})"
"${RUN[@]}" vllm serve "${MODEL}" \
  --served-model-name "${NAME}" --port "${PORT}" \
  --tensor-parallel-size "${TP}" --data-parallel-size "${DP}" --max-model-len "${MAXLEN}" \
  --gpu-memory-utilization "${GPU_MEM:-0.88}" \
  --mamba-ssm-cache-dtype float32 \
  --no-enable-prefix-caching \
  --language-model-only \
  "${GDN_ARGS[@]}" \
  --seed 0 "${SPEC_ARGS[@]}"
