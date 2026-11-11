#!/usr/bin/env bash
# STEP 02: numerical parity on THIS hardware before any training (report §为什么这样选: "A100 本身是风险").
#
#   bash scripts/02_parity.sh                 # workstation: smoke model; cluster: the 27B
#   bash scripts/02_parity.sh Qwen/Qwen3.8-27B # 27B on the workstation: TP4, or PP3 on 3 GPUs when one is busy
#   DPT_PARITY_LAYOUT="1 3 0.65" bash scripts/02_parity.sh Qwen/Qwen3.8-27B   # force TP PP util
#
# 1. Gated DeltaNet Triton kernels vs an FP64 reference on SM80, 1K–32K tokens: chunk kernel,
#    decode kernel, and decode with a BF16 state cache (the Incident 1 mechanism).
#      → env `dpt-gpu` (torch 2.11 CUDA 12.6 + flash-linear-attention)
# 2. vLLM vs a reference forward pass, per-token logprobs on NATURALLY long responses (xhigh effort,
#    up to 16K tokens), FP32 state → must PASS (mean |log-ppl diff| < 2e-3, no growth with position).
#    Scored twice: FLA chunk kernel (what Megatron trains with) and PyTorch FP32 delta rule.
# 3. Same with a BF16 recurrent state → documents why FP32 is mandatory.
#      → env `dpt-vllm` (vLLM 0.24 CUDA 12.9 + cuda-compat forward compatibility)
# 4. Megatron train-mode forward vs eval-mode forward on verl's model inputs (Qwen3.5-0.8B, same
#    attention shape as the 27B) → must PASS. PPO ratios compare exactly these two passes. On A100
#    they differed by 1-2 nats/token until dpt.verl_ext.patches fixed verl's inverted attention mask;
#    the report also says whether the upstream bug is still present.
#      → env `dpt-train` (only if it exists; bash scripts/00_setup_env.sh train-local)
# vLLM vs Megatron (eval-mode forward inside verl) is checked in step 05 (first smoke steps).
#
# Envs: bash scripts/00_setup_env.sh gpu-local && bash scripts/00_setup_env.sh vllm-local
# (override with DPT_GPU_ENV / DPT_VLLM_ENV / DPT_TRAIN_ENV). GPUs: the freest ones are picked automatically unless
# CUDA_VISIBLE_DEVICES is set. The workstation is shared, so these checks are sized to coexist.
source "$(dirname "$0")/common.sh"
GPU_ENV="${DPT_GPU_ENV:-dpt-gpu}"; VLLM_ENV="${DPT_VLLM_ENV:-dpt-vllm}"
[[ -n "$(env_python "${GPU_ENV}")" ]]  || die "conda env ${GPU_ENV} missing: bash scripts/00_setup_env.sh gpu-local"
[[ -n "$(env_python "${VLLM_ENV}")" ]] || die "conda env ${VLLM_ENV} missing: bash scripts/00_setup_env.sh vllm-local"
for e in "${GPU_ENV}" "${VLLM_ENV}"; do
  in_env "${e}" python -W ignore -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 3)" \
    || die "env ${e}: activation failed or torch cannot see a GPU (see the error above; in that env run: python -m dpt.launch.check_env)"
done

OUT="${DPT_REPO}/parity/$(date +%Y%m%d)"; mkdir -p "${OUT}"
MODEL="${1:-$([[ ${DPT_HW_PROFILE} == local_* ]] && cfg project.model.smoke || cfg project.model.base)}"

# ---- layout: fit the model into the GPUs that are actually free (the workstation is shared) ----------
# 27B BF16 = 54 GB of weights. vLLM's gpu_memory_utilization is a fraction of TOTAL memory and must be
# free at start-up, so the headroom left for other users' jobs is (free − util × total).
TP=1; PP=1; GPU_MEM=0.45; REF_CAP=""
if [[ "${MODEL}" == *27B* ]]; then
  if [[ -n "${DPT_PARITY_LAYOUT:-}" ]]; then read -r TP PP GPU_MEM <<< "${DPT_PARITY_LAYOUT}"
  elif [[ "$(count_free 26000)" -ge 4 ]]; then TP=4; PP=1; GPU_MEM=0.55   # ≈ 13.5 GB weights per GPU
  elif [[ "$(count_free 29000)" -ge 3 ]]; then TP=1; PP=3; GPU_MEM=0.65   # TP3 impossible (4 KV heads): pipeline
  elif [[ "$(count_free 37000)" -ge 2 ]]; then TP=2; PP=1; GPU_MEM=0.85
  else die "not enough free GPU memory for the 27B (need 4×26 GB, 3×29 GB or 2×37 GB free); try again later"
  fi
  REF_CAP=24
elif [[ "${MODEL}" == *9B* ]]; then
  GPU_MEM=0.85                                   # 19.3 GB of weights on one 40 GB GPU (the 9B line)
fi
NGPU=$((TP * PP))
GPUS="${CUDA_VISIBLE_DEVICES:-$(free_gpus "${NGPU}" 1)}"
NEED_MIB=$(awk -v u="${GPU_MEM}" 'BEGIN { printf "%d", u * 40960 + 512 }')
FREE="$(min_free_mib "${GPUS}")"
[[ "${FREE:-0}" -ge "${NEED_MIB}" ]] || die "GPUs ${GPUS} have only ${FREE} MiB free (need ${NEED_MIB}); other jobs are using them"
log "model ${MODEL} | TP${TP} × PP${PP} on GPUs ${GPUS} | vLLM util ${GPU_MEM} (min free ${FREE} MiB)"
export CUDA_VISIBLE_DEVICES="${GPUS%%,*}"   # kernel check: one GPU

log "1/4 GDN kernel parity (env ${GPU_ENV})"
in_env "${GPU_ENV}" python -m dpt.parity.deltanet_kernel_check --lengths 1024 4096 16384 32768 --heads 8 \
  --out "${OUT}/deltanet_kernel.json"

export CUDA_VISIBLE_DEVICES="${GPUS}"
# Natural long responses (the model line's longest-thinking effort, no forced length): forced
# continuations degenerate into loops. Qwen3.8: xhigh. Qwen3.5 (9B line): its only thinking mode,
# which the renderer calls "medium" (the template has no effort control).
EFFORT="${DPT_PARITY_EFFORT:-$("${PY}" -c "from dpt.common.config import load_config; \
print(list(load_config().project.thinking.eval_efforts)[-1])")}"
COMMON=(--model "${MODEL}" --tp "${TP}" --pp "${PP}" --n 8 --effort "${EFFORT}" --max-tokens 16384
        --max-model-len 17408 --gpu-mem "${GPU_MEM}" --ref-dtype bfloat16)
[[ -n "${REF_CAP}" ]] && COMMON+=(--ref-max-mem-gib "${REF_CAP}")
log "2/4 vLLM vs reference logprobs, FP32 recurrent state (env ${VLLM_ENV})"
RC=0
in_env "${VLLM_ENV}" python -m dpt.parity.logprob_parity "${COMMON[@]}" --ssm-dtype float32 --out "${OUT}/logprob_fp32_state" \
  || { RC=1; warn "FP32-state parity FAILED; see ${OUT}/logprob_fp32_state/parity.json and docs/runbook.md"; }
log "    same generations, second reference: PyTorch FP32 delta rule (separates kernel vs engine effects)"
in_env "${VLLM_ENV}" python -m dpt.parity.logprob_parity "${COMMON[@]}" --ssm-dtype float32 --ref-gdn torch \
  --phase score --out "${OUT}/logprob_fp32_state" \
  && in_env "${VLLM_ENV}" python -m dpt.parity.logprob_parity "${COMMON[@]}" --ssm-dtype float32 --ref-gdn torch \
       --phase compare --out "${OUT}/logprob_fp32_state" || warn "PyTorch-reference comparison did not pass"
log "3/4 same with a BF16 recurrent state (expected: FAIL, ~10x larger mismatch)"
in_env "${VLLM_ENV}" python -m dpt.parity.logprob_parity "${COMMON[@]}" --ssm-dtype bfloat16 --out "${OUT}/logprob_bf16_state" \
  || warn "BF16 state failed parity, as expected: this is why the FP32 state is mandatory"
TRAIN_ENV="${DPT_TRAIN_ENV:-dpt-train}"
if [[ -n "$(env_python "${TRAIN_ENV}")" ]]; then
  log "4/4 Megatron train-mode vs eval-mode forward on verl's inputs (env ${TRAIN_ENV}, 1 GPU)"
  CUDA_VISIBLE_DEVICES="$(free_gpus 1 12000)" in_env "${TRAIN_ENV}" python -m dpt.parity.train_forward_parity \
      --model "${DPT_TRAIN_PARITY_MODEL:-Qwen/Qwen3.5-0.8B}" --tokens 2048 --out "${OUT}/train_forward.json" \
    || { RC=1; warn "train-forward parity FAILED: PPO ratios would be noise; see ${OUT}/train_forward.json and dpt.verl_ext.patches"; }
else
  warn "4/4 skipped: env ${TRAIN_ENV} missing (bash scripts/00_setup_env.sh train-local)"
fi
log "parity results in ${OUT}"
exit "${RC}"
