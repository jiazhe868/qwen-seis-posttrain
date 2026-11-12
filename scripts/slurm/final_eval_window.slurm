#!/bin/bash
#SBATCH -J dpt-finaleval
#SBATCH -p gpu-a100
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 12:00:00
#SBATCH -o dpt-finaleval-%j.out
# Final evaluation of RL v2 (2026-11-09, README §8.3) on the reservation's 8 RL nodes, as a job that starts
# when the RL job ends. It is campaign 3532660's in-job final eval ($WORK/dpt/logs/final_eval_injob.sh) as a job
# script for row rl_v2_seed1:
#   select   pick the RL checkpoint on the dev curve (dpt.eval.select_checkpoint; ties go to the later step)
#   export   that checkpoint → HF at <ckpt>/rl/rl9b_v2_seed1/hf_export
#   main     every set at medium; D-Report judged pairwise against base_medium by Gemma
#   hard     D-Num-hard (the frozen 160 items), as the base, SFT and run 1 were scored
#   k64      D-Num with 64 samples (pass@k), only if listed in DPT_FINAL_PHASES
#   budgets  the answers whose thinking did not end within 32K, continued to 64K by the same model (D-Num-hard,
#            D-Fresh; SFT's too if missing), then the three-budget table with paired intervals against SFT
#            ($R/budgets_rl_v2_seed1.json, dpt.eval.budget compare). RL v2's success test is its 32K column.
#   report   scripts/07_final_eval.sh --report-only with rl_v2_seed1 as the gate candidate
# Layout as in October:
#   - this node: the T2 sandbox (:8080) and Gemma ×3 (:8101-8103, proxy :8100);
#   - the other 7 nodes: the model ×21 (proxy :8000).
# Run GPT-6 Luna on D-Report from a login node afterwards (scripts/slurm/second_judge.sh), then --report-only again.
#
#   sbatch -A PROJ --reservation=<name> --dependency=afterany:<RL job> scripts/slurm/final_eval_window.slurm
#
# DPT_FINAL_ROW (default rl_v2_seed1), DPT_FINAL_RUN (its RL run directory) and DPT_FINAL_EXPORT (default
# <run>/hf_export) name another model: job 2 evaluates SFT v2 (phases main,hard,budgets on its export) and RL v3.
# DPT_FINAL_PHASES (default select,export,main,hard,budgets,report); DPT_FINAL_STEP=<n> skips the dev selection;
# DPT_CANCEL_JOB=<id> cancels that job first (the RL run's judge server, no longer needed).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/slurm/env.sh
source scripts/slurm/lib.sh
export HF_HUB_OFFLINE=1
PHASES="${DPT_FINAL_PHASES:-select,export,main,hard,budgets,report}"
mapfile -t NODES < <(job_nodes)
(( ${#NODES[@]} >= 2 )) || tdie "this job needs the RL nodes (8), it has ${#NODES[@]}"
HEAD="${NODES[0]}"; WORKERS=("${NODES[@]:1}")
LOGS="${WORK}/dpt/logs/finaleval_${SLURM_JOB_ID:-local}"; PIDS="${LOGS}/pids"
mkdir -p "${LOGS}" "${PIDS}"
read -r R CK JUDGE_ID TAG BASE SECURE <<< "$(python - <<'PY'
from dpt.common.config import load_config
p = load_config().project
print(p.paths.eval_root, p.paths.ckpt_root, p.judges.primary.hf_id, p.model.tag, p.model.base, p.paths.eval_secure_root)
PY
)"
ROW="${DPT_FINAL_ROW:-rl_v2_seed1}"; RUN="${DPT_FINAL_RUN:-${CK}/rl/rl${TAG}_v2_seed1}"
EXPORT="${DPT_FINAL_EXPORT:-${RUN}/hf_export}"
EP="http://127.0.0.1:8000"
OV=(eval_suite.sandbox_endpoint=http://127.0.0.1:8080 eval_suite.judge.endpoint=http://127.0.0.1:8100
    eval_suite.judge.model=judge eval_suite.judge.max_concurrency=24 eval_suite.max_concurrency=384)
want() { [[ ",${PHASES}," == *",$1,"* ]]; }
phase() { tlog "final eval: $*"; }
[[ -z "${DPT_CANCEL_JOB:-}" ]] || run scancel "${DPT_CANCEL_JOB}" || true

# ---- servers --------------------------------------------------------------------------------------------------
start_replicas() {   # start_replicas <host> <model> <served name> <first port> [env] [max_model_len]: one per GPU
  local host="$1" model="$2" name="$3" port0="$4" extra="${5:-}" maxlen="${6:-40960}" g
  for g in 0 1 2; do
    on_node "${host}" bash -c "${extra} DPT_VLLM_DP=1 CUDA_VISIBLE_DEVICES=${g} GPU_MEM=0.9 setsid nohup bash \
scripts/serve_vllm.sh '${model}' '${name}' $((port0 + g)) ${maxlen} 1 > '${LOGS}/${name}_${host}_$((port0 + g)).log' \
2>&1 < /dev/null & echo \$! > '${PIDS}/${host}_$((port0 + g))'"
  done
}
start_here() {       # start_here <pid name> <command…>: in its own process group on this node
  local name="$1"; shift
  if dry; then run "$@"; return 0; fi
  setsid nohup "$@" > "${LOGS}/${name}.log" 2>&1 < /dev/null &
  echo $! > "${PIDS}/${HEAD}_${name}"
}
wait_backends() { local u; for u in "$@"; do wait_url "${u}" 3600 || tdie "${u} did not come up (logs ${LOGS})"; done; }
wait_sandbox() {     # the sandbox answers /health, not /v1/models
  dry && return 0
  local i=0
  until curl -sf --max-time 10 http://127.0.0.1:8080/health >/dev/null; do
    sleep 5; i=$((i + 5)); (( i < 600 )) || tdie "sandbox :8080 not up after 600 s (${LOGS}/sandbox.log)"
  done
}
stop_all() {
  local f host
  for f in "${PIDS}"/*; do
    [[ -e "${f}" ]] || continue
    host="$(basename "${f}")"; host="${host%%_*}"
    on_node "${host}" bash -c "kill -TERM -- -$(cat "${f}") 2>/dev/null; true" || true
    rm -f "${f}"
  done
}
trap stop_all EXIT
gpus_free() {        # wait up to 5 min until no process holds a GPU on this job's nodes
  dry && return 0
  local h i busy
  for i in $(seq 1 30); do
    busy=0
    for h in "${NODES[@]}"; do
      [[ -z "$(on_node "${h}" nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]] || busy=1
    done
    (( busy )) || return 0
    sleep 10
  done
  tlog "WARN GPUs still in use 5 min after stopping the servers"
}
serve_64k() {        # serve_64k <model> <served name>: one replica per GPU on every node at 64K + prompt, proxy :8000
  local urls=() args=() h p
  for h in "${NODES[@]}"; do start_replicas "${h}" "$1" "$2" 8001 DPT_GDN_MODEL=1 67584; done
  for h in "${NODES[@]}"; do for p in 8001 8002 8003; do urls+=("http://${h}:${p}"); done; done
  wait_backends "${urls[@]}"
  for p in "${urls[@]}"; do args+=(--backend "${p}"); done
  start_here 8000 python scripts/rr_proxy.py --port 8000 "${args[@]}"
  wait_backends "http://${HEAD}:8000"
}
harness() { run python -m dpt.eval.harness --endpoint "${EP}" "$@" "${OV[@]}"; }

# ---- 1-2: the RL checkpoint ----------------------------------------------------------------------------------
STEP="${DPT_FINAL_STEP:-}"
if want select && [[ -z "${STEP}" ]]; then
  if dry; then STEP=120; else
    SEL="$(python -m dpt.eval.select_checkpoint --run "${RUN}")"; echo "${SEL}" > "${LOGS}/selection.json"
    STEP="$(python -c 'import json, sys; print(json.loads(sys.argv[1])["selected_step"])' "${SEL}")"
  fi
  phase "dev selection: step ${STEP} (${LOGS}/selection.json)"
fi
if want export; then
  [[ -n "${STEP}" ]] || tdie "no RL step selected (phase select, or DPT_FINAL_STEP)"
  phase "export ${RUN}/global_step_${STEP} → ${EXPORT}"
  run rm -rf "${EXPORT}.partial"
  run bash scripts/export_hf.sh "${RUN}/global_step_${STEP}" "${EXPORT}.partial" "${BASE}"
  run rm -rf "${EXPORT}"; run mv "${EXPORT}.partial" "${EXPORT}"
  dry || echo "${STEP}" > "${EXPORT}/SELECTED_STEP"
fi

# ---- 3-5: the evaluation ---------------------------------------------------------------------------------------
if want main || want hard || want k64; then
  dry || [[ -f "${EXPORT}/config.json" ]] || tdie "no HF export at ${EXPORT}"
  sandbox_image || tdie "no sandbox image (check_node.slurm pulls it)"
  start_here sandbox python -m dpt.rewards.sandbox_server --host 127.0.0.1 --port 8080 \
    --hidden-tests-dir "${SECURE}/hidden_tests" --max-concurrency 32 --python "${DPT_SANDBOX_PYTHON}" \
    --container-image "${DPT_SANDBOX_SIF}"
  start_replicas "${HEAD}" "${JUDGE_ID}" judge 8101
  urls=(); for h in "${WORKERS[@]}"; do start_replicas "${h}" "${EXPORT}" "${ROW}" 8001 DPT_GDN_MODEL=1; done
  for h in "${WORKERS[@]}"; do for p in 8001 8002 8003; do urls+=("http://${h}:${p}"); done; done
  wait_backends "http://${HEAD}:8101" "http://${HEAD}:8102" "http://${HEAD}:8103" "${urls[@]}"
  start_here 8100 python scripts/rr_proxy.py --port 8100 --backend http://127.0.0.1:8101 \
    --backend http://127.0.0.1:8102 --backend http://127.0.0.1:8103
  args=(); for u in "${urls[@]}"; do args+=(--backend "${u}"); done
  start_here 8000 python scripts/rr_proxy.py --port 8000 "${args[@]}"
  wait_backends "http://${HEAD}:8100" "http://${HEAD}:8000"; wait_sandbox
  if want main; then
    phase "${ROW}: every set at medium"
    harness --model-name "${ROW}" --served-model "${ROW}" --sets d_num,d_code,d_report,d_fresh,general,refusal \
      --effort medium --reference-run "${R}/base_medium" --out "${R}/${ROW}" 2>&1 | tee "${LOGS}/main.log"
  fi
  if want hard; then
    phase "${ROW}: D-Num-hard"
    harness --model-name "${ROW}" --served-model "${ROW}" --sets d_num_hard --effort medium \
      --out "${R}/d_num_hard/${ROW}" 2>&1 | tee "${LOGS}/hard.log"
  fi
  if want k64; then
    phase "${ROW}: D-Num with 64 samples"
    harness --model-name "${ROW}" --served-model "${ROW}" --sets d_num --effort medium --samples-override 64 \
      --out "${R}/${ROW}_k64" 2>&1 | tee "${LOGS}/k64.log"
  fi
fi

# ---- 6: 64K by continuation, and the three-budget table --------------------------------------------------------
# A 64K sample equals the 32K one up to token 32K, so only the answers whose thinking did not end are continued
# (dpt.eval.budget). The judge and the sandbox are not needed: every GPU serves the model, at max_model_len 67,584.
if want budgets; then
  stop_all; gpus_free
  phase "${ROW}: answers cut at 32K continued to 64K"
  serve_64k "${EXPORT}" "${ROW}"
  for s in d_num_hard d_fresh; do
    run python -m dpt.eval.budget continue --set "${s}" --row "${ROW}" --endpoint "${EP}" --served-model "${ROW}" \
      --concurrency 96 2>&1 | tee -a "${LOGS}/budgets.log"
  done
  stop_all; gpus_free
  SFT_LEFT=0
  dry || SFT_LEFT="$(python -m dpt.eval.budget todo --set d_fresh --row sft)"
  if (( SFT_LEFT > 0 )); then
    phase "sft: ${SFT_LEFT} D-Fresh answers cut at 32K continued to 64K"
    serve_64k "${WORK}/dpt/exports/sft9b_seed1" sft
    run python -m dpt.eval.budget continue --set d_fresh --row sft --endpoint "${EP}" --served-model sft \
      --concurrency 96 2>&1 | tee -a "${LOGS}/budgets.log"
    stop_all
  fi
  phase "three budgets against SFT (${R}/budgets_${ROW}.json)"
  ROWS="sft,rl_seed1,rl_v2_seed1,sft_v2_seed1,rl_v3_seed1,base_medium"     # rows without generations are skipped
  [[ ",${ROWS}," == *",${ROW},"* ]] || ROWS="${ROWS},${ROW}"
  run python -m dpt.eval.budget compare --rows "${ROWS}" --baseline sft \
    --sets d_num_hard,d_fresh,d_num --out "${R}/budgets_${ROW}.json" 2>&1 | tee -a "${LOGS}/budgets.log"
fi

# ---- 7: report and ship gates ------------------------------------------------------------------------------------
if want report; then
  phase "report and ship gates (${ROW})"
  run env DPT_FINAL_CANDIDATE="${ROW}" bash scripts/07_final_eval.sh --report-only 2>&1 | tee "${LOGS}/report.log"
fi
phase "done (${PHASES})"
