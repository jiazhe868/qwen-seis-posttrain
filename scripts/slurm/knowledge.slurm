#!/bin/bash
#SBATCH -J dpt-knowledge
#SBATCH -p gpu-h100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 24:00:00
#SBATCH -o dpt-knowledge-%j.out
# Knowledge-dense report data for SFT v2 (2026-11-11, dpt.data.gen.knowledge), as one 1-node job. It is submitted
# to two partitions and starts on whichever frees a node first:
#
#   sbatch -A PROJ -p gpu-h100,gpu-a100 scripts/slurm/knowledge.slurm
#   sbatch -A PROJ -p gpu-a100-dev -t 02:00:00 scripts/slurm/knowledge.slurm     # 2 h steps meanwhile
#
#   H100 node (80 GB GPUs): the teacher, FP8, one replica per GPU but the last (:8011…), and Gemma on the last GPU
#   A100 node (3 × 40 GB):  the teacher, FP8 with TP2 on GPUs 0-1 (:8011), and Gemma on GPU 2. One 40 GB A100 does
#                           not hold the FP8 teacher (jobs 3530657 and 3561787).
#   rr_proxy on :8010 (teacher) and :8110 (Gemma, served as judge_v).
#
# The client reads $DATA/knowledge/chunks.jsonl (written on the workstation by `knowledge select`), appends to
# records.jsonl and caches every reply, so a second job resumes. It starts no new section 50 min before the job's
# limit, then writes the SFT rows to $DATA/gen/sft/knowledge_report.jsonl (+ .summary.json, counts only). The job that
# finishes the last section also builds SFT v2's data (data.slurm knowledge: the SFT stage's decontamination, with
# this job's teacher as its paraphrase judge, then the replay mix) as $DATA/sft_v2/sft_train.jsonl.
# Logs: $WORK/dpt/logs/knowledge_<job>/. Dry run: DPT_SLURM_DRY_RUN=1 DPT_KNOWLEDGE_GPUS="NVIDIA H100 80GB HBM3"×n.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/slurm/env.sh
source scripts/slurm/lib.sh
export HF_HUB_OFFLINE=1

DATA="$(python -c 'from dpt.common.config import load_config; print(load_config().project.paths.data_root)')"
K="${DATA}/knowledge"
TEACHER_MODEL="${DPT_TEACHER_MODEL:-Qwen/Qwen3.8-27B-FP8}"
JUDGE_MODEL="${DPT_JUDGE_MODEL_ID:-RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic}"
LOGS="${WORK}/dpt/logs/knowledge_${SLURM_JOB_ID:-local}"
mkdir -p "${LOGS}"
[[ -f "${K}/chunks.jsonl" ]] || dry || tdie "no ${K}/chunks.jsonl: run 'python -m dpt.data.gen.knowledge select' and copy it"
export_rows() {
  run mkdir -p "${DATA}/gen/sft"
  run python -m dpt.data.gen.knowledge export --records "${K}/records.jsonl" --chunks "${K}/chunks.jsonl" \
    --out "${DATA}/gen/sft/knowledge_report.jsonl" 2>&1 | tee -a "${LOGS}/export.log"
}
# One client at a time: records.jsonl and the reply caches are appended to without a lock. A 2 h dev-queue job
# leaves at once when another knowledge job runs; a longer job waits for it (a dev job ends within 2 h).
others() { squeue -h -u "${USER}" -n dpt-knowledge -t R -o %i 2>/dev/null | grep -vx "${SLURM_JOB_ID:-none}" || true; }
if ! dry; then
  waited=0
  while [[ -n "$(others)" ]]; do
    [[ "${SLURM_JOB_PARTITION:-}" != gpu-a100-dev ]] || { tlog "knowledge job $(others | paste -sd,) is running: leaving"; exit 0; }
    (( waited < 7800 )) || tdie "knowledge job $(others | paste -sd,) still runs after 2 h 10 min"
    (( waited % 1800 == 0 )) && tlog "waiting for knowledge job $(others | paste -sd,) to end"
    sleep 300; waited=$(( waited + 300 ))
  done
  LEFT_SECTIONS="$(python -m dpt.data.gen.knowledge todo --chunks "${K}/chunks.jsonl" --records "${K}/records.jsonl")"
  if (( LEFT_SECTIONS == 0 )) && [[ -f "${DATA}/sft_v2/sft_train.jsonl" ]]; then
    tlog "every section has a record and SFT v2's data exists: exporting only"; export_rows; exit 0
  fi
  tlog "${LEFT_SECTIONS} sections to go"
fi
# A dev job keeps 25 min for the sections in flight; a 24 h job 50 min, plus the export
MARGIN_S="${DPT_KNOWLEDGE_MARGIN_S:-$([[ "${SLURM_JOB_PARTITION:-}" == gpu-a100-dev ]] && echo 1500 || echo 3000)}"
GROUPS_=()                                          # process groups of this job's servers and proxies
cleanup() {
  local p
  [[ "$(endpoint_url knowledge_teacher)" == "http://$(hostname -s):"* ]] &&
    rm -f "${DPT_ENDPOINTS}/knowledge_teacher.url" "${DPT_ENDPOINTS}/knowledge_teacher.model"
  for p in "${GROUPS_[@]}"; do kill -TERM -- "-${p}" 2>/dev/null || true; done
}
trap cleanup EXIT
bg() {   # bg <log> <command…>: in its own process group, output in $LOGS/<log>; its pid in $BG
  local log="${LOGS}/$1"; shift
  if dry; then printf 'DRY:'; printf ' %q' "$@"; printf ' > %q\n' "${log}"; true & BG=$!; return; fi
  setsid "$@" > "${log}" 2>&1 < /dev/null & BG=$!
  GROUPS_+=("${BG}")
}
serve() {  # serve <served name> <model> <port> <tp> <gpus> [VAR=value…]
  local name="$1" model="$2" port="$3" tp="$4" gpus="$5"; shift 5
  bg "${name}_${port}.log" env "$@" DPT_VLLM_DP=1 CUDA_VISIBLE_DEVICES="${gpus}" GPU_MEM=0.9 \
    bash scripts/serve_vllm.sh "${model}" "${name}" "${port}" 40960 "${tp}"
  STARTED+=("${port}:${BG}")
}

# ---- servers: the layout follows the node's GPUs ----------------------------------------------------------------
if [[ -n "${DPT_KNOWLEDGE_GPUS:-}" ]]; then mapfile -t GPUS < <(tr ',' '\n' <<< "${DPT_KNOWLEDGE_GPUS}")
else mapfile -t GPUS < <(nvidia-smi --query-gpu=name --format=csv,noheader); fi
N="${#GPUS[@]}"; STARTED=(); TEACHER_URLS=()
if [[ "${GPUS[0]}" == *H100* ]] && (( N >= 2 )); then
  for (( g = 0; g < N - 1; g++ )); do serve teacher "${TEACHER_MODEL}" $((8011 + g)) 1 "${g}"; TEACHER_URLS+=("http://127.0.0.1:$((8011 + g))"); done
elif (( N >= 3 )); then
  serve teacher "${TEACHER_MODEL}" 8011 2 0,1; TEACHER_URLS+=("http://127.0.0.1:8011")
else
  tdie "need 2 H100 or 3 A100 GPUs, this node has ${N}: ${GPUS[*]}"
fi
serve judge_v "${JUDGE_MODEL}" 8111 1 $((N - 1)) DPT_SERVE_DP=1
tlog "${N} × ${GPUS[0]}: teacher ×${#TEACHER_URLS[@]}, Gemma on GPU $((N - 1))"
for s in "${STARTED[@]}"; do wait_url "http://127.0.0.1:${s%%:*}" 3600 "${s##*:}"; done
args=(); for u in "${TEACHER_URLS[@]}"; do args+=(--backend "${u}"); done
bg proxy_teacher.log python scripts/rr_proxy.py --port 8010 "${args[@]}"; wait_url http://127.0.0.1:8010 300 "${BG}"
bg proxy_judge.log python scripts/rr_proxy.py --port 8110 --backend http://127.0.0.1:8111; wait_url http://127.0.0.1:8110 300 "${BG}"

# ---- sections → questions → answers → checks, then the SFT rows ---------------------------------------------------
knowledge_run() {   # knowledge_run [run options]: the client against this node's proxies
  run python -m dpt.data.gen.knowledge run --chunks "${K}/chunks.jsonl" --out "${K}/records.jsonl" \
    --cache-dir "${K}/cache" --teacher-url http://127.0.0.1:8010 --teacher-model teacher \
    --tokenizer Qwen/Qwen3.8-27B --judge-url http://127.0.0.1:8110 --judge-model judge_v "$@" 2>&1 |
    tee -a "${LOGS}/client.log"
}
# A first look: 40 sections. The prompts were never run on the teacher before (the workstation's vLLM fails with
# CUDA error 803 since November), so stop here if fewer than 10% of these answers pass the checks.
knowledge_run --limit 40 --concurrency 40
if ! dry; then
  python - "${K}/records.jsonl" <<'PY' || tdie "fewer than 10% of the first answers passed: read ${K}/records.jsonl"
import collections, json, sys
status = collections.Counter(a["status"] for line in open(sys.argv[1]) for a in json.loads(line)["answers"])
n = sum(status.values())
print("first sections:", dict(status), file=sys.stderr)
sys.exit(0 if n and status["ok"] >= 0.1 * n else 1)
PY
fi
if dry; then LEFT=86400; else LEFT="$(python -c 'from dpt.launch.walltime import slurm_time_left; print(slurm_time_left())')"; fi
knowledge_run --concurrency "${DPT_KNOWLEDGE_CONCURRENCY:-64}" --stop-new-after $(( LEFT - MARGIN_S ))
export_rows
# SFT v2's data, once every section has its record. The embedder of the decontamination runs on the CPU: the GPUs
# hold the servers.
if dry || (( $(python -m dpt.data.gen.knowledge todo --chunks "${K}/chunks.jsonl" --records "${K}/records.jsonl") == 0 )); then
  if ! dry; then
    mkdir -p "${DPT_ENDPOINTS}"
    echo "http://$(hostname -s):8010" > "${DPT_ENDPOINTS}/knowledge_teacher.url"
    echo teacher > "${DPT_ENDPOINTS}/knowledge_teacher.model"
  fi
  run env DPT_TEACHER_ENDPOINTS=knowledge_teacher CUDA_VISIBLE_DEVICES= bash scripts/slurm/data.slurm knowledge 2>&1 |
    tee -a "${LOGS}/mix.log"
fi
tlog "done: ${DATA}/gen/sft/knowledge_report.jsonl (sections still to do are resumed by the next job)"
