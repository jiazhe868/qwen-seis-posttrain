#!/bin/bash
#SBATCH -J dpt-bankv2
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 48:00:00
#SBATCH -o dpt-bankv2-%j.out
# Finish bank v2 (2026-11-07, README §8.3) once its inputs exist:
#   1. merge the verified v1 problems (already in $DATA/v2/stages) with the verified hard and long T1 families,
#      applying the atomic T3 rubrics (dpt.data.bank_merge merge);
#   2. the SFT model's difficulty pass for the new problems (audit_all; the v1 replies come from the cache of the
#      first pass), then publish rl_main / rl_hard / rl_dev into $DATA/v2;
#   3. check the published bank (dpt.data.bank_merge check; exit 1 if it cannot train).
# The SFT server (`sft_audit`) is reused when it still answers, else a new serve job is submitted and awaited.
#
#   sbatch -A <alloc> --dependency=afterok:<hard verify>:<long verify>:<t3 atomic>:<v2 difficulty> \
#       scripts/slurm/bank_v2_finish.slurm
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/slurm/env.sh
source scripts/slurm/lib.sh
export HF_HUB_OFFLINE=1

DATA="$(python -c 'from dpt.common.config import load_config; print(load_config().project.paths.data_root)')"
V2="${DATA}/v2"
for f in "${V2}/stages/04_verify.jsonl" "${DATA}/v1_hard/stages/04_verify.jsonl" \
         "${DATA}/v1_long/stages/04_verify.jsonl" "${V2}/t3_atomic_map.jsonl"; do
  [[ -f "${f}" ]] || tdie "missing ${f}"
done

if [[ ! -e "${V2}/.merged" ]]; then
  tlog "merge: v1 + hard + long, atomic T3"
  python -m dpt.data.bank_merge merge --out "${V2}/stages/04_verify.jsonl" --t3-map "${V2}/t3_atomic_map.jsonl" \
    "${V2}/stages/04_verify.jsonl" "${DATA}/v1_hard/stages/04_verify.jsonl" "${DATA}/v1_long/stages/04_verify.jsonl"
  touch "${V2}/.merged"
fi

if ! endpoint_up sft_audit; then
  tlog "sft_audit does not answer: submitting a new SFT server"
  DPT_GDN_MODEL=1 sbatch -A "${SLURM_JOB_ACCOUNT:-PROJ}" -p gpu-a100 -t 24:00:00 \
    scripts/slurm/serve.slurm sft_audit "${WORK}/dpt/exports/sft9b_seed1" 1
fi

tlog "difficulty (new problems; v1 replies from the cache) and publish"
DPT_BANK_OUT="${V2}" DPT_BANK_NAME=v2b DPT_BANK_STAGES=difficulty,publish DPT_DIFFICULTY_ENDPOINT=sft_audit \
  DPT_DIFFICULTY_MODE=audit_all DPT_DIFFICULTY_SKIP="[t3_report,t2_code]" DPT_DIFFICULTY_CONCURRENCY=96 \
  DPT_ENDPOINT_WAIT_S="${DPT_ENDPOINT_WAIT_S:-86400}" bash scripts/slurm/data.slurm bank

tlog "check"
python -m dpt.data.bank_merge check "${V2}" --reference-dev "${DATA}/v1/rl_dev.jsonl"
tlog "bank v2 ready: ${V2}"
