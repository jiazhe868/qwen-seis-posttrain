#!/bin/bash
#SBATCH -J dpt-sftv2
#SBATCH -p gpu-a100
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 48:00:00
#SBATCH -o dpt-sftv2-%j.out
# Job 2 as one self-contained 8-node job (2026-11-12, the user's decision): SFT v2, continued from SFT v1 on the
# knowledge-dense report rows (dpt.data.gen.knowledge) plus a replay sample of the v1 mix; its eval; RL v3 from SFT
# v2; the final eval. All of it on this job's own nodes.
#
# Why this file is so short: Slurm copies a batch script when the job is submitted, so any logic written here would
# be frozen for the days the job waits in the queue. This script only waits for the go-ahead and then runs
# scripts/slurm/sft_v2_job.sh, which is read from the repository when the job starts, so it can still be written and
# tested while the job waits. The go-ahead is the file $WORK/dpt/sft_v2_job.ready, written once that script is synced
# and tested; without it the job waits up to 6 h and then exits, so an early start never runs unfinished code.
#
#   scripts/slurm/submit_window.sh sft-v2-queue     # 8 gpu-a100 nodes, 48 h
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/slurm/env.sh
source scripts/slurm/lib.sh

ready_flag="${WORK}/dpt/sft_v2_job.ready"
waited_seconds=0
until [[ -f "${ready_flag}" ]] || dry; do
  (( waited_seconds < 6 * 3600 )) || tdie "no ${ready_flag} after 6 h: sft_v2_job.sh is not marked ready, stopping"
  (( waited_seconds % 1800 == 0 )) && tlog "waiting for ${ready_flag} (sft_v2_job.sh not marked ready yet)"
  sleep 300; waited_seconds=$(( waited_seconds + 300 ))
done
exec bash scripts/slurm/sft_v2_job.sh
