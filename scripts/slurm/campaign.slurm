#!/bin/bash
#SBATCH -J dpt-campaign
#SBATCH -p gpu-a100
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 48:00:00
#SBATCH -o dpt-campaign-%j.out
# The whole training run in one 8-node allocation (plan §8, decided 2026-10-24): one queue wait instead of
# one per stage. scripts/slurm/campaign.sh runs the baseline eval spread over the nodes, SFT, the export, the
# SFT eval and the RL-entry gate, then RL until the walltime ends. ~3 SU per node-hour → 24 SU per hour.
#
#   sbatch -A <alloc> scripts/slurm/campaign.slurm        # submit with 48 h; the site extends it to 96 h on request
#
# Its RL judge is a serve.slurm job on gpu-a100-small submitted with --dependency=after:<this job>. Write that
# job's id to $WORK/dpt/campaign/judge_job: campaign.sh cancels it when the campaign ends without a successor.
#
# This file is copied when the job is submitted; campaign.sh is read when it starts, so the steps can still
# be finished after submission. Until campaign.sh is marked ready ($WORK/dpt/campaign/READY), an early
# start waits at most DPT_CAMPAIGN_READY_WAIT_S (1 h) and then ends, so it cannot burn SUs on unfinished steps.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
READY="${WORK}/dpt/campaign/READY"
waited=0
until [[ -e "${READY}" ]]; do
  (( waited >= ${DPT_CAMPAIGN_READY_WAIT_S:-3600} )) && { echo "$(date) ${READY} missing: campaign.sh not ready, ending"; exit 1; }
  sleep 60; waited=$((waited + 60))
done
exec bash scripts/slurm/campaign.sh "$@"
