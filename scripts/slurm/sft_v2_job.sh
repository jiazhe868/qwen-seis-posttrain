#!/bin/bash
# Job 2 as one 8-node job, run by scripts/slurm/sft_v2_queue.slurm when that job starts. The queued batch script is
# frozen and calls this path, so the name stays; the file is read from the repository at that moment.
#
# 2026-11-15, the user's decision: job 2 runs RL v2 again with seed 2 (rl/rl_9b_v2_seed2.yaml) and evaluates it as
# row rl_v2_seed2. SFT v2 and RL v3 (planned 11-12) were dropped: an expert audit of 100 knowledge-dense rows, SFT
# v2's data, kept 2, would fix 48 and drop 50, and 98 had at least one factual error (mostly earthquake facts the
# teacher wrote from memory). A second seed tells whether an RL v2 gain over SFT holds beyond seed noise.
#
# rl_v2_job.sh checks (and if needed builds) bank v2, runs the 24K test unless a cap is known, trains, and
# evaluates. Seed 2 keeps seed 1's cap when seed 1 chose one: the same model, layout and lengths, so the same memory.
#
# Logs: this job's output, $WORK/dpt/logs/rlv2_<job>/ (steps), $WORK/dpt/logs/finaleval_<job>/ (the eval).
# Dry run: DPT_SLURM_DRY_RUN=1 DPT_SLURM_NODES="<8 nodes>" bash scripts/slurm/sft_v2_job.sh
#
# The whole script is one { … } block: bash reads all of it before the first step, so a copy synced while the
# job runs cannot change the running job's steps (scripts it calls are read when they start).
{
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/slurm/env.sh
source scripts/slurm/lib.sh
export HF_HUB_OFFLINE=1

read -r CK TAG <<< "$(python - <<'PY'
from dpt.common.config import load_config
p = load_config().project
print(p.paths.ckpt_root, p.model.tag)
PY
)"
SEED1_CAP="${CK}/rl/rl${TAG}_v2_seed1.cap.json"
if [[ -z "${DPT_RL_CAP:-}" && -f "${SEED1_CAP}" ]]; then
  export DPT_RL_CAP="$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["cap"])' "${SEED1_CAP}")"
  tlog "seed 2 keeps seed 1's cap: ${DPT_RL_CAP}"
fi
tlog "job 2: RL v2 with seed 2, then its final eval as row rl_v2_seed2"
exec env DPT_RL_CONFIG=rl/rl_9b_v2_seed2.yaml DPT_FINAL_ROW=rl_v2_seed2 bash scripts/slurm/rl_v2_job.sh
}
