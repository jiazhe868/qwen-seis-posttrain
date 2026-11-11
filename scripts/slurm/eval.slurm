#!/bin/bash
#SBATCH -J dpt-eval
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 24:00:00
#SBATCH -o dpt-eval-%j.out
# Evaluate one row of the results table on one node: the fixed harness (dpt.eval.harness) against a
# vLLM server on this node (9B: TP1 × DP3), with D-Code solutions run in the container sandbox and
# D-Report judged pairwise by the `judge` endpoint (serve.slurm judge ...).
#
#   sbatch -A <alloc> scripts/slurm/eval.slurm base        # step 03: base, noise floor ×10, pass@64 (~1 day)
#   sbatch -A <alloc> scripts/slurm/eval.slurm sft         # step 07 --only sft, then the RL-entry gate
#   sbatch -A <alloc> scripts/slurm/eval.slurm rl_seed1    # after selecting + exporting the RL checkpoint
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/slurm/env.sh
source scripts/slurm/lib.sh
ROW="${1:?row: base | sft | rl_seed1 | rl_seed2}"
export HF_HUB_OFFLINE=1
SECURE="$(python -c 'from dpt.common.config import load_config; print(load_config().project.paths.eval_secure_root)')"

# Hidden tests for D-Code / LiveCodeBench live next to the eval sets and never leave this node.
python -m dpt.rewards.sandbox_server --host 127.0.0.1 --port 8080 --hidden-tests-dir "${SECURE}/hidden_tests" \
  --max-concurrency 32 --python "${DPT_SANDBOX_PYTHON}" --container-image "${DPT_SANDBOX_SIF}" &
SANDBOX=$!
trap 'kill ${SANDBOX} 2>/dev/null || true' EXIT
OV=("eval_suite.sandbox_endpoint=http://127.0.0.1:8080")
if endpoint_up judge; then
  OV+=("eval_suite.judge.endpoint=$(endpoint_url judge)" "eval_suite.judge.model=$(endpoint_model judge)")
else
  tlog "WARN no judge endpoint: D-Report pairwise judging will fail closed (serve.slurm judge ...)"
fi
export DPT_EVAL_OVERRIDES="${OV[*]}"
export DPT_VLLM_DP="$(nvidia-smi -L | wc -l)"     # one 9B replica per GPU behind one endpoint
if [[ "${ROW}" == base ]]; then
  bash scripts/03_baseline_eval.sh
else
  bash scripts/07_final_eval.sh --only "${ROW}"
fi
