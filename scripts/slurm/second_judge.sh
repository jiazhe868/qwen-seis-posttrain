#!/usr/bin/env bash
# The second D-Report judge (GPT-6 Luna through the OpenAI API) for rows already evaluated on the cluster.
#
#   bash scripts/slurm/second_judge.sh sft [rl_seed1 ...]
#
# Needed only when compute nodes cannot reach the API (check_node.slurm: "BLOCKED https://api.openai.com");
# otherwise eval.slurm runs both judges itself. Run it on a login node (light, network-bound: only HTTP
# calls, no generation) or anywhere else that has the run directories and the eval sets. It re-grades
# D-Report from the cached generations: the primary judge's verdicts come from its cache
# (<row>/d_report/judgments.primary.jsonl), the second judge's are new, and only D-Report is replaced in
# results.json (--update). Then the report and gates are rebuilt. Key: ~/.config/dpt/openai_key (env.sh).
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "${DPT_REPO}/scripts/common.sh"
[[ $# -ge 1 ]] || { echo "usage: $0 <row> [row ...]   (sft | rl_seed1 | rl_seed2)" >&2; exit 2; }
[[ -n "${OPENAI_API_KEY:-}" ]] || die "no OPENAI_API_KEY (put it in ~/.config/dpt/openai_key, see env.sh)"
R="$(cfg project.paths.eval_root)"
for row in "$@"; do
  [[ -f "${R}/${row}/results.json" ]] || die "no ${R}/${row}/results.json: evaluate ${row} first (eval.slurm ${row})"
  log "second judge: ${row}"
  "${PY}" -m dpt.eval.harness --model-name "${row}" --sets d_report --regrade-only --update \
    --reference-run "${R}/base_medium" --out "${R}/${row}"
done
bash "${DPT_REPO}/scripts/07_final_eval.sh" --report-only
