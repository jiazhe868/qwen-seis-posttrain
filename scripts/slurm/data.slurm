#!/bin/bash
#SBATCH -J dpt-data
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 48:00:00
#SBATCH -o dpt-data-%j.out
# Build the training data on a CPU node. The models are vLLM servers on GPU nodes (serve.slurm), found by
# name in $DPT_ENDPOINTS; this job only sends requests, runs the graders and runs model-written code in
# the sandbox container.
#
#   sbatch -A <alloc> scripts/slurm/data.slurm [gen | bank | sft | all]          (default: all)
#
#   gen   T1 seismology problems (held-out families → D-Num candidates), T2 code tasks (held-out → D-Code),
#         T4 public math + instruction prompts, T3 report prompts (teacher), T1 paraphrase (teacher)
#   bank  the RL problem bank: dpt.data.pipeline ingest → quality → dedup → decontam → verify → difficulty
#         → publish. Verification audits 20 instances per template with the teacher (medium effort) and the
#         judge and keeps or drops each template whole; difficulty from the base; decontaminated against every
#         eval set and the public benchmarks (embedding + teacher paraphrase judge)
#   sft   SFT subsets from the verified bank (dpt.data.gen.sft_gen: teacher rejection sampling and base
#         self-distillation; the RL dev split is excluded), then the pipeline's sft stage
#
# Endpoints (sbatch -A <alloc> scripts/slurm/serve.slurm <name> <model> [tp]):
#   teacher   Qwen/Qwen3.8-27B            gen, bank, sft. Several servers: serve them as teacher, teacher2, …
#                                          and set DPT_TEACHER_ENDPOINTS="teacher teacher2 …"
#   judge     RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic (-p gpu-a100-small, TP1)   bank, sft (report scores)
#   base      Qwen/Qwen3.5-9B             bank (difficulty, per template), sft (self-distillation); one A100 is
#                                          enough (-p gpu-a100-small, TP1)
#   (no embedding server: decontamination embeds in-process with Qwen/Qwen3-Embedding-0.6B on this node)
#
# The hard bank (dpt.data.gen.seismology_hard, 2026-10-27) runs the bank stages on its own candidates, into its
# own directory and markers, with its own verification judge (the RL judge keeps the name `judge`) and the
# SFT model for the difficulty audit:
#   DPT_BANK_NAME=bank_hard DPT_BANK_OUT=$WORK/dpt/data/v1_hard \
#   DPT_BANK_CANDIDATES=$WORK/dpt/data/gen/t1_hard_candidates.jsonl DPT_JUDGE_ENDPOINT=judge_v \
#   DPT_DIFFICULTY_ENDPOINT=sft_audit sbatch -A <alloc> scripts/slurm/data.slurm bank
#
# Layout: $DATA/gen (candidates, drafts, SFT subsets), $DATA/v1 (the bank = data_pipeline.output_dir), and
# the eval material (D-Num / D-Code candidates, decontamination audit) in the secure eval dir. Each step
# runs once: a rerun skips finished steps (markers in $DATA/gen/.done) and resumes the unfinished one.
# DPT_DATA_FORCE=1 reruns finished steps (resumable steps continue from their progress files; delete a
# step's outputs to start it over). Eval files are never replaced unless DPT_EVAL_SETS_FORCE=1 (before
# the freeze).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/slurm/env.sh
source scripts/slurm/lib.sh
export HF_HUB_OFFLINE=1                                  # tokenizers come from $HF_HOME (setup_login.sh models)
WHICH="${1:-all}"
case "${WHICH}" in gen | bank | sft | all) ;; *) tdie "usage: data.slurm [gen | bank | sft | all]" ;; esac

read -r DATA SECURE RAW TEACHER_ID BASE_ID SEED <<< "$(python - <<'PY'
from dpt.common.config import load_config
c = load_config().project
print(c.paths.data_root, c.paths.eval_secure_root, c.paths.raw_root, c.model.teacher, c.model.base, c.seed)
PY
)"
GEN="${DATA}/gen"; BANK="${DPT_BANK_OUT:-${DATA}/v1}"; SFT="${GEN}/sft"; DONE="${GEN}/.done"
T1="${GEN}/t1_candidates.jsonl"; T1P="${GEN}/t1_paraphrased.jsonl"
T2_DIR="${GEN}/t2"; T2="${T2_DIR}/candidates_t2.jsonl"; T2_HIDDEN="${T2_DIR}/hidden_tests"
T3="${GEN}/t3_candidates.jsonl"; T4="${GEN}/t4_candidates.jsonl"
# D-Num draft: a superset of D-Num (before the wording check), protected by decontamination as a whole
# D-Code and its hidden tests go to ${SECURE}/d_code.jsonl and ${SECURE}/hidden_tests (served by eval.slurm).
D_NUM_CAND="${SECURE}/d_num_draft.jsonl"; D_CODE="${SECURE}/d_code.jsonl"
D_REPORT="${SECURE}/d_report.jsonl"; D_FRESH="${SECURE}/d_fresh.jsonl"
REWARDS="${GEN}/rewards_data.yaml"                       # rl/rewards.yaml + hidden-test dir + judge endpoint
PUBLIC="${SECURE}/public_benchmarks"                     # benchmark prompts for decontamination (import_general)
EMBED_ID="${DPT_EMBED_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
TEACHERS="${DPT_TEACHER_ENDPOINTS:-teacher}"; TEACHER1="${TEACHERS%% *}"
JUDGE_EP="${DPT_JUDGE_ENDPOINT:-judge}"; DIFF_EP="${DPT_DIFFICULTY_ENDPOINT:-base}"     # bank stages only
BANK_OV=(); [[ -z "${DPT_BANK_OUT:-}" ]] || BANK_OV=("data_pipeline.output_dir=${BANK}")

# ---- helpers ---------------------------------------------------------------------------------------
force()  { [[ "${DPT_DATA_FORCE:-0}" == 1 ]]; }
done_before() { [[ -e "${DONE}/$1" ]] && ! force; }
step() {   # step <name> <command…>: run it unless it finished before; mark it finished on success
  local name="$1"; shift
  if done_before "${name}"; then tlog "${name}: finished before, skipping (DPT_DATA_FORCE=1 reruns it)"; return 0; fi
  tlog "${name}: start"
  run "$@"
  if ! dry; then mkdir -p "${DONE}"; date > "${DONE}/${name}"; fi
  tlog "${name}: done"
}
guard_eval() {   # an existing eval file may be frozen: never replaced unless DPT_EVAL_SETS_FORCE=1
  [[ ! -e "$1" || "${DPT_EVAL_SETS_FORCE:-0}" == 1 ]] ||
    tdie "$1 exists (frozen eval material?): delete it, or DPT_EVAL_SETS_FORCE=1 before the freeze"
}
need() {   # need <endpoint>: published by serve.slurm and answering, waiting up to $DPT_ENDPOINT_WAIT_S
  local u                                         # (a dry run checks only the URL file)
  if ! dry; then wait_endpoint "$1" || true; fi
  u="$(endpoint_url "$1" || true)"               # (a failed $(…) in an assignment would exit under set -e)
  [[ -n "${u}" ]] || tdie "no ${1} endpoint: sbatch -A <alloc> scripts/slurm/serve.slurm ${1} <model> ... (see the header)"
  dry || endpoint_up "$1" || tdie "the ${1} endpoint at ${u} does not answer"
}
url() { printf '%s/v1' "$(endpoint_url "$1")"; }
TEACHER=()
teacher_args() {   # --url/--model per teacher server, then the tokenizer that renders the prompts
  TEACHER=(); local e
  for e in ${TEACHERS}; do need "${e}"; TEACHER+=(--url "$(endpoint_url "${e}")" --model "$(endpoint_model "${e}")"); done
  TEACHER+=(--tokenizer "${TEACHER_ID}")
}
rewards_config() {   # graders of this job (and of the pipeline's difficulty stage) read $DPT_REWARDS_CONFIG
  # `rewards_config optional`: nothing graded afterwards calls the judge, so with no judge published (between
  # judge windows) the config gets a placeholder endpoint, which fails closed if it is ever called.
  local u="http://judge-not-up-yet:8000/v1" m=judge up=0
  if dry; then [[ -z "$(endpoint_url judge)" ]] || up=1; elif endpoint_up judge; then up=1; fi
  if [[ "${1:-}" != optional ]] || (( up )); then need judge; u="$(url judge)"; m="$(endpoint_model judge)"; fi
  run python -m dpt.data.gen.sft_gen rewards-config --out "${REWARDS}" \
    "rewards.t2_code.sandbox.hidden_tests_dir=${T2_HIDDEN}" \
    "rewards.t3_report.judge.base_url=${u}" "rewards.t3_report.judge.model=${m}"
}
DEC=()
decontam_args() {   # every eval set is protected; the paraphrase judge is the teacher (plan §5)
  local f canaries
  for f in "${D_NUM_CAND}" "${D_CODE}" "${D_REPORT}" "${D_FRESH}"; do
    [[ -f "${f}" ]] || tdie "decontamination needs every eval set: no ${f} (data.slurm gen; eval_sets.slurm)"
  done
  [[ -n "$(ls "${PUBLIC}"/*.jsonl 2>/dev/null)" ]] || dry ||
    tdie "no public benchmark prompts in ${PUBLIC} (python -m dpt.eval.import_general build --decontam-out ...)"
  need "${TEACHER1}"
  canaries="$(python -c 'import json; from dpt.data.decontam import make_canary as c
print(json.dumps([c(f"{s}/v1") for s in ("d_num", "d_code", "d_report", "d_fresh", "d_num_hard")]))')"
  DEC=("data_pipeline.inputs.eval_sets.d_num=${D_NUM_CAND}" "data_pipeline.inputs.eval_sets.d_code=${D_CODE}"
       "data_pipeline.inputs.eval_sets.d_report=${D_REPORT}" "data_pipeline.inputs.eval_sets.d_fresh=${D_FRESH}"
       "data_pipeline.decontam.canaries=${canaries}" "data_pipeline.decontam.audit_dir=${SECURE}/decontam_audit"
       "data_pipeline.inputs.public_benchmarks_dir=${PUBLIC}"
       "data_pipeline.backends.embed.local=true" "data_pipeline.backends.embed.model=${EMBED_ID}"
       "data_pipeline.backends.judge.base_url=$(url "${TEACHER1}")"
       "data_pipeline.backends.judge.model=$(endpoint_model "${TEACHER1}")")
  # D-Num-hard (2026-10-27): its draft is protected like D-Num's once it exists
  [[ ! -f "${SECURE}/d_num_hard_draft.jsonl" ]] || DEC+=("data_pipeline.inputs.eval_sets.d_num_hard=${SECURE}/d_num_hard_draft.jsonl")
}

# ---- gen -------------------------------------------------------------------------------------------
do_gen() {
  mkdir -p "${GEN}" "${SECURE}"
  if done_before t1; then step t1; else           # t1 also writes the D-Num candidates (eval material)
    guard_eval "${D_NUM_CAND}"
    step t1 python -m dpt.data.gen.t1 build --out "${T1}" --heldout-out "${D_NUM_CAND}" --seed "${SEED}"
  fi
  if done_before t2; then step t2; else           # t2 also writes D-Code, and runs reference code in the sandbox
    guard_eval "${D_CODE}"; sandbox_image || tdie "no sandbox image (check_node.slurm pulls it)"
    step t2 python -m dpt.data.gen.t2 build --out-dir "${T2_DIR}" --eval-out-dir "${SECURE}" \
      --n-train "${DPT_T2_TRAIN:-3000}" --n-heldout "${DPT_T2_HELDOUT:-150}" --seed "${SEED}" --check
  fi
  step t4 python -m dpt.data.gen.t4 build --raw-dir "${RAW}/t4" --out "${T4}"
  if ! done_before t3 || ! done_before paraphrase; then teacher_args; fi
  step t3 python -m dpt.data.gen.t3 build --work-dir "${GEN}/t3" --out "${T3}" \
    --spot-check "${GEN}/t3_spot_check.csv" --seed "${SEED}" "${TEACHER[@]}"
  step paraphrase python -m dpt.data.gen.paraphrase --in "${T1}" --out "${T1P}" \
    --fraction "${DPT_PARAPHRASE_FRACTION:-0.2}" --seed "${SEED}" "${TEACHER[@]}"
}

# ---- bank ------------------------------------------------------------------------------------------
do_bank() {
  # DPT_BANK_STAGES runs a subset, e.g. ingest,quality,dedup,decontam on a CPU node (decontamination embeds on
  # the CPU) and verify,difficulty,publish elsewhere; each subset has its own completion marker. Only the
  # servers the chosen stages use are required.
  local stages="${DPT_BANK_STAGES:-ingest,quality,dedup,decontam,verify,difficulty,publish}"
  local name="${DPT_BANK_NAME:-bank}" marker
  marker="${name}"; [[ "${stages}" == "ingest,quality,dedup,decontam,verify,difficulty,publish" ]] || marker="${name}_${stages//,/_}"
  done_before "${marker}" && { tlog "${marker}: finished before, skipping (DPT_DATA_FORCE=1 reruns it)"; return 0; }
  has() { [[ ",${stages}," == *",$1,"* ]]; }
  local f fams cands ov=()
  local inputs=("${T1P}" "${T2}" "${T3}" "${T4}")
  [[ -z "${DPT_BANK_CANDIDATES:-}" ]] || read -r -a inputs <<< "${DPT_BANK_CANDIDATES}"
  if has ingest; then
    for f in "${inputs[@]}"; do [[ -f "${f}" ]] || dry || tdie "no ${f}: run data.slurm gen first"; done
  fi
  cands="$(python -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "${inputs[@]}")"
  ov+=("data_pipeline.inputs.rl_candidates=${cands}" "${BANK_OV[@]}")
  if has decontam; then decontam_args; ov+=("${DEC[@]}"); fi
  if has verify; then
    teacher_args; need "${JUDGE_EP}"
    # url_file + wait_s: a request follows the endpoint to the next server window instead of failing.
    # timeout_s: with 256 problems in flight a request can queue ~30 min behind the others (2026-10-24).
    fams="$(python -c 'import json, sys
w, t = 2700, 7200
print(json.dumps([{"name": "teacher", "base_url": sys.argv[1], "model": sys.argv[2], "tokenizer": sys.argv[5],
                   "effort": "medium", "url_file": sys.argv[6], "wait_s": w, "timeout_s": t},
                  {"name": "judge", "base_url": sys.argv[3], "model": sys.argv[4], "url_file": sys.argv[7],
                   "wait_s": w, "timeout_s": t}]))' \
      "$(url "${TEACHER1}")" "$(endpoint_model "${TEACHER1}")" "$(url "${JUDGE_EP}")" "$(endpoint_model "${JUDGE_EP}")" \
      "${TEACHER_ID}" "${DPT_ENDPOINTS}/${TEACHER1}.url" "${DPT_ENDPOINTS}/${JUDGE_EP}.url")"
    ov+=("data_pipeline.verify.families=${fams}" "data_pipeline.verify.mode=audit" "data_pipeline.verify.family_samples=4")
  fi
  if has difficulty; then
    need "${DIFF_EP}"
    # Only T3 grading calls the judge: with t3_report skipped (the default) the difficulty pass needs none.
    if [[ "${DPT_DIFFICULTY_SKIP:-[t3_report]}" == *t3_report* ]]; then rewards_config optional; else rewards_config; fi
    # Grading T2 runs model-written code: only in the sandbox container (skip t2_code where there is none).
    [[ "${DPT_DIFFICULTY_SKIP:-[t3_report]}" == *t2_code* ]] || sandbox_image || tdie "no sandbox image (check_node.slurm pulls it)"
    # DPT_DIFFICULTY_MODE: audit (20 per template, the 10-25 pass), audit_all (audit, then every problem of the
    # surviving templates: per-problem pass rates for RL's difficulty weights, 2026-11-07) or all.
    ov+=("data_pipeline.difficulty.mode=${DPT_DIFFICULTY_MODE:-audit}" "data_pipeline.difficulty.skip_tasks=${DPT_DIFFICULTY_SKIP:-[t3_report]}"
         "data_pipeline.backends.base.base_url=$(url "${DIFF_EP}")" "data_pipeline.backends.base.model=$(endpoint_model "${DIFF_EP}")")
    # DPT_DIFFICULTY_CONCURRENCY: problems in flight when `base` is a proxy over several servers (pipeline.yaml's
    # max_concurrency is sized for one).
    [[ -z "${DPT_DIFFICULTY_CONCURRENCY:-}" ]] || ov+=("data_pipeline.difficulty.max_concurrency=${DPT_DIFFICULTY_CONCURRENCY}")
  fi
  # The SFT mix excludes the RL dev candidates fixed after verification (pipeline --dev-candidates), so publish
  # draws the dev split from them alone; written here first if verification finished in an earlier run.
  local cand="${BANK}/rl_dev_candidates.jsonl" verified="${BANK}/stages/04_verify.jsonl"
  if has publish && ! has verify; then
    [[ -f "${cand}" || ! -f "${verified}" ]] || run python -m dpt.data.pipeline --config data/pipeline.yaml --dev-candidates "${BANK_OV[@]}"
    if [[ -f "${cand}" ]] || { dry && [[ -f "${verified}" ]]; }; then ov+=("data_pipeline.publish.dev_candidates=true"); fi
  fi
  step "${marker}" env DPT_REWARDS_CONFIG="${REWARDS}" python -m dpt.data.pipeline --config data/pipeline.yaml \
    --stages "${stages}" "${ov[@]}"
}

# ---- sft -------------------------------------------------------------------------------------------
do_sft() {
  local verified="${BANK}/stages/04_verify.jsonl" dev="${BANK}/rl_dev.jsonl" cand="${BANK}/rl_dev_candidates.jsonl"
  [[ -f "${verified}" ]] || dry || tdie "no verified bank at ${verified}: run data.slurm bank first"
  # RL dev problems never reach SFT. Publish draws the dev split from candidates fixed after verification
  # (pipeline --dev-candidates), so SFT excludes those and need not wait for the difficulty pass (~a day of
  # base-model time, 2026-10-24); a split published before the candidates existed is excluded as well.
  [[ -f "${cand}" ]] || run python -m dpt.data.pipeline --config data/pipeline.yaml --dev-candidates
  local excl=("${cand}") ex=() f
  [[ -f "${dev}" ]] && excl+=("${dev}")
  for f in "${excl[@]}"; do ex+=(--exclude "${f}"); done
  mkdir -p "${SFT}"
  sandbox_image || tdie "no sandbox image (check_node.slurm pulls it)"
  teacher_args                                    # the teacher also judges paraphrases in the mix's decontamination
  done_before sft_distill || need base            # the judge grades only report samples, the base only distills
  if done_before sft_report; then rewards_config optional; else rewards_config; fi
  local common=(--problems "${verified}" "${ex[@]}" --rewards-config "${REWARDS}" --seed "${SEED}")
  # Teacher time is the bottleneck (2026-10-23): 2 samples per problem on up to 8,000 problems.
  step sft_domain python -m dpt.data.gen.sft_gen domain "${common[@]}" --out "${SFT}/domain_reasoning.jsonl" \
    --k "${DPT_SFT_K:-2}" --limit "${DPT_SFT_DOMAIN_LIMIT:-8000}" \
    "${TEACHER[@]}"
  step sft_tool python -m dpt.data.gen.sft_gen tool "${common[@]}" --out "${SFT}/tool_traj.jsonl" --k "${DPT_SFT_K:-2}" \
    --hidden-tests-dir "${T2_HIDDEN}" "${TEACHER[@]}"
  step sft_report python -m dpt.data.gen.sft_gen report "${common[@]}" --out "${SFT}/report.jsonl" \
    --judge-url "$(url judge)" --judge-model "$(endpoint_model judge)" "${TEACHER[@]}"
  step sft_distill python -m dpt.data.gen.sft_gen distill "${common[@]}" --out "${SFT}/self_distill.jsonl" \
    --url "$(endpoint_url base)" --model "$(endpoint_model base)" --tokenizer "${BASE_ID}"
  decontam_args
  # Samples drawn before verification finished (2026-10-24) are filtered here: verified problems only, never
  # an RL dev candidate.
  step sft_mix python -m dpt.data.pipeline --config data/pipeline.yaml --stages sft \
    "data_pipeline.sft.keep_problems_from=${verified}" "data_pipeline.sft.exclude_problems_from=[$(IFS=,; echo "${excl[*]}")]" \
    "data_pipeline.inputs.sft_subsets.domain_reasoning=${SFT}/domain_reasoning.jsonl" \
    "data_pipeline.inputs.sft_subsets.tool_traj=${SFT}/tool_traj.jsonl" \
    "data_pipeline.inputs.sft_subsets.report=${SFT}/report.jsonl" \
    "data_pipeline.inputs.sft_subsets.self_distill=${SFT}/self_distill.jsonl" "${DEC[@]}"
}

tlog "data.slurm ${WHICH}: data ${DATA}, teacher servers: ${TEACHERS}"
case "${WHICH}" in
  gen) do_gen ;; bank) do_bank ;; sft) do_sft ;;
  all) do_gen; do_bank; do_sft ;;
esac
tlog "data.slurm ${WHICH}: finished"
