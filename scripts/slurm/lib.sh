# shellcheck shell=bash
# Helpers for the Slurm job scripts. Source after env.sh, inside a Slurm job.
#
# Dry run: DPT_SLURM_DRY_RUN=1 prints every remote command (ssh, ray, sbatch, scancel) instead of
# running it, so the job logic can be checked on any machine (tests/slurm/test_job_scripts.py).

tlog()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
tdie()  { tlog "FAIL $*"; exit 1; }
dry()   { [[ "${DPT_SLURM_DRY_RUN:-0}" == 1 ]]; }
# A dry-run line is printed in one write: background steps (one per node) print at the same time, and three
# separate writes let their lines interleave (2026-11-12: a test counted 7 of 8 SFT launches).
run()   { if dry; then printf 'DRY:%s\n' "$(printf ' %q' "$@")"; else "$@"; fi; }
# The id of the job `sbatch --parsable` just queued: its last output line, without the cluster name. The cluster's
# submit filter prints its banner on stdout as well (2026-10-25).
job_id() { tail -n 1 | cut -d';' -f1; }

# Hostnames of this job's nodes, one per line (first = head).
job_nodes() {
  if [[ -n "${DPT_SLURM_NODES:-}" ]]; then tr ' ,' '\n\n' <<< "${DPT_SLURM_NODES}" | grep .; return; fi
  scontrol show hostnames "${SLURM_JOB_NODELIST:?not inside a Slurm job}"
}
job_head() { job_nodes | head -1; }

# Hostnames of component <n> of this heterogeneous job (sbatch A : B script), one per line (2026-11-10). From
# SLURM_JOB_NODELIST_HET_GROUP_<n>, else from squeue's "<job>+<n> <nodelist>" line, so a renamed variable cannot
# strand the job. Prints nothing outside a heterogeneous job.
het_nodes() {
  local v="SLURM_JOB_NODELIST_HET_GROUP_$1" list
  list="${!v:-}"
  [[ -n "${list}" ]] || list="$(squeue -h -j "${SLURM_JOB_ID:?not inside a Slurm job}" -o "%i %N" 2>/dev/null |
                                 awk -v c="${SLURM_JOB_ID}+$1" '$1 == c {print $2; exit}')"
  [[ -n "${list}" ]] || return 0
  scontrol show hostnames "${list}"
}

# Run a command on another node of this job, in the repo with env.sh sourced. Arguments are quoted,
# so `on_node host env A=1 B=2 bash script.sh x` works. The site allows ssh to your job's own nodes.
on_node() {
  local host="$1"; shift
  local cmd; cmd="cd $(printf '%q' "${DPT_REPO}") && source scripts/slurm/env.sh && $(printf '%q ' "$@")"
  run ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${host}" "${cmd}"
}

# ---- Ray across the job's nodes (verl schedules all GPUs through it) -----------------------------
ray_up() {     # ray_up <gpus_per_node>
  local g="${1:?gpus per node}" head ip n
  head="$(job_head)"; ip="$(getent hosts "${head}" 2>/dev/null | awk '{print $1; exit}' || true)"; ip="${ip:-${head}}"
  n="$(job_nodes | wc -l)"
  run ray stop --force >/dev/null 2>&1 || true
  # the cluster's token (env.sh: RAY_AUTH_MODE=token); the command prints it, so its output goes nowhere
  dry || [[ -s "${HOME}/.ray/auth_token" ]] || ray get-auth-token --generate >/dev/null
  run ray start --head --port=6379 --num-gpus "${g}" --temp-dir "${RAY_TMPDIR}" --disable-usage-stats
  local h
  for h in $(job_nodes | tail -n +2); do
    on_node "${h}" ray start --address="${ip}:6379" --num-gpus "${g}" --disable-usage-stats
  done
  dry && return 0
  local want=$((g * n)) have=0 i
  for i in $(seq 1 60); do
    have="$(python - <<'PY' 2>/dev/null
import ray
ray.init(address="auto", logging_level="ERROR")
print(int(ray.cluster_resources().get("GPU", 0)))
PY
)" || have=0
    if [[ "${have}" -ge "${want}" ]]; then
      # Drivers started after this join the cluster: a bare ray.init() (verl's main_ppo) does not look in our
      # custom RAY_TMPDIR, so without this it started a one-node Ray of its own (3 of 24 GPUs, 2026-10-29).
      export RAY_ADDRESS="${ip}:6379"
      tlog "Ray up: ${n} nodes, ${have} GPUs (head ${head} ${ip}); RAY_ADDRESS=${RAY_ADDRESS}"; return 0
    fi
    sleep 5
  done
  tdie "Ray cluster has ${have} of ${want} GPUs after 5 min"
}

ray_down() {
  local h
  for h in $(job_nodes | tail -n +2); do on_node "${h}" ray stop --force >/dev/null 2>&1 || true; done
  run ray stop --force >/dev/null 2>&1 || true
  unset RAY_ADDRESS
}

# ---- endpoints written by serve.slurm (one file per served model) --------------------------------
sandbox_image() {   # pull the T2 sandbox image if it is missing. Compute nodes only: on login nodes
                    # `apptainer` is a stub that prints a notice and exits 0
  [[ -f "${DPT_SANDBOX_SIF}" ]] && return 0
  command -v apptainer >/dev/null || { tlog "no apptainer (module load apptainer)"; return 1; }
  mkdir -p "$(dirname "${DPT_SANDBOX_SIF}")"
  run apptainer pull "${DPT_SANDBOX_SIF}" docker://debian:bookworm-slim
  dry || [[ -f "${DPT_SANDBOX_SIF}" ]]
}

endpoint_url()   { cat "${DPT_ENDPOINTS}/${1}.url" 2>/dev/null; }     # http://host:port (no /v1)
endpoint_model() { cat "${DPT_ENDPOINTS}/${1}.model" 2>/dev/null; }
endpoint_up()    { local u; u="$(endpoint_url "$1")" && [[ -n "${u}" ]] && curl -sf --max-time 10 "${u}/v1/models" >/dev/null; }
wait_endpoint() {   # wait_endpoint <name> [timeout_s]: until serve.slurm has published <name> and it answers.
  # The default wait is $DPT_ENDPOINT_WAIT_S (0 = check once), so a job queued behind a serve job
  # (sbatch --dependency=after:<serve job>) can wait out the model load instead of failing.
  local t="${2:-${DPT_ENDPOINT_WAIT_S:-0}}" waited=0
  until endpoint_up "$1"; do
    (( waited >= t )) && return 1
    sleep 30; waited=$((waited + 30))
  done
}

wait_url() {   # wait_url http://host:port [timeout_s] [server_pid]: with a pid, stop as soon as the server exits
  local url="$1" t="${2:-1800}" pid="${3:-}" i=0
  dry && return 0
  until curl -sf --max-time 10 "${url}/v1/models" >/dev/null; do
    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then tdie "server ${url} exited before it came up (see above)"; fi
    sleep 10; i=$((i + 10)); [[ ${i} -lt ${t} ]] || tdie "server ${url} not up after ${t}s"
  done
}

# Read one field of `python -m dpt.launch.rl --describe` JSON.
json_field() { python -c 'import json,sys; v=json.loads(sys.argv[1])[sys.argv[2]]; print(str(v).lower() if isinstance(v,bool) else v)' "$1" "$2"; }
