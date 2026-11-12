"""The Slurm job scripts, run in dry-run mode (DPT_SLURM_DRY_RUN=1: remote commands are printed).

What must hold, because each failure costs SUs or silently corrupts a run:
  * a finished run does nothing; a PAUSED run is never resubmitted (a human must look first)
  * a normal job queues its successor first, starts Ray on every node, launches with a deadline
  * a job that ends without a clean stop (crash) cancels its successor: no resubmission loop
  * RL refuses to start without a judge, because T3 rewards would silently be 0
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _run(script, args, tmp_path, **env):
    e = {**os.environ, "DPT_SLURM_DRY_RUN": "1", "DPT_SLURM_NODES": "node01-001 node01-002 node01-003",
         "SLURM_SUBMIT_DIR": str(REPO), "SLURM_JOB_ID": "4242", "SLURM_JOB_ACCOUNT": "PROJ-123",
         "WORK": str(tmp_path / "work"), "SCRATCH": str(tmp_path / "scratch"),
         "DPT_CONDA": str(tmp_path / "no-conda"), "DPT_SANDBOX_SIF": str(tmp_path / "sbx.sif"),
         "PYTHONPATH": str(REPO / "src"), **env}
    e.pop("DPT_HW_PROFILE", None)
    e.pop("DPT_MODEL_LINE", None)
    return subprocess.run(["bash", str(REPO / "scripts" / "slurm" / script), *args], capture_output=True,
                          text=True, env=e, cwd=REPO, timeout=120)


def test_env_survives_modules_that_read_unset_variables(tmp_path):
    """The cluster's apptainer module reads an unset variable; under `set -u` that killed every job."""
    script = tmp_path / "job.sh"
    script.write_text("set -euo pipefail\n"
                      "module() { : \"${BASH_COMPLETION_DEBUG}\"; }\n"      # what Lmod's module load runs
                      f"source {REPO / 'scripts' / 'slurm' / 'env.sh'}\n"
                      "echo \"reached $DPT_MODEL_LINE\"\n"
                      "( : \"$UNSET_AFTER_ENV\" ) 2>/dev/null && echo nounset-off || echo nounset-on\n")
    e = {k: v for k, v in os.environ.items() if k not in ("DPT_MODEL_LINE", "BASH_COMPLETION_DEBUG")}
    r = subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=60,
                       env={**e, "WORK": str(tmp_path), "DPT_CONDA": str(tmp_path / "no-conda")})
    assert r.returncode == 0, r.stderr
    assert "reached qwen35_9b" in r.stdout and "nounset-on" in r.stdout      # `set -u` is restored


def test_env_on_a_login_node_allows_many_client_threads(tmp_path):
    """Login nodes: 100 MB thread stacks under an 8 GB virtual-memory cap allowed only ~48 threads."""
    fake = tmp_path / "bin"
    fake.mkdir()
    for host in ("login2", "node01-001"):
        (fake / "hostname").write_text(f"#!/bin/sh\necho {host}\n")
        (fake / "hostname").chmod(0o755)
        cmd = (f"ulimit -s 102400; source {REPO / 'scripts' / 'slurm' / 'env.sh'} >/dev/null 2>&1; "
               "echo \"$(ulimit -s) ${MALLOC_ARENA_MAX:-unset} ${OPENBLAS_NUM_THREADS:-unset}\"")
        e = {k: v for k, v in os.environ.items() if k not in ("MALLOC_ARENA_MAX", "OPENBLAS_NUM_THREADS")}
        r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=60,
                           env={**e, "PATH": f"{fake}:{os.environ['PATH']}", "WORK": str(tmp_path),
                                "DPT_CONDA": str(tmp_path / "no-conda")})
        expected = "8192 2 1" if host.startswith("login") else "102400 unset unset"
        assert r.stdout.split()[-3:] == expected.split(), (host, r.stdout, r.stderr)


def test_env_keeps_megatron_config_locks_off_nfs_home(tmp_path):
    """2026-10-28: Megatron-Bridge's config lock under ~/.cache (NFSv3) hung one node's SFT ranks for good."""
    cmd = f"source {REPO / 'scripts' / 'slurm' / 'env.sh'} >/dev/null 2>&1; echo \"$MEGATRON_CONFIG_LOCK_DIR\""
    e = {k: v for k, v in os.environ.items() if k != "MEGATRON_CONFIG_LOCK_DIR"}
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=60,
                       env={**e, "WORK": str(tmp_path), "DPT_CONDA": str(tmp_path / "no-conda")})
    lock_dir = r.stdout.strip().splitlines()[-1]
    assert lock_dir.startswith("/tmp/"), (lock_dir, r.stderr)


def test_env_finds_apptainer_on_a_node_reached_over_ssh(tmp_path):
    """2026-10-29: Lmod is stubbed in an ssh shell (LMOD_CMD=:), so `module load apptainer` did nothing there
    and RL's reward workers on 7 of 8 nodes scored 79% of the T2 samples 0 as sandbox infra errors."""
    fake, appt = tmp_path / "bin", tmp_path / "apptainer" / "bin"
    fake.mkdir()
    appt.mkdir(parents=True)
    (appt / "apptainer").write_text("#!/bin/sh\nexit 0\n")
    (appt / "apptainer").chmod(0o755)
    for host, expected in (("node01-001", str(appt / "apptainer")), ("login2", "none")):
        (fake / "hostname").write_text(f"#!/bin/sh\necho {host}\n")
        (fake / "hostname").chmod(0o755)
        cmd = ("module() { :; }; "                                   # what Lmod does in an ssh shell
               f"source {REPO / 'scripts' / 'slurm' / 'env.sh'} >/dev/null 2>&1; command -v apptainer || echo none")
        r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=60,
                           env={**os.environ, "PATH": f"{fake}:{os.environ['PATH']}", "DPT_APPTAINER_BIN": str(appt),
                                "WORK": str(tmp_path), "DPT_CONDA": str(tmp_path / "no-conda")})
        assert r.stdout.strip().splitlines()[-1] == expected, (host, r.stdout, r.stderr)


def test_sandbox_image_is_pulled_on_a_compute_node_only_when_missing(tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "apptainer").write_text("#!/bin/sh\necho real-apptainer\n")
    (fake / "apptainer").chmod(0o755)
    sif = tmp_path / "sandbox" / "sandbox.sif"
    cmd = f"source {REPO / 'scripts' / 'slurm' / 'lib.sh'}; sandbox_image && echo ok"
    env = {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}", "DPT_SANDBOX_SIF": str(sif)}
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, env={**env, "DPT_SLURM_DRY_RUN": "1"})
    assert "DRY: apptainer pull" in r.stdout and "docker://debian:bookworm-slim" in r.stdout and "ok" in r.stdout
    # a stub apptainer that writes nothing (the login nodes' notice) is a failure, not "ready"
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, env=env)
    assert r.returncode != 0 and "ok" not in r.stdout
    sif.write_text("image")
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, env=env)
    assert r.stdout.strip() == "ok"                                     # present: nothing pulled


@pytest.fixture()
def run_dir(tmp_path):
    d = tmp_path / "rl9b"
    d.mkdir()
    return d


def test_normal_job_chains_starts_ray_everywhere_and_cancels_successor_on_unclean_exit(tmp_path, run_dir):
    r = _run("rl.slurm", [f"rl_run.output_dir={run_dir}"], tmp_path, DPT_ALLOW_NO_JUDGE="1")
    out = r.stdout + r.stderr
    assert "DRY: sbatch --parsable --dependency=afterany:4242 -A PROJ-123" in out
    assert "DPT_CHAIN_LEFT=5" in out
    assert "DRY: ray start --head" in out
    # the two worker nodes join the head over ssh (the remote command is shell-escaped in dry-run output)
    assert out.count("--address=node01-001:6379") == 2
    assert "node01-002" in out and "node01-003" in out
    assert "DRY: python -m dpt.launch.rl --config rl/rl_9b.yaml rl_run.output_dir=" in out
    # the dry "training" leaves no WALLTIME.json: an unclean exit must cancel the successor
    assert "DRY: scancel" in out and r.returncode != 0


def test_finished_run_does_nothing(tmp_path, run_dir):
    (run_dir / "latest_checkpointed_iteration.txt").write_text("120")
    r = _run("rl.slurm", [f"rl_run.output_dir={run_dir}"], tmp_path, DPT_ALLOW_NO_JUDGE="1")
    assert r.returncode == 0 and "run finished" in r.stderr and "sbatch" not in r.stdout


def test_paused_run_is_never_resubmitted(tmp_path, run_dir):
    (run_dir / "latest_checkpointed_iteration.txt").write_text("72")
    (run_dir / "PAUSED.json").write_text(json.dumps({"step": 72, "reason": "logppl_diff"}))
    r = _run("rl.slurm", [f"rl_run.output_dir={run_dir}"], tmp_path, DPT_ALLOW_NO_JUDGE="1")
    assert r.returncode != 0 and "PAUSED" in r.stderr and "sbatch" not in r.stdout


def test_rl_refuses_to_start_without_a_judge(tmp_path, run_dir):
    r = _run("rl.slurm", [f"rl_run.output_dir={run_dir}"], tmp_path)
    assert r.returncode != 0 and "no judge endpoint" in r.stderr and "ray start" not in r.stdout


def test_chain_budget_stops_resubmission(tmp_path, run_dir):
    r = _run("rl.slurm", [f"rl_run.output_dir={run_dir}"], tmp_path, DPT_ALLOW_NO_JUDGE="1", DPT_CHAIN_LEFT="1")
    assert "sbatch" not in r.stdout and "DRY: python -m dpt.launch.rl" in r.stdout


def test_sft_starts_one_rank_per_node_with_the_same_master(tmp_path):
    r = _run("sft.slurm", [], tmp_path)
    out = r.stdout
    assert r.returncode == 0, r.stderr
    for rank in range(3):
        assert f"NODE_RANK={rank}" in out
    assert out.count("NNODES=3") == 3 and out.count("MASTER_ADDR=") == 3
    assert out.count("ssh -o BatchMode=yes") == 2                         # node 0 runs locally


def test_second_judge_needs_a_key_and_existing_results(tmp_path):
    out = _run("second_judge.sh", ["sft"], tmp_path, HOME=str(tmp_path), OPENAI_API_KEY="")
    assert out.returncode != 0 and "no OPENAI_API_KEY" in out.stderr
    key = tmp_path / ".config" / "dpt" / "openai_key"
    key.parent.mkdir(parents=True)
    key.write_text("sk-test\n")
    out = _run("second_judge.sh", ["sft"], tmp_path, HOME=str(tmp_path), OPENAI_API_KEY="")
    assert out.returncode != 0 and "evaluate sft first" in out.stderr     # key read from the file


def test_eval_sets_needs_drafts_never_overwrites_and_needs_the_judge(tmp_path):
    drafts, secure = tmp_path / "drafts", tmp_path / "secure"
    env = {"DPT_DRAFTS": str(drafts), "DPT_EVAL_SECURE_ROOT": str(secure), "DPT_ENDPOINTS": str(tmp_path / "ep")}
    out = _run("eval_sets.slurm", ["fresh"], tmp_path, **env)
    assert out.returncode != 0 and "dpt.eval.drafts fresh" in out.stderr
    drafts.mkdir()
    (drafts / "fresh_drafts.jsonl").write_text("{}\n")
    secure.mkdir()
    (secure / "d_fresh.jsonl").write_text("{}\n")
    out = _run("eval_sets.slurm", ["fresh"], tmp_path, **env)
    assert out.returncode != 0 and "frozen?" in out.stderr               # an existing set is never replaced
    (secure / "d_fresh.jsonl").unlink()
    out = _run("eval_sets.slurm", ["fresh"], tmp_path, **env)
    assert out.returncode != 0 and "no judge endpoint" in out.stderr


def test_judge_drift_needs_a_key_and_validation_dumps(tmp_path):
    out = _run("judge_drift.sh", [], tmp_path, HOME=str(tmp_path), OPENAI_API_KEY="")
    assert out.returncode != 0 and "no OPENAI_API_KEY" in out.stderr
    out = _run("judge_drift.sh", ["rl9b_seed1"], tmp_path, HOME=str(tmp_path), OPENAI_API_KEY="sk-test")
    assert out.returncode != 0 and "val_generations" in out.stderr


@pytest.mark.parametrize("script", sorted(p.name for p in (REPO / "scripts" / "slurm").iterdir()
                                          if p.suffix in (".slurm", ".sh")))
def test_scripts_parse(script):
    assert subprocess.run(["bash", "-n", str(REPO / "scripts" / "slurm" / script)]).returncode == 0


def test_wait_endpoint_waits_for_a_late_server_then_gives_up(tmp_path):
    ep = tmp_path / "ep"
    ep.mkdir()
    lib = REPO / "scripts" / "slurm" / "lib.sh"
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "curl").write_text("#!/bin/sh\nexit 0\n")           # the server answers once its URL is published
    (fake / "curl").chmod(0o755)
    (fake / "sleep").write_text(f"#!/bin/sh\necho http://h:8000 > {ep}/judge.url\n")   # appears while waiting
    (fake / "sleep").chmod(0o755)
    env = {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}", "DPT_ENDPOINTS": str(ep)}
    r = subprocess.run(["bash", "-c", f"source {lib}; wait_endpoint judge 60 && echo up"], capture_output=True,
                       text=True, env=env)
    assert r.stdout.strip() == "up"
    (ep / "judge.url").unlink()
    (fake / "sleep").write_text("#!/bin/sh\nexit 0\n")
    r = subprocess.run(["bash", "-c", f"source {lib}; wait_endpoint judge 60 || echo gave-up"], capture_output=True,
                       text=True, env=env)
    assert r.stdout.strip() == "gave-up"
    r = subprocess.run(["bash", "-c", f"source {lib}; wait_endpoint judge || echo once"], capture_output=True,
                       text=True, env=env)
    assert r.stdout.strip() == "once"                               # default: check once, no wait


def test_env_activation_hook_survives_strict_mode_before_cudnn_is_installed(tmp_path):
    """The first cluster build died here: `ls` of the not-yet-installed cuDNN dir failed under pipefail."""
    src = (REPO / "scripts" / "env" / "build_train_local.sh").read_text()
    start = src.index('activate.d/cuda_compat.sh" <<\'HOOK\'\n') + len('activate.d/cuda_compat.sh" <<\'HOOK\'\n')
    hook = src[start:src.index("\nHOOK\n", start)]
    (tmp_path / "hook.sh").write_text(hook)
    (tmp_path / "cuda-compat").mkdir()
    r = subprocess.run(["bash", "-c", f"set -euo pipefail; . {tmp_path / 'hook.sh'}; echo \"ok $LD_LIBRARY_PATH\""],
                       capture_output=True, text=True, env={**os.environ, "CONDA_PREFIX": str(tmp_path),
                                                            "LD_LIBRARY_PATH": "/usr/lib64"})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == f"ok {tmp_path / 'cuda-compat'}:/usr/lib64"


def test_wait_url_stops_as_soon_as_the_server_process_exits(tmp_path):
    """The first teacher server died while loading and the job then waited its full hour."""
    lib = REPO / "scripts" / "slurm" / "lib.sh"
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "curl").write_text("#!/bin/sh\nexit 7\n")              # never answers
    (fake / "curl").chmod(0o755)
    cmd = f"source {lib}; (exit 3) & pid=$!; wait $pid; wait_url http://h:8000 3600 $pid"
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=60,
                       env={**os.environ, "PATH": f"{fake}:{os.environ['PATH']}"})
    assert r.returncode != 0 and "exited before it came up" in r.stderr


def test_qwen_hybrid_models_are_served_on_the_triton_gdn_kernels():
    src = (REPO / "scripts" / "serve_vllm.sh").read_text()
    assert "--gdn-prefill-backend triton" in src and '"${GDN_ARGS[@]}"' in src


def _campaign_inputs(tmp_path, bank=True):
    """The sandbox image, the RL judge job's id and (optionally) the published RL bank."""
    (tmp_path / "sbx.sif").write_text("image")          # the sandbox image is there (check_node pulled it)
    campaign = tmp_path / "work" / "dpt" / "campaign"
    campaign.mkdir(parents=True, exist_ok=True)
    (campaign / "judge_job").write_text("777\n")
    v1 = tmp_path / "work" / "dpt" / "data" / "v1"
    v1.mkdir(parents=True, exist_ok=True)
    for name in ("rl_main", "rl_hard", "rl_dev"):
        if bank:
            (v1 / f"{name}.parquet").write_text("")
    return campaign


def test_campaign_runs_every_step_once_in_order(tmp_path):
    nodes = " ".join(f"node01-00{i}" for i in range(1, 9))
    _campaign_inputs(tmp_path)
    r = _run("campaign.sh", [], tmp_path, DPT_SLURM_NODES=nodes)
    assert r.returncode == 0, r.stderr[-3000:]
    out = r.stdout + r.stderr
    order = [r.stdout.index(k) for k in ("scripts/03_baseline_eval.sh", "scripts/07_final_eval.sh", "dpt.eval.gates",
                                         "dpt.launch.rl --config rl/rl_9b.yaml")]          # the DRY commands
    assert order == sorted(order)
    steps = [r.stderr.index(f"campaign: {k} done") for k in ("baseline", "sft_data", "sft", "sft_eval", "rl_data")]
    assert steps == sorted(steps)
    # 3 judge replicas on the head node + 21 model replicas on the workers, per eval step, behind two proxies
    assert out.count("scripts/serve_vllm.sh") == 2 * 24
    assert out.count("rr_proxy.py --port 8000") == 2 and out.count("rr_proxy.py --port 8100") == 2
    logs = tmp_path / "work" / "dpt" / "campaign" / "logs" / "4242"
    sft = [(logs / f"sft_node{i}.log").read_text().replace("\\ ", " ") for i in range(8)]   # one rank per node
    assert all("scripts/04_sft.sh --config sft/sft_9b.yaml" in t and f"NODE_RANK={i}" in t for i, t in enumerate(sft))
    assert "--entry" in out
    assert "hold at step 32" in out                    # no judge endpoint: stage A only
    state = tmp_path / "work" / "dpt" / "campaign" / "state"
    assert {p.name for p in state.iterdir()} == {"baseline", "sft_data", "sft", "sft_eval", "rl_data"}
    assert "DRY: scancel 777" in r.stdout                 # no rl.slurm successor queued: the judge is released
    again = _run("campaign.sh", [], tmp_path, DPT_SLURM_NODES=nodes)
    assert again.returncode == 0 and "scripts/03_baseline_eval.sh" not in again.stdout + again.stderr
    (state / "GATE_FAILED").write_text("")
    stop = _run("campaign.sh", [], tmp_path, DPT_SLURM_NODES=nodes)
    assert stop.returncode != 0 and "RL-entry gate failed" in stop.stderr
    assert "DRY: scancel 777" in stop.stdout              # a failed campaign does not leave its judge idling


def test_campaign_runs_the_bank_pass_itself_while_the_bank_is_unpublished(tmp_path):
    """RL fails without the published bank. Its difficulty pass takes ~1.5 days on a one-GPU base job and ~2 h on
    the campaign's 24 GPUs. A login-node runner doing it must stop first: two clients appending to one cache
    duplicate work and can interleave lines."""
    nodes = " ".join(f"node01-00{i}" for i in range(1, 9))
    campaign = _campaign_inputs(tmp_path, bank=False)
    (campaign / "base_job").write_text("888\n")
    state = campaign / "state"
    state.mkdir()
    for step in ("baseline", "sft_data", "sft", "sft_eval"):
        (state / step).write_text("")
    r = _run("campaign.sh", [], tmp_path, DPT_SLURM_NODES=nodes)
    out = r.stdout + r.stderr
    assert (campaign / "bank_takeover").exists() and "DRY: scancel 888" in r.stdout    # the base job is released
    passes = [line for line in r.stdout.splitlines() if "scripts/slurm/data.slurm bank" in line]
    assert len(passes) == 2                          # at the start and once more before RL (dry: never published)
    assert "DPT_DIFFICULTY_CONCURRENCY=576" in passes[0] and "DPT_BANK_STAGES=difficulty" in passes[0]
    assert (campaign / "endpoints" / "4242" / "base.url").read_text() == "http://127.0.0.1:8000\n"
    assert out.count("scripts/serve_vllm.sh") == 2 * 24 and out.count("rr_proxy.py --port 8000") == 2
    assert r.returncode != 0 and "no RL bank" in r.stderr and "dpt.launch.rl" not in r.stdout
    assert "DRY: scancel 777" in r.stdout                 # the RL judge is released too
    (campaign / "bank_runner.alive").write_text("")       # a runner that is alive and has not answered
    busy = _run("campaign.sh", [], tmp_path, DPT_SLURM_NODES=nodes, DPT_CAMPAIGN_RUNNER_WAIT_S="0")
    assert busy.returncode != 0 and "difficulty runner is still running" in busy.stderr
    assert "scripts/slurm/data.slurm" not in busy.stdout and "scripts/serve_vllm.sh" not in busy.stdout + busy.stderr
    os.utime(campaign / "bank_runner.alive", (time.time() - 600,) * 2)    # its heartbeat stopped 10 min ago
    gone = _run("campaign.sh", [], tmp_path, DPT_SLURM_NODES=nodes, DPT_CAMPAIGN_RUNNER_WAIT_S="0")
    assert "scripts/slurm/data.slurm bank" in gone.stdout
    _campaign_inputs(tmp_path)                            # published
    ok = _run("campaign.sh", [], tmp_path, DPT_SLURM_NODES=nodes)
    assert ok.returncode == 0, ok.stderr[-3000:]
    assert "scripts/slurm/data.slurm" not in ok.stdout and "campaign: rl_data done" in ok.stderr
    assert "dpt.launch.rl --config rl/rl_9b.yaml" in ok.stdout


def test_job_id_skips_the_submit_banner():
    # The cluster's submit filter prints its banner on stdout even with sbatch --parsable (2026-10-25)
    banner = "-----\n   Welcome to the GPU cluster\n-----\n--> Verifying valid submit host (login1)...OK\n"
    for out, want in ((banner + "3533978\n", "3533978"), ("123;cluster\n", "123")):
        r = subprocess.run(["bash", "-c", f"source {REPO}/scripts/slurm/lib.sh; job_id"], input=out,
                           capture_output=True, text=True)
        assert r.stdout == want + "\n"


def test_campaign_slurm_waits_for_the_ready_marker(tmp_path):
    r = _run("campaign.slurm", [], tmp_path, DPT_CAMPAIGN_READY_WAIT_S="0")
    assert r.returncode != 0 and "not ready" in r.stdout + r.stderr


def _plain(out):
    """The dry-run lines with printf %q's backslashes removed, so remote commands read as typed."""
    return [line.replace("\\", "") for line in out.splitlines()]


def _model_servers(out, row, max_model_len):
    """The serve_vllm.sh lines of one row at one max_model_len (40960: the 32K eval; 67584: the 64K continuations)."""
    return [line for line in _plain(out) if "serve_vllm.sh" in line and f"'{row}' " in line
            and f" {max_model_len} 1 " in line]


def _served(lines, host, name):
    return [line for line in lines if f"StrictHostKeyChecking=no {host} " in line and "serve_vllm.sh" in line
            and f"' {name} " in line]


def test_data_window_runs_every_server_and_client_on_its_own_two_nodes(tmp_path):
    """Bank v2's data phase as a 2-node job (window 1): the teacher (FP8, TP2) and Gemma on node 1, the SFT model ×3
    on node 2, every client and proxy on node 1. In phase B node 1 serves the SFT model too, and the SFT proxy,
    which listed node 1's replicas from the start, sends them work (2026-11-11)."""
    r = _run("data_window.slurm", [], tmp_path, DPT_SLURM_NODES="node01-001 node01-002")
    assert r.returncode == 0, r.stderr
    out, lines = r.stdout, _plain(r.stdout)
    teacher = _served(lines, "node01-001", "teacher")
    assert len(teacher) == 1 and "CUDA_VISIBLE_DEVICES=0,1" in teacher[0] and "teacher 8010 40960 2" in teacher[0]
    judge = _served(lines, "node01-001", "judge_v")
    assert len(judge) == 1 and "CUDA_VISIBLE_DEVICES=2" in judge[0] and "DPT_SERVE_DP=1" in judge[0]
    assert len(_served(lines, "node01-002", "sft")) == 3
    assert all("DPT_GDN_MODEL=1" in line for line in _served(lines, "node01-002", "sft"))
    assert "--port 8000 --backend http://node01-001:8010 " in out and "--port 8100 --backend http://node01-001:8110 " in out
    sft_proxy = next(line for line in lines if "--port 8200" in line)
    assert sft_proxy.count("--backend") == 6 and "http://node01-001:8003" in sft_proxy     # node 1's replicas, for phase B
    for env in ("DPT_BANK_NAME=hard", "DPT_BANK_NAME=long", "DPT_BANK_NAME=v2a", "DPT_BANK_NAME=v2new",
                "DPT_BANK_NAME=v2pub", "t3_atomic.slurm"):
        assert env in out, env
    node1_sft = [i for i, line in enumerate(lines) if line in _served(lines, "node01-001", "sft")]
    assert len(node1_sft) == 3 and min(node1_sft) > out.count("\n", 0, out.index("DPT_BANK_NAME=hard"))
    assert out.index("DPT_BANK_NAME=v2new") > out.index("DPT_BANK_NAME=hard")      # phase B after phase A
    assert "DPT_DIFFICULTY_CONCURRENCY=96 " in out and "DPT_DIFFICULTY_CONCURRENCY=192 " in out
    assert "--name rl_bank.difficulty" in out and "--t3-map" in out and "bank_merge check" in out
    assert "--t3-not-trained" not in out
    pin = next(line for line in lines if "bank_merge pin-dev" in line)                 # run 1's dev split, converted
    assert "/v1/rl_dev.jsonl" in pin and "--t3-map" in pin
    assert out.index("DPT_BANK_NAME=v2pub") < out.index("bank_merge pin-dev") < out.index("bank_merge check")


def test_data_window_on_eight_nodes_with_two_teacher_nodes_and_no_t3(tmp_path):
    """Inside RL v2's job (2026-11-11): 2 teacher nodes behind one teacher proxy, 6 nodes of the SFT model, no atomic
    T3 rubrics, and both difficulty passes sized to the replicas they use."""
    nodes = " ".join(f"node01-00{i}" for i in range(1, 9))
    r = _run("data_window.slurm", [], tmp_path, DPT_SLURM_NODES=nodes, DPT_DATA_TEACHER_NODES="2", DPT_DATA_SKIP_T3="1")
    assert r.returncode == 0, r.stderr
    out, lines = r.stdout, _plain(r.stdout)
    for h in ("node01-001", "node01-002"):
        assert len(_served(lines, h, "teacher")) == 1 and len(_served(lines, h, "judge_v")) == 1
        assert len(_served(lines, h, "sft")) == 3                                 # in phase B
    for i in range(3, 9):
        assert len(_served(lines, f"node01-00{i}", "sft")) == 3 and not _served(lines, f"node01-00{i}", "teacher")
    assert next(line for line in lines if "--port 8000" in line).count("--backend") == 2
    assert next(line for line in lines if "--port 8200" in line).count("--backend") == 24
    assert "t3_atomic.slurm" not in out and "--t3-map" not in out and "--t3-not-trained" in out
    assert "bank_merge pin-dev" in out
    assert "DPT_DIFFICULTY_CONCURRENCY=576 " in out and "DPT_DIFFICULTY_CONCURRENCY=768 " in out


def test_data_window_needs_a_node_for_the_sft_model(tmp_path):
    r = _run("data_window.slurm", [], tmp_path, DPT_SLURM_NODES="node01-001")
    assert r.returncode != 0 and "needs 2 nodes" in r.stderr
    r = _run("data_window.slurm", [], tmp_path, DPT_SLURM_NODES="node01-001 node01-002", DPT_DATA_TEACHER_NODES="2")
    assert r.returncode != 0 and "needs 3 nodes" in r.stderr


def test_final_eval_window_evaluates_rl_v2_on_the_rl_nodes(tmp_path):
    """RL v2's final eval as a job after the RL job (2026-11-09): judge ×3 + sandbox on the head node, the model ×21
    on the 7 workers, every set and D-Num-hard for row rl_v2_seed1, then the report with it as the gate candidate."""
    nodes = " ".join(f"node01-00{i}" for i in range(1, 9))
    (tmp_path / "sbx.sif").write_text("")                               # the sandbox image, pulled on a compute node
    r = _run("final_eval_window.slurm", [], tmp_path, DPT_SLURM_NODES=nodes, DPT_CANCEL_JOB="777")
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "scancel 777" in out and "dpt.rewards.sandbox_server" in out and "sbx.sif" in out
    assert "rl9b_v2_seed1/global_step_120" in out and "export_hf.sh" in out
    assert len(_model_servers(out, "rl_v2_seed1", 40960)) == 21                 # the 32K eval on the 7 workers
    assert len(_model_servers(out, "rl_v2_seed1", 67584)) == 24                 # 64K continuations on all 8 nodes
    for s_ in ("d_num_hard", "d_fresh"):
        assert f"dpt.eval.budget continue --set {s_} --row rl_v2_seed1" in out
    assert ("dpt.eval.budget compare --rows sft,rl_seed1,rl_v2_seed1,sft_v2_seed1,rl_v3_seed1,base_medium "
            "--baseline sft") in "\n".join(_plain(out))
    assert out.index("budget compare") < out.index("07_final_eval.sh --report-only")
    assert sum(1 for line in out.splitlines() if "serve_vllm.sh" in line and "judge" in line) == 3
    assert "--sets d_num\\,d_code\\,d_report\\,d_fresh\\,general\\,refusal" in out and "--sets d_num_hard" in out
    assert "--samples-override 64" not in out                           # k64 only when asked for
    assert "DPT_FINAL_CANDIDATE=rl_v2_seed1" in out and "--report-only" in out


def _reservation(tmp_path, nodes):
    """A fake `scontrol show reservation` for a 24 h window that starts in 2 h, with <nodes> nodes."""
    fake = tmp_path / "bin"
    fake.mkdir(exist_ok=True)
    s = fake / "scontrol"
    s.write_text("#!/bin/bash\n"
                 "S=$(date -d '+2 hours' +%Y-%m-%dT%H:%M:%S); E=$(date -d '+26 hours' +%Y-%m-%dT%H:%M:%S)\n"
                 f"echo \"ReservationName=$3 StartTime=$S EndTime=$E Duration=1-00:00:00 Nodes=node01-[001-00{nodes}] "
                 f"NodeCnt={nodes} PartitionName=gpu-a100 Users=user Accounts=PROJ State=INACTIVE\"\n")
    s.chmod(0o755)
    return f"{fake}:{os.environ['PATH']}"


def _bank_check(tmp_path, problems):
    p = tmp_path / "work" / "dpt" / "data" / "v2" / "bank_check.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"problems": problems}))


def test_submit_window_data_fits_the_reservation(tmp_path):
    """Window 1 (2026-11-09): the data job's limit ends 10 min before the reservation does."""
    r = _run("submit_window.sh", ["data", "dpt_data"], tmp_path, PATH=_reservation(tmp_path, 2))
    assert r.returncode == 0, r.stderr
    assert "--reservation=dpt_data" in r.stdout and "-t 23:50:00" in r.stdout and "data_window.slurm" in r.stdout


def test_submit_window_rl_chains_judge_rl_and_final_eval(tmp_path):
    """Window 2: the judge to the end, RL 4 h less, the final eval after RL with the judge's id to cancel."""
    _bank_check(tmp_path, [])
    r = _run("submit_window.sh", ["rl", "dpt_rl"], tmp_path, PATH=_reservation(tmp_path, 9))
    assert r.returncode == 0, r.stderr
    judge, rl, ev = [line for line in r.stdout.splitlines() if line.startswith("DRY: sbatch")]
    assert "-t 23:50:00" in judge and "DPT_SERVE_REPLICAS=3" in judge and "serve.slurm judge" in judge
    assert "gemma-4-26B-A4B-it-FP8-dynamic" in judge
    assert "-t 19:50:00" in rl and "DPT_CHAIN_LEFT=1" in rl and "DPT_JUDGE_WAIT_S=3600" in rl
    assert "rl/rl_9b_v2.yaml" in rl
    assert "-t 4:00:00" in ev and "afterany:9002" in ev and "DPT_CANCEL_JOB=9001" in ev
    assert "final_eval_window.slurm" in ev
    assert all("--reservation=dpt_rl" in line for line in (judge, rl, ev))


def test_submit_window_rl_waits_for_a_passing_bank_check(tmp_path):
    path = _reservation(tmp_path, 9)
    r = _run("submit_window.sh", ["rl", "dpt_rl"], tmp_path, PATH=path)
    assert r.returncode != 0 and "window 1 (data) has not finished" in r.stderr and "sbatch" not in r.stdout
    _bank_check(tmp_path, ["only 12 T1 main problems at 5/8 or below"])
    r = _run("submit_window.sh", ["rl", "dpt_rl"], tmp_path, PATH=path)
    assert r.returncode != 0 and "only 12 T1 main problems" in r.stderr and "sbatch" not in r.stdout


def test_submit_window_rl_needs_nine_nodes(tmp_path):
    _bank_check(tmp_path, [])
    r = _run("submit_window.sh", ["rl", "dpt_rl"], tmp_path, PATH=_reservation(tmp_path, 8))
    assert r.returncode != 0 and "needs 9 nodes" in r.stderr


def test_rl_fallback_runs_judge_rl_and_final_eval_in_one_job(tmp_path):
    """RL v2 without a reservation (2026-11-09): Gemma ×3 on node 9; RL on nodes 1-8 with no successor and the final
    eval's hours kept free; then the final eval on all 9 nodes (the model ×24 on nodes 2-9), cancelling nothing."""
    _bank_check(tmp_path, [])
    (tmp_path / "sbx.sif").write_text("")
    nodes = " ".join(f"node01-00{i}" for i in range(1, 10))
    r = _run("rl_fallback.slurm", [], tmp_path, DPT_SLURM_NODES=nodes, DPT_CANCEL_JOB="777")
    out, err = r.stdout, r.stderr
    lines = out.splitlines()
    judge = [line for line in lines if "serve_vllm.sh" in line and "gemma-4-26B-A4B-it-FP8-dynamic" in line]
    assert len(judge) == 6                                              # RL's judge on node 9, the eval's on node 1
    assert sum("node01-009" in line for line in judge) == 3 and sum("node01-001" in line for line in judge) == 3
    assert any("node01-009" in line and "rr_proxy.py" in line for line in lines)
    assert "DRY: ray start --head" in out and out.count("--address=node01-001:6379") == 7   # Ray on the 8 RL nodes
    assert not any("node01-009" in line and "--address=" in line for line in lines)
    assert "DRY: python -m dpt.launch.rl --config rl/rl_9b_v2.yaml" in out and "sbatch" not in out
    assert "(15600 s before the job's limit)" in err                    # 4 h of final eval + 20 min
    models = _model_servers(out, "rl_v2_seed1", 40960)
    assert len(models) == 24 and not any("node01-001 " in line for line in models)
    assert out.index("DRY: python -m dpt.launch.rl") < out.index("dpt.rewards.sandbox_server")   # eval after RL
    assert "scancel" not in out and "DPT_FINAL_CANDIDATE=rl_v2_seed1" in out
    assert "final eval: done" in err
    assert r.returncode != 0                     # the dry "training" makes no clean stop, as for rl.slurm alone


def test_rl_fallback_refuses_without_a_passing_bank_or_a_judge_node(tmp_path):
    nodes = " ".join(f"node01-00{i}" for i in range(1, 10))
    r = _run("rl_fallback.slurm", [], tmp_path, DPT_SLURM_NODES=nodes)
    assert r.returncode != 0 and "build bank v2 first" in r.stderr and "serve_vllm" not in r.stdout
    _bank_check(tmp_path, [])
    r = _run("rl_fallback.slurm", [], tmp_path, DPT_SLURM_NODES=" ".join(nodes.split()[:8]))
    assert r.returncode != 0 and "needs 8 RL nodes and a judge node" in r.stderr


def test_rl_fallback_with_a_one_gpu_judge_node(tmp_path):
    """The judge component on gpu-a100-small (one A100): Gemma ×1, and the final eval on the 8 RL nodes only."""
    _bank_check(tmp_path, [])
    (tmp_path / "sbx.sif").write_text("")
    nodes = " ".join([f"node01-00{i}" for i in range(1, 9)] + ["node930-002"])
    r = _run("rl_fallback.slurm", [], tmp_path, DPT_SLURM_NODES=nodes, DPT_JUDGE_NODE_GPUS="1")
    lines = r.stdout.splitlines()
    judge = [line for line in lines if "serve_vllm.sh" in line and "gemma-4-26B-A4B-it-FP8-dynamic" in line]
    assert sum("node930-002" in line for line in judge) == 1 and sum("node01-001" in line for line in judge) == 3
    models = _model_servers(r.stdout, "rl_v2_seed1", 40960)
    assert len(models) == 21 and not any("node930-002" in line for line in models)
    assert "final eval: done" in r.stderr


def test_submit_window_queue_modes_need_no_reservation(tmp_path):
    r = _run("submit_window.sh", ["data-queue"], tmp_path)
    assert r.returncode == 0, r.stderr
    assert "DRY: sbatch -A PROJ scripts/slurm/data_window.slurm" in r.stdout and "reservation" not in r.stdout
    r = _run("submit_window.sh", ["data-queue", "-t", "48:00:00"], tmp_path)        # sbatch options pass through
    assert "DRY: sbatch -A PROJ -t 48:00:00 scripts/slurm/data_window.slurm" in r.stdout
    r = _run("submit_window.sh", ["rl-queue"], tmp_path)
    assert r.returncode != 0 and "window 1 (data) has not finished" in r.stderr and "sbatch" not in r.stdout
    _bank_check(tmp_path, [])
    r = _run("submit_window.sh", ["rl-queue"], tmp_path)
    assert r.returncode == 0, r.stderr
    # one heterogeneous job: the cluster refuses more than 8 gpu-a100 nodes per job (QOSMaxNodePerJobLimit, 2026-11-10)
    assert ("DRY: sbatch -A PROJ -p gpu-a100 -N 8 -t 24:00:00 : -A PROJ -p gpu-a100 -N 1 -t 24:00:00 "
            "scripts/slurm/rl_fallback.slurm") in r.stdout
    r = _run("submit_window.sh", ["rl-queue"], tmp_path, DPT_JUDGE_PARTITION="gpu-a100-small", DPT_FB_TIME="30:00:00")
    assert ": -A PROJ -p gpu-a100-small -N 1 -t 30:00:00 scripts/slurm/rl_fallback.slurm" in r.stdout


def test_submit_window_rl_build_queue_needs_no_bank_yet(tmp_path):
    """RL v2 as one self-contained job (2026-11-11): queued before bank v2 exists, because the job builds it."""
    r = _run("submit_window.sh", ["rl-build-queue"], tmp_path)
    assert r.returncode == 0, r.stderr
    # one plain 8-node job: gpu-a100's QOS counts the parts of a heterogeneous job together (refused at 8 + 1)
    assert "DRY: sbatch -A PROJ -p gpu-a100 -N 8 -t 48:00:00 scripts/slurm/rl_v2_queue.slurm" in r.stdout
    assert " : " not in r.stdout.split("DRY: sbatch", 1)[1].split("\n", 1)[0]       # no second job part
    r = _run("submit_window.sh", ["rl-build-queue", "-t", "1:00:00"], tmp_path)
    assert r.returncode != 0 and "takes no sbatch options" in r.stderr


EIGHT = " ".join(f"node01-00{i}" for i in range(1, 9))


def _fake_squeue(tmp_path, window_1_ids=""):
    """squeue that lists window 1 (dpt-datawin) for `-n dpt-datawin` and nothing else."""
    fake = tmp_path / "bin"
    fake.mkdir(exist_ok=True)
    (fake / "squeue").write_text('#!/bin/bash\n[[ " $* " == *" -n dpt-datawin "* ]] && '
                                 f'printf "%s" "{window_1_ids}"\nexit 0\n')
    (fake / "squeue").chmod(0o755)
    return f"{fake}:{os.environ['PATH']}"


def test_rl_v2_queue_waits_for_the_go_ahead_and_then_runs_the_job_script(tmp_path):
    """The batch script is frozen at submission, so it only hands over to rl_v2_job.sh, read when the job starts."""
    text = (REPO / "scripts" / "slurm" / "rl_v2_queue.slurm").read_text()
    assert "rl_v2_job.ready" in text and "exec bash scripts/slurm/rl_v2_job.sh" in text
    assert "#SBATCH -t 48:00:00" in text and "#SBATCH -N 8" in text
    assert "placeholder" not in (REPO / "scripts" / "slurm" / "rl_v2_job.sh").read_text()
    r = _run("rl_v2_queue.slurm", [], tmp_path, DPT_SLURM_NODES="node01-001 node01-002")   # a dry run skips the wait
    assert r.returncode != 0 and "RL v2 needs 8 nodes" in r.stderr


def test_rl_v2_job_builds_the_bank_tests_24k_trains_and_evaluates(tmp_path):
    """RL v2's job with no bank yet (2026-11-11): window 1 is stopped, the bank is built on all 8 nodes (2 teacher
    nodes, no T3), the dev split loses its T3 rows, the 24K test runs one forced full-length step, RL v2 runs at
    the cap it chose without a judge, and the final eval runs on the same nodes."""
    (tmp_path / "sbx.sif").write_text("")
    r = _run("rl_v2_job.sh", [], tmp_path, DPT_SLURM_NODES=EIGHT, PATH=_fake_squeue(tmp_path, "3571166"))
    out = r.stdout
    # a dry run of rl.slurm never records a clean stop, so RL "ends" with exit 1 and the final eval still runs
    assert "final eval: done" in r.stderr and "done: RL exit 1, final eval finished" in r.stderr, r.stderr
    assert "DRY: scancel 3571166" in out
    assert out.index("scancel 3571166") < out.index("DPT_BANK_NAME=hard")       # stopped before the build starts
    assert "--t3-not-trained" in out and "t3_atomic.slurm" not in out
    assert "bank_merge without-task --task t3_report" in out and "rl_dev_no_t3" in out
    test = next(line for line in out.splitlines() if "rl9b_v2_seed1_memtest24k" in line and "dpt.launch.rl" in line)
    for arg in ("project.thinking.rl_max_response=24576", "verl.actor_rollout_ref.model.mtp.enable_train=false",
                "verl.actor_rollout_ref.rollout.ignore_eos=true", "rl.filtering.drop_zero_advantage_groups=false",
                "rl.prompts_per_step=6", "rl.total_steps=1", "rl_run.save_freq=-1"):
        assert arg in test, arg
    train = [line for line in out.splitlines() if line.startswith("DRY: python -m dpt.launch.rl --config rl/rl_9b_v2.yaml")
             and "memtest" not in line]
    assert len(train) == 1 and "project.thinking.rl_max_response=24576" in train[0]   # dry: the test "passes"
    assert out.index("rl9b_v2_seed1_memtest24k") < out.index(train[0]) < out.index("rl9b_v2_seed1/global_step_120")
    assert "sbatch" not in out                                                     # RL has no successor job
    assert len(_model_servers(out, "rl_v2_seed1", 40960)) == 21
    assert len(_model_servers(out, "rl_v2_seed1", 67584)) == 24 and "budget compare" in out


def test_rl_v2_job_with_a_ready_bank_and_a_chosen_cap_goes_straight_to_rl(tmp_path):
    """A bank that passed its check is used as it is (window 1 keeps running), and a run that already chose its cap
    keeps it, so a second job resumes the same run the same way."""
    (tmp_path / "sbx.sif").write_text("")
    _bank_check(tmp_path, [])
    cap = tmp_path / "scratch" / "dpt" / "checkpoints" / "rl" / "rl9b_v2_seed1.cap.json"
    cap.parent.mkdir(parents=True)
    cap.write_text(json.dumps({"cap": "16k"}))
    r = _run("rl_v2_job.sh", [], tmp_path, DPT_SLURM_NODES=EIGHT, PATH=_fake_squeue(tmp_path, "3571166"))
    assert "final eval: done" in r.stderr, r.stderr
    out = r.stdout
    assert "scancel" not in out and "DPT_BANK_NAME" not in out and "memtest" not in out
    train = [line for line in out.splitlines() if line.startswith("DRY: python -m dpt.launch.rl --config")]
    assert len(train) == 1 and "rl_max_response" not in train[0]                   # 16K, as chosen before
    assert "the run already chose its cap: 16k" in r.stderr


def test_rl_v2_job_repairs_a_bank_window_1_published_before_building_one(tmp_path):
    """Window 1 runs its frozen copy of data_window.slurm (no pin-dev), so its bank fails the dev-split check. The RL
    job first puts back run 1's dev split on the CPU, and builds only if the check still fails."""
    (tmp_path / "sbx.sif").write_text("")
    _bank_check(tmp_path, ["dev split differs from v1/rl_dev.jsonl (61 vs 190 rows)"])
    done = tmp_path / "work" / "dpt" / "data" / "gen" / ".done"
    done.mkdir(parents=True)
    (done / "v2pub_publish").write_text("")
    (tmp_path / "work" / "dpt" / "data" / "v2" / "t3_atomic_map.jsonl").write_text("")
    r = _run("rl_v2_job.sh", [], tmp_path, DPT_SLURM_NODES=EIGHT, PATH=_fake_squeue(tmp_path, ""))
    out = r.stdout
    repair = next(line for line in out.splitlines() if "bank_merge pin-dev" in line)
    assert "--t3-map" in repair and "/v1/rl_dev.jsonl" in repair
    assert out.index(repair) < out.index("DPT_BANK_NAME=hard")      # the dry check cannot pass, so it builds next


def test_rl_v2_job_refuses_other_node_counts_and_unknown_caps(tmp_path):
    r = _run("rl_v2_job.sh", [], tmp_path, DPT_SLURM_NODES="node01-001 node01-002 node01-003")
    assert r.returncode != 0 and "needs 8 nodes" in r.stderr
    _bank_check(tmp_path, [])
    r = _run("rl_v2_job.sh", [], tmp_path, DPT_SLURM_NODES=EIGHT, DPT_RL_V2_CAP="32k")
    assert r.returncode != 0 and "unknown cap '32k'" in r.stderr


def test_het_nodes_reads_the_component_from_the_variable_or_from_squeue(tmp_path):
    """rl_fallback.slurm finds its RL and judge nodes (2026-11-10): Slurm 23.11 names the variable
    SLURM_JOB_NODELIST_HET_GROUP_<n>; if that name ever changes, squeue's "<job>+<n> <nodes>" lines still work."""
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "squeue").write_text('#!/bin/bash\nprintf "777+0 a1,a2\\n777+1 b1\\n"\n')
    (fake / "scontrol").write_text('#!/bin/bash\n[[ "$1 $2" == "show hostnames" ]] && tr "," "\\n" <<< "$3"\n')
    for f in ("squeue", "scontrol"):
        (fake / f).chmod(0o755)
    env = {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}", "SLURM_JOB_ID": "777"}
    env.pop("SLURM_JOB_NODELIST_HET_GROUP_1", None)
    cmd = f"source {REPO / 'scripts' / 'slurm' / 'lib.sh'}; het_nodes 0 | paste -sd' '; het_nodes 1; het_nodes 2; echo end"
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, env=env)
    assert r.stdout.split("\n")[:3] == ["a1 a2", "b1", "end"], r.stdout + r.stderr
    r = subprocess.run(["bash", "-c", f"source {REPO / 'scripts' / 'slurm' / 'lib.sh'}; het_nodes 1"],
                       capture_output=True, text=True, env={**env, "SLURM_JOB_NODELIST_HET_GROUP_1": "z9"})
    assert r.stdout.strip() == "z9"


def test_numhard_64k_serves_each_row_on_both_nodes_and_continues_the_unfinished_answers(tmp_path):
    """D-Num-hard at 64K (2026-11-10): per row, the model on both nodes (TP1 × DP3, 67,584 tokens), one proxy, the
    continuation client with a stop time before the job's limit; then the summary."""
    r = _run("numhard_64k.slurm", [], tmp_path, DPT_SLURM_NODES="node01-001 node01-002")
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    serve = [line for line in lines if "serve_vllm.sh" in line]
    assert len(serve) == 6 and all("67584" in line and "DPT_VLLM_DP=3" in line for line in serve)
    assert sum("node01-002" in line for line in serve) == 3                    # every row on the second node too
    for row in ("base_medium", "sft", "rl_seed1"):
        assert any(f"--served-model {row}" in line and "numhard_continue.py run" in line for line in lines), row
    assert sum("rr_proxy.py --port 8000" in line for line in lines) == 3
    assert "--stop-new-after 5100" in r.stdout                               # 7200 s left in a dry run − 35 min
    assert "numhard_continue.py summary" in r.stdout
    assert "exports/sft9b_seed1" in r.stdout and "exports/rl9b_seed1_step20" in r.stdout


def test_knowledge_job_lays_out_the_teacher_by_the_node_it_lands_on(tmp_path):
    """Knowledge data (2026-11-11), submitted to gpu-h100 and gpu-a100: on an H100 node the FP8 teacher fits one GPU,
    on an A100 node it needs two (TP2); Gemma takes the last GPU either way."""
    h100 = _run("knowledge.slurm", [], tmp_path, DPT_KNOWLEDGE_GPUS="NVIDIA H100 80GB HBM3,NVIDIA H100 80GB HBM3")
    assert h100.returncode == 0, h100.stderr
    out = h100.stdout
    assert "CUDA_VISIBLE_DEVICES=0 GPU_MEM=0.9 bash scripts/serve_vllm.sh Qwen/Qwen3.8-27B-FP8 teacher 8011 40960 1" in out
    assert "CUDA_VISIBLE_DEVICES=1 GPU_MEM=0.9 bash scripts/serve_vllm.sh" in out and "judge_v 8111 40960 1" in out
    runs = [line for line in out.splitlines() if "dpt.data.gen.knowledge run" in line]
    assert len(runs) == 2 and "--limit 40" in runs[0] and "--stop-new-after 83400" in runs[1]   # a first look, then all
    assert out.index("knowledge run") < out.index("knowledge export")
    mix = next(line for line in out.splitlines() if "data.slurm knowledge" in line)            # then SFT v2's data
    assert "DPT_TEACHER_ENDPOINTS=knowledge_teacher" in mix and "CUDA_VISIBLE_DEVICES=" in mix
    a100 = _run("knowledge.slurm", [], tmp_path, DPT_KNOWLEDGE_GPUS=",".join(["NVIDIA A100-PCIE-40GB"] * 3))
    assert a100.returncode == 0, a100.stderr
    assert "CUDA_VISIBLE_DEVICES=0\\,1 GPU_MEM=0.9 bash scripts/serve_vllm.sh Qwen/Qwen3.8-27B-FP8 teacher 8011 40960 2" in a100.stdout
    assert "CUDA_VISIBLE_DEVICES=2 GPU_MEM=0.9" in a100.stdout
    one = _run("knowledge.slurm", [], tmp_path, DPT_KNOWLEDGE_GPUS="NVIDIA A100-PCIE-40GB")
    assert one.returncode != 0 and "need 2 H100 or 3 A100 GPUs" in one.stderr


def test_job_2_is_queued_as_one_gated_8_node_job(tmp_path):
    """Job 2 (2026-11-12): queued now as a plain 8-node job that waits for its go-ahead flag and then runs
    sft_v2_job.sh, read when the job starts; the placeholder refuses to run."""
    r = _run("submit_window.sh", ["sft-v2-queue"], tmp_path)
    assert r.returncode == 0, r.stderr
    assert "DRY: sbatch -A PROJ -p gpu-a100 -N 8 -t 48:00:00 scripts/slurm/sft_v2_queue.slurm" in r.stdout
    text = (REPO / "scripts" / "slurm" / "sft_v2_queue.slurm").read_text()
    assert "sft_v2_job.ready" in text and "exec bash scripts/slurm/sft_v2_job.sh" in text and "#SBATCH -N 8" in text
    r = _run("sft_v2_queue.slurm", [], tmp_path)                    # a dry run skips the wait
    if "placeholder" in (REPO / "scripts" / "slurm" / "sft_v2_job.sh").read_text():
        assert r.returncode != 0 and "placeholder" in r.stderr


def test_data_knowledge_builds_the_knowledge_mix_and_the_replay_mix(tmp_path):
    """SFT v2's data (2026-11-11): the knowledge rows through the SFT stage alone, decontaminated with the teacher
    as paraphrase judge, then a replay sample of the v1 mix."""
    ep = tmp_path / "work" / "dpt" / "endpoints"
    ep.mkdir(parents=True)
    secure = tmp_path / "work" / "dpt" / "eval_secure"                   # decontamination needs every eval set
    secure.mkdir(parents=True)
    for name in ("d_num_draft", "d_code", "d_report", "d_fresh"):
        (secure / f"{name}.jsonl").write_text("")
    (ep / "knowledge_teacher.url").write_text("http://node18-001:8010")
    (ep / "knowledge_teacher.model").write_text("teacher")
    r = _run("data.slurm", ["knowledge"], tmp_path, DPT_TEACHER_ENDPOINTS="knowledge_teacher",
             DPT_ENDPOINTS=str(ep))
    out = "\n".join(_plain(r.stdout))
    assert "--stages sft" in out and "data_pipeline.output_dir=" in out and "/v2_knowledge" in out
    for s_ in ("domain_reasoning", "tool_traj", "report", "self_distill"):
        assert f"data_pipeline.inputs.sft_subsets.{s_}=null" in out
    assert "data_pipeline.inputs.sft_subsets.knowledge_report=" in out
    assert "data_pipeline.backends.judge.base_url=http://node18-001:8010" in out
    assert "sft_build replay-mix" in out and "--replay-token-ratio 2.0" in out and "/sft_v2/sft_train.jsonl" in out


def test_job_2_trains_sft_v2_evaluates_it_and_runs_rl_v3(tmp_path):
    """Job 2 (2026-11-12): SFT v2 on the 8 nodes, its export copied to $WORK, its eval as row sft_v2_seed1, then RL v3
    from it with RL v2's cap, and RL v3's final eval as row rl_v3_seed1."""
    (tmp_path / "sbx.sif").write_text("")
    _bank_check(tmp_path, [])
    cap = tmp_path / "scratch" / "dpt" / "checkpoints" / "rl" / "rl9b_v2_seed1.cap.json"
    cap.parent.mkdir(parents=True)
    cap.write_text(json.dumps({"cap": "24k"}))
    r = _run("sft_v2_job.sh", [], tmp_path, DPT_SLURM_NODES=EIGHT, PATH=_fake_squeue(tmp_path, ""))
    out, lines = r.stdout, _plain(r.stdout)
    assert "final eval: done" in r.stderr, r.stderr
    assert sum("04_sft.sh --config sft/sft_9b_v2.yaml" in line for line in lines) == 8      # every node trains
    assert "exports/sft9b_v2_seed1.partial" in out and "hf_export" in out
    assert len(_model_servers(out, "sft_v2_seed1", 40960)) == 21 and len(_model_servers(out, "sft_v2_seed1", 67584)) == 24
    assert "dpt.eval.budget continue --set d_fresh --row sft_v2_seed1" in out
    assert "RL v3 keeps RL v2's cap: 24k" in r.stderr and "memtest" not in out
    train = [line for line in out.splitlines() if line.startswith("DRY: python -m dpt.launch.rl --config rl/rl_9b_v3.yaml")]
    assert len(train) == 1 and "project.thinking.rl_max_response=24576" in train[0]
    assert "rl9b_v3_seed1/global_step_120" in out and len(_model_servers(out, "rl_v3_seed1", 40960)) == 21
    assert out.index("--row sft_v2_seed1") < out.index("--config rl/rl_9b_v3.yaml")


def test_a_second_job_waits_for_a_running_bank_build(tmp_path):
    """RL v2 and job 2 could start together with no bank: the second waits for the first's build (2026-11-12)."""
    text = (REPO / "scripts" / "slurm" / "rl_v2_job.sh").read_text()
    assert '"${V2}/BUILDING"' in text and "is building bank v2: waiting for its check" in text
