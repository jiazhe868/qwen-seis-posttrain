"""scripts/slurm/data.slurm in dry-run mode: the steps it runs, what it needs, and what it never overwrites."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _env(tmp_path, endpoints=("teacher", "judge", "base", "embed"), **extra):
    ep = tmp_path / "ep"
    ep.mkdir(exist_ok=True)
    for i, name in enumerate(endpoints):
        (ep / f"{name}.url").write_text(f"http://node01-00{i + 1}:8000\n")
        (ep / f"{name}.model").write_text(f"{name}\n")
    (tmp_path / "sbx.sif").write_text("image")                    # present: sandbox_image pulls nothing
    e = {**os.environ, "DPT_SLURM_DRY_RUN": "1", "SLURM_SUBMIT_DIR": str(REPO), "WORK": str(tmp_path / "work"),
         "SCRATCH": str(tmp_path / "scratch"), "DPT_CONDA": str(tmp_path / "no-conda"),
         "DPT_SANDBOX_SIF": str(tmp_path / "sbx.sif"), "DPT_ENDPOINTS": str(ep),
         "DPT_EVAL_SECURE_ROOT": str(tmp_path / "secure"), "PYTHONPATH": str(REPO / "src"), **extra}
    for k in ("DPT_HW_PROFILE", "DPT_MODEL_LINE", "DPT_DATA_ROOT", "DPT_TEACHER_ENDPOINTS", "DPT_DATA_FORCE"):
        if k not in extra:
            e.pop(k, None)
    return e


def _run(args, env):
    return subprocess.run(["bash", str(REPO / "scripts" / "slurm" / "data.slurm"), *args], capture_output=True,
                          text=True, env=env, cwd=REPO, timeout=180)


def _dry(out):
    """The DRY: command lines, split back into argument lists."""
    return [shlex.split(line[len("DRY: "):]) for line in out.stdout.splitlines() if line.startswith("DRY: ")]


def _cmd(cmds, *head):
    hits = [c for c in cmds if c[:len(head)] == list(head) or (c[:1] == ["env"] and head[0] in c)]
    assert hits, f"no command starting with {head}: {cmds}"
    return hits[0]


def test_gen_runs_every_generator_with_the_teacher(tmp_path):
    out = _run(["gen"], _env(tmp_path))
    assert out.returncode == 0, out.stderr
    cmds = _dry(out)
    data = tmp_path / "work" / "dpt" / "data" / "gen"
    t1 = _cmd(cmds, "python", "-m", "dpt.data.gen.t1")
    assert t1[t1.index("--out") + 1] == str(data / "t1_candidates.jsonl")
    assert t1[t1.index("--heldout-out") + 1] == str(tmp_path / "secure" / "d_num_draft.jsonl")
    t2 = _cmd(cmds, "python", "-m", "dpt.data.gen.t2")
    assert "--check" in t2 and t2[t2.index("--n-train") + 1] == "3000" and t2[t2.index("--n-heldout") + 1] == "150"
    _cmd(cmds, "python", "-m", "dpt.data.gen.t4")
    t3 = _cmd(cmds, "python", "-m", "dpt.data.gen.t3")
    assert t3[t3.index("--url") + 1] == "http://node01-001:8000" and t3[t3.index("--tokenizer") + 1] == "Qwen/Qwen3.8-27B"
    para = _cmd(cmds, "python", "-m", "dpt.data.gen.paraphrase")
    assert para[para.index("--in") + 1] == str(data / "t1_candidates.jsonl")


def test_finished_steps_are_skipped_and_eval_files_never_replaced(tmp_path):
    env = _env(tmp_path)
    done = tmp_path / "work" / "dpt" / "data" / "gen" / ".done"
    done.mkdir(parents=True)
    for s in ("t2", "t4", "t3", "paraphrase"):
        (done / s).write_text("x")
    (tmp_path / "secure").mkdir()
    (tmp_path / "secure" / "d_num_draft.jsonl").write_text("{}\n")
    out = _run(["gen"], env)
    assert out.returncode != 0 and "frozen eval material" in out.stderr       # t1 would rewrite D-Num candidates
    (done / "t1").write_text("x")
    out = _run(["gen"], _env(tmp_path, endpoints=()))                         # nothing to do: no endpoint needed
    assert out.returncode == 0 and not _dry(out) and out.stderr.count("finished before") == 5
    out = _run(["gen"], {**_env(tmp_path), "DPT_DATA_FORCE": "1", "DPT_EVAL_SETS_FORCE": "1"})
    assert out.returncode == 0 and len([c for c in _dry(out) if "dpt.data.gen.t1" in c]) == 1


def test_gen_needs_the_teacher_and_spreads_over_several_servers(tmp_path):
    out = _run(["gen"], _env(tmp_path, endpoints=("judge",)))
    assert out.returncode != 0 and "no teacher endpoint" in out.stderr
    out = _run(["gen"], _env(tmp_path, endpoints=("teacher", "teacher2"), DPT_TEACHER_ENDPOINTS="teacher teacher2"))
    assert out.returncode == 0, out.stderr
    t3 = _cmd(_dry(out), "python", "-m", "dpt.data.gen.t3")
    assert [t3[i + 1] for i, a in enumerate(t3) if a == "--url"] == ["http://node01-001:8000", "http://node01-002:8000"]
    assert [t3[i + 1] for i, a in enumerate(t3) if a == "--model"] == ["teacher", "teacher2"]


def test_bank_needs_every_eval_set_and_wires_the_pipeline(tmp_path):
    env = _env(tmp_path)
    out = _run(["bank"], env)
    assert out.returncode != 0 and "decontamination needs every eval set" in out.stderr
    secure = tmp_path / "secure"
    secure.mkdir(parents=True, exist_ok=True)
    for f in ("d_num_draft.jsonl", "d_code.jsonl", "d_report.jsonl", "d_fresh.jsonl"):
        (secure / f).write_text("{}\n")
    out = _run(["bank"], env)
    assert out.returncode == 0, out.stderr
    cmds = _dry(out)
    rc = _cmd(cmds, "python", "-m", "dpt.data.gen.sft_gen", "rewards-config")
    assert any(a.startswith("rewards.t2_code.sandbox.hidden_tests_dir=") for a in rc)
    assert "rewards.t3_report.judge.base_url=http://node01-002:8000/v1" in rc
    pipe = _cmd(cmds, "env")
    assert pipe[1].startswith("DPT_REWARDS_CONFIG=") and "dpt.data.pipeline" in pipe
    ov = dict(a.split("=", 1) for a in pipe if a.startswith("data_pipeline."))
    fams = json.loads(ov["data_pipeline.verify.families"])
    assert [f["name"] for f in fams] == ["teacher", "judge"] and fams[0]["base_url"] == "http://node01-001:8000/v1"
    assert fams[1]["url_file"] == str(tmp_path / "ep" / "judge.url") and fams[1]["wait_s"] > 600
    assert fams[0]["timeout_s"] == fams[1]["timeout_s"] >= 3600
    assert len(json.loads(ov["data_pipeline.inputs.rl_candidates"])) == 4
    assert ov["data_pipeline.backends.base.base_url"] == "http://node01-003:8000/v1"
    assert ov["data_pipeline.backends.embed.local"] == "true"
    assert ov["data_pipeline.backends.embed.model"] == "Qwen/Qwen3-Embedding-0.6B"     # in-process, no server
    assert ov["data_pipeline.verify.mode"] == "audit" and ov["data_pipeline.difficulty.mode"] == "audit"
    assert fams[0]["tokenizer"] == "Qwen/Qwen3.8-27B" and fams[0]["effort"] == "medium"
    assert ov["data_pipeline.inputs.public_benchmarks_dir"] == str(secure / "public_benchmarks")
    assert len(json.loads(ov["data_pipeline.decontam.canaries"])) == 5                      # D-Num-hard too
    assert ov["data_pipeline.inputs.eval_sets.d_fresh"] == str(secure / "d_fresh.jsonl")
    assert "data_pipeline.inputs.eval_sets.d_num_hard" not in ov and "data_pipeline.output_dir" not in ov


def test_hard_bank_has_its_own_inputs_output_markers_and_servers(tmp_path):
    secure = tmp_path / "secure"
    secure.mkdir(parents=True, exist_ok=True)
    for f in ("d_num_draft.jsonl", "d_code.jsonl", "d_report.jsonl", "d_fresh.jsonl", "d_num_hard_draft.jsonl"):
        (secure / f).write_text("{}\n")
    cand = tmp_path / "t1_hard.jsonl"
    cand.write_text("{}\n")
    out_dir = tmp_path / "work" / "dpt" / "data" / "v1_hard"
    env = _env(tmp_path, endpoints=("teacher", "judge", "judge_v", "sft_audit"), DPT_BANK_NAME="bank_hard",
               DPT_BANK_OUT=str(out_dir), DPT_BANK_CANDIDATES=str(cand), DPT_JUDGE_ENDPOINT="judge_v",
               DPT_DIFFICULTY_ENDPOINT="sft_audit")
    (tmp_path / "work" / "dpt" / "data" / "gen" / ".done").mkdir(parents=True)
    (tmp_path / "work" / "dpt" / "data" / "gen" / ".done" / "bank").write_text("x")      # the v1 bank is done
    out = _run(["bank"], env)
    assert out.returncode == 0, out.stderr
    pipe = _cmd(_dry(out), "env")
    ov = dict(a.split("=", 1) for a in pipe if a.startswith("data_pipeline."))
    assert json.loads(ov["data_pipeline.inputs.rl_candidates"]) == [str(cand)]
    assert ov["data_pipeline.output_dir"] == str(out_dir)
    fams = json.loads(ov["data_pipeline.verify.families"])
    assert fams[1]["base_url"] == "http://node01-003:8000/v1" and fams[1]["url_file"].endswith("judge_v.url")
    assert ov["data_pipeline.backends.base.base_url"] == "http://node01-004:8000/v1"           # the SFT model
    assert ov["data_pipeline.inputs.eval_sets.d_num_hard"] == str(secure / "d_num_hard_draft.jsonl")


def test_sft_samples_every_subset_then_mixes(tmp_path):
    secure = tmp_path / "secure"
    secure.mkdir(parents=True, exist_ok=True)
    for f in ("d_num_draft.jsonl", "d_code.jsonl", "d_report.jsonl", "d_fresh.jsonl"):
        (secure / f).write_text("{}\n")
    bank = tmp_path / "work" / "dpt" / "data" / "v1"
    (bank / "stages").mkdir(parents=True)
    (bank / "stages" / "04_verify.jsonl").write_text("{}\n")
    (bank / "rl_dev.jsonl").write_text("{}\n")
    out = _run(["sft"], _env(tmp_path))
    assert out.returncode == 0, out.stderr
    cmds = _dry(out)
    subs = [c[3] for c in cmds if c[:3] == ["python", "-m", "dpt.data.gen.sft_gen"] and c[3] != "rewards-config"]
    assert subs == ["domain", "tool", "report", "distill"]
    excl = [str(bank / "rl_dev_candidates.jsonl"), str(bank / "rl_dev.jsonl")]   # candidates + a published split
    assert ["python", "-m", "dpt.data.pipeline", "--config", "data/pipeline.yaml", "--dev-candidates"] in cmds
    for c in cmds:
        if c[:3] == ["python", "-m", "dpt.data.gen.sft_gen"] and c[3] != "rewards-config":
            assert [c[i + 1] for i, a in enumerate(c) if a == "--exclude"] == excl
    distill = [c for c in cmds if c[3:4] == ["distill"]][0]
    assert distill[distill.index("--tokenizer") + 1] == "Qwen/Qwen3.5-9B"
    mix = [c for c in cmds if c[:3] == ["python", "-m", "dpt.data.pipeline"] and "--stages" in c][0]
    assert mix[mix.index("--stages") + 1] == "sft"
    assert f"data_pipeline.sft.exclude_problems_from=[{','.join(excl)}]" in mix
    assert any(a.startswith("data_pipeline.inputs.sft_subsets.self_distill=") for a in mix)
    # sampled already (runners on the login node and in serve jobs): the mix needs only the teacher
    done = tmp_path / "work" / "dpt" / "data" / "gen" / ".done"
    done.mkdir(parents=True, exist_ok=True)
    for s in ("sft_domain", "sft_tool", "sft_report", "sft_distill"):
        (done / s).write_text("x")
    out = _run(["sft"], _env(tmp_path, endpoints=("teacher",)))
    assert out.returncode == 0, out.stderr
    cmds = _dry(out)
    assert [c[3] for c in cmds if c[:3] == ["python", "-m", "dpt.data.gen.sft_gen"]] == ["rewards-config"]
    assert [c[c.index("--stages") + 1] for c in cmds if c[:3] == ["python", "-m", "dpt.data.pipeline"]
            and "--stages" in c] == ["sft"]


def test_bank_runs_a_subset_of_stages_with_only_the_servers_it_needs(tmp_path):
    secure = tmp_path / "secure"
    secure.mkdir(parents=True, exist_ok=True)
    for f in ("d_num_draft.jsonl", "d_code.jsonl", "d_report.jsonl", "d_fresh.jsonl"):
        (secure / f).write_text("{}\n")
    env = _env(tmp_path, endpoints=("teacher",), DPT_BANK_STAGES="ingest,quality,dedup,decontam")
    out = _run(["bank"], env)
    assert out.returncode == 0, out.stderr                         # no judge, no base needed for these stages
    pipe = _cmd(_dry(out), "env")
    assert pipe[pipe.index("--stages") + 1] == "ingest,quality,dedup,decontam"
    ov = dict(a.split("=", 1) for a in pipe if a.startswith("data_pipeline."))
    assert "data_pipeline.verify.families" not in ov and "data_pipeline.backends.base.base_url" not in ov
    assert ov["data_pipeline.backends.embed.local"] == "true"
    env = _env(tmp_path, endpoints=("teacher", "judge"), DPT_BANK_STAGES="verify,difficulty,publish")
    out = _run(["bank"], env)
    assert out.returncode != 0 and "no base endpoint" in out.stderr  # difficulty needs the base 9B


def test_difficulty_without_t3_runs_between_judge_windows(tmp_path):
    env = _env(tmp_path, endpoints=("teacher", "base"), DPT_BANK_STAGES="difficulty,publish",
               DPT_DIFFICULTY_SKIP="[t3_report,t2_code]")
    out = _run(["bank"], env)
    assert out.returncode == 0, out.stderr
    rc = _cmd(_dry(out), "python", "-m", "dpt.data.gen.sft_gen", "rewards-config")
    assert "rewards.t3_report.judge.base_url=http://judge-not-up-yet:8000/v1" in rc   # never called: fails closed
    pipe = _cmd(_dry(out), "env")
    assert pipe[pipe.index("--stages") + 1] == "difficulty,publish"
    assert "data_pipeline.publish.dev_candidates=true" not in pipe                   # no verified bank yet
    assert not any(a.startswith("data_pipeline.difficulty.max_concurrency=") for a in pipe)   # one server
    proxy = _cmd(_dry(_run(["bank"], {**env, "DPT_DIFFICULTY_CONCURRENCY": "576"})), "env")
    assert "data_pipeline.difficulty.max_concurrency=576" in proxy                  # a proxy over 24 servers
    bank = tmp_path / "work" / "dpt" / "data" / "v1"
    (bank / "stages").mkdir(parents=True, exist_ok=True)
    (bank / "stages" / "04_verify.jsonl").write_text("{}\n")
    out = _run(["bank"], env)                         # verified earlier: the dev candidates come first
    cmds = _dry(out)
    assert ["python", "-m", "dpt.data.pipeline", "--config", "data/pipeline.yaml", "--dev-candidates"] in cmds
    assert "data_pipeline.publish.dev_candidates=true" in _cmd(cmds, "env")
    (bank / "rl_dev_candidates.jsonl").write_text('{"uid": "p-1"}\n')
    out = _run(["bank"], env)
    assert "--dev-candidates" not in sum(_dry(out), []) and "data_pipeline.publish.dev_candidates=true" in \
        _cmd(_dry(out), "env")
    out = _run(["bank"], {**env, "DPT_DIFFICULTY_SKIP": "[t2_code]"})                # grading T3 needs the judge
    assert out.returncode != 0 and "no judge endpoint" in out.stderr
