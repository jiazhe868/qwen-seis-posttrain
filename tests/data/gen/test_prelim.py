"""The timing run's preliminary bank: stage-A task shares, its own directory, and a config that composes."""

from __future__ import annotations

import json

import pytest

from dpt.common.io import read_jsonl, write_jsonl
from dpt.data.gen.families import by_split
from dpt.data.gen.prelim import main
from dpt.data.gen.t1 import build_train


def _t4(n):
    return [{"task": "t4_general", "question": f"Compute {i} + {i}. Give the number.",
             "answer": {"kind": "math", "value": 2 * i}, "source": {"license": "MIT"}} for i in range(n)]


def _t2(n):
    return [{"task": "t2_code", "question": f"Write f{i}(x) returning x + {i}.",
             "answer": {"entry_point": f"f{i}", "visible_tests": "def test_v():\n    pass\n",
                        "hidden_tests_id": f"h{i}"}, "meta": {"split": "train"}} for i in range(n)]


def test_prelim_bank_follows_stage_a_shares(tmp_path):
    t1 = build_train(by_split("train")[:6], 40, seed=1)
    write_jsonl(tmp_path / "t1.jsonl", t1)
    write_jsonl(tmp_path / "t2.jsonl", _t2(60))
    write_jsonl(tmp_path / "t4.jsonl", _t4(60))
    out = tmp_path / "prelim"
    assert main(["--t1", str(tmp_path / "t1.jsonl"), "--t2", str(tmp_path / "t2.jsonl"), "--t4",
                 str(tmp_path / "t4.jsonl"), "--out-dir", str(out), "--n", "200"]) == 0
    s = json.loads((out / "prelim_summary.json").read_text())
    assert s["by_task"] == {"t1_num": 100, "t2_code": 50, "t4_general": 50}
    rows = list(read_jsonl(out / "rl_main.jsonl")) + list(read_jsonl(out / "rl_dev.jsonl"))
    assert len(rows) == 200
    from dpt.common.manifest import verify_manifest

    assert verify_manifest(out / "rl_main.jsonl").name == "prelim.rl_main"
    with pytest.raises(SystemExit, match="real bank"):
        main(["--t1", str(tmp_path / "t1.jsonl"), "--t2", str(tmp_path / "t2.jsonl"), "--t4",
              str(tmp_path / "t4.jsonl"), "--out-dir", str(tmp_path / "v1")])


def test_timing_config_composes(monkeypatch):
    from dpt.launch.common import compose_stage

    monkeypatch.setenv("DPT_MODEL_LINE", "qwen35_9b")
    monkeypatch.setenv("DPT_DATA_ROOT", "/data")
    c = compose_stage("rl/rl_9b_timing.yaml", profile="slurm_3xa100_40g")
    assert c.rl.total_steps == 10 and c.rl_run.data_dir == "/data/prelim"
    assert list(c.rl_run.train_files) == ["/data/prelim/rl_main.parquet"]
    assert c.rl_run.init_checkpoint == "Qwen/Qwen3.5-9B"
    assert c.rewards.t2_code.sandbox.hidden_tests_dir == "/data/prelim/t2/hidden_tests"
    assert c.rewards.length.apply_penalty is False                     # inherited, not replaced
    assert c.verl["trainer.total_training_steps"] == 10
