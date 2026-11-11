"""CPU end-to-end: data pipeline → verl rows → reward adapter → step advantages → metrics → alerts.

This stitches the framework-free modules together exactly as the verl adapter does at run time,
using the toy bank. It catches contract drift between modules (answer formats, extra_info keys,
metric names) that per-module tests cannot see.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dpt.common.config import to_container
from dpt.common.io import read_jsonl
from dpt.launch.common import compose_stage
from dpt.monitoring.alerts import AlertManager
from dpt.monitoring.bridge import to_canonical
from dpt.rl.pipeline import compute_step_advantages
from dpt.verl_ext import reward_fn
from dpt.verl_ext.runtime import build_runtime

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def toy_bank(tmp_path_factory):
    out = tmp_path_factory.mktemp("bank")
    subprocess.run([sys.executable, "-m", "dpt.data.pipeline", "--offline",
                    "--stages", "ingest,quality,dedup,decontam,verify,difficulty,publish",
                    f"data_pipeline.output_dir={out}"],
                   check=True, cwd=REPO, env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"},
                   capture_output=True)
    rows = list(read_jsonl(out / "rl_main.jsonl")) + list(read_jsonl(out / "rl_hard.jsonl"))
    assert rows, "toy pipeline published nothing"
    return rows


def _answer(row, correct: bool) -> str:
    gt = json.loads(row["reward_model"]["ground_truth"])
    value = gt["value"] if correct else gt["value"] * 1.5 + 1
    unit = (" " + gt["unit"]) if gt.get("unit") else ""
    return f"Working through the physics step by step.\n</think>\n\nThe result is \\boxed{{{value:g}{unit}}}."


def _numeric_rows(rows):
    return [r for r in rows if r["data_source"] == "t1_num"
            or (r["data_source"] == "t4_general" and json.loads(r["reward_model"]["ground_truth"]).get("kind") == "math")]


def test_prompt_following_answers_are_rewarded_and_wrong_ones_are_not(toy_bank, monkeypatch):
    monkeypatch.delenv("DPT_RESOLVED_CONFIG", raising=False)
    rows = _numeric_rows(toy_bank)
    assert rows
    for r in rows:
        assert "\\boxed" in r["prompt"][0]["content"], "prompt must state the answer format"
        good = reward_fn.compute_score(r["data_source"], _answer(r, True), r["reward_model"]["ground_truth"],
                                       {**r["extra_info"], "split": "dev"})
        bad = reward_fn.compute_score(r["data_source"], _answer(r, False), r["reward_model"]["ground_truth"],
                                      {**r["extra_info"], "split": "dev"})
        assert good["score"] == pytest.approx(1.0), (r["extra_info"]["question"], good)
        assert good["gold_score"] == 1.0, "independent eval grader must agree with the training grader"
        assert bad["score"] == 0.0 and bad["gold_score"] == 0.0


def test_one_simulated_step_through_advantages_metrics_and_alerts(toy_bank, monkeypatch):
    monkeypatch.delenv("DPT_RESOLVED_CONFIG", raising=False)
    cfg = to_container(compose_stage("rl/rl_27b.yaml"))
    rt = build_runtime(cfg)
    rows = _numeric_rows(toy_bank)
    rng = np.random.default_rng(0)
    scores, gids, tasks, lens = [], [], [], []
    for r in rows:                                   # 8 samples per prompt, ~half correct
        for _ in range(8):
            ok = bool(rng.random() < 0.5)
            out = reward_fn.compute_score(r["data_source"], _answer(r, ok), r["reward_model"]["ground_truth"],
                                          r["extra_info"])
            scores.append(out["score"])
            gids.append(r["extra_info"]["uid"])
            tasks.append(r["data_source"])
            lens.append(int(rng.integers(500, 6000)))
    res = compute_step_advantages(scores, gids, tasks, lens, max_response=cfg["rl"]["sampling"]["max_response"],
                                  scaler=rt.scaler, normalizer=rt.normalizer, nominal_mix=rt.nominal_mix(0))
    assert np.isfinite(res.seq_adv).all()
    metrics = {**res.metrics, "mismatch/log_ppl_abs_diff": 3e-4, "actor/grad_norm": 0.8}
    metrics.update(to_canonical(metrics))
    am = AlertManager.from_config(cfg["monitoring"])
    alerts = am.evaluate(1, metrics)
    assert not am.should_pause(), [a.to_dict() for a in alerts]
