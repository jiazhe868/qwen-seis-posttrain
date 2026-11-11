"""The summary merges pass@k curves from the sibling <row>_k64 run (needed by the RL-entry gate)."""

from __future__ import annotations

from dpt.common.io import write_json
from dpt.eval.report import build_summary


def _results(path, k_curve=None):
    s = {"d_num": {"score": 60.0, "metric": "avg@8", "ci": {"halfwidth": 1.6}}}
    if k_curve:
        s["d_num"]["pass_at_k"] = k_curve
    write_json(path / "results.json", {"model_name": path.name, "sets": s, "general": {}})


def test_k64_sibling_is_merged(tmp_path):
    (tmp_path / "sft").mkdir()
    (tmp_path / "sft_k64").mkdir()
    _results(tmp_path / "sft")
    _results(tmp_path / "sft_k64", {"1": 0.6, "16": 0.83, "64": 0.9})
    rows = build_summary({"sft": tmp_path / "sft"})["rows"]
    assert rows["sft"]["pass_at_k"]["d_num"]["16"] == 0.83
