from __future__ import annotations

from dpt.common.io import write_jsonl
from dpt.eval.select_checkpoint import select


def test_prefers_later_step_within_noise_band(tmp_path):
    for s in (20, 40, 60):
        (tmp_path / f"global_step_{s}").mkdir()
    write_jsonl(tmp_path / "metrics.jsonl", [
        {"step": 20, "dev/d_num": 60.0}, {"step": 40, "dev/d_num": 66.0}, {"step": 60, "dev/d_num": 65.0},
        {"step": 50, "dev/d_num": 70.0},          # not saved → not selectable
    ])
    r = select(tmp_path)
    assert r["selected_step"] == 60 and r["best_value"] == 66.0
