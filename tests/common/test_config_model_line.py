"""Model-line layer (configs/model/<line>.yaml, DPT_MODEL_LINE): 27B stays the default, 9B is opt-in."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from dpt.common.config import load_config, to_container


def test_default_line_is_the_reports_27b(monkeypatch):
    monkeypatch.delenv("DPT_MODEL_LINE", raising=False)
    c = to_container(load_config())["project"]
    assert c["model"]["line"] == "qwen38_27b" and c["model"]["base"] == "Qwen/Qwen3.8-27B"
    assert c["model"]["arch"]["num_layers"] == 64
    assert c["thinking"]["eval_efforts"] == ["medium", "xhigh"]
    assert c["gates"]["d_num"]["ref"] == "base_xhigh"


def test_9b_line_overrides_model_arch_protocol_and_gates(monkeypatch):
    monkeypatch.setenv("DPT_MODEL_LINE", "qwen35_9b")
    c = to_container(load_config())["project"]
    assert c["model"]["base"] == "Qwen/Qwen3.5-9B" and c["model"]["tag"] == "9b"
    a = c["model"]["arch"]
    assert (a["num_layers"], a["num_linear_attn_layers"], a["num_full_attn_layers"]) == (32, 24, 8)
    assert a["num_layers"] == a["num_linear_attn_layers"] + a["num_full_attn_layers"]
    # Qwen3.5's template has no effort control: one thinking mode, no base@xhigh row to compare against
    assert c["thinking"]["eval_efforts"] == ["medium"]
    assert all(g.get("ref") != "base_xhigh" for g in c["gates"].values())
    # eval-set overrides merge into, not replace, the report's per-set protocol
    assert c["eval"]["sets"]["d_num"] == {"n_items": 300, "metric": "avg@8", "samples": 8, "expected_ci": 2.3}
    assert c["eval"]["sets"]["d_fresh"]["written_after"] == "2026-08-14"


def test_explicit_argument_beats_environment(monkeypatch):
    monkeypatch.setenv("DPT_MODEL_LINE", "qwen35_9b")
    assert OmegaConf.select(load_config(model_line="qwen38_27b"), "project.model.line") == "qwen38_27b"


def test_unknown_line_names_the_known_ones(monkeypatch):
    monkeypatch.setenv("DPT_MODEL_LINE", "qwen9_1t")
    with pytest.raises(FileNotFoundError, match="qwen35_9b"):
        load_config()
