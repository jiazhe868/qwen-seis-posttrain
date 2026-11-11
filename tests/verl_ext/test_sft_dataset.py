"""DPTSFTDataset: per-row effort rendering and verl-compatible output (no verl import needed)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from dpt.verl_ext.sft_dataset import DPTSFTDataset
from tests.fakes import CharTokenizer


class _Tok(CharTokenizer):
    pad_token_id = 0


def _frame():
    conv = [{"role": "user", "content": "q?"}, {"role": "assistant", "reasoning_content": "r", "content": "a"}]
    return pd.DataFrame({
        "messages": [json.dumps(conv), conv],          # JSON string and native list both accepted
        "reasoning_effort": ["low", "medium"],
    })


@pytest.fixture
def ds(monkeypatch):
    monkeypatch.setattr(pd, "read_parquet", lambda f: _frame())
    return DPTSFTDataset("x.parquet", _Tok(), {"pad_mode": "no_padding", "max_length": 4096})


def test_each_row_uses_its_own_effort(ds):
    tok = _Tok()
    low = tok.decode(ds[0]["input_ids"].tolist())
    med = tok.decode(ds[1]["input_ids"].tolist())
    assert "Reasoning effort is set to low." in low
    assert "Reasoning effort" not in med


def test_output_matches_verl_no_padding_contract(ds):
    item = ds[0]
    assert set(item) == {"input_ids", "position_ids", "loss_mask"}
    assert item["input_ids"].shape == item["loss_mask"].shape == item["position_ids"].shape
    assert item["loss_mask"].sum() > 0


def test_truncation_must_be_error(monkeypatch):
    monkeypatch.setattr(pd, "read_parquet", lambda f: _frame())
    with pytest.raises(ValueError):
        DPTSFTDataset("x.parquet", _Tok(), {"truncation": "right"})


def test_jsonl_rows_without_tools_render(tmp_path):
    """2026-10-28: pandas reads `tools` as NaN in rows that lack it; list(nan) stopped the 9B SFT at step 0."""
    conv = [{"role": "user", "content": "q?"}, {"role": "assistant", "reasoning_content": "r", "content": "a"}]
    tool = {"type": "function", "function": {"name": "run_python", "description": "run code",
                                             "parameters": {"type": "object", "properties": {}}}}
    rows = [{"messages": conv, "reasoning_effort": "medium", "tools": [tool]},
            {"messages": conv, "reasoning_effort": "medium"},
            {"messages": conv, "reasoning_effort": "low", "tools": None}]
    f = tmp_path / "mix.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    d = DPTSFTDataset(str(f), _Tok(), {"pad_mode": "no_padding", "max_length": 4096})
    assert d.tools[1] is None and d.tools[2] is None
    texts = [_Tok().decode(d[i]["input_ids"].tolist()) for i in range(len(d))]
    assert "run_python" in texts[0] and "run_python" not in texts[1] + texts[2]


def test_right_padding_mode(monkeypatch):
    monkeypatch.setattr(pd, "read_parquet", lambda f: _frame())
    d = DPTSFTDataset("x.parquet", _Tok(), {"pad_mode": "right", "max_length": 512})
    item = d[1]
    assert item["input_ids"].shape[0] == 512 and int(item["attention_mask"].sum()) < 512
