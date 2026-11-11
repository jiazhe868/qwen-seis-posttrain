"""Exact freezing on export: vision tensors restored from base, LM change verified."""

from __future__ import annotations

import json

import pytest
import torch

safetensors_torch = pytest.importorskip("safetensors.torch")

from dpt.launch.checkpoint_tools import restore_frozen, verify_export  # noqa: E402


def _write(d, tensors, shards=2):
    d.mkdir(parents=True, exist_ok=True)
    names = sorted(tensors)
    wm = {}
    for i in range(shards):
        part = {n: tensors[n] for n in names[i::shards]}
        fn = f"model-{i:05d}-of-{shards:05d}.safetensors"
        safetensors_torch.save_file(part, str(d / fn), metadata={"format": "pt"})
        wm.update({n: fn for n in part})
    (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}))


def test_restore_then_verify(tmp_path):
    base = {"model.visual.patch.weight": torch.ones(4, 4), "model.language_model.layers.0.w": torch.zeros(4, 4),
            "lm_head.weight": torch.zeros(4, 4)}
    trained = {"model.visual.patch.weight": torch.ones(4, 4) * 0.999,          # weight-decay drift
               "model.language_model.layers.0.w": torch.full((4, 4), 0.1), "lm_head.weight": torch.zeros(4, 4)}
    _write(tmp_path / "base", base)
    _write(tmp_path / "ckpt", trained)

    with pytest.raises(AssertionError):            # drifted vision tower is caught before restoring
        verify_export(str(tmp_path / "ckpt"), str(tmp_path / "base"), ["model.visual."])

    assert restore_frozen(str(tmp_path / "ckpt"), str(tmp_path / "base"), ["model.visual."]) == 1
    rep = verify_export(str(tmp_path / "ckpt"), str(tmp_path / "base"), ["model.visual."])
    assert rep["frozen_mismatch"] == 0 and rep["changed_trainable"] >= 1


def test_noop_export_is_rejected(tmp_path):
    base = {"model.visual.a": torch.ones(2), "model.language_model.b": torch.zeros(2)}
    _write(tmp_path / "base", base, shards=1)
    _write(tmp_path / "ckpt", base, shards=1)
    with pytest.raises(AssertionError, match="no-op"):
        verify_export(str(tmp_path / "ckpt"), str(tmp_path / "base"), ["model.visual."])
