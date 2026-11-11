"""Overlong penalty and truncation flag (report §算法配置 长度控制)."""

from __future__ import annotations

import pytest

from dpt.rewards.length import LengthConfig, is_truncated, overlong_penalty


def test_penalty_shape():
    cfg = LengthConfig(max_response_tokens=24576, overlong_start=20480, max_penalty=0.5)
    assert overlong_penalty(0, cfg) == 0.0
    assert overlong_penalty(20480, cfg) == 0.0
    assert overlong_penalty(22528, cfg) == pytest.approx(-0.25)
    assert overlong_penalty(24576, cfg) == pytest.approx(-0.5)
    assert overlong_penalty(30000, cfg) == pytest.approx(-0.5)


def test_truncation_prefers_finish_reason():
    cfg = LengthConfig()
    assert is_truncated(100, cfg, finish_reason="length")
    assert not is_truncated(24576, cfg, finish_reason="stop")
    assert is_truncated(24576, cfg)
    assert not is_truncated(24575, cfg)
