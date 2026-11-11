"""Reward adapter: fixed keys, gold grading only on dev rows (independent eval grader)."""

from __future__ import annotations

import json

import pytest

from dpt.verl_ext import reward_fn

GT = json.dumps({"value": 20.0, "unit": "s", "rel_tol": 0.01, "abs_tol": 0.0})
GOOD = "t = 120 km / 6 km/s = 20 s\n</think>\n\nThe travel time is \\boxed{20 s}."


@pytest.fixture(autouse=True)
def _no_resolved_config(monkeypatch):
    monkeypatch.delenv("DPT_RESOLVED_CONFIG", raising=False)


def test_dev_row_gets_gold_train_row_does_not():
    dev = reward_fn.compute_score("t1_num", GOOD, GT, {"split": "dev", "uid": "p-1"})
    train = reward_fn.compute_score("t1_num", GOOD, GT, {"split": "train", "uid": "p-1"})
    assert dev["score"] == pytest.approx(1.0) and dev["gold_score"] == 1.0
    assert train["gold_score"] == -1.0
    assert set(dev) == set(train), "verl needs identical keys for every sample"


def test_gold_is_stricter_on_missing_unit():
    no_unit = "work\n</think>\n\n\\boxed{20}"
    out = reward_fn.compute_score("t1_num", no_unit, GT, {"split": "dev"})
    assert out["gold_score"] == 0.0


def test_no_think_end_fails_both():
    out = reward_fn.compute_score("t1_num", "still thinking \\boxed{20 s}", GT, {"split": "dev"})
    assert out["score"] <= 0.0 and out["gold_score"] == 0.0
