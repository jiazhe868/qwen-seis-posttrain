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


def test_a_wrong_t1_dev_answer_is_either_a_wrong_value_or_unfinished():
    """Run 1 traded unfinished answers for wrong values (2026-11-11): the dev monitor needs to tell them apart."""
    wrong = reward_fn.compute_score("t1_num", "t = 120 / 4 = 30 s\n</think>\n\n\\boxed{30 s}", GT, {"split": "dev"})
    unfinished = reward_fn.compute_score("t1_num", "still thinking \\boxed{20 s}", GT, {"split": "dev"})
    right = reward_fn.compute_score("t1_num", GOOD, GT, {"split": "dev"})
    train = reward_fn.compute_score("t1_num", GOOD, GT, {"split": "train"})
    general = reward_fn.compute_score("t4_general", "x\n</think>\n\n\\boxed{20}",
                                      json.dumps({"kind": "math", "value": 20.0}), {"split": "dev"})
    assert (wrong["gold_score"], wrong["gold_wrong_value"], wrong["gold_unfinished"]) == (0.0, 1.0, 0.0)
    assert (unfinished["gold_score"], unfinished["gold_wrong_value"], unfinished["gold_unfinished"]) == (0.0, 0.0, 1.0)
    assert (right["gold_wrong_value"], right["gold_unfinished"]) == (0.0, 0.0)
    assert train["gold_wrong_value"] == train["gold_unfinished"] == -1.0
    assert general["gold_wrong_value"] == general["gold_unfinished"] == -1.0
    assert set(wrong) == set(train) == set(general), "verl needs identical keys for every sample"


def test_the_hard_dev_split_is_graded_as_t1_and_unfinished_answers_with_the_value_are_flagged():
    """2026-11-15: t1_num_hard rows are T1 under their own name; an unfinished answer whose trace already holds the
    right value is gold_unfinished_seen (the model kept checking instead of answering)."""
    hard = reward_fn.compute_score("t1_num_hard", GOOD, GT, {"split": "dev"})
    t1 = reward_fn.compute_score("t1_num", GOOD, GT, {"split": "dev"})
    assert hard["score"] == t1["score"] == pytest.approx(1.0) and hard["gold_score"] == 1.0
    assert hard["gold_unfinished_seen"] == 0.0 and hard["gold_len"] == pytest.approx(len(GOOD) / 3.2)
    looping = reward_fn.compute_score("t1_num_hard", "t = 120 / 6 = 20 s. Let me check that again, 120 / 6 ...",
                                      GT, {"split": "dev"})
    lost = reward_fn.compute_score("t1_num_hard", "t = 120 / 4 = 30 s. Hmm, let me redo the distance ...",
                                   GT, {"split": "dev"})
    assert (looping["gold_unfinished"], looping["gold_unfinished_seen"]) == (1.0, 1.0)
    assert (lost["gold_unfinished"], lost["gold_unfinished_seen"]) == (1.0, 0.0)
    train = reward_fn.compute_score("t1_num", GOOD, GT, {"split": "train"})
    assert train["gold_unfinished_seen"] == train["gold_len"] == -1.0
    assert set(hard) == set(train), "verl needs identical keys for every sample"
