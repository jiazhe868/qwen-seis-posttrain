"""RewardResult contract and the fail-closed decorator (report §防范原则 "fail closed")."""

from __future__ import annotations

import math

from dpt.rewards.base import RewardResult, fail_closed


def test_to_verl_dict_has_contract_keys():
    r = RewardResult(score=1.0, task="t1_num", correct=True, components={"x": 2})
    d = r.to_verl_dict()
    assert (
        d["score"] == 1.0
        and d["acc"] == 1.0
        and d["reason"] == "ok"
        and d["truncated"] is False
        and d["x"] == 2.0
    )


def test_acc_falls_back_to_clipped_score_for_continuous_rewards():
    assert RewardResult(score=0.7, task="t3_report").acc == 0.7
    assert RewardResult(score=-0.2, task="t3_report").acc == 0.0


def test_nan_score_is_neutralized():
    r = RewardResult(score=math.nan, task="t1_num")
    assert r.score == 0.0 and r.reason.startswith("invalid_score")


def test_exceptions_fail_closed_with_reason():
    @fail_closed(task="t2_code")
    def boom(x):
        raise KeyError("missing")

    r = boom(1)
    assert r.score == 0.0 and r.reason == "exception:KeyError" and r.task == "t2_code" and r.correct is False


def test_timeouts_map_to_timeout_reason():
    @fail_closed(task="t3_report")
    def slow():
        raise TimeoutError

    assert slow().reason == "timeout"


def test_bad_return_type_fails_closed():
    @fail_closed(task="t1_num")
    def wrong():
        return 1.0

    assert wrong().reason == "exception:BadReturnType"
