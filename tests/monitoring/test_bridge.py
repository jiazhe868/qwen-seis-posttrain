"""verl/dpt.rl metric names → canonical monitoring keys."""

from __future__ import annotations

from dpt.monitoring import keys as K
from dpt.monitoring.bridge import to_canonical


def test_systems_aliases():
    out = to_canonical({
        "mismatch/pos/12k+/log_ratio": -3e-3,
        "rollout/spec_accept_rate": 0.58,
        "actor/perf/max_memory_allocated_gb": 70.0,
        "actor/perf/max_memory_reserved_gb": 74.0,
        "timing_s/gen": 276.0,
    })
    assert out[K.LOG_PPL_DIFF_TAIL] == 3e-3
    assert out[K.MTP_ACCEPTANCE] == 0.58
    assert out[K.MEMORY_PEAK_MAX_RANK] == 74.0
    assert out[K.ROLLOUT_TIME] == 276.0


def test_zero_adv_and_validation_proxy_gold():
    out = to_canonical({
        "filter/t1_num/zero_adv_group_frac": 0.4,
        "val-core/t1_num/acc/mean@4": 0.70,           # verl's core var
        "val-aux/t1_num/score/mean@4": 0.70,          # our keys land in val-aux
        "val-aux/t1_num/gold_score/mean@4": 0.66,
        "val-aux/t2_code/score/mean@4": 0.5,
    })
    assert out["task/t1_num/zero_adv_frac"] == 0.4
    assert out["proxy/t1_num"] == 0.70 and out["gold/t1_num"] == 0.66
    assert out[K.DEV_D_NUM] == 66.0
    assert "gold/t2_code" not in out


def test_sandbox_infra_error_rate_from_the_dev_pass():
    """Incident 23: 79% of the dev T2 samples were ungraded (no apptainer) and scored 0, visible only here."""
    out = to_canonical({"val-aux/t2_code/t2_infra_error/mean@4": 0.79, "val-aux/t4_general/t2_infra_error/mean@4": 0.0})
    assert out == {K.SANDBOX_INFRA_ERROR: 0.79}


def test_missing_sources_are_skipped():
    assert to_canonical({"actor/grad_norm": 1.0}) == {}


def test_mtp_acceptance_falls_back_to_vllm_counters():
    assert to_canonical({"rollout/spec_accept_rate_counters": 0.66})[K.MTP_ACCEPTANCE] == 0.66
    both = {"rollout/spec_accept_rate": 0.7, "rollout/spec_accept_rate_counters": 0.5}
    assert to_canonical(both)[K.MTP_ACCEPTANCE] == 0.7          # verl's per-request metric wins


def test_dev_t1_wrong_values_and_unfinished_answers_in_percent():
    out = to_canonical({"val-aux/t1_num/gold_wrong_value/mean@4": 0.178, "val-aux/t1_num/gold_unfinished/mean@4": 0.0,
                        "val-aux/t4_general/gold_wrong_value/mean@4": -1.0})
    assert abs(out[K.DEV_T1_WRONG_VALUE] - 17.8) < 1e-9 and out[K.DEV_T1_UNFINISHED] == 0.0
    assert "gold/t1_num" not in out                                       # the gold score has its own key
