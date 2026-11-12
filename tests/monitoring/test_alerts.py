"""Alert engine: every panel rule fires on its failure pattern and stays quiet in the normal range."""

from __future__ import annotations

import json

import pytest

from dpt.common.config import load_config, to_container
from dpt.monitoring.alerts import AlertManager, build_rule


@pytest.fixture()
def mgr():
    cfg = to_container(load_config("rl/monitoring.yaml", profile="cluster_2x8xa100_80g"))["monitoring"]
    return AlertManager.from_config(cfg)


def _run(mgr, series):
    """series: list of metric dicts, one per step (starting at step 1). Returns alerts per step."""
    return [mgr.evaluate(i + 1, m) for i, m in enumerate(series)]


def test_incident1_logppl_warns_then_pauses(mgr):
    """Incident 1: the log-ppl gap rose ~60 steps before collapse. Warn at 2e-3, auto-pause at 5e-3."""
    vals = [5e-4] * 10 + [1.5e-3, 2.5e-3, 3e-3, 4e-3, 6e-3]
    severities, paused = [], []
    for i, v in enumerate(vals):
        alerts = mgr.evaluate(i + 1, {"mismatch/log_ppl_abs_diff": v})
        severities.append([a.severity for a in alerts])
        paused.append(mgr.should_pause())
    assert severities[:11] == [[]] * 11
    assert severities[11] == ["warn"] and not paused[11]
    assert severities[-1] == ["pause"] and paused[-1]


def test_incident3_parity_failure_blocks_rollout(mgr):
    mgr.evaluate(1, {"parity/post_sync_pass": 1.0})
    assert not mgr.rollout_blocked()
    alerts = mgr.evaluate(2, {"parity/post_sync_pass": 0.0})
    assert alerts[0].severity == "block_rollout" and mgr.rollout_blocked()


def test_incident5_mtp_acceptance(mgr):
    assert not mgr.evaluate(1, {"spec/mtp_acceptance_rate": 0.71})
    assert mgr.evaluate(2, {"spec/mtp_acceptance_rate": 0.46})[0].rule == "mtp_acceptance"


def test_incident23_sandbox_infra_errors_pause(mgr):
    assert not mgr.evaluate(1, {"sandbox/t2_infra_error": 0.0})
    assert [a.severity for a in mgr.evaluate(2, {"sandbox/t2_infra_error": 0.05})] == ["warn"]
    alerts = mgr.evaluate(3, {"sandbox/t2_infra_error": 0.79})
    assert alerts[0].rule == "sandbox_infra" and alerts[0].severity == "pause" and mgr.should_pause()


def test_memory_threshold_follows_hardware_profile():
    local = AlertManager.from_config(
        to_container(load_config("rl/monitoring.yaml", profile="local_4xa100_40g"))["monitoring"]
    )
    assert local.evaluate(1, {"memory/peak_gb_max_rank": 37.0})
    cluster = AlertManager.from_config(
        to_container(load_config("rl/monitoring.yaml", profile="cluster_2x8xa100_80g"))["monitoring"]
    )
    assert not cluster.evaluate(1, {"memory/peak_gb_max_rank": 37.0})
    assert cluster.evaluate(2, {"memory/peak_gb_max_rank": 77.0})


def test_entropy_collapse_per_task(mgr):
    series = [{"task/t1_num/entropy": 1.0, "task/t2_code/entropy": 1.0} for _ in range(12)]
    series += [{"task/t1_num/entropy": 0.65, "task/t2_code/entropy": 0.98}]
    out = _run(mgr, series)
    assert [a.metric for a in out[-1]] == ["task/t1_num/entropy"]


def test_entropy_explosion(mgr):
    out = _run(mgr, [{"task/t3_report/entropy": 1.0}] * 12 + [{"task/t3_report/entropy": 1.6}])
    assert out[-1] and "rose" in out[-1][0].message


def test_zero_adv_persistent_30_steps(mgr):
    out = _run(mgr, [{"task/t1_num/zero_adv_frac": 0.65}] * 31)
    fired = [i for i, a in enumerate(out) if a]
    assert fired == [29]  # fires exactly once, on the 30th consecutive step


def test_grad_norm_spike(mgr):
    out = _run(mgr, [{"actor/grad_norm": 1.0}] * 10 + [{"actor/grad_norm": 6.0}])
    assert out[-1] and out[-1][0].rule == "grad_norm_spike"


def test_wrong_length_grows_faster_than_right(mgr):
    series = []
    for s in range(25):
        series.append({"task/t1_num/len_incorrect": 5000 * (1 + 0.02 * s), "task/t1_num/len_correct": 5000.0})
    out = _run(mgr, series)
    assert any(a.rule == "wrong_length_growth" for step in out for a in step)


def test_proxy_gold_divergence_three_evals(mgr):
    evals = [(0.50, 0.40), (0.55, 0.40), (0.60, 0.39), (0.66, 0.38)]
    out = [
        mgr.evaluate(20 * (i + 1), {"proxy/t3_report": p, "gold/t3_report": g})
        for i, (p, g) in enumerate(evals)
    ]
    assert not any(out[:3]) and out[3][0].rule == "proxy_gold"


def test_hack_rate_growth(mgr):
    series = [{"hack/self_praise/rate": 0.004} for _ in range(20)] + [
        {"hack/self_praise/rate": 0.02} for _ in range(20)
    ]
    out = _run(mgr, series)
    assert any(a.rule == "hack_growth" for step in out for a in step)


def test_dev_d_num_three_drops_beyond_ci(mgr):
    vals = [(65.0, 1.6), (64.5, 1.6), (64.0, 1.6), (62.9, 1.6)]
    out = [
        mgr.evaluate(20 * (i + 1), {"dev/d_num": v, "dev/d_num_ci_halfwidth": c})
        for i, (v, c) in enumerate(vals)
    ]
    assert out[3] and out[3][0].rule == "dev_d_num"
    small = AlertManager(
        [build_rule({"name": "d", "kind": "consecutive_drop", "metric": "x", "default_ci": 1.6})]
    )
    assert not any(small.evaluate(i, {"x": v}) for i, v in enumerate([65.0, 64.9, 64.8, 64.7]))


def test_general_replay_drop(mgr):
    out = _run(
        mgr, [{"task/t4_general/reward_mean": 0.70}] * 10 + [{"task/t4_general/reward_mean": 0.67}] * 10
    )
    assert out[-1] and out[-1][0].rule == "general_replay"


def test_normal_ranges_are_quiet(mgr):
    normal = {
        "mismatch/log_ppl_abs_diff": 4e-4,
        "parity/post_sync_pass": 1.0,
        "actor/pg_clipfrac": 0.03,
        "actor/grad_norm": 1.0,
        "spec/mtp_acceptance_rate": 0.7,
        "memory/peak_gb_max_rank": 62.0,
        "task/t1_num/truncation_rate": 0.02,
        "task/t1_num/zero_adv_frac": 0.3,
        "task/t1_num/entropy": 0.9,
        "task/t4_general/reward_mean": 0.7,
    }
    assert not any(_run(mgr, [normal] * 50))


def test_warning_cooldown_but_pause_never_suppressed():
    m = AlertManager(
        [build_rule({"name": "c", "kind": "threshold", "metric": "x", "above": {"warn": 1.0, "pause": 5.0}})],
        cooldown_steps=10,
    )
    assert m.evaluate(1, {"x": 2.0}) and not m.evaluate(2, {"x": 2.0}) and m.evaluate(12, {"x": 2.0})
    assert m.evaluate(13, {"x": 6.0}) and m.evaluate(14, {"x": 6.0})


def test_state_dict_roundtrip_gives_identical_alerts(mgr):
    """Incident 2 lesson: resume must not change behavior. Same stream → same alerts after reload."""
    series = [{"task/t1_num/zero_adv_frac": 0.7, "actor/grad_norm": 1.0}] * 20
    _run(mgr, series)
    state = json.loads(json.dumps(mgr.state_dict()))  # must be JSON-serializable
    cfg = to_container(load_config("rl/monitoring.yaml", profile="cluster_2x8xa100_80g"))["monitoring"]
    resumed = AlertManager.from_config(cfg)
    resumed.load_state_dict(state)
    tail = [{"task/t1_num/zero_adv_frac": 0.7, "actor/grad_norm": 1.0}] * 10 + [{"actor/grad_norm": 9.0}]
    a = [[x.rule for x in step] for step in [mgr.evaluate(21 + i, m) for i, m in enumerate(tail)]]
    b = [[x.rule for x in step] for step in [resumed.evaluate(21 + i, m) for i, m in enumerate(tail)]]
    assert a == b and ["zero_adv"] in a and ["grad_norm_spike"] in a


def test_bad_config_rejected():
    with pytest.raises(ValueError):
        build_rule({"name": "x", "kind": "nope"})
    with pytest.raises(ValueError):
        build_rule({"name": "x", "kind": "threshold", "metric": "m", "action": "explode"})


def test_best_drop_pauses_on_the_first_dev_pass_after_the_turn():
    """Run 1's dev D-Num (92.0, 97.3, 78.9 at steps 0, 20, 40) would have paused at step 40, not 56."""
    from dpt.monitoring.alerts import AlertManager

    mgr = AlertManager.from_config({"rules": [], "extra_rules": [
        {"name": "dev_best_drop", "kind": "best_drop", "metric": "dev/d_num", "max_drop": 3.0, "action": "pause"}]})
    assert mgr.evaluate(0, {"dev/d_num": 92.0}) == []
    assert mgr.evaluate(20, {"dev/d_num": 97.3}) == []
    assert mgr.evaluate(30, {"dev/d_num": 95.0}) == []                 # 2.3 below the best: within the limit
    fired = mgr.evaluate(40, {"dev/d_num": 78.9})
    assert [a.severity for a in fired] == ["pause"] and "below its best 97.3" in fired[0].message
    state = mgr.rules[0].state_dict()
    again = AlertManager.from_config({"rules": [], "extra_rules": [
        {"name": "dev_best_drop", "kind": "best_drop", "metric": "dev/d_num", "max_drop": 3.0, "action": "pause"}]})
    again.rules[0].load_state_dict(state)
    assert again.rules[0].best["dev/d_num"] == 97.3                    # the best survives a resume


def test_best_drop_with_better_low_pauses_when_wrong_values_rise_above_their_lowest():
    """Run 1's T1 dev answers with a wrong value: 2.1%, 1.8%, 17.8% at steps 0, 20, 40 (val_generations)."""
    from dpt.monitoring.alerts import AlertManager

    spec = {"name": "dev_t1_wrong_value_rise", "kind": "best_drop", "better": "low", "metric": "dev/t1_wrong_value",
            "max_drop": 3.0, "action": "pause"}
    mgr = AlertManager.from_config({"rules": [], "extra_rules": [spec]})
    assert mgr.evaluate(0, {"dev/t1_wrong_value": 2.1}) == []
    assert mgr.evaluate(20, {"dev/t1_wrong_value": 1.8}) == []
    assert mgr.evaluate(30, {"dev/t1_wrong_value": 4.5}) == []          # 2.7 above the lowest: within the limit
    fired = mgr.evaluate(40, {"dev/t1_wrong_value": 17.8})
    assert [a.severity for a in fired] == ["pause"] and "above its best 1.8" in fired[0].message
    assert mgr.rules[0].best["dev/t1_wrong_value"] == 1.8
