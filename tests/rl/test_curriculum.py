"""Tests for the 3-stage curriculum, hard pool gating, auto-reweighting and exact state (Incident 2)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from dpt.rl.curriculum import AutoReweightConfig, Curriculum, CurriculumConfig, Stage

CFG_PATH = Path(__file__).resolve().parents[2] / "configs" / "rl" / "curriculum.yaml"


def _cfg(**over) -> CurriculumConfig:
    raw = OmegaConf.load(CFG_PATH)
    raw.curriculum.seed = 7
    raw.curriculum.prompts_per_step = 128
    d = OmegaConf.to_container(raw.curriculum, resolve=True)
    d.update(over)
    return CurriculumConfig.from_dict(d)


def _pools(n_main=500, n_hard=60):
    return {t: {"main": [f"{t}-m{i}" for i in range(n_main)], "hard": [f"{t}-h{i}" for i in range(n_hard)]}
            for t in ("t1_num", "t2_code", "t3_report", "t4_general")}


def _counts(draws):
    return Counter(d.task for d in draws)


def test_config_file_matches_report_stages():
    c = _cfg()
    assert [(s.start, s.end) for s in c.stages] == [(0, 120), (120, 300), (300, 420)]
    assert c.stages[0].mix["t3_report"] == 0.0 and c.total_steps == 420


def test_stage_mixes_and_t3_absent_in_stage_a():
    cur = Curriculum(_cfg(), _pools())
    assert _counts(cur.sample(0)) == {"t1_num": 64, "t2_code": 32, "t4_general": 32}
    assert _counts(cur.sample(200)) == {"t1_num": 51, "t2_code": 32, "t3_report": 19, "t4_general": 26}
    assert _counts(cur.sample(350)) == {"t1_num": 45, "t2_code": 32, "t3_report": 19, "t4_general": 32}


def test_hard_pool_joins_at_step_150():
    cur = Curriculum(_cfg(), _pools())
    assert not any(d.pool == "hard" for d in cur.sample(149))
    hard = [d for d in cur.sample(150) if d.pool == "hard"]
    assert hard and all(d.uid.split("-")[1].startswith("h") for d in hard)


def test_empty_hard_pool_falls_back_to_main():
    cur = Curriculum(_cfg(), _pools(n_hard=0))
    assert all(d.pool == "main" for d in cur.sample(200))


def test_sampling_is_deterministic_and_without_replacement_within_epoch():
    a, b = Curriculum(_cfg(), _pools()), Curriculum(_cfg(), _pools())
    for s in range(5):
        assert a.sample(s) == b.sample(s)
    c = Curriculum(_cfg(), _pools(n_main=1000))
    uids = [d.uid for s in range(10) for d in c.sample(s) if d.task == "t1_num"]
    assert len(uids) == len(set(uids)) == 640


def test_double_sampling_a_step_is_an_error():
    cur = Curriculum(_cfg(), _pools())
    cur.sample(3)
    with pytest.raises(RuntimeError):
        cur.sample(3)


def test_state_round_trip_continues_identically():
    a = Curriculum(_cfg(), _pools(n_main=100, n_hard=10))
    for s in range(160):
        a.sample(s)
        a.observe(s, {"t1_num": {"zero_adv_frac": 0.3}})
    state = a.state_dict()
    b = Curriculum(_cfg(), _pools(n_main=100, n_hard=10))
    b.load_state_dict(state)
    for s in range(160, 170):
        assert a.sample(s) == b.sample(s)


def test_resume_against_different_data_is_refused():
    a = Curriculum(_cfg(), _pools())
    a.sample(0)
    b = Curriculum(_cfg(), _pools(n_main=499))
    with pytest.raises(ValueError, match="fingerprint"):
        b.load_state_dict(a.state_dict())


def test_resume_with_changed_config_is_refused():
    a = Curriculum(_cfg(), _pools())
    b = Curriculum(_cfg(hard_pool_join_step=100), _pools())
    with pytest.raises(ValueError, match="config changed"):
        b.load_state_dict(a.state_dict())


def _small_cfg(**ar):
    stages = [Stage("A", 0, 1000, {"t1_num": 0.5, "t2_code": 0.25, "t3_report": 0.0, "t4_general": 0.25})]
    return CurriculumConfig(stages=stages, prompts_per_step=16, hard_pool_join_step=0, hard_share=0.1,
                            seed=0, auto_reweight=AutoReweightConfig(window=5, cooldown=5, **ar))


def test_auto_reweight_triggers_on_stall_plus_saturation():
    cur = Curriculum(_small_cfg(), _pools())
    events = []
    for s in range(12):
        events += cur.observe(s, {"t1_num": {"zero_adv_frac": 0.8, "dev_metric": 50.0 if s == 0 else 49.0}})
    assert [e["step"] for e in events] == [5, 10]            # window 5, cooldown 5
    assert cur.weight_factor["t1_num"] == pytest.approx(0.64)
    assert cur.hard_share_extra["t1_num"] == pytest.approx(0.2)
    assert cur.effective_mix(20)["t1_num"] < 0.5


def test_auto_reweight_does_not_trigger_while_dev_improves():
    cur = Curriculum(_small_cfg(), _pools())
    for s in range(20):
        assert not cur.observe(s, {"t1_num": {"zero_adv_frac": 0.9, "dev_metric": float(s)}})


def test_auto_reweight_does_not_trigger_when_zero_adv_is_healthy():
    cur = Curriculum(_small_cfg(), _pools())
    for s in range(20):
        assert not cur.observe(s, {"t2_code": {"zero_adv_frac": 0.4, "dev_metric": 1.0}})


def test_mostly_failed_groups_do_not_get_more_hard_problems():
    cur = Curriculum(_small_cfg(), _pools())
    ev = []
    for s in range(6):
        ev += cur.observe(s, {"t1_num": {"zero_adv_frac": 0.8, "all_solved_frac": 0.1, "dev_metric": 1.0}})
    assert ev and cur.hard_share_extra["t1_num"] == 0.0 and cur.weight_factor["t1_num"] < 1.0


def test_inactive_task_is_never_reweighted():
    cur = Curriculum(_small_cfg(), _pools())
    for s in range(20):
        assert not cur.observe(s, {"t3_report": {"zero_adv_frac": 1.0, "dev_metric": 1.0}})


# ---- difficulty weights and the hard-pool size guard (2026-11-07) ---------------------------------------
def test_difficulty_bands_map_pass_rates_to_copies():
    from dpt.rl.curriculum import DifficultyWeights

    dw = DifficultyWeights.from_dict({"enabled": True})
    assert [dw.copies(r) for r in (0.125, 0.375, 0.5, 0.625, 0.875, None)] == [3, 3, 2, 2, 1, 1]
    with pytest.raises(ValueError):
        DifficultyWeights.from_dict({"enabled": True, "bands": [[0.6, 2], [0.3, 3]]})     # unsorted
    with pytest.raises(ValueError):
        DifficultyWeights.from_dict({"enabled": True, "typo": 1})


def test_hard_problems_are_drawn_more_often_and_the_state_stays_exact():
    pools = {t: {"main": [f"{t}-m{i}" for i in range(300)], "hard": []}
             for t in ("t1_num", "t2_code", "t3_report", "t4_general")}
    rates = {}
    for t in pools:
        for i, u in enumerate(pools[t]["main"]):
            rates[u] = (0.25, 0.5, 0.875)[i % 3]                 # copies 3, 2, 1
    cfg = _cfg(difficulty_weights={"enabled": True})
    cur = Curriculum(cfg, pools, rates)
    seen = Counter()
    for s in range(60):
        seen.update(d.uid for d in cur.sample(s) if d.task == "t1_num")
    by_band = Counter()
    for u, n in seen.items():
        by_band[rates[u]] += n
    assert by_band[0.25] > 1.3 * by_band[0.5] > 1.3 * 1.3 * by_band[0.875]
    again = Curriculum(_cfg(difficulty_weights={"enabled": True}), pools, rates)
    again.load_state_dict(cur.state_dict())                       # same weights: resumes
    with pytest.raises(ValueError):                              # other weights: refused
        Curriculum(_cfg(difficulty_weights={"enabled": True, "bands": [[1.0, 1]]}), pools, rates) \
            .load_state_dict(cur.state_dict())


def test_disabled_weights_keep_the_old_fingerprint_and_draws():
    a, b = Curriculum(_cfg(), _pools()), Curriculum(_cfg(difficulty_weights={"enabled": False}), _pools(), {})
    assert a.state_dict()["config_fingerprint"] == b.state_dict()["config_fingerprint"]
    assert [d.uid for d in a.sample(0)] == [d.uid for d in b.sample(0)]


def test_a_small_hard_pool_gets_a_proportionally_smaller_share():
    cfg = _cfg(hard_pool_join_step=0, hard_share=0.2, hard_pool_min_size=300)
    cur = Curriculum(cfg, _pools(n_hard=30))
    assert abs(cur.hard_share("t1_num", 0) - 0.02) < 1e-9
    assert Curriculum(_cfg(hard_pool_join_step=0, hard_share=0.2), _pools(n_hard=30)).hard_share("t1_num", 0) == 0.2
