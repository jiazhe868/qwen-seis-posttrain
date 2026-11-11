from __future__ import annotations

from dpt.common.io import write_jsonl
from dpt.monitoring.compare_runs import compare


def _log(path, stage_at_11, *, lr_restart=False, reward_noise=0.0, roundtrip=None):
    rows = []
    for s in range(1, 21):
        lr = 1e-6 * min(1.0, (s - 10 if lr_restart and s > 10 else s) / 10)    # 10-step warm-up
        r = {"step": s, "training/global_step": s, "curriculum/stage": (stage_at_11 if s >= 11 else 0.0),
             "actor/lr": lr, "task/t1_num/reward_mean": 0.5 + 0.01 * s + reward_noise * (-1) ** s}
        if roundtrip is not None and s == 11:
            r["resume/state_roundtrip_ok"] = roundtrip
        rows.append(r)
    write_jsonl(path, rows)


def _cmp(tmp_path, **kw):
    return compare(str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl"), range(11, 21), **kw)


def test_identical_resume_passes(tmp_path):
    _log(tmp_path / "a.jsonl", 1.0)
    _log(tmp_path / "b.jsonl", 1.0, roundtrip=1.0)
    rep = _cmp(tmp_path)
    assert rep["pass"] and rep["keys"]["resume/state_roundtrip_ok"]["ok"]


def test_curriculum_restart_is_caught(tmp_path):
    """Incident 2: the resumed run fell back to stage A."""
    _log(tmp_path / "a.jsonl", 1.0)
    _log(tmp_path / "b.jsonl", 0.0)
    rep = _cmp(tmp_path)
    assert not rep["pass"] and not rep["keys"]["curriculum/stage"]["ok"]


def test_lr_warmup_restart_is_caught(tmp_path):
    _log(tmp_path / "a.jsonl", 1.0)
    _log(tmp_path / "b.jsonl", 1.0, lr_restart=True)
    rep = _cmp(tmp_path)
    assert not rep["pass"] and not rep["keys"]["actor/lr"]["ok"]


def test_failed_roundtrip_is_caught(tmp_path):
    _log(tmp_path / "a.jsonl", 1.0)
    _log(tmp_path / "b.jsonl", 1.0, roundtrip=0.0)
    assert not _cmp(tmp_path)["pass"]


def test_stochastic_rewards_are_informational_unless_deterministic(tmp_path):
    """Async partial rollout makes samples differ between correct runs; rewards must not decide the test."""
    _log(tmp_path / "a.jsonl", 1.0)
    _log(tmp_path / "b.jsonl", 1.0, reward_noise=0.2)
    rep = _cmp(tmp_path)
    assert rep["pass"] and not rep["informational"]["task/t1_num/reward_mean"]["ok"]
    assert not _cmp(tmp_path, deterministic=True)["pass"]
