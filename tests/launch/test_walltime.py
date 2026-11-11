"""Wall-clock stop before the batch scheduler's limit (the cluster: 48 h jobs, RL spans several)."""

from __future__ import annotations

import subprocess

import pytest

from dpt.launch.walltime import WallClock, parse_slurm_duration, slurm_time_left


@pytest.mark.parametrize("text, seconds", [
    ("47:59:59", 47 * 3600 + 59 * 60 + 59),
    ("1-23:30:00", 47 * 3600 + 30 * 60),
    ("2-00", 48 * 3600),
    ("1-02:03", 26 * 3600 + 3 * 60),
    ("12:34", 12 * 60 + 34),
    ("45", 45 * 60),
])
def test_parse_slurm_duration(text, seconds):
    assert parse_slurm_duration(text) == seconds


@pytest.mark.parametrize("text", ["", "UNLIMITED", "INVALID", "abc", "1:2:3:4"])
def test_parse_slurm_duration_rejects_garbage(text):
    with pytest.raises(ValueError):
        parse_slurm_duration(text)


def test_slurm_time_left_reads_squeue():
    def fake(cmd, **kw):
        assert cmd[:4] == ["squeue", "-h", "-j", "123"]
        return subprocess.CompletedProcess(cmd, 0, stdout="1-10:00:00\n")

    assert slurm_time_left("123", runner=fake) == 34 * 3600


def test_no_deadline_never_stops():
    wc = WallClock(None)
    for t in range(0, 10_000, 600):
        wc.observe_step(t)
        assert wc.should_stop(t) == (False, "")


def test_stops_when_next_step_and_save_no_longer_fit():
    deadline = 10 * 3600
    wc = WallClock(deadline, safety=1.5, save_estimate_s=900)
    t = 0.0
    stopped_at = None
    while t < deadline:
        wc.observe_step(t)
        stop, why = wc.should_stop(t)
        if stop:
            stopped_at = t
            assert "wall-clock limit" in why
            break
        t += 12 * 60                                    # 12-minute steps
    # One more step + save needs 1.5 × 12 min + 15 min = 33 min. The clock stops at the first step
    # boundary with less than that left (another step would not fit), and the checkpoint (15 min)
    # still fits in what is left.
    assert stopped_at is not None
    left = deadline - stopped_at
    assert 15 * 60 <= left < 33 * 60
    assert deadline - (stopped_at - 12 * 60) >= 33 * 60    # one step earlier, there was still room


def test_measured_save_time_replaces_the_estimate():
    wc = WallClock(3600 * 5, save_estimate_s=900)
    wc.observe_step(0)
    wc.observe_step(600)
    assert wc.needed_s() == pytest.approx(1.5 * 600 + 900)
    wc.observe_save(100)
    assert wc.needed_s() == pytest.approx(1.5 * 600 + 120)


def test_slowest_recent_step_dominates():
    wc = WallClock(3600 * 5, window=3)
    for t in (0, 600, 1200, 3000, 3600):               # one 30-minute step (a dev validation step)
        wc.observe_step(t)
    assert wc.needed_s() == pytest.approx(1.5 * 1800 + 900)


def test_before_any_timed_step_only_stops_past_the_line():
    wc = WallClock(1000.0, save_estimate_s=900)
    wc.observe_step(0)                                  # first call only starts the clock
    assert wc.should_stop(50)[0] is False               # 950 s left > 900 s save estimate
    assert wc.should_stop(200)[0] is True


def test_an_extended_time_limit_is_picked_up_while_running():
    """Site staff may extend a running job; the trainer must not stop at the old limit."""
    limits = iter([10_000.0, 400_000.0])                     # Slurm deadline before / after the extension
    wc = WallClock(10_000.0, refresh=lambda now: next(limits), refresh_every_s=1800, save_estimate_s=600)
    t = 0.0
    for _ in range(4):                                        # 15-minute steps
        wc.observe_step(t)
        t += 900
    assert wc.should_stop(8_500)[0]                           # old limit: 1,500 s left < 1,950 s needed
    wc._refreshed = 0.0
    assert not wc.should_stop(9_000)[0] and wc.deadline == 400_000.0   # refreshed: keep going


def test_from_env_refreshes_from_squeue_only_inside_slurm(monkeypatch):
    import dpt.launch.walltime as w

    monkeypatch.setattr(w, "slurm_time_left", lambda job=None, **kw: 3600)
    wc = w.WallClock.from_env({"DPT_DEADLINE": "1000", "DPT_DEADLINE_MARGIN_S": "600", "SLURM_JOB_ID": "7"})
    assert wc.refresh is not None and wc.refresh(100.0) == 100.0 + 3600 - 600
    assert w.WallClock.from_env({"DPT_DEADLINE": "1000"}).refresh is None


def test_a_failing_refresh_keeps_the_deadline():
    def boom(now):
        raise RuntimeError("squeue down")

    wc = WallClock(5_000.0, refresh=boom)
    wc.should_stop(100.0)
    assert wc.deadline == 5_000.0
