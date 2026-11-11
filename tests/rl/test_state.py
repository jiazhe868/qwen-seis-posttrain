"""Tests for the extra-state registry, the LR schedule, and resume equivalence (Incident 2)."""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import pytest

from dpt.common.seeding import seed_everything
from dpt.rl.advantage import TaskAdvantageScaler, group_mean_advantage
from dpt.rl.curriculum import AutoReweightConfig, Curriculum, CurriculumConfig, Stage
from dpt.rl.partial_rollout import PartialRolloutBuffer
from dpt.rl.schedule import RLLRSchedule
from dpt.rl.state import DictState, ExtraStateManager, resume_equivalence


def test_lr_schedule_matches_report():
    s = RLLRSchedule()
    assert s.lr(0) == pytest.approx(1e-7)
    assert s.lr(9) == pytest.approx(1e-6)
    assert s.lr(10) == s.lr(299) == 1e-6
    assert s.lr(300) == s.lr(419) == 5e-7
    lin = RLLRSchedule(decay="linear")
    assert 5e-7 < lin.lr(360) < 1e-6


def _curriculum(seed=0):
    stages = [Stage("A", 0, 10, {"t1_num": 0.5, "t2_code": 0.5}),
              Stage("B", 10, 40, {"t1_num": 0.25, "t2_code": 0.25, "t3_report": 0.25, "t4_general": 0.25})]
    cfg = CurriculumConfig(stages=stages, prompts_per_step=8, hard_pool_join_step=12, hard_share=0.25,
                           seed=seed, auto_reweight=AutoReweightConfig(window=3, cooldown=3))
    pools = {t: {"main": [f"{t}{i}" for i in range(37)], "hard": [f"{t}h{i}" for i in range(5)]}
             for t in ("t1_num", "t2_code", "t3_report", "t4_general")}
    return Curriculum(cfg, pools)


class ToyRun:
    """Deterministic stand-in for an RL step: curriculum draws + RNG noise + LR schedule + a parameter.

    ``forget`` drops components from the checkpoint, reproducing Incident 2.
    """

    def __init__(self, forget: tuple[str, ...] = ()):
        seed_everything(0)  # a fresh "process"; resume must restore the RNG it had at save time
        self.cur = _curriculum()
        self.sched = RLLRSchedule(base_lr=0.5, warmup_steps=3, total_steps=40, final_lr=0.1, final_phase_steps=10)
        self.scaler = TaskAdvantageScaler(["t3_report"])
        self.buf = PartialRolloutBuffer(chunk_tokens=4, max_response=8)
        self.alerts = DictState({"fired": 0})
        self.theta = np.zeros(4)
        self.mgr = ExtraStateManager()
        for name, comp in [("curriculum", self.cur), ("scheduler", self.sched), ("adv_scaler", self.scaler),
                           ("partial_rollout", self.buf), ("alerts", self.alerts)]:
            if name not in forget:
                self.mgr.register(name, comp)

    def step(self, step: int) -> float:
        draws = self.cur.sample(step)
        feats = np.array([[zlib.crc32(d.uid.encode()) % 97 / 97.0, d.pool == "hard", step % 3, 1.0]
                          for d in draws])
        rewards = (feats @ np.array([1.0, 0.5, 0.1, 0.0]) + np.random.normal(0, 0.1, len(draws)) > 0.7) * 1.0
        adv = group_mean_advantage(rewards, [i // 2 for i in range(len(draws))])
        adv = self.scaler.update_and_scale(adv, [d.task for d in draws])
        lr = self.sched.step(step)
        self.theta += lr * (feats * adv[:, None]).mean(0)
        self.alerts.value["fired"] += int(rewards.mean() > 0.5)
        self.cur.observe(step, {d.task: {"zero_adv_frac": float(np.mean(adv == 0))} for d in draws})
        return float(-(feats @ self.theta).mean())

    def save(self, ckpt_dir: str, step: int) -> None:
        Path(ckpt_dir, "theta.json").write_text(json.dumps(self.theta.tolist()))
        self.mgr.save(ckpt_dir, step)

    def load(self, ckpt_dir: str) -> int:
        self.theta = np.array(json.loads(Path(ckpt_dir, "theta.json").read_text()))
        return self.mgr.load(ckpt_dir)


def test_resume_equivalence_holds_with_full_state():
    rep = resume_equivalence(ToyRun, total_steps=20, split_step=13, atol=1e-12)
    assert rep.passed, rep


def test_resume_equivalence_detects_missing_curriculum_state():
    """Incident 2 reproduced: without the curriculum in the checkpoint the curves diverge."""

    def make():
        return ToyRun(forget=("curriculum",))

    rep = resume_equivalence(make, total_steps=20, split_step=13, atol=1e-12)
    assert not rep.passed


def test_loading_a_checkpoint_missing_a_component_fails_loudly(tmp_path):
    old = ToyRun(forget=("adv_scaler",))
    old.step(0)
    old.save(str(tmp_path), 0)
    new = ToyRun()
    with pytest.raises(ValueError, match="missing in checkpoint"):
        new.load(str(tmp_path))
    assert new.mgr.load(str(tmp_path), allow_missing=True) == 0


def test_register_rejects_non_stateful_and_duplicates():
    mgr = ExtraStateManager()
    mgr.register("a", DictState())
    with pytest.raises(ValueError):
        mgr.register("a", DictState())
    with pytest.raises(TypeError):
        mgr.register("b", object())


def test_data_hash_change_is_refused(tmp_path):
    mgr = ExtraStateManager()
    mgr.register("x", DictState({"k": 1}))
    mgr.save(tmp_path, 5, data_hashes={"rl_bank": "abc"})
    assert mgr.load(tmp_path, expected_data_hashes={"rl_bank": "abc"}) == 5
    with pytest.raises(ValueError, match="changed since checkpoint"):
        mgr.load(tmp_path, expected_data_hashes={"rl_bank": "def"})


def test_roundtrip_catches_a_component_that_drops_state(tmp_path):
    from dpt.rl.state import DictState, ExtraStateManager

    class Forgetful:
        """Saves two fields but restores only one: load_state_dict 'succeeds' silently."""

        def __init__(self):
            self.a, self.b = 0, 0

        def state_dict(self):
            return {"a": self.a, "b": self.b}

        def load_state_dict(self, s):
            self.a = s["a"]

    src = ExtraStateManager()
    good, bad = DictState({"cursor": 7, "pending": (1, 2)}), Forgetful()
    bad.a, bad.b = 3, 4
    src.register("good", good)
    src.register("bad", bad)
    src.save(tmp_path, step=6)

    dst = ExtraStateManager()
    dst.register("good", DictState())
    dst.register("bad", Forgetful())
    assert dst.load(tmp_path) == 6
    assert dst.last_roundtrip_mismatches == ["bad"]          # tuple vs list in "good" is not a mismatch
