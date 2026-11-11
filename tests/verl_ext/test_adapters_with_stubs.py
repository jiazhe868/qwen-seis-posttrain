"""Import the verl-facing adapters against stub verl/TransferQueue modules.

verl is not installed in the CPU test environment. These stubs provide only the names the
adapters import, so import-time errors, registration and the replay-buffer filtering logic are
still exercised on CPU. Full behaviour is validated by scripts/05_rl_smoke.sh on GPUs.
"""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest
import torch


class _KVBatchMeta:
    def __init__(self, partition_id, keys, tags):
        self.partition_id, self.keys, self.tags, self.extra_info = partition_id, list(keys), list(tags), {}


class _FakeTQ(types.ModuleType):
    def __init__(self):
        super().__init__("transfer_queue")
        self.store: dict[str, dict] = {}
        self.cleared: list[str] = []
        self.KVBatchMeta = _KVBatchMeta

    def kv_batch_get(self, keys, partition_id, select_fields):
        return {f: (torch.stack([self.store[k][f] for k in keys]) if f != "data_source"
                    else np.array([self.store[k][f] for k in keys], dtype=object)) for f in select_fields}

    def kv_clear(self, partition_id, keys):
        self.cleared.extend(keys)


@pytest.fixture
def stubs(monkeypatch):
    tq = _FakeTQ()
    mods = {"transfer_queue": tq}

    def mod(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        mods[name] = m
        return m

    registry: dict = {}
    trainers: dict = {}

    def register_adv_est(name):
        def deco(fn):
            registry[name] = fn
            return fn
        return deco

    def register_trainer(name):
        def deco(cls):
            trainers[name] = cls
            return cls
        return deco

    class _RB:
        def __init__(self, *a, **k):
            self.cleared_groups = []

        def sample(self, global_steps, partition_id, batch_size):
            return self._next, {}

        def _clear_groups(self, partition_id, uids):
            self.cleared_groups.append(set(uids))

    class _Trainer:
        def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
            pass

        def _save_checkpoint(self):
            self.base_saved = True

        class checkpoint_manager:                      # noqa: N801 - mimics verl's attribute
            calls: list = []

            @classmethod
            def abort_replicas(cls):
                cls.calls.append("abort")

            @classmethod
            def sleep_replicas(cls):
                cls.calls.append("sleep")

        def _load_checkpoint(self):
            return None

    mod("verl")
    mod("verl.trainer")
    mod("verl.trainer.ppo")
    mod("verl.trainer.ppo.core_algos", ADV_ESTIMATOR_REGISTRY=registry, register_adv_est=register_adv_est)
    mods["verl.trainer.ppo"].core_algos = mods["verl.trainer.ppo.core_algos"]
    mod("verl.trainer.ppo.v1")
    mod("verl.trainer.ppo.v1.replay_buffer", ReplayBufferAsync=_RB)
    mod("verl.trainer.ppo.v1.trainer_base", register_trainer=register_trainer)
    mod("verl.trainer.ppo.v1.trainer_colocate_async", PPOTrainerColocateAsync=_Trainer)
    mod("verl.trainer.ppo.ray_trainer", apply_kl_penalty=None)
    mod("verl.trainer.ppo.rollout_corr_helper", compute_rollout_correction_and_add_to_batch=None)
    mod("verl.protocol", DataProto=object)
    mod("verl.utils")
    mod("verl.utils.dataset")
    mod("verl.utils.dataset.rl_dataset", collate_fn=None)
    mod("verl.workers")
    mod("verl.workers.utils")
    mod("verl.workers.utils.padding", response_to_nested=None)
    mod("tensordict", TensorDict=dict)
    mod("torchdata")
    mod("torchdata.stateful_dataloader", StatefulDataLoader=object)
    for name, m in mods.items():
        monkeypatch.setitem(sys.modules, name, m)
    for name in ("dpt.verl_ext.register", "dpt.verl_ext.trainer", "dpt.verl_ext.replay_buffer"):
        sys.modules.pop(name, None)
    yield types.SimpleNamespace(tq=tq, registry=registry, trainers=trainers)
    for name in ("dpt.verl_ext.register", "dpt.verl_ext.trainer", "dpt.verl_ext.replay_buffer"):
        sys.modules.pop(name, None)


def test_register_installs_estimator_and_trainer(stubs):
    importlib.import_module("dpt.verl_ext.register")
    assert "dpt_task_grpo" in stubs.registry
    assert "dpt_colocate_async" in stubs.trainers
    adv, _ = stubs.registry["dpt_task_grpo"](torch.tensor([[1.0], [0.0]]), torch.ones(2, 3), np.array(["g", "g"]))
    assert torch.allclose(adv[:, 0], torch.tensor([0.5, -0.5]))


def test_replay_buffer_drops_zero_advantage_groups(stubs):
    from dpt.common.config import to_container
    from dpt.launch.common import compose_stage
    from dpt.verl_ext import runtime

    runtime.set_runtime(runtime.build_runtime(to_container(compose_stage("rl/rl_27b.yaml"))))
    try:
        rb_mod = importlib.import_module("dpt.verl_ext.replay_buffer")
        keys, rewards = [], {"gA": [1, 0, 1, 0], "gB": [1, 1, 1, 1], "gC": [0, 0, 0, 0],
                             "gD": [1, 0, 0, 0], "gE": [0, 1, 0, 0], "gF": [1, 1, 0, 0]}
        for g, rs in rewards.items():
            for i, r in enumerate(rs):
                k = f"{g}_0_{i}"
                keys.append(k)
                stubs.tq.store[k] = {"rm_scores": torch.tensor([0.0, 0.0, float(r)]),
                                     "response_mask": torch.ones(3), "data_source": "t1_num"}
        rb = rb_mod.ZeroAdvantageFilteringReplayBuffer()
        rb._next = _KVBatchMeta("train", keys, [{} for _ in keys])
        batch, metrics = rb.sample(1, "train", len(keys))
        kept_groups = {k.split("_")[0] for k in batch.keys}
        assert kept_groups == {"gA", "gD", "gE", "gF"}               # all-1 and all-0 groups dropped
        assert {k for k in stubs.tq.cleared} == {f"g{x}_0_{i}" for x in "BC" for i in range(4)}
        assert metrics["filter/t1_num/zero_adv_group_frac"] == pytest.approx(2 / 6)
        # panel metrics are computed on the full batch (24 samples, 10 correct), not the kept 16
        assert metrics["task/t1_num/n_samples"] == 24
        assert metrics["task/t1_num/acc"] == pytest.approx(10 / 24)
    finally:
        runtime.set_runtime(None)


def test_trainer_auto_pauses_on_red_mismatch_alert(stubs, tmp_path):
    """Incident 1 guard end to end: log-ppl diff above 5e-3 → checkpoint + PAUSED.json + loop exit."""
    import json

    from omegaconf import OmegaConf

    from dpt.common.config import to_container
    from dpt.launch.common import compose_stage
    from dpt.rl.state import DictState, ExtraStateManager
    from dpt.verl_ext import runtime

    runtime.set_runtime(runtime.build_runtime(to_container(compose_stage("rl/rl_27b.yaml"))))
    try:
        trainer_mod = importlib.import_module("dpt.verl_ext.trainer")
        t = trainer_mod.DomainRLTrainer()
        t.config = OmegaConf.create({"trainer": {"default_local_dir": str(tmp_path)}})
        t.global_steps, t.total_training_steps = 37, 420
        t.dpt_state = ExtraStateManager()
        t._alert_state = DictState()
        t.dpt_state.register("alerts", t._alert_state)

        healthy = {"mismatch/log_ppl_abs_diff": 4e-4, "task/t1_num/reward_mean": 0.5}
        t._compute_metrics(None, dict(healthy), {}, 36, 0)
        assert t.total_training_steps == 420 and not (tmp_path / "PAUSED.json").exists()

        red = {"mismatch/log_ppl_abs_diff": 8e-3, "task/t1_num/reward_mean": 0.5}
        t._compute_metrics(None, red, {}, 37, 0)
        assert t.total_training_steps == 37                       # fit loop exits
        assert getattr(t, "base_saved", False)                    # verl checkpoint written
        assert t.checkpoint_manager.calls[-2:] == ["abort", "sleep"], "vLLM must sleep before the save"
        assert (tmp_path / "global_step_37" / "dpt_extra_state.json").exists()
        pause = json.loads((tmp_path / "PAUSED.json").read_text())
        assert pause["step"] == 37 and "log_ppl" in pause["reason"]
        assert any(k.startswith("alert/") for k in red)
        assert (tmp_path / "metrics.jsonl").exists()
        # the pause checkpoint's extra state was refreshed AFTER this step's alerts were evaluated
        state = json.loads((tmp_path / "global_step_37" / "dpt_extra_state.json").read_text())
        assert state["components"]["alerts"], "alert history must be in the pause checkpoint"
    finally:
        runtime.set_runtime(None)
