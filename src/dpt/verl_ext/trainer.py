"""DomainRLTrainer: verl's colocate_async v1 trainer plus the project's RL rules.

Registered as ``trainer.v1.trainer_mode=dpt_colocate_async`` by
:mod:`dpt.verl_ext.register`. Base class: ``PPOTrainerColocateAsync``.
Training and vLLM rollout share all GPUs, and responses still generating when
a batch completes are aborted and resumed after the weight sync (partial
rollout, report §单步时间预算).

Overrides and why
-----------------
``_init_dataloader``   curriculum sampler (3 stages, hard pool at 150, auto-reweighting)
``_compute_advantage`` task-aware advantages (``dpt.rl.pipeline``): group mean without std,
                       T3 running-std scaling, fixed per-task loss budgets folded into
                       advantages, truncated samples out of loss *and* baseline; per-task
                       and per-position train/rollout mismatch metrics (Incidents 1, 3)
``_compute_metrics``   curriculum feedback + alert engine; auto-pause on red alerts
``_save_checkpoint``   writes ``dpt_extra_state.json`` (sampler/curriculum, scaler, alerts,
                       RNG, data hashes) *before* verl marks the checkpoint as latest
``_load_checkpoint``   restores it (Incident 2: resume must be exact)

Everything numerical is delegated to framework-free functions in ``dpt.rl`` that
are unit-tested on CPU. This class only moves data between TransferQueue and those
functions. Verified against verl 0.9.1 source; validate on the smoke run
(``scripts/05_rl_smoke.sh``) before every verl upgrade.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import transfer_queue as tq
from omegaconf import OmegaConf
from tensordict import TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader
from transfer_queue import KVBatchMeta

from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import apply_kl_penalty
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_colocate_async import PPOTrainerColocateAsync
from verl.utils.dataset.rl_dataset import collate_fn
from verl.workers.utils.padding import response_to_nested

from dpt.common.logging import get_logger
from dpt.common.types import TASKS
from dpt.launch.walltime import STOP_AT_ENV, WallClock
from dpt.monitoring.bridge import to_canonical
from dpt.monitoring.hack_rate import HackRateTracker
from dpt.monitoring.sinks import JsonlSink
from dpt.monitoring.spec_scrape import SpecDecodeScraper
from dpt.rl.mismatch import fresh_sample_mask, fresh_stale_metrics
from dpt.rl.pipeline import compute_step_advantages
from dpt.rl.state import DictState, ExtraStateManager
from dpt.verl_ext.runtime import get_runtime, to_str_list
from dpt.verl_ext.sampler import CurriculumSampler

log = get_logger(__name__)


class _ScalerState:
    def __init__(self, rt):
        self.rt = rt

    def state_dict(self):
        return self.rt.scaler.state_dict()

    def load_state_dict(self, s):
        self.rt.scaler.load_state_dict(s)


class _HackState:
    """Window history of the hack-rate tracker, so growth detection survives a resume."""

    def __init__(self, tracker: HackRateTracker, seen: set):
        self.tracker, self.seen = tracker, seen

    def state_dict(self):
        return {"steps": [[s, n, c] for s, n, c in self.tracker.steps], "seen": sorted(self.seen)}

    def load_state_dict(self, st):
        self.tracker.steps.clear()
        self.tracker.steps.extend((int(s), int(n), dict(c)) for s, n, c in st.get("steps", []))
        self.seen.clear()
        self.seen.update(st.get("seen", []))


@register_trainer("dpt_colocate_async")
class DomainRLTrainer(PPOTrainerColocateAsync):
    # ------------------------------------------------------------------ data / curriculum
    def _init_dataloader(self):
        super()._init_dataloader()
        rt = get_runtime()
        self.dpt_sampler = CurriculumSampler(self.train_dataset, rt.cfg["curriculum"], rt.cfg["rl"]["total_steps"])
        rt.curriculum = self.dpt_sampler.curriculum
        gen_bs = self.config.data.get("gen_batch_size", None) or self.config.data.train_batch_size
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset, batch_size=gen_bs,
            num_workers=self.config.data["dataloader_num_workers"], drop_last=True,
            collate_fn=collate_fn, sampler=self.dpt_sampler,
        )
        self.train_dataloader_it = None
        # verl derived steps_per_epoch from len(dataset) // batch_size. The curriculum is one continuous
        # stream with no epochs (it cycles pools as needed), so one "epoch" spans the whole run. A bank
        # smaller than one batch (the toy smoke bank) would otherwise give 0 and divide by zero.
        self.steps_per_epoch = max(1, len(self.dpt_sampler) // int(self.config.data.train_batch_size))
        self.dpt_state = ExtraStateManager()
        self.dpt_state.register("sampler", self.dpt_sampler)
        self.dpt_state.register("adv_scaler", _ScalerState(rt))
        self._alert_state = DictState()
        self.dpt_state.register("alerts", self._alert_state)
        self._hack_tracker, self._hack_seen = HackRateTracker(), set()
        self.dpt_state.register("hack_rates", _HackState(self._hack_tracker, self._hack_seen))
        self._last_saved_step = -1

    # ------------------------------------------------------------------ advantages
    def _compute_advantage(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        rt = get_runtime()
        rl = rt.cfg["rl"]
        fields = ["uid", "response_mask", "rm_scores", "rollout_log_probs", "old_log_probs"]
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
        nested_mask = data["response_mask"]
        # verl pads the batch to a multiple of lcm(dp, mini_batch×n) with synthetic samples tagged
        # is_padding (loss mask 0). Their non-tensor fields are copies of a real sample's, so they are
        # identified by the tag and excluded from every statistic.
        is_padding = np.array([bool(t.get("is_padding", False)) if isinstance(t, dict) else False
                               for t in batch.tags])
        real_keys = [k for k, p in zip(batch.keys, is_padding) if not p]
        meta = tq.kv_batch_get(keys=real_keys, partition_id=batch.partition_id, select_fields=["data_source"])
        it = iter(to_str_list(meta["data_source"]))
        task_ids = ["" if p else next(it) for p in is_padding]
        metrics.update(self._hack_rates(batch.partition_id, real_keys))

        data = DataProto(batch=data.to_padded_tensor())
        data.batch["token_level_scores"] = data.batch["rm_scores"]
        data.non_tensor_batch["uid"] = np.array(data.batch.pop("uid").tolist(), dtype=object)
        lengths = data.batch["response_mask"].sum(-1).cpu().numpy()       # before any RS masking

        if self.config.algorithm.use_kl_in_reward:                            # off in our config (KL = 0)
            data, kl_metrics = apply_kl_penalty(data, kl_ctrl=self.kl_ctrl_in_reward,
                                                kl_penalty=self.config.algorithm.kl_penalty)
            metrics.update(kl_metrics)
        else:
            data.batch["token_level_rewards"] = data.batch["token_level_scores"]

        # Train/rollout mismatch, sliced by task and token position (Incident 1 lived past 12K
        # tokens; Incident 3 only on T2). Computed before RS changes the mask.
        #
        # Partial rollout: a sample started under an older policy version carries that version's
        # log-probs for its early tokens, while old_log_probs come from the current weights. Its gap
        # is engine mismatch PLUS policy drift. The Incident 1/3 alerts need engine mismatch alone,
        # so `mismatch/*` uses fresh samples only (all tokens from the current weights, version
        # global_steps − 1 in verl's tags). Stale samples are reported as `drift/*`. Found on the
        # smoke run: with lr 1e-5 the mixed metric hit 4.4e-2 at step 4 (pause at 5e-3); step 1 had 8e-4.
        if "rollout_log_probs" in data.batch and "old_log_probs" in data.batch:
            mm = rl["mismatch"]
            real = ~is_padding
            fresh = fresh_sample_mask(batch.tags, version=self.global_steps - 1)[real]
            rmask = torch.as_tensor(real)
            metrics.update(fresh_stale_metrics(
                data.batch["old_log_probs"][rmask], data.batch["rollout_log_probs"][rmask],
                data.batch["response_mask"][rmask], fresh,
                task_ids=[t for t, p in zip(task_ids, is_padding) if not p],
                position_edges=mm["position_edges"], length_edges=mm["length_edges"],
                min_fresh_sequences=int(mm.get("min_fresh_sequences", 1)),
                min_fresh_tokens=float(mm.get("min_fresh_tokens", 0))))

        # Per-task entropy estimate: E_{x~π}[-log π(x)] is the policy entropy per token, so the masked mean
        # of -old_log_probs over sampled tokens is an unbiased Monte-Carlo estimate (temperature 1, top_p 1).
        if "old_log_probs" in data.batch:
            nll = -data.batch["old_log_probs"].float() * data.batch["response_mask"].float()
            tok = data.batch["response_mask"].float().sum(-1)
            for t in sorted(set(task_ids) - {""}):
                sel = torch.as_tensor([x == t for x in task_ids])
                if sel.any() and tok[sel].sum() > 0:
                    metrics[f"task/{t}/entropy"] = float(nll[sel].sum() / tok[sel].sum())

        # TIS (token-level, cap 2.0) + sequence-level rejection, verl's implementation.
        rc_cfg = self.config.algorithm.get("rollout_correction", None)
        use_rc = rc_cfg is not None and "rollout_log_probs" in data.batch and not rc_cfg.get("bypass_mode", False)
        if use_rc:
            data, is_metrics = compute_rollout_correction_and_add_to_batch(data, rc_cfg)
            metrics.update(is_metrics)

        scores = data.batch["token_level_rewards"].sum(-1).cpu().numpy()
        res = compute_step_advantages(
            scores, list(data.non_tensor_batch["uid"]), task_ids, lengths,
            max_response=int(rl["sampling"]["max_response"]),
            scaler=rt.scaler, normalizer=rt.normalizer,
            # verl's global_steps is 1-indexed; curriculum stages are 0-indexed.
            nominal_mix=rt.mix_for_batch(self.global_steps - 1, {t for t in task_ids if t}),
            is_padding=is_padding, eps=float(rl["advantage"]["zero_adv_eps"]),
            normalize_by_group_std=rl["advantage"].get("estimator") == "group_mean_std",   # ablation A2
        )
        metrics.update(res.metrics)
        mask = data.batch["response_mask"] * torch.as_tensor(res.valid, dtype=data.batch["response_mask"].dtype,
                                                             device=data.batch["response_mask"].device)[:, None]
        adv = torch.as_tensor(res.seq_adv, dtype=torch.float32, device=mask.device)[:, None] * mask
        data.batch["advantages"] = adv
        data.batch["returns"] = adv
        data.batch["response_mask"] = mask

        out_fields = ["advantages", "returns", "response_mask"]
        if use_rc and "rollout_is_weights" in data.batch:
            out_fields.append("rollout_is_weights")
        output = TensorDict({f: response_to_nested(data.batch[f], nested_mask) for f in out_fields},
                            batch_size=len(batch))
        return tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=output)

    def _hack_rates(self, partition_id: str, keys: list[str]) -> dict[str, float]:
        """``hack/<category>/rate`` from the per-sample reward flags (report §检测体系 2).

        The reward function returns the hack-pattern flags it found as a comma-joined string.
        verl keeps it in each trajectory's ``extra_fields.reward_extra_info``. Categories seen
        earlier in the run get an explicit 0, so the window-over-window growth rule sees a
        continuous series.
        """
        if not hasattr(self, "_hack_tracker"):          # normally created in _init_dataloader
            self._hack_tracker, self._hack_seen = HackRateTracker(), set()
        try:
            ex = tq.kv_batch_get(keys=keys, partition_id=partition_id, select_fields=["extra_fields"])
            vals = ex["extra_fields"]
            vals = vals.tolist() if hasattr(vals, "tolist") else list(vals)
        except Exception as e:  # monitoring only; never fail the step
            log.debug("no extra_fields for hack rates: %s", e)
            return {}
        flags = []
        for v in vals:
            info = v.get("reward_extra_info", {}) if isinstance(v, dict) else {}
            flags.append([f for f in str(info.get("flags", "") or "").split(",") if f])
        out = self._hack_tracker.update(int(self.global_steps), flags)
        self._hack_seen.update(k for k in out)
        return {k: out.get(k, 0.0) for k in self._hack_seen}

    # ------------------------------------------------------------------ metrics, curriculum, alerts
    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)
        self._spec_acceptance_fallback(metrics)
        if getattr(self, "_resume_roundtrip_ok", None) is not None:
            metrics["resume/state_roundtrip_ok"] = float(self._resume_roundtrip_ok)
            self._resume_roundtrip_ok = None
        rt = get_runtime()
        step = int(global_steps) - 1                       # curriculum steps are 0-indexed
        stats: dict[str, dict[str, float]] = {}
        for k, v in rt.last_filter_stats.items():          # filter/<task>/zero_adv_group_frac etc.
            parts = k.split("/")
            if len(parts) == 3 and parts[2] in ("zero_adv_group_frac", "all_solved_group_frac"):
                key = "zero_adv_frac" if parts[2] == "zero_adv_group_frac" else "all_solved_frac"
                stats.setdefault(parts[1], {})[key] = float(v)
        # Dev metric on validation steps: the independent gold grader when available, else the proxy.
        canon = to_canonical(metrics)
        for t in TASKS:
            dev = canon.get(f"gold/{t}", canon.get(f"proxy/{t}"))
            if dev is not None:
                stats.setdefault(t, {})["dev_metric"] = float(dev)
        if rt.curriculum is not None and stats:
            for ev in rt.curriculum.observe(step, stats):
                log.warning("curriculum auto-reweight: %s", ev)
                metrics[f"curriculum/event/{ev.get('task', 'unknown')}"] = 1.0
            for t, w in rt.curriculum.effective_mix(step).items():
                metrics[f"curriculum/mix/{t}"] = w
            metrics["curriculum/stage"] = float(rt.curriculum.cfg.stages.index(rt.curriculum.stage_at(step)))

        metrics.update(canon)                             # canonical names for panels + alert rules
        alerts = []
        if rt.alerts is not None:
            alerts = rt.alerts.evaluate(int(global_steps), metrics)
            for a in alerts:
                metrics[f"alert/{a.rule}"] = 1.0
                log.warning("ALERT [%s] %s: %s=%.4g (threshold %s): %s",
                            a.severity, a.rule, a.metric, a.value, a.threshold, a.message)
            self._alert_state.value = rt.alerts.state_dict()
        # Local, append-only step log: offline analysis, dev-curve checkpoint selection
        # (dpt.eval.select_checkpoint) and the resume-equivalence comparison (dpt.monitoring.compare_runs).
        if not hasattr(self, "_dpt_sink"):
            self._dpt_sink = JsonlSink(Path(self.config.trainer.default_local_dir) / "metrics.jsonl")
        self._dpt_sink.log(int(global_steps), metrics)
        # verl saved this step's checkpoint BEFORE this hook ran. Refresh its extra state so it also
        # holds this step's curriculum observation and alert history (exact resume, Incident 2).
        if getattr(self, "_last_saved_step", -1) == self.global_steps:
            self.dpt_state.save(self._ckpt_folder(self.global_steps), self.global_steps, data_hashes=_data_hashes())
        # In colocated mode rollout cannot be blocked on its own: a failed post-sync parity check
        # ("block_rollout") also stops the run until a human has looked (runbook §3).
        if rt.alerts is not None and (rt.alerts.should_pause() or rt.alerts.rollout_blocked()):
            red = [a for a in alerts if a.severity in ("pause", "block_rollout")] or alerts
            self._pause(global_steps, "; ".join(f"{a.rule}: {a.message}" for a in red))
            return
        # Batch-scheduler wall-clock limit (the cluster: 48 h): save and stop at a step boundary while
        # one more step + a checkpoint still fit; the job script resubmits (dpt.launch.walltime).
        if getattr(self, "_wallclock", None) is None:
            self._wallclock = WallClock.from_env()
        now = time.time()
        self._wallclock.observe_step(now)
        stop, why = self._wallclock.should_stop(now)
        if stop and self.global_steps < self.total_training_steps:
            self._stop(global_steps, why, marker="WALLTIME.json", severity="info")
            return
        # A hold: at a set step (campaign.sh: the judge's job is still queued when stage B would need it), or
        # when HOLD_REQUEST appears in the run dir (campaign.sh: the judge went away during stage B/C).
        hold = os.environ.get(STOP_AT_ENV)
        request = Path(self.config.trainer.default_local_dir) / "HOLD_REQUEST"
        if self.global_steps < self.total_training_steps:
            if hold and int(hold) <= self.global_steps:
                self._stop(global_steps, f"hold at step {hold} ({STOP_AT_ENV})", marker="HOLD.json", severity="info")
            elif request.exists():
                self._stop(global_steps, f"hold requested: {request.read_text().strip()[:200]}", marker="HOLD.json",
                           severity="info")

    def _spec_acceptance_fallback(self, metrics: dict) -> None:
        """MTP acceptance from vLLM's counters when verl cannot report it (vLLM 0.24; see spec_scrape).

        Without it the Incident 5 alert (`spec/mtp_acceptance_rate` < 0.6) has no input. Needs
        `rollout.disable_log_stats: false`, otherwise vLLM keeps no counters.
        """
        sel = lambda key, default=None: OmegaConf.select(self.config, key, default=default)  # noqa: E731
        mtp_rollout = sel("actor_rollout_ref.model.mtp.enable") and sel("actor_rollout_ref.model.mtp.enable_rollout")
        if not mtp_rollout or "rollout/spec_accept_rate" in metrics:
            return
        if getattr(self, "_spec_scraper", None) is None:
            if sel("actor_rollout_ref.rollout.disable_log_stats", True):
                log.warning("MTP rollout is on but rollout.disable_log_stats=true: no acceptance metric, "
                            "the mtp_acceptance alert is blind")
                self._spec_scraper = False
                return
            addrs = list(getattr(getattr(self, "llm_server_manager", None), "server_addresses", None) or [])
            self._spec_scraper = SpecDecodeScraper(addrs) if addrs else False
            if not addrs:
                log.warning("no rollout server addresses; cannot read vLLM acceptance counters")
        if self._spec_scraper:
            rate = self._spec_scraper.poll()
            if rate is not None:
                metrics["rollout/spec_accept_rate_counters"] = rate

    def _pause(self, global_steps: int, reason: str) -> None:
        """Alert auto-pause: checkpoint, write ``PAUSED.json``, end the fit loop (runbook §auto-pause).

        A human must look before the run continues: job scripts never resubmit a paused run.
        """
        self._stop(global_steps, reason, marker="PAUSED.json", severity="error")

    def _stop(self, global_steps: int, reason: str, *, marker: str, severity: str) -> None:
        """Graceful stop at a step boundary: checkpoint, write ``marker``, end the fit loop.

        Used for alert pauses (``PAUSED.json``) and the wall-clock limit (``WALLTIME.json``).
        This hook runs after ``on_step_end`` has woken vLLM and resumed generation. A Megatron save
        moves parameters back onto the GPU, so first abort the in-flight requests and put vLLM to
        sleep (what ``on_sample_end`` does). Otherwise the save competes with vLLM's share of HBM.
        """
        (log.error if severity == "error" else log.info)(
            "%s at step %d: %s", "AUTO-PAUSE" if marker == "PAUSED.json" else "STOP", global_steps, reason)
        try:
            self.checkpoint_manager.abort_replicas()
            self.checkpoint_manager.sleep_replicas()
        except Exception as e:  # still try to save; worst case the save itself reports OOM
            log.warning("could not put rollout replicas to sleep before the stop checkpoint: %s", e)
        if getattr(self, "_last_saved_step", -1) != self.global_steps:
            self._save_checkpoint()
        path = Path(self.config.trainer.default_local_dir) / marker
        path.write_text(json.dumps({"step": int(global_steps), "reason": reason, "time": time.time()}, indent=2))
        self.total_training_steps = self.global_steps     # the v1 fit loop exits on its next check

    # ------------------------------------------------------------------ checkpoint (Incident 2)
    def _ckpt_folder(self, step: int) -> str:
        return os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}")

    def _save_checkpoint(self):
        folder = self._ckpt_folder(self.global_steps)
        os.makedirs(folder, exist_ok=True)
        # Written BEFORE super() updates latest_checkpointed_iteration.txt, so a crash in between
        # can never leave a "latest" checkpoint without its extra state.
        t0 = time.time()
        self.dpt_state.save(folder, self.global_steps, data_hashes=_data_hashes())
        super()._save_checkpoint()
        self._last_saved_step = self.global_steps
        if getattr(self, "_wallclock", None) is not None:
            self._wallclock.observe_save(time.time() - t0)

    def _load_checkpoint(self):
        out = super()._load_checkpoint()
        if self.global_steps > 0:
            t = self.config.trainer
            if t.get("resume_mode") == "resume_path" and t.get("resume_from_path"):
                folder = str(t.resume_from_path)                     # explicit rollback target
            else:
                folder = self._ckpt_folder(self.global_steps)
            step = self.dpt_state.load(folder, expected_data_hashes=_data_hashes())
            # Reported once at the first step after the resume (alert rule `resume_state`, pause).
            self._resume_roundtrip_ok = not getattr(self.dpt_state, "last_roundtrip_mismatches", [])
            rt = get_runtime()
            if rt.alerts is not None and self._alert_state.value:
                rt.alerts.load_state_dict(self._alert_state.value)
            log.info("restored dpt extra state from step %d", step)
        return out


def _data_hashes() -> dict[str, str]:
    path = os.environ.get("DPT_DATA_HASHES")
    return json.loads(Path(path).read_text()) if path and Path(path).exists() else {}
