"""Opt-in per-micro-batch diagnostics of the PPO ratio (train-time log-probs vs old log-probs).

Enable with ``DPT_DEBUG_PPO=<path.jsonl>`` in the environment of the Ray cluster (export it before
``ray start``; workers inherit the raylet's environment). ``dpt.verl_ext.register`` then imports
this module, which wraps ``verl.workers.utils.losses.ppo_loss``. For every micro-batch it appends:

* the response length and the number of tokens the loss mask keeps;
* mean(old − new) (verl's ``ppo_kl``) and mean |old − new|, over the loss mask and over the whole
  response (the loss mask excludes truncated / zero-advantage samples);
* where the largest gap sits, and the gaps at the first and last response positions.

Before the first optimizer step the ratio must be 1 up to kernel noise (~1e-2 per token in BF16).
A gap far above that means the training forward and the old-log-prob forward disagree, so PPO
clipping and TIS act on noise. Use with ``rl.num_updates_per_step=1`` and one step to see it in isolation.
"""

from __future__ import annotations

import json
import os
import sys

import torch
import verl.workers.utils.losses as _losses

_PATH = os.environ.get("DPT_DEBUG_PPO", "")
_orig_ppo_loss = _losses.ppo_loss


def _record(model_output, data) -> dict:
    new = _losses.no_padding_2_padding(model_output["log_probs"], data).detach().float()
    d = data.select("response_mask", "old_log_probs").to_padded_tensor()
    old = d["old_log_probs"].float().to(new.device)
    mask = d["response_mask"].bool().to(new.device)
    T = min(new.shape[-1], old.shape[-1])
    new, old, mask = new[..., :T], old[..., :T], mask[..., :T]
    resp_lens = data["responses"].offsets().diff().tolist() if data["responses"].is_nested else [T] * new.shape[0]
    full = torch.zeros_like(mask)
    for i, n in enumerate(resp_lens):
        full[i, : min(n, T)] = True
    x = old - new
    out = {"pid": os.getpid(), "shape": list(new.shape), "old_shape": list(d["old_log_probs"].shape),
           "resp_lens": resp_lens, "mask_tokens": int(mask.sum())}
    for name, m in (("mask", mask), ("full", full)):
        n = m.sum().clamp(min=1)
        out[f"kl_{name}"] = float((x * m).sum() / n)
        out[f"abs_{name}"] = float((x.abs() * m).sum() / n)
        out[f"frac_abs_gt_1_{name}"] = float(((x.abs() > 1) & m).sum() / n)
    xf = (x.abs() * full)[0]
    out["argmax_pos"], out["max_abs"] = int(xf.argmax()), float(xf.max())
    n0 = resp_lens[0]
    out["first4"] = [round(v, 4) for v in x[0, :4].tolist()]
    out["last4"] = [round(v, 4) for v in x[0, max(0, n0 - 4): n0].tolist()]
    out["old_first4"] = [round(v, 4) for v in old[0, :4].tolist()]
    out["new_first4"] = [round(v, 4) for v in new[0, :4].tolist()]
    return out


def ppo_loss(config, model_output, data, dp_group=None):
    try:
        rec = _record(model_output, data)
    except Exception as e:  # diagnostics must never break training
        rec = {"pid": os.getpid(), "error": f"{type(e).__name__}: {e}"}
    with open(_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return _orig_ppo_loss(config, model_output, data, dp_group=dp_group)


def _fingerprint(chunks) -> dict:
    """Sums of a few parameters, to tell whether two forward passes used the same weights."""
    named = [(n, p) for c in chunks for n, p in c.named_parameters()]
    pick = named[:2] + named[len(named) // 2: len(named) // 2 + 1] + named[-2:]
    return {n.split(".", 2)[-1][-48:]: [round(float(p.detach().float().sum()), 4), str(p.dtype), str(p.device)]
            for n, p in pick}


def _wrap_lm_head():
    """Also log what reaches the LM-head processor: temperature values, logits scale, train/eval mode."""
    from verl.workers.engine.megatron import transformer_impl as _ti

    cls = _ti.MegatronEngineWithLMHead
    orig = cls._lm_head_logits_processor

    def wrapped(self, logits, label, temperature, **kw):
        try:
            t = temperature.values() if getattr(temperature, "is_nested", False) else temperature
            lf = logits.detach().float()
            rec = {"pid": os.getpid(), "where": "lm_head", "training": bool(self.module[0].training),
                   "grad": torch.is_grad_enabled(), "requires_grad": bool(logits.requires_grad),
                   "calc_entropy": bool(kw.get("calculate_entropy")), "shape": list(logits.shape),
                   "temp_unique": [round(v, 4) for v in t.float().unique()[:6].tolist()],
                   "logits_std": float(lf.std()), "logits_max_mean": float(lf.max(-1).values.mean()),
                   "param_fp": _fingerprint(self.module)}
        except Exception as e:
            rec = {"pid": os.getpid(), "where": "lm_head", "error": f"{type(e).__name__}: {e}"}
        with open(_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return orig(self, logits, label, temperature, **kw)

    cls._lm_head_logits_processor = wrapped


if _PATH:
    _losses.ppo_loss = ppo_loss
    try:
        _wrap_lm_head()
    except Exception as e:  # pragma: no cover - environment dependent
        print(f"[dpt.debug_ppo] cannot wrap the LM-head processor: {e}")
    _ew = sys.modules.get("verl.workers.engine_workers")   # binds the name at import; patch if loaded
    if _ew is not None:
        _ew.ppo_loss = ppo_loss
