"""Runtime fixes for the pinned stack (verl 0.9.1, Megatron-Bridge 0.6.2, Transformer Engine 2.20.2).

Imported in every training process: by ``dpt.verl_ext.register`` (RL, via ``VERL_USE_EXTERNAL_MODULES``)
and directly through ``VERL_USE_EXTERNAL_MODULES=dpt.verl_ext.patches`` (SFT, see ``dpt.launch.sft``).
Importing this module only installs an import hook; nothing heavy is imported here.

Fix 1: inverted attention mask for Qwen3.5/3.8 on A100 (found by the workstation RL smoke run).

    verl's vision-language BSHD path (``verl.models.mcore.util.build_vlm_attn_mask_bshd``, used for
    ``Qwen3_5ForConditionalGeneration``) builds an HF-style *keep* mask, ``[B, S]`` with True = real
    token. Megatron-Bridge's ``Qwen3VLModel`` passes it unchanged to Megatron attention, where a
    boolean mask means True = *masked out*. Whether the mask is honoured depends on Transformer
    Engine's backend:

    * forward-only (old log-probs, eval): TE picks cuDNN FusedAttention with a causal mask type,
      and the supplied mask has no effect, so the result is correct;
    * training forward on A100 (SM80) with head_dim 256 (Qwen3.5/3.8 attention): cuDNN has no fused
      backward for that shape, so TE falls back to UnfusedDotProductAttention. That backend applies
      the mask and blanks every real token in the 1-in-4 full-attention layers.

    Effect: the training forward's log-probs on sampled tokens were 1.0-1.8 nats lower than the
    old-log-prob forward on identical weights. PPO ratios, clipping and TIS then act on noise,
    and SFT would train a model whose attention layers see nothing. Measured on Qwen3.5-0.8B with
    standalone Megatron-Bridge (train vs eval mode, verl-style inputs): mean(old − new) = +1.80 nats/token;
    with the mask converted below: 3e-4, the same as kernel noise. GPUs whose fused attention
    supports d=256 in training (H100) would not show it, which is presumably why it went unnoticed.

    Fix: convert the keep mask to Megatron's convention, ``[B, 1, 1, S]`` with True = masked. It is
    correct for every backend: fused attention still uses its causal mask; unfused attention applies
    causal + padding masks. Only the model call consumes this mask; verl's post-processing uses its
    own ``attention_mask_bshd``.

Fix 2: FP32 copy of the logits in verl's log-probs (the 9B line, measured 2026-10-28).

    verl's Megatron engine takes SFT and PPO log-probs from ``vocab_parallel_log_probs_from_logits``, i.e.
    Megatron's ``vocab_parallel_cross_entropy``, which upcasts the logits to FP32 and keeps that copy for
    backward. At TP1 the last pipeline stage holds the whole 248,320-word vocabulary: 1.43 MiB per token
    (28.9 GiB for the longest SFT row, 20,841 tokens) on a GPU whose fixed state already takes most of its
    40 GB, so rows above ~3.5K tokens did not fit.

    Fix: the same log-probs from Transformer Engine's Triton cross entropy. It also computes in FP32, from
    the BF16 logits, and keeps only the logits; with ``overwrite_input`` it writes the gradient into them
    (verl does not read the logits after this call). Labels are token ids, never TE's ignore index -100,
    so values and gradients match Megatron's to rounding. The MTP head's loss is Megatron's own
    ``compute_language_model_loss``, which takes the same kernel through ``cross_entropy_loss_fusion`` and
    ``cross_entropy_fusion_impl: te`` (``configs/sft/sft_9b.yaml``).

    Before that call verl (``MegatronEngineWithLMHead._lm_head_logits_processor``) divides the logits by the
    temperature in place. SFT and our RL use temperature 1, where the division changes no value, but its
    backward allocates a second ``[S, V]`` gradient. When every real token has temperature 1 the division is
    skipped (padding carries temperature 0, which verl clamps to 1e-8 and masks out of the loss); any other
    temperature runs verl's code. Measured on an A100, main + MTP losses, forward + backward: 2.38 MiB per
    token before; after, 0.94 MiB per token plus the output layer's 2 GiB weight gradient (21.7 GiB at
    20,841 tokens). Until 2026-10-29 the skip looked for the method on the base class ``MegatronEngine``,
    found nothing and patched nothing; a missing target now raises.

Fix 3: FP32 copy of the logits in Megatron's ``Float16Module`` (found by the 8-node SFT pre-flight, 2026-10-29).

    On the last pipeline stage the wrapper upcasts the model's output to FP32 (``forward(..., fp32_output=True)``).
    Megatron's own training returns a loss there; verl's engine returns the logits, so the wrapper copies all
    of them to FP32 before fix 2's kernel sees them: 0.95 MiB per token at TP1. Stage 7 ran out of memory
    allocating exactly that copy (18.8 GiB for a 20.4K-token row); the workstation measurement had called the
    loss without the wrapper.

    Fix: during verl's LM-head forward step the wrapper returns the logits in model precision
    (``fp32_output=False``), the BF16 values the copy would have held. Fix 2's kernel computes in FP32 from
    them; entropy, Σπ² and distillation get FP32 copies (chunk by chunk for chunked entropy), and another
    temperature runs verl's code on an FP32 copy, so none of their values change.
"""

from __future__ import annotations

import contextvars
import importlib.abc
import importlib.machinery
import sys
from typing import Any, Callable

_FIXES: dict[str, Callable[[Any], None]] = {}

# Set while verl's LM-head forward step runs (fix 3): Float16Module then keeps the logits in BF16.
_HALF_LOGITS: contextvars.ContextVar[bool] = contextvars.ContextVar("dpt_half_logits", default=False)


def _own_attr(cls: Any, name: str, what: str) -> Any:
    """``cls``'s own ``name`` (not inherited). Raises if missing: a fix must never be a silent no-op."""
    attr = vars(cls).get(name)
    if attr is None:
        raise RuntimeError(f"dpt.verl_ext.patches: {what} not found; did the pinned stack change?")
    return attr


def keep_to_megatron_mask(keep):
    """HF keep mask ``[B, S]`` (True = token) → Megatron mask ``[B, 1, 1, S]`` (True = masked)."""
    if keep is None or keep.dim() != 2:
        return keep
    return ~keep.bool()[:, None, None, :]


def _fix_vlm_bshd_mask(module: Any) -> None:
    orig = getattr(module, "build_vlm_attn_mask_bshd", None)
    if orig is None or getattr(orig, "_dpt_fixed", False):
        return

    def build_vlm_attn_mask_bshd(*args, **kwargs):
        input_ids_bshd, keep = orig(*args, **kwargs)
        return input_ids_bshd, keep_to_megatron_mask(keep)

    build_vlm_attn_mask_bshd._dpt_fixed = True
    build_vlm_attn_mask_bshd.__wrapped__ = orig
    module.build_vlm_attn_mask_bshd = build_vlm_attn_mask_bshd


_FIXES["verl.models.mcore.model_forward"] = _fix_vlm_bshd_mask


def te_log_probs_from_logits(logits, labels, fallback: Callable):
    """Per-token log-probs ``[B, S]`` from logits ``[B, S, V/TP]`` with TE's cross entropy (fix 2).

    Anything TE's kernel does not take (CPU tensors, other dtypes or ranks) goes to ``fallback``.
    """
    import torch

    if not (logits.is_cuda and logits.dim() == 3 and logits.dtype in (torch.bfloat16, torch.float32)):
        return fallback(logits, labels)
    from megatron.core import parallel_state as mpu
    from transformer_engine.pytorch.cross_entropy import parallel_cross_entropy

    tp_group = mpu.get_tensor_model_parallel_group() if mpu.model_parallel_is_initialized() else None
    loss = parallel_cross_entropy(logits, labels.contiguous().long(), 0.0, False, tp_group,
                                  overwrite_input=logits.is_contiguous())
    return -loss


def _fix_te_cross_entropy(module: Any) -> None:
    orig = getattr(module, "vocab_parallel_log_probs_from_logits", None)
    if orig is None or getattr(orig, "_dpt_fixed", False):
        return

    def vocab_parallel_log_probs_from_logits(logits, labels):
        return te_log_probs_from_logits(logits, labels, fallback=orig)

    vocab_parallel_log_probs_from_logits._dpt_fixed = True
    vocab_parallel_log_probs_from_logits.__wrapped__ = orig
    module.vocab_parallel_log_probs_from_logits = vocab_parallel_log_probs_from_logits


def _fp32_copy(x):
    """A new FP32 tensor (verl's entropy and Σπ² are written for FP32 logits and modify their input)."""
    import torch

    return x.clone() if x.dtype == torch.float32 else x.float()


def _fix_unit_temperature(module: Any) -> None:
    engine = getattr(module, "MegatronEngineWithLMHead", None)
    if engine is None:
        return
    orig = _own_attr(engine, "_lm_head_logits_processor", "MegatronEngineWithLMHead._lm_head_logits_processor")
    if getattr(orig, "_dpt_fixed", False):
        return

    def _lm_head_logits_processor(self, logits, label, temperature, **kw):
        if not bool(((temperature == 1) | (temperature <= 0)).all()):
            return orig(self, _fp32_copy(logits), label, temperature, **kw)
        # verl 0.9.1's body without `logits.div_(temperature)`, on BF16 logits (fix 3)
        assert logits.shape[:2] == label.shape[:2]
        ret = {}
        if kw["calculate_sum_pi_squared"]:
            ret["sum_pi_squared"] = module.vocab_parallel_sum_pi_squared(_fp32_copy(logits))
        if kw["calculate_entropy"]:
            if self.engine_config.entropy_from_logits_with_chunking:
                chunk_size = self.engine_config.entropy_from_logits_chunk_size
                ret["entropy"] = _chunked_fp32_entropy(module, logits, chunk_size)
            else:
                ret["entropy"] = module.vocab_parallel_entropy(_fp32_copy(logits))
        if kw["distillation_use_topk"]:
            ret.update(kw["logits_processor_func"](student_logits=_fp32_copy(logits), data=kw["batch"],
                                                   data_format=kw["data_format"]))
        if not kw["distillation_only"]:
            ret["log_probs"] = module.vocab_parallel_log_probs_from_logits(logits, label)
        return ret

    _lm_head_logits_processor._dpt_fixed = True
    _lm_head_logits_processor.__wrapped__ = orig
    engine._lm_head_logits_processor = _lm_head_logits_processor


def _chunked_fp32_entropy(module: Any, logits, chunk_size: int):
    """verl's ``vocab_parallel_entropy_with_chunking`` with each chunk upcast to FP32 on its own."""
    import torch

    entropy = torch.zeros(logits.shape[:-1], device=logits.device)
    for i in range(0, logits.shape[1], chunk_size):
        entropy[:, i:i + chunk_size] = module.vocab_parallel_entropy(_fp32_copy(logits[:, i:i + chunk_size, :]))
    return entropy


def _fix_half_logits_forward(module: Any) -> None:
    engine = getattr(module, "MegatronEngineWithLMHead", None)
    if engine is None:
        return
    orig = _own_attr(engine, "forward_step", "MegatronEngineWithLMHead.forward_step")
    if getattr(orig, "_dpt_fixed", False):
        return

    def forward_step(self, *args, **kwargs):
        token = _HALF_LOGITS.set(True)
        try:
            return orig(self, *args, **kwargs)
        finally:
            _HALF_LOGITS.reset(token)

    forward_step._dpt_fixed = True
    forward_step.__wrapped__ = orig
    engine.forward_step = forward_step


def _fix_engine_loss(module: Any) -> None:
    _fix_te_cross_entropy(module)
    _fix_unit_temperature(module)
    _fix_half_logits_forward(module)


_FIXES["verl.workers.engine.megatron.transformer_impl"] = _fix_engine_loss


def _fix_float16_output(module: Any) -> None:
    wrapper = getattr(module, "Float16Module", None)
    if wrapper is None:
        return
    orig = _own_attr(wrapper, "forward", "megatron.core.transformer.module.Float16Module.forward")
    if getattr(orig, "_dpt_fixed", False):
        return

    def forward(self, *inputs, fp32_output=True, **kwargs):
        return orig(self, *inputs, fp32_output=fp32_output and not _HALF_LOGITS.get(), **kwargs)

    forward._dpt_fixed = True
    forward.__wrapped__ = orig
    wrapper.forward = forward


_FIXES["megatron.core.transformer.module"] = _fix_float16_output


class _PostImportPatcher(importlib.abc.MetaPathFinder):
    """Applies a fix right after its target module has executed (the target imports Megatron, so it
    cannot be imported eagerly while ``import verl`` is still running)."""

    def find_spec(self, name, path=None, target=None):
        fix = _FIXES.get(name)
        if fix is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return None
        exec_module = spec.loader.exec_module

        def exec_and_fix(module):
            exec_module(module)
            fix(module)

        spec.loader.exec_module = exec_and_fix
        return spec


def install() -> None:
    """Idempotent: fix already-imported targets now, the others when they are imported."""
    for name, fix in _FIXES.items():
        if name in sys.modules:
            fix(sys.modules[name])
    if not any(isinstance(f, _PostImportPatcher) for f in sys.meta_path):
        sys.meta_path.insert(0, _PostImportPatcher())


install()
