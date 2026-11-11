"""dpt.verl_ext.patches: the Qwen3.5 attention-mask fix (inverted keep mask in A100 training forwards), the
TE cross entropy for verl's log-probs and the BF16 logits out of Megatron's Float16Module (the FP32 logits
copies on the last pipeline stage)."""

from __future__ import annotations

import importlib
import inspect
import sys
import types

import pytest
import torch

from dpt.verl_ext import patches


def test_keep_mask_becomes_megatron_mask():
    keep = torch.tensor([[True, True, True, False], [True, True, False, False]])
    m = patches.keep_to_megatron_mask(keep)
    assert m.shape == (2, 1, 1, 4)
    # Megatron: True = masked out. Real tokens must never be masked.
    assert m[:, 0, 0].tolist() == [[False, False, False, True], [False, False, True, True]]
    assert patches.keep_to_megatron_mask(None) is None


def test_fix_wraps_builder_once():
    mod = types.SimpleNamespace(build_vlm_attn_mask_bshd=lambda ids, bs: (ids, torch.ones(1, 3, dtype=torch.bool)))
    patches._fix_vlm_bshd_mask(mod)
    patches._fix_vlm_bshd_mask(mod)                      # idempotent
    ids, mask = mod.build_vlm_attn_mask_bshd("ids", 1)
    assert ids == "ids" and mask.shape == (1, 1, 1, 3) and not mask.any()
    assert not getattr(mod.build_vlm_attn_mask_bshd.__wrapped__, "_dpt_fixed", False)


def test_post_import_hook_applies_fix(tmp_path, monkeypatch):
    (tmp_path / "dpt_fake_mf.py").write_text("def build_vlm_attn_mask_bshd(x):\n    return x, None\nflag = 0\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(patches._FIXES, "dpt_fake_mf", lambda m: setattr(m, "flag", 1))
    patches.install()
    sys.modules.pop("dpt_fake_mf", None)
    m = importlib.import_module("dpt_fake_mf")
    assert m.flag == 1
    sys.modules.pop("dpt_fake_mf", None)


def test_te_cross_entropy_fix_wraps_once_and_falls_back_off_gpu():
    mod = types.SimpleNamespace(vocab_parallel_log_probs_from_logits=lambda logits, labels: ("orig", logits.shape))
    orig = mod.vocab_parallel_log_probs_from_logits
    patches._fix_te_cross_entropy(mod)
    patches._fix_te_cross_entropy(mod)                   # idempotent
    assert mod.vocab_parallel_log_probs_from_logits.__wrapped__ is orig
    assert mod.vocab_parallel_log_probs_from_logits(torch.zeros(1, 4, 8), torch.zeros(1, 4)) == ("orig", (1, 4, 8))
    assert "verl.workers.engine.megatron.transformer_impl" in patches._FIXES


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TE's cross entropy is a GPU kernel")
def test_te_log_probs_match_fp32_reference():
    pytest.importorskip("megatron.core")
    pytest.importorskip("transformer_engine.pytorch.cross_entropy")
    torch.manual_seed(0)
    h = torch.randn(1, 512, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = (torch.randn(32000, 256, device="cuda", dtype=torch.bfloat16) * 0.05).requires_grad_()
    labels = torch.randint(0, 32000, (1, 512), device="cuda")
    mask = (torch.rand(1, 512, device="cuda") > 0.3).float()

    def reference(logits, y):                            # Megatron's math: FP32 log-softmax of the BF16 logits
        return torch.log_softmax(logits.float(), -1).gather(-1, y[..., None])[..., 0]

    def run(fn):
        h.grad = w.grad = None
        logits = h @ w.t()
        logits.div_(0.7)                                 # verl divides by the temperature in place first
        lp = fn(logits, labels)
        (-(lp * mask).sum()).backward()
        return lp.detach(), h.grad.float(), w.grad.float()

    ref = run(reference)
    new = run(lambda logits, y: patches.te_log_probs_from_logits(logits, y, fallback=reference))
    assert new[0].shape == ref[0].shape == (1, 512)
    assert (new[0] - ref[0]).abs().max() < 1e-4
    for a, b in zip(new[1:], ref[1:]):
        assert (a - b).norm() / b.norm() < 1e-2


def _fake_engine_module(with_processor=True):
    class MegatronEngine:                                # verl 0.9.1: the processor lives on the subclass only
        pass

    class MegatronEngineWithLMHead(MegatronEngine):
        def forward_step(self, *args, **kwargs):
            return patches._HALF_LOGITS.get()

    if with_processor:
        def _lm_head_logits_processor(self, logits, label, temperature, *, calculate_sum_pi_squared,
                                      calculate_entropy, distillation_use_topk, distillation_only,
                                      logits_processor_func, batch, data_format):
            return {"verl": logits.dtype}
        MegatronEngineWithLMHead._lm_head_logits_processor = _lm_head_logits_processor

    return types.SimpleNamespace(MegatronEngine=MegatronEngine, MegatronEngineWithLMHead=MegatronEngineWithLMHead,
                                 vocab_parallel_log_probs_from_logits=lambda logits, labels: ("lp", logits.dtype))


_OPTS = dict(calculate_sum_pi_squared=False, calculate_entropy=False, distillation_use_topk=False,
             distillation_only=False, logits_processor_func=None, batch=None, data_format="bshd")


def test_engine_fixes_patch_the_lm_head_engine():
    mod = _fake_engine_module()
    patches._fix_engine_loss(mod)
    patches._fix_engine_loss(mod)                        # idempotent
    eng = mod.MegatronEngineWithLMHead
    assert eng._lm_head_logits_processor._dpt_fixed and eng.forward_step._dpt_fixed
    assert not hasattr(mod.MegatronEngine, "_lm_head_logits_processor")
    logits = torch.zeros(1, 4, 8, dtype=torch.bfloat16)
    unit = torch.tensor([[1.0, 1.0, 0.0, 0.0]])          # padding carries temperature 0
    # temperature 1 on every real token: no division, log-probs from the BF16 logits
    assert eng()._lm_head_logits_processor(logits, torch.zeros(1, 4), unit, **_OPTS) == {"log_probs": ("lp", torch.bfloat16)}
    # any other temperature: verl's own code, on an FP32 copy as before fix 3
    assert eng()._lm_head_logits_processor(logits, torch.zeros(1, 4), unit * 0.7, **_OPTS) == {"verl": torch.float32}
    assert eng().forward_step() is True and patches._HALF_LOGITS.get() is False


def test_engine_fix_refuses_a_missing_target():
    with pytest.raises(RuntimeError, match="_lm_head_logits_processor not found"):
        patches._fix_engine_loss(_fake_engine_module(with_processor=False))


def test_float16_output_stays_bf16_only_inside_the_lm_head_step():
    class Float16Module:
        def forward(self, *inputs, fp32_output=True, **kwargs):
            return fp32_output

    mod = types.SimpleNamespace(Float16Module=Float16Module)
    patches._fix_float16_output(mod)
    patches._fix_float16_output(mod)                     # idempotent
    m = Float16Module()
    assert m.forward() is True and m.forward(fp32_output=False) is False
    token = patches._HALF_LOGITS.set(True)
    try:
        assert m.forward() is False
    finally:
        patches._HALF_LOGITS.reset(token)
    assert "megatron.core.transformer.module" in patches._FIXES


def test_fixes_reach_the_installed_stack():
    """The fixes are installed on the real classes (a fake module once hid a fix that patched nothing)."""
    pytest.importorskip("megatron.core")
    pytest.importorskip("verl")
    patches.install()
    ti = importlib.import_module("verl.workers.engine.megatron.transformer_impl")
    f16 = importlib.import_module("megatron.core.transformer.module").Float16Module
    eng = ti.MegatronEngineWithLMHead
    for f in (vars(eng)["_lm_head_logits_processor"], vars(eng)["forward_step"],
              ti.vocab_parallel_log_probs_from_logits, vars(f16)["forward"]):
        assert getattr(f, "_dpt_fixed", False), f
    # the body fix 2 mirrors still takes verl 0.9.1's options; the wrapper still has fp32_output
    params = inspect.signature(vars(eng)["_lm_head_logits_processor"].__wrapped__).parameters
    assert [n for n, p in params.items() if p.kind is p.KEYWORD_ONLY] == list(_OPTS)
    assert "fp32_output" in inspect.signature(vars(f16)["forward"].__wrapped__).parameters


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TE's cross entropy is a GPU kernel")
def test_patched_processor_on_bf16_matches_verl_on_fp32(tmp_path):
    """Fixes 2 + 3 end to end: verl's processor on the FP32 copy (as before) vs the patched one on BF16 logits."""
    pytest.importorskip("megatron.core")
    pytest.importorskip("verl")
    import torch.distributed as dist
    from megatron.core import parallel_state as mpu

    patches.install()
    ti = importlib.import_module("verl.workers.engine.megatron.transformer_impl")
    proc = vars(ti.MegatronEngineWithLMHead)["_lm_head_logits_processor"]
    created = not dist.is_initialized()
    if created:
        torch.cuda.set_device(0)
        dist.init_process_group("nccl", init_method=f"file://{tmp_path}/pg", world_size=1, rank=0)
        mpu.initialize_model_parallel(1, 1)
    try:
        engine = types.SimpleNamespace(engine_config=types.SimpleNamespace(
            entropy_from_logits_with_chunking=True, entropy_from_logits_chunk_size=128))
        torch.manual_seed(0)
        h = torch.randn(1, 512, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w = (torch.randn(32000, 256, device="cuda", dtype=torch.bfloat16) * 0.05).requires_grad_()
        labels = torch.randint(0, 32000, (1, 512), device="cuda")
        temperature = torch.ones(1, 512, device="cuda")
        temperature[:, -16:] = 0                         # padding
        keep = temperature > 0

        def run(fn, fp32):
            def call(**opts):
                logits = h @ w.t()
                return fn(engine, logits.float() if fp32 else logits, labels, temperature.clone(), **{**_OPTS, **opts})

            with torch.no_grad():                        # RL's old-log-prob pass: entropy, no grad
                out = call(calculate_entropy=True)       # (verl's chunked entropy cannot take a backward)
            lp_ent, ent = out["log_probs"][keep], out["entropy"][keep]
            h.grad = w.grad = None                       # the update pass: grad, no entropy
            out = call()
            (-(out["log_probs"] * keep).sum()).backward()
            return lp_ent, ent, out["log_probs"][keep].detach(), h.grad.float(), w.grad.float()

        ref = run(proc.__wrapped__, fp32=True)
        new = run(proc, fp32=False)
        logp = torch.log_softmax((h @ w.t()).float(), -1)
        exact = (logp.gather(-1, labels[..., None])[..., 0][keep], -(logp.exp() * logp).sum(-1)[keep])
        for got in (new, ref):
            for value, want in zip((got[0], got[1], got[2]), (exact[0], exact[1], exact[0])):
                assert (value - want).abs().max() < 1e-3
        for a, b in zip(new[:3], ref[:3]):
            assert (a - b).abs().max() < 1e-4
        for a, b in zip(new[3:], ref[3:]):
            assert (a - b).norm() / b.norm() < 1e-2
    finally:
        if created:
            mpu.destroy_model_parallel()
            dist.destroy_process_group()


def test_sft_launch_imports_the_patches(tmp_path, monkeypatch):
    from dpt.launch.sft import build

    monkeypatch.setenv("DPT_HW_PROFILE", "local_4xa100_40g")
    run = build("sft/sft_smoke_0p8b.yaml", [f"sft.output_dir={tmp_path}"], require_manifest=False)
    assert "dpt.verl_ext.patches" in run.env["VERL_USE_EXTERNAL_MODULES"]
