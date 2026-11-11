"""GPU check: the training-mode forward must give the same log-probs as the eval-mode forward.

PPO's ratio π_new/π_old compares log-probs from two forwards of the same weights: the old-log-prob
pass (eval mode, no grad) and the actor-update pass (train mode, grad). If those two passes
disagree before any update, the ratio, clipping and TIS act on noise, and SFT trains the wrong graph.
Nothing else catches this; the vLLM-vs-Megatron parity only checks the eval-mode pass.

Found on the workstation RL smoke run: on A100 the train-mode pass of Qwen3.5/3.8 was 1.0-1.8
nats/token below the eval pass, because Transformer Engine switches attention backend in training
and verl's VLM path hands Megatron an inverted mask (``dpt.verl_ext.patches``, fix 1).

What it does (single GPU, Megatron-Bridge, TP1/PP1, verl's recompute settings):

1. tokenizes a long text and builds verl's model inputs with verl's own
   ``build_vlm_attn_mask_bshd``, exactly as the actor does (``position_ids=None`` + mask);
2. scores the tokens in eval/no-grad mode and in train/grad mode, (a) with the mask as patched by
   ``dpt.verl_ext.patches`` (production path), (b) with verl's raw mask (shows whether the
   underlying bug is present in the installed stack);
3. passes if (a) agrees within kernel noise: |mean Δ| < 5e-3 and mean |Δ| < 5e-2 nats/token.

    python -m dpt.parity.train_forward_parity --model Qwen/Qwen3.5-0.8B --tokens 2048 \\
        --out parity/train_forward.json

The 0.8B smoke model has the same attention shape (head_dim 256) as the 27B, so it exercises the
same Transformer Engine backend choice. Run it after any upgrade of verl, Megatron-Bridge, TE or cuDNN.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

SIGNED_MAX = 5e-3      # |mean(ref − lp)|, nats/token: a systematic shift means a different graph
ABS_MAX = 5e-2         # mean |ref − lp|: BF16 kernel differences give ~2.5e-2 on Qwen3.5-0.8B


def _stats(lp: torch.Tensor, ref: torch.Tensor) -> dict:
    d = ref - lp
    return {"mean_signed": float(d.mean()), "mean_abs": float(d.abs().mean()), "max_abs": float(d.abs().max())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--text-file", type=Path, default=Path(__file__).resolve().parents[3] / "README.md")
    ap.add_argument("--out", type=Path, default=Path("parity/train_forward.json"))
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; this check must run on the target GPU.")

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(29600 + os.getpid() % 1000))
    torch.cuda.set_device(0)
    torch.distributed.init_process_group("nccl", rank=0, world_size=1)

    import dpt.verl_ext.patches  # noqa: F401  (installs the fix before verl's model_forward is imported)
    from megatron.bridge import AutoBridge
    from megatron.core.transformer.enums import AttnBackend
    from transformers import AutoTokenizer
    from verl.models.mcore import model_forward as mf

    tok = AutoTokenizer.from_pretrained(a.model)
    ids = tok(a.text_file.read_text(), return_tensors="pt").input_ids[0, : a.tokens].cuda()

    provider = AutoBridge.from_hf_pretrained(a.model).to_megatron_provider(load_weights=True)
    provider.tensor_model_parallel_size = provider.pipeline_model_parallel_size = 1
    provider.sequence_parallel = False
    provider.gradient_accumulation_fusion = False
    provider.attention_backend = AttnBackend.auto
    provider.bf16, provider.params_dtype = True, torch.bfloat16
    provider.mtp_num_layers = None
    provider.recompute_granularity, provider.recompute_method, provider.recompute_num_layers = "full", "uniform", 1
    provider.finalize()
    provider.initialize_model_parallel(seed=0)
    model = provider.provide_distributed_model(wrap_with_ddp=False)[0]

    nested = torch.nested.as_nested_tensor([ids], layout=torch.jagged)
    builder = mf.build_vlm_attn_mask_bshd
    patched = getattr(builder, "_dpt_fixed", False)
    raw_builder = getattr(builder, "__wrapped__", builder)

    def score(mask_builder, train: bool) -> torch.Tensor:
        input_ids, mask = mask_builder(nested, 1, tok.pad_token_id)
        model.train(train)
        with torch.set_grad_enabled(train):
            logits = model(input_ids=input_ids, position_ids=None, attention_mask=mask)
        n = ids.shape[0]
        lp = torch.log_softmax(logits[0, : n - 1].float(), -1).gather(-1, ids[1:, None])[:, 0]
        return lp.detach()

    res = {"model": a.model, "tokens": int(ids.shape[0]), "device": torch.cuda.get_device_name(),
           "capability": "sm%d%d" % torch.cuda.get_device_capability(), "patch_installed": patched}
    ref = score(builder, train=False)
    res["production"] = _stats(score(builder, train=True), ref)
    res["raw_verl_mask"] = _stats(score(raw_builder, train=True), score(raw_builder, train=False))
    p = res["production"]
    res["pass"] = patched and abs(p["mean_signed"]) < SIGNED_MAX and p["mean_abs"] < ABS_MAX
    raw = res["raw_verl_mask"]
    res["upstream_bug_present"] = abs(raw["mean_signed"]) >= SIGNED_MAX or raw["mean_abs"] >= ABS_MAX

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    torch.distributed.destroy_process_group()
    sys.stdout.flush()
    sys.stderr.flush()
    # os._exit, not sys.exit: on the cluster Transformer Engine's RTC kernel cache segfaults in its static
    # destructor at interpreter exit, which turned a passing check into a non-zero exit code.
    os._exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
