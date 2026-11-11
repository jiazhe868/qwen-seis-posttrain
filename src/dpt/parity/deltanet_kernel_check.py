"""GPU check: flash-linear-attention Gated DeltaNet kernels vs. the FP64 reference, on this GPU.

Run before any training (report §工程设计 "A100 本身是风险"):

    python -m dpt.parity.deltanet_kernel_check --lengths 1024 4096 16384 32768 --out parity/deltanet.json

What it measures, for each sequence length:

* ``chunk``: ``chunk_gated_delta_rule``, the chunked Triton kernel used for
  training forward/backward and for prefill, in BF16 inputs with FP32 state.
* ``recurrent``: ``fused_recurrent_gated_delta_rule``, the token-by-token
  kernel used for decode, with an FP32 state.
* ``recurrent_bf16_state``: the decode kernel with a BF16 initial state carried
  across calls of ``step`` tokens each. This emulates a BF16 recurrent-state
  cache (Incident 1). Expect it to be orders of magnitude worse.

For each variant it records mean/max |Δo| per position quartile and |Δ final state|,
and checks them against thresholds. The exit code is non-zero if a required
variant fails, so ``scripts/02_parity.sh`` can gate the project on it.

Shapes default to one Qwen3.8 GDN layer after GQA expansion (48 heads, K=V=128).
The reference runs on GPU in FP64; the recurrence is O(T) sequential steps,
so 32K tokens takes a few minutes. Use ``--heads`` to subsample heads.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from dpt.common.logging import get_logger
from dpt.parity.deltanet_reference import error_by_position, gated_delta_rule_reference, random_inputs

log = get_logger(__name__)

# Thresholds on mean |Δo| relative to mean |o_ref|. BF16 inputs bound achievable accuracy to ~1e-2
# relative per element; we require the *mean* relative error to stay below these values and, more
# importantly, not to grow with position (late/early ratio).
REL_ERR_MAX = {"chunk": 2e-2, "recurrent": 2e-2}
LATE_EARLY_RATIO_MAX = 2.0


def _import_fla():
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    except ImportError as e:  # pragma: no cover - environment dependent
        raise SystemExit(
            "flash-linear-attention is not installed (pip install flash-linear-attention). "
            f"Import error: {e}"
        )
    return chunk_gated_delta_rule, fused_recurrent_gated_delta_rule


def _call(fn, x, *, initial_state=None):
    """Call an FLA kernel with KEYWORD arguments for everything after q, k, v.

    The positional order differs between kernels and releases. In fla 0.5.x,
    ``fused_recurrent_gated_delta_rule(q, k, v, g, gk, gv, beta, ...)`` takes per-key
    and per-value decays before ``beta``, so a positional ``beta`` lands in ``gk``.
    The kernel then reads past the end of a too-small tensor (illegal memory access).
    """
    kwargs = dict(g=x["g"], beta=x["beta"], initial_state=initial_state, output_final_state=True,
                  use_qk_l2norm_in_kernel=True)
    try:
        return fn(x["q"], x["k"], x["v"], **kwargs)
    except TypeError:                     # very old releases without in-kernel L2 norm
        kwargs.pop("use_qk_l2norm_in_kernel")
        from dpt.parity.deltanet_reference import l2norm

        return fn(l2norm(x["q"]), l2norm(x["k"]), x["v"], **kwargs)


def check_length(T: int, heads: int, dim: int, step: int, seed: int) -> dict:
    chunk_fn, rec_fn = _import_fla()
    dev = "cuda"
    x32 = random_inputs(B=1, T=T, H=heads, K=dim, V=dim, seed=seed, device=dev, dtype=torch.float32)
    x16 = {n: t.to(torch.bfloat16) for n, t in x32.items()}
    x16["g"], x16["beta"] = x32["g"], x32["beta"].to(torch.bfloat16)   # log-decay stays FP32 as in the model

    o_ref, h_ref = gated_delta_rule_reference(**x32, compute_dtype=torch.float64)
    ref_scale = float(o_ref.abs().mean())
    out: dict = {"T": T, "heads": heads, "dim": dim, "ref_mean_abs": ref_scale, "variants": {}}

    def record(name: str, o: torch.Tensor, h: torch.Tensor | None):
        buckets = error_by_position(o, o_ref)
        rel = [b / ref_scale for b in buckets]
        entry = {
            "rel_err_by_quartile": rel,
            "late_early_ratio": rel[-1] / max(rel[0], 1e-30),
            "final_state_max_abs": float((h.double() - h_ref).abs().max()) if h is not None else None,
        }
        thr = REL_ERR_MAX.get(name)
        if thr is not None:
            entry["pass"] = max(rel) < thr and entry["late_early_ratio"] < LATE_EARLY_RATIO_MAX
        out["variants"][name] = entry
        log.info("T=%d %-22s rel_err=%s ratio=%.2f", T, name, ["%.1e" % r for r in rel], entry["late_early_ratio"])

    o, h = _call(chunk_fn, x16)
    record("chunk", o.float(), h)
    o, h = _call(rec_fn, x16)
    record("recurrent", o.float(), h)

    # Decode with a BF16 state cache: feed `step` tokens per call and round the carried state.
    outs, state = [], None
    for s in range(0, T, step):
        piece = {n: t[:, s : s + step] for n, t in x16.items()}
        o, state = _call(rec_fn, piece, initial_state=state)
        state = state.to(torch.bfloat16).float()
        outs.append(o.float())
    record("recurrent_bf16_state", torch.cat(outs, dim=1), state)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Gated DeltaNet kernel parity vs FP64 reference")
    p.add_argument("--lengths", type=int, nargs="+", default=[1024, 4096, 16384])
    p.add_argument("--heads", type=int, default=8, help="heads to test (48 = full Qwen3.8 layer)")
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--step", type=int, default=1, help="tokens per decode call in the bf16-state variant")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("parity/deltanet_kernel.json"))
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; this check must run on the target GPU (A100/SM80).")
    cap = torch.cuda.get_device_capability()
    log.info("device=%s capability=sm%d%d", torch.cuda.get_device_name(), *cap)

    results = {"device": torch.cuda.get_device_name(), "capability": f"sm{cap[0]}{cap[1]}", "runs": []}
    ok = True
    for T in args.lengths:
        r = check_length(T, args.heads, args.dim, args.step, args.seed)
        results["runs"].append(r)
        ok &= all(v.get("pass", True) for v in r["variants"].values())
    results["pass"] = ok
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    log.info("wrote %s (pass=%s)", args.out, ok)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
