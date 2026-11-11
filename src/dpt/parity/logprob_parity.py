"""Token-level logprob parity: vLLM (rollout engine) vs. a reference forward pass.

The report's first engineering action on A100 is a parity test: take one
checkpoint, compare per-token logprobs between the inference engine and the
training stack, and measure the GDN kernels on SM80. This tool does the
engine-side half offline, in three isolated phases (separate processes, because
vLLM does not release GPU memory reliably within a process):

    generate  vLLM samples N responses at temperature 1 / top_p 1 and records each sampled
              token's logprob (= what TIS uses as π_rollout)
    score     Hugging Face forward pass over the same token ids → reference logprobs
    compare   dpt.rl.mismatch metrics, sliced by token position; PASS iff the mean |log-ppl
              diff| is under the alert line (2e-3) and does not grow with position

    python -m dpt.parity.logprob_parity --model Qwen/Qwen3.8-27B --tp 4 --phase all \
        --out parity/logprob_27b_fp32state
    # Reproduce the Incident 1 mechanism (expect drift that grows with position):
    python -m dpt.parity.logprob_parity --model ... --ssm-dtype bfloat16 --out parity/logprob_bf16state

Caveat on ``--min-tokens``. Forcing a small model to 16K tokens makes its late text repetitive
and near-deterministic, and confident tokens have tiny logprob differences. The late-position
buckets are therefore *less* sensitive than on natural long reasoning. A pass rules out gross
drift. The decisive long-position check is the train/rollout mismatch on real long responses in
the RL smoke run (step 05) and on the 27B.

vLLM vs **Megatron** (the actual training forward) is measured inside verl.
The first steps of ``scripts/05_rl_smoke.sh`` log ``rollout_corr/log_ppl_abs_diff``
and our sliced ``mismatch/pos/*`` metrics for the same comparison. The pass
criteria are identical.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from dpt.common.io import read_jsonl, write_json, write_jsonl
from dpt.common.logging import get_logger

log = get_logger(__name__)

# Eight distinct problems that make the model reason at length on its own (at xhigh effort the 27B
# writes ~10-16K tokens). Natural long responses are what RL rollouts look like; responses forced
# past their natural end (--min-tokens) degenerate into loops and are not representative.
DEFAULT_PROMPTS = [
    "A layered crust has a 12 km upper layer (Vp 5.8 km/s) over a lower layer (Vp 6.8 km/s) above the Moho "
    "at 35 km, mantle Vp 8.1 km/s. Derive the head-wave travel-time equations for both refractors, the "
    "crossover distances, and check them numerically. Be rigorous.",
    "Derive the Hanks-Kanamori moment-magnitude relation from first principles, discuss its assumptions, "
    "then compute Mw, rupture area and average slip for M0 = 3.2e20 N·m with stress drop 3 MPa.",
    "Write and verify a Python implementation of an STA/LTA earthquake detector with a recursive filter, "
    "including edge cases, complexity analysis and tests.",
    "Explain, with quantitative evidence, why S-waves do not propagate through the outer core, and how "
    "PKiKP and PKJKP observations constrain inner-core shear velocity.",
    "Locate an earthquake from P arrivals at four stations (give coordinates and times you choose), using "
    "Geiger's method by hand for two iterations, and discuss the error ellipse.",
    "Derive the Gutenberg-Richter b-value maximum-likelihood estimator (Aki 1965), its standard error, and "
    "apply it to a synthetic catalog you construct; discuss magnitude-of-completeness bias.",
    "A strike-slip fault has strike 30°, dip 90°, rake 0°. Derive the P-wave radiation pattern, find the "
    "nodal planes and the P/T axes, and sketch the beach ball in words.",
    "Compute the surface-wave group velocity dispersion for a single layer over a half-space (Love waves), "
    "deriving the dispersion relation and solving it numerically for three periods.",
]
PASS_MEAN_ABS = 2e-3          # the dashboard's warning line for train/rollout log-ppl difference
MAX_LATE_EARLY_RATIO = 3.0    # late-position drift vs early-position drift (Incident 1 signature)


# ------------------------------------------------------------------------------------------------
def phase_generate(a: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from dpt.chat.render import render_chat

    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = [r["prompt"] for r in read_jsonl(a.prompts)] if a.prompts else DEFAULT_PROMPTS
    prompts = (prompts * ((a.n + len(prompts) - 1) // len(prompts)))[: a.n]
    texts = [render_chat(tok, [{"role": "user", "content": p}], effort=a.effort, add_generation_prompt=True)
             for p in prompts]
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, pipeline_parallel_size=a.pp,
              gpu_memory_utilization=a.gpu_mem, max_model_len=a.max_model_len,
              mamba_ssm_cache_dtype=a.ssm_dtype, enable_prefix_caching=False, seed=a.seed,
              language_model_only=True,          # vision tower is out of scope; frees its memory
              max_num_seqs=max(1, a.n))
    # min_tokens forces long responses: drift from a low-precision recurrent state shows up late
    # (Incident 1: after ~12K tokens), and natural responses are often much shorter.
    sp = SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=a.max_tokens,
                        min_tokens=a.min_tokens, logprobs=0)          # engine seed makes the run reproducible
    rows = []
    for text, out in zip(texts, llm.generate(texts, sp)):
        o = out.outputs[0]
        lps = [float(d[t].logprob) for t, d in zip(o.token_ids, o.logprobs)]
        rows.append({"prompt_ids": list(out.prompt_token_ids), "response_ids": list(o.token_ids),
                     "rollout_logp": lps, "finish_reason": o.finish_reason})
    write_jsonl(Path(a.out) / "generated.jsonl", rows)
    log.info("generated %d responses (mean %.0f tokens)", len(rows),
             sum(len(r["response_ids"]) for r in rows) / max(1, len(rows)))


def _load_reference_model(name: str, dtype: str, max_mem_gib: float | None = None):
    import torch
    import transformers

    td = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
    try:
        import accelerate  # noqa: F401  (needed for device_map="auto", i.e. sharding a 27B over GPUs)

        place: dict = {"device_map": "auto"}
        if max_mem_gib:   # shared GPUs: never take more than this per visible device
            place["max_memory"] = {i: f"{max_mem_gib}GiB" for i in range(torch.cuda.device_count())}
    except ImportError:
        log.warning("accelerate not installed: loading the reference model on one GPU (fine for small models)")
        place = {}
    # Qwen3.5/3.8 checkpoints are multimodal (Qwen3_5ForConditionalGeneration); try the text-only head first.
    errors = []
    for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModelForMultimodalLM"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            # SDPA picks a flash / memory-efficient kernel for BF16, so no T×T score matrix is built.
            model = cls.from_pretrained(name, dtype=td, attn_implementation="sdpa", **place)
            return (model if place else model.to("cuda")).eval()
        except (ValueError, KeyError) as e:
            errors.append(f"{cls_name}: {e}")
            log.info("%s cannot load %s (%s); trying next", cls_name, name, e)
    raise RuntimeError(f"no transformers auto-class could load {name}: " + " | ".join(errors))


def chunked_token_logprobs(hidden, head, targets, chunk: int = 1024):
    """log p(target_t | h_t) for every position, never materializing [T × vocab] logits at once.

    With a 248K vocabulary, full FP32 logits for one 16K-token sequence are ~16 GB (the same trap
    the report hits on the training side). Projecting 1K positions at a time keeps this at ~1 GB.
    """
    import torch

    out = []
    for s in range(0, hidden.shape[0], chunk):
        logits = head(hidden[s:s + chunk]).float()
        lp = torch.log_softmax(logits, dim=-1)
        out.append(lp.gather(-1, targets[s:s + chunk, None])[:, 0])
    return torch.cat(out)


def _tag(a: argparse.Namespace) -> str:
    return "" if a.ref_gdn == "fla" else f".ref-{a.ref_gdn}"


def phase_score(a: argparse.Namespace) -> None:
    import sys

    import torch

    if a.ref_gdn == "torch":
        # transformers tries `import fla` for the GDN kernels and falls back to its PyTorch implementation
        # (which computes the delta rule in FP32) on any import error. Blocking the import selects it.
        sys.modules["fla"] = None  # type: ignore[assignment]
    model = _load_reference_model(a.model, a.ref_dtype, a.ref_max_mem_gib)
    backbone, head = model.model, model.get_output_embeddings()
    rows = list(read_jsonl(Path(a.out) / "generated.jsonl"))
    for r in rows:
        ids = torch.tensor([r["prompt_ids"] + r["response_ids"]], device=head.weight.device)
        p = len(r["prompt_ids"])
        with torch.no_grad():
            h = backbone(input_ids=ids).last_hidden_state[0]
            # position t predicts token t+1: response tokens p..end are predicted by hidden p-1..end-1
            lp = chunked_token_logprobs(h[p - 1 : -1], head, ids[0, p:])
        r["ref_logp"] = lp.cpu().tolist()
        del h
        torch.cuda.empty_cache()
    write_jsonl(Path(a.out) / f"scored{_tag(a)}.jsonl", rows)


def compare(rows: list[dict[str, Any]], position_edges=(4096, 8192, 12288)) -> dict[str, Any]:
    """Pure function (unit-tested): mismatch metrics + PASS/FAIL from generated+scored rows."""
    import torch

    from dpt.rl.mismatch import mismatch_metrics

    L = max(len(r["response_ids"]) for r in rows)
    roll = torch.zeros(len(rows), L)
    ref = torch.zeros(len(rows), L)
    mask = torch.zeros(len(rows), L)
    for i, r in enumerate(rows):
        n = len(r["response_ids"])
        roll[i, :n] = torch.tensor(r["rollout_logp"])
        ref[i, :n] = torch.tensor(r["ref_logp"])
        mask[i, :n] = 1
    m = mismatch_metrics(ref, roll, mask, position_edges=position_edges)
    pos = {k: v for k, v in m.items() if k.startswith("mismatch/pos/") and k.endswith("/abs_log_ratio")}
    vals = list(pos.values())
    ratio = (vals[-1] / max(vals[0], 1e-12)) if len(vals) >= 2 else 1.0
    ok = m["mismatch/log_ppl_abs_diff"] < PASS_MEAN_ABS and ratio < MAX_LATE_EARLY_RATIO
    return {"pass": bool(ok), "late_early_ratio": ratio, "metrics": m, "n": len(rows)}


def phase_compare(a: argparse.Namespace) -> int:
    res = compare(list(read_jsonl(Path(a.out) / f"scored{_tag(a)}.jsonl")))
    res["settings"] = {k: getattr(a, k) for k in
                       ("model", "ssm_dtype", "ref_dtype", "ref_gdn", "effort", "tp", "pp", "max_tokens", "min_tokens")}
    write_json(Path(a.out) / f"parity{_tag(a)}.json", res)
    log.info("parity %s: log_ppl_abs_diff=%.2e late/early=%.2f", "PASS" if res["pass"] else "FAIL",
             res["metrics"]["mismatch/log_ppl_abs_diff"], res["late_early_ratio"])
    return 0 if res["pass"] else 1


def main() -> None:
    p = argparse.ArgumentParser(description="vLLM vs reference token logprob parity")
    p.add_argument("--model", required=True)
    p.add_argument("--phase", choices=["generate", "score", "compare", "all"], default="all")
    p.add_argument("--prompts", default=None, help="JSONL with a 'prompt' field (default: built-in)")
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--min-tokens", type=int, default=0, help="force long responses to probe late positions")
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--pp", type=int, default=1, help="vLLM pipeline parallel size (e.g. 3 GPUs for the 27B)")
    p.add_argument("--ref-max-mem-gib", type=float, default=None,
                   help="cap per GPU for the HF reference model (device_map=auto); leave room for other jobs")
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--ssm-dtype", default="float32", choices=["float32", "bfloat16", "auto"])
    p.add_argument("--ref-dtype", default="bfloat16", choices=["float32", "bfloat16"],
                   help="bfloat16 mirrors the training forward; float32 only for short responses (memory)")
    p.add_argument("--effort", default="xhigh", help="xhigh → naturally long reasoning (positions past 12K)")
    p.add_argument("--ref-gdn", default="fla", choices=["fla", "torch"],
                   help="reference GDN kernel: fla (chunked Triton, like Megatron training) or torch (FP32 math)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.phase == "all":
        args = sys.argv[1:]
        if "--phase" in args:                       # drop "--phase all"; each child gets its own phase
            i = args.index("--phase")
            args = args[:i] + args[i + 2:]
        base = [sys.executable, "-m", "dpt.parity.logprob_parity", *args]
        for ph in ("generate", "score", "compare"):
            rc = subprocess.call(base + ["--phase", ph])
            if rc:
                raise SystemExit(rc)
        return
    if a.phase == "generate":
        phase_generate(a)
    elif a.phase == "score":
        phase_score(a)
    else:
        raise SystemExit(phase_compare(a))


if __name__ == "__main__":
    main()
