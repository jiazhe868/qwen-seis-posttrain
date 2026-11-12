"""SFT mixture builder (report §SFT 语料, §阶段一).

Inputs are four subsets of :class:`SFTSample` JSONL:

=================  ==========  =================================================================
subset             target      admission rule (applied here, visible in the funnel)
=================  ==========  =================================================================
domain_reasoning   210M tok    ``meta.verified``: rejection-sampled, the answer passed the verifier
tool_traj           80M tok    ``meta.exec_ok`` and ``meta.hidden_tests_passed``: really executed
                               in the sandbox and passed the hidden tests
self_distill       105M tok    ``meta.generated_by == base model``: the base's own answers on
                   (25%)       general prompts, keeping its format, tone and tool-call habits
report              25M tok    ``meta.rubric`` present: expert-written. The rubric is reused by T3.
knowledge_report               ``meta.knowledge_checked``: written from an openly licensed paper and passed
                               the Gemma check (dpt.data.gen.knowledge, 2026-11-11; SFT v2 only)
=================  ==========  =================================================================

Then, in order:

1. **Quality**: 4-gram repetition > 0.30 → drop (the SFT loss-spike lesson).
2. **Decontamination**: prompt-side layers plus the teacher-recitation check
   (``decontam.Decontaminator.check_sft``).
3. **Render + tokenize** with the model's own chat template, with the
   reasoning effort passed **explicitly** and verified
   (``dpt.chat.render``). Samples longer than ``max_length`` are dropped,
   never truncated. A truncated sample teaches the model to stop mid-thought.
4. **Mixture**: subsample ``self_distill`` so it is exactly
   ``self_distill_token_share`` (25%) of the final *tokens*. The share is
   measured in tokens because the loss is per token. Ablation: SFT without
   replay was −3.2 on general benchmarks, with 25% replay −1.1.
5. **Length buckets** 4K/8K/16K/32K. Megatron's Gated DeltaNet path has no
   packed-sequence support, so every micro-batch is padded. Bucketing keeps
   that padding small. :func:`padding_stats` reports the overhead against the
   < 8% target.

SFT v2 (2026-11-11) continues from SFT v1 on the knowledge rows (dpt.data.gen.knowledge), built by the same stages,
plus a replay sample of the v1 mix so the model keeps what v1 taught (:func:`replay_mix`)::

    python -m dpt.data.sft_build replay-mix --new $DATA/v2_knowledge/sft/sft_train.jsonl \
        --old $DATA/v1/sft/sft_train.jsonl --replay-token-ratio 2.0 --out $DATA/sft_v2/sft_train.jsonl
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from dpt.common.hashing import short_hash
from dpt.common.io import read_jsonl, write_json, write_jsonl
from dpt.common.logging import get_logger
from dpt.data.quality import QualityConfig, quality_issues
from dpt.data.schema import SchemaError, SFTSample

log = get_logger(__name__)


@dataclass
class SFTBuildConfig:
    self_distill_token_share: float = 0.25
    buckets: list[int] = field(default_factory=lambda: [4096, 8192, 16384, 32768])
    max_length: int = 32768
    micro_batch_size: int = 2
    pad_to_multiple_of: int = 128
    padding_overhead_target: float = 0.08
    self_distill_generated_by: str = "Qwen/Qwen3.8-27B"
    seed: int = 1

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "SFTBuildConfig":
        d = dict(d or {})
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def admission_reason(s: SFTSample, cfg: SFTBuildConfig) -> str | None:
    """Return why ``s`` is *not* admitted to its subset, or ``None`` if it is admitted."""
    m = s.meta
    if s.subset == "domain_reasoning" and m.get("verified") is not True:
        return "unverified_answer"
    if s.subset == "tool_traj" and not (m.get("exec_ok") is True and m.get("hidden_tests_passed") is True):
        return "trajectory_not_executed_or_failed_hidden_tests"
    if s.subset == "self_distill" and m.get("generated_by") != cfg.self_distill_generated_by:
        return "not_generated_by_base"
    if s.subset == "report" and not m.get("rubric"):
        return "missing_rubric"
    if s.subset == "knowledge_report" and m.get("knowledge_checked") is not True:
        return "knowledge_not_checked"
    return None


def load_subsets(paths: dict[str, str]) -> tuple[list[SFTSample], Counter]:
    """Load ``{subset: path}``; the subset name from the mapping overrides any per-record value."""
    out, errors = [], Counter()
    for subset, path in paths.items():
        for rec in read_jsonl(path):
            rec = {**rec, "subset": subset}
            try:
                out.append(SFTSample.from_dict(rec))
            except SchemaError:
                errors[f"schema:{subset}"] += 1
    return out, errors


def bucket_for(n_tokens: int, buckets: Sequence[int]) -> int | None:
    for b in sorted(buckets):
        if n_tokens <= b:
            return b
    return None


def _round_up(x: int, m: int) -> int:
    return int(math.ceil(x / m) * m) if m > 1 else x


def padding_stats(
    lengths: Sequence[int],
    buckets: Sequence[int],
    *,
    micro_batch_size: int,
    pad_to_multiple_of: int,
    seed: int = 1,
) -> dict[str, Any]:
    """Padding overhead (pad tokens / padded tokens) with and without length bucketing.

    *Bucketed*: within each bucket, sort by length, cut into micro-batches,
    and pad each to its longest member rounded up to ``pad_to_multiple_of``
    (TP × SP divisibility), capped at the bucket size.
    *Unbucketed*: the same micro-batch size on a shuffled stream, so each
    micro-batch is padded to its own longest member. Comparing the two shows
    what bucketing buys.
    """

    def _overhead(groups: Iterable[list[int]], cap: int | None = None) -> tuple[int, int]:
        real = padded = 0
        for g in groups:
            if not g:
                continue
            target = _round_up(max(g), pad_to_multiple_of)
            if cap is not None:  # every member of a bucket fits in the bucket, so never pad past it
                target = min(target, cap)
            real += sum(g)
            padded += target * len(g)
        return real, padded

    by_bucket: dict[int, list[int]] = defaultdict(list)
    for n in lengths:
        b = bucket_for(n, buckets)
        if b is not None:
            by_bucket[b].append(n)
    per_bucket, tot_real, tot_padded = {}, 0, 0
    for b in sorted(by_bucket):
        ls = sorted(by_bucket[b])
        groups = [ls[i:i + micro_batch_size] for i in range(0, len(ls), micro_batch_size)]
        r, p = _overhead(groups, cap=b)
        per_bucket[b] = {"n": len(ls), "tokens": r, "overhead": round(1 - r / p, 4) if p else 0.0}
        tot_real += r
        tot_padded += p
    shuffled = list(lengths)
    random.Random(seed).shuffle(shuffled)
    ug = [shuffled[i:i + micro_batch_size] for i in range(0, len(shuffled), micro_batch_size)]
    ur, up = _overhead(ug)
    return {
        "bucketed_overhead": round(1 - tot_real / tot_padded, 4) if tot_padded else 0.0,
        "unbucketed_overhead": round(1 - ur / up, 4) if up else 0.0,
        "per_bucket": per_bucket,
    }


@dataclass
class SFTBuildResult:
    samples: list[SFTSample]
    dropped: list[dict[str, Any]] = field(default_factory=list)
    stage_stats: list[dict[str, Any]] = field(default_factory=list)
    mixture: dict[str, Any] = field(default_factory=dict)
    padding: dict[str, Any] = field(default_factory=dict)
    decontam: Any = None          # DecontamResult of the SFT check (for the contamination report)


def select_self_distill(
    domain_tokens: int, replay: list[SFTSample], share: float, seed: int
) -> tuple[list[SFTSample], int]:
    """Pick replay samples so replay / (domain + replay) ≈ ``share`` in tokens (seeded, no upsampling)."""
    if not 0 <= share < 1:
        raise ValueError("self_distill_token_share must be in [0, 1)")
    target = share / (1 - share) * domain_tokens
    pool = list(replay)
    random.Random(seed).shuffle(pool)
    chosen, total = [], 0
    for s in pool:
        n = int(s.meta["n_tokens"])
        if total >= target:
            break
        # Stop if adding this sample overshoots by more than skipping it would undershoot.
        if total + n - target > target - total:
            continue
        chosen.append(s)
        total += n
    return chosen, total


def build_sft(
    samples: Sequence[SFTSample],
    tokenizer: Any,
    cfg: SFTBuildConfig,
    *,
    quality: QualityConfig | None = None,
    decontaminator: Any | None = None,
) -> SFTBuildResult:
    """Run admission → quality → decontam → render/tokenize → mixture → buckets."""
    from dpt.chat.render import EffortRenderingError, tokenize_with_assistant_mask

    quality = quality or QualityConfig()
    res = SFTBuildResult(samples=[])
    stats: list[dict[str, Any]] = [{"stage": "loaded", "count": len(samples)}]

    def _drop(s: SFTSample, reason: str) -> None:
        res.dropped.append({"uid": s.uid, "subset": s.subset, "reason": reason})

    cur: list[SFTSample] = []
    for s in samples:
        r = admission_reason(s, cfg)
        _drop(s, r) if r else cur.append(s)
    stats.append({"stage": "admission", "count": len(cur)})

    nxt = []
    for s in cur:
        issues = quality_issues(s.assistant_text(), quality)
        _drop(s, "quality:" + ",".join(issues)) if issues else nxt.append(s)
    cur = nxt
    stats.append({"stage": "quality", "count": len(cur)})

    if decontaminator is not None:
        dres = decontaminator.check_sft(cur)
        for r in dres.removed:
            res.dropped.append({"uid": r["uid"], "reason": f"decontam:{r['layer']}", "match": r["match"],
                                "source": r.get("source")})
        cur = dres.kept
        res.decontam = dres
        res.mixture["decontam"] = dres.stats()
    stats.append({"stage": "decontam", "count": len(cur)})

    nxt = []
    for s in cur:
        try:
            enc = tokenize_with_assistant_mask(tokenizer, s.messages, effort=s.reasoning_effort,
                                               tools=s.tools)
        except EffortRenderingError as e:
            _drop(s, f"render_effort:{e}")
            continue
        except Exception as e:  # template errors (bad role order, malformed tool calls) → drop, report
            _drop(s, f"render_error:{type(e).__name__}")
            continue
        n = len(enc["input_ids"])
        if n > cfg.max_length:
            _drop(s, "too_long")
            continue
        s.meta["n_tokens"] = n
        s.meta["n_loss_tokens"] = int(sum(enc["loss_mask"]))
        s.meta["bucket"] = bucket_for(n, cfg.buckets)
        nxt.append(s)
    cur = nxt
    stats.append({"stage": "render_tokenize", "count": len(cur)})

    domain = [s for s in cur if s.subset != "self_distill"]
    replay = [s for s in cur if s.subset == "self_distill"]
    domain_tokens = sum(int(s.meta["n_tokens"]) for s in domain)
    chosen, replay_tokens = select_self_distill(domain_tokens, replay, cfg.self_distill_token_share, cfg.seed)
    chosen_ids = {s.uid for s in chosen}
    for s in replay:
        if s.uid not in chosen_ids:
            _drop(s, "mixture:replay_surplus")
    total = domain_tokens + replay_tokens
    actual = replay_tokens / total if total else 0.0
    if replay and abs(actual - cfg.self_distill_token_share) > 0.02:
        log.warning("self-distill share %.3f differs from target %.3f (not enough replay data?)",
                    actual, cfg.self_distill_token_share)
    final = sorted(domain + chosen, key=lambda s: s.uid)
    stats.append({"stage": "mixture", "count": len(final)})

    tokens_by_subset: Counter = Counter()
    for s in final:
        tokens_by_subset[s.subset] += int(s.meta["n_tokens"])
    res.mixture.update({
        "tokens_by_subset": dict(tokens_by_subset),
        "self_distill_token_share": round(actual, 4),
        "target_share": cfg.self_distill_token_share,
        "total_tokens": total,
    })
    res.padding = padding_stats([int(s.meta["n_tokens"]) for s in final], cfg.buckets,
                                micro_batch_size=cfg.micro_batch_size,
                                pad_to_multiple_of=cfg.pad_to_multiple_of, seed=cfg.seed)
    if res.padding["bucketed_overhead"] > cfg.padding_overhead_target:
        log.warning("bucketed padding overhead %.3f exceeds target %.3f",
                    res.padding["bucketed_overhead"], cfg.padding_overhead_target)
    res.samples = final
    res.stage_stats = stats
    return res


def replay_mix(new: Sequence[dict[str, Any]], old: Sequence[dict[str, Any]], *, replay_token_ratio: float,
               seed: int = 1) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The new built rows plus a seeded random sample of old built rows holding ``replay_token_ratio`` × the new rows'
    tokens. Rows are drawn whole and in a hashed order, so the sample keeps the old mix's subset shares on average,
    its 25% self-distilled replay included."""
    new_tokens = sum(int(r["meta"]["n_tokens"]) for r in new)
    picked, replay_tokens = [], 0
    for r in sorted(old, key=lambda r: short_hash(f"{seed}|replay|{r['uid']}")):
        if replay_tokens >= replay_token_ratio * new_tokens:
            break
        picked.append(r)
        replay_tokens += int(r["meta"]["n_tokens"])
    rows = sorted(list(new) + picked, key=lambda r: r["uid"])       # the trainer shuffles; this order is stable
    tokens: Counter = Counter()
    for r in rows:
        tokens[r["subset"]] += int(r["meta"]["n_tokens"])
    return rows, {"rows": len(rows), "new_rows": len(new), "replay_rows": len(picked), "new_tokens": new_tokens,
                  "replay_tokens": replay_tokens, "tokens_by_subset": dict(tokens)}


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    from pathlib import Path

    from dpt.common.manifest import build_manifest, load_manifest, manifest_path_for, write_manifest

    ap = argparse.ArgumentParser(description="SFT mixture tools")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay-mix", help="new built rows + a replay sample of an older built mix")
    r.add_argument("--new", required=True, type=Path)
    r.add_argument("--old", required=True, type=Path)
    r.add_argument("--replay-token-ratio", type=float, default=2.0)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    rows, summary = replay_mix(list(read_jsonl(a.new)), list(read_jsonl(a.old)),
                               replay_token_ratio=a.replay_token_ratio, seed=a.seed)
    write_jsonl(a.out, rows)
    parents = {str(p): load_manifest(p).content_hash for p in (a.new, a.old) if manifest_path_for(p).exists()}
    write_manifest(build_manifest(a.out, name="sft_train", version="v2", parents=parents,
                                  notes=f"SFT v2: {a.new.name} + replay of {a.old} ({a.replay_token_ratio}× tokens)"))
    write_json(a.out.with_suffix(".summary.json"), summary)
    log.info("replay mix: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
