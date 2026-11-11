"""The fixed evaluation harness.

Every number in the report comes out of this harness. Base, SFT and RL
checkpoints are all run through it with identical settings (report §harness 规则):

* **Same prompt construction for every model.** By default
  (``mode: completions``) the harness renders prompts itself with the
  model's own chat template through :func:`dpt.chat.render.render_chat`,
  with an explicit, verified reasoning effort, and sends raw text to
  ``/v1/completions``. Server-side template or reasoning-parser differences
  therefore cannot leak in. ``mode: chat`` exists for API-only models; it
  needs a server *without* a reasoning parser, because the harness has to see
  ``</think>`` itself.
* **Fixed sampling.** temperature 1.0, top-p 1.0, effort ``medium``, max
  32K output tokens, with a per-item seed derived from the run seed.
* **Only final answers are graded.** Thinking is stripped. A sample with no
  ``</think>`` (usually truncated) scores 0 and is never graded on its raw
  thinking.
* **Truncation rate and output-token counts are always recorded.** Base at
  xhigh truncates 11% at 32K, so ``--extra-max-tokens 65536`` reruns the
  suite at 64K and reports, per set, the higher of the two scores
  (``score_by_max_tokens`` keeps both).
* **Cache and resume.** Generations are appended to
  ``<out>/<set>/generations.jsonl`` as items finish. A rerun skips items
  already generated under an identical sampling spec, and ``--regrade-only``
  re-scores cached generations, e.g. after a grader fix such as
  grader v1 → v3.

Output: ``<out>/results.json``. The schema is documented in
docs/evaluation.md and consumed by :mod:`dpt.eval.report` and
:mod:`dpt.eval.gates`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

import httpx
import numpy as np

from dpt.chat.thinking import split_thinking
from dpt.common.hashing import sha256_obj
from dpt.common.io import read_json, read_jsonl, write_json
from dpt.common.logging import get_logger
from dpt.common.seeding import derive_seed
from dpt.eval import metrics as M
from dpt.eval import stats as S
from dpt.eval.datasets import EvalItem, get_grader, load_eval_items
from dpt.eval.graders import GradeResult

log = get_logger(__name__)

Messages = list[dict[str, Any]]
Renderer = Callable[[Messages, str], str]  # (messages, effort) -> prompt text
TokenCounter = Callable[[str], int]


# --------------------------------------------------------------------------- data classes
@dataclass(frozen=True)
class SamplingSpec:
    effort: str = "medium"
    max_tokens: int = 32768
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 8
    seed: int = 0

    def cache_key(self, model: str, mode: str) -> str:
        return sha256_obj({**asdict(self), "model": model, "mode": mode})[:16]


@dataclass
class Sample:
    text: str
    finish_reason: str
    n_tokens: int


@dataclass
class SetSpec:
    name: str
    path: str
    grader: str
    samples: int
    metric: str = "avg@k"  # "avg@k" | "pass@1" | "winrate" | "refusal_rate"
    task: str | None = None
    grader_kwargs: dict[str, Any] = field(default_factory=dict)
    relative_only: bool = False

    @classmethod
    def from_cfg(cls, name: str, cfg: dict[str, Any], defaults: dict[str, Any]) -> "SetSpec":
        kw = {**defaults, **dict(cfg.get("grader_kwargs") or {})}
        return cls(
            name=name,
            path=str(cfg["path"]),
            grader=str(cfg["grader"]),
            samples=int(cfg.get("samples", 1)),
            metric=str(cfg.get("metric", "avg@k")),
            task=cfg.get("task"),
            grader_kwargs=kw,
            relative_only=bool(cfg.get("relative_only", False)),
        )


# --------------------------------------------------------------------------- prompts
def build_messages(item: EvalItem) -> Messages:
    """Prompt for an item via the shared prompt builder (same builder RL uses, so formats match).

    Falls back to a bare user turn when ``dpt.data.prompts`` is unavailable
    or the item has no task (general benchmarks whose question already
    contains the instructions).
    """
    if item.task:
        try:
            from dpt.data.prompts import build_messages as _build  # shared with RL export (rl_build)
        except ImportError:
            log.warning("dpt.data.prompts unavailable; using bare user prompt")
        else:
            extra = dict(item.answer.get("prompt_extra") or {})
            if item.task == "t2_code" and item.answer.get("entry_point"):
                extra.setdefault("entry_point", item.answer["entry_point"])
            if item.task == "t4_general":
                extra.setdefault("kind", item.answer.get("kind"))
            return list(_build(item.task, item.question, extra=extra))
    return [{"role": "user", "content": item.question}]


def hf_renderer(tokenizer_name_or_path: str) -> Renderer:
    """Renderer backed by the model's own tokenizer chat template (with effort verification)."""
    from transformers import AutoTokenizer

    from dpt.chat.render import render_chat

    tok = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    return lambda messages, effort: render_chat(tok, messages, effort=effort, add_generation_prompt=True)


def hf_token_counter(tokenizer_name_or_path: str) -> TokenCounter:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    return lambda text: len(tok(text, add_special_tokens=False)["input_ids"])


# --------------------------------------------------------------------------- backend

def _host_only(endpoint: str) -> str:
    e = endpoint.rstrip("/")
    return e[: -len("/v1")] if e.endswith("/v1") else e


class OpenAICompatBackend:
    """Async client for an OpenAI-compatible server (vLLM ``serve``)."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        mode: str = "completions",
        renderer: Renderer | None = None,
        token_counter: TokenCounter | None = None,
        api_key: str | None = None,
        timeout_s: float = 7200.0,
        max_retries: int = 3,
        max_concurrency: int = 64,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        if mode not in ("completions", "chat"):
            raise ValueError("mode must be 'completions' or 'chat'")
        if mode == "completions" and renderer is None:
            raise ValueError("completions mode needs a renderer (the model's chat template)")
        # Accept both "http://host:8000" and the OpenAI-style "http://host:8000/v1"; paths add /v1 themselves.
        self.endpoint, self.model, self.mode = _host_only(endpoint), model, mode
        self.renderer, self.token_counter = renderer, token_counter
        self.max_retries = max_retries
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        # The pool must hold ``max_concurrency`` connections: httpx allows 100, which held the 21-replica
        # baseline eval of campaign 3532660 at 100 requests in flight, ~38 samples per GPU (2026-10-27).
        limits = httpx.Limits(max_connections=max_concurrency, max_keepalive_connections=max_concurrency)
        self.client = httpx.AsyncClient(timeout=timeout_s, headers=headers, transport=transport, limits=limits)

    async def aclose(self) -> None:
        await self.client.aclose()

    def _payload(
        self, messages: Messages, spec: SamplingSpec, n: int, seed: int
    ) -> tuple[str, dict[str, Any]]:
        common = {
            "model": self.model,
            "n": n,
            "temperature": spec.temperature,
            "top_p": spec.top_p,
            "max_tokens": spec.max_tokens,
            "seed": seed,
        }
        if self.mode == "completions":
            assert self.renderer is not None
            return "/v1/completions", {
                **common,
                "prompt": self.renderer(messages, spec.effort),
                "skip_special_tokens": True,
            }
        return "/v1/chat/completions", {
            **common,
            "messages": messages,
            "chat_template_kwargs": {"reasoning_effort": spec.effort, "enable_thinking": True},
        }

    def _parse(self, data: dict[str, Any], n: int) -> list[Sample]:
        usage_tokens = int((data.get("usage") or {}).get("completion_tokens", 0))
        out = []
        for ch in data["choices"]:
            if self.mode == "completions":
                text = ch.get("text") or ""
            else:
                msg = ch.get("message") or {}
                content, reasoning = msg.get("content") or "", msg.get("reasoning_content")
                if reasoning is not None:
                    # Server split the reasoning off; rebuild the marker only for cleanly finished samples.
                    finished = ch.get("finish_reason") == "stop" and content.strip()
                    text = reasoning + ("\n</think>\n\n" + content if finished else content)
                else:
                    text = content
            n_tok = self.token_counter(text) if self.token_counter else usage_tokens // max(1, n)
            out.append(
                Sample(text=text, finish_reason=str(ch.get("finish_reason") or "unknown"), n_tokens=n_tok)
            )
        return out

    async def generate(self, messages: Messages, spec: SamplingSpec, n: int, seed: int) -> list[Sample]:
        path, body = self._payload(messages, spec, n, seed)
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            try:
                r = await self.client.post(self.endpoint + path, json=body)
                if r.status_code >= 500:
                    raise httpx.HTTPStatusError("server error", request=r.request, response=r)
                r.raise_for_status()
                return self._parse(r.json(), n)
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                if attempt == self.max_retries or (
                    isinstance(e, httpx.HTTPStatusError) and e.response.status_code < 500
                ):
                    raise
                log.warning("request failed (%s); retry %d in %.0fs", e, attempt + 1, delay)
                await asyncio.sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")


Generator = Callable[[Messages, SamplingSpec, int, int], Awaitable[list[Sample]]]


# --------------------------------------------------------------------------- cache
class GenerationCache:
    """Append-only JSONL cache: one line per item, valid only for an identical sampling spec."""

    def __init__(self, path: str | os.PathLike, spec_key: str):
        self.path, self.spec_key = Path(path), spec_key
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data: dict[str, list[Sample]] = {}
        if self.path.exists():
            for rec in read_jsonl(self.path):
                if rec.get("spec_key") == spec_key:
                    self.data[rec["item_id"]] = [Sample(**s) for s in rec["samples"]]
        self._lock = asyncio.Lock()

    async def put(self, item_id: str, samples: list[Sample]) -> None:
        async with self._lock:
            self.data[item_id] = samples
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "item_id": item_id,
                            "spec_key": self.spec_key,
                            "samples": [asdict(s) for s in samples],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


async def generate_set(
    generate: Generator,
    items: Sequence[EvalItem],
    spec: SamplingSpec,
    cache: GenerationCache,
    *,
    max_concurrency: int = 64,
) -> dict[str, list[Sample]]:
    """Generate ``spec.n`` samples per item, skipping items already in the cache.

    At most ``max_concurrency`` requests and 16 × ``max_concurrency`` samples are in flight: all 300 pass@64
    requests at once would queue ~900 samples on each of 21 replicas, and the last would wait past the
    request timeout (7,200 s).
    """
    sem = asyncio.Semaphore(max(1, min(max_concurrency, 16 * max_concurrency // max(1, spec.n))))
    todo = [it for it in items if len(cache.data.get(it.id, [])) < spec.n]
    log.info("generating %d/%d items (n=%d, max_tokens=%d)", len(todo), len(items), spec.n, spec.max_tokens)

    async def one(it: EvalItem) -> None:
        async with sem:
            seed = derive_seed(spec.seed, it.set, it.id) % (2**31 - 1)
            samples = await generate(build_messages(it), spec, spec.n, seed)
            await cache.put(it.id, samples)

    await asyncio.gather(*(one(it) for it in todo))
    return {it.id: cache.data[it.id][: spec.n] for it in items}


# --------------------------------------------------------------------------- grading
def grade_sample(
    sample: Sample, item: EvalItem, grader: Callable[[str, dict[str, Any]], GradeResult]
) -> GradeResult:
    """Strip thinking, fail closed on a missing ``</think>`` or empty answer, else grade the final answer."""
    split = split_thinking(sample.text)
    if not split.has_end:
        return GradeResult.fail("no_think_end", finish_reason=sample.finish_reason)
    if not split.answer:
        return GradeResult.fail("empty_answer")
    try:
        return grader(split.answer, item.answer)
    except Exception as e:  # graders should not raise; if one does, the sample fails closed
        return GradeResult.fail("grader_exception", error=repr(e))


def score_set(
    spec: SetSpec,
    items: Sequence[EvalItem],
    gens: dict[str, list[Sample]],
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Grade every sample and summarize a set into the results.json schema."""
    grader = get_grader(spec.grader, **spec.grader_kwargs)
    correct = np.zeros((len(items), spec.samples), dtype=bool)
    reasons: Counter[str] = Counter()
    finish, tokens = [], []
    per_item = []
    for i, it in enumerate(items):
        samples = gens[it.id]
        if len(samples) < spec.samples:
            raise ValueError(f"{spec.name}/{it.id}: {len(samples)} samples < {spec.samples}")
        for j, smp in enumerate(samples[: spec.samples]):
            g = grade_sample(smp, it, grader)
            correct[i, j] = g.correct
            reasons[g.reason] += 1
            finish.append(smp.finish_reason)
            tokens.append(smp.n_tokens)
        per_item.append({"id": it.id, "n_correct": int(correct[i].sum()), "mean": float(correct[i].mean())})

    scores = correct.astype(float)
    out: dict[str, Any] = {
        "set": spec.name,
        "grader": spec.grader,
        "metric": spec.metric if spec.metric != "avg@k" else f"avg@{spec.samples}",
        "n_items": len(items),
        "n_samples": spec.samples,
        "truncation_rate": M.truncation_rate(finish),
        "mean_output_tokens": M.mean_output_tokens(tokens),
        "reasons": dict(reasons),
        "relative_only": spec.relative_only,
        "per_item": per_item,
    }
    out["score"] = M.avg_at_k(scores)
    out["item_ci"] = S.bootstrap_ci(M.per_item_means(scores), n_boot=n_boot, seed=seed).to_dict()
    if spec.samples >= 2:
        out["ci"] = S.sampling_ci(scores, n_boot=n_boot, seed=seed).to_dict()
        out["pass_at_k"] = {str(k): v for k, v in M.pass_at_k_curve(correct).items()}
    else:
        out["ci"] = out["item_ci"]
    return out


def score_refusal(spec: SetSpec, items: Sequence[EvalItem], gens: dict[str, list[Sample]]) -> dict[str, Any]:
    """Refusal rate (%) on items that should be refused, plus over-refusal on benign controls."""
    from dpt.eval.graders.refusal import is_refusal

    def answer(s: Sample) -> str | None:
        sp = split_thinking(s.text)
        return sp.answer if sp.has_end else None

    should, benign = [], []
    for it in items:
        for s in gens[it.id][: spec.samples]:
            (should if it.answer.get("should_refuse", True) else benign).append(is_refusal(answer(s)))
    rate = 100.0 * sum(should) / max(1, len(should))
    return {
        "set": spec.name,
        "metric": "refusal_rate",
        "refusal_rate": rate,
        "score": rate,
        "ci": S.wilson_ci(sum(should), max(1, len(should))).to_dict(),
        "over_refusal_rate": 100.0 * sum(benign) / max(1, len(benign)),
        "n_items": len(items),
    }


GATE_JUDGES = ("primary", "second", "min")


def _judge_pairs(
    judge: Any,
    items: Sequence[EvalItem],
    cand_ans: dict[str, str],
    base_ans: dict[str, str],
    max_concurrency: int,
) -> list[Any]:
    """Both-order verdicts for every item, ``max_concurrency`` items in flight (judges do HTTP I/O)."""
    from concurrent.futures import ThreadPoolExecutor

    from dpt.eval.graders.pairwise import PairResult, judge_both_orders

    def one(it: EvalItem) -> Any:
        if not cand_ans[it.id]:
            return PairResult(it.id, "loss", "B", "A")  # no final answer: fail closed
        return judge_both_orders(
            judge, it.id, it.question, cand_ans[it.id], base_ans[it.id], it.answer.get("rubric")
        )

    with ThreadPoolExecutor(max(1, int(max_concurrency))) as ex:
        return list(ex.map(one, items))


def _judge_summary(
    judge: Any,
    results: Sequence[Any],
    items: Sequence[EvalItem],
    cand_len: dict[str, int],
    base_len: dict[str, int],
    *,
    seed: int,
    max_unparsed_rate: float,
) -> dict[str, Any]:
    from dpt.eval.graders.pairwise import length_matched

    outcomes = {r.item_id: r.outcome for r in results}
    credits = M.win_rate_items(outcomes[it.id] for it in items)
    lm = length_matched(results, cand_len, base_len)
    out: dict[str, Any] = {
        "status": "ok",
        "score": M.win_rate(outcomes.values()),
        "ci": S.bootstrap_ci(credits, seed=seed).to_dict(),
        "outcomes": dict(Counter(outcomes.values())),
        "length_matched": {"n": len(lm), "score": M.win_rate(r.outcome for r in lm) if lm else None},
        "_per_item": outcomes,
    }
    stats = getattr(judge, "stats", None)
    if stats is not None:
        calls = int(stats.get("calls", 0))
        rate = stats.get("unparsed", 0) / max(1, calls)
        out["judge_calls"] = {"calls": calls, "cached": int(stats.get("cached", 0)),
                              "unparsed": int(stats.get("unparsed", 0)), "unparsed_rate": rate}
        if getattr(judge, "served_models", None):
            out["served_models"] = sorted(judge.served_models)
        if rate > max_unparsed_rate:
            # Unparsed verdicts are ties and pull the win rate toward 50%: do not score on them.
            out.update(status="judge_unparsed", score=None,
                       note=f"{rate:.1%} of verdicts unparsed (limit {max_unparsed_rate:.0%}): "
                            "raise the judge's max_tokens or check its output")
    return out


def _per_item_row(item_id: str, outcome: str, per_item_by: dict[str, dict[str, str]]) -> dict[str, Any]:
    row: dict[str, Any] = {"id": item_id, "outcome": outcome}
    if len(per_item_by) > 1:
        row["by_judge"] = {role: oc[item_id] for role, oc in per_item_by.items()}
    return row


def score_pairwise(
    spec: SetSpec,
    items: Sequence[EvalItem],
    cand: dict[str, list[Sample]],
    base: dict[str, list[Sample]] | None,
    judge: Any,
    *,
    second_judge: Any = None,
    gate_judge: str = "primary",
    max_unparsed_rate: float = 0.05,
    max_concurrency: int = 8,
    expert_fraction: float = 0.3,
    blind_review: dict[str, Any] | None = None,
    out_dir: Path | None = None,
    seed: int = 0,
    self_reference: bool = False,
) -> dict[str, Any]:
    """D-Report: both-order pairwise judging against base outputs, length-matched rate, expert export.

    ``judge`` is the primary judge (on the cluster also the T3 reward), ``second_judge`` an optional judge of
    another family that never touched training. ``gate_judge`` picks the number in ``score`` (what the
    gates read): ``primary``, ``second``, or ``min`` = the lower of the two, so the gate needs both
    judges to agree that the candidate is better. A judge whose verdicts are not cached and cannot be
    obtained, or whose unparsed rate exceeds ``max_unparsed_rate``, leaves ``score`` empty: the gate then
    fails closed instead of reading a number that was not measured.
    """
    from dpt.eval.graders.pairwise import JudgeUnavailable, export_blind_review

    if gate_judge not in GATE_JUDGES:
        raise ValueError(f"gate_judge must be one of {GATE_JUDGES}, got {gate_judge!r}")

    def final(s: Sample) -> str:
        sp = split_thinking(s.text)
        return sp.answer if sp.has_end else ""

    cand_ans = {it.id: final(cand[it.id][0]) for it in items}
    cand_len = {it.id: cand[it.id][0].n_tokens for it in items}
    common = {
        "set": spec.name,
        "n_items": len(items),
        "mean_output_tokens": M.mean_output_tokens(cand_len.values()),
        "truncation_rate": M.truncation_rate(cand[it.id][0].finish_reason for it in items),
    }
    if self_reference:
        outcomes = {it.id: "tie" for it in items}  # base vs base is 50% by construction
        credits = M.win_rate_items(outcomes[it.id] for it in items)
        return {
            **common,
            "metric": "winrate_vs_base",
            "score": M.win_rate(outcomes.values()),
            "ci": S.bootstrap_ci(credits, seed=seed).to_dict(),
            "outcomes": dict(Counter(outcomes.values())),
            "per_item": [{"id": k, "outcome": v} for k, v in outcomes.items()],
        }
    if base is None or judge is None:
        return {
            "set": spec.name,
            "metric": "winrate",
            "status": "pending",
            "note": "needs --reference-run (base generations) and a configured judge",
        }
    base_ans = {it.id: final(base[it.id][0]) for it in items}
    base_len = {it.id: base[it.id][0].n_tokens for it in items}

    judges = {"primary": judge, **({"second": second_judge} if second_judge is not None else {})}
    by_judge: dict[str, dict[str, Any]] = {}
    for role, j in judges.items():
        try:
            res = _judge_pairs(j, items, cand_ans, base_ans, max_concurrency)
        except (JudgeUnavailable, httpx.TransportError, httpx.HTTPStatusError) as e:
            # Verdicts obtained so far are cached; rerun with --regrade-only to finish.
            by_judge[role] = {"status": "pending", "score": None, "note": f"{type(e).__name__}: {e}"}
            continue
        by_judge[role] = _judge_summary(
            j, res, items, cand_len, base_len, seed=seed, max_unparsed_rate=max_unparsed_rate
        )
    if out_dir is not None:
        br = blind_review or {}
        review, key = export_blind_review(
            [asdict(it) for it in items], cand_ans, base_ans, fraction=expert_fraction, seed=seed,
            n_first=br.get("n_first"), n_extra=int(br.get("n_extra") or 0),
        )
        from dpt.common.io import write_jsonl

        write_jsonl(out_dir / "expert_review.jsonl", review)
        write_jsonl(out_dir / "expert_review.KEY.jsonl", key)

    per_item_by = {r: b.pop("_per_item") for r, b in by_judge.items() if "_per_item" in b}
    out: dict[str, Any] = {**common, "metric": "winrate_vs_base", "gate_judge": gate_judge,
                           "by_judge": by_judge}
    if len(per_item_by) == 2:
        a, b = per_item_by["primary"], per_item_by["second"]
        out["judge_agreement"] = sum(a[i] == b[i] for i in a) / max(1, len(a))

    needed = ["primary", "second"] if gate_judge == "min" else [gate_judge]
    missing = [r for r in needed if (by_judge.get(r) or {}).get("score") is None]
    if missing:
        reasons = "; ".join(f"{r}: {(by_judge.get(r) or {}).get('note', 'not configured')}" for r in missing)
        out.update(status="pending", score=None, note=f"gate_judge={gate_judge} needs {missing} ({reasons})")
        return out
    chosen = min(needed, key=lambda r: by_judge[r]["score"])
    g = by_judge[chosen]
    out.update(
        score=g["score"],
        ci=g["ci"],
        score_judge=chosen,
        outcomes=g["outcomes"],
        length_matched=g["length_matched"],
        per_item=[_per_item_row(k, v, per_item_by) for k, v in per_item_by[chosen].items()],
    )
    return out


# --------------------------------------------------------------------------- suite runner
def _set_specs(cfg: dict[str, Any], names: Sequence[str]) -> list[SetSpec]:
    defaults = {"sandbox_endpoint": cfg.get("sandbox_endpoint")}
    specs = []
    for name in names:
        if name in cfg.get("sets", {}):
            specs.append(SetSpec.from_cfg(name, cfg["sets"][name], defaults))
        elif name == "general":
            specs += [SetSpec.from_cfg(b, c, defaults) for b, c in cfg.get("general", {}).items()]
        elif name in ("refusal", "perturbation") and cfg.get(name):
            specs.append(SetSpec.from_cfg(name, cfg[name], defaults))
        elif name in cfg.get("general", {}):
            specs.append(SetSpec.from_cfg(name, cfg["general"][name], defaults))
        else:
            raise KeyError(f"unknown eval set {name!r}")
    return specs


async def run_suite(
    cfg: dict[str, Any],
    set_names: Sequence[str],
    *,
    generate: Generator | None,
    model_name: str,
    out_dir: Path,
    spec: SamplingSpec,
    mode: str,
    judge: Any = None,
    second_judge: Any = None,
    reference_run: Path | None = None,
    limit: int | None = None,
    regrade_only: bool = False,
    verify_manifests: bool = True,
) -> dict[str, Any]:
    """Generate (unless ``regrade_only``), grade and summarize every requested set at one sampling spec.

    D-Report verdicts are cached per judge in ``<out>/d_report/judgments.{primary,second}.jsonl``. A judge
    given as None replays its cache, so ``--regrade-only`` on a machine that reaches only one of the two
    judges still produces both win rates.
    """
    boot = cfg.get("bootstrap", {})
    n_boot, boot_seed = int(boot.get("n_boot", 2000)), int(boot.get("seed", 0))
    results: dict[str, Any] = {"sets": {}, "general": {}}
    data_hashes: dict[str, str] = {}
    for ss in _set_specs(cfg, set_names):
        items = load_eval_items(ss.path, ss.name, verify=verify_manifests, limit=limit)
        try:
            from dpt.common.manifest import load_manifest

            data_hashes[ss.name] = load_manifest(ss.path).content_hash
        except FileNotFoundError:
            data_hashes[ss.name] = "unmanifested"
        set_spec = SamplingSpec(**{**asdict(spec), "n": ss.samples})
        cache = GenerationCache(out_dir / ss.name / "generations.jsonl", set_spec.cache_key(model_name, mode))
        if not regrade_only:
            assert generate is not None
            await generate_set(
                generate, items, set_spec, cache, max_concurrency=int(cfg.get("max_concurrency", 64))
            )
        missing = [it.id for it in items if len(cache.data.get(it.id, [])) < ss.samples]
        if missing:
            raise RuntimeError(
                f"{ss.name}: {len(missing)} items have no cached generations (e.g. {missing[:3]})"
            )
        gens = {it.id: cache.data[it.id] for it in items}

        if ss.grader == "pairwise":
            base_gens = None
            self_ref = reference_run is not None and reference_run.resolve() == out_dir.resolve()
            if reference_run is not None and not self_ref:
                base_cache_path = reference_run / ss.name / "generations.jsonl"
                base_gens = {
                    r["item_id"]: [Sample(**s) for s in r["samples"]] for r in read_jsonl(base_cache_path)
                }
            from dpt.eval.graders.pairwise import CachedJudge

            def _cached(j: Any, role: str) -> Any:
                path = out_dir / ss.name / f"judgments.{role}.jsonl"
                if j is None and not path.exists():
                    return None
                return CachedJudge(j, path, model=getattr(j, "model", None))

            pw = cfg.get("pairwise") or {}
            res = score_pairwise(
                ss,
                items,
                gens,
                base_gens,
                _cached(judge, "primary"),
                second_judge=_cached(second_judge, "second"),
                gate_judge=str(pw.get("gate_judge", "primary")),
                max_unparsed_rate=float(pw.get("max_unparsed_rate", 0.05)),
                max_concurrency=int(pw.get("max_concurrency", 8)),
                out_dir=out_dir / ss.name,
                seed=boot_seed,
                expert_fraction=float(cfg.get("judge", {}).get("expert_blind_fraction", 0.3)),
                blind_review=pw.get("blind_review"),
                self_reference=self_ref,
            )
        elif ss.grader == "refusal":
            res = score_refusal(ss, items, gens)
        else:
            res = score_set(ss, items, gens, n_boot=n_boot, seed=boot_seed)
        bucket = "general" if ss.name in cfg.get("general", {}) else "sets"
        results[bucket][ss.name] = res
        log.info("%s: score=%.2f", ss.name, res.get("score") or float("nan"))
    results["data_hashes"] = data_hashes
    return results


def merge_max_tokens(
    primary: dict[str, Any], extra: dict[str, Any], primary_len: int, extra_len: int
) -> dict[str, Any]:
    """Base@xhigh rule: per set, keep the higher score of the 32K and 64K runs; record both."""
    merged = json.loads(json.dumps(primary))
    for bucket in ("sets", "general"):
        for name, res in primary.get(bucket, {}).items():
            other = extra.get(bucket, {}).get(name)
            if other is None or res.get("score") is None or other.get("score") is None:
                continue
            best = other if other["score"] > res["score"] else res
            merged[bucket][name] = json.loads(json.dumps(best))
            merged[bucket][name]["score_by_max_tokens"] = {
                str(primary_len): res["score"],
                str(extra_len): other["score"],
            }
            merged[bucket][name]["selected_max_tokens"] = extra_len if best is other else primary_len
    return merged


def overall_mean_output_tokens(
    results: dict[str, Any], domain_sets: Sequence[str] = ("d_num", "d_code", "d_report", "d_fresh")
) -> float | None:
    """Sample-weighted mean output tokens over the domain sets (the report's "平均输出 token")."""
    tot, cnt = 0.0, 0
    for s in domain_sets:
        r = results.get("sets", {}).get(s)
        if r and r.get("mean_output_tokens") is not None:
            n = r.get("n_items", 0) * r.get("n_samples", 1)
            tot += r["mean_output_tokens"] * n
            cnt += n
    return tot / cnt if cnt else None


def build_pairwise_judge(jcfg: dict[str, Any] | None) -> Any:
    """HttpPairwiseJudge from an eval-suite judge block; None when unset or unusable from this machine."""
    from dpt.eval.graders.pairwise import HttpPairwiseJudge
    from dpt.eval.hosted import endpoint_from_config

    return endpoint_from_config(jcfg, HttpPairwiseJudge)


# --------------------------------------------------------------------------- CLI
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="eval/eval_suite.yaml")
    ap.add_argument(
        "--model-name", required=True, help="row name for reports, e.g. base_medium, sft, rl_seed1"
    )
    ap.add_argument(
        "--served-model", default=None, help="model id the server expects (default: --model-name)"
    )
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--sets", default="d_num,d_code,d_report,d_fresh")
    ap.add_argument("--effort", default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--extra-max-tokens", type=int, default=None, help="base@xhigh: also run at this length")
    ap.add_argument("--samples-override", type=int, default=None, help="e.g. 64 for pass@k curves")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["completions", "chat"], default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--reference-run", default=None, help="base run dir for D-Report pairwise judging")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="first N items per set (smoke tests)")
    ap.add_argument("--regrade-only", action="store_true")
    ap.add_argument(
        "--update",
        action="store_true",
        help="replace only the requested sets in an existing results.json (e.g. a D-Report pass with the "
        "second judge on a machine that reaches its API), keeping every other set",
    )
    ap.add_argument("--no-verify", action="store_true", help="skip manifest verification (smoke tests only)")
    ap.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. eval_suite.max_concurrency=32")
    args = ap.parse_args(argv)

    from dpt.common.config import config_hash, load_config, to_container

    cfg_all = load_config(args.config, overrides=args.overrides)
    cfg = to_container(cfg_all)["eval_suite"]
    if args.samples_override:
        for bucket in ("sets", "general"):
            for c in cfg.get(bucket, {}).values():
                c["samples"] = args.samples_override
    sampling = cfg["sampling"]
    spec = SamplingSpec(
        effort=args.effort or sampling["effort"],
        max_tokens=args.max_tokens or sampling["max_tokens"],
        temperature=float(sampling["temperature"]),
        top_p=float(sampling["top_p"]),
        seed=args.seed,
    )
    mode = args.mode or cfg.get("mode", "completions")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    judge = build_pairwise_judge(cfg.get("judge"))
    second_judge = build_pairwise_judge(cfg.get("second_judge"))

    async def _run(max_tokens: int, sub: Path) -> dict[str, Any]:
        backend = None
        gen = None
        if not args.regrade_only:
            tok_name = args.tokenizer or cfg.get("tokenizer")
            renderer = hf_renderer(tok_name) if mode == "completions" else None
            backend = OpenAICompatBackend(
                args.endpoint or cfg["endpoint"],
                args.served_model or args.model_name,
                mode=mode,
                renderer=renderer,
                token_counter=hf_token_counter(tok_name),
                timeout_s=float(cfg.get("request_timeout_s", 7200)),
                max_concurrency=int(cfg.get("max_concurrency", 64)),
            )
            gen = backend.generate
        try:
            s = SamplingSpec(**{**asdict(spec), "max_tokens": max_tokens})
            return await run_suite(
                cfg,
                [x for x in args.sets.split(",") if x],
                generate=gen,
                model_name=args.served_model or args.model_name,
                out_dir=sub,
                spec=s,
                mode=mode,
                judge=judge,
                second_judge=second_judge,
                reference_run=Path(args.reference_run) if args.reference_run else None,
                limit=args.limit,
                regrade_only=args.regrade_only,
                verify_manifests=not args.no_verify,
            )
        finally:
            if backend is not None:
                await backend.aclose()

    results = asyncio.run(_run(spec.max_tokens, out_dir))
    if args.extra_max_tokens:
        extra = asyncio.run(_run(args.extra_max_tokens, out_dir / f"max{args.extra_max_tokens}"))
        results = merge_max_tokens(results, extra, spec.max_tokens, args.extra_max_tokens)
    results.update(
        {
            "model_name": args.model_name,
            "effort": spec.effort,
            "max_tokens": spec.max_tokens,
            "mode": mode,
            "seed": args.seed,
            "config_hash": config_hash(cfg_all),
            "sampling": asdict(spec),
            "mean_output_tokens": overall_mean_output_tokens(results),
        }
    )
    if args.update and (out_dir / "results.json").exists():
        results = merge_into_existing(read_json(out_dir / "results.json"), results)
    write_json(out_dir / "results.json", results)
    log.info("wrote %s", out_dir / "results.json")
    return 0


def merge_into_existing(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """``old`` with the sets graded in ``new`` replaced; the update is recorded under ``updates``."""
    merged = json.loads(json.dumps(old))
    changed = []
    for bucket in ("sets", "general"):
        for name, res in new.get(bucket, {}).items():
            merged.setdefault(bucket, {})[name] = res
            changed.append(name)
    merged.setdefault("data_hashes", {}).update(
        {k: v for k, v in new.get("data_hashes", {}).items() if k in changed}
    )
    merged["mean_output_tokens"] = overall_mean_output_tokens(merged)
    merged.setdefault("updates", []).append({"sets": changed, "config_hash": new.get("config_hash")})
    return merged


if __name__ == "__main__":
    raise SystemExit(main())
