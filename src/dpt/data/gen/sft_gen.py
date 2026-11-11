"""SFT data: teacher rejection sampling (domain_reasoning, tool_traj, report) and base self-distillation.

    python -m dpt.data.gen.sft_gen domain  --problems BANK --out $SFT/domain_reasoning.jsonl  <teacher args>
    python -m dpt.data.gen.sft_gen tool    --problems BANK --out $SFT/tool_traj.jsonl --hidden-tests-dir DIR  <teacher>
    python -m dpt.data.gen.sft_gen report  --problems BANK --out $SFT/report.jsonl --judge-url URL --judge-model judge
    python -m dpt.data.gen.sft_gen distill --problems BANK --out $SFT/self_distill.jsonl  <base model args>
    python -m dpt.data.gen.sft_gen rewards-config --out FILE rewards.t2_code.sandbox.hidden_tests_dir=DIR ...

``BANK`` is a JSONL of problems: the pipeline's verify-stage output (decontaminated, labels verified).
Problems from held-out families (``meta.split == "heldout"``) are always skipped, and ``--exclude`` takes
files whose uids must not be used (the RL dev split).

Per problem the model is sampled ``--k`` times with the RL/eval prompt (``dpt.data.prompts``) at
``--effort`` (medium), and each reply is graded with the **training** grader (``dpt.rewards.router``, the
config in ``--rewards-config`` or ``$DPT_REWARDS_CONFIG``, plus dotlist overrides). A reply cut off at
``max_tokens`` or without an end of thinking is never kept. Up to ``--keep`` passing replies become
:class:`dpt.data.schema.SFTSample` records, thinking in ``reasoning_content``, with the meta that
``dpt.data.sft_build`` admits:

=================  ===============================  ===========================================
subset             passes when                      meta
=================  ===============================  ===========================================
domain_reasoning   the grader marks it correct      verified, pass_at_k
tool_traj          the hidden tests all pass        exec_ok, hidden_tests_passed (single-turn code)
report             judge score ≥ ``--min-score``    rubric, judge_score
self_distill       well-formed (math: correct)      generated_by = the base model's HF id
=================  ===============================  ===========================================

Every subset writes ``<out>.report.json``: per task and per family, the histogram of passing replies
out of k, the share of problems at 0/k (for domain_reasoning: how much of the bank the teacher cannot
solve, the number that decides whether a stronger teacher is worth it for that slice), truncation, and
token counts (exact with the model's tokenizer).

Resumable: one progress line per finished problem in ``<out>.progress.jsonl``; samples are appended to
``--out`` as they pass. A rerun skips finished problems. A problem is not finished when the judge or the
sandbox could not grade its replies (:data:`INFRA_REASONS`), and after 8 of those the run stops with exit
code 1, so a judge window that ends mid-run costs no samples. Replies are kept in ``<out>.replies.jsonl``
until graded: ``--generate-only`` samples while no judge is up, and the next run only grades.
``--stop-at`` ends sampling at a given time, keeping what is done (a server job that is about to end).
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from dpt.common.hashing import short_hash
from dpt.common.io import read_jsonl, write_json
from dpt.common.logging import get_logger
from dpt.common.seeding import derive_seed
from dpt.common.types import EFFORTS, T1_NUM, T2_CODE, T3_REPORT, T4_GENERAL
from dpt.data.gen.teacher import Appender, ChatModel, Reply, add_model_args, model_from_args, read_existing
from dpt.data.prompts import messages_for_problem
from dpt.data.schema import Problem, SFTSample

log = get_logger(__name__)


@dataclass(frozen=True)
class SubsetSpec:
    subset: str
    tasks: tuple[str, ...]
    k: int
    keep: int


SPECS = {
    "domain": SubsetSpec("domain_reasoning", (T1_NUM,), k=4, keep=2),
    "tool": SubsetSpec("tool_traj", (T2_CODE,), k=4, keep=1),
    "report": SubsetSpec("report", (T3_REPORT,), k=2, keep=1),
    "distill": SubsetSpec("self_distill", (T4_GENERAL,), k=1, keep=1),
}


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reason: str
    score: float = 0.0
    meta: dict[str, Any] | None = None


GradeFn = Callable[[Problem, Reply], Any]            # → RewardResult-like (score, reason, correct)
AcceptFn = Callable[[Problem, Reply], Verdict]


# --------------------------------------------------------------------------- grading


def build_router(config: str | None = None, overrides: Sequence[str] = ()) -> Any:
    """The training reward router from ``config`` (default ``$DPT_REWARDS_CONFIG`` or rl/rewards.yaml)."""
    from dpt.common.config import load_config, to_container
    from dpt.rewards.router import CONFIG_ENV, RewardRouter

    path = config or os.environ.get(CONFIG_ENV) or "rl/rewards.yaml"
    return RewardRouter(to_container(load_config(path, overrides=list(overrides)))["rewards"])


def router_grade(router: Any) -> GradeFn:
    def grade(p: Problem, reply: Reply) -> Any:
        return router.grade(p.task, reply.text, json.dumps(p.answer, sort_keys=True),
                            {"uid": p.uid, "task": p.task, "question": p.question,
                             "finish_reason": reply.finish_reason})
    return grade


def _well_formed(reply: Reply) -> str | None:
    if reply.truncated:
        return "truncated"
    s = reply.split
    if not s.has_end:
        return "no_think_end"
    if not s.answer.strip() or not s.thinking.strip():
        return "empty"
    return None


def accept_fn(name: str, grade: GradeFn, *, min_score: float = 0.7) -> AcceptFn:
    """The pass rule of one subset (table in the module docstring)."""

    def accept(p: Problem, reply: Reply) -> Verdict:
        bad = _well_formed(reply)
        if bad:
            return Verdict(False, bad)
        if name == "distill" and not (p.task == T4_GENERAL and p.answer.get("kind") == "math"):
            return Verdict(True, "well_formed")
        res = grade(p, reply)
        score = float(getattr(res, "score", 0.0))
        reason = str(getattr(res, "reason", ""))
        if name == "report":
            ok = reason == "ok" and score >= min_score
            why = "ok" if ok else (reason if reason != "ok" else "below_min_score")
            return Verdict(ok, why, score, {"judge_score": round(score, 4)})
        ok = reason == "ok" and getattr(res, "correct", None) is True
        failed = "hidden_tests_failed" if name == "tool" else "wrong"
        return Verdict(ok, "ok" if ok else (reason if reason != "ok" else failed), score)

    return accept


# --------------------------------------------------------------------------- samples


def sft_sample(spec: SubsetSpec, p: Problem, reply: Reply, i: int, *, model_id: str, effort: str,
               extra_meta: dict[str, Any]) -> dict[str, Any]:
    s = reply.split
    msgs = messages_for_problem(p) + [{"role": "assistant", "reasoning_content": s.thinking, "content": s.answer}]
    meta: dict[str, Any] = {"task": p.task, "family": p.meta.get("family"), **extra_meta}
    if spec.subset == "domain_reasoning":
        meta["verified"] = True
    elif spec.subset == "tool_traj":
        meta.update(exec_ok=True, hidden_tests_passed=True)
    elif spec.subset == "report":
        meta["rubric"] = p.answer.get("rubric")
    elif spec.subset == "self_distill":
        meta["generated_by"] = model_id
    rec = {"subset": spec.subset, "messages": msgs, "reasoning_effort": effort,
           "source": {"origin": "dpt.data.gen.sft_gen", "model": model_id, "problem_uid": p.uid, "sample": i},
           "meta": meta}
    return SFTSample.from_dict(rec).to_dict()


# --------------------------------------------------------------------------- problems


def load_problems(paths: Iterable[str | Path], tasks: Sequence[str], *, exclude: Iterable[str | Path] = (),
                  limit: int | None = None) -> tuple[list[Problem], Counter]:
    skipped: Counter = Counter()
    ex: set[str] = set()
    for e in exclude:
        for r in read_jsonl(e):
            uid = r.get("uid") or (r.get("extra_info") or {}).get("uid")
            if uid:
                ex.add(uid)
    seen, out = set(), []
    for path in paths:
        for r in read_jsonl(path):
            p = Problem.from_dict(r)
            if p.task not in tasks:
                continue
            if p.meta.get("split") == "heldout":       # held-out families are eval only
                skipped["heldout"] += 1
            elif p.uid in ex:
                skipped["excluded"] += 1
            elif p.uid in seen:
                skipped["duplicate"] += 1
            else:
                seen.add(p.uid)
                out.append(p)
    out.sort(key=lambda p: short_hash(p.uid))          # a deterministic, shuffled order; --limit takes a prefix
    if limit is not None:
        out = out[:limit]
    return out, skipped


# --------------------------------------------------------------------------- sampling


# Rejections that say a grading server was unreachable (the T3 judge, the T2 sandbox), not that the reply was
# bad. Such a problem gets no progress line, so a rerun samples it again; after ``max_infra_failures`` of them
# the run stops, because the server is down (a judge window ended) and every further problem would fail too.
INFRA_REASONS = frozenset({"judge_error", "judge_version_mismatch", "sandbox_error"})


def sample_problems(problems: Sequence[Problem], model: ChatModel, accept: AcceptFn, spec: SubsetSpec,
                    out: str | Path, *, seed: int = 1, concurrency: int = 32,
                    max_infra_failures: int = 8, generate_only: bool = False,
                    stop_at: float | None = None) -> dict[str, Any]:
    progress_path = Path(f"{out}.progress.jsonl")
    # Replies are kept until their problem is finished, so a rerun grades them instead of sampling again: the
    # report subset is sampled while no judge window is up (generate_only) and graded in the next one.
    replies_path = Path(f"{out}.replies.jsonl")
    done = {r["uid"] for r in read_existing(progress_path)}
    cached = {r["uid"]: [Reply(x["text"], x.get("finish_reason")) for x in r["replies"]]
              for r in read_existing(replies_path) if r.get("k") == spec.k and r.get("seed") == seed}
    todo = [p for p in problems if p.uid not in done and not (generate_only and p.uid in cached)]
    samples, progress, reply_log = Appender(out), Appender(progress_path), Appender(replies_path)
    effort = model.effort if model.effort in EFFORTS else "medium"
    failures = Counter()
    lock, stop = threading.Lock(), threading.Event()

    def one(p: Problem) -> None:
        if stop.is_set() or (stop_at is not None and time.time() >= stop_at):   # past --stop-at: none left
            return
        replies = cached.get(p.uid)
        if replies is None:
            try:
                replies = model.generate(messages_for_problem(p), n=spec.k,
                                         seed=derive_seed(seed, spec.subset, p.uid))
            except Exception as e:  # retried on the next run (no progress line)
                with lock:
                    failures["request"] += 1
                log.warning("%s %s: %s", spec.subset, p.uid, e)
                return
            reply_log.write([{"uid": p.uid, "k": spec.k, "seed": seed,
                              "replies": [{"text": r.text, "finish_reason": r.finish_reason} for r in replies]}])
        if generate_only:
            return
        verdicts = []
        for r in replies:
            try:
                verdicts.append(accept(p, r))
            except Exception as e:  # a grader error rejects the reply, never the run
                verdicts.append(Verdict(False, f"grader_error:{type(e).__name__}"))
        infra = sorted({v.reason for v in verdicts} & INFRA_REASONS)
        if infra:  # retried on the next run (no progress line)
            with lock:
                failures["infra"] += 1
                if failures["infra"] >= max_infra_failures and not stop.is_set():
                    stop.set()
                    log.error("%s: %d problems could not be graded (%s): stopping", spec.subset,
                              failures["infra"], ",".join(infra))
            log.warning("%s %s: not graded (%s), left for the next run", spec.subset, p.uid, ",".join(infra))
            return
        passing = [(i, r, v) for i, (r, v) in enumerate(zip(replies, verdicts)) if v.ok]
        kept = passing[:spec.keep]
        pass_k = len(passing) / max(1, len(replies))
        recs = [sft_sample(spec, p, r, i, model_id=model.model_id, effort=effort,
                           extra_meta={"pass_at_k": round(pass_k, 4), **(v.meta or {})}) for i, r, v in kept]
        samples.write(recs)
        progress.write([{
            "uid": p.uid, "task": p.task, "family": p.meta.get("family") or "", "k": len(replies),
            "passed": len(passing), "kept": len(kept), "truncated": sum(r.truncated for r in replies),
            "gen_tokens": sum(model.count_tokens(r.text) for r in replies),
            "kept_tokens": sum(model.count_tokens(r.text) for _, r, _ in kept),
            "reasons": dict(Counter(v.reason for v in verdicts)),
            "scores": [round(v.score, 4) for v in verdicts],
        }])

    with ThreadPoolExecutor(max(1, concurrency)) as ex:
        list(ex.map(one, todo))
    rows = read_existing(progress_path)
    report = pass_rate_report(rows, spec, exact_tokens=model.tokenizer is not None)
    report.update(model=model.model_id, effort=effort, attempted_this_run=len(todo),
                  failed_requests=failures["request"], ungraded=failures["infra"], stopped=stop.is_set(),
                  problems_in_input=len(problems), generate_only=generate_only,
                  past_stop_at=stop_at is not None and time.time() >= stop_at)
    write_json(f"{out}.report.json", report)
    log.info("%s: %d problems, %d samples, 0/k share %.3f, failed requests %d, ungraded %d%s", spec.subset,
             report["problems"], report["kept_samples"], report["zero_of_k"]["share"], failures["request"],
             failures["infra"], " (stopped)" if report["stopped"] else "")
    return report


def _hist(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    h = Counter(int(r["passed"]) for r in rows)
    n = len(rows)
    return {"problems": n, "histogram": {str(k): h[k] for k in sorted(h)},
            "zero_share": round(h[0] / n, 4) if n else None,
            "mean_pass_rate": round(sum(r["passed"] / max(1, r["k"]) for r in rows) / n, 4) if n else None}


def pass_rate_report(rows: Sequence[dict[str, Any]], spec: SubsetSpec, *, exact_tokens: bool) -> dict[str, Any]:
    by_task: dict[str, list] = defaultdict(list)
    by_family: dict[str, list] = defaultdict(list)
    reasons: Counter = Counter()
    for r in rows:
        by_task[r["task"]].append(r)
        by_family[r.get("family") or "(none)"].append(r)
        reasons.update(r.get("reasons") or {})
    n = len(rows)
    zero = sum(1 for r in rows if r["passed"] == 0)
    return {
        "subset": spec.subset, "k": spec.k, "keep": spec.keep, "problems": n,
        "kept_samples": sum(r["kept"] for r in rows),
        "zero_of_k": {"count": zero, "share": round(zero / n, 4) if n else 0.0},
        "all_of_k": sum(1 for r in rows if r["passed"] == r["k"]),
        "by_task": {t: _hist(v) for t, v in sorted(by_task.items())},
        "by_family": {f: _hist(v) for f, v in sorted(by_family.items())},
        "reply_outcomes": dict(sorted(reasons.items())),
        "truncated_replies": sum(r["truncated"] for r in rows),
        "tokens": {"generated": sum(r["gen_tokens"] for r in rows), "kept": sum(r["kept_tokens"] for r in rows),
                   "exact": exact_tokens},
    }


# --------------------------------------------------------------------------- rewards config


def write_rewards_config(out: str | Path, overrides: Sequence[str], *, base: str | Path | None = None) -> Path:
    """rl/rewards.yaml with ``overrides`` merged and its ``${project.*}`` interpolations kept.

    Point ``$DPT_REWARDS_CONFIG`` at the result so every grader in a process (the pipeline's difficulty
    stage included) sees, e.g., the hidden-test directory and the judge endpoint.
    """
    from omegaconf import OmegaConf

    from dpt.common.config import CONFIG_ROOT

    cfg = OmegaConf.load(base or CONFIG_ROOT / "rl" / "rewards.yaml")
    merged = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(merged, p)
    return p


# --------------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None, *, model: ChatModel | None = None,
         grade: GradeFn | None = None) -> int:
    ap = argparse.ArgumentParser(description="SFT data by rejection sampling and self-distillation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rc = sub.add_parser("rewards-config", help="write rl/rewards.yaml with overrides (for $DPT_REWARDS_CONFIG)")
    rc.add_argument("--out", required=True)
    rc.add_argument("overrides", nargs="*")
    for name, spec in SPECS.items():
        p = sub.add_parser(name, help=f"{spec.subset} from {'/'.join(spec.tasks)} problems")
        p.add_argument("--problems", action="append", required=True, help="JSONL of problems (repeatable)")
        p.add_argument("--out", required=True, help="SFTSample JSONL (appended; resumable)")
        p.add_argument("--exclude", action="append", default=[], help="JSONL whose uids must not be used")
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--k", type=int, default=spec.k, help="samples per problem")
        p.add_argument("--keep", type=int, default=spec.keep, help="passing samples kept per problem")
        p.add_argument("--seed", type=int, default=1)
        p.add_argument("--concurrency", type=int, default=32)
        p.add_argument("--rewards-config", default=None)
        if name == "tool":
            p.add_argument("--hidden-tests-dir", required=True)
            p.add_argument("--allow-no-container", action="store_true",
                           help="run model code without the Apptainer image (never on a shared account)")
        if name == "report":
            p.add_argument("--judge-url", default=None, help="judge server (default: rewards config)")
            p.add_argument("--judge-model", default=None, help="served judge name, pinned (default: config)")
            p.add_argument("--min-score", type=float, default=0.7)
        p.add_argument("--override", action="append", default=[], help="rewards dotlist override (repeatable)")
        p.add_argument("--stop-at", default=None,
                       help="ISO time: start no problem after it (in-flight ones finish), then end as usual")
        p.add_argument("--generate-only", action="store_true",
                       help="sample and keep the replies (<out>.replies.jsonl) without grading them; "
                            "a later run grades them")
        if model is None:
            add_model_args(p, max_tokens=20480)
    args = ap.parse_args(argv)

    if args.cmd == "rewards-config":
        log.info("rewards config → %s", write_rewards_config(args.out, args.overrides))
        return 0

    spec = SPECS[args.cmd]
    spec = SubsetSpec(spec.subset, spec.tasks, k=args.k, keep=args.keep)
    overrides = list(args.override)
    if args.cmd == "tool":
        overrides.append(f"rewards.t2_code.sandbox.hidden_tests_dir={args.hidden_tests_dir}")
    if args.cmd == "report":
        if args.judge_url:
            u = args.judge_url.rstrip("/")
            overrides.append(f"rewards.t3_report.judge.base_url={u if u.endswith('/v1') else u + '/v1'}")
        if args.judge_model:
            overrides.append(f"rewards.t3_report.judge.model={args.judge_model}")
    if grade is None:
        router = build_router(args.rewards_config, overrides)
        if args.cmd == "tool" and not router.cfg.get("t2_code", {}).get("sandbox", {}).get("container_image") \
                and not args.allow_no_container:
            raise SystemExit("model-written code runs only inside the sandbox container: set DPT_SANDBOX_SIF "
                             "(scripts/slurm/env.sh), or pass --allow-no-container on a machine of your own")
        grade = router_grade(router)
    problems, skipped = load_problems(args.problems, spec.tasks, exclude=args.exclude, limit=args.limit)
    log.info("%s: %d problems (skipped %s)", spec.subset, len(problems), dict(skipped))
    if not problems:
        log.error("%s: no %s problems in %s", spec.subset, "/".join(spec.tasks), args.problems)
        return 1
    m = model or model_from_args(args, concurrency=args.concurrency)
    accept = accept_fn(args.cmd, grade, min_score=getattr(args, "min_score", 0.7))
    stop_at = datetime.fromisoformat(args.stop_at).timestamp() if args.stop_at else None
    report = sample_problems(problems, m, accept, spec, args.out, seed=args.seed, concurrency=args.concurrency,
                             generate_only=args.generate_only, stop_at=stop_at)
    attempted = max(1, report["attempted_this_run"])
    return 1 if report["stopped"] or report["failed_requests"] > 0.05 * attempted else 0


if __name__ == "__main__":
    raise SystemExit(main())
