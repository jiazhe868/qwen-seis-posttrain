"""Judge drift during RL: does a judge that never touched training agree with the reward judge?

The T3 reward comes from the Gemma judge, so RL optimizes against it. Reward hacking shows up as the
reward judge liking the policy's reports more and more while an independent judge does not (in the
report, the step-245 citation hack raised the judge score while people's verdicts fell). With no expert
review during RL (README §3.1), GPT-6 Luna is that independent judge.

verl dumps every dev validation (every 20 steps) to ``<run>/val_generations/<step>.jsonl``
(``trainer.validation_data_dir``), with Gemma's rubric score of each response (``t3_rubric``). This tool
runs on the eval side, outside the training job. It has Luna score the same T3 responses with the same
rubric prompt (``dpt.rewards.judge.rubric_messages``) and the same preprocessing (thinking, self-praise
sentences and the reference list removed):

    python -m dpt.eval.judge_drift --run $SCRATCH/dpt/checkpoints/rl/rl9b_seed1 \\
        --dev $WORK/dpt/data/v1/rl_dev.jsonl          # scripts/slurm/judge_drift.sh wraps it

Only validation steps not scored yet are sent; scores are cached in ``<run>/judge_drift/``. For each step
it reports both judges' mean rubric score (0–100) on the same responses and the gap between them. It
alerts (exit status 3) when

* the gap (Gemma − Luna) has grown by more than ``max_gap_growth`` points since the first step, on two
  checks in a row (the paired comparison, the less noisy signal); or
* Gemma's mean rose while Luna's stayed flat (rose by at most ``gold_tolerance`` points) or fell for
  ``consecutive`` checks in a row (the ``proxy_gold`` rule of ``configs/rl/monitoring.yaml``).

An alert is for a person: treat it like a ``proxy_gold`` alert (docs/runbook.md). Nothing here feeds back
into training. OpenAI's terms forbid using Luna's output to develop a model, so the trainer never reads
these files and no training package imports this module (``tests/eval/test_independence.py``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from dpt.common.io import read_jsonl, write_json
from dpt.common.logging import get_logger
from dpt.eval.hosted import ChatEndpoint, endpoint_from_config

log = get_logger(__name__)

T3 = "t3_report"


@dataclass
class DriftConfig:
    n_prompts: int = 40              # T3 dev prompts scored per validation step (first sample of each)
    min_responses: int = 40          # fewer prompts (the 9B has 12): more samples of each, up to verl's val n
    max_gap_growth: float = 8.0      # points (0–100) the Gemma − Luna gap may grow over the first step
    consecutive: int = 3             # proxy up / gold flat streak that alerts (monitoring.yaml proxy_gold)
    gold_tolerance: float = 1.0      # Luna "flat" = rose by at most this many points (40 responses are noisy)
    max_unparsed_rate: float = 0.10  # a step with more unreadable Luna scores is not used for alerts


def _sha(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_dev_questions(path: str | Path) -> dict[str, str]:
    """ground_truth string → question, for T3 dev rows (a ground truth shared by two prompts is skipped)."""
    seen: Counter[str] = Counter()
    questions: dict[str, str] = {}
    for row in read_jsonl(path):
        extra = row.get("extra_info") or {}
        if (extra.get("task") or row.get("data_source")) != T3:
            continue
        gt = (row.get("reward_model") or {}).get("ground_truth")
        if isinstance(gt, str) and extra.get("question"):
            seen[gt] += 1
            questions[gt] = str(extra["question"])
    return {g: q for g, q in questions.items() if seen[g] == 1}


def judge_input(output: str) -> str | None:
    """What the T3 judge sees: the final answer without self-praise sentences and the reference list."""
    from dpt.chat.thinking import strip_thinking
    from dpt.rewards.citations import extract_citations
    from dpt.rewards.hack_patterns import strip_self_praise

    answer = strip_thinking(output)
    if answer is None:
        return None
    body, _ = extract_citations(strip_self_praise(answer)[0])
    return body if body.strip() else None


def select_rows(rows: Sequence[dict[str, Any]], questions: dict[str, str], n: int,
                min_responses: int = 0) -> list[dict[str, Any]]:
    """The first graded sample of up to ``n`` T3 dev prompts, in a fixed (hashed) prompt order. While that is
    fewer than ``min_responses``, every prompt adds its next sample, so each prompt keeps the same weight."""
    graded: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        gt = r.get("gts")
        if r.get("task") != T3 or gt not in questions or r.get("t3_rubric") is None:
            continue
        if (r.get("reason") or "ok") != "ok":
            continue
        answer = judge_input(str(r.get("output") or ""))
        if answer is not None:
            graded.setdefault(gt, []).append(
                {"gt": gt, "question": questions[gt], "answer": answer, "gemma": float(r["t3_rubric"])})
    prompts = sorted(graded, key=_sha)[:n]
    k, most = 1, max((len(graded[g]) for g in prompts), default=0)
    while k < most and sum(min(k, len(graded[g])) for g in prompts) < min_responses:
        k += 1
    return [s for g in prompts for s in graded[g][:k]]


class LunaScorer:
    """The T3 rubric judge's prompt on another endpoint, with a per-run cache of scores."""

    def __init__(self, chat: ChatEndpoint | Any, cache_path: Path):
        self.chat, self.cache_path = chat, cache_path
        self.model = getattr(chat, "model", "?")
        self.cache: dict[str, dict[str, Any]] = {}
        if cache_path.exists():
            for r in read_jsonl(cache_path):
                self.cache[r["key"]] = r
        self.served: set[str] = set()

    def score(self, question: str, answer: str, rubric: Sequence[Any]) -> float | None:
        from dpt.rewards.judge import JudgeParseError, RubricJudge, rubric_messages, weighted_score

        key = _sha(self.model, question, answer, rubric)
        if key in self.cache:
            return self.cache[key]["score"]
        reply = self.chat.request(rubric_messages(question, answer, rubric))
        score = None
        if reply.finish_reason != "length":
            try:
                score = weighted_score(RubricJudge.parse(reply.content, len(rubric)), rubric)
            except JudgeParseError:
                score = None
        rec = {"key": key, "score": score, "model": self.model, "served_model": reply.model}
        self.cache[key] = rec
        if reply.model:
            self.served.add(reply.model)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return score


def score_step(rows: Sequence[dict[str, Any]], scorer: LunaScorer, *, max_concurrency: int = 8) -> dict[str, Any]:
    def one(r: dict[str, Any]) -> float | None:
        rubric = (json.loads(r["gt"]) or {}).get("rubric") or []
        return scorer.score(r["question"], r["answer"], rubric) if rubric else None

    with ThreadPoolExecutor(max(1, max_concurrency)) as ex:
        luna = list(ex.map(one, rows))
    pairs = [(100.0 * r["gemma"], 100.0 * s) for r, s in zip(rows, luna) if s is not None]
    out: dict[str, Any] = {"n": len(rows), "scored": len(pairs), "unparsed": len(rows) - len(pairs)}
    if pairs:
        out["gemma"] = statistics.fmean(g for g, _ in pairs)
        out["luna"] = statistics.fmean(s for _, s in pairs)
        out["gap"] = statistics.fmean(g - s for g, s in pairs)
    return out


def assess(steps: Sequence[dict[str, Any]], cfg: DriftConfig) -> dict[str, Any]:
    """Alert state over the per-step summaries (sorted by step)."""
    from dpt.monitoring.divergence import ProxyGoldDivergence

    usable = [s for s in steps if s.get("scored") and s["unparsed"] / max(1, s["n"]) <= cfg.max_unparsed_rate]
    reasons: list[str] = []
    if len(usable) >= 3:
        base = usable[0]["gap"]
        growth = [s["gap"] - base for s in usable]
        if all(g > cfg.max_gap_growth for g in growth[-2:]):
            reasons.append(f"Gemma − Luna gap grew by {growth[-2]:.1f} and {growth[-1]:.1f} points since step "
                           f"{usable[0]['step']} (limit {cfg.max_gap_growth:g}) on the last two checks")
    det = ProxyGoldDivergence(consecutive=cfg.consecutive, gold_tolerance=cfg.gold_tolerance)
    hit = False
    for s in usable:
        hit = det.observe(s["step"], s["gemma"], s["luna"])
    if hit:
        reasons.append(f"Gemma's score rose while Luna's stayed flat or fell for {det.streak} checks in a row")
    return {"alert": bool(reasons), "reasons": reasons, "usable_steps": [s["step"] for s in usable]}


def step_files(run: Path) -> list[tuple[int, Path]]:
    out = []
    for p in (run / "val_generations").glob("*.jsonl"):
        try:
            out.append((int(p.stem), p))
        except ValueError:
            continue
    return sorted(out)


def run_check(run: Path, dev: Path, scorer: LunaScorer, cfg: DriftConfig, *, max_concurrency: int = 8
              ) -> dict[str, Any]:
    questions = load_dev_questions(dev)
    if not questions:
        raise SystemExit(f"{dev}: no T3 dev rows (wrong file?)")
    summary_path = run / "judge_drift" / "summary.json"
    done = {}
    if summary_path.exists():
        done = {s["step"]: s for s in json.loads(summary_path.read_text()).get("steps", [])}
    steps = []
    for step, path in step_files(run):
        if step in done and done[step].get("scored"):
            steps.append(done[step])
            continue
        raw = list(read_jsonl(path))
        rows = select_rows(raw, questions, cfg.n_prompts, cfg.min_responses)
        if not rows:
            t3 = [r for r in raw if r.get("task") == T3]
            log.warning("step %d: no T3 dev response to score (%d T3 rows; %d match a dev prompt; %d have an "
                        "end-of-thinking marker)", step, len(t3), sum(r.get("gts") in questions for r in t3),
                        sum("</think>" in str(r.get("output") or "") for r in t3))
            continue
        s = {"step": step, **score_step(rows, scorer, max_concurrency=max_concurrency)}
        log.info("step %d: Gemma %.1f, Luna %s on %d responses", step, s.get("gemma", float("nan")),
                 f"{s['luna']:.1f}" if "luna" in s else "—", s["scored"])
        steps.append(s)
    result = {"steps": steps, **assess(steps, cfg), "served_models": sorted(scorer.served),
              "config": cfg.__dict__}
    write_json(summary_path, result)
    return result


def main(argv: Sequence[str] | None = None, *, chat: Any = None) -> int:
    from dpt.common.config import load_config, to_container

    ap = argparse.ArgumentParser(description="Compare the T3 reward judge with GPT-6 Luna on RL dev responses")
    ap.add_argument("--run", required=True, help="RL run dir (has val_generations/)")
    ap.add_argument("--dev", default=None, help="rl_dev.jsonl (default: <data_root>/v1/rl_dev.jsonl)")
    ap.add_argument("--n-prompts", type=int, default=DriftConfig.n_prompts)
    ap.add_argument("--min-responses", type=int, default=DriftConfig.min_responses)
    ap.add_argument("--max-gap-growth", type=float, default=DriftConfig.max_gap_growth)
    ap.add_argument("--consecutive", type=int, default=DriftConfig.consecutive)
    ap.add_argument("--max-concurrency", type=int, default=8)
    ap.add_argument("--model-line", default=None)
    args = ap.parse_args(argv)
    cfg_all = to_container(load_config("eval/eval_suite.yaml", model_line=args.model_line))
    run = Path(args.run)
    dev = Path(args.dev or f"{cfg_all['project']['paths']['data_root']}/v1/rl_dev.jsonl")
    if chat is None:
        chat = endpoint_from_config(cfg_all["eval_suite"].get("second_judge"), ChatEndpoint)
        if chat is None:
            raise SystemExit("second judge not usable here: needs project.judges.second, its API key, and a "
                             "machine that reaches the API (a login node)")
    cfg = DriftConfig(n_prompts=args.n_prompts, min_responses=args.min_responses, max_gap_growth=args.max_gap_growth,
                      consecutive=args.consecutive)
    scorer = LunaScorer(chat, run / "judge_drift" / "luna_scores.jsonl")
    result = run_check(run, dev, scorer, cfg, max_concurrency=args.max_concurrency)
    print(f"{'step':>6} {'Gemma':>7} {'Luna':>7} {'gap':>7} {'n':>4}")
    for s in result["steps"]:
        if s.get("scored"):
            print(f"{s['step']:>6} {s['gemma']:>7.1f} {s['luna']:>7.1f} {s['gap']:>7.1f} {s['scored']:>4}")
    if result["alert"]:
        for r in result["reasons"]:
            log.error("JUDGE DRIFT: %s", r)
        log.error("treat it like a proxy_gold alert (docs/runbook.md §3): read the behavior audit, then decide "
                  "whether to stop the run (scancel) and roll back")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
