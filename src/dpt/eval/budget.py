"""Accuracy under a token budget, from a run's cached generations (no new generation).

A sample counts as correct at budget B only if it is correct and used at most B output tokens: with a cap of
B it would have been cut off. The curve over B still separates models on a set the base nearly saturates
(the base 9B scores 91.6 on D-Num, 2026-10-27), and it measures what RL's length penalty pushes on. Tokens per
correct answer (all output tokens / correct samples) is the cost of one right answer.

    python -m dpt.eval.budget --eval-root $WORK/dpt/eval_results --out $WORK/dpt/talk/budget

Writes one JSON of counts per run directory (no text), and skips a run whose JSON is newer than its
results.json. Numeric sets only: D-Code needs the sandbox to grade.

Three budgets and the paired comparison with SFT (2026-11-11)
-------------------------------------------------------------
The budget only caps an answer; it does not change sampling. So one 32K run also gives a 64K score: an answer whose
thinking did not end within 32K is continued by the same model for up to 32K more tokens (``continue``), and the
others count as they are. This is the method of the D-Num-hard 64K pass of 2026-11-10, whose continuations it
reuses. At 16K an answer counts if it used at most 16,384 tokens, as in the curve above.

RL v2's success test, set by the user on 2026-11-11, is at 32K: D-Num-hard and D-Fresh above SFT by more than their
noise. ``compare`` calls that "above noise" when the 95% bootstrap interval of the paired per-item difference
(candidate − SFT, items resampled) lies above 0. 16K and 64K are diagnostics: run 1 beat SFT at 16K (+10.2 on
D-Num-hard) and lost at 64K (−6.4), because it finished answers sooner and got more of them wrong.

    python -m dpt.eval.budget todo --set d_fresh --row rl_v2_seed1
    python -m dpt.eval.budget continue --set d_fresh --row rl_v2_seed1 --endpoint http://127.0.0.1:8000 \
        --served-model rl_v2_seed1
    python -m dpt.eval.budget compare --rows sft,rl_seed1,rl_v2_seed1 --sets d_num_hard,d_fresh --out $R/budgets.json

Runs: ``<eval_root>/d_num_hard/<row>`` for D-Num-hard, ``<eval_root>/<row>`` for the other sets. Continuations:
``<eval_root>/<set>_64k/<row>/continuations.jsonl``, appended as they arrive, so a rerun resumes. Outcomes follow
the failure-type analysis: thinking not ended = unfinished, a value outside tolerance = wrong_value, any other
failure = format.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from dpt.chat.thinking import split_thinking
from dpt.common.io import read_json, read_jsonl, write_json
from dpt.common.logging import get_logger
from dpt.eval.datasets import EvalItem, get_grader, load_eval_items
from dpt.eval.harness import Sample, SetSpec, _set_specs, grade_sample

log = get_logger(__name__)
BUDGETS = (2048, 4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768)
THREE_BUDGETS = (16384, 32768, 65536)
OUTCOMES = ("right", "wrong_value", "unfinished", "format")


def _quantile(xs: Sequence[int], q: float) -> int | None:
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))] if s else None


def latest_samples(name: str, items: Sequence[EvalItem], records: Iterable[dict[str, Any]],
                   n: int) -> dict[str, list[Sample]]:
    """The first ``n`` samples of every item, read as GenerationCache reads them (the last record per item wins, per
    spec key), from the last spec key whose records cover every item with ``n`` samples."""
    ids = {it.id for it in items}
    by_key: dict[str, dict[str, list[Sample]]] = defaultdict(dict)
    for rec in records:                           # streamed: a pass@64 cache is ~1 GB of text
        if rec["item_id"] in ids:
            by_key[rec.get("spec_key", "")][rec["item_id"]] = [Sample(**s) for s in rec["samples"][:n]]
    keys = [k for k, g in by_key.items() if all(len(g.get(i, ())) >= n for i in ids)]
    if not keys:
        raise ValueError(f"{name}: no cached generations cover all {len(ids)} items with {n} samples")
    return by_key[keys[-1]]


def graded_samples(
    spec: SetSpec, items: Sequence[EvalItem], records: Iterable[dict[str, Any]], n: int
) -> list[tuple[bool, int]]:
    """(correct, output tokens) for the first ``n`` samples of every item, graded as the harness grades."""
    grader = get_grader(spec.grader, **spec.grader_kwargs)
    gens = latest_samples(spec.name, items, records, n)
    return [(grade_sample(s, it, grader).correct, int(s.n_tokens)) for it in items for s in gens[it.id]]


def budget_curve(samples: Sequence[tuple[bool, int]], max_tokens: int,
                 budgets: Sequence[int] = BUDGETS) -> dict[str, Any]:
    bs = [b for b in budgets if b < max_tokens] + [max_tokens]
    right = [t for c, t in samples if c]
    wrong = [t for c, t in samples if not c]
    return {
        "n_samples": len(samples),
        "n_correct": len(right),
        "max_tokens": max_tokens,
        "budgets": bs,
        # at the run's own cap nothing more is cut, whatever the re-tokenized length says
        "correct_within": [sum(1 for c, t in samples if c and (t <= b or b >= max_tokens)) for b in bs],
        "tokens_total": sum(t for _, t in samples),
        "tokens_correct_p50": _quantile(right, 0.5),
        "tokens_correct_p90": _quantile(right, 0.9),
        "tokens_wrong_p50": _quantile(wrong, 0.5),
    }


def analyze_run(run_dir: Path, cfg: dict[str, Any], sets: Sequence[str], *, verify: bool = True) -> dict[str, Any]:
    res = read_json(run_dir / "results.json")
    out: dict[str, Any] = {"model_name": res.get("model_name"), "seed": res.get("seed"),
                           "max_tokens": res.get("max_tokens"), "sets": {}}
    for spec in _set_specs(cfg, [s for s in sets if s in (res.get("sets") or {})]):
        cache = run_dir / spec.name / "generations.jsonl"
        if not cache.exists():
            continue
        n = int(res["sets"][spec.name].get("n_samples") or spec.samples)
        items = load_eval_items(spec.path, spec.name, verify=verify)
        curve = budget_curve(graded_samples(spec, items, read_jsonl(cache), n), int(res.get("max_tokens") or 32768))
        score = res["sets"][spec.name].get("score")
        if score is not None and abs(100 * curve["n_correct"] / curve["n_samples"] - score) > 1e-6:
            log.warning("%s/%s: regraded %.3f%% vs results.json %.3f%%", run_dir.name, spec.name,
                        100 * curve["n_correct"] / curve["n_samples"], score)
        out["sets"][spec.name] = curve
    return out


# ---- three budgets and the paired comparison with SFT (module docstring) ----------------------------------------
def outcome(sample: Sample, item: EvalItem, grader: Any) -> str:
    g = grade_sample(sample, item, grader)
    if g.correct:
        return "right"
    if g.reason == "no_think_end":
        return "unfinished"
    return "wrong_value" if g.reason == "wrong_value" else "format"


def run_dir_of(root: Path, set_name: str, row: str) -> Path:
    return root / "d_num_hard" / row if set_name == "d_num_hard" else root / row


def continuations_path(root: Path, set_name: str, row: str) -> Path:
    return root / f"{set_name}_64k" / row / "continuations.jsonl"


def read_continuations(path: Path) -> dict[str, dict[str, Any]]:
    return {r["key"]: r for r in read_jsonl(path)} if path.exists() else {}


def row_samples(spec: SetSpec, run_dir: Path, *, verify: bool = True) -> tuple[list[EvalItem], dict[str, list[Sample]]]:
    res = read_json(run_dir / "results.json")
    n = int(res["sets"][spec.name].get("n_samples") or spec.samples)
    items = load_eval_items(spec.path, spec.name, verify=verify)
    return items, latest_samples(spec.name, items, read_jsonl(run_dir / spec.name / "generations.jsonl"), n)


def three_budget_scores(spec: SetSpec, items: Sequence[EvalItem], gens: dict[str, list[Sample]],
                        continuations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per-item scores (% right over the item's samples) and outcome counts at 16K, 32K and 64K."""
    grader = get_grader(spec.grader, **spec.grader_kwargs)
    per_item: dict[int, dict[str, float]] = {b: {} for b in THREE_BUDGETS}
    counts: dict[int, Counter[str]] = {b: Counter() for b in THREE_BUDGETS}
    missing = 0
    for it in items:
        right: Counter[int] = Counter()
        for i, s in enumerate(gens[it.id]):
            at_32k = outcome(s, it, grader)
            at_16k = at_32k if int(s.n_tokens) <= THREE_BUDGETS[0] else "unfinished"
            at_64k = at_32k
            if at_32k == "unfinished":
                c = continuations.get(f"{it.id}|{i}")
                if c is None:
                    missing += 1
                else:
                    longer = Sample(text=s.text + c["text"], finish_reason=c.get("finish_reason") or "stop",
                                    n_tokens=int(s.n_tokens) + int(c.get("n_tokens") or 0))
                    at_64k = outcome(longer, it, grader)
            for b, o in zip(THREE_BUDGETS, (at_16k, at_32k, at_64k)):
                counts[b][o] += 1
                right[b] += o == "right"
        for b in THREE_BUDGETS:
            per_item[b][it.id] = 100.0 * right[b] / max(1, len(gens[it.id]))
    return {"per_item": per_item, "counts": counts, "missing_continuations": missing}


def paired_interval(candidate: dict[str, float], baseline: dict[str, float], *, n_boot: int = 10000,
                    seed: int = 1) -> dict[str, float]:
    """Mean of candidate − baseline over the items both have, and its 95% bootstrap interval (items resampled)."""
    ids = sorted(set(candidate) & set(baseline))
    d = np.array([candidate[i] - baseline[i] for i in ids], dtype=float)
    if not len(d):
        return {"items": 0, "diff": float("nan"), "low": float("nan"), "high": float("nan")}
    means = d[np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"items": len(ids), "diff": round(float(d.mean()), 2), "low": round(float(low), 2),
            "high": round(float(high), 2)}


def compare(cfg: dict[str, Any], root: Path, rows: Sequence[str], baseline: str, sets: Sequence[str], *,
            verify: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {"collected_at": dt.datetime.now().strftime("%Y-%m-%dT%H:%M"), "baseline": baseline,
                           "budgets": list(THREE_BUDGETS), "sets": {}}
    for spec in _set_specs(cfg, sets):
        scored = {}
        for row in rows:
            run_dir = run_dir_of(root, spec.name, row)
            if not (run_dir / spec.name / "generations.jsonl").exists():
                continue
            items, gens = row_samples(spec, run_dir, verify=verify)
            scored[row] = three_budget_scores(spec, items, gens,
                                              read_continuations(continuations_path(root, spec.name, row)))
        entry: dict[str, Any] = {}
        for row, sc in scored.items():
            e: dict[str, Any] = {"missing_continuations": sc["missing_continuations"]}
            for b in THREE_BUDGETS:
                n = sum(sc["counts"][b].values())
                e[f"at_{b // 1024}k"] = {
                    "score": round(float(np.mean(list(sc["per_item"][b].values()))), 1),
                    "outcomes_pct": {o: round(100 * sc["counts"][b][o] / max(1, n), 1) for o in OUTCOMES}}
                if row != baseline and baseline in scored:
                    iv = paired_interval(sc["per_item"][b], scored[baseline]["per_item"][b])
                    e[f"at_{b // 1024}k"]["vs_baseline"] = {**iv, "above_noise": bool(iv["low"] > 0)}
            entry[row] = e
        out["sets"][spec.name] = entry
    return out


async def continue_answers(a: argparse.Namespace, cfg: dict[str, Any], root: Path, base_model: str) -> None:
    """Continue every answer of a row whose thinking did not end within 32K, up to 64K tokens in all."""
    import httpx

    from dpt.eval.harness import build_messages, hf_renderer

    spec = _set_specs(cfg, [a.set])[0]
    items, gens = row_samples(spec, run_dir_of(root, a.set, a.row))
    by_id = {it.id: it for it in items}
    render = hf_renderer(base_model)
    path = continuations_path(root, a.set, a.row)
    done = read_continuations(path)
    jobs = [(it.id, i, s) for it in items for i, s in enumerate(gens[it.id])
            if not split_thinking(s.text).has_end and f"{it.id}|{i}" not in done]
    print(f"{a.set} {a.row}: {len(jobs)} continuations to do ({len(done)} cached)", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    stop_at = (t0 + a.stop_new_after) if a.stop_new_after else None
    sem, lock, stats = asyncio.Semaphore(a.concurrency), asyncio.Lock(), Counter()
    async with httpx.AsyncClient(timeout=a.timeout_s) as client:

        async def one(item_id: str, i: int, s: Sample) -> None:
            async with sem:
                if stop_at and time.time() > stop_at:
                    stats["skipped_for_time"] += 1
                    return
                key = f"{item_id}|{i}"
                # the harness's own prompt at medium effort, then the stored 32K text; temperature 1, top-p 1
                prompt = render(build_messages(by_id[item_id]), "medium") + s.text
                seed = int(hashlib.sha256(f"{a.row}|{key}|64k".encode()).hexdigest()[:8], 16) % (2 ** 31)
                body = {"model": a.served_model, "prompt": prompt, "n": 1, "seed": seed,
                        "max_tokens": max(1, THREE_BUDGETS[-1] - int(s.n_tokens)), "temperature": 1.0, "top_p": 1.0}
                for attempt in range(4):
                    try:
                        r = await client.post(f"{a.endpoint.rstrip('/')}/v1/completions", json=body)
                        r.raise_for_status()
                        break
                    except (httpx.TransportError, httpx.HTTPStatusError) as e:
                        if attempt == 3:
                            print(f"FAIL {key}: {type(e).__name__}: {e}", flush=True)
                            stats["failed"] += 1
                            return
                        await asyncio.sleep(10 * (attempt + 1))
                data = r.json()
                ch = data["choices"][0]
                rec = {"key": key, "item": item_id, "sample": i, "text": ch.get("text") or "",
                       "finish_reason": ch.get("finish_reason"),
                       "n_tokens": int((data.get("usage") or {}).get("completion_tokens") or 0),
                       "prefix_tokens": int(s.n_tokens), "served": data.get("model")}
                async with lock:
                    with open(path, "a") as fh:
                        fh.write(json.dumps(rec) + "\n")
                    stats["done"] += 1
                    if stats["done"] % 20 == 0:
                        print(f"{a.set} {a.row}: {stats['done']}/{len(jobs)} in {time.time() - t0:.0f}s", flush=True)

        await asyncio.gather(*(one(*j) for j in jobs))
    print(f"{a.set} {a.row}: finished {dict(stats)} in {time.time() - t0:.0f}s", flush=True)


def three_budget_main(argv: Sequence[str]) -> int:
    from dpt.common.config import load_config, to_container

    ap = argparse.ArgumentParser(prog="python -m dpt.eval.budget", description="three budgets (module docstring)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("todo")
    t.add_argument("--set", required=True)
    t.add_argument("--row", required=True)
    c = sub.add_parser("continue")
    c.add_argument("--set", required=True)
    c.add_argument("--row", required=True)
    c.add_argument("--endpoint", required=True)
    c.add_argument("--served-model", required=True)
    c.add_argument("--concurrency", type=int, default=60)
    c.add_argument("--timeout-s", type=float, default=10800)
    c.add_argument("--stop-new-after", type=float, default=0, help="seconds from start after which no new request")
    m = sub.add_parser("compare")
    m.add_argument("--rows", required=True, help="comma-separated, e.g. sft,rl_seed1,rl_v2_seed1")
    m.add_argument("--baseline", default="sft")
    m.add_argument("--sets", default="d_num_hard,d_fresh")
    m.add_argument("--out", type=Path, default=None)
    m.add_argument("--config", default="eval/eval_suite.yaml")
    a = ap.parse_args(argv)
    full = load_config(getattr(a, "config", "eval/eval_suite.yaml"))
    cfg = to_container(full)["eval_suite"]
    root = Path(str(full.project.paths.eval_root))
    if a.cmd == "todo":
        spec = _set_specs(cfg, [a.set])[0]
        run_dir = run_dir_of(root, a.set, a.row)
        if not (run_dir / a.set / "generations.jsonl").exists():
            print(0)
            return 0
        items, gens = row_samples(spec, run_dir)
        done = read_continuations(continuations_path(root, a.set, a.row))
        print(sum(1 for it in items for i, s in enumerate(gens[it.id])
                  if not split_thinking(s.text).has_end and f"{it.id}|{i}" not in done))
        return 0
    if a.cmd == "continue":
        asyncio.run(continue_answers(a, cfg, root, str(full.project.model.base)))
        return 0
    res = compare(cfg, root, a.rows.split(","), a.baseline, a.sets.split(","))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))
    for set_name, rows in res["sets"].items():
        for row, e in rows.items():
            parts = []
            for b in THREE_BUDGETS:
                v = e[f"at_{b // 1024}k"]
                vs = v.get("vs_baseline")
                parts.append(f"{b // 1024}K {v['score']:.1f}" + (
                    f" ({vs['diff']:+.1f} [{vs['low']:+.1f}, {vs['high']:+.1f}])" if vs else ""))
            print(f"{set_name} {row}: " + " | ".join(parts) + f" | missing {e['missing_continuations']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("todo", "continue", "compare"):
        return three_budget_main(argv)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default="d_num,d_fresh,d_num_hard")
    ap.add_argument("--config", default="eval/eval_suite.yaml")
    ap.add_argument("runs", nargs="*", help="run directories under --eval-root (default: every finished run)")
    args = ap.parse_args(argv)

    from dpt.common.config import load_config, to_container

    cfg = to_container(load_config(args.config))["eval_suite"]
    root, out = Path(args.eval_root), Path(args.out)
    runs = args.runs or sorted(str(p.parent.relative_to(root)) for p in root.glob("**/results.json"))
    for r in runs:
        dst = out / (r.replace("/", "__") + ".json")
        if dst.exists() and dst.stat().st_mtime >= (root / r / "results.json").stat().st_mtime:
            continue
        a = analyze_run(root / r, cfg, args.sets.split(","))
        if a["sets"]:
            write_json(dst, {"run": r, **a})
            log.info("%s: %s", r, {s: f"{v['n_correct']}/{v['n_samples']}" for s, v in a["sets"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
