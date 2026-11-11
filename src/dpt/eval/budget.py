"""Accuracy under a token budget, from a run's cached generations (no new generation).

A sample counts as correct at budget B only if it is correct and used at most B output tokens: with a cap of
B it would have been cut off. The curve over B still separates models on a set the base nearly saturates
(the base 9B scores 91.6 on D-Num, 2026-10-27), and it measures what RL's length penalty pushes on. Tokens per
correct answer (all output tokens / correct samples) is the cost of one right answer.

    python -m dpt.eval.budget --eval-root $WORK/dpt/eval_results --out $WORK/dpt/talk/budget

Writes one JSON of counts per run directory (no text), and skips a run whose JSON is newer than its
results.json. Numeric sets only: D-Code needs the sandbox to grade.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from dpt.common.io import read_json, read_jsonl, write_json
from dpt.common.logging import get_logger
from dpt.eval.datasets import EvalItem, get_grader, load_eval_items
from dpt.eval.harness import Sample, SetSpec, _set_specs, grade_sample

log = get_logger(__name__)
BUDGETS = (2048, 4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768)


def _quantile(xs: Sequence[int], q: float) -> int | None:
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))] if s else None


def graded_samples(
    spec: SetSpec, items: Sequence[EvalItem], records: Iterable[dict[str, Any]], n: int
) -> list[tuple[bool, int]]:
    """(correct, output tokens) for the first ``n`` samples of every item, graded as the harness grades.

    Records are read as GenerationCache reads them (the last one per item wins, per spec key). The spec key
    used is the last one whose records cover every item with ``n`` samples.
    """
    by_id = {it.id: it for it in items}
    grader = get_grader(spec.grader, **spec.grader_kwargs)
    by_key: dict[str, dict[str, list[tuple[bool, int]]]] = defaultdict(dict)
    for rec in records:                           # streamed: a pass@64 cache is ~1 GB of text
        it = by_id.get(rec["item_id"])
        if it is None:
            continue
        smps = [Sample(**s) for s in rec["samples"][:n]]
        by_key[rec.get("spec_key", "")][it.id] = [(grade_sample(s, it, grader).correct, int(s.n_tokens)) for s in smps]
    keys = [k for k, g in by_key.items() if all(len(g.get(i, ())) >= n for i in by_id)]
    if not keys:
        raise ValueError(f"{spec.name}: no cached generations cover all {len(by_id)} items with {n} samples")
    gens = by_key[keys[-1]]
    return [x for it in items for x in gens[it.id]]


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


def main(argv: Sequence[str] | None = None) -> int:
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
