"""Collect per-run ``results.json`` files into one summary and render the report's results table.

Usage::

    python -m dpt.eval.report \\
        --row base_medium=eval_results/base_medium \\
        --row base_xhigh=eval_results/base_xhigh \\
        --row sft=eval_results/sft \\
        --row rl_seed1=eval_results/rl_seed1 --row rl_seed2=eval_results/rl_seed2 \\
        --label rl_seed1="SFT + RL (seed 1, delivered)" \\
        --summary-out eval_results/summary.json --markdown-out eval_results/report.md

``--exclude FILE`` (item ids, one per line) scores every row without those items: the screened subset of the
2026-11-14 review of the frozen sets, which flagged items with a faulty gold answer, wording or rubric. The list
stays private; the table notes how many items each set lost.

Row names matter: :mod:`dpt.eval.gates` looks up the reference rows by the
names used in ``configs/project.yaml`` (``base_medium``, ``base_xhigh``).
``general_mean_delta`` is always computed against the ``base_medium`` row.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from dpt.common.io import read_json, write_json
from dpt.common.io import atomic_write_text

DOMAIN_SETS = ("d_num", "d_code", "d_report", "d_fresh")
BASE_ROW = "base_medium"


def row_from_results(results: dict[str, Any]) -> dict[str, Any]:
    """Flatten one results.json into a summary row."""
    row: dict[str, Any] = {
        "model_name": results.get("model_name"),
        "effort": results.get("effort"),
        "config_hash": results.get("config_hash"),
    }
    for name, r in results.get("sets", {}).items():
        if name == "refusal":
            row["refusal_rate"] = r.get("refusal_rate")
            continue
        if r.get("by_judge"):  # D-Report with two judges: keep both, even when the gated score is pending
            row[f"{name}_by_judge"] = {
                "scores": {role: b.get("score") for role, b in r["by_judge"].items()},
                "gate_judge": r.get("gate_judge"),
                "agreement": r.get("judge_agreement"),
            }
        for review in ("blind_review", "blind_review_model"):  # dpt.eval.blind_review import --reviewer person|model
            if r.get(review):
                row[f"{name}_{review}"] = r[review]
        if r.get("score") is None:
            continue
        row[name] = {
            "score": r["score"],
            "ci_halfwidth": (r.get("ci") or {}).get("halfwidth"),
            "metric": r.get("metric"),
            "truncation_rate": r.get("truncation_rate"),
            "mean_output_tokens": r.get("mean_output_tokens"),
        }
        if "pass_at_k" in r:
            row.setdefault("pass_at_k", {})[name] = r["pass_at_k"]
        if "length_matched" in r:
            row[name]["length_matched"] = r["length_matched"]
        if "score_by_max_tokens" in r:
            row[name]["score_by_max_tokens"] = r["score_by_max_tokens"]
    row["general"] = {
        b: r["score"] for b, r in results.get("general", {}).items() if r.get("score") is not None
    }
    row["general_relative_only"] = [
        b for b, r in results.get("general", {}).items() if r.get("relative_only")
    ]
    row["mean_output_tokens"] = results.get("mean_output_tokens")
    row["data_hashes"] = results.get("data_hashes", {})
    return row


def screened(results: dict[str, Any], exclude: set[str]) -> tuple[dict[str, Any], dict[str, int]]:
    """One results.json scored without the excluded item ids, and how many items each set lost.

    Only sets that lose an item are rescored. Numeric sets: the mean of the per-item means. D-Report: each judge's
    win rate over the items left, and the score of the judge the run's gate picks ("min": the lower one). The
    intervals and length-matched scores are dropped, since the samples behind them are not in results.json."""
    from copy import deepcopy

    from dpt.eval.metrics import win_rate

    out, lost = deepcopy(results), {}
    for name, r in out.get("sets", {}).items():
        per_item = r.get("per_item") or []
        kept = [x for x in per_item if x["id"] not in exclude]
        if len(kept) == len(per_item):
            continue
        lost[name] = len(per_item) - len(kept)
        r["per_item"], r["n_items"] = kept, len(kept)
        for k in ("ci", "item_ci", "length_matched", "pass_at_k"):
            r.pop(k, None)
        if r.get("metric") == "winrate_vs_base" and not r.get("by_judge"):     # the base row: ties by definition
            r["score"] = win_rate([x["outcome"] for x in kept])
            r["outcomes"] = {o: sum(x["outcome"] == o for x in kept) for o in ("win", "tie", "loss")}
        elif r.get("metric") == "winrate_vs_base":
            for role, b in r["by_judge"].items():
                b["score"] = win_rate([x["by_judge"][role] for x in kept if role in (x.get("by_judge") or {})])
                b.pop("ci", None)
            gate = r.get("gate_judge", "primary")
            roles = ["primary", "second"] if gate == "min" else [gate]
            r["score_judge"] = min(roles, key=lambda role: r["by_judge"][role]["score"])
            r["score"] = r["by_judge"][r["score_judge"]]["score"]
            r["outcomes"] = {o: sum(x["by_judge"][r["score_judge"]] == o for x in kept) for o in ("win", "tie", "loss")}
        else:
            r["score"] = 100.0 * sum(float(x["mean"]) for x in kept) / max(1, len(kept))
    return out, lost


def build_summary(run_dirs: dict[str, str | Path], exclude: set[str] | None = None) -> dict[str, Any]:
    results = {name: read_json(Path(d) / "results.json") for name, d in run_dirs.items()}
    lost: dict[str, int] = {}
    if exclude:
        for name in results:
            results[name], lost_here = screened(results[name], exclude)
            for set_name, n in lost_here.items():
                lost[set_name] = max(lost.get(set_name, 0), n)
    rows = {name: row_from_results(r) for name, r in results.items()}
    # pass@k curves (k up to 64) come from a sibling run "<dir>_k64" made with --samples-override 64
    # (scripts 03/07). Merge them so the RL-entry gate (pass@16 ≥ base) and the pass@64 floor see them.
    for name, d in run_dirs.items():
        k64 = Path(str(d).rstrip("/") + "_k64") / "results.json"
        if k64.exists():
            extra = row_from_results(read_json(k64)).get("pass_at_k", {})
            rows[name].setdefault("pass_at_k", {}).update(extra)
    base = rows.get(BASE_ROW)
    for name, row in rows.items():
        if base and base.get("general") and row.get("general"):
            common = sorted(set(base["general"]) & set(row["general"]))
            if common:
                row["general_mean_delta"] = sum(row["general"][b] - base["general"][b] for b in common) / len(
                    common
                )
                row["general_benchmarks"] = common
    out: dict[str, Any] = {"rows": rows, "base_row": BASE_ROW}
    if exclude:
        out["screened"] = {"items_left_out": lost}
    return out


def _cell(row: dict[str, Any], key: str, pct: bool = False) -> str:
    r = row.get(key)
    if not isinstance(r, dict):
        return "—"
    s = f"{r['score']:.1f}{'%' if pct else ''}"
    if r.get("ci_halfwidth") is not None:
        s += f" ±{r['ci_halfwidth']:.1f}"
    return s


def render_markdown(summary: dict[str, Any], labels: dict[str, str] | None = None) -> str:
    """Markdown table in the layout of the report's "最终设想结果" table."""
    labels = labels or {}
    head = (
        "| Model | D-Num | D-Code | D-Report win rate | D-Fresh | General 5 avg Δ | Mean output tokens |\n"
        "|---|---|---|---|---|---|---|"
    )
    lines = [head]
    for name, row in summary["rows"].items():
        gd = row.get("general_mean_delta")
        tok = row.get("mean_output_tokens")
        lines.append(
            f"| {labels.get(name, name)} | {_cell(row, 'd_num')} | {_cell(row, 'd_code')} | "
            f"{_cell(row, 'd_report', pct=True)} | {_cell(row, 'd_fresh')} | "
            f"{'—' if gd is None else f'{gd:+.1f}'} | {'—' if tok is None else f'{tok / 1000:.1f}K'} |"
        )
    notes = [
        "",
        "± is the 95% generation-noise interval (resampling within items; see docs/evaluation.md).",
        "Gains count only when they exceed the noise band.",
    ]
    if summary.get("screened"):
        lost = summary["screened"]["items_left_out"]
        notes.append("Screened subset: items flagged by the 2026-11-14 review are left out ("
                     + (", ".join(f"{k} {v}" for k, v in sorted(lost.items())) or "none in these sets")
                     + "); the rescored sets have no interval.")
    judged = {n: r["d_report_by_judge"] for n, r in summary["rows"].items() if r.get("d_report_by_judge")}
    if judged:
        parts = []
        for n, j in judged.items():
            sc = ", ".join(f"{role} {'pending' if v is None else f'{v:.1f}%'}" for role, v in j["scores"].items())
            agr = j.get("agreement")
            parts.append(f"{labels.get(n, n)}: {sc}" + ("" if agr is None else f" (agree {agr:.0%})"))
        gate = next(iter(judged.values())).get("gate_judge")
        notes.append(f"D-Report by judge ({'; '.join(parts)}); the table shows the gated value ({gate}).")
    for n, r in summary["rows"].items():
        for review, title in (("d_report_blind_review", "Blind review"),
                              ("d_report_blind_review_model", "Blind review by a model")):
            br = r.get(review)
            if not br or br.get("reviewer_win_rate") is None:
                continue
            js = ", ".join(f"{role} {v['win_rate_same_pairs']:.1f}% (agree {v['agreement']:.0%})"
                           for role, v in br.get("by_judge", {}).items())
            status = "" if br.get("status") == "done" else f"; {br.get('status')}"
            notes.append(f"{title}, {labels.get(n, n)} ({br['n']} pairs{status}): reviewer win rate "
                         f"{br['reviewer_win_rate']:.1f}%; judges on the same pairs: {js or '—'}.")
    rel = sorted({b for r in summary["rows"].values() for b in r.get("general_relative_only", [])})
    if rel:
        notes.append(
            f"Relative-only benchmarks (base memorization evidence): {', '.join(rel)}; read deltas, "
            "not absolute scores."
        )
    return "\n".join(lines + notes) + "\n"


def _kv(items: Sequence[str]) -> dict[str, str]:
    out = {}
    for it in items:
        k, _, v = it.partition("=")
        if not v:
            raise SystemExit(f"expected NAME=VALUE, got {it!r}")
        out[k] = v
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build eval summary + markdown table")
    ap.add_argument("--row", action="append", default=[], help="NAME=RUN_DIR (repeatable, ordered)")
    ap.add_argument("--label", action="append", default=[], help="NAME=display label")
    ap.add_argument("--summary-out", required=True)
    ap.add_argument("--markdown-out", default=None)
    ap.add_argument("--exclude", default=None, help="file of item ids to leave out: the screened subset")
    args = ap.parse_args(argv)
    from dpt.eval.budget import read_exclude

    summary = build_summary(_kv(args.row), read_exclude(Path(args.exclude)) if args.exclude else None)
    write_json(args.summary_out, summary)
    md = render_markdown(summary, _kv(args.label))
    if args.markdown_out:
        atomic_write_text(args.markdown_out, md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
