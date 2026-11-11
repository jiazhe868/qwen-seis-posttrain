"""Collect per-run ``results.json`` files into one summary and render the report's results table.

Usage::

    python -m dpt.eval.report \\
        --row base_medium=eval_results/base_medium \\
        --row base_xhigh=eval_results/base_xhigh \\
        --row sft=eval_results/sft \\
        --row rl_seed1=eval_results/rl_seed1 --row rl_seed2=eval_results/rl_seed2 \\
        --label rl_seed1="SFT + RL (seed 1, delivered)" \\
        --summary-out eval_results/summary.json --markdown-out eval_results/report.md

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
        if r.get("blind_review"):  # dpt.eval.blind_review import
            row[f"{name}_blind_review"] = r["blind_review"]
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


def build_summary(run_dirs: dict[str, str | Path]) -> dict[str, Any]:
    rows = {name: row_from_results(read_json(Path(d) / "results.json")) for name, d in run_dirs.items()}
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
    return {"rows": rows, "base_row": BASE_ROW}


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
        br = r.get("d_report_blind_review")
        if not br or br.get("reviewer_win_rate") is None:
            continue
        js = ", ".join(f"{role} {v['win_rate_same_pairs']:.1f}% (agree {v['agreement']:.0%})"
                       for role, v in br.get("by_judge", {}).items())
        status = "" if br.get("status") == "done" else f"; {br.get('status')}"
        notes.append(f"Blind review, {labels.get(n, n)} ({br['n']} pairs{status}): reviewer win rate "
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
    args = ap.parse_args(argv)
    summary = build_summary(_kv(args.row))
    write_json(args.summary_out, summary)
    md = render_markdown(summary, _kv(args.label))
    if args.markdown_out:
        atomic_write_text(args.markdown_out, md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
