"""Render the talk's running numbers file and its figures from the JSON scripts/talk/collect.py writes on the cluster.

    rsync cluster:$WORK/dpt/talk/data.json docs/talk/data/run.json
    ~/miniconda3/bin/python scripts/talk/render.py      # → docs/talk/numbers.md, docs/talk/figures/*.png|pdf

docs/talk/numbers.md is generated: hand-written notes go in the run log, which is copied in at the end.
Figures follow the reference data-viz palette (light mode): one colour per task in every figure, thin marks with
a rounded data end, hairline grid, text in ink colours. PNGs are 200 dpi for slides; the PDFs are vector.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TALK = ROOT / "docs" / "talk"
NOTES = TALK / "notes.md"

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, BLUE_LIGHT = "#2a78d6", "#eb6834", "#b7d3f6"
TASKS = ["t1_num", "t2_code", "t3_report", "t4_general"]
TASK_COLOR = dict(zip(TASKS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))   # validated in this order
TASK_NAME = {"t1_num": "T1 numeric", "t2_code": "T2 code", "t3_report": "T3 report", "t4_general": "T4 general"}
DPI = 200
PX = DPI / 96            # one CSS pixel in device pixels
PT = 72 / 96             # one CSS pixel in points
FUNNEL = [("ingest", "Generated"), ("after_quality", "Quality"), ("after_dedup", "Dedup"),
          ("after_decontam", "Decontam"), ("after_verify", "Verify"), ("after_difficulty", "Published")]
REASON = {
    "exact_duplicate": "Ingest: exact duplicate", "repetition_4gram": "Quality: repeated 4-grams",
    "too_short": "Quality: too short", "near_duplicate": "Dedup: near duplicate",
    "decontam:ngram_minhash": "Decontam: n-gram / MinHash match", "decontam:semantic": "Decontam: semantic match",
    "verify:disagreement": "Verify: solver and models disagree", "verify:family_undecided": "Verify: models undecided",
    "verify:template_rejected": "Verify: whole template rejected", "all_correct": "Difficulty: problem solved 8/8",
    "group_all_correct": "Difficulty: whole template solved 8/8"}
# Eval grader reasons (dpt.eval.graders), grouped; anything else (sandbox runner errors, …) is "other"
OUTCOME = [("correct", {"ok"}), ("thinking never closed", {"no_think_end"}),
           ("answer format", {"multiple_answers", "no_answer", "missing_unit", "parse_error", "unit_parse_error",
                              "no_code_block", "empty_answer"}),
           ("wrong answer", {"wrong_value", "unit_mismatch", "tests_failed"})]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "svg.fonttype": "none", "pdf.fonttype": 42})


# ---- chart chrome -----------------------------------------------------------------------------------------
def figure(w: float = 10.0, h: float = 5.6):
    return plt.figure(figsize=(w, h), dpi=DPI, facecolor=SURFACE)


def axes(fig, rect, grid: str | None = "y"):
    ax = fig.add_axes(rect, facecolor=SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(1 * PT)
    ax.tick_params(length=0, labelsize=10.5, pad=6)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=1 * PT)
    ax.set_axisbelow(True)
    return ax


def heading(fig, title: str, sub: str = "") -> None:
    fig.text(0.06, 0.955, title, fontsize=15, fontweight="semibold", color=INK, ha="left", va="top")
    if sub:
        fig.text(0.06, 0.895, sub, fontsize=10.5, color=INK2, ha="left", va="top")


def legend(fig, items, y: float = 0.835) -> None:
    handles = [Rectangle((0, 0), 1, 1, fc=c, ec="none") if kind == "box" else
               Rectangle((0, 0), 1, 1, fc=SURFACE, ec=c, lw=1 * PT) if kind == "outline" else
               Line2D([0], [0], color=c, lw=2 * PT, solid_capstyle="round") if kind == "line" else
               Line2D([0], [0], color=c, lw=0, marker="o", ms=8 * PT, mec=c)
               for _, c, kind in items]
    fig.legend(handles, [n for n, _, _ in items], loc="upper left", bbox_to_anchor=(0.055, y), ncol=len(items),
               frameon=False, fontsize=10.5, labelcolor=INK2, handlelength=1.0, handleheight=1.0, columnspacing=1.6)


def thousands(v: float) -> str:
    return f"{v:,.0f}"


def compact(v: float) -> str:
    return f"{v / 1e6:.0f}M" if v >= 1e6 else f"{v / 1e3:.0f}K" if v >= 1e3 else f"{v:,.0f}"


def per_unit(ax):
    (x0, y0), (x1, y1) = ax.transData.transform([(0, 0), (1, 1)])
    return abs(x1 - x0), abs(y1 - y0)                    # device pixels per data unit in x and y


def bar(ax, center, base, length, thick, color, *, horizontal=False, rounded=True, zorder=2) -> None:
    """A bar with a 4 px rounded data end, square at the baseline. Axis limits must be final."""
    if length <= 0:
        return
    px, py = per_unit(ax)
    x, y, w, h = (base, center - thick / 2, length, thick) if horizontal else (center - thick / 2, base, thick, length)
    r = min(4 * PX, w * px / 2, h * py / 2)
    if not rounded or r < 0.5:
        ax.add_patch(Rectangle((x, y), w, h, fc=color, ec="none", zorder=zorder))
        return
    rx, ry = r / px, r / py
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rx}", mutation_aspect=ry / rx,
                                fc=color, ec="none", zorder=zorder))
    ax.add_patch(Rectangle((x, y), min(w, rx) if horizontal else w, h if horizontal else min(h, ry),
                           fc=color, ec="none", zorder=zorder))


def save(fig, name: str) -> str:
    out = TALK / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=DPI, facecolor=SURFACE)
    fig.savefig(out / f"{name}.pdf", facecolor=SURFACE)
    plt.close(fig)
    return f"figures/{name}.png"


# ---- figures ----------------------------------------------------------------------------------------------
def fig_funnel(p: dict[str, Any]) -> str:
    bt = p["by_task"]
    totals = [sum(bt[t][k] for t in TASKS) for k, _ in FUNNEL]
    fig = figure()
    heading(fig, "RL problem bank: 57,037 generated → 36,884 published",
            "Problems left after each stage, by task")
    legend(fig, [(TASK_NAME[t], TASK_COLOR[t], "box") for t in TASKS])
    ax = axes(fig, (0.08, 0.1, 0.89, 0.64))
    ax.set_xlim(-0.6, len(FUNNEL) - 0.4)
    ax.set_ylim(0, max(totals) * 1.12)
    ax.set_xticks(range(len(FUNNEL)), [n for _, n in FUNNEL])
    ax.yaxis.set_major_formatter(lambda v, _: compact(v))
    px, py = per_unit(ax)
    thick, gap = 24 * PX / px, 2 * PX / py
    for i, (k, _) in enumerate(FUNNEL):
        base = 0.0
        vals = [(t, bt[t][k]) for t in TASKS if bt[t][k] > 0]
        for j, (t, v) in enumerate(vals):
            last = j == len(vals) - 1
            lo, hi = base + (gap / 2 if j else 0), base + v - (0 if last else gap / 2)
            bar(ax, i, lo, hi - lo, thick, TASK_COLOR[t], rounded=last)
            base += v
        ax.text(i, totals[i] + max(totals) * 0.02, thousands(totals[i]), ha="center", va="bottom", fontsize=10.5,
                color=INK2)
    return save(fig, "bank_funnel")


def fig_retention(p: dict[str, Any]) -> str:
    bt = p["by_task"]
    fig = figure()
    heading(fig, "Share of each task's generated problems that survive each stage",
            "T4 general keeps half (decontamination, the rejected gsm8k source, too-easy templates); "
            "T3 report keeps all")
    legend(fig, [(TASK_NAME[t], TASK_COLOR[t], "line") for t in TASKS])
    ax = axes(fig, (0.08, 0.1, 0.78, 0.64))
    ax.set_xlim(-0.3, len(FUNNEL) - 0.7)
    ax.set_ylim(0, 108)
    ax.set_xticks(range(len(FUNNEL)), [n for _, n in FUNNEL])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    for t in TASKS:
        ys = [100 * bt[t][k] / bt[t]["ingest"] for k, _ in FUNNEL]
        ax.plot(range(len(FUNNEL)), ys, color=TASK_COLOR[t], lw=2 * PT, solid_capstyle="round",
                solid_joinstyle="round", marker="o", ms=8 * PT, mec=SURFACE, mew=2 * PT, zorder=3)
        ax.text(len(FUNNEL) - 1 + 0.12, ys[-1], f"{TASK_NAME[t]} {ys[-1]:.0f}%", va="center", fontsize=10.5,
                color=INK2, clip_on=False)
    return save(fig, "bank_retention")


def fig_removals(p: dict[str, Any]) -> str:
    rows = sorted(((REASON.get(r, r), n) for s in p["stages"] for r, n in (s.get("reasons") or {}).items()),
                  key=lambda x: x[1])
    fig = figure(10, 6.2)
    heading(fig, f"What the pipeline removed: {sum(n for _, n in rows):,} problems",
            "Removals by reason; whole-template decisions (verify, difficulty) dominate")
    ax = axes(fig, (0.34, 0.06, 0.6, 0.78), grid="x")
    ax.spines["bottom"].set_visible(False)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(0, max(n for _, n in rows) * 1.15)
    ax.set_yticks(range(len(rows)), [r for r, _ in rows])
    ax.tick_params(axis="y", labelcolor=INK2)
    ax.xaxis.set_major_formatter(lambda v, _: compact(v))
    px, py = per_unit(ax)
    for i, (_, n) in enumerate(rows):
        bar(ax, i, 0, n, 18 * PX / py, BLUE, horizontal=True)
        ax.text(n + ax.get_xlim()[1] * 0.01, i, thousands(n), va="center", fontsize=10, color=INK2)
    return save(fig, "bank_removals")


def fig_difficulty(p: dict[str, Any]) -> str:
    hist = next(s for s in p["stages"] if s["stage"] == "difficulty")["histogram"]
    series = [t for t in TASKS if t in hist]
    shares = {t: [100 * hist[t].get(str(k), 0) / sum(hist[t].values()) for k in range(9)] for t in series}
    n = {t: sum(hist[t].values()) for t in series}
    fig = figure()
    heading(fig, "The base 9B already solves most audited problems",
            "Correct answers out of 8 samples per problem; audit of 20 problems per template ("
            + ", ".join(f"{n[t]:,} {TASK_NAME[t].split()[0]}" for t in series) + ")")
    legend(fig, [(TASK_NAME[t], TASK_COLOR[t], "box") for t in series])
    ax = axes(fig, (0.08, 0.12, 0.89, 0.62))
    ax.set_xlim(-0.6, 8.6)
    ax.set_ylim(0, 85)
    ax.set_xticks(range(9), [f"{k}/8" for k in range(9)])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    px, py = per_unit(ax)
    thick, gap = 20 * PX / px, 2 * PX / px
    for i, t in enumerate(series):
        off = (i - (len(series) - 1) / 2) * (thick + gap)
        for k in range(9):
            bar(ax, k + off, 0, shares[t][k], thick, TASK_COLOR[t])
        for k in (0, 8):
            ax.text(k + off, shares[t][k] + 1.2, f"{shares[t][k]:.0f}%", ha="center", va="bottom", fontsize=9.5,
                    color=INK2)
    ax.text(0, 14, "hard pool\n(0/8)", ha="center", va="bottom", fontsize=9.5, color=MUTED)
    ax.text(8 - 0.62, 78, "dropped: no learning signal (8/8) →", ha="right", va="center", fontsize=9.5, color=MUTED)
    return save(fig, "difficulty_audit")


def fig_templates(groups: dict[str, Any], drop_share: float) -> str:
    rows = sorted(groups.values(), key=lambda g: -g["share_all_correct"])
    fig = figure()
    drop = sum(g["decision"] == "drop" for g in rows)
    heading(fig, f"Template-level decisions: {drop} of {len(rows)} templates dropped as too easy",
            f"Share of each template's 20 audited problems the base solved 8/8; "
            f"a template with ≥ {drop_share:.0%} is dropped whole")
    legend(fig, [("dropped", ORANGE, "dot"), ("kept", BLUE, "dot")])
    ax = axes(fig, (0.08, 0.12, 0.89, 0.62))
    ax.set_xlim(-1.5, len(rows) + 0.5)
    ax.set_ylim(-4, 104)
    ax.set_xlabel("templates, ranked", fontsize=10.5, color=MUTED)
    ax.set_xticks([])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.axhline(100 * drop_share, color=INK2, lw=1 * PT, zorder=1)
    ax.text(len(rows) + 0.5, 100 * drop_share + 1.5, f"drop rule {drop_share:.0%}", ha="right", va="bottom",
            fontsize=9.5, color=INK2)
    for i, g in enumerate(rows):
        ax.plot(i, 100 * g["share_all_correct"], "o", ms=8 * PT, color=ORANGE if g["decision"] == "drop" else BLUE,
                mec=SURFACE, mew=2 * PT, zorder=3)
    return save(fig, "difficulty_templates")


def fig_sft(s: dict[str, Any]) -> str:
    tok = s["mixture"]["tokens_by_subset"]
    conv = s.get("conversations_by_subset", {})
    rows = sorted(tok.items(), key=lambda x: x[1])
    names = {"domain_reasoning": "Domain reasoning (teacher)", "tool_traj": "Tool trajectories (teacher)",
             "report": "Reports (teacher, judged)", "self_distill": "Self-distillation (base 9B)"}
    fig = figure(10, 4.6)
    heading(fig, f"SFT mix: {sum(conv.values()):,} conversations, {s['mixture']['total_tokens'] / 1e6:.1f}M tokens",
            f"Tokens by subset; self-distillation is {s['mixture']['self_distill_token_share']:.1%} of tokens "
            f"(target {s['mixture']['target_share']:.0%})")
    ax = axes(fig, (0.3, 0.1, 0.66, 0.66), grid="x")
    ax.spines["bottom"].set_visible(False)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0, max(v for _, v in rows) * 1.62)
    ax.set_yticks(range(len(rows)), [names.get(k, k) for k, _ in rows])
    ax.tick_params(axis="y", labelcolor=INK2)
    ax.xaxis.set_major_formatter(lambda v, _: compact(v))
    px, py = per_unit(ax)
    for i, (k, v) in enumerate(rows):
        bar(ax, i, 0, v, 22 * PX / py, BLUE, horizontal=True)
        ax.text(v + ax.get_xlim()[1] * 0.01, i, f"{v / 1e6:.1f}M tokens · {conv.get(k, 0):,} conversations",
                va="center", fontsize=10, color=INK2)
    return save(fig, "sft_mix")


def fig_replies(b: dict[str, Any]) -> str:
    edges, hist = b["hist_edges"], b["hist"]
    labels = [f"{lo // 1000}–{hi // 1000}K" if hi < 10 ** 8 else f"≥{lo // 1000}K" for lo, hi in zip(edges, edges[1:])]
    labels[0] = f"<{edges[1] // 1000}K"
    fig = figure()
    heading(fig, f"Base 9B replies in the difficulty audit: median {b['chars_p50'] / 1000:.0f}K characters",
            f"{b['replies']:,} replies (8 per problem); {b['closed_think'] / b['replies']:.0%} finish their reasoning "
            f"(close </think>); about 2.76 characters per token")
    ax = axes(fig, (0.08, 0.13, 0.89, 0.66))
    ax.set_xlim(-0.6, len(hist) - 0.4)
    ax.set_ylim(0, max(hist) * 1.15)
    ax.set_xticks(range(len(hist)), labels)
    ax.set_xlabel("reply length (characters)", fontsize=10.5, color=MUTED)
    ax.yaxis.set_major_formatter(lambda v, _: compact(v))
    px, py = per_unit(ax)
    for i, v in enumerate(hist):
        bar(ax, i, 0, v, 24 * PX / px, BLUE)
        ax.text(i, v + max(hist) * 0.015, thousands(v), ha="center", va="bottom", fontsize=9.5, color=INK2)
    return save(fig, "base_replies")


ROLE_ORDER = ["Teacher (Qwen3.8-27B)", "Judge (Gemma 4 26B-A4B)", "Base (Qwen3.5-9B)", "8-node RL and campaign",
              "Env checks and tests", "CPU data jobs"]


def role(j: dict[str, Any]) -> str:
    ep = j.get("endpoint", "")
    if j["name"] == "dpt-serve":
        if ep.startswith("teacher") or (not ep and j["partition"] == "gpu-h100"):
            return ROLE_ORDER[0]
        return ROLE_ORDER[1] if ep == "judge" else ROLE_ORDER[2] if ep == "base" else ROLE_ORDER[1]
    if j["name"] in ("dpt-rl", "dpt-campaign"):
        return ROLE_ORDER[3]
    if j["name"] in ("dpt-build", "dpt-check", "dpt-smoke", "dpt-parity"):
        return ROLE_ORDER[4]
    return ROLE_ORDER[5]


def when(s: str) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None


def ended(j: dict[str, Any], now: dt.datetime) -> dt.datetime | None:
    """A job's end; a running job (sacct: End Unknown) counts up to the collection time."""
    return when(j["end"]) or (now if j["state"] == "RUNNING" else None)


def fig_compute(c: dict[str, Any], now: dt.datetime) -> str:
    jobs = c["jobs"]
    t0 = min(when(j["submit"]) or now for j in jobs).replace(hour=0, minute=0)
    days = lambda t: (t - t0).total_seconds() / 86400   # noqa: E731
    end = days(now)
    fig = figure(10, 6.2)
    total = sum(j["su"] or 0 for j in jobs)
    heading(fig, f"Compute so far: {total:,.0f} SU over {sum(1 for j in jobs if j['elapsed_s'])} jobs",
            "Jobs by role (top) and SU used, estimated from Slurm's accounting (bottom)")
    legend(fig, [("ran", BLUE, "box"), ("failed", MUTED, "box"),
                 ("waiting in queue (8-node jobs)", BLUE_LIGHT, "box")], y=0.85)
    ax = axes(fig, (0.25, 0.4, 0.72, 0.38), grid="x")
    ax.spines["bottom"].set_visible(False)
    ax.set_ylim(len(ROLE_ORDER) - 0.5, -0.5)
    ax.set_xlim(0, end + 0.2)
    ax.set_yticks(range(len(ROLE_ORDER)), ROLE_ORDER)
    ax.tick_params(axis="y", labelcolor=INK2)
    ticks = range(0, int(end) + 1)
    ax.set_xticks(list(ticks), [""] * len(ticks))
    px, py = per_unit(ax)
    thick = 14 * PX / py
    for j in jobs:
        r = ROLE_ORDER.index(role(j))
        sub, st, en = when(j["submit"]), when(j["start"]), ended(j, now)
        if j["nodes"] >= 8 and sub and (st or j["state"] == "PENDING"):
            q_end = st or now
            ax.add_patch(Rectangle((days(sub), r - thick / 2), days(q_end) - days(sub), thick, fc=BLUE_LIGHT,
                                   ec="none", zorder=1))
        if st and en and j["elapsed_s"] > 0:
            w = max(days(en) - days(st), 2 * PX / px)        # very short jobs stay visible
            ax.add_patch(Rectangle((days(st), r - thick / 2), w, thick,
                                   fc=MUTED if j["state"] == "FAILED" else BLUE, ec="none", zorder=2))
    ax2 = axes(fig, (0.25, 0.1, 0.72, 0.24), grid="y")
    ax2.set_xlim(0, end + 0.2)
    hours = [i / 24 for i in range(int(end * 24) + 2)]
    cum, acc = [], 0.0
    for h0, h1 in zip(hours, hours[1:]):
        for j in jobs:
            st, en = when(j["start"]), ended(j, now)
            if st and en and j["su"]:
                a, b = max(days(st), h0), min(days(en), h1)
                if b > a:
                    acc += j["su"] * (b - a) / max(days(en) - days(st), 1e-9)
        cum.append(acc)
    ax2.plot(hours[1:], cum, color=BLUE, lw=2 * PT, solid_capstyle="round")
    ax2.plot(hours[len(cum)], cum[-1], "o", ms=8 * PT, color=BLUE, mec=SURFACE, mew=2 * PT)
    ax2.text(hours[len(cum)] - 0.1, cum[-1], f"{cum[-1]:,.0f} SU ", ha="right", va="bottom", fontsize=10, color=INK2)
    ax2.set_ylim(0, max(cum) * 1.25 if cum and max(cum) else 1)
    ax2.set_xticks(list(ticks), [(t0 + dt.timedelta(days=d)).strftime("%b %d") for d in ticks])
    ax2.set_ylabel("SU used", fontsize=10.5, color=MUTED)
    return save(fig, "compute")


STAGE_COLOR = {"base": MUTED, "sft": BLUE, "rl": ORANGE}       # de-emphasised base, then the trained models


def stage_of(name: str) -> str:
    return "rl" if name.startswith("rl") else "sft" if name.startswith("sft") else "base"


def fig_pass_at_k(runs: dict[str, Any]) -> str | None:
    curves = {n: r["sets"]["d_num"]["pass_at_k"] for n, r in runs.items()
              if n.endswith("_k64") and (r["sets"].get("d_num") or {}).get("pass_at_k")}
    if not curves:
        return None
    fig = figure()
    heading(fig, "D-Num pass@k: how often at least one of k samples is right",
            "64 samples per item; RL can only sharpen what some sample already gets right")
    legend(fig, [(n.removesuffix("_k64"), STAGE_COLOR[stage_of(n)], "line") for n in curves])
    ax = axes(fig, (0.08, 0.12, 0.8, 0.62))
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ks = sorted({int(k) for c in curves.values() for k in c})
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.minorticks_off()
    ends = sorted(c[str(max(int(k) for k in c))] for c in curves.values())
    apart = all(b - a >= 4 for a, b in zip(ends, ends[1:]))       # converging end labels: legend only
    for n, c in curves.items():
        xs = sorted(int(k) for k in c)
        ys = [c[str(k)] for k in xs]
        col = STAGE_COLOR[stage_of(n)]
        ax.plot(xs, ys, color=col, lw=2 * PT, solid_capstyle="round", zorder=3)
        ax.plot(xs[-1], ys[-1], "o", ms=8 * PT, color=col, mec=SURFACE, mew=2 * PT, zorder=4)
        if apart:
            ax.text(xs[-1] * 1.08, ys[-1], f"{n.removesuffix('_k64')} {ys[-1]:.0f}%", va="center", fontsize=10.5,
                    color=INK2, clip_on=False)
    return save(fig, "pass_at_k")


SET_NAME = {"d_num": "D-Num", "d_fresh": "D-Fresh", "d_code": "D-Code", "d_num_hard": "D-Num-hard",
            "gpqa_diamond": "GPQA Diamond", "mmlu_pro_2k": "MMLU-Pro (2K)", "ifeval": "IFEval",
            "livecodebench_post_2025_01": "LiveCodeBench (2025+)", "tool_calling": "Tool calling"}
FAIL_COLOR = {"thinking never closed": "#e87ba4", "answer format": "#008300", "wrong answer": "#6250d6"}  # slots 5–7


def fig_scorecard(main: dict[str, Any]) -> str | None:
    """Every scored set with its 95% CI, one dot per model (D-Report is a win rate against the base: not here)."""
    order = ["d_num", "d_num_hard", "d_fresh", "d_code", "gpqa_diamond", "mmlu_pro_2k", "ifeval",
             "livecodebench_post_2025_01", "tool_calling"]
    models = by_stage(main)
    rows = [k for k in order if any(k in main[m]["sets"] or k in main[m]["general"] for m in models)]
    if not rows:
        return None
    fig = figure(10, 0.9 + 0.48 * len(rows) + 1.3)
    one = len(models) == 1
    heading(fig, "Scores with 95% confidence intervals",
            (f"{models[0]}; " if one else "") + "D-Report is a win rate against the base and is reported separately")
    h = fig.get_figheight()
    if not one:
        legend(fig, [(m, STAGE_COLOR[stage_of(m)], "dot") for m in models], y=1 - 0.72 / h)
    top = 1 - (1.0 if one else 1.25) / h
    ax = axes(fig, (0.3, 0.07, 0.6, top - 0.07), grid="x")
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_yticks(range(len(rows)), [SET_NAME.get(k, k) for k in rows])
    ax.tick_params(axis="y", labelcolor=INK2)
    for j, m in enumerate(models):
        off = 0 if one else (j - (len(models) - 1) / 2) * 0.22
        col = STAGE_COLOR[stage_of(m)]
        for i, k in enumerate(rows):
            v = main[m]["sets"].get(k) or main[m]["general"].get(k)
            if not v or v.get("score") is None:
                continue
            hw = (v.get("ci") or {}).get("halfwidth") or 0
            ax.plot([v["score"] - hw, v["score"] + hw], [i + off] * 2, color=col, lw=2 * PT, solid_capstyle="round",
                    zorder=3)
            ax.plot(v["score"], i + off, "o", ms=8 * PT, color=col, mec=SURFACE, mew=2 * PT, zorder=4)
            if one:
                ax.text(v["score"] + hw + 1.5, i, f"{v['score']:.1f} ± {hw:.1f}", va="center", fontsize=10,
                        color=INK2)
    return save(fig, "scorecard")


def fig_failures(main: dict[str, Any]) -> str | None:
    """The failed samples of each numeric/code set, split by grader reason (share of all samples)."""
    models = by_stage(main)
    rows = []
    for k in ("d_num", "d_num_hard", "d_fresh", "d_code"):
        for m in models:
            rs = (main[m]["sets"].get(k) or {}).get("reasons")
            if rs:
                total = sum(rs.values())
                shares = {h: 100 * sum(c for r, c in rs.items() if r in keys) / total for h, keys in OUTCOME[1:]}
                shares["other"] = max(0.0, 100 - 100 * rs.get("ok", 0) / total - sum(shares.values()))
                rows.append((SET_NAME[k] + ("" if len(models) == 1 else f" · {m}"), shares))
    if not rows:
        return None
    fig = figure(10, 1.0 + 0.62 * len(rows) + 1.4)
    heading(fig, "Where the failures go", "Failed samples by grader reason, as a share of all samples")
    keys = list(FAIL_COLOR) + ["other"]
    hgt = fig.get_figheight()
    legend(fig, [(h, FAIL_COLOR.get(h, AXIS), "box") for h in keys], y=1 - 0.72 / hgt)
    ax = axes(fig, (0.22, 0.08, 0.72, 1 - 1.3 / hgt - 0.08), grid="x")
    ax.spines["bottom"].set_visible(False)
    xmax = max(sum(sh.values()) for _, sh in rows) * 1.22
    ax.set_xlim(0, xmax)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_yticks(range(len(rows)), [r for r, _ in rows])
    ax.tick_params(axis="y", labelcolor=INK2)
    px, py = per_unit(ax)
    gap, thick = 2 * PX / px, 20 * PX / py
    for i, (_, sh) in enumerate(rows):
        x = 0.0
        segs = [(h, sh[h]) for h in keys if sh[h] > 0]
        for n, (h, w) in enumerate(segs):
            last = n == len(segs) - 1
            bar(ax, i, x, max(w - (0 if last else gap), 0), thick, FAIL_COLOR.get(h, AXIS), horizontal=True,
                rounded=last)
            if w >= 2.2:
                ax.text(x + w / 2, i, f"{w:.1f}", ha="center", va="center", fontsize=9.5,
                        color=SURFACE if h != "other" else INK2, zorder=5)
            x += w
        ax.text(x + xmax * 0.012, i, f"{x:.1f}% fail", va="center", fontsize=10, color=INK2)
    return save(fig, "failure_modes")


def fig_noise(runs: dict[str, Any]) -> str | None:
    seeds = sorted((int(n.rsplit("_s", 1)[1]), r["sets"]["d_num"]) for n, r in runs.items()
                   if n.startswith("noise/") and "d_num" in r["sets"])
    if len(seeds) < 3:
        return None
    ys = [v["score"] for _, v in seeds]
    mean = sum(ys) / len(ys)
    sd = (sum((y - mean) ** 2 for y in ys) / (len(ys) - 1)) ** 0.5
    hw = sum((v.get("ci") or {}).get("halfwidth", 0) for _, v in seeds) / len(seeds)
    fig = figure(10, 4.6)
    heading(fig, f"D-Num across {len(seeds)} seeds: {min(ys):.1f}–{max(ys):.1f}, sd {sd:.2f}",
            f"Base, avg@8 per seed; the band is one run's mean 95% CI (±{hw:.2f}) around the mean of the seeds")
    ax = axes(fig, (0.08, 0.14, 0.86, 0.62))
    lo, hi = mean - 2.5 * hw, mean + 2.5 * hw
    ax.set_ylim(min(lo, min(ys) - 0.3), max(hi, max(ys) + 0.3))
    ax.set_xlim(-0.6, len(seeds) - 0.4)
    ax.axhspan(mean - hw, mean + hw, color="#cde2fb", lw=0, zorder=1)           # blue ramp step 100
    ax.axhline(mean, color=AXIS, lw=1 * PT, zorder=2)
    ax.text(-0.55, mean + hw + 0.05, f"mean {mean:.2f} ± one run's CI", ha="left", va="bottom", fontsize=10,
            color=INK2)
    ax.plot(range(len(seeds)), ys, "o", ms=8 * PT, color=MUTED, mec=SURFACE, mew=2 * PT, zorder=4)
    ax.set_xticks(range(len(seeds)), [f"s{s}" for s, _ in seeds])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1f}%")
    return save(fig, "noise_floor")


def fig_families(fs: dict[str, Any], set_name: str = "d_num_hard") -> str | None:
    """Mean score per problem family of one set, one bar group per model (base de-emphasised)."""
    per_model: dict[str, dict[str, Any]] = {}
    for run, sets in fs.items():
        if set_name in sets and not run.startswith("noise/") and not run.endswith("_k64"):
            per_model[run.split("/")[-1]] = sets[set_name]
    if not per_model:
        return None
    models = by_stage(per_model)
    fams = sorted(per_model[models[0]], key=lambda f: per_model[models[0]][f]["score"])
    fig = figure(10, 1.2 + 0.5 * len(fams) + 1.0)
    hgt = fig.get_figheight()
    heading(fig, f"{SET_NAME.get(set_name, set_name)} by problem family",
            (f"{models[0]}, " if len(models) == 1 else "") + "avg@8 per family, 20 items each")
    if len(models) > 1:
        legend(fig, [(m, STAGE_COLOR[stage_of(m)], "box") for m in models], y=1 - 0.72 / hgt)
    ax = axes(fig, (0.36, 0.08, 0.56, 1 - (1.25 if len(models) > 1 else 1.0) / hgt - 0.08), grid="x")
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(fams) - 0.4)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_yticks(range(len(fams)), [f.replace("_", " ") for f in fams])
    ax.tick_params(axis="y", labelcolor=INK2)
    px, py = per_unit(ax)
    thick = (22 if len(models) == 1 else 12) * PX / py
    for j, m in enumerate(models):
        off = 0 if len(models) == 1 else (j - (len(models) - 1) / 2) * (thick + 2 * PX / py)
        for i, f in enumerate(fams):
            v = per_model[m].get(f)
            if v is None:
                continue
            bar(ax, i + off, 0, v["score"], thick, STAGE_COLOR[stage_of(m)], horizontal=True)
            if len(models) == 1:
                ax.text(v["score"] + 1.2, i, f"{v['score']:.0f}%" + (f" (never solved: {v['never_solved']} of "
                        f"{v['items']})" if v["never_solved"] else ""), va="center", fontsize=10, color=INK2)
    return save(fig, f"families_{set_name}")


# Plan (README §6.1, rescheduled 2026-10-21) against what happened or is now expected (notes.md timeline).
TIMELINE = [  # (phase, planned start, planned end, actual or expected start, end)
    ("Environment, parity, smoke tests", "10-21", "10-23", "10-21", "10-23"),
    ("Eval sets drafted, checked, frozen", "10-21", "10-23 16:00", "10-21", "10-23 16:23"),
    ("Training data: problems, bank, SFT mix", "10-23", "10-27", "10-23", "10-27 02:17"),
    ("8-node campaign in the queue", "", "", "10-24 14:43", "10-27 11:16"),
    ("Baseline eval", "10-24", "10-25 23:59", "10-27 11:16", "10-27 23:00"),
    ("SFT, SFT eval, RL-entry gate", "10-26", "10-27 23:59", "10-27 23:00", "10-28 06:30"),
    ("RL (420 steps planned)", "10-28", "11-01 23:59", "10-28 06:30", "10-31 11:16"),
    ("Hard bank: verify, audit on SFT", "", "", "10-27 19:00", "10-29 12:00"),
    ("Final eval, second judge, blind review", "11-02", "11-06 23:59", "11-01", "11-05 23:59"),
    ("Second RL phase on the hard pool", "", "", "11-03", "11-05 12:00"),
    ("Buffer / talk preparation", "11-07", "11-09 23:59", "11-06", "11-09 23:59"),
]
TALK_DAY = "11-10"


def fig_timeline(now: dt.datetime) -> str:
    year = now.year
    t = lambda s: dt.datetime.strptime(f"{year}-{s}" + ("" if " " in s else " 00:00"), "%Y-%m-%d %H:%M")  # noqa: E731
    t0 = t("10-20")
    days = lambda x: (x - t0).total_seconds() / 86400   # noqa: E731
    fig = figure(10, 6.0)
    heading(fig, "Schedule: plan against what happened",
            f"Plan of 2026-10-21 (outline) and actual or expected (filled); as of {now:%b %d %H:%M}")
    legend(fig, [("done", BLUE, "box"), ("expected", BLUE_LIGHT, "box"), ("planned", AXIS, "outline")], y=0.86)
    ax = axes(fig, (0.33, 0.1, 0.63, 0.66), grid="x")
    ax.spines["bottom"].set_visible(False)
    n = len(TIMELINE)
    end = days(t(TALK_DAY)) + 1.5
    ax.set_xlim(0, end)
    ax.set_ylim(n - 0.4, -0.6)
    ax.set_yticks(range(n), [r[0] for r in TIMELINE])
    ax.tick_params(axis="y", labelcolor=INK2)
    ticks = list(range(1, int(end) + 1, 3))
    ax.set_xticks(ticks, [(t0 + dt.timedelta(days=d)).strftime("%b %d") for d in ticks])
    px, py = per_unit(ax)
    for i, (_, ps, pe, as_, ae) in enumerate(TIMELINE):
        if ps:
            ax.add_patch(Rectangle((days(t(ps)), i - 0.32), days(t(pe)) - days(t(ps)), 0.2, fc=SURFACE, ec=AXIS,
                                   lw=1 * PT, zorder=2))
        a, b = days(t(as_)), days(t(ae))
        cut = min(max(days(now), a), b)                # done up to now, expected after
        if cut > a:
            bar(ax, i + 0.06, a, cut - a, 0.3, BLUE, horizontal=True, rounded=cut >= b)
        if b > cut:
            bar(ax, i + 0.06, cut, b - cut, 0.3, BLUE_LIGHT, horizontal=True)
    for x, label in ((days(now), "now"), (days(t(TALK_DAY)), "talk")):
        ax.axvline(x, color=INK2 if label == "talk" else MUTED, lw=1 * PT, zorder=1)
        ax.text(x, -0.75, label, ha="center", va="bottom", fontsize=10, color=INK2)
    return save(fig, "timeline")


BUDGET_SETS = [("d_num", "D-Num"), ("d_fresh", "D-Fresh"), ("d_num_hard", "D-Num-hard")]


def pool_budget(budget: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Every run of a model (main run, noise seeds, pass@64) pooled per set: the same settings, more samples."""
    pooled: dict[tuple[str, str], dict[str, Any]] = {}
    for r in budget.values():
        for s, c in r["sets"].items():
            p = pooled.setdefault((s, r["model_name"]), {"budgets": c["budgets"], "within": [0] * len(c["budgets"]),
                                                         "n": 0, "correct": 0, "tokens": 0})
            if p["budgets"] != c["budgets"]:          # another length cap: not comparable
                continue
            p["within"] = [a + b for a, b in zip(p["within"], c["correct_within"])]
            p["n"], p["correct"], p["tokens"] = (p["n"] + c["n_samples"], p["correct"] + c["n_correct"],
                                                 p["tokens"] + c["tokens_total"])
    return pooled


def by_stage(models) -> list[str]:
    return sorted(models, key=lambda m: (("base", "sft", "rl").index(stage_of(m)), m))


def fig_budget(pooled: dict[tuple[str, str], dict[str, Any]]) -> str | None:
    sets = [(s, t) for s, t in BUDGET_SETS if any(k[0] == s for k in pooled)]
    if not sets:
        return None
    models = by_stage({m for _, m in pooled})
    fig = figure(10, 5.4)
    who = f"{models[0]}, " if len(models) == 1 else ""          # one series: no legend, the subtitle names it
    heading(fig, "Accuracy under a token budget",
            f"{who}every run pooled; a sample counts only if it is right within the budget")
    if len(models) > 1:
        legend(fig, [(m, STAGE_COLOR[stage_of(m)], "line") for m in models])
    w = 0.86 / len(sets)
    for i, (s, title) in enumerate(sets):
        ax = axes(fig, (0.07 + i * w, 0.11, w - 0.07, 0.6))
        ax.set_xlim(0, 34)
        ax.set_ylim(0, 100)
        ax.set_xticks([4, 8, 16, 24, 32], ["4K", "8K", "16K", "24K", "32K"])
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax.axvline(16, color=AXIS, lw=1 * PT, zorder=1)                     # RL's answer cap (16K since 10-29)
        ax.text(15.4, 3, "RL cap", ha="right", fontsize=9.5, color=MUTED)
        ax.set_title(title, loc="left", fontsize=11.5, color=INK, pad=10)
        ends = sorted(100 * p["within"][-1] / p["n"] for (ss, _), p in pooled.items() if ss == s and p["n"])
        apart = all(b - a >= 4 for a, b in zip(ends, ends[1:]))
        for m in models:
            p = pooled.get((s, m))
            if not p or not p["n"]:
                continue
            xs, ys = [b / 1024 for b in p["budgets"]], [100 * v / p["n"] for v in p["within"]]
            col = STAGE_COLOR[stage_of(m)]
            ax.plot(xs, ys, color=col, lw=2 * PT, solid_capstyle="round", zorder=3)
            ax.plot(xs[-1], ys[-1], "o", ms=8 * PT, color=col, mec=SURFACE, mew=2 * PT, zorder=4)
            if apart:
                ax.text(xs[-1] - 0.8, ys[-1] + 2.5, f"{ys[-1]:.0f}%", ha="right", va="bottom", fontsize=10,
                        color=INK2)
    return save(fig, "token_budget")


def budget_table(pooled: dict[tuple[str, str], dict[str, Any]]) -> str:
    rows = []
    for s, _ in BUDGET_SETS:
        for m in by_stage({mm for ss, mm in pooled if ss == s}):
            p = pooled[(s, m)]
            acc = dict(zip(p["budgets"], p["within"]))
            rows.append([s, m, f"{p['n']:,}"] + [f"{100 * acc[b] / p['n']:.1f}" if b in acc else "—"
                                                 for b in (8192, 16384, 24576, p["budgets"][-1])]
                        + [f"{p['tokens'] / max(1, p['correct']):,.0f}"])
    return table(["set", "model", "samples", "8K", "16K (RL cap)", "24K", "eval cap", "tokens per right answer"],
                 rows, left=2)


def eval_section(ev: dict[str, Any], budget: dict[str, Any], fs: dict[str, Any] | None = None) -> str:
    runs = ev.get("runs") or {}
    main: dict[str, Any] = {}
    # One column per model: a set scored in a run of its own (d_num_hard/<model>) joins that model's column
    for n, r in sorted(runs.items(), key=lambda kv: kv[0].count("/")):
        if n.startswith("noise/") or n.endswith("_k64"):
            continue
        col = main.setdefault(r.get("model_name") or n, {**r, "sets": {}, "general": {}})
        col["sets"].update(r.get("sets") or {})
        col["general"].update(r.get("general") or {})
    if not main:
        return "## 7. Evaluation (base → SFT → RL)\n\n_Pending: the campaign's baseline eval (job 3532660)._\n"
    md = ["## 7. Evaluation (base → SFT → RL)\n",
          "Scores are the harness's own (every model re-run, nothing copied); ± is the 95% CI half-width.\n"]

    def cell(v: dict[str, Any] | None, key: str = "score") -> str:
        if not v or v.get(key) is None:
            return "—"
        hw = (v.get("ci") or {}).get("halfwidth")
        return f"{v[key]:.1f}" + (f" ± {hw:.1f}" if hw is not None and key == "score" else "")

    names = sorted(main, key=lambda n: ("base", "sft", "rl").index(stage_of(n)))
    rows = []
    for s in ("d_num", "d_code", "d_report", "d_fresh", "d_num_hard"):
        if any(s in main[n]["sets"] for n in names):
            metric = next(main[n]["sets"][s].get("metric", "") for n in names if s in main[n]["sets"])
            rows.append([f"{s} ({metric})"] + [cell(main[n]["sets"].get(s)) for n in names])
    if any("refusal" in main[n]["sets"] for n in names):
        rows.append(["refusal rate (%)"] + [cell(main[n]["sets"].get("refusal"), "refusal_rate") for n in names])
        rows.append(["over-refusal rate (%)"]
                    + [cell(main[n]["sets"].get("refusal"), "over_refusal_rate") for n in names])
    for g in sorted({g for n in names for g in main[n]["general"]}):
        rows.append([f"general: {g}"] + [cell(main[n]["general"].get(g)) for n in names])
    rows.append(["mean output tokens"] + [f"{main[n]['mean_output_tokens']:,.0f}"
                                          if main[n].get("mean_output_tokens") else "—" for n in names])
    md.append(table(["set"] + names, rows) + "\n")
    sc = fig_scorecard(main)
    if sc:
        md.append(f"![scorecard]({sc})\n")
    trunc = [[s] + [f"{main[n]['sets'][s]['truncation_rate']:.1%}" if main[n]["sets"].get(s, {}).get(
        "truncation_rate") is not None else "—" for n in names] for s in ("d_num", "d_code", "d_fresh", "d_num_hard")
             if any(s in main[n]["sets"] for n in names)]
    md.append("Truncated at the length cap:\n\n" + table(["set"] + names, trunc) + "\n")
    pk_rows, out_rows = [], []
    for s in ("d_num", "d_fresh", "d_code", "d_num_hard"):
        for n in names:
            v = main[n]["sets"].get(s) or {}
            if v.get("pass_at_k"):
                pk_rows.append((s, n, v["pass_at_k"]))
            if v.get("reasons"):
                total = sum(v["reasons"].values())
                shares = [sum(c for r, c in v["reasons"].items() if r in keys) / total for _, keys in OUTCOME]
                out_rows.append([s, n] + [f"{x:.1%}" for x in shares] + [f"{1 - sum(shares):.1%}", f"{total:,}"])
    if pk_rows:
        ks = sorted({k for _, _, p in pk_rows for k in p}, key=int)
        md.append("pass@k from the same samples (unbiased estimator):\n\n"
                  + table(["set", "run"] + [f"pass@{k}" for k in ks],
                          [[s, n] + [f"{p[k]:.1f}" if k in p else "—" for k in ks] for s, n, p in pk_rows],
                          left=2) + "\n")
    if out_rows:
        md.append("Where the samples go (share of all samples; \"thinking never closed\" is almost always the "
                  "32K cap):\n\n" + table(["set", "run"] + [h for h, _ in OUTCOME] + ["other", "samples"], out_rows,
                                          left=2) + "\n")
        fm = fig_failures(main)
        if fm:
            md.append(f"![failure modes]({fm})\n")
    nf = ev.get("noise_floor")
    if isinstance(nf, dict):
        scal = {k: v for k, v in nf.items() if isinstance(v, (int, float))}
        md.append("D-Num noise floor (10 seeds): "
                  + ", ".join(f"{k} {v}" if isinstance(v, int) else f"{k} {v:.2f}" for k, v in scal.items()) + "\n")
    else:
        seeds = sorted(((int(n.rsplit("_s", 1)[1]), r["sets"]["d_num"]["score"]) for n, r in runs.items()
                        if n.startswith("noise/") and "d_num" in r["sets"]))
        if seeds:
            md.append(f"D-Num noise floor: {len(seeds)} of 10 seeds so far, avg@8 "
                      + ", ".join(f"s{s} {v:.1f}" for s, v in seeds)
                      + f" (spread {max(v for _, v in seeds) - min(v for _, v in seeds):.1f} points)\n")
    nz = fig_noise(runs)
    if nz:
        md.append(f"![noise floor]({nz})\n")
    pooled = pool_budget(budget)
    if pooled:
        md.append("Accuracy (%) under a token budget, every run of a model pooled. A sample counts only if it is "
                  "right within the budget, as if the length cap were that budget:\n\n" + budget_table(pooled) + "\n")
        md.append(f"![token budget]({fig_budget(pooled)})\n")
    ff = fig_families(fs or {})
    if ff:
        md.append(f"![D-Num-hard by family]({ff})\n")
    pk = fig_pass_at_k(runs)
    if pk:
        md.append(f"![pass@k]({pk})\n")
    return "\n".join(md)


# ---- markdown ---------------------------------------------------------------------------------------------
def table(header: list[str], rows: list[list[Any]], left: int = 1) -> str:
    """A Markdown table: the first ``left`` columns (labels, text) left-aligned, the rest (numbers) right-aligned."""
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" if i < left else "---:" for i in range(len(header))) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def render(d: dict[str, Any], drop_share: float) -> str:
    now = when(d["collected_at"]) or dt.datetime.now()
    p, g, s, b, c, e, k = (d.get(x) for x in ("pipeline", "groups", "sft_mix", "base_replies", "compute",
                                               "eval_sets", "contamination"))  # noqa: E127
    md = ["# Talk numbers: the Qwen3.5-9B run (running file)\n",
          f"Collected on the cluster {d['collected_at']}. Generated by `scripts/talk/render.py` from "
          f"`docs/talk/data/run.json`; do not edit by hand (notes go in `the run log`, copied in below). "
          f"Refresh on a login node: `cd $WORK/nanogpt_seis_qwen && source scripts/slurm/env.sh && python -m "
          f"dpt.eval.budget --eval-root $WORK/dpt/eval_results --out $WORK/dpt/talk/budget && python3 "
          f"scripts/talk/collect.py`; rsync the JSON, then `~/miniconda3/bin/python scripts/talk/render.py`. "
          f"Each figure has a vector PDF next to its PNG.\n"]
    pub = p["published"]
    bal = c.get("balance") or {}
    used = sum(j["su"] or 0 for j in c["jobs"])
    sets = e["sets"]
    md.append("## At a glance\n")
    md.append(table(["", ""], [
        ["RL bank", f"{sum(sum(v.values()) for v in pub.values()):,} problems from "
                    f"{p['stages'][0]['in']:,} generated: main {sum(pub['rl_main'].values()):,}, "
                    f"dev {sum(pub['rl_dev'].values()):,}, hard {sum(pub['rl_hard'].values())} (not trained on)"],
        ["SFT mix", f"{sum(s['conversations_by_subset'].values()):,} conversations, "
                    f"{s['mixture']['total_tokens'] / 1e6:.1f}M tokens"],
        ["Eval sets (frozen " + (e.get("frozen_at") or "?")[:10] + ")",
         ", ".join(f"{n} {v['records']}" for n, v in sets.items())],
        ["Contamination removed", f"RL {k['rl']['total_removed']:,} of {k['rl']['scanned']:,} "
                                  f"({k['rl']['rate']:.1%}); SFT {k['sft']['total_removed']:,} of "
                                  f"{k['sft']['scanned']:,} ({k['sft']['rate']:.1%}); canary hits "
                                  f"{k['rl']['canary']['hits'] + k['sft']['canary']['hits']}"],
        ["Compute so far", f"~{used:,.0f} SU over {sum(1 for j in c['jobs'] if j['elapsed_s'])} jobs since "
                           f"{c['since']}; balance {bal.get('available_su', '?'):,} SU (expires {bal.get('expires')})"
         if bal else f"~{used:,.0f} SU"],
        ["Campaign steps finished", ", ".join(f"{a} ({t})" for a, t in d["campaign"]["finished_steps"].items())
         or "none yet"],
    ], left=2) + "\n")
    md.append(f"![schedule]({fig_timeline(now)})\n")

    md.append("## 1. RL problem bank\n")
    md.append(f"![bank funnel]({fig_funnel(p)})\n")
    bt = p["by_task"]
    kept_all = sum(bt[t]["after_difficulty"] for t in TASKS) / sum(bt[t]["ingest"] for t in TASKS)
    md.append(table(["task"] + [n for _, n in FUNNEL] + ["kept"],
                    [[TASK_NAME[t]] + [thousands(bt[t][x]) for x, _ in FUNNEL]
                     + [f"{bt[t]['after_difficulty'] / bt[t]['ingest']:.0%}"] for t in TASKS]
                    + [["all"] + [thousands(sum(bt[t][x] for t in TASKS)) for x, _ in FUNNEL]
                       + [f"{kept_all:.0%}"]])
              + "\n")
    md.append(f"![retention by task]({fig_retention(p)})\n")
    md.append(f"![removals]({fig_removals(p)})\n")
    rb = p["removed_by_task"]
    rows = []
    for st in p["stages"]:
        for r, n in sorted((st.get("reasons") or {}).items(), key=lambda x: -x[1]):
            per = {t: rb.get(st["stage"], {}).get(t, {}).get(r, 0) for t in TASKS}
            rows.append([st["stage"], REASON.get(r, r), thousands(n)] + [thousands(per[t]) for t in TASKS])
    md.append(table(["stage", "reason", "removed"] + [TASK_NAME[t] for t in TASKS], rows, left=2) + "\n")
    ver = next(st for st in p["stages"] if st["stage"] == "verify")
    vg = g["verify"]
    md.append(f"**Verification audit.** {ver['groups']} templates/sources; 20 problems of each solved by the teacher "
              f"(4 samples) and judged; {len(ver['groups_dropped'])} rejected whole "
              f"({sum(1 for x in ver['groups_dropped'] if x.startswith('t1'))} T1 templates and "
              + ", ".join(x.split(':', 1)[1] for x in ver['groups_dropped'] if not x.startswith('t1'))
              + f"), {ver['expert_queue']} problems queued for a human check. "
              f"Accepted share per template: median "
              f"{sorted(v['accepted'] / v['audited'] for v in vg.values() if v.get('audited'))[len(vg) // 2]:.0%}.\n")
    md.append("Rejected templates: " + ", ".join(f"`{x}`" for x in ver["groups_dropped"]) + "\n")
    dif = next(st for st in p["stages"] if st["stage"] == "difficulty")
    md.append("### Difficulty audit\n")
    md.append(f"![difficulty audit]({fig_difficulty(p)})\n")
    hist = dif["histogram"]
    md.append(table(["task", "audited"] + [f"{i}/8" for i in range(9)],
                    [[TASK_NAME[t], sum(hist[t].values())] + [hist[t].get(str(i), 0) for i in range(9)]
                     for t in TASKS if t in hist]) + "\n")
    md.append(f"![template decisions]({fig_templates(g['difficulty'], drop_share)})\n")
    dg = g["difficulty"]
    md.append(f"Groups: {dif['groups']['keep']} kept, {dif['groups']['drop']} dropped (whole template solved 8/8 "
              f"by ≥ {drop_share:.0%} of its audited problems: {dif['reasons']['group_all_correct']:,} problems); "
              f"{dif['reasons']['all_correct']:,} more audited problems solved 8/8 were dropped individually. "
              f"Hard pool (0/8): {dif['hard_pool']} problems. Dropped templates: "
              + ", ".join(f"`{n}`" for n, v in dg.items() if v["decision"] == "drop") + "\n")
    md.append("### Published bank\n")
    md.append(table(["split"] + [TASK_NAME[t] for t in TASKS] + ["all"],
                    [[sp] + [thousands(pub[sp].get(t, 0)) for t in TASKS] + [thousands(sum(pub[sp].values()))]
                     for sp in ("rl_main", "rl_dev", "rl_hard")]) + "\n")

    md.append("## 2. SFT mix\n")
    md.append(f"![sft mix]({fig_sft(s)})\n")
    fun = {x["stage"]: x for x in s["funnel"]}
    pf = fun.get("problem_filter", {})
    md.append(f"Funnel: {pf.get('kept', 0) + pf.get('problem_not_verified', 0) + pf.get('rl_dev_problem', 0):,} "
              f"samples → {pf.get('kept', 0):,} after dropping unverified problems "
              f"({pf.get('problem_not_verified', 0):,}) and RL dev problems ({pf.get('rl_dev_problem', 0)}) → "
              f"{fun.get('quality', {}).get('count', 0):,} after quality → {fun.get('decontam', {}).get('count', 0):,} "
              f"after decontamination.\n")
    pad = s["padding"]
    md.append(table(["length bucket", "conversations", "tokens", "padding overhead"],
                    [[b_, v["n"], thousands(v["tokens"]), f"{v['overhead']:.1%}"]
                     for b_, v in pad["per_bucket"].items()])
              + f"\n\nBucketed padding overhead {pad['bucketed_overhead']:.1%} (unbucketed would be "
              f"{pad['unbucketed_overhead']:.0%}).\n")

    md.append("## 3. Eval sets\n")
    md.append(table(["set", "items", "built"], [[n, v["records"], v["created"]] for n, v in sets.items()]
                    + [[n, v, ""] for n, v in e["extra_files"].items()]) + "\n")
    md.append("## 4. Contamination\n")
    md.append(table(["", "scanned", "removed", "rate", "how"], left=1, rows=[
        ["RL bank", thousands(k["rl"]["scanned"]), thousands(k["rl"]["total_removed"]), f"{k['rl']['rate']:.2%}",
         f"n-gram / MinHash {k['rl']['ngram_minhash']:,}, semantic {k['rl']['semantic']:,} "
         f"(judge rejected {k['rl']['semantic_judge_rejected']:,} candidate matches), answer-level "
         f"{k['rl']['answer_level']}"],
        ["SFT mix", thousands(k["sft"]["scanned"]), thousands(k["sft"]["total_removed"]), f"{k['sft']['rate']:.2%}",
         f"prompt side {k['sft']['prompt_side']}, teacher verbatim {k['sft']['teacher_verbatim']}"]]) + "\n")
    md.append("## 5. The base model in the difficulty audit\n")
    md.append(f"![base replies]({fig_replies(b)})\n")
    md.append(f"{b['problems']:,} problems × 8 = {b['replies']:,} replies; characters p10 {b['chars_p10']:,}, "
              f"median {b['chars_p50']:,}, p90 {b['chars_p90']:,}, max {b['chars_max']:,}; "
              f"{b['replies'] - b['closed_think']:,} replies ({1 - b['closed_think'] / b['replies']:.0%}) never close "
              f"</think> (cut at the 24,576-token cap).\n")
    md.append("## 6. Compute and queue\n")
    md.append(f"![compute]({fig_compute(c, now)})\n")
    by_part: dict[str, list[float]] = {}
    for j in c["jobs"]:
        v = by_part.setdefault(j["partition"], [0, 0.0, 0.0])
        v[0] += 1
        v[1] += j["nodes"] * j["elapsed_s"] / 3600
        v[2] += j["su"] or 0
    md.append(table(["partition", "jobs", "node-hours", "SU"],
                    [[pt, v[0], f"{v[1]:,.1f}", f"{v[2]:,.1f}"]
                     for pt, v in sorted(by_part.items(), key=lambda x: -x[1][2])])
              + "\n")
    md.append(eval_section(d.get("eval") or {}, d.get("budget") or {}, d.get("family_scores")))
    for title in ("8. SFT training", "9. RL training", "10. Blind review"):
        md.append(f"## {title}\n\n_Pending: the campaign (job 3532660) produces it._\n")
    if NOTES.exists():
        md.append(NOTES.read_text())
    return "\n".join(md)


def main() -> None:
    global TALK
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", nargs="?", default=str(TALK / "data" / "run.json"))
    ap.add_argument("--drop-share", type=float, default=0.9, help="difficulty audit drop_share (pipeline.yaml)")
    ap.add_argument("--out", default=str(TALK), help="output directory (numbers.md, figures/)")
    args = ap.parse_args()
    TALK = Path(args.out)
    TALK.mkdir(parents=True, exist_ok=True)
    d = json.loads(Path(args.data).read_text())
    (TALK / "numbers.md").write_text(render(d, args.drop_share))
    print(f"wrote {TALK / 'numbers.md'} and {len(list((TALK / 'figures').glob('*.png')))} figures")


if __name__ == "__main__":
    main()
