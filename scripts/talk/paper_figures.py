"""Figures and number macros for the story report (docs/reports/latex), from the talk data.

    ~/miniconda3/bin/python scripts/talk/paper_figures.py      # → docs/reports/latex/figures/*.pdf, numbers.tex

Figures are sized for the page (6.4 in wide) and carry as little text as possible: no titles (the LaTeX captions
say what they show), set names and numbers only where a word is needed, and an English and a Chinese variant
(``*_en.pdf``, ``*_zh.pdf``) for the few figures that need words. ``numbers.tex`` defines every number the report
quotes, so the two languages cannot drift apart. Colours follow scripts/talk/render.py (the reference palette).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render as R  # noqa: E402  (sets the Agg backend and the palette)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

OUT = R.ROOT / "docs" / "reports" / "latex"
FIG = OUT / "figures"
W = 6.4                                       # inches: the text width of the report
CJK_TTC = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
CJK_OTF = Path(os.environ.get("DPT_CJK_FONT", Path.home() / ".cache" / "dpt" / "NotoSansCJKsc-Regular.otf"))
DEEP, MID, LIGHT = "#184f95", "#2a78d6", "#86b6ef"     # blue ramp 600 / 450 / 250: an ordered trio
BASE = R.INK2                                         # the base model's marks (SFT and RL will add colour)

T = {  # the few words the figures need
    "en": {"stages": ["generated", "quality", "dedup", "decontam.", "verified", "difficulty"],
           "fail": ["thinking never closed", "answer format", "wrong answer", "other"],
           "families": {"rupture_length_from_directivity": "rupture length (directivity)",
                        "gmpe_distance_for_pga": "distance (ground-motion model)",
                        "coulomb_stress_change": "Coulomb stress change",
                        "mw_from_spectral_plateau": "Mw (spectral plateau)",
                        "gradient_ray_range": "ray range (gradient)",
                        "gradient_ray_travel_time": "ray travel time (gradient)",
                        "moho_depth_from_ps_delay": "Moho depth (Ps)",
                        "crustal_vpvs_from_ps_and_ppps": "vp/vs (H-κ)"}},
    "zh": {"stages": ["生成", "质量", "去重", "去污染", "校验", "难度"],
           "fail": ["思考未闭合", "答案格式", "答错", "其他"],
           "families": {"rupture_length_from_directivity": "方向性 → 破裂长度",
                        "gmpe_distance_for_pga": "地震动模型 → 距离",
                        "coulomb_stress_change": "库仑应力变化",
                        "mw_from_spectral_plateau": "谱平台 → Mw",
                        "gradient_ray_range": "梯度介质射线距离",
                        "gradient_ray_travel_time": "梯度介质射线走时",
                        "moho_depth_from_ps_delay": "Ps → Moho 深度",
                        "crustal_vpvs_from_ps_and_ppps": "H-κ → vp/vs"}},
}


def cjk_font() -> Path:
    """matplotlib reads only a font collection's first face (Japanese): extract the Simplified Chinese one once."""
    if not CJK_OTF.exists():
        from fontTools.ttLib import TTCollection

        for face in TTCollection(CJK_TTC).fonts:
            if face["name"].getDebugName(1) == "Noto Sans CJK SC":
                CJK_OTF.parent.mkdir(parents=True, exist_ok=True)
                face.save(str(CJK_OTF))
                break
    return CJK_OTF


def lang(code: str) -> None:
    if code == "zh":
        font_manager.fontManager.addfont(str(cjk_font()))
        plt.rcParams["font.family"] = ["DejaVu Sans", "Noto Sans CJK SC"]
    else:
        plt.rcParams["font.family"] = ["DejaVu Sans"]
    plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})


PAGE = "white"                                      # the figures sit on a white page, not the chart surface


def fig(h: float, w: float = W):
    return plt.figure(figsize=(w, h), dpi=R.DPI, facecolor=PAGE)


def axes(f, rect, grid="y"):
    ax = R.axes(f, rect, grid=grid)
    ax.set_facecolor(PAGE)
    ax.tick_params(labelsize=8.5, pad=4)
    return ax


def save(f, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    f.savefig(FIG / f"{name}.pdf", facecolor=PAGE)
    f.savefig(FIG / f"{name}.png", dpi=150, facecolor=PAGE)
    plt.close(f)


def pooled(d: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return R.pool_budget(d.get("budget") or {})


def main_runs(d: dict[str, Any]) -> dict[str, Any]:
    main: dict[str, Any] = {}
    for n, r in sorted(d["eval"]["runs"].items(), key=lambda kv: kv[0].count("/")):
        if n.startswith("noise/") or n.endswith("_k64"):
            continue
        col = main.setdefault(r.get("model_name") or n, {**r, "sets": {}, "general": {}})
        col["sets"].update(r.get("sets") or {})
        col["general"].update(r.get("general") or {})
    return main


# ---- figures ------------------------------------------------------------------------------------------------
def budget(d):
    pb = pooled(d)
    f = fig(2.7)
    ax = axes(f, (0.08, 0.15, 0.7, 0.8))
    ax.set_xlim(0, 33)
    ax.set_ylim(0, 100)
    ax.set_xticks([4, 8, 16, 24, 32], ["4K", "8K", "16K", "24K", "32K"])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.axvline(16, color=R.AXIS, lw=1 * R.PT, ls=(0, (3, 3)), zorder=1)      # RL's answer cap (16K since 10-29)
    for s, col in (("d_num", DEEP), ("d_fresh", MID), ("d_num_hard", LIGHT)):
        p = pb.get((s, "base_medium"))
        if not p:
            continue
        xs, ys = [b / 1024 for b in p["budgets"]], [100 * v / p["n"] for v in p["within"]]
        ax.plot(xs, ys, color=col, lw=2.2 * R.PT, solid_capstyle="round", zorder=3)
        ax.plot(xs[-1], ys[-1], "o", ms=7 * R.PT, color=col, mec=PAGE, mew=2 * R.PT, zorder=4)
        ax.text(xs[-1] + 0.8, ys[-1], f"{R.SET_NAME[s]}  {ys[-1]:.0f}%", va="center", fontsize=8.5, color=R.INK2,
                clip_on=False)
    save(f, "budget")


def scorecard(d):
    m = main_runs(d)["base_medium"]
    order = ["d_num", "d_num_hard", "d_fresh", "d_code", "gpqa_diamond", "mmlu_pro_2k", "ifeval",
             "livecodebench_post_2025_01", "tool_calling"]
    rows = [k for k in order if k in m["sets"] or k in m["general"]]
    short = {"livecodebench_post_2025_01": "LiveCodeBench", "mmlu_pro_2k": "MMLU-Pro", "gpqa_diamond": "GPQA ◆"}
    f = fig(0.28 * len(rows) + 0.5)
    ax = axes(f, (0.2, 0.12, 0.72, 0.85), grid="x")
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(0, 100)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_yticks(range(len(rows)), [short.get(k, R.SET_NAME.get(k, k)) for k in rows])
    ax.tick_params(axis="y", labelcolor=R.INK2)
    for i, k in enumerate(rows):
        v = m["sets"].get(k) or m["general"].get(k)
        hw = (v.get("ci") or {}).get("halfwidth") or 0
        ax.plot([v["score"] - hw, v["score"] + hw], [i, i], color=BASE, lw=2 * R.PT, solid_capstyle="round")
        ax.plot(v["score"], i, "o", ms=7 * R.PT, color=BASE, mec=PAGE, mew=1.5 * R.PT, zorder=4)
        ax.text(v["score"] + hw + 1.5, i, f"{v['score']:.1f}", va="center", fontsize=8, color=R.INK2)
    save(f, "scorecard")


def funnel(d, code):
    st = d["pipeline"]["stages"]
    counts = [st[0]["in"]] + [s["out"] for s in st[1:]]
    labels = T[code]["stages"]
    f = fig(2.2)
    ax = axes(f, (0.06, 0.16, 0.92, 0.8))
    ax.set_ylim(0, max(counts) * 1.18)
    ax.set_xlim(-0.6, len(counts) - 0.4)
    ax.set_xticks(range(len(counts)), labels)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}K")
    px, py = R.per_unit(ax)
    for i, n in enumerate(counts):
        R.bar(ax, i, 0, n, 34 * R.PX / px, MID if i == len(counts) - 1 else LIGHT)
        ax.text(i, n + max(counts) * 0.03, f"{n:,}", ha="center", fontsize=8, color=R.INK2)
    save(f, f"funnel_{code}")


def failures(d, code):
    m = main_runs(d)["base_medium"]
    rows = []
    for k in ("d_num", "d_fresh", "d_num_hard", "d_code"):
        rs = (m["sets"].get(k) or {}).get("reasons")
        if rs:
            total = sum(rs.values())
            sh = [100 * sum(c for r, c in rs.items() if r in keys) / total for _, keys in R.OUTCOME[1:]]
            rows.append((R.SET_NAME[k], sh + [max(0.0, 100 - 100 * rs.get("ok", 0) / total - sum(sh))]))
    cols = list(R.FAIL_COLOR.values()) + [R.AXIS]
    f = fig(0.38 * len(rows) + 0.75)
    ax = axes(f, (0.15, 0.2, 0.8, 0.62), grid="x")
    ax.spines["bottom"].set_visible(False)
    xmax = max(sum(sh) for _, sh in rows) * 1.15
    ax.set_xlim(0, xmax)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_yticks(range(len(rows)), [r for r, _ in rows])
    ax.tick_params(axis="y", labelcolor=R.INK2)
    px, py = R.per_unit(ax)
    gap, thick = 2 * R.PX / px, 16 * R.PX / py
    for i, (_, sh) in enumerate(rows):
        x = 0.0
        segs = [(c, w) for c, w in zip(cols, sh) if w > 0]
        for n, (c, w) in enumerate(segs):
            last = n == len(segs) - 1
            R.bar(ax, i, x, max(w - (0 if last else gap), 0), thick, c, horizontal=True, rounded=last)
            if w >= 3:
                ax.text(x + w / 2, i, f"{w:.0f}", ha="center", va="center", fontsize=7.5,
                        color=R.SURFACE if c != R.AXIS else R.INK2)
            x += w
        ax.text(x + xmax * 0.012, i, f"{x:.1f}%", va="center", fontsize=8, color=R.INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="none") for c in cols]
    f.legend(handles, T[code]["fail"], loc="upper left", bbox_to_anchor=(0.14, 0.99), ncol=4, frameon=False,
             fontsize=8, handlelength=1.0, handleheight=1.0, columnspacing=1.2, labelcolor=R.INK2)
    save(f, f"failures_{code}")


def families(d, code):
    fs = (d.get("family_scores") or {}).get("d_num_hard/base_medium", {}).get("d_num_hard")
    if not fs:
        return
    fams = sorted(fs, key=lambda k: fs[k]["score"])
    f = fig(0.27 * len(fams) + 0.4)
    ax = axes(f, (0.33, 0.06, 0.6, 0.92), grid="x")
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(fams) - 0.4)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_yticks(range(len(fams)), [T[code]["families"][k] for k in fams])
    ax.tick_params(axis="y", labelcolor=R.INK2)
    px, py = R.per_unit(ax)
    for i, k in enumerate(fams):
        v = fs[k]["score"]
        R.bar(ax, i, 0, v, 13 * R.PX / py, MID if v > 0 else R.AXIS, horizontal=True)
        ax.text(max(v, 0) + 1.2, i, f"{v:.0f}", va="center", fontsize=8, color=R.INK2)
    save(f, f"families_{code}")


def seeds(d):
    sm = (d.get("diagnostics") or {}).get("seed_minutes") or []
    if not sm:
        return
    f = fig(1.9)
    ax = axes(f, (0.08, 0.17, 0.9, 0.78))
    ax.set_xlim(-0.6, len(sm) - 0.4)
    ax.set_ylim(0, max(s["minutes"] for s in sm) * 1.2)
    ax.set_xticks(range(len(sm)), [f"s{s['seed']}" for s in sm])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}′")
    px, py = R.per_unit(ax)
    for i, s in enumerate(sm):
        c = R.MUTED if s["seed"] < 4 else (MID if s["seed"] < 8 else LIGHT)
        R.bar(ax, i, 0, s["minutes"], 22 * R.PX / px, c)
        ax.text(i, s["minutes"] + 0.8, f"{s['minutes']:.0f}", ha="center", fontsize=8, color=R.INK2)
    save(f, "seeds")


def cliff(d):
    fl = (d.get("diagnostics") or {}).get("k64_failures") or []
    if not fl:
        return
    f = fig(1.9)
    ax = axes(f, (0.08, 0.17, 0.9, 0.78))
    end = max(x["minutes"] for x in fl) + 5
    ax.set_xlim(0, end)
    counts = {}
    for x in fl:
        key = (int(x["minutes"]), x["kind"])
        counts[key] = counts.get(key, 0) + 1
    ax.set_ylim(0, max(counts.values()) * 1.15)
    ax.axvline(60, color=R.AXIS, lw=1 * R.PT, ls=(0, (3, 3)), zorder=1)
    px, py = R.per_unit(ax)
    for (minute, kind), n in counts.items():
        R.bar(ax, minute + 0.5, 0, n, 0.85, R.ORANGE if kind == "timeout" else R.MUTED)
    ticks = [t for t in (0, 30, 60, 90, 120) if t <= end]
    ax.set_xticks(ticks, [f"{t}′" if t else "0" for t in ticks])
    save(f, "cliff")


def receiver(d):
    rc = ((d.get("diagnostics") or {}).get("receiver") or {}).get("ratios") or []
    if not rc:
        return
    f = fig(2.6, W / 2)
    ax = axes(f, (0.22, 0.17, 0.74, 0.79), grid=None)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(R.AXIS)
    lo, hi = 0.98, 1.22
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.plot([lo, hi], [lo, hi], color=R.AXIS, lw=1 * R.PT, zorder=1)
    ax.axhline(1.0, color=R.GRID, lw=1 * R.PT, zorder=0)
    ax.plot([r["ray_path"] for r in rc], [r["model"] for r in rc], "o", ms=7 * R.PT, color=MID, mec=PAGE,
            mew=1.5 * R.PT, zorder=3)
    ax.set_xlabel(r"$H_{\mathrm{ray\ path}}\,/\,H$", fontsize=9, color=R.INK2)
    ax.set_ylabel(r"$H_{\mathrm{model}}\,/\,H$", fontsize=9, color=R.INK2)
    ax.set_aspect("equal")
    save(f, "receiver")


def noise(d):
    nf = d["eval"].get("noise_floor") or {}
    ys = nf.get("scores") or []
    if not ys:
        return
    mean = sum(ys) / len(ys)
    runs = d["eval"]["runs"]
    hw = sum(r["sets"]["d_num"]["ci"]["halfwidth"] for n, r in runs.items() if n.startswith("noise/")) / len(ys)
    f = fig(1.7)
    ax = axes(f, (0.09, 0.17, 0.89, 0.78))
    ax.set_xlim(-0.6, len(ys) - 0.4)
    ax.set_ylim(mean - 2.2 * hw, mean + 2.2 * hw)
    ax.axhspan(mean - hw, mean + hw, color="#cde2fb", lw=0, zorder=1)
    ax.axhline(mean, color=R.AXIS, lw=1 * R.PT, zorder=2)
    ax.plot(range(len(ys)), ys, "o", ms=7 * R.PT, color=BASE, mec=PAGE, mew=1.5 * R.PT, zorder=4)
    ax.set_xticks(range(len(ys)), [f"s{i}" for i in range(len(ys))])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    save(f, "noise")


def compute(d):
    import datetime as dt
    jobs = d["compute"]["jobs"]
    now = R.when(d["collected_at"])
    t0 = min(R.when(j["submit"]) or now for j in jobs).replace(hour=0, minute=0)
    days = lambda t: (t - t0).total_seconds() / 86400  # noqa: E731
    end = days(now)
    hours = [i / 24 for i in range(int(end * 24) + 2)]
    cum, acc = [], 0.0
    for h0, h1 in zip(hours, hours[1:]):
        for j in jobs:
            st, en = R.when(j["start"]), R.ended(j, now)
            if st and en and j["su"]:
                a, b = max(days(st), h0), min(days(en), h1)
                if b > a:
                    acc += j["su"] * (b - a) / max(days(en) - days(st), 1e-9)
        cum.append(acc)
    f = fig(1.8)
    ax = axes(f, (0.09, 0.17, 0.88, 0.78))
    ax.set_xlim(0, end + 0.2)
    ax.set_ylim(0, max(cum) * 1.2)
    ax.plot(hours[1:], cum, color=MID, lw=2 * R.PT, solid_capstyle="round")
    ax.plot(hours[len(cum)], cum[-1], "o", ms=7 * R.PT, color=MID, mec=PAGE, mew=1.5 * R.PT)
    ticks = list(range(0, int(end) + 1, 2))
    ax.set_xticks(ticks, [(t0 + dt.timedelta(days=x)).strftime("%m-%d") for x in ticks])
    save(f, "compute")


# ---- numbers -------------------------------------------------------------------------------------------------
def signed(x: float, fmt: str = "{:+.1f}") -> str:
    return fmt.format(x).replace("-", "−")


def trained_numbers(d, rl=None, jd=None, rf=None) -> dict[str, str]:
    """Macros for the trained models (prefix S = SFT, R = RL step 20), the RL dynamics (rl.json), the judge-drift
    audit (judge_drift.json) and the refusal breakdown (refusal.json). Macro names are letters only, so the prefixes
    stand in for the model."""
    runs = d["eval"]["runs"]
    fam = d.get("family_scores") or {}
    v: dict[str, str] = {}

    def general(r):
        g = r.get("general") or {}
        return sum(x["score"] for x in g.values()) / len(g)

    for m, k in (("base_medium", "B"), ("sft", "S"), ("rl_seed1", "R")):
        r, S, G = runs[m], runs[m]["sets"], runs[m]["general"]
        hard = runs[f"d_num_hard/{m}"]["sets"]["d_num_hard"]
        hf = fam.get(f"d_num_hard/{m}", {}).get("d_num_hard", {})
        v.update({
            f"{k}DNum": f"{S['d_num']['score']:.1f}", f"{k}DNumCI": f"{S['d_num']['ci']['halfwidth']:.1f}",
            f"{k}DNumTrunc": f"{100 * S['d_num']['truncation_rate']:.1f}",
            f"{k}DCode": f"{S['d_code']['score']:.1f}", f"{k}DCodeCI": f"{S['d_code']['ci']['halfwidth']:.1f}",
            f"{k}DFresh": f"{S['d_fresh']['score']:.1f}", f"{k}DFreshCI": f"{S['d_fresh']['ci']['halfwidth']:.1f}",
            f"{k}DHard": f"{hard['score']:.1f}", f"{k}DHardTrunc": f"{100 * hard['truncation_rate']:.0f}",
            f"{k}DHardPassEight": f"{hard['pass_at_k']['8']:.1f}",
            f"{k}HardRF": f"{hf.get('moho_depth_from_ps_delay', {}).get('score', 0):.0f}",
            f"{k}HardHK": f"{hf.get('crustal_vpvs_from_ps_and_ppps', {}).get('score', 0):.0f}",
            f"{k}GPQA": f"{G['gpqa_diamond']['score']:.1f}", f"{k}MMLU": f"{G['mmlu_pro_2k']['score']:.1f}",
            f"{k}IFEval": f"{G['ifeval']['score']:.1f}", f"{k}LCB": f"{G['livecodebench_post_2025_01']['score']:.1f}",
            f"{k}Tools": f"{G['tool_calling']['score']:.1f}", f"{k}General": f"{general(r):.1f}",
            f"{k}Refusal": f"{S['refusal']['refusal_rate']:.1f}",
            f"{k}OverRefusal": f"{S['refusal']['over_refusal_rate']:.1f}",
            f"{k}Tokens": f"{r['mean_output_tokens']:,.0f}",
        })
        if m != "base_medium":
            dr = S["d_report"]
            v[f"{k}DReport"] = f"{dr['score']:.1f}"
            v[f"{k}DReportCI"] = f"{dr['ci']['halfwidth']:.1f}"
            for role, name in (("primary", "Gemma"), ("second", "Luna")):
                bj = dr["by_judge"][role]
                v[f"{k}{name}"] = f"{bj['score']:.1f}"
                for o in ("win", "tie", "loss"):
                    v[f"{k}{name}{o.capitalize()}"] = str(bj["outcomes"][o])
    b, sf, r = (runs[m] for m in ("base_medium", "sft", "rl_seed1"))
    v["GeneralDelta"] = signed(general(r) - general(b))
    v["RefusalDelta"] = signed(r["sets"]["refusal"]["refusal_rate"] - b["sets"]["refusal"]["refusal_rate"])
    v["TokenRatio"] = f"{r['mean_output_tokens'] / b['mean_output_tokens']:.2f}"
    v["TokenCutSFT"] = f"{100 * (1 - r['mean_output_tokens'] / sf['mean_output_tokens']):.0f}"
    v["TokenCutBase"] = f"{100 * (1 - r['mean_output_tokens'] / b['mean_output_tokens']):.0f}"
    v["DNumGateDelta"] = signed(r["sets"]["d_num"]["score"] - b["sets"]["d_num"]["score"])
    v["DCodeGateDelta"] = signed(r["sets"]["d_code"]["score"] - b["sets"]["d_code"]["score"])
    if rl:
        m = rl["metrics"]
        dev = rl["dev"]["steps"]
        last = max(x["step"] for x in m)
        early = [x for x in m if 1 <= x["step"] <= 20]

        def med(key):
            import statistics

            return statistics.median(x[key] for x in early if isinstance(x.get(key), (int, float)))

        solved = [x["filter/t1_num/all_solved_group_frac"] for x in early
                  if isinstance(x.get("filter/t1_num/all_solved_group_frac"), (int, float))]
        v.update({
            "DevZero": f"{100 * dev['0']['t1_num']['acc']:.1f}", "DevTwenty": f"{100 * dev['20']['t1_num']['acc']:.1f}",
            "DevForty": f"{100 * dev['40']['t1_num']['acc']:.1f}",
            "DevLenZero": f"{dev['0']['t1_num']['len']['p50']:,.0f}",
            "DevLenTwenty": f"{dev['20']['t1_num']['len']['p50']:,.0f}",
            "DevLenForty": f"{dev['40']['t1_num']['len']['p50']:,.0f}",
            "DevWrongTwenty": str(dev["20"]["t1_num"]["reasons"].get("wrong_value", 0)),
            "DevWrongForty": str(dev["40"]["t1_num"]["reasons"].get("wrong_value", 0)),
            "PauseStep": str(last),
            "LenEnd": f"{next(x for x in m if x['step'] == last).get('task/t1_num/len_mean', 0):.0f}",
            "TThreeJoin": str(next(x["step"] for x in m if (x.get("task/t3_report/n_samples") or 0) > 0)),
            "TFourPromptShare": f"{100 * med('task/t4_general/prompt_share'):.0f}",
            "TFourGradShare": f"{100 * med('task/t4_general/grad_share'):.0f}",
            "SolvedLo": f"{100 * min(solved):.0f}", "SolvedHi": f"{100 * max(solved):.0f}",
            "TFourLenOk": f"{med('task/t4_general/len_correct'):,.0f}",       # steps 1-20: right vs wrong answers
            "TFourLenBad": f"{med('task/t4_general/len_incorrect'):,.0f}",
        })
    if jd:
        st = {x["step"]: x for x in jd["steps"]}
        ch = next(c for c in jd["changes"] if c["from"] == min(st) and c["to"] == max(st))
        v.update({
            "DriftGemmaZero": f"{st[0]['gemma']['point']:.1f}", "DriftLunaZero": f"{st[0]['luna']['point']:.1f}",
            "DriftGapZero": f"{st[0]['gap']['point']:.1f}", "DriftGapForty": f"{st[max(st)]['gap']['point']:.1f}",
            "DriftGapTwenty": f"{st[20]['gap']['point']:.1f}" if 20 in st else "–",
            "DriftPrompts": str(st[0]["prompts"]),
            "DriftGemmaChange": signed(ch["gemma"]["point"]), "DriftGemmaLo": signed(ch["gemma"]["lo"]),
            "DriftGemmaHi": signed(ch["gemma"]["hi"]), "DriftLunaChange": signed(ch["luna"]["point"]),
            "DriftGapChange": signed(ch["gap"]["point"]), "DriftGapLo": signed(ch["gap"]["lo"]),
            "DriftGapHi": signed(ch["gap"]["hi"]),
            "DriftSpearmanLo": f"{min(x['spearman'] for x in jd['steps']):.2f}",
            "DriftSpearmanHi": f"{max(x['spearman'] for x in jd['steps']):.2f}",
        })
    if rf:
        def src(run, s, key):
            rows = [x for k, x in rf["groups"][run].items() if k.startswith(s + "/")]
            return 100 * sum(x[key] for x in rows) / sum(x["n"] for x in rows)

        b, r = rf["runs"]["base_medium"]["should_refuse"], rf["runs"]["rl_seed1"]["should_refuse"]
        v.update({
            "RefBroadB": f"{b['broad']:.1f}", "RefBroadR": f"{r['broad']:.1f}",
            "RefBroadDelta": signed(r["broad"] - b["broad"]),
            "RefJbbBroadB": f"{src('base_medium', 'jbb', 'broad'):.0f}",
            "RefJbbBroadR": f"{src('rl_seed1', 'jbb', 'broad'):.0f}",
            "RefXsStrictB": f"{src('base_medium', 'xstest', 'strict'):.1f}",
            "RefXsStrictR": f"{src('rl_seed1', 'xstest', 'strict'):.1f}",
            "RefUnclear": str(rf["transitions"]["rl_seed1"]["lost"].get("neither", 0)),
        })
    return v


def numbers(d, rl=None, jd=None, rf=None) -> str:
    m = main_runs(d)["base_medium"]
    S, G = m["sets"], m["general"]
    pb = pooled(d)
    nf = d["eval"]["noise_floor"] or {}
    st = d["pipeline"]["stages"]
    pub = d["pipeline"]["published"]
    k = d["contamination"]
    sm = d["sft_mix"]
    c = d["compute"]
    diag = d.get("diagnostics") or {}
    rc = diag.get("receiver") or {}

    def bud(s, b):
        p = pb[(s, "base_medium")]
        return 100 * dict(zip(p["budgets"], p["within"]))[b] / p["n"]

    def per_right(s):
        p = pb[(s, "base_medium")]
        return p["tokens"] / p["correct"] / 1000

    def fails(s):
        rs = S[s]["reasons"]
        tot = sum(rs.values())
        return [100 * sum(n for r, n in rs.items() if r in keys) / tot for _, keys in R.OUTCOME[1:]]

    def mean_minutes(lo, hi):
        ms = [s["minutes"] for s in seeds if lo <= s["seed"] < hi]
        return sum(ms) / max(1, len(ms))

    fam = (d.get("family_scores") or {}).get("d_num_hard/base_medium", {}).get("d_num_hard", {})
    seeds = diag.get("seed_minutes") or []
    v = {
        "DNum": f"{S['d_num']['score']:.1f}", "DNumCI": f"{S['d_num']['ci']['halfwidth']:.1f}",
        "DNumPassEight": f"{S['d_num']['pass_at_k']['8']:.1f}",
        "DFresh": f"{S['d_fresh']['score']:.1f}", "DFreshCI": f"{S['d_fresh']['ci']['halfwidth']:.1f}",
        "DCode": f"{S['d_code']['score']:.1f}", "DCodeCI": f"{S['d_code']['ci']['halfwidth']:.1f}",
        "DHard": f"{S['d_num_hard']['score']:.1f}", "DHardCI": f"{S['d_num_hard']['ci']['halfwidth']:.1f}",
        "DHardPassEight": f"{S['d_num_hard']['pass_at_k']['8']:.1f}",
        "DHardTrunc": f"{100 * S['d_num_hard']['truncation_rate']:.0f}",
        "GPQA": f"{G['gpqa_diamond']['score']:.1f}", "MMLU": f"{G['mmlu_pro_2k']['score']:.1f}",
        "IFEval": f"{G['ifeval']['score']:.1f}", "LCB": f"{G['livecodebench_post_2025_01']['score']:.1f}",
        "Tools": f"{G['tool_calling']['score']:.1f}",
        "BudEight": f"{bud('d_num', 8192):.0f}", "BudSixteen": f"{bud('d_num', 16384):.0f}",
        "BudRL": f"{bud('d_num', 24576):.0f}", "HardBudRL": f"{bud('d_num_hard', 24576):.0f}",
        "TokPerRight": f"{per_right('d_num'):.1f}", "HardTokPerRight": f"{per_right('d_num_hard'):.1f}",
        "FailUnclosed": f"{fails('d_num')[0]:.1f}", "FailFormat": f"{fails('d_num')[1]:.1f}",
        "FailWrong": f"{fails('d_num')[2]:.1f}",
        "NoiseSD": f"{nf.get('std', 0):.2f}", "NoiseMeanDiff": f"{nf.get('mean_pairwise_abs_diff', 0):.2f}",
        "NoiseMaxDiff": f"{nf.get('max_pairwise_abs_diff', 0):.2f}",
        "Generated": f"{st[0]['in']:,}", "Published": f"{st[-1]['out']:,}",
        "Removed": f"{st[0]['in'] - st[-1]['out']:,}",
        "TOneMixed": "53", "TOneSolved": "46",
        "Templates": "82", "TemplatesAll": "102",
        "SFTConv": f"{sum(sm['conversations_by_subset'].values()):,}",
        "SFTTokens": f"{sm['mixture']['total_tokens'] / 1e6:.1f}",
        "ContamRL": f"{100 * k['rl']['rate']:.1f}", "ContamSFT": f"{100 * k['sft']['rate']:.1f}",
        "SU": f"{sum(j['su'] or 0 for j in c['jobs']):,.0f}",
        "SULeft": f"{(c.get('balance') or {}).get('available_su', 0):,}",
        "HardFamilies": "8", "HardItems": "160", "HardTrain": "7,200",
        "MohoFinished": str(rc.get("finished", 0)), "MohoSamples": str(rc.get("samples", 0)),
        "SeedBefore": f"{mean_minutes(0, 4):.0f}", "SeedAfter": f"{mean_minutes(4, 8):.0f}",
        "CliffCount": str(sum(1 for x in diag.get("k64_failures") or [] if 59.5 <= x["minutes"] < 61)),
        "HardRF": f"{fam.get('moho_depth_from_ps_delay', {}).get('score', 0):.0f}",
        "Published_main": f"{sum(pub['rl_main'].values()):,}",
    }
    v.update(trained_numbers(d, rl, jd, rf))
    lines = ["% Generated by scripts/talk/paper_figures.py from docs/talk/data/run.json; do not edit.",
             f"% Collected on the cluster {d['collected_at']}."]
    lines += [f"\\newcommand{{\\N{key.replace('_', '')}}}{{{val}}}" for key, val in v.items()]
    return "\n".join(lines) + "\n"


def main() -> None:
    d = json.loads((R.TALK / "data" / "run.json").read_text())
    lang("en")
    for g in (budget, scorecard, seeds, cliff, receiver, noise, compute):
        g(d)
    for code in ("en", "zh"):
        lang(code)
        funnel(d, code)
        failures(d, code)
        families(d, code)
    OUT.mkdir(parents=True, exist_ok=True)
    extra = {}
    for name in ("rl", "judge_drift", "refusal"):
        f = R.TALK / "data" / f"{name}.json"
        extra[name] = json.loads(f.read_text()) if f.exists() else None
    (OUT / "numbers.tex").write_text(numbers(d, extra["rl"], extra["judge_drift"], extra["refusal"]))
    print(f"wrote {FIG} and {OUT / 'numbers.tex'}")


if __name__ == "__main__":
    main()
