"""The talk deck's technical figures: why each result came out as it did. English and Chinese variants, no titles
(the slide says what a figure shows), sized for a 13.33 × 7.5 in slide.

    ~/miniconda3/bin/python scripts/talk/deck_figures.py      # → docs/talk/figures/deck/<name>_{en,zh}.png|pdf, base/*.png

Inputs: docs/talk/data/run.json (scripts/talk/collect.py: eval scores, failure reasons, token budgets, pass@k,
judges, family scores) and docs/talk/data/rl.json (scripts/talk/collect_rl.py: RL metrics per step, the dev passes,
the SFT steps, host and GPU memory) and docs/talk/data/judge_drift.json (scripts/talk/judge_drift_ci.py: both
judges' scores of the T3 dev answers) and docs/talk/data/refusal.json (scripts/talk/refusal_breakdown.py). Aggregates
only. Colours follow scripts/talk/render.py: one colour per task,
one per model (base grey, SFT blue, RL orange), failure reasons in render's slots 5–7.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_figures as P  # noqa: E402  (CJK font)
import render as R  # noqa: E402  (Agg backend, palette, bar helper)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

DATA = R.TALK / "data"
OUT = R.TALK / "figures" / "deck"
LATEX_FIG = R.ROOT / "docs" / "reports" / "latex" / "figures"      # the story report reuses some of these
REPORT_FIGS = ("rl_collapse", "rl_signal", "judge_drift", "eval_failures", "frontier", "host_memory", "time_accounting")
# The story report wants figures with as little text as possible (its captions carry the words): these labels are
# blanked in its renders, and "quiet" drops the data labels the slides print.
QUIET_KEYS = ("selected", "t3_joins", "paused", "cap", "ev_resume", "ev_mtpoff", "ev_save", "ev_pause", "kill",
              "drift_gap", "t3_train", "drift_gemma_end", "drift_luna_end")
PAGE = "#ffffff"
MODELS = ["base_medium", "sft", "rl_seed1"]
MODEL_COLOR = {"base_medium": R.INK2, "sft": R.BLUE, "rl_seed1": R.ORANGE}
STEP_COLOR = {"0": P.LIGHT, "20": P.MID, "40": P.DEEP}          # ordered: a light-to-dark blue ramp
WIN, TIE, LOSS = "#184f95", "#d9d7cf", "#c2410c"
PLANNED, INCIDENT, IDLE = R.BLUE, "#e87ba4", "#c3c2b7"
FAIL_KEYS = list(R.FAIL_COLOR)                                   # thinking never closed, answer format, wrong answer
FS, FS_SMALL = 16, 14                      # the figures show at ~80% on a slide: ~13 and ~11 pt

L = {
    "en": {
        "model": {"base_medium": "base", "sft": "SFT", "rl_seed1": "RL step 20"},
        "task": {"t1_num": "T1 numeric", "t2_code": "T2 code", "t3_report": "T3 report", "t4_general": "T4 general"},
        "set": {"d_num": "D-Num", "d_fresh": "D-Fresh", "d_code": "D-Code", "d_num_hard": "D-Num-hard"},
        "fail": FAIL_KEYS + ["other"],
        "step": "RL step", "train_acc": "training accuracy", "resp_len": "mean answer length (tokens)",
        "dev": "dev D-Num (16K cap)", "selected": "step 20: selected", "t3_joins": "T3 joins (step 27)",
        "paused": "alert pause", "pen": "penalty from 12K", "cap": "16K cap",
        "groups_t1": "T1 groups of 8 samples", "groups_t2": "T2 groups of 8 samples",
        "solved": "all right: no signal", "mixed": "mixed: signal", "failed": "all wrong: no signal",
        "grad": "share of the gradient", "t2t3": "T2 + T3",
        "kept": "informative samples", "of": "of 1,008 per step",
        "entropy": "token entropy", "gap": "train − rollout |log-ppl| gap", "pause_at": "pause at 5e-3",
        "fails_of": "failed samples (% of all)", "dev_t1": "T1 dev failures (% of 512 samples)",
        "t1_len": "T1 dev answer length (share of samples)", "tokens": "tokens", "step_n": "step {s}",
        "median": "median {v}", "budget": "token budget", "right_within": "right within the budget",
        "rl_cap": "RL cap", "mean_tokens": "mean output tokens", "score": "score",
        "pass_k": "pass@k (D-Num, 64 samples)", "k": "k samples", "win": "win", "tie": "tie", "loss": "loss",
        "judge": {"primary": "Gemma 4 (primary)", "second": "GPT-6 Luna (second)"},
        "winrate": "win rate {v:.1f}", "lm": "length-matched pairs: {n}",
        "drift_y": "mean rubric score of the same T3 dev answers (0–100)", "drift_gap": "gap {v:.1f}",
        "drift_gemma": "Gemma 4: the T3 reward", "drift_luna": "GPT-6 Luna: never in training",
        "drift_gemma_end": "Gemma 4", "drift_luna_end": "GPT-6 Luna", "t3_train": "T3 in training (from step {s})",
        "ref_jbb": "JailbreakBench: harmful requests (100)", "ref_xs": "XSTest: unsafe prompts worded like safe ones (200)",
        "ref_strict": "refused: gate's detector", "ref_broad": "+ broader lexicon (upper bound)",
        "in_use": "host memory in use (GiB)", "kill": "Ray kills at 95%", "total": "node total",
        "ev_resume": "22:45 resume, MTP drafting on:\nthree CPU weight copies → killed",
        "ev_mtpoff": "22:50 restart,\nMTP drafting off", "ev_save": "checkpoint saves", "ev_pause": "02:31 pause",
        "gpu_mib": "GPU memory, last stage (MiB)", "capacity": "40,437 MiB", "sft_run": "SFT (Oct 29)",
        "rl_run": "RL (Oct 29–30)", "crash50": "crash at step 50", "relaunch": "relaunch",
        "planned": "planned stages", "incident": "incidents", "idle": "waits and holds", "hours": "hours since start",
        "seg": ["baseline eval", "SFT blocked: hang, takeover, fixes", "SFT", "export", "SFT eval",
                "RL launch", "RL", "wait", "final eval", "held"],
        "sft_loss": "SFT loss", "mtp_loss": "MTP head loss", "sft_step": "SFT step",
    },
    "zh": {
        "model": {"base_medium": "基座", "sft": "SFT", "rl_seed1": "RL 第 20 步"},
        "task": {"t1_num": "T1 数值", "t2_code": "T2 代码", "t3_report": "T3 报告", "t4_general": "T4 通用"},
        "set": {"d_num": "D-Num", "d_fresh": "D-Fresh", "d_code": "D-Code", "d_num_hard": "D-Num-hard"},
        "fail": ["思考未闭合", "答案格式", "答错", "其他"],
        "step": "RL 步数", "train_acc": "训练准确率", "resp_len": "平均回答长度（token）",
        "dev": "开发集 D-Num（16K 上限）", "selected": "第 20 步：选中", "t3_joins": "T3 加入（第 27 步）",
        "paused": "警报暂停", "pen": "12K 起惩罚", "cap": "16K 上限",
        "groups_t1": "T1：每组 8 个样本", "groups_t2": "T2：每组 8 个样本",
        "solved": "全对：无信号", "mixed": "有对有错：有信号", "failed": "全错：无信号",
        "grad": "梯度占比", "t2t3": "T2 + T3",
        "kept": "有信号的样本", "of": "每步共 1,008",
        "entropy": "token 熵", "gap": "训练与 rollout 的 |log-ppl| 差", "pause_at": "5e-3 时暂停",
        "fails_of": "失败样本（占全部的 %）", "dev_t1": "T1 开发集失败（占 512 个样本的 %）",
        "t1_len": "T1 开发集回答长度（样本占比）", "tokens": "token", "step_n": "第 {s} 步",
        "median": "中位数 {v}", "budget": "token 预算", "right_within": "预算内答对",
        "rl_cap": "RL 上限", "mean_tokens": "平均输出 token", "score": "得分",
        "pass_k": "pass@k（D-Num，64 个样本）", "k": "样本数 k", "win": "胜", "tie": "平", "loss": "负",
        "judge": {"primary": "Gemma 4（主评审）", "second": "GPT-6 Luna（第二评审）"},
        "winrate": "胜率 {v:.1f}", "lm": "长度相近的样本对：{n}",
        "drift_y": "同一批 T3 开发集回答的平均评分（0–100）", "drift_gap": "差 {v:.1f}",
        "drift_gemma": "Gemma 4：T3 的奖励", "drift_luna": "GPT-6 Luna：从不参与训练",
        "drift_gemma_end": "Gemma 4", "drift_luna_end": "GPT-6 Luna", "t3_train": "T3 参与训练（第 {s} 步起）",
        "ref_jbb": "JailbreakBench：有害请求（100）", "ref_xs": "XSTest：措辞像安全提示的不安全提示（200）",
        "ref_strict": "拒绝：门槛的检测器", "ref_broad": "+ 更宽的词表（上限）",
        "in_use": "主机内存占用（GiB）", "kill": "Ray 在 95% 时杀进程", "total": "节点总量",
        "ev_resume": "22:45 恢复运行，MTP 草拟开启：\n三份 CPU 权重副本 → 被杀",
        "ev_mtpoff": "22:50 重启，\nMTP 草拟关闭", "ev_save": "保存检查点", "ev_pause": "02:31 暂停",
        "gpu_mib": "最后一段的 GPU 显存（MiB）", "capacity": "40,437 MiB", "sft_run": "SFT（10 月 29 日）",
        "rl_run": "RL（10 月 29–30 日）", "crash50": "第 50 步崩溃", "relaunch": "重新启动",
        "planned": "计划内阶段", "incident": "事件", "idle": "等待与保留", "hours": "开始后的小时数",
        "seg": ["基线评测", "SFT 受阻：卡住、接管、修复", "SFT", "导出", "SFT 评测",
                "RL 启动", "RL", "等待", "最终评测", "保留"],
        "sft_loss": "SFT 损失", "mtp_loss": "MTP 头损失", "sft_step": "SFT 步数",
    },
}


# ---- chrome ---------------------------------------------------------------------------------------------------
def fig(w: float, h: float):
    return plt.figure(figsize=(w, h), dpi=R.DPI, facecolor=PAGE)


def axes(f, rect, grid: str | None = "y"):
    ax = R.axes(f, rect, grid=grid)
    ax.set_facecolor(PAGE)
    ax.tick_params(labelsize=FS_SMALL, pad=5)
    ax.tick_params(which="minor", length=0)
    return ax


def label(ax, text: str, y: float = 1.03) -> None:
    """A panel's caption, top left, in ink."""
    ax.text(0, y, text, transform=ax.transAxes, fontsize=FS, color=R.INK, ha="left", va="bottom")


def legend(f, items, x: float, y: float, ncol: int | None = None) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="none") if kind == "box" else
               Line2D([0], [0], color=c, lw=2.4 * R.PT, ls=(0, (3, 2)) if kind == "dash" else "-") if kind in
               ("line", "dash") else Line2D([0], [0], color=c, lw=0, marker="o", ms=9 * R.PT, mec=c)
               for _, c, kind in items]
    f.legend(handles, [n for n, _, _ in items], loc="upper left", bbox_to_anchor=(x, y), ncol=ncol or len(items),
             frameon=False, fontsize=FS_SMALL, labelcolor=R.INK2, handlelength=1.2, handleheight=1.0,
             columnspacing=1.6)


def vline(ax, x: float, text: str = "", y: float = 0.98, ha: str = "left", color: str = R.MUTED) -> None:
    ax.axvline(x, color=R.AXIS, lw=1.2 * R.PT, ls=(0, (3, 3)), zorder=1)
    if text:
        ax.text(x + (0.6 if ha == "left" else -0.6), y, text, transform=ax.get_xaxis_transform(), fontsize=FS_SMALL,
                color=color, ha=ha, va="top")


def save(f, name: str, code: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    f.savefig(OUT / f"{name}_{code}.png", dpi=R.DPI, facecolor=PAGE)
    f.savefig(OUT / f"{name}_{code}.pdf", facecolor=PAGE)
    plt.close(f)


def series(rows: list[dict[str, Any]], key: str, scale: float = 1.0):
    xs, ys = [], []
    for r in rows:
        v = r.get(key)
        if isinstance(v, (int, float)) and v == v:
            xs.append(r["step"])
            ys.append(v * scale)
    return xs, ys


def line(ax, xs, ys, color, lw=2.2, **kw):
    ax.plot(xs, ys, color=color, lw=lw * R.PT, solid_capstyle="round", **kw)


def step_axis(ax, last: int = 56, labels: bool = True) -> None:
    ax.set_xlim(0, last + 1)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 56])
    if not labels:
        ax.tick_params(labelbottom=False)


def outcome_shares(reasons: dict[str, int]) -> list[float]:
    total = sum(reasons.values())
    sh = [100 * sum(c for r, c in reasons.items() if r in keys) / total for _, keys in R.OUTCOME[1:]]
    return sh + [max(0.0, 100 - 100 * reasons.get("ok", 0) / total - sum(sh))]


def hbars(ax, rows, cols, xmax, fmt="{:.1f}%") -> None:
    """Horizontal stacked bars; rows = [(label, [values…])]; the total is printed at the end."""
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(0, xmax)
    ax.set_ylim(len(rows) - 0.45, -0.55)
    ax.set_yticks(range(len(rows)), [r for r, _ in rows])
    ax.tick_params(axis="y", labelcolor=R.INK2, labelsize=FS)
    px, py = R.per_unit(ax)
    gap, thick = 2 * R.PX / px, 26 * R.PX / py
    for i, (_, vals) in enumerate(rows):
        x = 0.0
        segs = [(c, w) for c, w in zip(cols, vals) if w > 0]
        for n, (c, w) in enumerate(segs):
            last = n == len(segs) - 1
            R.bar(ax, i, x, max(w - (0 if last else gap), 0), thick, c, horizontal=True, rounded=last)
            if w >= xmax * 0.06:
                ax.text(x + w / 2, i, f"{w:.0f}", ha="center", va="center", fontsize=FS_SMALL,
                        color=PAGE if c not in (R.AXIS, TIE) else R.INK2, zorder=5)
            x += w
        ax.text(x + xmax * 0.012, i, fmt.format(x), va="center", fontsize=FS_SMALL, color=R.INK2)


# ---- RL dynamics -------------------------------------------------------------------------------------------------
def rl_collapse(rl, t):
    m, dev = rl["metrics"], rl["dev"]["steps"]
    f = fig(12.4, 5.6)
    a1 = axes(f, (0.075, 0.53, 0.9, 0.32))
    a2 = axes(f, (0.075, 0.1, 0.9, 0.34))
    for ax in (a1, a2):
        step_axis(ax, labels=ax is a2)
        vline(ax, 20, t["selected"] if ax is a1 else "", y=0.42)
        vline(ax, 27, t["t3_joins"] if ax is a1 else "", y=0.24)
        vline(ax, 56, t["paused"] if ax is a1 else "", ha="right", y=1.1)
    a1.set_ylim(0, 105)
    a1.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    for k in R.TASKS:
        xs, ys = series(m, f"task/{k}/acc", 100)
        if xs:
            line(a1, xs, ys, R.TASK_COLOR[k])
    ds = sorted((int(s), 100 * v["t1_num"]["acc"]) for s, v in dev.items())
    a1.plot([s for s, _ in ds], [v for _, v in ds], color=R.INK, lw=1.4 * R.PT, ls=(0, (2, 2)), zorder=4)
    a1.plot([s for s, _ in ds], [v for _, v in ds], "o", ms=9 * R.PT, mfc=PAGE, mec=R.INK, mew=2 * R.PT, zorder=5)
    for s, v in ds:
        if not t.get("quiet"):
            a1.text(s + 0.7, v - 4, f"{v:.1f}", fontsize=FS_SMALL, color=R.INK, va="top")
    label(a1, t["train_acc"])
    a2.set_yscale("log")
    a2.set_ylim(10, 30000)
    a2.set_yticks([10, 100, 1000, 10000], ["10", "100", "1K", "10K"])
    a2.axhline(12288, color=R.AXIS, lw=1.2 * R.PT, ls=(0, (1, 2)))
    a2.axhline(16384, color=R.MUTED, lw=1.2 * R.PT, ls=(0, (3, 2)))
    a2.text(0.6, 16384 * 1.12, t["cap"], fontsize=FS_SMALL, color=R.MUTED, va="bottom")
    for k in R.TASKS:
        xs, ys = series(m, f"task/{k}/len_mean")
        if xs:
            line(a2, xs, ys, R.TASK_COLOR[k])
    a2.set_xlabel(t["step"], fontsize=FS_SMALL, color=R.MUTED)
    label(a2, t["resp_len"])
    legend(f, [(t["task"][k], R.TASK_COLOR[k], "line") for k in R.TASKS] + [(t["dev"], R.INK, "dot")], 0.07, 0.995)
    return f


def groups_panel(ax, m, task, t):
    xs, solved, mixed, failed = [], [], [], []
    for r in m:
        a, z = r.get(f"filter/{task}/all_solved_group_frac"), r.get(f"filter/{task}/zero_adv_group_frac")
        if isinstance(a, (int, float)) and isinstance(z, (int, float)):
            xs.append(r["step"])
            solved.append(100 * a)
            failed.append(100 * max(0.0, z - a))
            mixed.append(100 * max(0.0, 1 - z))
    ax.stackplot(xs, solved, mixed, failed, colors=[R.BLUE_LIGHT, R.ORANGE, R.INK2], lw=0)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    step_axis(ax)
    ax.grid(False)
    vline(ax, 20)
    ax.set_xlabel(t["step"], fontsize=FS_SMALL, color=R.MUTED)


def rl_signal(rl, t):
    m = rl["metrics"]
    f = fig(12.4, 4.8)
    a1 = axes(f, (0.06, 0.15, 0.27, 0.62), grid=None)
    a2 = axes(f, (0.385, 0.15, 0.27, 0.62), grid=None)
    a3 = axes(f, (0.71, 0.15, 0.27, 0.62), grid=None)
    groups_panel(a1, m, "t1_num", t)
    label(a1, t["groups_t1"])
    groups_panel(a2, m, "t2_code", t)
    label(a2, t["groups_t2"])
    xs, t1 = series(m, "task/t1_num/grad_share", 100)
    xs4, t4 = series(m, "task/t4_general/grad_share", 100)
    common = sorted(set(xs) & set(xs4))
    g1 = [t1[xs.index(s)] for s in common]
    g4 = [t4[xs4.index(s)] for s in common]
    other = [max(0.0, 100 - a - b) for a, b in zip(g1, g4)]
    a3.stackplot(common, g1, other, g4, colors=[R.TASK_COLOR["t1_num"], R.AXIS, R.TASK_COLOR["t4_general"]], lw=0)
    a3.set_ylim(0, 100)
    a3.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    step_axis(a3)
    vline(a3, 20)
    a3.set_xlabel(t["step"], fontsize=FS_SMALL, color=R.MUTED)
    label(a3, t["grad"])
    legend(f, [(t["solved"], R.BLUE_LIGHT, "box"), (t["mixed"], R.ORANGE, "box"), (t["failed"], R.INK2, "box")],
           0.055, 0.995)
    legend(f, [(t["task"]["t1_num"], R.TASK_COLOR["t1_num"], "box"), (t["t2t3"], R.AXIS, "box"),
               (t["task"]["t4_general"], R.TASK_COLOR["t4_general"], "box")], 0.705, 0.995, ncol=2)
    return f


def rl_monitors(rl, t):
    m = rl["metrics"]
    f = fig(12.4, 4.4)
    a1 = axes(f, (0.06, 0.16, 0.26, 0.64))
    a2 = axes(f, (0.385, 0.16, 0.26, 0.64))
    a3 = axes(f, (0.715, 0.16, 0.26, 0.64))
    xs, ys = series(m, "filter/kept_samples")
    px_thick = 0.62
    a1.set_ylim(0, 1008)
    step_axis(a1)
    for x, y in zip(xs, ys):
        R.bar(a1, x, 0, y, px_thick, R.ORANGE)
    a1.axhline(1008, color=R.AXIS, lw=1 * R.PT)
    label(a1, t["kept"])
    a1.text(1, 1008 * 0.97, t["of"], fontsize=FS_SMALL, color=R.MUTED, va="top")
    for k in ("t1_num", "t4_general"):
        xs, ys = series(m, f"task/{k}/entropy")
        line(a2, xs, ys, R.TASK_COLOR[k])
    a2.set_ylim(0, None)
    step_axis(a2)
    label(a2, t["entropy"])
    xs, ys = series(m, "mismatch/log_ppl_abs_diff")
    a3.set_yscale("log")
    line(a3, xs, ys, R.INK)
    a3.axhline(5e-3, color=LOSS, lw=1.4 * R.PT, ls=(0, (3, 2)))
    a3.text(1, 5e-3 * 1.15, t["pause_at"], fontsize=FS_SMALL, color=LOSS, va="bottom")
    a3.set_ylim(2e-4, 3e-2)
    step_axis(a3)
    label(a3, t["gap"])
    for ax in (a1, a2, a3):
        vline(ax, 20)
        ax.set_xlabel(t["step"], fontsize=FS_SMALL, color=R.MUTED)
    legend(f, [(t["task"][k], R.TASK_COLOR[k], "line") for k in ("t1_num", "t4_general")], 0.38, 0.995)
    return f


def rl_dev_shift(rl, t):
    dev, bins = rl["dev"]["steps"], rl["dev"]["len_bins"]
    f = fig(12.4, 4.8)
    a1 = axes(f, (0.075, 0.14, 0.33, 0.62), grid="x")
    a2 = axes(f, (0.5, 0.14, 0.47, 0.62))
    groups = [("thinking never closed", {"no_think_end"}),
              ("answer format", {"missing_unit", "unit_parse_fail", "no_final_answer", "multi_answer"}),
              ("wrong answer", {"wrong_value"})]
    rows = []
    for s in ("0", "20", "40"):
        rs, n = dev[s]["t1_num"]["reasons"], dev[s]["t1_num"]["n"]
        vals = [100 * sum(c for r, c in rs.items() if r in keys) / n for _, keys in groups]
        rows.append((t["step_n"].format(s=s), vals + [max(0.0, 100 * (1 - dev[s]["t1_num"]["acc"]) - sum(vals))]))
    hbars(a1, rows, [R.FAIL_COLOR[g] for g, _ in groups] + [R.AXIS], 26)
    a1.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    label(a1, t["dev_t1"])
    edges = [bins[0] / 2 ** 0.5] + bins + [bins[-1] * 2 ** 0.5]
    centers = [(edges[i] * edges[i + 1]) ** 0.5 for i in range(len(edges) - 1)]
    a2.set_xscale("log")
    for s in ("0", "20", "40"):
        d = dev[s]["t1_num"]
        h = [a + b for a, b in zip(d["hist_correct"], d["hist_wrong"])]
        tot = sum(h)
        ys = [100 * v / tot for v in h]
        a2.plot(centers, ys, color=STEP_COLOR[s], lw=2.6 * R.PT, solid_capstyle="round", drawstyle="default")
        med = d["len"]["p50"]
        a2.plot([med], [max(ys) * 1.06], "v", ms=10 * R.PT, color=STEP_COLOR[s])
        a2.text(med, max(ys) * 1.12, t["step_n"].format(s=s) + f" · {med:,.0f}", fontsize=FS_SMALL,
                color=R.INK2, ha="center", va="bottom")
    a2.axvline(16384, color=R.MUTED, lw=1.2 * R.PT, ls=(0, (3, 2)))
    a2.text(16384 * 0.92, 0.98, t["cap"], transform=a2.get_xaxis_transform(), fontsize=FS_SMALL, color=R.MUTED,
            ha="right", va="top")
    a2.set_xticks([100, 300, 1000, 3000, 10000], ["100", "300", "1K", "3K", "10K"])
    a2.set_xlim(60, 24000)
    a2.set_ylim(0, a2.get_ylim()[1] * 1.22)
    a2.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    a2.set_xlabel(t["tokens"], fontsize=FS_SMALL, color=R.MUTED)
    label(a2, t["t1_len"])
    legend(f, [(n, R.FAIL_COLOR[k] if k in R.FAIL_COLOR else R.AXIS, "box")
               for n, k in zip(t["fail"], FAIL_KEYS + ["other"])], 0.07, 0.995)
    return f


# ---- evaluation ------------------------------------------------------------------------------------------------
def eval_failures(d, t):
    runs = d["eval"]["runs"]
    f = fig(12.4, 4.2)
    specs = [("d_num", {m: m for m in MODELS}), ("d_num_hard", {m: f"d_num_hard/{m}" for m in MODELS}),
             ("d_code", {m: m for m in MODELS})]
    for i, (s, src) in enumerate(specs):
        ax = axes(f, (0.12 + i * 0.315, 0.12, 0.19, 0.62), grid="x")
        rows = [(t["model"][m], outcome_shares(runs[src[m]]["sets"][s]["reasons"])) for m in MODELS]
        xmax = max(sum(v) for _, v in rows) * 1.32
        hbars(ax, rows, [R.FAIL_COLOR[k] for k in FAIL_KEYS] + [R.AXIS], xmax)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        label(ax, t["set"][s])
    legend(f, [(n, R.FAIL_COLOR[k] if k in R.FAIL_COLOR else R.AXIS, "box")
               for n, k in zip(t["fail"], FAIL_KEYS + ["other"])], 0.07, 0.995)
    return f


def budget3(d, t):
    pb = R.pool_budget(d.get("budget") or {})
    f = fig(12.4, 4.4)
    for i, s in enumerate(("d_num", "d_fresh", "d_num_hard")):
        ax = axes(f, (0.07 + i * 0.325, 0.15, 0.25, 0.64))
        ax.set_xlim(0, 33)
        ax.set_ylim(0, 100)
        ax.set_xticks([4, 8, 16, 24, 32], ["4K", "8K", "16K", "24K", "32K"])
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax.axvline(16, color=R.AXIS, lw=1.2 * R.PT, ls=(0, (3, 3)), zorder=1)
        if i == 0:
            ax.text(16.5, 4, t["rl_cap"], fontsize=FS_SMALL, color=R.MUTED)
        for m in MODELS:
            p = pb.get((s, m))
            if not p:
                continue
            xs, ys = [b / 1024 for b in p["budgets"]], [100 * v / p["n"] for v in p["within"]]
            line(ax, xs, ys, MODEL_COLOR[m], lw=2.4, zorder=3)
            ax.plot(xs[-1], ys[-1], "o", ms=8 * R.PT, color=MODEL_COLOR[m], mec=PAGE, mew=2 * R.PT, zorder=4)
            k16 = p["budgets"].index(16384)
            ax.plot(16, ys[k16], "o", ms=7 * R.PT, color=MODEL_COLOR[m], mec=PAGE, mew=1.5 * R.PT, zorder=4)
        label(ax, t["set"][s])
        ax.set_xlabel(t["budget"], fontsize=FS_SMALL, color=R.MUTED)
    legend(f, [(t["model"][m], MODEL_COLOR[m], "line") for m in MODELS], 0.065, 0.995)
    f.text(0.97, 0.955, t["right_within"], fontsize=FS_SMALL, color=R.MUTED, ha="right", va="top")
    return f


def frontier(d, t):
    runs = d["eval"]["runs"]
    f = fig(8.4, 5.4)
    ax = axes(f, (0.11, 0.13, 0.84, 0.8), grid="both")
    ax.set_xlim(0, 31)
    ax.set_ylim(55, 95)
    for s in ("d_num", "d_code", "d_fresh", "d_num_hard"):
        pts = []
        for m in MODELS:
            v = runs[f"d_num_hard/{m}" if s == "d_num_hard" else m]["sets"][s]
            pts.append((v["mean_output_tokens"] / 1000, v["score"], m))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=R.AXIS, lw=1.6 * R.PT, zorder=2)
        for x, y, m in pts:
            ax.plot(x, y, "o", ms=11 * R.PT, color=MODEL_COLOR[m], mec=PAGE, mew=2 * R.PT, zorder=3)
        x0, y0, _ = pts[0]
        left = s == "d_code"
        ax.text(x0 + (-0.5 if left else 0.5), y0, t["set"][s], fontsize=FS, color=R.INK, va="center",
                ha="right" if left else "left")
    ax.set_xticks([0, 5, 10, 15, 20, 25], ["0", "5K", "10K", "15K", "20K", "25K"])
    ax.set_xlabel(t["mean_tokens"], fontsize=FS_SMALL, color=R.MUTED)
    ax.set_ylabel(t["score"], fontsize=FS_SMALL, color=R.MUTED)
    legend(f, [(t["model"][m], MODEL_COLOR[m], "dot") for m in MODELS], 0.1, 0.995)
    return f


def families3(d, t, code):
    fs = d.get("family_scores") or {}
    sc = {m: fs.get(f"d_num_hard/{m}", {}).get("d_num_hard", {}) for m in MODELS}
    fams = sorted(sc["base_medium"], key=lambda k: sc["base_medium"][k]["score"])
    f = fig(8.4, 5.4)
    ax = axes(f, (0.44, 0.08, 0.53, 0.8), grid="x")
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(-2, 102)
    ax.set_ylim(len(fams) - 0.4, -0.6)
    ax.set_yticks(range(len(fams)), [P.T[code]["families"].get(k, k) for k in fams])
    ax.tick_params(axis="y", labelcolor=R.INK2, labelsize=FS_SMALL)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    for i, k in enumerate(fams):
        vals = [sc[m].get(k, {}).get("score") for m in MODELS]
        vs = [v for v in vals if v is not None]
        ax.plot([min(vs), max(vs)], [i, i], color=R.AXIS, lw=2 * R.PT, zorder=2)
        for m, v in zip(MODELS, vals):
            if v is not None:
                ax.plot(v, i, "o", ms=10 * R.PT, color=MODEL_COLOR[m], mec=PAGE, mew=1.6 * R.PT, zorder=3)
    legend(f, [(t["model"][m], MODEL_COLOR[m], "dot") for m in MODELS], 0.43, 0.995)
    return f


def families_base(d, t, code):
    sc = (d.get("family_scores") or {}).get("d_num_hard/base_medium", {}).get("d_num_hard", {})
    fams = sorted(sc, key=lambda k: -sc[k]["score"])
    f = fig(8.4, 4.6)
    ax = axes(f, (0.44, 0.1, 0.5, 0.86), grid="x")
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(0, 100)
    ax.set_ylim(len(fams) - 0.4, -0.6)
    ax.set_yticks(range(len(fams)), [P.T[code]["families"].get(k, k) for k in fams])
    ax.tick_params(axis="y", labelcolor=R.INK2, labelsize=FS_SMALL)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    px, py = R.per_unit(ax)
    for i, k in enumerate(fams):
        v = sc[k]["score"]
        R.bar(ax, i, 0, v, 22 * R.PX / py, R.INK2 if v > 0 else R.AXIS, horizontal=True)
        ax.text(max(v, 0) + 1.5, i, f"{v:.0f}", va="center", fontsize=FS_SMALL, color=R.INK2)
    return f


def passk3(d, t):
    runs = d["eval"]["runs"]
    f = fig(6.2, 5.4)
    ax = axes(f, (0.16, 0.13, 0.8, 0.72), grid="both")
    ax.set_xscale("log", base=2)
    ks = [1, 2, 4, 8, 16]
    for m in MODELS:
        pk = runs[f"{m}_k64"]["sets"]["d_num"]["pass_at_k"]
        ys = [pk[str(k)] for k in ks]
        line(ax, ks, ys, MODEL_COLOR[m], lw=2.4)
        ax.plot(ks, ys, "o", ms=7 * R.PT, color=MODEL_COLOR[m], mec=PAGE, mew=1.5 * R.PT)
    ax.set_xlim(0.75, 19)
    ax.set_ylim(88, 100.5)
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xlabel(t["k"], fontsize=FS_SMALL, color=R.MUTED)
    label(ax, t["pass_k"], y=1.04)
    ax.legend([Line2D([0], [0], color=MODEL_COLOR[m], lw=2.4 * R.PT) for m in MODELS], [t["model"][m] for m in MODELS],
              loc="lower right", frameon=False, fontsize=FS_SMALL, labelcolor=R.INK2)
    return f


def judges(d, t):
    runs = d["eval"]["runs"]
    rows = []
    for m in ("sft", "rl_seed1"):
        bj = runs[m]["sets"]["d_report"]["by_judge"]
        for j in ("primary", "second"):
            o = bj[j]["outcomes"]
            n = sum(o.values())
            rows.append((f"{t['model'][m]} · {t['judge'][j]}", [100 * o["win"] / n, 100 * o["tie"] / n,
                                                                  100 * o["loss"] / n], bj[j]["score"]))
    f = fig(12.4, 3.9)
    ax = axes(f, (0.35, 0.1, 0.52, 0.68), grid=None)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(labelbottom=False)
    ax.set_xlim(0, 100)
    ax.set_ylim(len(rows) - 0.45, -0.55)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.tick_params(axis="y", labelcolor=R.INK2, labelsize=FS)
    px, py = R.per_unit(ax)
    gap, thick = 2 * R.PX / px, 26 * R.PX / py
    for i, (_, vals, score) in enumerate(rows):
        x = 0.0
        for n, (c, w) in enumerate(zip((WIN, TIE, LOSS), vals)):
            R.bar(ax, i, x, max(w - (0 if n == 2 else gap), 0), thick, c, horizontal=True, rounded=False)
            if w >= 6:
                ax.text(x + w / 2, i, f"{w:.0f}", ha="center", va="center", fontsize=FS_SMALL,
                        color=PAGE if c != TIE else R.INK2)
            x += w
        ax.text(101.5, i, t["winrate"].format(v=score), va="center", fontsize=FS_SMALL, color=R.INK2)
    ax.axvline(50, color=R.INK2, lw=1 * R.PT, zorder=6)
    legend(f, [(t["win"], WIN, "box"), (t["tie"], TIE, "box"), (t["loss"], LOSS, "box")], 0.345, 0.995)
    return f


def judge_drift(jd, rl, t):
    """GPT-6 Luna re-scoring the T3 dev answers the Gemma reward judge scored (scripts/talk/judge_drift_ci.py):
    prompt-level means with 95% whiskers. Reward hacking would lift Gemma's line and widen the gap."""
    rows = jd["steps"]
    first = next(r["step"] for r in rl["metrics"] if (r.get("task/t3_report/n_samples") or 0) > 0)
    f = fig(8.4, 4.6)
    ax = axes(f, (0.1, 0.13, 0.86, 0.68))
    step_axis(ax)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.axvspan(first, 57, color=R.TASK_COLOR["t3_report"], alpha=0.10, lw=0, zorder=0)
    ax.text(first + 0.8, 0.98, t["t3_train"].format(s=first), transform=ax.get_xaxis_transform(),
            fontsize=FS_SMALL, color=R.INK2, va="top")
    vline(ax, 20, t["selected"], ha="right")
    xs = [r["step"] for r in rows]
    ax.fill_between(xs, [r["luna"]["point"] for r in rows], [r["gemma"]["point"] for r in rows], color=R.GRID,
                    alpha=0.7, lw=0, zorder=1)
    for key, ls, face in (("gemma", "-", R.INK), ("luna", (0, (4, 2)), PAGE)):
        ys = [r[key]["point"] for r in rows]
        for r in rows:
            ax.plot([r["step"]] * 2, [r[key]["lo"], r[key]["hi"]], color=R.INK2, lw=1.4 * R.PT, zorder=3,
                    solid_capstyle="butt", clip_on=False)
        ax.plot(xs, ys, color=R.INK, lw=2.4 * R.PT, ls=ls, zorder=4)
        ax.plot(xs, ys, lw=0, marker="o", ms=10 * R.PT, mfc=face, mec=R.INK, mew=1.6 * R.PT, zorder=5, clip_on=False)
        if t[f"drift_{key}_end"]:
            ax.text(xs[-1] + 1.6, ys[-1], t[f"drift_{key}_end"], fontsize=FS_SMALL, color=R.INK, va="center")
    for r in rows:
        if t["drift_gap"]:
            mid = (r["gemma"]["point"] + r["luna"]["point"]) / 2
            ax.text(r["step"] + 1.0, mid, t["drift_gap"].format(v=r["gap"]["point"]), fontsize=FS_SMALL,
                    color=R.INK2, va="center")
    ax.set_xlabel(t["step"], fontsize=FS_SMALL, color=R.MUTED)
    label(ax, t["drift_y"])
    legend(f, [(t["drift_gemma"], R.INK, "line"), (t["drift_luna"], R.INK, "dash")], 0.095, 0.995)
    return f


def refusal_breakdown(rf, t):
    """Refusal rate on the should-refuse prompts by source (scripts/talk/refusal_breakdown.py): the gate's phrase
    detector (solid) and what a broader lexicon adds (light), per model."""
    from matplotlib.colors import to_rgb

    def light(c):                                      # an opaque tint, so the grid does not show through
        return tuple(0.3 * x + 0.7 for x in to_rgb(c))

    g = rf["groups"]
    f = fig(8.4, 4.6)
    for i, (src, title) in enumerate((("jbb", t["ref_jbb"]), ("xstest", t["ref_xs"]))):
        ax = axes(f, (0.17, 0.56 - i * 0.44, 0.74, 0.25), grid="x")
        ax.set_xlim(0, 100)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax.set_ylim(len(MODELS) - 0.45, -0.55)
        ax.set_yticks(range(len(MODELS)), [t["model"][m] for m in MODELS])
        ax.tick_params(axis="y", labelcolor=R.INK2, labelsize=FS_SMALL, length=0)
        ax.spines["left"].set_visible(False)
        _, py = R.per_unit(ax)
        thick = 22 * R.PX / py
        for y, m in enumerate(MODELS):
            rows = [v for k, v in g[m].items() if k.startswith(src + "/")]
            n = sum(v["n"] for v in rows)
            strict, broad = (100 * sum(v[k] for v in rows) / n for k in ("strict", "broad"))
            R.bar(ax, y, 0, strict, thick, MODEL_COLOR[m], horizontal=True, rounded=False)
            R.bar(ax, y, strict, broad - strict, thick, light(MODEL_COLOR[m]), horizontal=True, rounded=False)
            ax.text(strict - 1.2, y, f"{strict:.0f}", ha="right", va="center", fontsize=FS_SMALL, color=PAGE)
            ax.text(broad + 1.2, y, f"{broad:.0f}", ha="left", va="center", fontsize=FS_SMALL, color=R.INK2)
        label(ax, title, y=1.06)
    legend(f, [(t["ref_strict"], R.INK2, "box"), (t["ref_broad"], light(R.INK2), "box")], 0.03, 0.995)
    return f


# ---- engineering --------------------------------------------------------------------------------------------------
def clock(s: str) -> dt.datetime:
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M")


def host_memory(rl, t):
    h = rl["host_memory_stage8"]
    t0 = clock("2026-10-29 22:30")
    rows = [((clock(r["t"]) - t0).total_seconds() / 3600, r["in_use_mb"] / 1024) for r in h
            if clock(r["t"]) <= clock("2026-10-30 03:10")]
    f = fig(12.4, 4.6)
    ax = axes(f, (0.07, 0.15, 0.9, 0.7))
    ax.fill_between([x for x, _ in rows], [y for _, y in rows], color=R.BLUE_LIGHT, lw=0, zorder=2)
    line(ax, [x for x, _ in rows], [y for _, y in rows], R.BLUE, lw=1.6, zorder=3)
    ax.axhline(244000 / 1024, color=LOSS, lw=1.4 * R.PT, ls=(0, (3, 2)))
    ax.text(2.3, 244000 / 1024 - 4, t["kill"], fontsize=FS_SMALL, color=LOSS, ha="left", va="top")  # clear of 02:31
    ax.axhline(256872 / 1024, color=R.AXIS, lw=1 * R.PT)
    ax.set_ylim(0, 270)
    ax.set_xlim(0, 4.67)
    ticks = [clock(s) for s in ("2026-10-29 22:30", "2026-10-29 23:30", "2026-10-30 00:30", "2026-10-30 01:30",
                                "2026-10-30 02:30")]
    ax.set_xticks([(c - t0).total_seconds() / 3600 for c in ticks], [c.strftime("%H:%M") for c in ticks])
    peak = max(rows[:30], key=lambda r: r[1])
    if t["ev_resume"]:
        ax.annotate(t["ev_resume"], xy=peak, xytext=(peak[0] + 0.28, 205), fontsize=FS_SMALL, color=R.INK,
                    va="center", arrowprops={"arrowstyle": "-", "color": R.MUTED, "lw": 1 * R.PT})
    x50 = (clock("2026-10-29 22:50") - t0).total_seconds() / 3600
    if t["ev_mtpoff"]:
        ax.annotate(t["ev_mtpoff"], xy=(x50, 10), xytext=(x50 + 0.28, 75), fontsize=FS_SMALL, color=R.INK,
                    va="center", arrowprops={"arrowstyle": "-", "color": R.MUTED, "lw": 1 * R.PT})
    for s, txt in (("2026-10-30 00:42", t["ev_save"]), ("2026-10-30 01:45", ""), ("2026-10-30 02:30", t["ev_pause"])):
        x = (clock(s) - t0).total_seconds() / 3600
        ax.axvline(x, color=R.AXIS, lw=1 * R.PT, ls=(0, (3, 3)), zorder=1)
        if txt:
            ax.text(x - 0.04, 262, txt, fontsize=FS_SMALL, color=R.MUTED, ha="right", va="top")
    label(ax, t["in_use"])
    return f


def gpu_memory(rl, t):
    f = fig(12.4, 4.4)
    specs = [(rl["gpu_memory_stage7_sft"], "2026-10-29 11:05", "2026-10-29 14:35", t["sft_run"]),
             (rl["gpu_memory_stage7_rl"], "2026-10-29 22:45", "2026-10-30 02:40", t["rl_run"])]
    for i, (rows, a, b, name) in enumerate(specs):
        ax = axes(f, (0.08 + i * 0.47, 0.14, 0.42, 0.66))
        t0, t1 = clock(a), clock(b)
        pts = [((clock(r["t"]) - t0).total_seconds() / 3600, r["mib"] if r["mib"] >= 30000 else float("nan"))
               for r in rows if t0 <= clock(r["t"]) <= t1]          # a restart (no process) is a gap
        line(ax, [x for x, _ in pts], [y for _, y in pts], R.BLUE, lw=1.6)
        ax.axhline(40437, color=LOSS, lw=1.4 * R.PT, ls=(0, (3, 2)))
        ax.text(0.02, 40437 + 80, t["capacity"], fontsize=FS_SMALL, color=LOSS, va="bottom")
        ax.set_ylim(30000, 41500)
        ax.set_xlim(0, (t1 - t0).total_seconds() / 3600)
        hrs = range(0, int((t1 - t0).total_seconds() // 3600) + 1)
        ax.set_xticks(list(hrs), [(t0 + dt.timedelta(hours=k)).strftime("%H:%M") for k in hrs])
        ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}K")
        label(ax, name)
        if i == 0:
            for s, txt in (("2026-10-29 12:03", t["crash50"]), ("2026-10-29 12:07", "")):
                x = (clock(s) - t0).total_seconds() / 3600
                ax.axvline(x, color=R.AXIS, lw=1 * R.PT, ls=(0, (3, 3)), zorder=1)
                if txt:
                    ax.text(x - 0.03, 31000, txt, fontsize=FS_SMALL, color=R.MUTED, ha="right")
    f.text(0.075, 0.97, t["gpu_mib"], fontsize=FS, color=R.INK, va="top")
    return f


def time_accounting(t):
    segs = [(13.74, PLANNED), (35.1, INCIDENT), (2.32, PLANNED), (2.07, INCIDENT), (4.75, PLANNED), (1.58, INCIDENT),
            (3.68, PLANNED), (0.83, IDLE), (4.17, PLANNED), (3.63, IDLE)]
    f = fig(12.4, 3.0)
    ax = axes(f, (0.03, 0.3, 0.94, 0.32), grid=None)
    ax.set_xlim(0, 72)
    ax.set_ylim(-0.6, 0.6)
    ax.tick_params(labelleft=False)
    ax.set_xticks([0, 12, 24, 36, 48, 60, 72])
    def width(name: str) -> float:                      # hours a label needs: ~0.62 h a Latin character, 1.2 a CJK one
        return sum(1.2 if ord(ch) > 0x2E80 else 0.62 for ch in name) + 1

    x = 0.0
    for i, ((w, c), name) in enumerate(zip(segs, t["seg"])):
        ax.barh(0, w - 0.08, left=x, height=0.8, color=c, lw=0)
        if w >= width(name):
            ax.text(x + w / 2, 0, name, ha="center", va="center", fontsize=FS_SMALL,
                    color=PAGE if c != IDLE else R.INK2)
        else:
            up = sum(1 for (w2, _), n2 in zip(segs[:i], t["seg"][:i]) if w2 < width(n2)) % 2 == 0
            ax.text(x + w / 2, 0.5 if up else -0.5, name, ha="center", va="bottom" if up else "top",
                    fontsize=FS_SMALL - 1, color=R.INK2)
        x += w
    ax.set_xlabel(t["hours"], fontsize=FS_SMALL, color=R.MUTED)
    legend(f, [(t["planned"], PLANNED, "box"), (t["incident"], INCIDENT, "box"), (t["idle"], IDLE, "box")], 0.025,
           0.99)
    return f


def sft_loss(rl, t):
    s = rl["sft_steps"]
    f = fig(8.4, 4.6)
    ax = axes(f, (0.1, 0.14, 0.86, 0.72))
    for key, col, name in (("train/loss", R.BLUE, t["sft_loss"]), ("mtp_losses/mtp_1_loss", P.LIGHT, t["mtp_loss"])):
        xs = [r["step"] for r in s if key in r]
        ys = [r[key] for r in s if key in r]
        line(ax, xs, ys, col, lw=1.8)
    ax.set_xlim(0, 210)
    ax.set_ylim(0.3, 0.95)
    ax.set_xlabel(t["sft_step"], fontsize=FS_SMALL, color=R.MUTED)
    legend(f, [(t["sft_loss"], R.BLUE, "line"), (t["mtp_loss"], P.LIGHT, "line")], 0.095, 0.995)
    return f


def base_figures(d) -> None:
    """The report's base-model figures (scripts/talk/paper_figures.py), saved again at slide resolution in base/."""
    def save300(f, name: str) -> None:
        (OUT / "base").mkdir(parents=True, exist_ok=True)
        f.savefig(OUT / "base" / f"{name}.png", dpi=300, facecolor=P.PAGE)
        plt.close(f)

    def small_fig(h: float, w: float = P.W):
        # the report's 6.4-in figures on a 0.85 canvas: the same 8.5-pt text is ~18% larger on the slide
        return plt.figure(figsize=(w * 0.85, h * 0.85), dpi=R.DPI, facecolor=P.PAGE)

    P.save, P.fig = save300, small_fig
    P.lang("en")
    for g in (P.budget, P.scorecard, P.receiver, P.noise):
        g(d)
    for code in ("en", "zh"):
        P.lang(code)
        for g in (P.funnel, P.failures, P.families):
            g(d, code)


def main() -> None:
    d = json.loads((DATA / "run.json").read_text())
    rl = json.loads((DATA / "rl.json").read_text())
    jd = json.loads((DATA / "judge_drift.json").read_text())
    rf = json.loads((DATA / "refusal.json").read_text())
    base_figures(d)
    makers = {
        "rl_collapse": lambda t, c: rl_collapse(rl, t), "rl_signal": lambda t, c: rl_signal(rl, t),
        "rl_monitors": lambda t, c: rl_monitors(rl, t), "rl_dev_shift": lambda t, c: rl_dev_shift(rl, t),
        "eval_failures": lambda t, c: eval_failures(d, t), "budget3": lambda t, c: budget3(d, t),
        "frontier": lambda t, c: frontier(d, t), "families3": lambda t, c: families3(d, t, c),
        "families_base": lambda t, c: families_base(d, t, c), "passk3": lambda t, c: passk3(d, t),
        "judges": lambda t, c: judges(d, t), "judge_drift": lambda t, c: judge_drift(jd, rl, t),
        "refusal_breakdown": lambda t, c: refusal_breakdown(rf, t),
        "host_memory": lambda t, c: host_memory(rl, t), "gpu_memory": lambda t, c: gpu_memory(rl, t),
        "time_accounting": lambda t, c: time_accounting(t), "sft_loss": lambda t, c: sft_loss(rl, t),
    }
    LATEX_FIG.mkdir(parents=True, exist_ok=True)
    for code in ("en", "zh"):
        P.lang(code)
        plt.rcParams.update({"font.size": FS, "axes.unicode_minus": False})
        for name, make in makers.items():
            save(make(L[code], code), name, code)
        quiet = {**L[code], **{k: "" for k in QUIET_KEYS}, "quiet": True}
        for name in REPORT_FIGS:                       # the story report's text-light renders, deck_<name>_<code>.pdf
            f = makers[name](quiet, code)
            f.savefig(LATEX_FIG / f"deck_{name}_{code}.pdf", facecolor=PAGE)
            plt.close(f)
        print(code, "done")
    print(f"{len(REPORT_FIGS)} text-light figures × 2 in {LATEX_FIG}")


if __name__ == "__main__":
    main()
