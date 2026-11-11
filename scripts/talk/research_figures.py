"""Figures of the reward / rubric / scaling recap (2026-11-10): docs/reports/reward_rubric_scaling_1110_{en,zh}.md.

    ~/miniconda3/bin/python scripts/talk/research_figures.py     # → docs/reports/figures/recap_<name>_{en,zh}.png|pdf

Inputs, aggregates only:
- docs/talk/data/reward_vs_eval.json (scripts/talk/reward_vs_eval.py): T1 reward vs the eval grader, T3 Gemma vs Luna;
- docs/talk/data/research_1110.json (scripts/talk/research_1110.py): maj@k / pass@k on D-Num, rubric statistics;
- docs/talk/data/run_results.json: the base model's difficulty audit per template, mean tokens per answer;
- docs/talk/data/rl.json: run 1's GRPO group outcomes per step.

Colours: the deck's (scripts/talk/deck_figures.py). Models are base grey, SFT blue, RL orange. Group outcomes are as in
rl_signal. Agreement uses categorical slots 1, 3, 2 and 7, checked with the data-viz validator on 2026-11-10 (adjacent
CVD ΔE ≥ 9.2, normal vision ≥ 24). Aqua is below 3:1 contrast, so every segment carries its count.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deck_figures as D  # noqa: E402  (chrome, fonts, palette)

R = D.R
OUT = R.ROOT / "docs" / "reports" / "figures"
AGREE = {"TP": "#2a78d6", "TN": "#1baf7a", "FP": "#eb6834", "FN": "#6250d6"}
OUTCOME = {"right": R.BLUE_LIGHT, "mixed": R.ORANGE, "wrong": R.INK2}          # as in rl_signal
MODEL_NAME = {"en": {"base_medium": "base", "sft": "SFT", "rl_seed1": "RL step 20"},
              "zh": {"base_medium": "基座", "sft": "SFT", "rl_seed1": "RL 第 20 步"}}

T = {
    "en": {
        "TP": "both right", "TN": "both wrong", "FP": "reward right, reference wrong",
        "FN": "reward wrong, reference right",
        "t1": "T1 reward vs eval grader (D-Num, 2,400 answers each)",
        "t3": "T3 reward judge (Gemma) vs GPT-6 Luna (141 RL dev reports)",
        "mark": "pass mark {m:.0f}", "agree": "agree {v:.1f}%", "none": "no disagreement",
        "dis": "FP {fp} · FN {fn}",
        "words": "words per T3 rubric criterion (5,765 criteria, median {m:.0f})", "median": "median {m:.0f}",
        "n_crit": "criteria", "joins": "“and” / “or” / “;” per criterion\n{v:.0f}% of criteria have at least one",
        "share": "share of criteria",
        "right": "all 8 right: no signal", "mixed": "mixed: signal", "wrong": "all 8 wrong: no signal",
        "rows": ["T1, base (bank audit)", "T1, SFT (RL steps 1–20)", "T1, SFT (steps 21–40)",
                 "T2, SFT (steps 1–20)", "T4, base (bank audit)", "T4, SFT (steps 1–20)"],
        "groups": "GRPO groups of 8 samples",
        "per_tpl": "base model: 81 T1 templates", "tpl": "share of problems solved 8/8",
        "n_tpl": "templates", "drop": "",
        "k": "samples per problem (k)", "tokens": "tokens generated per problem",
        "maj": "majority vote (maj@k)", "pass": "any sample right (pass@k)",
        "acc": "D-Num accuracy", "vs_k": "D-Num accuracy vs samples\nsolid: majority vote · dashed: pass@k",
        "vs_tok": "majority vote vs tokens per problem",
    },
    "zh": {
        "TP": "两者都判对", "TN": "两者都判错", "FP": "奖励判对、参照判错", "FN": "奖励判错、参照判对",
        "t1": "T1 奖励 vs 评测判分器（D-Num，每个模型 2,400 个回答）",
        "t3": "T3 奖励裁判（Gemma）vs GPT-6 Luna（141 份 RL 开发集报告）",
        "mark": "及格线 {m:.0f}", "agree": "一致 {v:.1f}%", "none": "无分歧", "dis": "FP {fp} · FN {fn}",
        "words": "T3 评分细则每条标准的词数（5,765 条，中位数 {m:.0f}）", "median": "中位数 {m:.0f}",
        "n_crit": "标准条数", "joins": "每条标准中的 “and” / “or” / “;”\n{v:.0f}% 的标准至少含一个",
        "share": "占标准的比例",
        "right": "8 个全对：无信号", "mixed": "有对有错：有信号", "wrong": "8 个全错：无信号",
        "rows": ["T1，基座（题库审计）", "T1，SFT（RL 第 1–20 步）", "T1，SFT（第 21–40 步）",
                 "T2，SFT（第 1–20 步）", "T4，基座（题库审计）", "T4，SFT（第 1–20 步）"],
        "groups": "GRPO 组（每组 8 个样本）",
        "per_tpl": "基座模型：81 个 T1 模板", "tpl": "8/8 全对的题目比例",
        "n_tpl": "模板数", "drop": "",
        "k": "每题样本数（k）", "tokens": "每题生成的 token 数",
        "maj": "多数投票（maj@k）", "pass": "任一样本正确（pass@k）",
        "acc": "D-Num 准确率", "vs_k": "D-Num 准确率 vs 样本数\n实线：多数投票 · 虚线：pass@k",
        "vs_tok": "多数投票 vs 每题 token 数",
    },
}


def save(f, name: str, code: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    f.savefig(OUT / f"recap_{name}_{code}.png", dpi=R.DPI, facecolor=D.PAGE)
    f.savefig(OUT / f"recap_{name}_{code}.pdf", facecolor=D.PAGE)
    D.plt.close(f)


def stacked(ax, y: float, parts: list[tuple[str, float, str]], total: float, h: float = 0.62,
            min_label: float = 6.0, fmt=lambda v: f"{v:,.0f}") -> None:
    """One horizontal 100% bar: parts are (key, value, colour); a 2 px page-coloured gap between segments."""
    if not total:
        return
    x = 0.0
    for key, v, c in parts:
        w = 100 * v / total
        if w <= 0:
            continue
        ax.barh(y, w, left=x, height=h, color=c, edgecolor=D.PAGE, linewidth=2 * R.PT)
        if w >= min_label:
            ink = "#ffffff" if key in ("TP", "FN", "FP", "mixed", "wrong", "partial", "missed", "quantitative",
                                       "coverage") else R.INK
            ax.text(x + w / 2, y, fmt(v), ha="center", va="center", fontsize=D.FS_SMALL, color=ink)
        x += w


# ---------------------------------------------------------------------------------------------- figure 1
def reward_agreement(rv, t, code):
    f = D.fig(13.0, 7.4)
    a1 = D.axes(f, [0.15, 0.55, 0.66, 0.28], grid=None)
    a2 = D.axes(f, [0.15, 0.09, 0.66, 0.28], grid=None)
    t1 = rv["t1_reward_vs_eval_grader"]
    order = ["rl_seed1", "sft", "base_medium"]
    for i, m in enumerate(order):
        c = t1[m]
        stacked(a1, i, [(k, c[k], AGREE[k]) for k in ("TP", "TN", "FP", "FN")], c["n"])
        dis = c["FP"] + c["FN"]
        a1.text(101.5, i, t["dis"].format(fp=c["FP"], fn=c["FN"]) if dis else t["none"], va="center",
                fontsize=D.FS_SMALL, color=R.INK2)
    a1.set_yticks(range(3))
    a1.set_yticklabels([MODEL_NAME[code][m] for m in order], fontsize=D.FS_SMALL)
    D.label(a1, t["t1"], y=1.06)
    t3 = sorted(rv["t3_gemma_vs_luna"], key=lambda r: -r["threshold"])
    for i, c in enumerate(t3):
        stacked(a2, i, [(k, c[k], AGREE[k]) for k in ("TP", "TN", "FP", "FN")], c["n"], min_label=4.0)
        a2.text(101.5, i, t["agree"].format(v=100 * c["agreement"]), va="center", fontsize=D.FS_SMALL, color=R.INK2)
    a2.set_yticks(range(len(t3)))
    a2.set_yticklabels([t["mark"].format(m=c["threshold"]) for c in t3], fontsize=D.FS_SMALL)
    D.label(a2, t["t3"], y=1.06)
    for ax in (a1, a2):
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.set_xticklabels(["0%", "50%", "100%"])
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", colors=R.MUTED)
    D.legend(f, [(t[k], AGREE[k], "box") for k in ("TP", "TN", "FP", "FN")], 0.04, 0.99, ncol=4)
    save(f, "reward_agreement", code)


# ---------------------------------------------------------------------------------------------- figure 2
def rubric_atomicity(rs, t, code):
    t3 = rs["bank_v1"]["t3"]
    f = D.fig(13.0, 5.0)
    a1 = D.axes(f, [0.07, 0.14, 0.50, 0.68])
    a2 = D.axes(f, [0.67, 0.14, 0.30, 0.68])
    edges, hist = t3["words_hist_edges"], t3["words_hist"]
    cut = edges.index(80)                                   # 80 words and longer in one bar
    counts = hist[:cut] + [sum(hist[cut:])]
    labels = [f"{lo}–{hi - 1}" for lo, hi in zip(edges[:cut], edges[1:cut + 1])] + ["80+"]
    a1.bar(range(len(counts)), counts, width=0.78, color=R.BLUE, edgecolor=D.PAGE, linewidth=2 * R.PT)
    a1.set_xticks(range(len(counts)))
    a1.set_xticklabels(labels, fontsize=D.FS_SMALL - 1)
    m = t3["words_median"]
    mid = next(i for i, (lo, hi) in enumerate(zip(edges, edges[1:])) if lo <= m < hi)
    a1.axvline(mid + (m - edges[mid]) / (edges[mid + 1] - edges[mid]) - 0.5, color=R.INK2, lw=1.4 * R.PT,
               ls=(0, (3, 3)), zorder=4)
    a1.set_ylabel(t["n_crit"], fontsize=D.FS_SMALL, color=R.MUTED)
    D.label(a1, t["words"].format(m=m))
    joins = t3["joins_per_criterion"]
    keys = sorted(joins, key=int)
    n = sum(joins.values())
    vals = [100 * joins[k] / n for k in keys]
    a2.bar(range(len(keys)), vals, width=0.74, color=R.BLUE, edgecolor=D.PAGE, linewidth=2 * R.PT)
    a2.set_xticks(range(len(keys)))
    a2.set_xticklabels([k if k != "6" else "6+" for k in keys], fontsize=D.FS_SMALL)
    a2.set_ylabel(t["share"], fontsize=D.FS_SMALL, color=R.MUTED)
    a2.yaxis.set_major_formatter(D.plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    D.label(a2, t["joins"].format(v=100 * t3["joined_share"]))
    save(f, "rubric_atomicity", code)


# ---------------------------------------------------------------------------------------------- figure 3
def base_audit_shares(groups: dict, task: str) -> tuple[float, float, float]:
    """Problem shares of the audit over the templates kept in the bank, the population RL then drew from."""
    rows = [g for k, g in groups.items() if k.startswith(task + ":") and g["decision"] == "keep"]
    n = sum(g["audited"] for g in rows)
    right = sum(g["share_all_correct"] * g["audited"] for g in rows) / n
    wrong = sum(g["share_none_correct"] * g["audited"] for g in rows) / n
    return right, 1 - right - wrong, wrong


def policy_shares(metrics: list[dict], task: str, lo: int, hi: int) -> tuple[float, float, float]:
    rows = [m for m in metrics if lo <= m["step"] <= hi and m.get(f"filter/{task}/all_solved_group_frac") is not None]
    right = st.fmean(m[f"filter/{task}/all_solved_group_frac"] for m in rows)
    zero = st.fmean(m[f"filter/{task}/zero_adv_group_frac"] for m in rows)
    return right, 1 - zero, zero - right


def grpo_groups(run_results, rl, t, code):
    groups, metrics = run_results["groups"]["difficulty"], rl["metrics"]
    rows = [base_audit_shares(groups, "t1_num"), policy_shares(metrics, "t1_num", 1, 20),
            policy_shares(metrics, "t1_num", 21, 40), policy_shares(metrics, "t2_code", 1, 20),
            base_audit_shares(groups, "t4_general"), policy_shares(metrics, "t4_general", 1, 20)]
    f = D.fig(13.0, 5.6)
    a1 = D.axes(f, [0.24, 0.12, 0.38, 0.68], grid=None)
    a2 = D.axes(f, [0.72, 0.17, 0.25, 0.63])
    for i, (r, m, w) in enumerate(reversed(rows)):
        stacked(a1, i, [("right", r, OUTCOME["right"]), ("mixed", m, OUTCOME["mixed"]), ("wrong", w, OUTCOME["wrong"])],
                1.0, fmt=lambda v: f"{100 * v:.0f}%", min_label=7.0)
    a1.set_yticks(range(len(rows)))
    a1.set_yticklabels(list(reversed(t["rows"])), fontsize=D.FS_SMALL)
    a1.set_xlim(0, 100)
    a1.set_xticks([0, 50, 100])
    a1.set_xticklabels(["0%", "50%", "100%"])
    a1.spines["bottom"].set_visible(False)
    a1.tick_params(axis="x", colors=R.MUTED)
    D.label(a1, t["groups"], y=1.04)
    shares = [100 * g["share_all_correct"] for k, g in groups.items() if k.startswith("t1_num:")]
    bins = list(range(0, 101, 10))
    counts = [sum(lo <= v < hi or (hi == 100 and v == 100) for v in shares) for lo, hi in zip(bins, bins[1:])]
    a2.bar([b + 5 for b in bins[:-1]], counts, width=8.4, color=R.BLUE, edgecolor=D.PAGE, linewidth=2 * R.PT)
    a2.axvline(90, color=R.INK2, lw=1.2 * R.PT, ls=(0, (3, 3)))
    a2.set_xlim(0, 100)
    a2.set_xticks([0, 50, 100])
    a2.set_xticklabels(["0%", "50%", "100%"])
    a2.set_xlabel(t["tpl"], fontsize=D.FS_SMALL, color=R.MUTED)
    a2.set_ylabel(t["n_tpl"], fontsize=D.FS_SMALL, color=R.MUTED)
    D.label(a2, t["per_tpl"], y=1.04)
    D.legend(f, [(t[k], OUTCOME[k], "box") for k in ("right", "mixed", "wrong")], 0.04, 0.99, ncol=3)
    save(f, "grpo_groups", code)


# ---------------------------------------------------------------------------------------------- figure 4
def test_time(rs, run_results, t, code):
    vote = rs["d_num_voting"]
    runs = run_results["eval"]["runs"]
    toks = {m: runs[f"{m}_k64"]["sets"]["d_num"]["mean_output_tokens"] for m in D.MODELS}
    ks = [int(k) for k in vote["base_medium"]["maj_at_k"]]
    f = D.fig(13.0, 5.4)
    a1 = D.axes(f, [0.07, 0.15, 0.38, 0.66])
    a2 = D.axes(f, [0.57, 0.15, 0.38, 0.66])
    for m in D.MODELS:
        c = D.MODEL_COLOR[m]
        maj = [vote[m]["maj_at_k"][str(k)] for k in ks]
        pk = [vote[m]["pass_at_k"][str(k)] for k in ks]
        D.line(a1, ks, maj, c, marker="o", ms=7 * R.PT, mec=D.PAGE, mew=1.5 * R.PT)
        D.line(a1, ks, pk, c, lw=1.4, ls=(0, (3, 2)))
        xs = [k * toks[m] for k in ks]
        D.line(a2, xs, maj, c, marker="o", ms=7 * R.PT, mec=D.PAGE, mew=1.5 * R.PT)
        if m == "rl_seed1":                     # left of its first point: the SFT line starts just below it
            a2.text(xs[0] * 0.93, maj[0] + 0.2, MODEL_NAME[code][m], ha="right", va="bottom", fontsize=D.FS_SMALL,
                    color=R.INK2)
        else:
            a2.text(xs[0] * 1.12, maj[0] - 0.15, MODEL_NAME[code][m], ha="left", va="top", fontsize=D.FS_SMALL,
                    color=R.INK2)
    for ax in (a1, a2):
        ax.set_xscale("log", base=2)
        ax.set_ylim(88, 100.6)
        ax.yaxis.set_major_formatter(D.plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.grid(axis="x", color=R.GRID, linewidth=1 * R.PT)
    a1.set_xticks(ks)
    a1.set_xticklabels([str(k) for k in ks])
    a1.set_xlabel(t["k"], fontsize=D.FS_SMALL, color=R.MUTED)
    D.label(a1, t["vs_k"])
    ticks = [4096, 16384, 65536, 262144, 1048576]
    a2.set_xticks(ticks)
    a2.set_xticklabels(["4K", "16K", "64K", "256K", "1M"])
    a2.set_xlim(1800, 1.4e6)
    a2.set_xlabel(t["tokens"], fontsize=D.FS_SMALL, color=R.MUTED)
    D.label(a2, t["vs_tok"])
    D.legend(f, [(MODEL_NAME[code][m], D.MODEL_COLOR[m], "line") for m in D.MODELS], 0.56, 0.99, ncol=3)
    save(f, "test_time", code)


# ---------------------------------------------------------------------------------------------- figures 5-7
FAIL = {"right": "#2a78d6", "wrong": "#eb6834", "unfinished": "#1baf7a", "format": "#6250d6"}
WHY = {"concept": "#e34948", "execution": "#eda100", "unclear": R.MUTED}
# concept is red in Figures 5 and 6; this order passed the validator on 11-10 (adjacent CVD ΔE ≥ 9.6, normal ≥ 24)
SHORT = {"concept": "#e34948", "quantitative": "#2a78d6", "evidence": "#1baf7a", "coverage": "#6250d6",
         "clarity": "#e87ba4"}
VERDICT = {"met": "#d9d7cf", "partial": R.MUTED, "missed": R.INK2}             # ordinal: a neutral light-to-dark ramp

T2 = {
    "en": {"what": "D-Num-hard: what happens to each answer (1,280 per model)", "right": "right",
           "wrong": "wrong value", "unfinished": "unfinished at 32K", "format": "answer format",
           "why": "why, as judged by GPT-6 Luna (sample of each kind; n per bar)", "concept": "concept error",
           "execution": "execution error", "unclear": "unclear", "unf_of": "unfinished, {m}",
           "wrong_of": "wrong value, {m}",
           "verdicts": "D-Report: rubric criteria, as judged by GPT-6 Luna", "met": "met", "partial": "partial",
           "missed": "missed", "short": "type of each shortfall (partial or missed)", "quantitative": "quantitative",
           "evidence": "evidence", "coverage": "coverage", "clarity": "clarity",
           "cut": "answers cut off at 32K tokens", "score": "score", "sets": {
               "gpqa_diamond": "GPQA", "livecodebench_post_2025_01": "LiveCodeBench", "d_num_hard": "D-Num-hard",
               "d_fresh": "D-Fresh", "mmlu_pro_2k": "MMLU-Pro", "d_code": "D-Code", "d_num": "D-Num"}},
    "zh": {"what": "D-Num-hard：每个回答的结果（每个模型 1,280 个）", "right": "正确", "wrong": "数值错误",
           "unfinished": "32K 内未完成", "format": "答案格式", "why": "原因（GPT-6 Luna 判定；各类抽样，柱旁为样本数）",
           "concept": "概念错误", "execution": "执行错误", "unclear": "无法判断", "unf_of": "未完成，{m}",
           "wrong_of": "数值错误，{m}", "verdicts": "D-Report：评分标准（GPT-6 Luna 判定）", "met": "满足",
           "partial": "部分满足", "missed": "未满足", "short": "每处不足的类型（部分满足或未满足）",
           "quantitative": "定量", "evidence": "证据", "coverage": "覆盖", "clarity": "表达",
           "cut": "在 32K token 处被截断的回答比例", "score": "得分", "sets": {
               "gpqa_diamond": "GPQA", "livecodebench_post_2025_01": "LiveCodeBench", "d_num_hard": "D-Num-hard",
               "d_fresh": "D-Fresh", "mmlu_pro_2k": "MMLU-Pro", "d_code": "D-Code", "d_num": "D-Num"}},
}


def numhard_failures(ft, t, code):
    nh = ft["d_num_hard"]
    f = D.fig(13.0, 7.6)
    a1 = D.axes(f, [0.20, 0.64, 0.62, 0.22], grid=None)
    a2 = D.axes(f, [0.20, 0.07, 0.62, 0.36], grid=None)
    order = ["rl_seed1", "sft", "base_medium"]
    for i, m in enumerate(order):
        k = nh[m]["kinds"]
        parts = [("right", k.get("right", 0)), ("wrong", k.get("wrong_value", 0)),
                 ("unfinished", k.get("unfinished", 0)), ("format", k.get("format", 0))]
        stacked(a1, i, [(n, v, FAIL[n]) for n, v in parts], nh[m]["answers"], fmt=lambda v: f"{v:,.0f}")
    a1.set_yticks(range(3))
    a1.set_yticklabels([MODEL_NAME[code][m] for m in order], fontsize=D.FS_SMALL)
    D.label(a1, t["what"], y=1.08)
    rows = [(t["wrong_of"].format(m=MODEL_NAME[code][m]), nh[m]["label_categories"]) for m in order] + \
           [(t["unf_of"].format(m=MODEL_NAME[code][m]), nh[m]["unfinished_label_categories"]) for m in order]
    for i, (_, c) in enumerate(rows):
        n = sum(c.values())
        if n:
            stacked(a2, i, [(k, c.get(k, 0), WHY[k]) for k in ("concept", "execution", "unclear")], n,
                    fmt=lambda v: f"{v:,.0f}", min_label=7.0)
        a2.text(101.5, i, f"n = {n}", va="center", fontsize=D.FS_SMALL, color=R.INK2)
    a2.set_yticks(range(len(rows)))
    a2.set_yticklabels([r[0] for r in rows], fontsize=D.FS_SMALL)
    D.label(a2, t["why"], y=1.04)
    for ax in (a1, a2):
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.set_xticklabels(["0%", "50%", "100%"])
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", colors=R.MUTED)
    D.legend(f, [(t[k], FAIL[k], "box") for k in ("right", "wrong", "unfinished", "format")], 0.04, 0.995, ncol=4)
    D.legend(f, [(t[k], WHY[k], "box") for k in ("concept", "execution", "unclear")], 0.04, 0.555, ncol=3)
    save(f, "numhard_failures", code)


def report_shortfalls(ft, t, code):
    rp = ft["d_report"]
    order = ["rl_seed1", "sft", "base_medium"]
    f = D.fig(13.0, 6.4)
    a1 = D.axes(f, [0.16, 0.56, 0.66, 0.26], grid=None)
    a2 = D.axes(f, [0.16, 0.08, 0.66, 0.26], grid=None)
    for i, m in enumerate(order):
        v = rp[m]["criteria_verdicts"]
        n = sum(v.values())
        stacked(a1, i, [(k, v.get(k, 0), VERDICT[k]) for k in ("met", "partial", "missed")], n,
                fmt=lambda x: f"{x:,.0f}", min_label=5.0)
        a1.text(101.5, i, f"n = {n}", va="center", fontsize=D.FS_SMALL, color=R.INK2)
        s = rp[m]["shortfall_types"]
        ns = sum(s.values())
        stacked(a2, i, [(k, s.get(k, 0), SHORT[k]) for k in SHORT], ns, fmt=lambda x: f"{x:,.0f}", min_label=5.0)
        a2.text(101.5, i, f"n = {ns}", va="center", fontsize=D.FS_SMALL, color=R.INK2)
    for ax, key in ((a1, "verdicts"), (a2, "short")):
        ax.set_yticks(range(3))
        ax.set_yticklabels([MODEL_NAME[code][m] for m in order], fontsize=D.FS_SMALL)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.set_xticklabels(["0%", "50%", "100%"])
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", colors=R.MUTED)
        D.label(ax, t[key], y=1.06)
    D.legend(f, [(t[k], VERDICT[k], "box") for k in VERDICT], 0.04, 0.995, ncol=3)
    D.legend(f, [(t[k], SHORT[k], "box") for k in SHORT], 0.04, 0.505, ncol=5)
    save(f, "report_shortfalls", code)


def cut_off(ft, t, code):
    tr = ft["cut_off_32k"]
    sets = [k for k in ("gpqa_diamond", "livecodebench_post_2025_01", "d_num_hard", "d_fresh", "mmlu_pro_2k",
                        "d_code", "d_num") if all(k in tr[m] for m in D.MODELS)]
    f = D.fig(13.0, 5.2)
    ax = D.axes(f, [0.07, 0.15, 0.90, 0.64])
    w = 0.26
    for j, m in enumerate(D.MODELS):
        xs = [i + (j - 1) * w for i in range(len(sets))]
        ys = [100 * tr[m][k]["cut_off"] for k in sets]
        ax.bar(xs, ys, width=w * 0.92, color=D.MODEL_COLOR[m], edgecolor=D.PAGE, linewidth=2 * R.PT)
        for x, y in zip(xs, ys):
            if y >= 1:
                ax.text(x, y + 0.8, f"{y:.0f}", ha="center", va="bottom", fontsize=D.FS_SMALL - 3, color=R.INK2)
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels([t["sets"][k] for k in sets], fontsize=D.FS_SMALL)
    ax.yaxis.set_major_formatter(D.plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    D.label(ax, t["cut"])
    D.legend(f, [(MODEL_NAME[code][m], D.MODEL_COLOR[m], "box") for m in D.MODELS], 0.04, 0.99, ncol=3)
    save(f, "cut_off", code)


def main() -> None:
    data = D.DATA
    rv = json.loads((data / "reward_vs_eval.json").read_text())
    rs = json.loads((data / "research_1110.json").read_text())
    run_results = json.loads((data / "run_results.json").read_text())
    rl = json.loads((data / "rl.json").read_text())
    for code in ("en", "zh"):
        D.P.lang(code)
        t = T[code]
        reward_agreement(rv, t, code)
        rubric_atomicity(rs, t, code)
        grpo_groups(run_results, rl, t, code)
        test_time(rs, run_results, t, code)
        ftp = data / "failure_types.json"
        if ftp.exists():
            ft = json.loads(ftp.read_text())
            numhard_failures(ft, T2[code], code)
            report_shortfalls(ft, T2[code], code)
            cut_off(ft, T2[code], code)
    print(OUT)


if __name__ == "__main__":
    main()
