"""Draw every figure in README.md and README.zh-CN.md from the result files in docs/talk/data/.

Why a module of its own: the talk and report figures (scripts/talk/) carry titles, notes and
provenance tags, because each one stands alone on a slide or a page. A README figure sits under
a heading and above a caption that already say what it shows, so here every figure carries as
little text as possible: no titles, short axis labels, one legend at most, and values only where
the README text points at them. Figures with words come in two languages, `_en` and `_zh`.

Every number is read from a result file; none is typed into this file (the lesson of
nanogpt-seis-posttrain PITFALLS F2, where figures drifted away from the logs they claimed to show):
- docs/talk/data/run.json            scores, token budgets, the bank pipeline, the SFT mix
                                     (written on the cluster by scripts/talk/collect.py)
- docs/talk/data/rl.json             per-step RL metrics and GPU memory samples (scripts/talk/collect_rl.py)
- docs/talk/data/reward_vs_eval.json, judge_drift.json, refusal.json, failure_types.json, research_1110.json
- docs/talk/data/numhard_64k.json   D-Num-hard with a 64K budget (scripts/talk/numhard_continue.py)
- docs/talk/data/campaign_hours.json and last_stage_memory.json: values measured by hand during the
  run, each with the note it came from (the run log)

Colours follow the talk deck (scripts/talk/render.py), whose categorical slots were checked with the
data-viz palette validator: models are base grey, SFT blue, RL orange; tasks T1 blue, T2 orange,
T3 green, T4 amber; answer outcomes use the deck's failure slots.

    ~/miniconda3/bin/python scripts/readme_figures.py            # → assets/<name>_{en,zh}.png and .pdf
    ~/miniconda3/bin/python scripts/readme_figures.py workflow   # one figure
"""

from __future__ import annotations

import datetime
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

repository_root = Path(__file__).resolve().parents[1]
result_directory = repository_root / "docs" / "talk" / "data"
asset_directory = repository_root / "assets"
figure_languages = ("en", "zh")
png_dots_per_inch = 300

# ---------------------------------------------------------------------------
# colours: the talk deck's, so a reader who saw the slides reads these the same way
# ---------------------------------------------------------------------------
page_color = "#ffffff"
ink_color = "#0b0b0b"
secondary_ink_color = "#52514e"
muted_ink_color = "#898781"
grid_color = "#e1e0d9"
axis_color = "#c3c2b7"
model_keys = ("base_medium", "sft", "rl_seed1")
model_color = {"base_medium": "#8a8984", "sft": "#2a78d6", "rl_seed1": "#eb6834"}
task_color = {"t1_num": "#2a78d6", "t2_code": "#eb6834", "t3_report": "#1baf7a", "t4_general": "#eda100"}
outcome_color = {"right": "#b7d3f6", "unfinished": "#e87ba4", "wrong_value": "#6250d6", "format": "#008300"}
cause_color = {"execution": "#1baf7a", "concept": "#6250d6", "other": "#c3c2b7"}
shortfall_color = {"concept": "#6250d6", "coverage": "#eda100", "quantitative": "#86b6ef"}
judge_color = {"primary": "#1baf7a", "second": "#6250d6"}
planned_time_color, lost_time_color = "#2a78d6", "#e87ba4"

# ---------------------------------------------------------------------------
# the only words in the figures, in both languages
# ---------------------------------------------------------------------------
label_text = {
    "en": {
        "model": {"base_medium": "base", "sft": "SFT", "rl_seed1": "RL"},
        "workflow_model": {"base_medium": "Qwen3.5-9B", "sft": "SFT", "rl_seed1": "RL, step 20"},
        "workflow_score": "D-Num-hard", "workflow_tokens": "tokens / answer",
        "conversations": "conversations", "problems": "problems",
        "score": "score", "tokens_thousands": "tokens (K)",
        "domain_sets": ["D-Num", "D-Num-hard", "D-Fresh", "D-Code"],
        "general_sets": ["GPQA", "MMLU-Pro", "IFEval", "LiveCode", "tools"],
        "percent_right": "% right", "budget_thousands": "token budget (K)", "samples_k": "samples (k)",
        "outcome": {"right": "right", "unfinished": "unfinished", "wrong_value": "wrong value",
                    "format": "format"},
        "bank_stages": ["generated", "quality", "dedup", "decontam.", "verified", "difficulty"],
        "task": {"t1_num": "T1 numeric", "t2_code": "T2 code", "t3_report": "T3 report", "t4_general": "T4 general"},
        "gib": "GiB", "before_fixes": "before", "after_fixes": "after", "hours": "hours",
        "fixed_state": "weights, gradients, Adam", "loss_path": "loss at 20,841 tokens",
        "step": "step", "tokens": "tokens", "accuracy": "accuracy", "no_signal": "no-signal groups",
        "reward": "reward", "grader": "eval grader", "gemma": "Gemma", "luna": "GPT-6 Luna",
        "pass": "pass", "fail": "fail",
        "win_rate": "win rate vs base (%)", "dev_score": "dev score",
        "judge": {"primary": "Gemma (reward judge)", "second": "GPT-6 Luna (eval only)"},
        "cause": {"execution": "execution", "concept": "concept", "other": "format, unclear"},
        "answers": "answers of 1,280", "share_of_shortfalls": "% of shortfalls",
        "shortfall": {"concept": "concept", "coverage": "coverage", "quantitative": "quantitative"},
        "refused": "% refused", "should_refuse": "should refuse (300)", "benign": "benign (350)",
        "strict": "strict", "broad": "broad",
        "kind": {"planned": "planned stages", "lost": "incidents, waits, holds"},
        "segment": {"baseline_eval": "baseline eval", "sft_blocked": "SFT blocked", "sft": "SFT",
                    "sft_eval_and_gate": "SFT eval", "rl": "RL", "final_eval": "final eval"},
        "hours_unit": "h",
    },
    "zh": {
        "model": {"base_medium": "基座", "sft": "SFT", "rl_seed1": "RL"},
        "workflow_model": {"base_medium": "Qwen3.5-9B", "sft": "SFT", "rl_seed1": "RL，第 20 步"},
        "workflow_score": "D-Num-hard", "workflow_tokens": "每个回答的 token",
        "conversations": "条对话", "problems": "道题",
        "score": "得分", "tokens_thousands": "token 数（千）",
        "domain_sets": ["D-Num", "D-Num-hard", "D-Fresh", "D-Code"],
        "general_sets": ["GPQA", "MMLU-Pro", "IFEval", "LiveCode", "工具调用"],
        "percent_right": "正确率（%）", "budget_thousands": "token 预算（千）", "samples_k": "样本数 k",
        "outcome": {"right": "正确", "unfinished": "未完成", "wrong_value": "数值错误", "format": "格式"},
        "bank_stages": ["生成", "质量", "去重", "去污染", "验证", "难度"],
        "task": {"t1_num": "T1 数值", "t2_code": "T2 代码", "t3_report": "T3 报告", "t4_general": "T4 通用"},
        "gib": "GiB", "before_fixes": "修复前", "after_fixes": "修复后", "hours": "小时",
        "fixed_state": "权重、梯度、Adam", "loss_path": "20,841 token 时的损失",
        "step": "步", "tokens": "token 数", "accuracy": "准确率", "no_signal": "无信号的组",
        "reward": "奖励", "grader": "评测判分器", "gemma": "Gemma", "luna": "GPT-6 Luna",
        "pass": "通过", "fail": "不通过",
        "win_rate": "对基座的胜率（%）", "dev_score": "开发集得分",
        "judge": {"primary": "Gemma（奖励裁判）", "second": "GPT-6 Luna（仅评测）"},
        "cause": {"execution": "执行", "concept": "概念", "other": "格式、不明"},
        "answers": "1,280 个回答中", "share_of_shortfalls": "占不足之处（%）",
        "shortfall": {"concept": "概念", "coverage": "覆盖", "quantitative": "定量"},
        "refused": "拒答率（%）", "should_refuse": "应拒答（300）", "benign": "无害（350）",
        "strict": "严格", "broad": "宽泛",
        "kind": {"planned": "计划内阶段", "lost": "故障、等待、挂起"},
        "segment": {"baseline_eval": "基线评测", "sft_blocked": "SFT 受阻", "sft": "SFT",
                    "sft_eval_and_gate": "SFT 评测", "rl": "RL", "final_eval": "最终评测"},
        "hours_unit": "小时",
    },
}


# ---------------------------------------------------------------------------
# style and saving
# ---------------------------------------------------------------------------
def find_simplified_chinese_font() -> Path:
    """Return a Simplified Chinese Noto Sans font file, extracting it from the system collection once.

    matplotlib reads only the first face of a .ttc collection, which in Noto Sans CJK is the Japanese
    one; its glyph shapes differ from Simplified Chinese, so the SC face is saved as its own file.
    """
    simplified_chinese_font = Path(os.environ.get("DPT_CJK_FONT", Path.home() / ".cache" / "dpt" /
                                                  "NotoSansCJKsc-Regular.otf"))
    if not simplified_chinese_font.exists():
        from fontTools.ttLib import TTCollection

        system_collection = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        for font_face in TTCollection(system_collection).fonts:
            if font_face["name"].getDebugName(1) == "Noto Sans CJK SC":
                simplified_chinese_font.parent.mkdir(parents=True, exist_ok=True)
                font_face.save(str(simplified_chinese_font))
                break
    return simplified_chinese_font


def set_figure_style(language: str) -> None:
    """One quiet style for every figure: light dashed y-grid, no top or right spine, no frames."""
    font_families = ["DejaVu Sans"]
    if language == "zh":
        font_manager.fontManager.addfont(str(find_simplified_chinese_font()))
        font_families = ["DejaVu Sans", "Noto Sans CJK SC"]        # Latin glyphs from DejaVu, Chinese from Noto
    plt.rcParams.update({
        "font.family": font_families, "font.size": 13, "text.color": ink_color,
        "axes.edgecolor": axis_color, "axes.linewidth": 1.0, "axes.labelcolor": secondary_ink_color,
        "axes.labelsize": 13, "axes.spines.top": False, "axes.spines.right": False,
        "axes.facecolor": page_color, "figure.facecolor": page_color, "savefig.facecolor": page_color,
        "xtick.color": secondary_ink_color, "ytick.color": secondary_ink_color,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "grid.color": grid_color, "grid.linestyle": "--",
        "grid.linewidth": 0.8, "legend.frameon": False, "legend.fontsize": 12, "axes.unicode_minus": False,
        "pdf.fonttype": 42,
    })


def add_horizontal_grid(axes) -> None:
    axes.grid(True, axis="y")
    axes.set_axisbelow(True)


def save_figure(figure, figure_name: str, language: str) -> None:
    """Write assets/<name>_<language>.png (300 dpi, for the README) and a vector .pdf next to it."""
    asset_directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        output_path = asset_directory / f"{figure_name}_{language}.{suffix}"
        figure.savefig(output_path, dpi=png_dots_per_inch, bbox_inches="tight", pad_inches=0.06)
    plt.close(figure)
    print(f"  assets/{figure_name}_{language}.png + .pdf")


def load_result_file(file_name: str) -> dict:
    return json.loads((result_directory / file_name).read_text())


def load_all_result_files() -> dict:
    return {
        "run": load_result_file("run.json"),
        "rl": load_result_file("rl.json"),
        "reward_vs_eval": load_result_file("reward_vs_eval.json"),
        "judge_drift": load_result_file("judge_drift.json"),
        "refusal": load_result_file("refusal.json"),
        "failure_types": load_result_file("failure_types.json"),
        "numhard_64k": load_result_file("numhard_64k.json"),
        "research_1110": load_result_file("research_1110.json"),
        "campaign_hours": load_result_file("campaign_hours.json"),
        "last_stage_memory": load_result_file("last_stage_memory.json"),
    }


def eval_set_result(results: dict, model_key: str, set_name: str) -> dict:
    """D-Num-hard was scored in its own runs (d_num_hard/<model>); every other set in the model's main run."""
    run_key = f"d_num_hard/{model_key}" if set_name == "d_num_hard" else model_key
    return results["run"]["eval"]["runs"][run_key]["sets"][set_name]


def thousands_label(value: float) -> str:
    return f"{value / 1000:.1f}K"


# ---------------------------------------------------------------------------
# 1. the pipeline strip at the top of the README
# ---------------------------------------------------------------------------
def draw_workflow_strip(results: dict, language: str) -> None:
    """Three models in boxes; each arrow carries the data that trained the next model.

    The big number is D-Num-hard (avg@8), the held-out slice with headroom; the small number is the
    mean output length on it.
    """
    text = label_text[language]
    hard_scores = [eval_set_result(results, model, "d_num_hard")["score"] for model in model_keys]
    hard_mean_tokens = [eval_set_result(results, model, "d_num_hard")["mean_output_tokens"] for model in model_keys]
    sft_conversation_count = sum(results["run"]["sft_mix"]["conversations_by_subset"].values())
    rl_problem_count = sum(task["after_difficulty"] for task in results["run"]["pipeline"]["by_task"].values())

    figure = plt.figure(figsize=(12.4, 2.7))
    axes = figure.add_axes([0, 0, 1, 1])
    axes.set_xlim(0, 12.4)
    axes.set_ylim(0, 2.7)
    axes.axis("off")
    box_width, box_height, box_bottom = 2.45, 2.2, 0.25
    box_lefts = [2.1, 5.95, 9.8]
    score_height, token_height, name_height = box_bottom + 1.05, box_bottom + 0.42, box_bottom + 1.78

    axes.text(1.95, score_height, text["workflow_score"], ha="right", va="center", fontsize=12,
              color=muted_ink_color)
    axes.text(1.95, token_height, text["workflow_tokens"], ha="right", va="center", fontsize=12,
              color=muted_ink_color)
    for box_index, model in enumerate(model_keys):
        box_left = box_lefts[box_index]
        axes.add_patch(FancyBboxPatch((box_left, box_bottom), box_width, box_height,
                                      boxstyle="round,pad=0.02,rounding_size=0.12", linewidth=1.6,
                                      edgecolor=model_color[model], facecolor=model_color[model] + "14"))
        box_center = box_left + box_width / 2
        axes.text(box_center, name_height, text["workflow_model"][model], ha="center", va="center",
                  fontsize=15, fontweight="bold", color=ink_color)
        axes.text(box_center, score_height, f"{hard_scores[box_index]:.1f}", ha="center", va="center",
                  fontsize=26, fontweight="bold", color=model_color[model])
        axes.text(box_center, token_height, thousands_label(hard_mean_tokens[box_index]), ha="center",
                  va="center", fontsize=13, color=secondary_ink_color)

    arrow_tags = [(f"{sft_conversation_count:,}", text["conversations"]), (f"{rl_problem_count:,}", text["problems"])]
    for arrow_index, (count_text, unit_text) in enumerate(arrow_tags):
        arrow_start = box_lefts[arrow_index] + box_width + 0.08
        arrow_end = box_lefts[arrow_index + 1] - 0.08
        arrow_height = box_bottom + box_height / 2
        axes.add_patch(FancyArrowPatch((arrow_start, arrow_height - 0.25), (arrow_end, arrow_height - 0.25),
                                       arrowstyle="-|>", mutation_scale=18, linewidth=1.6, color=secondary_ink_color))
        arrow_center = (arrow_start + arrow_end) / 2
        axes.text(arrow_center, arrow_height + 0.38, count_text, ha="center", va="center", fontsize=13,
                  fontweight="bold", color=secondary_ink_color)
        axes.text(arrow_center, arrow_height + 0.05, unit_text, ha="center", va="center", fontsize=11.5,
                  color=muted_ink_color)
    save_figure(figure, "workflow", language)


# ---------------------------------------------------------------------------
# 2. results by stage: domain sets, general benchmarks, answer length
# ---------------------------------------------------------------------------
def draw_grouped_bars(axes, values_by_model: dict, halfwidths_by_model: dict, group_count: int) -> None:
    bar_width = 0.25
    for model_index, model in enumerate(model_keys):
        positions = [group + (model_index - 1) * (bar_width + 0.02) for group in range(group_count)]
        axes.bar(positions, values_by_model[model], width=bar_width, color=model_color[model], zorder=2)
        axes.errorbar(positions, values_by_model[model], yerr=halfwidths_by_model[model], fmt="none",
                      ecolor=ink_color, elinewidth=1.0, capsize=2.5, zorder=3)


def draw_results_by_stage(results: dict, language: str) -> None:
    """Scores with 95% intervals on four domain sets and five general benchmarks, then answer length.

    D-Report is left out here: it is a win rate against the base (the base is 50 by definition), and
    its two judges disagree, so it gets a figure of its own (two_judges).
    """
    text = label_text[language]
    runs = results["run"]["eval"]["runs"]
    domain_set_names = ["d_num", "d_num_hard", "d_fresh", "d_code"]
    general_set_names = ["gpqa_diamond", "mmlu_pro_2k", "ifeval", "livecodebench_post_2025_01", "tool_calling"]

    domain_scores, domain_halfwidths, general_scores, general_halfwidths = {}, {}, {}, {}
    for model in model_keys:
        domain_results = [eval_set_result(results, model, set_name) for set_name in domain_set_names]
        domain_scores[model] = [result["score"] for result in domain_results]
        domain_halfwidths[model] = [result["ci"]["halfwidth"] for result in domain_results]
        general_results = [runs[model]["general"][set_name] for set_name in general_set_names]
        general_scores[model] = [result["score"] for result in general_results]
        general_halfwidths[model] = [result["ci"]["halfwidth"] for result in general_results]
    mean_output_tokens = [runs[model]["mean_output_tokens"] for model in model_keys]

    figure, (domain_axes, general_axes, token_axes) = plt.subplots(
        1, 3, figsize=(14.5, 4.3), gridspec_kw={"width_ratios": [4, 5, 1.6], "wspace": 0.16})
    panels = ((domain_axes, domain_scores, domain_halfwidths, text["domain_sets"]),
              (general_axes, general_scores, general_halfwidths, text["general_sets"]))
    for axes, scores, halfwidths, tick_labels in panels:
        draw_grouped_bars(axes, scores, halfwidths, len(tick_labels))
        axes.set_xticks(range(len(tick_labels)), tick_labels)
        axes.set_ylim(0, 102)
        axes.set_yticks([0, 25, 50, 75, 100])
        add_horizontal_grid(axes)
        axes.tick_params(axis="x", length=0)
    domain_axes.set_ylabel(text["score"])
    general_axes.set_yticklabels([])

    for model_index, model in enumerate(model_keys):
        token_axes.bar(model_index, mean_output_tokens[model_index] / 1000, width=0.62, color=model_color[model],
                       zorder=2)
        token_axes.text(model_index, mean_output_tokens[model_index] / 1000 + 0.3,
                        f"{mean_output_tokens[model_index] / 1000:.1f}", ha="center", va="bottom", fontsize=12,
                        color=secondary_ink_color)
    token_axes.set_xticks([])
    token_axes.set_ylabel(text["tokens_thousands"])
    token_axes.set_ylim(0, max(mean_output_tokens) / 1000 * 1.18)
    add_horizontal_grid(token_axes)

    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=model_color[model]) for model in model_keys]
    domain_axes.legend(legend_handles, [text["model"][model] for model in model_keys], ncol=3,
                       loc="lower left", bbox_to_anchor=(0.0, 1.0), handlelength=1.0, columnspacing=1.4)
    save_figure(figure, "results_by_stage", language)


# ---------------------------------------------------------------------------
# 3. test-time compute: accuracy within a token budget, and majority votes
# ---------------------------------------------------------------------------
def pooled_accuracy_within_budget(results: dict, set_name: str, budget_entry_keys: list[str]):
    """Pool every run of one model: a sample counts if it is right within the budget, as if the cap were there."""
    budgets, right_within_budget, sample_count = None, None, 0
    for entry_key in budget_entry_keys:
        set_budget = results["run"]["budget"][entry_key]["sets"][set_name]
        if budgets is None:
            budgets, right_within_budget = set_budget["budgets"], [0] * len(set_budget["budgets"])
        assert set_budget["budgets"] == budgets, f"{entry_key} uses other budgets; pooling would mix them"
        right_within_budget = [total + count for total, count in zip(right_within_budget,
                                                                     set_budget["correct_within"])]
        sample_count += set_budget["n_samples"]
    # budgets are powers of two (2,048 … 32,768), so "K" here is 1,024 tokens and 16K is the 16,384-token cap
    return [budget / 1024 for budget in budgets], [100 * right / sample_count for right in right_within_budget]


def read_rl_response_cap_tokens() -> int:
    """The RL response cap from the model config, read with a plain line scan to avoid a YAML dependency."""
    model_config_lines = (repository_root / "configs" / "model" / "qwen35_9b.yaml").read_text().splitlines()
    cap_lines = [line for line in model_config_lines if line.strip().startswith("rl_max_response:")]
    assert len(cap_lines) == 1, "expected exactly one rl_max_response in configs/model/qwen35_9b.yaml"
    return int(cap_lines[0].split(":", 1)[1].split("#", 1)[0])


def draw_test_time_compute(results: dict, language: str) -> None:
    text = label_text[language]
    d_num_budget_entries = {
        "base_medium": ["base_medium", "base_medium_k64"] + [f"noise__base_medium_s{seed}" for seed in range(10)],
        "sft": ["sft", "sft_k64"],
        "rl_seed1": ["rl_seed1", "rl_seed1_k64"],
    }
    figure, (d_num_axes, hard_axes, vote_axes) = plt.subplots(
        1, 3, figsize=(14.5, 4.1), gridspec_kw={"width_ratios": [1, 1, 1], "wspace": 0.22})
    # RL cut answers at 16K tokens (configs/model/qwen35_9b.yaml thinking.rl_max_response; README §6.1);
    # the dotted line shows how much of each model's accuracy fits under that cap
    rl_response_cap_thousands = read_rl_response_cap_tokens() / 1024
    for axes, set_name, set_tag in ((d_num_axes, "d_num", "D-Num"), (hard_axes, "d_num_hard", "D-Num-hard")):
        for model in model_keys:
            entry_keys = d_num_budget_entries[model] if set_name == "d_num" else [f"d_num_hard__{model}"]
            budgets, accuracy = pooled_accuracy_within_budget(results, set_name, entry_keys)
            axes.plot(budgets, accuracy, color=model_color[model], linewidth=2.2, marker="o", markersize=4.5,
                      label=text["model"][model])
        axes.axvline(rl_response_cap_thousands, color=muted_ink_color, linewidth=1.0, linestyle=":", zorder=1)
        axes.set_xticks([2, 8, 16, 24, 32])
        axes.set_xlabel(text["budget_thousands"])
        axes.set_ylim(0, 100)
        axes.text(0.03, 0.96, set_tag, transform=axes.transAxes, ha="left", va="top", fontsize=13,
                  fontweight="bold", color=ink_color)
        add_horizontal_grid(axes)
    d_num_axes.set_ylabel(text["percent_right"])
    d_num_axes.legend(loc="lower right", handlelength=1.4)

    for model in model_keys:
        majority_vote = results["research_1110"]["d_num_voting"][model]["maj_at_k"]
        sample_counts = sorted(int(k) for k in majority_vote)
        vote_axes.plot(sample_counts, [majority_vote[str(k)] for k in sample_counts], color=model_color[model],
                       linewidth=2.2, marker="o", markersize=4.5)
    vote_axes.set_xscale("log", base=2)
    vote_axes.set_xticks([1, 2, 4, 8, 16, 32, 64], ["1", "2", "4", "8", "16", "32", "64"])
    vote_axes.set_xlabel(text["samples_k"])
    vote_axes.set_ylim(88, 100.5)
    vote_axes.text(0.97, 0.05, "D-Num maj@k", transform=vote_axes.transAxes, ha="right", va="bottom", fontsize=13,
                   fontweight="bold", color=ink_color)
    add_horizontal_grid(vote_axes)
    save_figure(figure, "test_time_compute", language)


# ---------------------------------------------------------------------------
# 4. where the samples go: right, unfinished, wrong value, format
# ---------------------------------------------------------------------------
def outcome_shares(reasons: dict) -> dict:
    """Fold the grader's reasons into four outcomes; every reason that is not one of three is 'format'."""
    total = sum(reasons.values())
    right, unfinished, wrong = reasons.get("ok", 0), reasons.get("no_think_end", 0), reasons.get("wrong_value", 0)
    return {"right": 100 * right / total, "unfinished": 100 * unfinished / total,
            "wrong_value": 100 * wrong / total, "format": 100 * (total - right - unfinished - wrong) / total}


def draw_d_num_hard_at_64k(results: dict, language: str) -> None:
    """D-Num-hard outcomes at the eval's 32K budget and at 64K, for each model.

    The 64K outcome continues every answer that was cut off at 32K, with the same model and sampling
    (scripts/talk/numhard_continue.py): a 64K sample is the same as a 32K sample up to token 32K.
    """
    text = label_text[language]
    budget_rows = results["numhard_64k"]["rows"]
    outcome_order = ("right", "unfinished", "wrong_value", "format")
    figure, axes = plt.subplots(figsize=(12.4, 4.4))
    row_positions, row_labels = [], []
    for model_index, model in enumerate(model_keys):
        group_top = model_index * 3.0
        axes.text(-1.0, group_top - 0.75, text["model"][model], ha="left", va="center", fontsize=13,
                  fontweight="bold", color=ink_color, clip_on=False)
        for budget_index, (budget_key, budget_label) in enumerate((("at_32k_pct", "32K"), ("at_64k_pct", "64K"))):
            row_position = group_top + budget_index
            row_positions.append(row_position)
            row_labels.append(budget_label)
            shares = budget_rows[model][budget_key]
            segment_left = 0.0
            for outcome in outcome_order:
                share = shares.get(outcome, 0.0)
                axes.barh(row_position, share, left=segment_left, height=0.68, color=outcome_color[outcome],
                          edgecolor=page_color, linewidth=1.2)
                if share >= 4.0:
                    label_color = ink_color if outcome == "right" else page_color
                    axes.text(segment_left + share / 2, row_position, f"{share:.0f}", ha="center", va="center",
                              fontsize=11.5, color=label_color)
                segment_left += share
    axes.set_yticks(row_positions, row_labels)
    axes.invert_yaxis()
    axes.set_xlim(0, 100)
    axes.set_xticks([0, 25, 50, 75, 100])
    axes.tick_params(axis="y", length=0)
    axes.spines["left"].set_visible(False)
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=outcome_color[outcome]) for outcome in outcome_order]
    axes.legend(legend_handles, [text["outcome"][outcome] for outcome in outcome_order],
                ncol=4, loc="lower left", bbox_to_anchor=(0.0, 1.0), handlelength=1.0, columnspacing=1.4)
    save_figure(figure, "d_num_hard_at_64k", language)


def draw_where_samples_go(results: dict, language: str) -> None:
    text = label_text[language]
    figure, axes = plt.subplots(figsize=(12.4, 4.0))
    row_positions, row_labels = [], []
    for group_index, (set_name, set_tag) in enumerate((("d_num", "D-Num"), ("d_num_hard", "D-Num-hard"))):
        group_top = group_index * 4.0
        axes.text(-1.0, group_top - 0.75, set_tag, ha="left", va="center", fontsize=13, fontweight="bold",
                  color=ink_color, clip_on=False)
        for model_index, model in enumerate(model_keys):
            row_position = group_top + model_index
            row_positions.append(row_position)
            row_labels.append(text["model"][model])
            shares = outcome_shares(eval_set_result(results, model, set_name)["reasons"])
            segment_left = 0.0
            for outcome in ("right", "unfinished", "wrong_value", "format"):
                axes.barh(row_position, shares[outcome], left=segment_left, height=0.68,
                          color=outcome_color[outcome], edgecolor=page_color, linewidth=1.2)
                if shares[outcome] >= 4.0:
                    label_color = ink_color if outcome == "right" else page_color
                    axes.text(segment_left + shares[outcome] / 2, row_position, f"{shares[outcome]:.0f}",
                              ha="center", va="center", fontsize=11.5, color=label_color)
                segment_left += shares[outcome]
    axes.set_yticks(row_positions, row_labels)
    axes.invert_yaxis()
    axes.set_xlim(0, 100)
    axes.set_xticks([0, 25, 50, 75, 100])
    axes.tick_params(axis="y", length=0)
    axes.spines["left"].set_visible(False)
    outcome_order = ("right", "unfinished", "wrong_value", "format")
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=outcome_color[outcome]) for outcome in outcome_order]
    axes.legend(legend_handles, [text["outcome"][outcome] for outcome in outcome_order],
                ncol=4, loc="lower left", bbox_to_anchor=(0.0, 1.0), handlelength=1.0, columnspacing=1.4)
    save_figure(figure, "where_samples_go", language)


# ---------------------------------------------------------------------------
# 5. the RL problem bank, stage by stage
# ---------------------------------------------------------------------------
def draw_bank_funnel(results: dict, language: str) -> None:
    text = label_text[language]
    by_task = results["run"]["pipeline"]["by_task"]
    stage_keys = ["ingest", "after_quality", "after_dedup", "after_decontam", "after_verify", "after_difficulty"]
    figure, axes = plt.subplots(figsize=(11.5, 4.0))
    stage_totals = [0] * len(stage_keys)
    for task in ("t1_num", "t2_code", "t3_report", "t4_general"):
        task_counts = [by_task[task][stage] for stage in stage_keys]
        axes.bar(range(len(stage_keys)), task_counts, bottom=stage_totals, width=0.62, color=task_color[task],
                 edgecolor=page_color, linewidth=1.0, label=text["task"][task], zorder=2)
        stage_totals = [total + count for total, count in zip(stage_totals, task_counts)]
    for stage_index, stage_total in enumerate(stage_totals):
        axes.text(stage_index, stage_total + 700, f"{stage_total:,}", ha="center", va="bottom", fontsize=12,
                  color=secondary_ink_color)
    axes.set_xticks(range(len(stage_keys)), text["bank_stages"])
    axes.tick_params(axis="x", length=0)
    axes.set_ylim(0, max(stage_totals) * 1.13)
    axes.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _: f"{value / 1000:.0f}K"))
    add_horizontal_grid(axes)
    axes.legend(ncol=4, loc="lower left", bbox_to_anchor=(0.0, 1.0), handlelength=1.0, columnspacing=1.4)
    save_figure(figure, "bank_funnel", language)


# ---------------------------------------------------------------------------
# 6. memory on the last pipeline stage
# ---------------------------------------------------------------------------
def parse_log_time(timestamp_text: str) -> datetime.datetime:
    return datetime.datetime.strptime(timestamp_text, "%Y-%m-%d %H:%M")


def draw_last_stage_memory(results: dict, language: str) -> None:
    """Left: the budget at the longest SFT row before and after the fixes. Right: memory during the SFT run."""
    text = label_text[language]
    memory = results["last_stage_memory"]
    capacity = memory["gpu_capacity_gib"]
    longest_row_tokens = memory["longest_sft_row_tokens"]
    loss_before = memory["before_fixes"]["loss_mib_per_token"] * longest_row_tokens / 1024
    fixed_state = [memory["before_fixes"]["fixed_state_gib"], memory["after_fixes"]["fixed_state_gib"]]
    loss_path = [loss_before, memory["after_fixes"]["loss_at_longest_row_gib"]]

    figure, (budget_axes, series_axes) = plt.subplots(
        1, 2, figsize=(12.4, 4.0), gridspec_kw={"width_ratios": [1, 2.2], "wspace": 0.18})
    budget_axes.bar([0, 1], fixed_state, width=0.58, color=secondary_ink_color, label=text["fixed_state"], zorder=2)
    budget_axes.bar([0, 1], loss_path, bottom=fixed_state, width=0.58, color=model_color["sft"],
                    label=text["loss_path"], zorder=2)
    for bar_index in (0, 1):
        total = fixed_state[bar_index] + loss_path[bar_index]
        budget_axes.text(bar_index, total - 1.5, f"{total:.1f}", ha="center", va="top", fontsize=12.5,
                         fontweight="bold", color=page_color)
    budget_axes.set_xticks([0, 1], [text["before_fixes"], text["after_fixes"]])
    budget_axes.tick_params(axis="x", length=0)
    budget_axes.set_ylabel(text["gib"])
    budget_axes.set_ylim(0, 90)

    sft_segment = next(segment for segment in results["campaign_hours"]["segments"] if segment["name"] == "sft")
    sft_start, sft_end = parse_log_time(sft_segment["start"]), parse_log_time(sft_segment["end"])
    samples = [sample for sample in results["rl"]["gpu_memory_stage7_sft"]
               if sft_start <= parse_log_time(sample["t"]) <= sft_end]
    series_axes.plot([(parse_log_time(sample["t"]) - sft_start).total_seconds() / 3600 for sample in samples],
                     [sample["mib"] / 1024 for sample in samples], color=model_color["sft"], linewidth=1.8)
    series_axes.set_xlabel(text["hours"])
    series_axes.set_ylim(0, 45)
    for axes in (budget_axes, series_axes):
        axes.axhline(capacity, color=ink_color, linewidth=1.1, linestyle="--", zorder=3)
        add_horizontal_grid(axes)
    budget_axes.legend(ncol=2, loc="lower left", bbox_to_anchor=(0.0, 1.0), handlelength=1.0, columnspacing=1.4)
    save_figure(figure, "last_stage_memory", language)


# ---------------------------------------------------------------------------
# 7. RL run 1 step by step: length, training accuracy, groups without signal
# ---------------------------------------------------------------------------
def draw_rl_steps(results: dict, language: str) -> None:
    text = label_text[language]
    step_metrics = results["rl"]["metrics"]
    steps = [metric["step"] for metric in step_metrics]
    selected_step = 20                  # picked on the dev pass (dpt.eval.select_checkpoint); see README §6.4
    # metric keys in rl.json, one per panel; "{task}" is filled in per line. A group without signal is one
    # whose 8 samples all got the same reward, so its advantages are all zero (README §6.3).
    panel_metrics = (("task/{task}/len_mean", text["tokens"], "log"), ("task/{task}/acc", text["accuracy"], "linear"),
                     ("filter/{task}/zero_adv_group_frac", text["no_signal"], "linear"))
    figure, panel_axes = plt.subplots(1, 3, figsize=(14.5, 4.0), gridspec_kw={"wspace": 0.24})
    for axes, (metric_key_pattern, axis_label, y_scale) in zip(panel_axes, panel_metrics):
        for task in ("t1_num", "t2_code", "t4_general"):
            values = [metric.get(metric_key_pattern.format(task=task)) for metric in step_metrics]
            if y_scale == "linear":                     # shares in [0, 1], drawn as percent
                values = [None if value is None else 100 * value for value in values]
            axes.plot(steps, values, color=task_color[task], linewidth=2.0, label=text["task"][task])
        axes.axvline(selected_step, color=muted_ink_color, linewidth=1.0, linestyle=":", zorder=1)
        axes.set_yscale(y_scale)
        axes.set_xlabel(text["step"])
        axes.set_ylabel(axis_label)
        axes.set_xticks([1, 20, 40, 56])
        add_horizontal_grid(axes)
    panel_axes[1].set_ylim(-3, 103)
    panel_axes[2].set_ylim(-3, 103)
    panel_axes[0].legend(loc="lower left", handlelength=1.4)
    save_figure(figure, "rl_steps", language)


# ---------------------------------------------------------------------------
# 8. do the rewards agree with an independent check?
# ---------------------------------------------------------------------------
def draw_confusion_square(axes, counts: dict, reward_label: str, reference_label: str, text: dict) -> None:
    """Rows: what the reward said. Columns: what the reference said. Agreement on the diagonal, in blue."""
    cell_layout = [[("TP", True), ("FP", False)], [("FN", False), ("TN", True)]]
    total = sum(counts[name] for name in ("TP", "FP", "FN", "TN"))
    for row_index, row in enumerate(cell_layout):
        for column_index, (cell_name, is_agreement) in enumerate(row):
            share = counts[cell_name] / total
            base_color = "#2a78d6" if is_agreement else "#eb6834"
            cell_alpha = 0.12 + 0.88 * min(1.0, share * 1.6) if counts[cell_name] else 0.04
            axes.add_patch(plt.Rectangle((column_index, row_index), 1, 1, facecolor=base_color, alpha=cell_alpha,
                                         edgecolor=page_color, linewidth=3))
            axes.text(column_index + 0.5, row_index + 0.5, f"{counts[cell_name]:,}", ha="center", va="center",
                      fontsize=17, fontweight="bold", color=page_color if cell_alpha > 0.55 else ink_color)
    axes.set_xlim(0, 2)
    axes.set_ylim(2, 0)
    axes.set_xticks([0.5, 1.5], [text["pass"], text["fail"]])
    axes.set_yticks([0.5, 1.5], [text["pass"], text["fail"]])
    axes.xaxis.set_label_position("top")
    axes.xaxis.tick_top()
    axes.set_xlabel(reference_label, labelpad=8)
    axes.set_ylabel(reward_label)
    axes.tick_params(length=0)
    for spine in axes.spines.values():
        spine.set_visible(False)
    axes.set_aspect("equal")


def draw_reward_agreement(results: dict, language: str) -> None:
    text = label_text[language]
    t1_counts = results["reward_vs_eval"]["t1_reward_vs_eval_grader"]["base_medium"]
    t3_at_pass_mark_70 = next(entry for entry in results["reward_vs_eval"]["t3_gemma_vs_luna"]
                              if entry["threshold"] == 70.0)
    figure, (t1_axes, t3_axes) = plt.subplots(1, 2, figsize=(9.0, 4.2), gridspec_kw={"wspace": 0.55})
    draw_confusion_square(t1_axes, t1_counts, f"T1 {text['reward']}", text["grader"], text)
    draw_confusion_square(t3_axes, t3_at_pass_mark_70, text["gemma"], text["luna"], text)
    save_figure(figure, "reward_agreement", language)


# ---------------------------------------------------------------------------
# 9. two judges on the same reports
# ---------------------------------------------------------------------------
def draw_two_judges(results: dict, language: str) -> None:
    text = label_text[language]
    figure, (win_rate_axes, drift_axes) = plt.subplots(
        1, 2, figsize=(12.4, 4.1), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.22})
    bar_width = 0.32
    for judge_index, judge in enumerate(("primary", "second")):
        scores, halfwidths = [], []
        for model in ("sft", "rl_seed1"):
            judge_result = eval_set_result(results, model, "d_report")["by_judge"][judge]
            scores.append(judge_result["score"])
            halfwidths.append(judge_result["ci"]["halfwidth"])
        positions = [model_index + (judge_index - 0.5) * (bar_width + 0.03) for model_index in range(2)]
        win_rate_axes.bar(positions, scores, width=bar_width, color=judge_color[judge], zorder=2,
                          label=text["judge"][judge])
        win_rate_axes.errorbar(positions, scores, yerr=halfwidths, fmt="none", ecolor=ink_color, elinewidth=1.0,
                               capsize=2.5, zorder=3)
    win_rate_axes.axhline(50, color=ink_color, linewidth=1.0, linestyle="--", zorder=1)
    win_rate_axes.set_xticks([0, 1], [text["model"]["sft"], text["model"]["rl_seed1"]])
    win_rate_axes.tick_params(axis="x", length=0)
    win_rate_axes.set_ylim(0, 100)
    win_rate_axes.set_ylabel(text["win_rate"])
    add_horizontal_grid(win_rate_axes)

    drift_steps = results["judge_drift"]["steps"]
    dev_steps = [entry["step"] for entry in drift_steps]
    for judge, judge_key in (("primary", "gemma"), ("second", "luna")):
        points = [entry[judge_key]["point"] for entry in drift_steps]
        lower_errors = [entry[judge_key]["point"] - entry[judge_key]["lo"] for entry in drift_steps]
        upper_errors = [entry[judge_key]["hi"] - entry[judge_key]["point"] for entry in drift_steps]
        drift_axes.errorbar(dev_steps, points, yerr=[lower_errors, upper_errors], color=judge_color[judge],
                            linewidth=2.2, marker="o", markersize=6, capsize=3)
    drift_axes.set_xticks(dev_steps)
    drift_axes.set_xlabel(text["step"])
    drift_axes.set_ylabel(text["dev_score"])
    drift_axes.set_ylim(0, 100)
    drift_axes.set_xlim(-4, 44)
    add_horizontal_grid(drift_axes)
    win_rate_axes.legend(ncol=2, loc="lower left", bbox_to_anchor=(0.0, 1.0), handlelength=1.0, columnspacing=1.4)
    save_figure(figure, "two_judges", language)


# ---------------------------------------------------------------------------
# 10. what kind of failure: execution or concept (D-Num-hard), and what reports miss (D-Report)
# ---------------------------------------------------------------------------
def draw_failure_causes(results: dict, language: str) -> None:
    """Counts are GPT-6 Luna's labels on a stratified sample, weighted back to every failed answer
    (scripts/talk/failure_types.py; docs/reports/failure_types_and_tests_1110_en.md §1)."""
    text = label_text[language]
    hard_failures = results["failure_types"]["d_num_hard"]
    report_failures = results["failure_types"]["d_report"]
    figure, (hard_axes, report_axes) = plt.subplots(
        1, 2, figsize=(12.4, 3.9), gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.3})
    for model_index, model in enumerate(model_keys):
        weighted_unfinished = hard_failures[model]["weighted_unfinished"]
        weighted_wrong = hard_failures[model]["weighted_wrong"]
        cause_counts = {
            "execution": weighted_unfinished.get("execution", 0) + weighted_wrong.get("execution", 0),
            "concept": weighted_unfinished.get("concept", 0) + weighted_wrong.get("concept", 0),
            "other": hard_failures[model]["kinds"]["format"] + weighted_wrong.get("unclear", 0),
        }
        segment_left = 0.0
        for cause in ("execution", "concept", "other"):
            legend_label = text["cause"][cause] if model_index == 0 else None      # one legend entry per cause
            hard_axes.barh(model_index, cause_counts[cause], left=segment_left, height=0.62, color=cause_color[cause],
                           edgecolor=page_color, linewidth=1.2, label=legend_label)
            if cause_counts[cause] >= 40:
                hard_axes.text(segment_left + cause_counts[cause] / 2, model_index, f"{cause_counts[cause]:.0f}",
                               ha="center", va="center", fontsize=11.5,
                               color=ink_color if cause == "other" else page_color)
            segment_left += cause_counts[cause]
        shortfall_shares = report_failures[model]["shortfall_types_pct"]
        segment_left = 0.0
        for shortfall in ("concept", "coverage", "quantitative"):
            share = shortfall_shares.get(shortfall, 0.0)
            report_axes.barh(model_index, share, left=segment_left, height=0.62, color=shortfall_color[shortfall],
                             edgecolor=page_color, linewidth=1.2,
                             label=text["shortfall"][shortfall] if model_index == 0 else None)
            if share >= 8:
                report_axes.text(segment_left + share / 2, model_index, f"{share:.0f}", ha="center", va="center",
                                 fontsize=11.5, color=page_color if shortfall == "concept" else ink_color)
            segment_left += share
    for axes in (hard_axes, report_axes):
        axes.set_yticks(range(len(model_keys)), [text["model"][model] for model in model_keys])
        axes.invert_yaxis()
        axes.tick_params(axis="y", length=0)
        axes.spines["left"].set_visible(False)
        axes.legend(ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.0), handlelength=1.0, columnspacing=1.2,
                    fontsize=11.5)
    hard_axes.set_xlabel(f"D-Num-hard, {text['answers']}")
    report_axes.set_xlabel(f"D-Report, {text['share_of_shortfalls']}")
    report_axes.set_xlim(0, 100)
    save_figure(figure, "failure_causes", language)


# ---------------------------------------------------------------------------
# 11. refusals
# ---------------------------------------------------------------------------
def draw_refusals(results: dict, language: str) -> None:
    text = label_text[language]
    refusal_runs = results["refusal"]["runs"]
    figure, panel_axes = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"wspace": 0.25})
    for axes, prompt_group, y_limit in ((panel_axes[0], "should_refuse", 100), (panel_axes[1], "benign", 10)):
        for model_index, model in enumerate(model_keys):
            detector_rates = [refusal_runs[model][prompt_group][detector] for detector in ("strict", "broad")]
            positions = [detector_index + (model_index - 1) * 0.27 for detector_index in range(2)]
            axes.bar(positions, detector_rates, width=0.25, color=model_color[model], zorder=2,
                     label=text["model"][model])
        axes.set_xticks([0, 1], [text["strict"], text["broad"]])
        axes.tick_params(axis="x", length=0)
        axes.set_ylim(0, y_limit)
        axes.set_xlabel(text[prompt_group])
        add_horizontal_grid(axes)
    panel_axes[0].set_ylabel(text["refused"])
    panel_axes[0].legend(ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.0), handlelength=1.0, columnspacing=1.4)
    save_figure(figure, "refusals", language)


# ---------------------------------------------------------------------------
# 12. where the 8-node job's hours went
# ---------------------------------------------------------------------------
def draw_campaign_hours(results: dict, language: str) -> None:
    text = label_text[language]
    segments = results["campaign_hours"]["segments"]
    figure, axes = plt.subplots(figsize=(13.0, 1.9))
    segment_left = 0.0
    for segment in segments:
        segment_hours = (parse_log_time(segment["end"]) - parse_log_time(segment["start"])).total_seconds() / 3600
        segment_color = planned_time_color if segment["kind"] == "planned" else lost_time_color
        axes.barh(0, segment_hours, left=segment_left, height=0.55, color=segment_color, edgecolor=page_color,
                  linewidth=1.5)
        if segment_hours >= 2.0:                        # narrower segments have no room for their number
            axes.text(segment_left + segment_hours / 2, 0, f"{segment_hours:.1f}", ha="center", va="center",
                      fontsize=11.5, color=page_color)
        # names only for the planned stages and the one long block; the README table lists every segment
        if segment["name"] in text["segment"]:
            axes.text(segment_left + segment_hours / 2, 0.52, text["segment"][segment["name"]], ha="center",
                      va="center", fontsize=11, color=secondary_ink_color)
        segment_left += segment_hours
    axes.text(segment_left + 0.6, 0, f"{segment_left:.1f} {text['hours_unit']}", ha="left", va="center",
              fontsize=13, fontweight="bold", color=ink_color)
    axes.set_xlim(0, segment_left * 1.08)
    axes.set_ylim(0.85, -0.45)
    axes.axis("off")
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in (planned_time_color, lost_time_color)]
    axes.legend(legend_handles, [text["kind"]["planned"], text["kind"]["lost"]], ncol=2, loc="upper left",
                bbox_to_anchor=(0.0, 0.02), handlelength=1.0, columnspacing=1.4)
    save_figure(figure, "campaign_hours", language)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
figure_drawers = {
    "workflow": draw_workflow_strip,
    "results_by_stage": draw_results_by_stage,
    "test_time_compute": draw_test_time_compute,
    "where_samples_go": draw_where_samples_go,
    "d_num_hard_at_64k": draw_d_num_hard_at_64k,
    "bank_funnel": draw_bank_funnel,
    "last_stage_memory": draw_last_stage_memory,
    "rl_steps": draw_rl_steps,
    "reward_agreement": draw_reward_agreement,
    "two_judges": draw_two_judges,
    "failure_causes": draw_failure_causes,
    "refusals": draw_refusals,
    "campaign_hours": draw_campaign_hours,
}


def main() -> None:
    requested_figures = sys.argv[1:] or list(figure_drawers)
    unknown_figures = [name for name in requested_figures if name not in figure_drawers]
    assert not unknown_figures, f"unknown figure(s) {unknown_figures}; choose from {list(figure_drawers)}"
    results = load_all_result_files()
    for language in figure_languages:
        set_figure_style(language)
        for figure_name in requested_figures:
            figure_drawers[figure_name](results, language)


if __name__ == "__main__":
    main()
