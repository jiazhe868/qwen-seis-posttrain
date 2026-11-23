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
- docs/talk/data/numhard_64k.json   D-Num-hard with a 64K budget (scripts/talk/numhard_continue.py summary)
- docs/talk/data/campaign_hours.json and last_stage_memory.json: values measured by hand during the
  run, each with the note it came from (the run log)
- docs/talk/data/difficulty_pilot_1115.json, run1_step0_replay_1116.json, value_positions_1115.json: the SFT model's
  answers on bank v2's new families and on run 1's first step (scripts/talk/difficulty_pilot.py)
- docs/talk/data/rubric_formats_1116.json: the reward judge in three formats against gold labels
  (python -m dpt.eval.rubric_calibration --formats-out)
- docs/talk/data/rl_runs_compared.json: run 1 against RL v2, step by step (scripts/talk/rl_compare.py)
- docs/talk/data/stage_example.json: one new question answered 16 times by every model (scripts/talk/stage_example.py)
- docs/talk/data/rl_v2_seed1_local_1118.json: RL v2 seed 1 on D-Num-hard and D-Fresh, run on the workstation while its
  final evaluation waited for the cluster (scripts/talk/collect.py --eval-only); run.json's own results replace it
- docs/talk/data/large_models_1120.json: RL v2 against Qwen3.8-27B and Qwen3.5-122B-A10B on D-Num-hard at 16K and 32K
  (scripts/talk/large_models.py)

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
import random
import re
import sys
import textwrap
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
model_keys = ("base_medium", "sft", "rl_seed1")          # the three models every figure has
rl_v2_key = "rl_v2_seed1"                                 # RL v2 joins a figure once its results on that figure exist
model_color = {"base_medium": "#8a8984", "sft": "#2a78d6", "rl_seed1": "#eb6834", "rl_v2_seed1": "#6250d6"}
task_color = {"t1_num": "#2a78d6", "t2_code": "#eb6834", "t3_report": "#1baf7a", "t4_general": "#eda100"}
outcome_color = {"right": "#b7d3f6", "unfinished": "#e87ba4", "wrong_value": "#6250d6", "format": "#008300"}
cause_color = {"execution": "#1baf7a", "concept": "#6250d6", "other": "#c3c2b7"}
shortfall_color = {"concept": "#6250d6", "coverage": "#eda100", "quantitative": "#86b6ef"}
judge_color = {"primary": "#1baf7a", "second": "#6250d6"}
planned_time_color, lost_time_color = "#2a78d6", "#e87ba4"
length_only_color = "#c3c2b7"           # gradient from the length penalty alone, not from right against wrong
partial_level_color = "#9fdcc3"         # a three-level judge's "partial", a tint of the judge's green
run_color = {"run1": model_color["rl_seed1"], "rl_v2_seed1": model_color["rl_v2_seed1"]}   # run 1 keeps RL's orange
# the larger Qwen models in a cool grey ramp, darker for the larger one (the base model keeps the warm grey)
larger_model_color = {"rl_v2": model_color["rl_v2_seed1"], "sft": model_color["sft"],
                      "base_medium": model_color["base_medium"], "qwen38_27b_nvfp4": "#7d8c99",
                      "qwen35_122b_int4": "#3f4d5a"}

# ---------------------------------------------------------------------------
# the only words in the figures, in both languages
# ---------------------------------------------------------------------------
label_text = {
    "en": {
        "model": {"base_medium": "base", "sft": "SFT", "rl_seed1": "RL run 1", "rl_v2_seed1": "RL v2"},
        "workflow_model": {"base_medium": "Qwen3.5-9B", "sft": "SFT", "rl_seed1": "RL run 1, step 20",
                           "rl_v2_seed1": "RL v2, step 110"},
        "workflow_score": "D-Num-hard", "workflow_tokens": "tokens / answer",
        "conversations": "conversations", "problems": "problems",
        "score": "score", "tokens_thousands": "tokens (K)",
        "larger_model": {"rl_v2": "RL v2, 9B (ours)", "sft": "SFT, 9B (ours)", "base_medium": "Qwen3.5-9B (base)",
                         "qwen38_27b_nvfp4": "Qwen3.8-27B (teacher)", "qwen35_122b_int4": "Qwen3.5-122B-A10B"},
        "larger_budgets": ["16K cap", "32K cap"], "larger_score": "D-Num-hard, % right",
        "larger_tokens": "tokens per answer (K)",
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
        "rule": {"left out + penalty": "left out + penalty", "left out": "left out", "counts wrong": "counts wrong"},
        "groups_with_gradient": "groups with gradient (%)", "run1": "run 1",
        "right_and_wrong": "right and wrong answers", "length_only": "length penalty only",
        "value_first_written": "first number matching the gold (K tokens)", "answer_length": "answer length (K tokens)",
        "ran_out": "ran out", "right": "right", "wrong": "wrong",
        "right_at_24k": "right at 24K (%)", "at_24k": "24K", "at_16k": "16K",
        "judge_format": {"0-10 per criterion": "0–10 per criterion", "yes/no checks": "yes/no checks with a quote",
                         "three levels": "three levels"},
        "gold_mark": {"yes": "met", "partly": "partly met", "no": "not met"},
        "judged_met_share": "judged met (% of criteria)", "judged_met": "met", "judged_partial": "partial",
        "run": {"run1": "run 1", "rl_v2_seed1": "RL v2"}, "lr_sum": "learning-rate sum (×10⁻⁶)",
        "curve_run": {"run1": "run 1", "rl_v2_seed1": "RL v2 seed 1", "rl_v2_seed2": "RL v2 seed 2"},
        "curve_panel": {"t1_num_reward": "numeric reward", "t2_code_reward": "code reward",
                        "t3_report_reward": "report reward", "t4_general_reward": "general reward",
                        "t1_num_entropy": "numeric entropy", "t2_code_entropy": "code entropy",
                        "drift_kl": "KL, stale samples", "mismatch_kl": "KL, fresh samples"},
        "entropy_unit": "nats / token", "kl_unit": "KL (k3)", "reward_unit": "mean reward",
        "numeric_grad_share": "gradient from numeric (%)",
        "numeric_groups_with_gradient": "numeric groups with gradient (%)",
        "right_answer_length": "right numeric answers (tokens)",
        "stage_title": {"base": "Base  ·  Qwen3.5-9B", "sft": "SFT", "rl_seed1": "RL run 1  ·  step 20",
                        "rl_v2_seed1": "RL v2  ·  seed 1, step 110"},
        "question": "Question", "expected": "expected", "tokens_of_reasoning": "… {n:,} tokens of reasoning in all …",
        "cut_off": "cut off at {n:,} tokens: no answer", "verdict_right": "right", "verdict_cut": "ran out of tokens",
        "verdict_wrong": "wrong value", "of_n_right": "{k} of {n} right", "of_n_cut": "{u} of {n} ran out",
        "median_tokens": "median {n:.1f}K tokens", "pending": "not answered yet",
    },
    "zh": {
        "model": {"base_medium": "基座", "sft": "SFT", "rl_seed1": "第 1 次 RL", "rl_v2_seed1": "RL v2"},
        "workflow_model": {"base_medium": "Qwen3.5-9B", "sft": "SFT", "rl_seed1": "第 1 次 RL，第 20 步",
                           "rl_v2_seed1": "RL v2，第 110 步"},
        "workflow_score": "D-Num-hard", "workflow_tokens": "每个回答的 token",
        "conversations": "条对话", "problems": "道题",
        "score": "得分", "tokens_thousands": "token 数（千）",
        "larger_model": {"rl_v2": "RL v2，9B（我们的）", "sft": "SFT，9B（我们的）", "base_medium": "Qwen3.5-9B（基座）",
                         "qwen38_27b_nvfp4": "Qwen3.8-27B（教师）", "qwen35_122b_int4": "Qwen3.5-122B-A10B"},
        "larger_budgets": ["16K 上限", "32K 上限"], "larger_score": "D-Num-hard 答对 %",
        "larger_tokens": "每个回答的 token 数 (K)",
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
        "rule": {"left out + penalty": "移出损失 + 惩罚", "left out": "移出损失", "counts wrong": "记为错误"},
        "groups_with_gradient": "带有梯度的组（%）", "run1": "第 1 次\n运行",
        "right_and_wrong": "有对有错的回答", "length_only": "只来自长度惩罚",
        "value_first_written": "首个与标准答案相符的数（千 token）", "answer_length": "回答长度（千 token）",
        "ran_out": "用光 token", "right": "正确", "wrong": "错误",
        "right_at_24k": "24K 下正确（%）", "at_24k": "24K", "at_16k": "16K",
        "judge_format": {"0-10 per criterion": "每条标准 0–10 分", "yes/no checks": "带引文的是/否检查项",
                         "three levels": "三个等级"},
        "gold_mark": {"yes": "满足", "partly": "部分满足", "no": "不满足"},
        "judged_met_share": "判为满足（占标准的 %）", "judged_met": "满足", "judged_partial": "部分满足",
        "run": {"run1": "第 1 次运行", "rl_v2_seed1": "RL v2"}, "lr_sum": "学习率之和（×10⁻⁶）",
        "curve_run": {"run1": "第 1 次运行", "rl_v2_seed1": "RL v2 种子 1", "rl_v2_seed2": "RL v2 种子 2"},
        "curve_panel": {"t1_num_reward": "数值题奖励", "t2_code_reward": "代码奖励", "t3_report_reward": "报告奖励",
                        "t4_general_reward": "通用奖励", "t1_num_entropy": "数值题熵", "t2_code_entropy": "代码熵",
                        "drift_kl": "KL，旧样本", "mismatch_kl": "KL，新样本"},
        "entropy_unit": "nats / token", "kl_unit": "KL（k3）", "reward_unit": "平均奖励",
        "numeric_grad_share": "来自数值题的梯度（%）", "numeric_groups_with_gradient": "带有梯度的数值题组（%）",
        "right_answer_length": "数值题正确回答（token）",
        "stage_title": {"base": "基座  ·  Qwen3.5-9B", "sft": "SFT", "rl_seed1": "第 1 次 RL  ·  第 20 步",
                        "rl_v2_seed1": "RL v2  ·  种子 1，第 110 步"},
        "question": "问题", "expected": "标准答案", "tokens_of_reasoning": "…… 推理共 {n:,} 个 token ……",
        "cut_off": "在 {n:,} 个 token 处截断：没有答案", "verdict_right": "正确", "verdict_cut": "用光 token",
        "verdict_wrong": "数值错误", "of_n_right": "{n} 次中 {k} 次正确", "of_n_cut": "{n} 次中 {u} 次用光 token",
        "median_tokens": "中位数 {n:.1f}K 个 token", "pending": "尚未作答",
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
    cluster = load_result_file("run.json")
    # RL v2 seed 1's two headline sets ran on the workstation while its final evaluation waited in the cluster's queue;
    # they fill in only what run.json does not have yet, so the cluster's own runs replace them when collected
    workstation = load_result_file("rl_v2_seed1_local_1118.json")
    for run_name, run in workstation["eval"]["runs"].items():
        cluster["eval"]["runs"].setdefault(run_name, run)
    for entry_name, entry in workstation["budget"].items():
        cluster["budget"].setdefault(entry_name, entry)
    return {
        "run": cluster,
        "rl": load_result_file("rl.json"),
        "reward_vs_eval": load_result_file("reward_vs_eval.json"),
        "judge_drift": load_result_file("judge_drift.json"),
        "refusal": load_result_file("refusal.json"),
        "failure_types": load_result_file("failure_types.json"),
        "numhard_64k": load_result_file("numhard_64k.json"),
        "research_1110": load_result_file("research_1110.json"),
        "campaign_hours": load_result_file("campaign_hours.json"),
        "last_stage_memory": load_result_file("last_stage_memory.json"),
        "difficulty_pilot": load_result_file("difficulty_pilot_1115.json"),
        "run1_step0_replay": load_result_file("run1_step0_replay_1116.json"),
        "value_positions": load_result_file("value_positions_1115.json"),
        "rubric_formats": load_result_file("rubric_formats_1116.json"),
        "rl_runs_compared": load_result_file("rl_runs_compared.json"),
        "stage_example": load_result_file("stage_example.json"),
    }


def eval_set_result(results: dict, model_key: str, set_name: str) -> dict:
    """D-Num-hard was scored in its own runs (d_num_hard/<model>); every other set in the model's main run."""
    run_key = f"d_num_hard/{model_key}" if set_name == "d_num_hard" else model_key
    return results["run"]["eval"]["runs"][run_key]["sets"][set_name]


def models_scored_on(results: dict, set_names: list[str]) -> list[str]:
    """The three models, and RL v2 once it has a result on every one of these sets."""
    runs = results["run"]["eval"]["runs"]

    def has_set(model: str, set_name: str) -> bool:
        run = runs.get(f"d_num_hard/{model}" if set_name == "d_num_hard" else model) or {}
        return set_name in (run.get("sets") or {})

    return [model for model in model_keys + (rl_v2_key,) if all(has_set(model, name) for name in set_names)]


def thousands_label(value: float) -> str:
    return f"{value / 1000:.1f}K"


# ---------------------------------------------------------------------------
# 1. the pipeline strip at the top of the README
# ---------------------------------------------------------------------------
def draw_workflow_strip(results: dict, language: str) -> None:
    """The models in boxes; each arrow carries the data that trained the next model. Both RL runs start from SFT, so
    once RL v2 has a D-Num-hard score the strip branches after SFT: run 1 above, RL v2 below.

    The big number is D-Num-hard (avg@8), the held-out slice with headroom; the small number is the
    mean output length on it.
    """
    text = label_text[language]
    strip_models = models_scored_on(results, ["d_num_hard"])
    branched = rl_v2_key in strip_models
    sft_conversation_count = sum(results["run"]["sft_mix"]["conversations_by_subset"].values())
    rl_problem_count = sum(task["after_difficulty"] for task in results["run"]["pipeline"]["by_task"].values())
    # RL v2 trains on numeric, code and general problems; its bank's report problems are never drawn (README §8.3)
    rl_v2_problem_count = sum(count for pool in results["run"]["bank_v2"]["trained"].values()
                              for task, count in pool.items() if task != "t3_report")

    figure_height = 5.0 if branched else 2.7
    figure = plt.figure(figsize=(12.4, figure_height))
    axes = figure.add_axes([0, 0, 1, 1])
    axes.set_xlim(0, 12.4)
    axes.set_ylim(0, figure_height)
    axes.axis("off")
    box_width, box_height = 2.45, 2.2
    middle_bottom = (figure_height - box_height) / 2
    box_corner = {"base_medium": (2.1, middle_bottom), "sft": (5.95, middle_bottom),
                  "rl_seed1": (9.8, figure_height - box_height - 0.12 if branched else middle_bottom),
                  rl_v2_key: (9.8, 0.12)}

    axes.text(1.95, middle_bottom + 1.05, text["workflow_score"], ha="right", va="center", fontsize=12,
              color=muted_ink_color)
    axes.text(1.95, middle_bottom + 0.42, text["workflow_tokens"], ha="right", va="center", fontsize=12,
              color=muted_ink_color)
    for model in strip_models:
        box_left, box_bottom = box_corner[model]
        hard_result = eval_set_result(results, model, "d_num_hard")
        axes.add_patch(FancyBboxPatch((box_left, box_bottom), box_width, box_height,
                                      boxstyle="round,pad=0.02,rounding_size=0.12", linewidth=1.6,
                                      edgecolor=model_color[model], facecolor=model_color[model] + "14"))
        box_center = box_left + box_width / 2
        axes.text(box_center, box_bottom + 1.78, text["workflow_model"][model], ha="center", va="center",
                  fontsize=15 if len(text["workflow_model"][model]) < 16 else 13.5, fontweight="bold",
                  color=ink_color)
        axes.text(box_center, box_bottom + 1.05, f"{hard_result['score']:.1f}", ha="center", va="center",
                  fontsize=26, fontweight="bold", color=model_color[model])
        axes.text(box_center, box_bottom + 0.42, thousands_label(hard_result["mean_output_tokens"]), ha="center",
                  va="center", fontsize=13, color=secondary_ink_color)

    def arrow(start_model: str, end_model: str, count: int, unit_text: str, label_side: str,
              start_shift: float = 0.0) -> None:
        start_left, start_bottom = box_corner[start_model]
        end_left, end_bottom = box_corner[end_model]
        start = (start_left + box_width + 0.08, start_bottom + box_height / 2 - 0.25 + start_shift)
        end = (end_left - 0.08, end_bottom + box_height / 2 - 0.25)
        axes.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, linewidth=1.6,
                                       color=secondary_ink_color))
        label_x = (start[0] + end[0]) / 2
        # a straight arrow carries its label above; a branch labels its far side, away from the other branch
        label_y = {"above": max(start[1], end[1]) + 0.63, "below": min(start[1], end[1]) - 0.33}[label_side]
        axes.text(label_x, label_y, f"{count:,}", ha="center", va="center", fontsize=13, fontweight="bold",
                  color=secondary_ink_color)
        axes.text(label_x, label_y - 0.33, unit_text, ha="center", va="center", fontsize=11.5, color=muted_ink_color)

    arrow("base_medium", "sft", sft_conversation_count, text["conversations"], "above")
    arrow("sft", "rl_seed1", rl_problem_count, text["problems"], "above", start_shift=0.35 if branched else 0.0)
    if branched:
        arrow("sft", rl_v2_key, rl_v2_problem_count, text["problems"], "below", start_shift=-0.1)
    save_figure(figure, "workflow", language)


# ---------------------------------------------------------------------------
# 2. results by stage: domain sets, general benchmarks, answer length
# ---------------------------------------------------------------------------
def draw_grouped_bars(axes, values_by_model: dict, halfwidths_by_model: dict, group_count: int) -> None:
    bar_width = 0.75 / len(values_by_model)
    middle = (len(values_by_model) - 1) / 2
    for model_index, model in enumerate(values_by_model):
        positions = [group + (model_index - middle) * (bar_width + 0.02) for group in range(group_count)]
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
    stage_models = [model for model in models_scored_on(results, domain_set_names)
                    if set(general_set_names) <= set(runs[model].get("general") or {})]
    for model in stage_models:
        domain_results = [eval_set_result(results, model, set_name) for set_name in domain_set_names]
        domain_scores[model] = [result["score"] for result in domain_results]
        domain_halfwidths[model] = [result["ci"]["halfwidth"] for result in domain_results]
        general_results = [runs[model]["general"][set_name] for set_name in general_set_names]
        general_scores[model] = [result["score"] for result in general_results]
        general_halfwidths[model] = [result["ci"]["halfwidth"] for result in general_results]
    mean_output_tokens = [runs[model]["mean_output_tokens"] for model in stage_models]

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

    for model_index, model in enumerate(stage_models):
        token_axes.bar(model_index, mean_output_tokens[model_index] / 1000, width=0.62, color=model_color[model],
                       zorder=2)
        token_axes.text(model_index, mean_output_tokens[model_index] / 1000 + 0.3,
                        f"{mean_output_tokens[model_index] / 1000:.1f}", ha="center", va="bottom", fontsize=12,
                        color=secondary_ink_color)
    token_axes.set_xticks([])
    token_axes.set_ylabel(text["tokens_thousands"])
    token_axes.set_ylim(0, max(mean_output_tokens) / 1000 * 1.18)
    add_horizontal_grid(token_axes)

    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=model_color[model]) for model in stage_models]
    domain_axes.legend(legend_handles, [text["model"][model] for model in stage_models], ncol=len(stage_models),
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
        "rl_v2_seed1": ["rl_v2_seed1"],
    }
    figure, (d_num_axes, hard_axes, vote_axes) = plt.subplots(
        1, 3, figsize=(14.5, 4.1), gridspec_kw={"width_ratios": [1, 1, 1], "wspace": 0.22})
    # RL cut answers at 16K tokens (configs/model/qwen35_9b.yaml thinking.rl_max_response; README §6.1);
    # the dotted line shows how much of each model's accuracy fits under that cap
    rl_response_cap_thousands = read_rl_response_cap_tokens() / 1024
    for axes, set_name, set_tag in ((d_num_axes, "d_num", "D-Num"), (hard_axes, "d_num_hard", "D-Num-hard")):
        for model in model_keys + (rl_v2_key,):
            entry_keys = d_num_budget_entries.get(model) if set_name == "d_num" else [f"d_num_hard__{model}"]
            if not entry_keys or any(key not in results["run"]["budget"] for key in entry_keys):
                continue
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
    hard_axes.legend(loc="lower right", handlelength=1.4)

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
    # a model joins once every one of its cut-off answers has been continued
    figure_models = [model for model in model_keys + (rl_v2_key,)
                     if model in budget_rows and budget_rows[model]["missing_continuations"] == 0]
    outcome_order = ("right", "unfinished", "wrong_value", "format")
    figure, axes = plt.subplots(figsize=(12.4, 1.47 * len(figure_models)))
    row_positions, row_labels = [], []
    for model_index, model in enumerate(figure_models):
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
    groups = [(set_name, set_tag, models_scored_on(results, [set_name]))
              for set_name, set_tag in (("d_num", "D-Num"), ("d_num_hard", "D-Num-hard"))]
    figure, axes = plt.subplots(figsize=(12.4, 4.0 + 0.45 * (sum(len(models) for *_, models in groups) - 6)))
    row_positions, row_labels = [], []
    group_top = 0.0
    for set_name, set_tag, group_models in groups:
        axes.text(-1.0, group_top - 0.75, set_tag, ha="left", va="center", fontsize=13, fontweight="bold",
                  color=ink_color, clip_on=False)
        for model_index, model in enumerate(group_models):
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
        group_top += len(group_models) + 1.0
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
    judged_models = [model for model in ("sft", "rl_seed1", rl_v2_key)
                     if "by_judge" in (results["run"]["eval"]["runs"].get(model, {}).get("sets", {}).get("d_report")
                                       or {})]
    for judge_index, judge in enumerate(("primary", "second")):
        scores, halfwidths = [], []
        for model in judged_models:
            judge_result = eval_set_result(results, model, "d_report")["by_judge"][judge]
            scores.append(judge_result["score"])
            halfwidths.append(judge_result["ci"]["halfwidth"])
        positions = [model_index + (judge_index - 0.5) * (bar_width + 0.03)
                     for model_index in range(len(judged_models))]
        win_rate_axes.bar(positions, scores, width=bar_width, color=judge_color[judge], zorder=2,
                          label=text["judge"][judge])
        win_rate_axes.errorbar(positions, scores, yerr=halfwidths, fmt="none", ecolor=ink_color, elinewidth=1.0,
                               capsize=2.5, zorder=3)
    win_rate_axes.axhline(50, color=ink_color, linewidth=1.0, linestyle="--", zorder=1)
    win_rate_axes.set_xticks(range(len(judged_models)), [text["model"][model] for model in judged_models])
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
    figure_models = [model for model in model_keys + (rl_v2_key,) if model in refusal_runs]
    bar_width = 0.8 / len(figure_models)
    figure, panel_axes = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"wspace": 0.25})
    for axes, prompt_group, y_limit in ((panel_axes[0], "should_refuse", 100), (panel_axes[1], "benign", 10)):
        for model_index, model in enumerate(figure_models):
            detector_rates = [refusal_runs[model][prompt_group][detector] for detector in ("strict", "broad")]
            offset = (model_index - (len(figure_models) - 1) / 2) * (bar_width + 0.02)
            positions = [detector_index + offset for detector_index in range(2)]
            axes.bar(positions, detector_rates, width=bar_width, color=model_color[model], zorder=2,
                     label=text["model"][model])
        axes.set_xticks([0, 1], [text["strict"], text["broad"]])
        axes.tick_params(axis="x", length=0)
        axes.set_ylim(0, y_limit)
        axes.set_xlabel(text[prompt_group])
        add_horizontal_grid(axes)
    panel_axes[0].set_ylabel(text["refused"])
    panel_axes[0].legend(ncol=len(figure_models), loc="lower left", bbox_to_anchor=(0.0, 1.0), handlelength=1.0,
                         columnspacing=1.2)
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
# ---------------------------------------------------------------------------
# 13. the judge in three formats against gold labels, per gold mark
# ---------------------------------------------------------------------------
judge_formats = ("0-10 per criterion", "yes/no checks", "three levels")
gold_marks = ("yes", "partly", "no")


def draw_judge_formats(results: dict, language: str) -> None:
    """Per format, the share of the gold's met, partly met and not met criteria that the judge called met; for three
    levels, also the share it called partial. Test split of the gold set (README §6.5)."""
    text = label_text[language]
    format_counts = results["rubric_formats"]["formats"]
    figure, panel_axes = plt.subplots(1, 3, figsize=(12.4, 3.9), sharey=True, gridspec_kw={"wspace": 0.08})
    for axes, format_name in zip(panel_axes, judge_formats):
        counts = format_counts[format_name]
        for position, mark in enumerate(gold_marks):
            met_share = 100 * counts["judged_met"][mark] / counts["criteria"][mark]
            axes.bar(position, met_share, width=0.62, color=judge_color["primary"], zorder=2)
            if counts["levels"]:
                partial_share = 100 * counts["levels"][mark]["partial"] / counts["criteria"][mark]
                axes.bar(position, partial_share, bottom=met_share, width=0.62, color=partial_level_color,
                         edgecolor=page_color, linewidth=1.2, zorder=2)
            axes.text(position, met_share - 2.5, f"{met_share:.0f}", ha="center", va="top", fontsize=11.5,
                      color=page_color, zorder=3)
        axes.set_xticks(range(len(gold_marks)), [text["gold_mark"][mark] for mark in gold_marks])
        axes.tick_params(axis="x", length=0)
        axes.set_xlabel(text["judge_format"][format_name])
        axes.set_ylim(0, 100)
        add_horizontal_grid(axes)
    panel_axes[0].set_ylabel(text["judged_met_share"])
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color)
                      for color in (judge_color["primary"], partial_level_color)]
    panel_axes[2].legend(legend_handles, [text["judged_met"], text["judged_partial"]], ncol=2, loc="lower right",
                         bbox_to_anchor=(1.0, 1.0), handlelength=1.0, columnspacing=1.4)
    save_figure(figure, "judge_formats", language)


# ---------------------------------------------------------------------------
# 14. RL v2's first step replayed: which groups carry gradient under each rule for a cut-off answer
# ---------------------------------------------------------------------------
truncation_rules = ("left out + penalty", "left out", "counts wrong")


def draw_gradient_by_rule(results: dict, language: str) -> None:
    """The first GRPO step computed with the trainer's own code (scripts/talk/difficulty_pilot.py): the share of groups
    whose answers got different rewards, split into groups with right and wrong answers and groups whose answers
    differ only by the soft length penalty. One panel per rule; in each, the pilot on bank v2's new families at three
    budgets, and run 1's first step at its 16K on run 1's own problems (README §8.3)."""
    text = label_text[language]
    pilot_budgets = results["difficulty_pilot"]["families"]["all"]["by_budget"]
    run1_budgets = results["run1_step0_replay"]["families"]["all"]["by_budget"]
    bar_sources = [(pilot_budgets[str(budget)], f"{budget // 1024}K") for budget in (16384, 20480, 24576)]
    bar_sources.append((run1_budgets["16384"], text["run1"]))
    bar_positions = (0.0, 1.0, 2.0, 3.45)          # run 1 set apart: other problems
    figure, panel_axes = plt.subplots(1, 3, figsize=(13.0, 4.1), sharey=True, gridspec_kw={"wspace": 0.08})
    for axes, rule in zip(panel_axes, truncation_rules):
        for position, (budget_result, _) in zip(bar_positions, bar_sources):
            first_step = budget_result[rule]
            right_and_wrong, length_only = 100 * first_step["right_and_wrong"], 100 * first_step["length_only"]
            axes.bar(position, right_and_wrong, width=0.72, color=task_color["t1_num"], zorder=2)
            axes.bar(position, length_only, bottom=right_and_wrong, width=0.72, color=length_only_color,
                     edgecolor=page_color, linewidth=1.2, zorder=2)
            axes.text(position, right_and_wrong + length_only + 1.2, f"{100 * first_step['with_gradient']:.0f}",
                      ha="center", va="bottom", fontsize=11.5, color=secondary_ink_color)
        axes.axvline(2.72, color=grid_color, linewidth=1.0, zorder=1)
        axes.set_xticks(bar_positions, [label for _, label in bar_sources])
        axes.tick_params(axis="x", length=0)
        axes.set_xlabel(text["rule"][rule])
        axes.set_ylim(0, 64)
        add_horizontal_grid(axes)
    panel_axes[0].set_ylabel(text["groups_with_gradient"])
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in (task_color["t1_num"], length_only_color)]
    panel_axes[0].legend(legend_handles, [text["right_and_wrong"], text["length_only"]], ncol=2, loc="lower left",
                         bbox_to_anchor=(0.0, 1.0), handlelength=1.0, columnspacing=1.4)
    save_figure(figure, "gradient_by_rule", language)


# ---------------------------------------------------------------------------
# 15. where in an answer the value first appears, against the answer's length
# ---------------------------------------------------------------------------
def draw_value_positions(results: dict, language: str) -> None:
    """Every pilot answer that contains a number matching the gold: the token where it first appears against the
    answer's length. Answers that ran out of tokens have no length of their own, so they sit in a strip on top, spread
    at random in height only. Any matching number counts, so a position can be early by chance (README §8.3)."""
    text = label_text[language]
    positions = results["value_positions"]
    short_budget_thousands = positions["short_budget"] / 1000
    answers = positions["answers"]
    axis_end = 24.8
    figure, (strip_axes, scatter_axes) = plt.subplots(
        2, 1, figsize=(8.4, 6.6), sharex=True, gridspec_kw={"height_ratios": [1.0, 3.4], "hspace": 0.05})
    strip_spread = random.Random(20261116)
    unfinished_positions = [value_at / 1000 for value_at, _, outcome in answers if outcome == "unfinished"]
    strip_axes.scatter(unfinished_positions, [strip_spread.uniform(0.0, 1.0) for _ in unfinished_positions], s=11,
                       color=outcome_color["unfinished"], alpha=0.6, linewidths=0, zorder=2)
    strip_axes.set_ylim(-0.3, 1.3)
    strip_axes.set_yticks([0.5], [text["ran_out"]])
    strip_axes.tick_params(axis="y", length=0)
    strip_axes.tick_params(axis="x", length=0)
    strip_axes.spines["left"].set_visible(False)
    strip_axes.spines["bottom"].set_visible(False)
    marker_style = {"right": {"facecolor": outcome_color["right"], "edgecolor": task_color["t1_num"]},
                    "wrong": {"facecolor": outcome_color["wrong_value"], "edgecolor": outcome_color["wrong_value"]}}
    for outcome in ("right", "wrong"):
        finished = [(value_at / 1000, tokens / 1000) for value_at, tokens, answer_outcome in answers
                    if answer_outcome == outcome]
        scatter_axes.scatter([x for x, _ in finished], [y for _, y in finished], s=16, linewidths=0.6, alpha=0.8,
                             zorder=3, label=text[outcome], **marker_style[outcome])
    scatter_axes.plot([0, axis_end], [0, axis_end], color=axis_color, linewidth=1.0, zorder=1)  # value before end
    for axes in (strip_axes, scatter_axes):
        axes.axvline(short_budget_thousands, color=muted_ink_color, linewidth=1.0, linestyle="--", zorder=1)
    scatter_axes.axhline(short_budget_thousands, color=muted_ink_color, linewidth=1.0, linestyle="--", zorder=1)
    scatter_axes.set_xlim(0, axis_end)
    scatter_axes.set_ylim(0, axis_end)
    scatter_axes.set_xticks([0, 4, 8, 12, 16, 20, 24])
    scatter_axes.set_yticks([0, 4, 8, 12, 16, 20, 24])
    scatter_axes.set_xlabel(text["value_first_written"])
    scatter_axes.set_ylabel(text["answer_length"])
    add_horizontal_grid(scatter_axes)
    scatter_axes.legend(loc="lower right", handletextpad=0.3, markerscale=1.4)
    save_figure(figure, "value_positions", language)


# ---------------------------------------------------------------------------
# 16. the pilot's families: groups with gradient at 24K and at 16K
# ---------------------------------------------------------------------------
def draw_pilot_families(results: dict, language: str) -> None:
    """Per family, with a cut-off answer counted as wrong: the share of problems whose 8 answers were not all the
    same at 24K (open) and at 16K (filled), next to the share of answers right at 24K. Sorted by that share, so the
    families the model solves quickly sit on top (README §8.3)."""
    text = label_text[language]
    families = {name: family for name, family in results["difficulty_pilot"]["families"].items()
                if not name.startswith("all")}
    family_order = sorted(families, key=lambda name: (-families[name]["pass_rate"], name))
    figure, (rate_axes, gradient_axes) = plt.subplots(
        1, 2, figsize=(12.4, 6.4), sharey=True, gridspec_kw={"width_ratios": [1.0, 2.6], "wspace": 0.06})
    for row, name in enumerate(family_order):
        family = families[name]
        rate_axes.barh(row, 100 * family["pass_rate"], height=0.62, color=outcome_color["right"], zorder=2)
        at_24k = 100 * family["by_budget"]["24576"]["counts wrong"]["with_gradient"]
        at_16k = 100 * family["by_budget"]["16384"]["counts wrong"]["with_gradient"]
        if abs(at_16k - at_24k) >= 1:
            gradient_axes.annotate("", xy=(at_16k, row), xytext=(at_24k, row), zorder=2,
                                   arrowprops={"arrowstyle": "-|>", "color": muted_ink_color, "linewidth": 1.2,
                                               "shrinkA": 5, "shrinkB": 5, "mutation_scale": 11})
        gradient_axes.scatter([at_24k], [row], s=46, facecolor=page_color, edgecolor=task_color["t1_num"],
                              linewidths=1.6, zorder=3, label=text["at_24k"] if row == 0 else None)
        gradient_axes.scatter([at_16k], [row], s=46, color=task_color["t1_num"], zorder=3,
                              label=text["at_16k"] if row == 0 else None)
    rate_axes.set_yticks(range(len(family_order)), family_order, fontsize=11)
    rate_axes.invert_yaxis()
    rate_axes.tick_params(axis="y", length=0)
    rate_axes.spines["left"].set_visible(False)
    rate_axes.set_xlim(0, 100)
    rate_axes.set_xticks([0, 50, 100])
    rate_axes.set_xlabel(text["right_at_24k"])
    gradient_axes.set_xlim(-4, 104)
    gradient_axes.set_xticks([0, 25, 50, 75, 100])
    gradient_axes.set_xlabel(text["groups_with_gradient"])
    gradient_axes.tick_params(axis="y", length=0)
    gradient_axes.spines["left"].set_visible(False)
    for axes in (rate_axes, gradient_axes):
        axes.grid(True, axis="x")
        axes.set_axisbelow(True)
    gradient_axes.legend(ncol=2, loc="lower left", bbox_to_anchor=(0.0, 1.0), handletextpad=0.3, columnspacing=1.4)
    save_figure(figure, "pilot_families", language)


# ---------------------------------------------------------------------------
# 17. run 1 against RL v2, at equal learning-rate sums
# ---------------------------------------------------------------------------
def rolling_mean(values: list, width: int = 5) -> list:
    """Centered mean over ``width`` steps, skipping missing values (the per-step series are noisy)."""
    half, out = width // 2, []
    for i in range(len(values)):
        window = [v for v in values[max(0, i - half): i + half + 1] if isinstance(v, (int, float))]
        out.append(sum(window) / len(window) if window else None)
    return out


def draw_rl_runs_compared(results: dict, language: str) -> None:
    """Three series for both runs against the sum of the learning rates so far (with Adam, how far a run has pushed
    the policy): the share of the gradient from numeric problems, the share of numeric groups whose answers differ,
    and the length of right numeric answers. 5-step centered means; the dotted line is run 1's collapse (step 40)."""
    text = label_text[language]
    compared = results["rl_runs_compared"]
    collapse_lr_sum = compared["run1_collapse"]["lr_sum"]
    panels = (("t1_num_grad_share", text["numeric_grad_share"], 100, "linear"),
              ("t1_groups_with_gradient", text["numeric_groups_with_gradient"], 100, "linear"),
              ("t1_len_right", text["right_answer_length"], 1, "log"))
    figure, panel_axes = plt.subplots(1, 3, figsize=(14.5, 4.0), gridspec_kw={"wspace": 0.28})
    for axes, (key, axis_label, scale, y_scale) in zip(panel_axes, panels):
        ends = {}
        for run in ("run1", "rl_v2_seed1", "rl_v2_seed2"):
            if run not in compared["per_step"]:
                continue
            rows = compared["per_step"][run]
            values = rolling_mean([r.get(key) for r in rows])
            points = [(r["lr_sum"], scale * v) for r, v in zip(rows, values) if v is not None and v > 0]
            seed2 = run == "rl_v2_seed2"                  # dashed, unlabelled: it repeats seed 1
            axes.plot([x for x, _ in points], [y for _, y in points], color=run_color["rl_v2_seed1" if seed2 else run],
                      linewidth=1.6 if seed2 else 2.0, linestyle="--" if seed2 else "-", zorder=3)
            if not seed2:
                ends[run] = points[-1]
        # direct labels at the line ends, pushed apart when the two ends nearly meet
        label_y = {run: y for run, (_, y) in ends.items()}
        if y_scale == "linear" and abs(label_y["run1"] - label_y["rl_v2_seed1"]) < 7:
            upper = max(label_y, key=label_y.get)
            middle = (label_y["run1"] + label_y["rl_v2_seed1"]) / 2
            label_y = {run: middle + (3.5 if run == upper else -3.5) for run in label_y}
        for run, (x, _) in ends.items():
            axes.text(x + 1.5, label_y[run], text["run"][run], color=run_color[run], fontsize=11.5, va="center",
                      ha="left")
        axes.axvline(collapse_lr_sum, color=muted_ink_color, linewidth=1.0, linestyle=":", zorder=1)
        axes.set_yscale(y_scale)
        axes.set_xlabel(text["lr_sum"])
        axes.set_ylabel(axis_label)
        axes.set_xlim(0, 125)
        add_horizontal_grid(axes)
    panel_axes[0].set_ylim(0, 103)
    panel_axes[1].set_ylim(0, 80)
    save_figure(figure, "rl_runs_compared", language)


def draw_rl_training_curves(results: dict, language: str) -> None:
    """Reward per task, entropy per token and the train/rollout KL for run 1 and RL v2's seeds, against the sum of the
    learning rates so far (as rl_runs_compared). 5-step centered means; the dotted line is run 1's collapse. There is
    no reference model, so no KL to SFT: "stale samples" are the ones drawn up to 3 steps before (policy change plus
    engine), "fresh samples" the ones drawn by the current weights (engine only)."""
    text = label_text[language]
    compared = results["rl_runs_compared"]
    runs = [run for run in ("run1", "rl_v2_seed1", "rl_v2_seed2") if run in compared["per_step"]]
    style = {"run1": (run_color["run1"], "-"), "rl_v2_seed1": (run_color["rl_v2_seed1"], "-"),
             "rl_v2_seed2": (run_color["rl_v2_seed1"], "--")}
    panels = ["t1_num_reward", "t2_code_reward", "t3_report_reward", "t4_general_reward",
              "t1_num_entropy", "t2_code_entropy", "drift_kl", "mismatch_kl"]
    figure, axes_grid = plt.subplots(2, 4, figsize=(16.5, 7.2), gridspec_kw={"wspace": 0.28, "hspace": 0.18},
                                     sharex=True)
    for axes, key in zip(axes_grid.flat, panels):
        for run in runs:
            rows = compared["per_step"][run]
            values = rolling_mean([r.get(key) for r in rows])
            points = [(r["lr_sum"], v) for r, v in zip(rows, values) if v is not None and (v > 0 or "kl" not in key)]
            if points:
                color, dash = style[run]
                axes.plot([x for x, _ in points], [y for _, y in points], color=color, linestyle=dash, linewidth=1.8,
                          label=text["curve_run"][run], zorder=3)
        axes.axvline(compared["run1_collapse"]["lr_sum"], color=muted_ink_color, linewidth=1.0, linestyle=":",
                     zorder=1)
        # reward curves run near 1, so their names go to the empty lower left
        label_y, label_va = (0.04, "bottom") if key.endswith("_reward") else (0.96, "top")
        axes.text(0.03, label_y, text["curve_panel"][key], transform=axes.transAxes, ha="left", va=label_va,
                  fontsize=12, fontweight="bold", color=ink_color)
        if key.endswith("_reward"):
            axes.set_ylim(0, 1.08)
        elif key.endswith("_entropy"):
            axes.set_ylim(0, 1.0)
        else:                                  # one range for both KLs, so a flat one looks flat beside a rising one
            axes.set_yscale("log")
            axes.set_ylim(1.5e-4, 2e-2)
        axes.set_xlim(0, 125)
        add_horizontal_grid(axes)
    axes_grid[0, 0].set_ylabel(text["reward_unit"])
    axes_grid[1, 0].set_ylabel(text["entropy_unit"])
    axes_grid[1, 2].set_ylabel(text["kl_unit"])
    for axes in axes_grid[1]:
        axes.set_xlabel(text["lr_sum"])
    handles, labels = axes_grid[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=len(runs), loc="lower left", bbox_to_anchor=(0.12, 0.92), frameon=False,
                  handlelength=2.2, columnspacing=1.6)
    save_figure(figure, "rl_training_curves", language)


# ---------------------------------------------------------------------------
# 18. one question, every stage
# ---------------------------------------------------------------------------
stage_order = ("base", "sft", "rl_seed1", "rl_v2_seed1")
stage_color = {"base": model_color["base_medium"], "sft": model_color["sft"], "rl_seed1": model_color["rl_seed1"],
               "rl_v2_seed1": run_color["rl_v2_seed1"]}
stage_tint = {"base": "#f4f4f2", "sft": "#eef4fc", "rl_seed1": "#fdf1eb", "rl_v2_seed1": "#f2f0fc"}
right_color, wrong_color = "#1a8f5a", "#c0392b"


def plain_text(text: str) -> str:
    """Model text without the markup that would not read in a box: LaTeX delimiters and \\text{}, markdown bold. Also
    no '$', which matplotlib would read as math."""
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\[ ;,!]", " ", text)                    # LaTeX spacing commands
    text = re.sub(r"\\[()\[\]]", "", text).replace("$", "").replace("**", "")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def wrapped_lines(text: str, width: int, max_lines: int, keep: str = "start") -> list[str]:
    lines = []
    for line in plain_text(text).splitlines():
        lines += textwrap.wrap(line.strip(), width=width) or [""]
    lines = [line for i, line in enumerate(lines) if line or (i and lines[i - 1])]
    return lines[:max_lines] if keep == "start" else lines[-max_lines:]


def draw_stage_outputs(results: dict, language: str) -> None:
    """The same new question answered by every model: each model's first answer (sample 0 of 16) shortened to the start
    and the end of its reasoning and its final answer, and a verdict that counts all 16 answers."""
    text = label_text[language]
    example = results["stage_example"]
    question, gold = example["question"]["question"], example["question"]["answer"]
    figure = plt.figure(figsize=(14.2, 10.6))
    # the answers are English; the Chinese marker lines need a CJK fallback behind the monospace font
    mono_family = ["DejaVu Sans Mono", "Noto Sans CJK SC"] if language == "zh" else ["DejaVu Sans Mono"]
    width_chars, mono = 88, {"family": mono_family, "fontsize": 8.2}
    question_axes = figure.add_axes([0.02, 0.83, 0.96, 0.15])
    question_axes.axis("off")
    question_axes.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.02",
                                           facecolor="#f4f4f2", edgecolor=axis_color, linewidth=1.4,
                                           transform=question_axes.transAxes))
    question_axes.text(0.02, 0.8, text["question"], fontsize=13, fontweight="bold", color=ink_color,
                       transform=question_axes.transAxes)
    question_axes.text(0.98, 0.8, f"{text['expected']}: {gold['value']:.1f} {gold['unit']}", fontsize=12, ha="right",
                       color=secondary_ink_color, transform=question_axes.transAxes)
    question_axes.text(0.02, 0.58, "\n".join(textwrap.wrap(question, 150)), fontsize=11.5, va="top", color=ink_color,
                       transform=question_axes.transAxes, linespacing=1.4)
    for index, stage in enumerate(stage_order):
        row, column = divmod(index, 2)
        axes = figure.add_axes([0.02 + column * 0.49, 0.42 - row * 0.405, 0.47, 0.39])
        axes.axis("off")
        axes.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.02",
                                      facecolor=stage_tint[stage], edgecolor=stage_color[stage], linewidth=1.6,
                                      transform=axes.transAxes))
        axes.text(0.04, 0.92, text["stage_title"][stage], fontsize=13, fontweight="bold", color=ink_color,
                  transform=axes.transAxes)
        model = example["models"].get(stage)
        if model is None:
            axes.text(0.5, 0.5, text["pending"], ha="center", va="center", color=muted_ink_color,
                      transform=axes.transAxes)
            continue
        sample = model["example"]
        body = wrapped_lines(sample["thinking_start"], width_chars, 7)
        if sample["thinking_end"]:
            body += ["", text["tokens_of_reasoning"].format(n=sample["tokens"]), ""]
            body += wrapped_lines(sample["thinking_end"], width_chars, 3, keep="end")
        if sample["finished"]:
            if sample["final_answer_end"]:
                final = (wrapped_lines(sample["final_answer_start"], width_chars, 2) + ["…"]
                         + wrapped_lines(sample["final_answer_end"], width_chars, 3, keep="end"))
            else:                                 # short enough to be kept whole; its end holds the boxed result
                final = wrapped_lines(sample["final_answer_start"], width_chars, 99)
                final = final if len(final) <= 6 else final[:2] + ["…"] + final[-3:]
            body += ["</think>"] + final
        axes.text(0.04, 0.83, "\n".join(body), va="top", color=ink_color, transform=axes.transAxes, linespacing=1.25,
                  **mono)
        if not sample["finished"]:
            axes.text(0.04, 0.2, text["cut_off"].format(n=sample["tokens"]), fontsize=10.5, color=wrong_color,
                      style="italic", transform=axes.transAxes)
        verdict = (text["verdict_right"] if sample["right"] else
                   text["verdict_cut"] if not sample["finished"] else text["verdict_wrong"])
        if sample.get("answer") and sample["finished"]:
            verdict += f":  {plain_text(sample['answer'])}"
        n_samples = len(model["samples"])
        counts = [text["of_n_right"].format(k=model["right"], n=n_samples)]
        if model["unfinished"]:
            counts.append(text["of_n_cut"].format(u=model["unfinished"], n=n_samples))
        counts.append(text["median_tokens"].format(n=model["median_tokens"] / 1000))
        axes.text(0.04, 0.105, f"{'✓' if sample['right'] else '✗'}  {verdict}", fontsize=11.5, fontweight="bold",
                  color=right_color if sample["right"] else wrong_color, transform=axes.transAxes)
        axes.text(0.04, 0.04, "  ·  ".join(counts), fontsize=10.5, color=secondary_ink_color,
                  transform=axes.transAxes)
    save_figure(figure, "stage_outputs", language)


def draw_larger_models(results: dict, language: str) -> None:
    """RL v2 and SFT against two larger Qwen models on D-Num-hard: % right at a 16K and a 32K cap with 95% item
    intervals, and the mean output tokens per answer at the 32K cap (docs/talk/data/large_models_1120.json, written
    by scripts/talk/large_models.py). The layout follows nanogpt-seis-posttrain's comparison with general models."""
    text = label_text[language]
    comparison = load_result_file("large_models_1120.json")
    models = ["rl_v2", "sft", "base_medium", "qwen38_27b_nvfp4", "qwen35_122b_int4"]
    budgets = ["at_16k", "at_32k"]
    rows = comparison["rows"]

    figure, (score_axes, token_axes) = plt.subplots(1, 2, figsize=(12.5, 4.6),
                                                    gridspec_kw={"width_ratios": [2.6, 1.0], "wspace": 0.22})
    bar_width = 0.155                # five models in a group of width 1
    middle = (len(models) - 1) / 2
    for model_index, model in enumerate(models):
        positions = [group + (model_index - middle) * (bar_width + 0.015) for group in range(len(budgets))]
        scores = [rows[model][budget]["score"] for budget in budgets]
        lower = [rows[model][budget]["score"] - rows[model][budget]["score_interval"]["low"] for budget in budgets]
        upper = [rows[model][budget]["score_interval"]["high"] - rows[model][budget]["score"] for budget in budgets]
        score_axes.bar(positions, scores, width=bar_width, color=larger_model_color[model], zorder=2)
        score_axes.errorbar(positions, scores, yerr=[lower, upper], fmt="none", ecolor=ink_color, elinewidth=1.0,
                            capsize=2.5, zorder=3)
        for position, score, top in zip(positions, scores, upper):
            # ours carry their own colour, as in the comparison this figure follows; the others stay in grey ink
            score_axes.text(position, score + top + 1.2, f"{score:.1f}", ha="center", va="bottom", fontsize=11.5,
                            fontweight="bold", color=larger_model_color[model] if model in ("rl_v2", "sft") else
                            secondary_ink_color)
    score_axes.set_xticks(range(len(budgets)), text["larger_budgets"])
    score_axes.tick_params(axis="x", length=0, labelsize=13)
    score_axes.set_ylim(0, 105)
    score_axes.set_yticks([0, 20, 40, 60, 80, 100])
    score_axes.set_ylabel(text["larger_score"])
    add_horizontal_grid(score_axes)

    tokens = [rows[model]["at_32k"]["mean_tokens"] / 1000 for model in models]
    for model_index, (model, value) in enumerate(zip(models, tokens)):
        token_axes.bar(model_index, value, width=0.62, color=larger_model_color[model], zorder=2)
        token_axes.text(model_index, value + 0.4, f"{value:.1f}", ha="center", va="bottom", fontsize=11.5,
                        fontweight="bold", color=larger_model_color[model] if model in ("rl_v2", "sft") else
                        secondary_ink_color)
    token_axes.set_xticks([])
    token_axes.set_ylim(0, max(tokens) * 1.18)
    token_axes.set_ylabel(text["larger_tokens"])
    add_horizontal_grid(token_axes)

    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=larger_model_color[model]) for model in models]
    figure.legend(legend_handles, [text["larger_model"][model] for model in models], ncol=len(models),
                  loc="upper center", bbox_to_anchor=(0.5, 0.02), handlelength=1.0, columnspacing=1.6)
    save_figure(figure, "larger_models", language)


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
    "judge_formats": draw_judge_formats,
    "gradient_by_rule": draw_gradient_by_rule,
    "value_positions": draw_value_positions,
    "pilot_families": draw_pilot_families,
    "rl_runs_compared": draw_rl_runs_compared,
    "rl_training_curves": draw_rl_training_curves,
    "stage_outputs": draw_stage_outputs,
    "larger_models": draw_larger_models,
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
