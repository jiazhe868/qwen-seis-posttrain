"""One page to watch RL v2 seeds against the rules that would pause them (2026-11-16).

RL v2 counts a cut-off numeric answer as wrong, so hard answers are meant to get shorter, and a fall in length says
little on its own (README §8.3, "What to watch during training"). This page draws the signals that tell a useful fall
from a harmful one, step by step, next to the thresholds of the seed's own pause rules. The rules are not written
again here: the metrics are replayed through dpt.monitoring.alerts.AlertManager, built from the seed's config, so the
alerts marked on the page are the ones the trainer raised or would have raised.

  top row, the hard dev split (10 held-out problems of each hard and long family, 4 answers each):
    right                 dev/t1_hard, and its best so far minus the pause margin
    wrong value           dev/t1_hard_wrong_value, and its lowest so far plus the margin
    unfinished, value in  dev/t1_hard_unfinished_seen: answers that ran out with the right value already written
    length                dev/t1_hard_len, the floor, and half of its highest value within the halving window
  bottom row, training and the main dev split:
    length gap in groups  task/t1_num/len_gap_within_finished: up-weighted minus down-weighted answers, finished ones
                          only, inside the groups with gradient. Near 0 is healthy; clearly negative once hard answers
                          are short means right answers themselves are getting shorter. The dashed line is the same
                          over all answers, negative by design, since cut-off answers are the longest and count wrong.
    groups with gradient  1 - task/t1_num/zero_adv_frac
    dev D-Num             dev/d_num, and its best so far minus the margin
    dev wrong value       dev/t1_wrong_value, and its lowest so far plus the margin

    python scripts/talk/rl_v2_watch.py --run seed1=<run dir>/metrics.jsonl --run seed3=<...>/metrics.jsonl \\
        --reference run1=<run 1>/metrics.jsonl --out rl_v2_watch.png

Prints, per run, the latest dev pass and how far each pause rule is from firing, and every alert the replay raised.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dpt.common.config import MODEL_LINE_ENV, to_container  # noqa: E402
from dpt.launch.common import compose_stage  # noqa: E402
from dpt.monitoring.alerts import AlertManager  # noqa: E402

# the talk deck's categorical order (scripts/talk/render.py), RL orange first; run 1 underneath in grey
run_colors = ("#eb6834", "#2a78d6", "#1baf7a", "#eda100")
reference_color = "#b9b8b0"
ink_color, muted_ink_color, grid_color = "#0b0b0b", "#898781", "#e1e0d9"

# panel: (metric, title, scale, the pause rule whose margin is drawn, or None)
panels = (
    ("dev/t1_hard", "hard dev: right (%)", 1.0, "dev_t1_hard_best_drop"),
    ("dev/t1_hard_wrong_value", "hard dev: wrong value (%)", 1.0, "dev_t1_hard_wrong_value_rise"),
    ("dev/t1_hard_unfinished_seen", "hard dev: ran out, value written (%)", 1.0, "dev_t1_hard_unfinished_seen_rise"),
    ("dev/t1_hard_len", "hard dev: length (tokens)", 1.0, None),
    ("task/t1_num/len_gap_within_finished", "length gap in groups, finished (tokens)", 1.0, None),
    ("task/t1_num/zero_adv_frac", "T1 groups with gradient (%)", -100.0, None),
    ("dev/d_num", "dev D-Num (%)", 1.0, "dev_d_num_best_drop"),
    ("dev/t1_wrong_value", "dev wrong value (%)", 1.0, "dev_t1_wrong_value_rise"),
)


def read_metrics(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    return sorted((row for row in rows if "step" in row), key=lambda row: row["step"])


def metric_series(rows: list[dict], metric: str, scale: float) -> tuple[list[int], list[float]]:
    """Steps and values of one metric; a negative scale draws 1 - value, in percent (no-signal share → gradient)."""
    points = [(int(row["step"]), float(row[metric])) for row in rows if isinstance(row.get(metric), (int, float))]
    values = [100.0 * (1.0 - value) if scale < 0 else scale * value for _, value in points]
    return [step for step, _ in points], values


def running_best(values: list[float], lower_is_better: bool) -> list[float]:
    best, out = None, []
    for value in values:
        best = value if best is None else (min(best, value) if lower_is_better else max(best, value))
        out.append(best)
    return out


def replay_alerts(rows: list[dict], monitoring: dict) -> list:
    """Every alert the trainer's rules raise on these metrics, evaluated step by step as in training."""
    alert_manager = AlertManager.from_config(monitoring)
    alerts = []
    for row in rows:
        alerts += alert_manager.evaluate(int(row["step"]), row)
    return alerts


def draw_page(runs: dict[str, list[dict]], reference: dict[str, list[dict]], rules: dict[str, dict],
              alerts_by_run: dict[str, list], out_path: Path) -> None:
    figure, axes_grid = plt.subplots(2, 4, figsize=(19.0, 8.2), gridspec_kw={"hspace": 0.42, "wspace": 0.26})
    every_run = list(runs.values()) + list(reference.values())
    last_step = max([int(row["step"]) for rows in every_run for row in rows] or [1])
    axes_by_metric = {}
    for axes, (metric, title, scale, rule_name) in zip(axes_grid.flat, panels):
        axes_by_metric[metric] = axes
        if not any(metric_series(rows, metric, scale)[0] for rows in every_run):
            axes.text(0.5, 0.5, "not logged yet", transform=axes.transAxes, ha="center", va="center",
                      color=muted_ink_color, fontsize=11)
            axes.set_yticks([])
        for reference_name, rows in reference.items():
            steps, values = metric_series(rows, metric, scale)
            if steps:
                axes.plot(steps, values, color=reference_color, linewidth=1.6, marker="o" if len(steps) < 8 else None,
                          markersize=4, label=reference_name, zorder=1)
        for run_index, (run_name, rows) in enumerate(runs.items()):
            color = run_colors[run_index % len(run_colors)]
            steps, values = metric_series(rows, metric, scale)
            if not steps:
                continue
            axes.plot(steps, values, color=color, linewidth=2.0, marker="o" if len(steps) < 30 else None,
                      markersize=4, label=run_name, zorder=3)
            rule = rules.get(rule_name or "")
            if rule and rule.get("kind") == "best_drop":
                lower_is_better = rule.get("better", "high") == "low"
                margin = float(rule["max_drop"]) * (1 if lower_is_better else -1)
                axes.plot(steps, [best + margin for best in running_best(values, lower_is_better)], color=color,
                          linewidth=1.0, linestyle="--", zorder=2)
            if metric == "dev/t1_hard_len" and "dev_t1_hard_length_halved" in rules:
                halving = rules["dev_t1_hard_length_halved"]
                window, drop_frac = int(halving.get("window", 30)), float(halving["drop_frac"])
                half_line = [(1 - drop_frac) * max(v for s, v in zip(steps, values) if s >= step - window)
                             for step in steps]
                axes.plot(steps, half_line, color=color, linewidth=1.0, linestyle=":", zorder=2)
            if metric == "task/t1_num/len_gap_within_finished":
                all_steps, all_values = metric_series(rows, "task/t1_num/len_gap_within", 1.0)
                axes.plot(all_steps, all_values, color=color, linewidth=1.0, linestyle="--", alpha=0.7, zorder=2)
        has_data = any(metric_series(rows, metric, scale)[0] for rows in every_run)
        if metric == "dev/t1_hard_len" and "dev_t1_hard_length_floor" in rules and has_data:
            floor = float(rules["dev_t1_hard_length_floor"]["below"]["pause"])
            axes.axhline(floor, color=ink_color, linewidth=1.0, linestyle="--", zorder=2)
        if metric == "task/t1_num/len_gap_within_finished" and has_data:
            axes.axhline(0.0, color=muted_ink_color, linewidth=1.0, zorder=1)
        axes.set_title(title, fontsize=12, loc="left", color=ink_color)
        axes.set_xlim(0, last_step + 1)
        axes.set_xlabel("step", fontsize=11)
        axes.grid(True, axis="y", color=grid_color, linestyle="--", linewidth=0.8)
        axes.set_axisbelow(True)
        for side in ("top", "right"):
            axes.spines[side].set_visible(False)
    # pauses: a triangle on the panel of the metric that fired, at the top, in the run's colour
    for run_index, (run_name, alerts) in enumerate(alerts_by_run.items()):
        for alert in alerts:
            axes = axes_by_metric.get(alert.metric)
            if axes is not None and alert.severity == "pause":
                axes.plot([alert.step], [1.0], transform=axes.get_xaxis_transform(), marker="v", markersize=9,
                          color=run_colors[run_index % len(run_colors)], clip_on=False, zorder=4)
    handles, labels = axes_grid.flat[0].get_legend_handles_labels()
    for axes in axes_grid.flat[1:]:
        for handle, label in zip(*axes.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    figure.legend(handles, labels, loc="upper left", ncol=len(labels), frameon=False, bbox_to_anchor=(0.06, 1.0))
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def print_status(run_name: str, rows: list[dict], rules: dict[str, dict], alerts: list) -> None:
    print(f"\n{run_name}: {len(rows)} steps logged, last step {rows[-1]['step'] if rows else '-'}")
    for metric, title, scale, rule_name in panels:
        steps, values = metric_series(rows, metric, scale)
        if not steps:
            continue
        line = f"  {title:42s} step {steps[-1]:4d}: {values[-1]:9.1f}"
        rule = rules.get(rule_name or "")
        if rule and rule.get("kind") == "best_drop":
            lower_is_better = rule.get("better", "high") == "low"
            best = running_best(values, lower_is_better)[-1]
            limit = best + float(rule["max_drop"]) * (1 if lower_is_better else -1)
            line += f"   best {best:.1f}, pauses {'above' if lower_is_better else 'below'} {limit:.1f}"
        if metric == "dev/t1_hard_len" and "dev_t1_hard_length_floor" in rules:
            line += f"   floor {float(rules['dev_t1_hard_length_floor']['below']['pause']):.0f}"
        print(line)
    # every pause, and the warnings of the rules this page is about; the other warnings only counted
    watched_rules = {rule_name for _, _, _, rule_name in panels if rule_name} | {
        "t1_length_drop", "dev_t1_hard_length_halved", "dev_t1_hard_length_drop", "dev_t1_hard_length_floor"}
    other_warnings: dict[str, int] = {}
    for alert in alerts:
        if alert.severity != "warn" or alert.rule in watched_rules:
            print(f"  alert at step {alert.step}: [{alert.severity}] {alert.rule}: {alert.message}")
        else:
            other_warnings[alert.rule] = other_warnings.get(alert.rule, 0) + 1
    if other_warnings:
        print("  other warnings: " + ", ".join(f"{rule} ×{count}" for rule, count in sorted(other_warnings.items())))


def named_paths(pairs: list[str]) -> dict[str, Path]:
    named = {}
    for pair in pairs:
        name, _, path = pair.partition("=")
        named[name if path else Path(name).parent.name] = Path(path or name)
    return named


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", action="append", default=[], help="name=metrics.jsonl, once per seed")
    parser.add_argument("--reference", action="append", default=[], help="name=metrics.jsonl, drawn in grey")
    parser.add_argument("--config", default="rl/rl_9b_v2.yaml", help="the config whose pause rules apply")
    parser.add_argument("--out", type=Path, default=Path("rl_v2_watch.png"))
    arguments = parser.parse_args()
    os.environ.setdefault(MODEL_LINE_ENV, "qwen35_9b")
    monitoring = to_container(compose_stage(arguments.config))["monitoring"]
    rules = {rule["name"]: rule for rule in list(monitoring["rules"]) + list(monitoring.get("extra_rules") or [])}
    runs = {name: read_metrics(path) for name, path in named_paths(arguments.run).items()}
    reference = {name: read_metrics(path) for name, path in named_paths(arguments.reference).items()}
    alerts_by_run = {name: replay_alerts(rows, monitoring) for name, rows in runs.items()}
    draw_page(runs, reference, rules, alerts_by_run, arguments.out)
    for name, rows in runs.items():
        print_status(name, rows, rules, alerts_by_run[name])
    print(f"\nwrote {arguments.out}")


if __name__ == "__main__":
    main()
