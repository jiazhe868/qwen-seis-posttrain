"""Gates and report: the report's final-results table must pass all 8 ship gates; regressions must fail."""

from __future__ import annotations

import copy
import os
import time

import pytest

import dpt.eval.gates as gates
from dpt.common.config import load_config, to_container
from dpt.common.io import write_json
from dpt.eval.gates import evaluate_entry_gate, evaluate_ship_gates, format_table, hold_for_decision
from dpt.eval.report import build_summary, render_markdown

GENERAL = ["gpqa_diamond", "mmlu_pro_2k", "ifeval", "livecodebench_post_2025_01", "tool_calling"]


def _results(name, d_num, d_code, d_report, d_fresh, gen_delta, tokens, refusal, p16=None):
    def s(score, ci, metric):
        return {
            "score": score,
            "ci": {"halfwidth": ci},
            "metric": metric,
            "mean_output_tokens": tokens,
            "n_items": 1,
            "n_samples": 1,
        }

    sets = {
        "d_num": s(d_num, 1.6, "avg@8"),
        "d_code": s(d_code, 2.4, "pass@1"),
        "d_report": s(d_report, 4.0, "winrate_vs_base"),
        "d_fresh": s(d_fresh, 3.1, "avg@8"),
        "refusal": {"refusal_rate": refusal},
    }
    if p16 is not None:
        sets["d_num"]["pass_at_k"] = {"16": p16}
    general = {b: {"score": 60.0 + gen_delta, "relative_only": b == "mmlu_pro_2k"} for b in GENERAL}
    return {"model_name": name, "sets": sets, "general": general, "mean_output_tokens": tokens}


# Numbers from the report's final table (refusal rates: base 10.0, delivered +0.7 pp).
ROWS = {
    "base_medium": _results("base_medium", 52.1, 38.0, 50.0, 44.7, 0.0, 5900, 10.0, p16=81.7),
    "base_xhigh": _results("base_xhigh", 55.3, 40.2, 53.0, 47.0, 0.9, 14800, 10.0),
    "sft": _results("sft", 60.8, 44.5, 61.0, 51.2, -1.1, 5200, 10.2, p16=83.0),
    "rl_seed1": _results("rl_seed1", 71.4, 55.2, 68.0, 60.3, -0.4, 6400, 10.7),
    "rl_seed2": _results("rl_seed2", 70.1, 54.6, 67.0, 59.5, -0.6, 6200, 10.5),
}


@pytest.fixture()
def summary(tmp_path):
    dirs = {}
    for name, res in ROWS.items():
        d = tmp_path / name
        write_json(d / "results.json", res)
        dirs[name] = d
    return build_summary(dirs)


@pytest.fixture()
def project():
    # the report's numbers belong to the report's line, whatever DPT_MODEL_LINE says (the cluster sets qwen35_9b)
    return to_container(load_config(model_line="qwen38_27b"))["project"]


def test_report_numbers_pass_all_ship_gates(summary, project):
    res = evaluate_ship_gates(summary, project["gates"], "rl_seed1", seed2="rl_seed2")
    table = format_table(res)
    assert all(r.passed for r in res), table
    assert len(res) == 8
    by = {r.name: r for r in res}
    assert by["cost"].observed == "1.08×"
    assert by["repro"].observed == "1.30"


def test_regressions_fail(summary, project):
    bad = copy.deepcopy(summary)
    bad["rows"]["rl_seed1"]["d_num"]["score"] = 62.0  # only +6.7 over base xhigh
    bad["rows"]["rl_seed1"]["general_mean_delta"] = -1.4
    bad["rows"]["rl_seed1"]["mean_output_tokens"] = 9000
    res = {r.name: r for r in evaluate_ship_gates(bad, project["gates"], "rl_seed1", seed2="rl_seed2")}
    assert not res["d_num"].passed and not res["general"].passed and not res["cost"].passed


def test_missing_seed2_fails_repro_gate(summary, project):
    res = {r.name: r for r in evaluate_ship_gates(summary, project["gates"], "rl_seed1")}
    assert not res["repro"].passed and res["repro"].observed == "missing"


def test_winrate_gate_requires_lower_bound_above_50(summary, project):
    s = copy.deepcopy(summary)
    s["rows"]["rl_seed1"]["d_report"].update(score=61.0, ci_halfwidth=12.0)
    res = {r.name: r for r in evaluate_ship_gates(s, project["gates"], "rl_seed1", seed2="rl_seed2")}
    assert not res["d_report"].passed


def test_rl_entry_gate(summary, project):
    res = evaluate_entry_gate(summary, project["rl_entry_gate"], "sft")
    assert all(r.passed for r in res), format_table(res)
    s = copy.deepcopy(summary)
    s["rows"]["sft"]["pass_at_k"]["d_num"]["16"] = 80.0
    assert not all(r.passed for r in evaluate_entry_gate(s, project["rl_entry_gate"], "sft"))


def _entry_summary(root, sft_num, sft_p16):
    """The 9B baseline (D-Num 91.6, pass@16 99.8) and an SFT row that clears every other entry check."""
    rows = {"base_medium": _results("base_medium", 91.6, 80.5, 50.0, 70.6, 0.0, 14700, 56.0, p16=99.8),
            "sft": _results("sft", sft_num, 82.0, 58.0, 72.0, -0.5, 12000, 56.5, p16=sft_p16)}
    for name, res in rows.items():
        write_json(root / name / "results.json", res)
    return build_summary({name: root / name for name in rows})


def test_9b_entry_gate_allows_noise_at_the_d_num_ceiling(tmp_path, monkeypatch):
    monkeypatch.setenv("WORK", str(tmp_path))
    gate = to_container(load_config(model_line="qwen35_9b"))["project"]["rl_entry_gate"]
    assert gate["domain_beats_base"] == ["d_code", "d_report"] and gate["domain_not_below_base"] == ["d_num"]
    assert gate["on_fail"] == "hold" and gate["decision_file"] == f"{tmp_path}/dpt/campaign/GATE_DECISION"
    report = to_container(load_config(model_line="qwen38_27b"))["project"]["rl_entry_gate"]
    assert report["domain_beats_base"] == ["d_num", "d_code", "d_report"] and "on_fail" not in report
    # D-Num CIs ±1.6 each → ±2.26 combined; pass@16 may trail by 1 point
    ok = {r.name: r.passed for r in evaluate_entry_gate(_entry_summary(tmp_path / "a", 89.5, 99.0), gate, "sft")}
    assert all(ok.values()), ok
    bad = {r.name: r.passed for r in evaluate_entry_gate(_entry_summary(tmp_path / "b", 89.0, 98.7), gate, "sft")}
    assert not bad["entry:d_num"] and not bad["entry:pass@16"] and bad["entry:d_code"] and bad["entry:general"]


def test_hold_waits_for_a_fresh_go_or_stop(tmp_path):
    f = tmp_path / "GATE_DECISION"
    gate, t = {"decision_file": str(f), "hold_hours": 1.0}, [time.time()]

    def answer(word):
        def sleep(s):
            t[0] += s
            f.write_text(word + "\n")
        return sleep

    assert hold_for_decision(gate, clock=lambda: t[0], sleep=answer("go")) == 0
    f.unlink()
    assert hold_for_decision(gate, clock=lambda: t[0], sleep=answer("STOP")) == 1
    # an answer from before the hold is ignored, and no answer in time ends the job
    f.write_text("go\n")
    os.utime(f, (t[0] - 3600, t[0] - 3600))
    assert hold_for_decision(gate, clock=lambda: t[0], sleep=lambda s: t.__setitem__(0, t[0] + s)) == 1


def test_failed_entry_gate_holds_then_starts_rl_on_go(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DPT_MODEL_LINE", "qwen35_9b")
    monkeypatch.setenv("WORK", str(tmp_path))
    write_json(tmp_path / "summary.json", _entry_summary(tmp_path / "runs", 85.0, 99.8))   # D-Num far below
    decision = tmp_path / "dpt" / "campaign" / "GATE_DECISION"

    def answer(_):
        decision.parent.mkdir(parents=True, exist_ok=True)
        decision.write_text("go")

    monkeypatch.setattr(gates.time, "sleep", answer)
    assert gates.main(["--summary", str(tmp_path / "summary.json"), "--candidate", "sft", "--entry"]) == 0
    out = capsys.readouterr().out
    assert "| entry:d_num | FAIL |" in out and "RL is held" in out and "says go or stop" in out


def test_summary_and_markdown(summary):
    assert summary["rows"]["sft"]["general_mean_delta"] == pytest.approx(-1.1)
    md = render_markdown(summary, {"rl_seed1": "SFT + RL (seed 1)"})
    assert "| SFT + RL (seed 1) | 71.4 ±1.6 | 55.2 ±2.4 | 68.0% ±4.0 | 60.3 ±3.1 | -0.4 | 6.4K |" in md
    assert "mmlu_pro_2k" in md
