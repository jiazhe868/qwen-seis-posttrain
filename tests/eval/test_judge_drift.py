"""Judge drift during RL: Luna re-scores the T3 dev responses Gemma scored, from verl's validation dumps."""

from __future__ import annotations

import json

import pytest

from dpt.common.io import write_jsonl
from dpt.eval.hosted import ChatReply
from dpt.eval.judge_drift import DriftConfig, assess, judge_input, load_dev_questions, main, select_rows

RUBRIC = [{"criterion": "explains diffusion", "weight": 2}, {"criterion": "gives a time scale", "weight": 1}]


def _gt(i):
    return json.dumps({"rubric": RUBRIC, "reference_length": 700, "n": i}, sort_keys=True)


def _dev(tmp_path, n=6):
    rows = [{"data_source": "t3_report", "reward_model": {"ground_truth": _gt(i)},
             "extra_info": {"task": "t3_report", "question": f"Report question {i}?"}} for i in range(n)]
    rows.append({"data_source": "t1_num", "reward_model": {"ground_truth": "{}"},
                 "extra_info": {"task": "t1_num", "question": "numeric"}})
    write_jsonl(tmp_path / "rl_dev.jsonl", rows)
    return tmp_path / "rl_dev.jsonl"


def _dump(run, step, gemma, n=6, text="Report body {i}."):
    rows = []
    for i in range(n):
        for k in range(2):                                 # val n=2
            rows.append({"input": "...",
                         "output": f"<think>plan</think>{text.format(i=i)} ({k})\n\nReferences\n1. X (2020)",
                         "gts": _gt(i), "score": gemma, "step": step, "task": "t3_report", "reason": "ok",
                         "t3_rubric": gemma})
    rows.append({"output": "x", "gts": "{}", "task": "t1_num", "t3_rubric": 0.0, "reason": "ok"})
    write_jsonl(run / "val_generations" / f"{step}.jsonl", rows)


class Luna:
    model = "gpt-6-luna"

    def __init__(self, score):
        self.score, self.calls = score, 0

    def request(self, messages):
        self.calls += 1
        assert "<<<ANSWER" in messages[-1]["content"] and "References" not in messages[-1]["content"]
        s = self.score
        return ChatReply(json.dumps({"criteria": [{"id": 1, "score": s}, {"id": 2, "score": s}]}), "stop",
                         "gpt-6-luna-2026-09-22")


def test_dev_questions_and_row_selection(tmp_path):
    q = load_dev_questions(_dev(tmp_path))
    assert len(q) == 6 and q[_gt(0)] == "Report question 0?"
    run = tmp_path / "run"
    _dump(run, 0, 0.5)
    rows = select_rows(json_rows(run / "val_generations" / "0.jsonl"), q, 4)
    assert len(rows) == 4 and all(r["answer"].startswith("Report body") for r in rows)
    assert all("References" not in r["answer"] and "plan" not in r["answer"] for r in rows)
    assert all(r["answer"].endswith("(0)") for r in rows)                   # the first sample of each prompt
    assert judge_input("no end of thinking") is None


def test_a_small_dev_split_scores_every_sample_of_each_prompt(tmp_path):
    q, run = load_dev_questions(_dev(tmp_path)), tmp_path / "run"
    _dump(run, 0, 0.5)
    raw = json_rows(run / "val_generations" / "0.jsonl")
    assert len(select_rows(raw, q, 40, min_responses=6)) == 6               # 6 prompts reach it: first samples
    rows = select_rows(raw, q, 40, min_responses=10)                         # both samples of all 6, not 10 rows
    assert len(rows) == 12 and sorted({r["gt"] for r in rows}) == sorted(q)
    assert len(select_rows(raw, q, 40, min_responses=40)) == 12             # val n caps it


def json_rows(path):
    return [json.loads(line) for line in open(path)]


def test_assess_alerts_on_a_growing_gap_or_a_divergence_streak():
    def s(step, g, lu, unparsed=0):
        return {"step": step, "n": 40, "scored": 40 - unparsed, "unparsed": unparsed, "gemma": g, "luna": lu,
                "gap": g - lu}

    cfg = DriftConfig(max_gap_growth=8.0)
    steady = [s(0, 50, 48), s(20, 55, 53), s(40, 60, 57), s(60, 63, 61)]
    assert not assess(steady, cfg)["alert"]
    pulling = [s(0, 50, 48), s(20, 58, 49), s(40, 62, 50), s(60, 66, 51)]
    a = assess(pulling, cfg)
    assert a["alert"] and any("gap grew" in r for r in a["reasons"]) and any("flat or fell" in r for r in a["reasons"])
    noisy = [s(0, 50, 48), s(20, 58, 49), s(40, 62, 50, unparsed=10), s(60, 66, 51, unparsed=10)]
    assert assess(noisy, cfg)["usable_steps"] == [0, 20]                 # unreliable steps are not used


def test_cli_scores_new_steps_only_and_alerts(tmp_path):
    dev, run = _dev(tmp_path), tmp_path / "run"
    for step, g in [(0, 0.50), (20, 0.62), (40, 0.70)]:
        _dump(run, step, g)
    luna = Luna(5.0)                                       # Luna: 50 at every step
    args = ["--run", str(run), "--dev", str(dev), "--n-prompts", "4", "--model-line", "qwen35_9b"]
    assert main(args, chat=luna) == 3                      # Gemma +12, +20 over Luna's flat 50 → alert
    assert luna.calls == 8                                 # 4 prompts < 40 responses: both samples of each,
    #                                                        the same 8 every step → cached after step 0
    summary = json.loads((run / "judge_drift" / "summary.json").read_text())
    assert [st["step"] for st in summary["steps"]] == [0, 20, 40]
    assert summary["steps"][2]["gap"] == pytest.approx(20.0)
    assert summary["served_models"] == ["gpt-6-luna-2026-09-22"]

    _dump(run, 60, 0.70, text="A different report {i}.")
    luna2 = Luna(7.0)
    main(args, chat=luna2)
    assert luna2.calls == 8                                # only step 60 is new
