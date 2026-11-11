"""SFT data by rejection sampling: the training grader decides, cut-off replies never pass, reports, resume."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from dpt.chat.thinking import THINK_END
from dpt.common.io import read_json, read_jsonl, write_jsonl
from dpt.data.gen.common import candidate
from dpt.data.gen.sft_gen import build_router, main
from dpt.data.gen.teacher import Reply
from dpt.data.schema import SFTSample
from dpt.data.sft_build import SFTBuildConfig, admission_reason


class FakeModel:
    """Replies chosen per problem by ``script(question) -> list[Reply]``; counts requests."""

    effort = "medium"
    tokenizer = None

    def __init__(self, script, model_id="Qwen/Qwen3.8-27B"):
        self.script, self.model_id, self.calls = script, model_id, 0

    def generate(self, messages, *, n=1, seed=None, max_tokens=None, temperature=None):
        self.calls += 1
        out = self.script(messages[-1]["content"])
        assert len(out) == n
        return out

    def count_tokens(self, text):
        return len(text) // 4


def _ok(answer, thinking="thinking about it"):
    return Reply(f"{thinking}\n{THINK_END}\n\n{answer}", "stop")


def _t1(d, **meta):
    q = (f"A P wave travels {d} km through the crust at 6.0 km/s before reaching a seismometer. "
         "How long does the trip take, in seconds?")
    return candidate("t1_num", q, {"value": d / 6.0, "unit": "s", "rel_tol": 0.01}, origin="test",
                     family="travel_time", **meta)


def test_domain_keeps_verified_replies_and_reports_pass_rates(tmp_path):
    probs = [_t1(120), _t1(60), _t1(30), _t1(90, split="heldout"), _t1(150)]
    write_jsonl(tmp_path / "bank.jsonl", probs)
    write_jsonl(tmp_path / "dev.jsonl", [{"extra_info": {"uid": probs[4]["uid"]}}])   # a verl dev row

    def script(q):
        if "120 km" in q:
            return [_ok("t = 120/6.0, so \\boxed{20 s}")] * 4
        if "60 km" in q:
            return [_ok("\\boxed{10 s}"), Reply("long thinking", "length"), Reply("no end", "stop"),
                    _ok("\\boxed{12 s}")]
        return [_ok("\\boxed{7 s}")] * 4

    out = tmp_path / "domain.jsonl"
    args = ["domain", "--problems", str(tmp_path / "bank.jsonl"), "--exclude", str(tmp_path / "dev.jsonl"),
            "--out", str(out)]
    m = FakeModel(script)
    assert main(args, model=m) == 0
    assert m.calls == 3                                                     # held-out and dev problems skipped
    samples = list(read_jsonl(out))
    assert len(samples) == 3                                                # 2 of the 4/4 problem, 1 of the 1/4
    for s in samples:
        rec = SFTSample.from_dict(s)
        assert admission_reason(rec, SFTBuildConfig()) is None and rec.meta["verified"] is True
        assert rec.messages[0]["role"] == "system" and rec.messages[-1]["reasoning_content"] == "thinking about it"
        assert "\\boxed" in rec.messages[-1]["content"] and THINK_END not in rec.messages[-1]["content"]
        assert rec.source["model"] == "Qwen/Qwen3.8-27B"
    rep = read_json(f"{out}.report.json")
    assert rep["problems"] == 3 and rep["zero_of_k"] == {"count": 1, "share": 0.3333}
    assert rep["by_family"]["travel_time"]["histogram"] == {"0": 1, "1": 1, "4": 1}
    assert rep["truncated_replies"] == 1 and rep["reply_outcomes"]["no_think_end"] == 1
    assert rep["reply_outcomes"]["truncated"] == 1 and rep["reply_outcomes"]["wrong_value"] >= 5
    assert rep["tokens"]["kept"] > 0 and rep["tokens"]["exact"] is False
    # resumable: every problem is done, so a rerun makes no request and writes nothing new
    m2 = FakeModel(lambda q: pytest.fail("no request expected"))
    assert main(args, model=m2) == 0 and m2.calls == 0 and len(list(read_jsonl(out))) == 3


def _t2(tmp_path):
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    (hidden / "tt-hidden.py").write_text(
        "import solution\n\n\ndef test_a():\n    assert abs(solution.travel_time(120.0, 6.0) - 20.0) < 1e-9\n\n\n"
        "def test_b():\n    assert abs(solution.travel_time(30.0, 3.0) - 10.0) < 1e-9\n")
    vis = "import solution\n\n\ndef test_v():\n    assert solution.travel_time(6.0, 6.0) == 1.0\n"
    t2 = candidate("t2_code", "Write travel_time(distance_km, velocity_km_s) returning the travel time in seconds.",
                   {"entry_point": "travel_time", "visible_tests": vis, "hidden_tests_id": "tt-hidden"},
                   origin="test", family="t2:travel_time")
    write_jsonl(tmp_path / "t2.jsonl", [t2])
    return hidden


def test_tool_runs_hidden_tests_in_the_sandbox(tmp_path, monkeypatch):
    monkeypatch.delenv("DPT_SANDBOX_SIF", raising=False)
    monkeypatch.setenv("DPT_SANDBOX_PYTHON", sys.executable)
    hidden = _t2(tmp_path)
    good = "```python\ndef travel_time(distance_km, velocity_km_s):\n    return distance_km / velocity_km_s\n```"
    bad = "```python\ndef travel_time(distance_km, velocity_km_s):\n    return distance_km * velocity_km_s\n```"
    m = FakeModel(lambda q: [_ok(bad), _ok("no code here"), _ok(good), _ok(good)])
    args = ["tool", "--problems", str(tmp_path / "t2.jsonl"), "--out", str(tmp_path / "tool.jsonl"),
            "--hidden-tests-dir", str(hidden)]
    with pytest.raises(SystemExit, match="sandbox container"):
        main(args, model=m)
    assert main(args + ["--allow-no-container"], model=m) == 0
    samples = list(read_jsonl(tmp_path / "tool.jsonl"))
    assert len(samples) == 1 and "distance_km / velocity_km_s" in samples[0]["messages"][-1]["content"]
    assert admission_reason(SFTSample.from_dict(samples[0]), SFTBuildConfig()) is None
    rep = read_json(tmp_path / "tool.jsonl.report.json")
    assert rep["by_task"]["t2_code"]["histogram"] == {"2": 1}
    assert rep["reply_outcomes"].get("no_code") == 1


def test_report_keeps_scores_above_the_threshold(tmp_path):
    rubric = [{"criterion": "explains impedance contrast", "weight": 3},
              {"criterion": "explains resonance", "weight": 2}, {"criterion": "names HVSR", "weight": 1}]
    t3 = candidate("t3_report", "Explain how soft sediments amplify earthquake ground motion and how it is measured.",
                   {"rubric": rubric, "reference_length": 1031, "prompt_extra": {"length_hint": 550}},
                   origin="test", family="t3:site")
    write_jsonl(tmp_path / "t3.jsonl", [t3])
    scores = iter([0.5, 0.82])
    grade = (lambda p, r: SimpleNamespace(score=next(scores), reason="ok", correct=None))
    m = FakeModel(lambda q: [_ok("Report A body.\n\nReferences\n1. X"), _ok("Report B body.")])
    assert main(["report", "--problems", str(tmp_path / "t3.jsonl"), "--out", str(tmp_path / "rep.jsonl")],
                model=m, grade=grade) == 0
    (s,) = list(read_jsonl(tmp_path / "rep.jsonl"))
    assert s["meta"]["rubric"] == rubric and s["meta"]["judge_score"] == 0.82
    assert s["messages"][-1]["content"] == "Report B body."
    assert s["messages"][1]["content"].endswith("Aim for roughly 550 words.")
    assert admission_reason(SFTSample.from_dict(s), SFTBuildConfig()) is None


def test_report_problems_the_judge_could_not_grade_are_left_for_the_next_run(tmp_path):
    t3 = [candidate("t3_report", f"Explain how soft sediments amplify ground motion, case {i} of twelve.",
                    {"rubric": [{"criterion": "explains impedance contrast", "weight": 1}], "reference_length": 800},
                    origin="test", family="t3:site") for i in range(12)]
    write_jsonl(tmp_path / "t3.jsonl", t3)
    args = ["report", "--problems", str(tmp_path / "t3.jsonl"), "--out", str(tmp_path / "rep.jsonl"),
            "--concurrency", "1"]
    down = (lambda p, r: SimpleNamespace(score=0.0, reason="judge_error", correct=None))   # a judge window ended
    m = FakeModel(lambda q: [_ok("Report A."), _ok("Report B.")])
    assert main(args, model=m, grade=down) == 1
    assert m.calls == 8                                       # stopped after 8 ungraded problems, not all 12
    rep = read_json(tmp_path / "rep.jsonl.report.json")
    assert rep["stopped"] is True and rep["ungraded"] == 8 and rep["problems"] == 0
    assert not (tmp_path / "rep.jsonl").exists() and not (tmp_path / "rep.jsonl.progress.jsonl").exists()
    up = (lambda p, r: SimpleNamespace(score=0.9, reason="ok", correct=None))
    assert main(args, model=FakeModel(lambda q: [_ok("Report A."), _ok("Report B.")]), grade=up) == 0
    assert len(list(read_jsonl(tmp_path / "rep.jsonl"))) == 12
    assert read_json(tmp_path / "rep.jsonl.report.json")["stopped"] is False


def test_report_replies_sampled_without_a_judge_are_graded_later_without_sampling_again(tmp_path):
    t3 = [candidate("t3_report", f"Explain how soft sediments amplify ground motion, case {i} of four.",
                    {"rubric": [{"criterion": "explains impedance contrast", "weight": 1}], "reference_length": 800},
                    origin="test", family="t3:site") for i in range(4)]
    write_jsonl(tmp_path / "t3.jsonl", t3)
    args = ["report", "--problems", str(tmp_path / "t3.jsonl"), "--out", str(tmp_path / "rep.jsonl")]
    m = FakeModel(lambda q: [_ok("Report A."), _ok("Report B.")])
    no_judge = (lambda p, r: pytest.fail("nothing is graded while only generating"))
    assert main(args + ["--generate-only"], model=m, grade=no_judge) == 0 and m.calls == 4
    assert not (tmp_path / "rep.jsonl.progress.jsonl").exists()
    assert main(args + ["--generate-only"], model=m, grade=no_judge) == 0 and m.calls == 4   # all sampled already
    up = (lambda p, r: SimpleNamespace(score=0.9, reason="ok", correct=None))
    m2 = FakeModel(lambda q: pytest.fail("graded from the kept replies"))
    assert main(args, model=m2, grade=up) == 0 and m2.calls == 0
    assert len(list(read_jsonl(tmp_path / "rep.jsonl"))) == 4


def test_stop_at_ends_sampling_and_keeps_what_is_done(tmp_path):
    write_jsonl(tmp_path / "bank.jsonl", [_t1(d) for d in (30, 60, 90, 120)])
    args = ["domain", "--problems", str(tmp_path / "bank.jsonl"), "--out", str(tmp_path / "d.jsonl")]
    m = FakeModel(lambda q: pytest.fail("nothing starts after --stop-at"))
    assert main(args + ["--stop-at", "2000-01-01T00:00"], model=m) == 0 and m.calls == 0
    assert read_json(tmp_path / "d.jsonl.report.json")["past_stop_at"] is True
    m = FakeModel(lambda q: [_ok("\\boxed{5 s}")] * 4)
    assert main(args + ["--stop-at", "2999-01-01T00:00"], model=m) == 0 and m.calls == 4


def test_distill_math_must_be_correct_ifeval_well_formed(tmp_path):
    math = candidate("t4_general", "What is two plus two, counted carefully on your fingers?",
                     {"kind": "math", "value": 4, "unit": ""}, origin="test", family="t4:math")
    ife = candidate("t4_general", "Describe a seismometer in plain words for a child.",
                    {"kind": "ifeval", "constraints": [{"type": "all_lowercase"}]}, origin="test", family="t4:if")
    write_jsonl(tmp_path / "t4.jsonl", [math, ife])

    def script(q):
        if "two plus two" in q:
            return [_ok("\\boxed{5}")]
        return [_ok("a seismometer is a box that feels the ground shake.")]

    m = FakeModel(script, model_id="Qwen/Qwen3.5-9B")
    assert main(["distill", "--problems", str(tmp_path / "t4.jsonl"), "--out", str(tmp_path / "sd.jsonl")],
                model=m) == 0
    (s,) = list(read_jsonl(tmp_path / "sd.jsonl"))                       # the wrong math answer is dropped
    assert s["meta"]["generated_by"] == "Qwen/Qwen3.5-9B" and s["messages"][0]["role"] == "user"
    cfg = SFTBuildConfig(self_distill_generated_by="Qwen/Qwen3.5-9B")
    assert admission_reason(SFTSample.from_dict(s), cfg) is None


def test_rewards_config_keeps_interpolations_and_applies_overrides(tmp_path):
    p = tmp_path / "rewards.yaml"
    assert main(["rewards-config", "--out", str(p), "rewards.t2_code.sandbox.hidden_tests_dir=/hidden",
                 "rewards.t3_report.judge.model=judge"]) == 0
    assert "${project.thinking" in p.read_text()
    r = build_router(str(p))
    assert r.cfg["t2_code"]["sandbox"]["hidden_tests_dir"] == "/hidden"
    assert r.cfg["t3_report"]["judge"]["model"] == "judge"
    assert isinstance(r.cfg["length"]["max_response_tokens"], int)
    json.dumps(r.cfg)                                                       # fully resolved


def test_model_requests_may_wait_in_a_busy_server_queue():
    import argparse

    from dpt.data.gen.teacher import add_model_args, model_from_args
    ap = argparse.ArgumentParser()
    add_model_args(ap)
    m = model_from_args(ap.parse_args(["--url", "http://h100:8000", "--model", "teacher"]))
    assert m.clients[0]._http.timeout.read == 7200.0          # a 20K-token reply took ~26 min on the shared H100
    m = model_from_args(ap.parse_args(["--url", "http://h100:8000", "--model", "teacher", "--timeout", "60"]))
    assert m.clients[0]._http.timeout.read == 60.0
