"""Atomic yes/no checks for T3 rubrics (dpt.data.gen.t3_atomic, 2026-11-07)."""

from __future__ import annotations

import json

from dpt.data.gen.t3_atomic import AtomicConfig, cmd_apply, convert_one, json_object, structural_problems, summarize
from dpt.data.schema import Problem

RUBRIC = [{"criterion": "States the travel time of P and S and their ratio", "weight": 3},
          {"criterion": "Explains why the S-P time grows with distance", "weight": 1}]


def _teacher(replies):
    calls = []

    def gen(messages, n, **kw):
        calls.append(messages[-1]["content"])
        return [replies[min(len(calls), len(replies)) - 1]]
    gen.calls = calls
    return gen


def _reply(checks):
    return "thinking ...</think>\n" + json.dumps({"criteria": [{"id": i, "checks": c}
                                                               for i, c in enumerate(checks, start=1)]})


GOOD = [["The answer gives the P travel time in seconds.", "The answer gives the S travel time in seconds.",
         "The answer gives the ratio of the S to the P travel time."],
        ["The answer explains that S travels slower than P over the same path."]]


def test_convert_splits_weights_and_keeps_the_criteria():
    judge = _teacher([json.dumps({"criteria": [{"id": 1, "covers": True, "no_additions": True},
                                               {"id": 2, "covers": True, "no_additions": True}]})])
    out = convert_one("u1", "Why does S-P time grow?", RUBRIC, _teacher([_reply(GOOD)]), judge, AtomicConfig())
    assert out["status"] == "ok" and out["attempts"] == 1
    assert [c["criterion_id"] for c in out["rubric"]] == [1, 1, 1, 2]
    assert abs(sum(c["weight"] for c in out["rubric"] if c["criterion_id"] == 1) - 3) < 1e-6   # weight kept
    assert out["rubric_criteria"] == RUBRIC and out["coverage"] == {"criteria": 2, "covers": 2, "no_additions": 2}
    answer = {"rubric": out["rubric"], "reference_length": 500}
    Problem.from_dict({"task": "t3_report", "question": "q", "answer": answer})  # schema accepts


def test_a_rule_violation_is_sent_back_once_then_falls_back():
    bad = [["The answer gives the P time; the S time; the ratio."], GOOD[1]]
    teacher = _teacher([_reply(bad), _reply(GOOD)])
    out = convert_one("u2", "q", RUBRIC, teacher, None, AtomicConfig())
    assert out["status"] == "ok" and out["attempts"] == 2
    assert "previous attempt was rejected" in teacher.calls[1] and "';'" in teacher.calls[1]
    out = convert_one("u3", "q", RUBRIC, _teacher([_reply(bad)]), None, AtomicConfig())
    assert out["status"] == "fallback" and out["rubric"][0]["check"] == RUBRIC[0]["criterion"]


def test_a_cut_off_reply_and_missing_criteria_are_rejected():
    out = convert_one("u4", "q", RUBRIC, _teacher(["thinking that never ends"]), None, AtomicConfig(retries=0))
    assert out["status"] == "fallback" and out["problems"] == ["cut_off"]
    out = convert_one("u5", "q", RUBRIC, _teacher([_reply(GOOD[:1])]), None, AtomicConfig(retries=0))
    assert out["status"] == "fallback" and "missing" in out["problems"][0]


def test_structural_rules():
    assert structural_problems(GOOD, AtomicConfig()) == []
    assert structural_problems([["Too short."]], AtomicConfig())
    assert structural_problems([[f"The answer states fact number {i} here." for i in range(7)]], AtomicConfig())


def test_json_object_reads_thinking_then_json():
    assert json_object('thought\nI will answer.\n{"a": 1}') == {"a": 1}
    assert json_object('```json\n{"a": {"b": 2}}\n```') == {"a": {"b": 2}}


def test_apply_replaces_t3_rubrics_and_drops_unconverted(tmp_path):
    rows = [{"uid": "a", "task": "t3_report", "question": "q", "answer": {"rubric": RUBRIC, "reference_length": 5}},
            {"uid": "b", "task": "t3_report", "question": "q", "answer": {"rubric": RUBRIC, "reference_length": 5}},
            {"uid": "c", "task": "t3_report", "question": "q", "answer": {"rubric": RUBRIC, "reference_length": 5}},
            {"uid": "d", "task": "t1_num", "question": "q", "answer": {"value": 1.0}}]
    ok = convert_one("a", "q", RUBRIC, _teacher([_reply(GOOD)]), None, AtomicConfig())
    fb = {**convert_one("b", "q", RUBRIC, _teacher(["x"]), None, AtomicConfig(retries=0))}
    (tmp_path / "map.jsonl").write_text("\n".join(json.dumps(r) for r in (ok, fb)) + "\n")
    (tmp_path / "in.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    class A:
        map, inp, out, keep_fallback = (str(tmp_path / "map.jsonl"), str(tmp_path / "in.jsonl"),
                                        str(tmp_path / "out.jsonl"), False)
    cmd_apply(A)
    got = {json.loads(line)["uid"]: json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()}
    assert sorted(got) == ["a", "d"]                         # b: fallback, c: not converted
    assert got["a"]["answer"]["rubric"][0]["check"].startswith("The answer")
    assert got["a"]["meta"]["rubric_criteria"] == RUBRIC and got["d"] == rows[3]
    assert summarize([ok, fb])["ok"] == 1
