"""Bank v2 merge and check (dpt.data.bank_merge, 2026-11-07)."""

from __future__ import annotations

import json

from dpt.common.io import read_jsonl, write_jsonl
from dpt.common.manifest import build_manifest, load_manifest, write_manifest
from dpt.data.bank_merge import check, merge

RUB = [{"criterion": "States X", "weight": 2}]


def _t3(uid):
    return {"uid": uid, "task": "t3_report", "question": "q", "answer": {"rubric": RUB, "reference_length": 5}}


def test_merge_dedups_applies_the_map_and_may_overwrite_an_input(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_jsonl(a, [{"uid": "x", "task": "t1_num", "question": "q", "answer": {"value": 1}}, _t3("r1"), _t3("r2")])
    write_manifest(build_manifest(a, name="a", version="v1"))
    write_jsonl(b, [{"uid": "x", "task": "t1_num", "question": "dup", "answer": {"value": 2}},
                    {"uid": "y", "task": "t1_num", "question": "q2", "answer": {"value": 3}}])
    m = tmp_path / "map.jsonl"
    write_jsonl(m, [{"uid": "r1", "status": "ok", "rubric": [{"check": "The answer states X.", "weight": 2,
                                                              "criterion_id": 1}], "rubric_criteria": RUB}])
    res = merge([a, b], a, m)                                   # the output is the first input
    rows = {r["uid"]: r for r in read_jsonl(a)}
    assert sorted(rows) == ["r1", "x", "y"] and rows["x"]["question"] == "q"     # first uid wins; r2 unconverted
    assert rows["r1"]["answer"]["rubric"][0]["check"] == "The answer states X."
    assert res["t3_dropped"] == {"missing": 1, "fallback": 0} and res["per_input"][str(b)] == 1
    assert str(a) in load_manifest(a).parents                  # the input's hash before it was overwritten


def test_check_flags_non_atomic_rubrics_and_easy_banks(tmp_path):
    def row(uid, task, rate=None, rubric=None):
        gt = json.dumps({"rubric": rubric}) if rubric else json.dumps({"value": 1})
        return {"data_source": task, "reward_model": {"ground_truth": gt},
                "extra_info": {"uid": uid, "task": task, "pass_at_8_medium": rate}}

    write_jsonl(tmp_path / "rl_main.jsonl", [row(f"t{i}", "t1_num", 0.875) for i in range(20)]
                + [row("r", "t3_report", rubric=RUB)])
    write_jsonl(tmp_path / "rl_dev.jsonl", [row("d", "t1_num")])
    summary, problems = check(tmp_path)
    assert summary["pass_rate_bands"]["t1_num"]["6_7_of_8"] == 20
    assert any("not atomic" in p for p in problems) and any("5/8 or below" in p for p in problems)
    write_jsonl(tmp_path / "rl_main.jsonl", [row(f"t{i}", "t1_num", 0.25) for i in range(20)]
                + [row("r", "t3_report", rubric=[{"check": "The answer states X.", "weight": 2}])])
    summary, problems = check(tmp_path, reference_dev=tmp_path / "rl_dev.jsonl")
    assert problems == [] and summary["dev_same_as_reference"]
