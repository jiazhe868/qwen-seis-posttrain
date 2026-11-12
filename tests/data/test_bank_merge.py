"""Bank v2 merge and check (dpt.data.bank_merge, 2026-11-07)."""

from __future__ import annotations

import json

from dpt.common.io import read_jsonl, write_jsonl
from dpt.common.manifest import build_manifest, load_manifest, write_manifest
from dpt.data.bank_merge import check, expected_mixed_share, merge, pin_dev, without_task

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


def _row(uid, task, rate=None, rubric=None):
    gt = json.dumps({"rubric": rubric}) if rubric else json.dumps({"value": 1})
    return {"data_source": task, "reward_model": {"ground_truth": gt},
            "extra_info": {"uid": uid, "task": task, "pass_at_8_medium": rate}}


def test_a_run_without_t3_keeps_its_rubrics_but_the_easy_bank_gate_still_holds(tmp_path):
    """RL v2 (2026-11-11) trains no T3: non-atomic rubrics are counted, not fatal. The step-1 mixed-group estimate
    fails a bank whose T1 problems are nearly solved (run 1: 13% of T1 groups mixed in steps 1-20)."""
    write_jsonl(tmp_path / "rl_main.jsonl", [_row(f"t{i}", "t1_num", 0.25) for i in range(20)]
                + [_row("r", "t3_report", rubric=RUB)])
    summary, problems = check(tmp_path, t3_not_trained=True)
    assert problems == [] and summary["t3_not_atomic"] == 1 and summary["t3_trained"] is False
    assert summary["expected_mixed_t1_at_step_1"] == round(1 - 0.25 ** 8 - 0.75 ** 8, 3)
    write_jsonl(tmp_path / "rl_main.jsonl", [_row(f"t{i}", "t1_num", 0.95) for i in range(20)])
    summary, problems = check(tmp_path, t3_not_trained=True)
    assert any("mixed results at step 1" in p for p in problems)
    assert expected_mixed_share([0.5]) == 1 - 2 * 0.5 ** 8 and expected_mixed_share([]) == 0.0


def test_without_task_writes_the_split_and_a_manifest_traced_to_it(tmp_path):
    dev = tmp_path / "rl_dev.jsonl"
    write_jsonl(dev, [_row("d1", "t1_num"), _row("d2", "t3_report", rubric=RUB), _row("d3", "t2_code")])
    write_manifest(build_manifest(dev, name="rl_dev", version="v2"))
    res = without_task(dev, "t3_report", tmp_path / "rl_dev_no_t3")
    out = tmp_path / "rl_dev_no_t3.jsonl"
    assert res["rows"] == 2 and res["dropped"] == 1
    assert [r["extra_info"]["uid"] for r in read_jsonl(out)] == ["d1", "d3"]
    m = load_manifest(out)
    assert m.name == "rl_dev_without_t3_report" and m.version == "v2"
    assert m.parents == {str(dev): load_manifest(dev).content_hash}


def test_pin_dev_restores_the_reference_split_and_keeps_it_out_of_training(tmp_path):
    """Bank v2 (2026-11-11): the SFT model drops dev problems it solves 8/8, so publish draws a different split.
    pin-dev puts run 1's split back and removes its problems from both training pools."""
    v1, v2 = tmp_path / "v1", tmp_path / "v2"
    v1.mkdir()
    v2.mkdir()
    write_jsonl(v1 / "rl_dev.jsonl", [_row("d1", "t1_num"), _row("d2", "t1_num"), _row("d3", "t3_report", rubric=RUB)])
    write_manifest(build_manifest(v1 / "rl_dev.jsonl", name="rl_dev", version="v1"))
    write_jsonl(v2 / "rl_main.jsonl", [_row("d1", "t1_num", 0.5), _row("m1", "t1_num", 0.5)])
    write_manifest(build_manifest(v2 / "rl_main.jsonl", name="rl_main", version="v2"))
    write_jsonl(v2 / "rl_hard.jsonl", [_row("d2", "t1_num", 0.0), _row("h1", "t1_num", 0.0)])
    write_jsonl(v2 / "rl_dev.jsonl", [_row("x9", "t1_num", 0.25)])                      # what publish drew
    summary, problems = check(v2, reference_dev=v1 / "rl_dev.jsonl", t3_not_trained=True)
    assert any("dev split differs" in p for p in problems)
    res = pin_dev(v2, v1 / "rl_dev.jsonl")
    assert res["removed_from_training"] == {"rl_main": 1, "rl_hard": 1} and res["dev_rows"] == 3
    assert [r["extra_info"]["uid"] for r in read_jsonl(v2 / "rl_main.jsonl")] == ["m1"]
    assert [r["extra_info"]["uid"] for r in read_jsonl(v2 / "rl_hard.jsonl")] == ["h1"]
    assert load_manifest(v2 / "rl_main.jsonl").name == "rl_main"
    assert load_manifest(v2 / "rl_dev.jsonl").parents["pinned_dev"] == load_manifest(v1 / "rl_dev.jsonl").content_hash
    summary, problems = check(v2, reference_dev=v1 / "rl_dev.jsonl", t3_not_trained=True)
    assert problems == [] and summary["dev_same_as_reference"]


def test_pin_dev_gives_the_dev_reports_their_atomic_checks(tmp_path):
    v1, v2 = tmp_path / "v1", tmp_path / "v2"
    v1.mkdir()
    v2.mkdir()
    write_jsonl(v1 / "rl_dev.jsonl", [_row("d3", "t3_report", rubric=RUB), _row("d4", "t3_report", rubric=RUB)])
    m = tmp_path / "map.jsonl"
    checks = [{"check": "The answer states X.", "weight": 2, "criterion_id": 1}]
    write_jsonl(m, [{"uid": "d3", "status": "ok", "rubric": checks}, {"uid": "d4", "status": "fallback", "rubric": []}])
    res = pin_dev(v2, v1 / "rl_dev.jsonl", t3_map=m)
    rows = {r["extra_info"]["uid"]: json.loads(r["reward_model"]["ground_truth"]) for r in read_jsonl(v2 / "rl_dev.jsonl")}
    assert res["t3_rubrics_converted"] == 1 and rows["d3"]["rubric"] == checks and rows["d4"]["rubric"] == RUB
