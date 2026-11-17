"""Bank v2 merge and check (dpt.data.bank_merge, 2026-11-07)."""

from __future__ import annotations

import json

from dpt.common.io import read_jsonl, write_jsonl
from dpt.common.manifest import build_manifest, load_manifest, write_manifest
from dpt.data.bank_merge import (FAULTY_FAMILIES, carve_hard_dev, check, expected_mixed_share, merge, pin_dev,
                                 without_task)

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
    summary, problems = check(tmp_path, min_learnable=0)
    assert summary["pass_rate_bands"]["t1_num"]["6_7_of_8"] == 20
    assert any("not atomic" in p for p in problems) and any("5/8 or below" in p for p in problems)
    write_jsonl(tmp_path / "rl_main.jsonl", [row(f"t{i}", "t1_num", 0.25) for i in range(20)]
                + [row("r", "t3_report", rubric=[{"check": "The answer states X.", "weight": 2}])])
    summary, problems = check(tmp_path, reference_dev=tmp_path / "rl_dev.jsonl", min_learnable=0)
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
    summary, problems = check(tmp_path, t3_not_trained=True, min_learnable=0)
    assert problems == [] and summary["t3_not_atomic"] == 1 and summary["t3_trained"] is False
    assert summary["expected_mixed_t1_at_step_1"] == round(1 - 0.25 ** 8 - 0.75 ** 8, 3)
    write_jsonl(tmp_path / "rl_main.jsonl", [_row(f"t{i}", "t1_num", 0.95) for i in range(20)])
    summary, problems = check(tmp_path, t3_not_trained=True, min_learnable=0)
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
    summary, problems = check(v2, reference_dev=v1 / "rl_dev.jsonl", t3_not_trained=True, min_learnable=0)
    assert any("dev split differs" in p for p in problems)
    res = pin_dev(v2, v1 / "rl_dev.jsonl")
    assert res["removed_from_training"] == {"rl_main": 1, "rl_hard": 1} and res["dev_rows"] == 3
    assert [r["extra_info"]["uid"] for r in read_jsonl(v2 / "rl_main.jsonl")] == ["m1"]
    assert [r["extra_info"]["uid"] for r in read_jsonl(v2 / "rl_hard.jsonl")] == ["h1"]
    assert load_manifest(v2 / "rl_main.jsonl").name == "rl_main"
    assert load_manifest(v2 / "rl_dev.jsonl").parents["pinned_dev"] == load_manifest(v1 / "rl_dev.jsonl").content_hash
    summary, problems = check(v2, reference_dev=v1 / "rl_dev.jsonl", t3_not_trained=True, min_learnable=0)
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


def test_merge_leaves_out_the_faulty_local_magnitude_families(tmp_path):
    """The 2026-11-14 screening: local_magnitude printed the IASPEI constant (A in nm) with A in mm, so its labels
    were 2.681 too low. Bank v2 drops the T1 family and the T2 code task; the long family that also ends in a
    local magnitude uses a relation calibrated for mm and stays."""
    assert FAULTY_FAMILIES == {"local_magnitude", "network_local_magnitude"}
    a = tmp_path / "a.jsonl"
    write_jsonl(a, [
        {"uid": "t1", "task": "t1_num", "source": {"family": "local_magnitude"}, "answer": {"value": 0.3}},
        {"uid": "t2", "task": "t2_code", "source": {"family": "network_local_magnitude"}},
        {"uid": "t2b", "task": "t2_code", "source": {"family": "local_magnitude", "kind": "physics_function"}},
        {"uid": "t2c", "task": "t2_code", "source": {"family": None, "kind": "network_local_magnitude"}},
        {"uid": "t2ok", "task": "t2_code", "source": {"family": None, "kind": "travel_time_residuals"}},
        {"uid": "long", "task": "t1_num", "source": {"family": "locate_then_local_magnitude"}},
        {"uid": "gen", "task": "t1_num", "meta": {"generator": {"name": "local_magnitude"}}},
        {"uid": "ok", "task": "t1_num", "source": {"family": "head_wave_time"}}])
    res = merge([a], tmp_path / "out.jsonl")
    assert sorted(r["uid"] for r in read_jsonl(tmp_path / "out.jsonl")) == ["long", "ok", "t2ok"]
    assert res["faulty_dropped"] == {"local_magnitude": 3, "network_local_magnitude": 2}
    assert "faulty families left out" in load_manifest(tmp_path / "out.jsonl").notes


def test_carve_hard_dev_holds_out_each_family_and_writes_the_family_map(tmp_path):
    """2026-11-15: a few problems of each hard and long family become rl_dev_hard (T1 under its own data_source),
    leave the training files, and the uid → family map comes from the stage file; a second call only reads."""
    stage = tmp_path / "stages" / "05_difficulty.jsonl"
    stage.parent.mkdir()
    fams = {f"h{i}": ("hard_a" if i < 12 else "long_b") for i in range(20)} | {f"v{i}": "v1_fam" for i in range(10)}
    write_jsonl(stage, [{"uid": u, "task": "t1_num", "source": {"family": f}} for u, f in fams.items()])
    write_jsonl(tmp_path / "rl_main.jsonl", [_row(u, "t1_num", 0.5) for u in fams if u != "h3"])
    write_jsonl(tmp_path / "rl_hard.jsonl", [_row("h3", "t1_num", 0.0)])
    res = carve_hard_dev(tmp_path, [stage], ["hard_a", "long_b", "missing_c"], per_family=3)
    # long_b has 8 problems: at most a third of them (2) go to the dev split
    assert res["held_out"] == 5 and res["per_family_held"] == {"hard_a": 3, "long_b": 2, "missing_c": 0}
    assert res["per_family_train"] == {"hard_a": 9, "long_b": 6, "missing_c": 0}
    assert res["missing_families"] == ["missing_c"]
    dev = list(read_jsonl(tmp_path / "rl_dev_hard.jsonl"))
    assert {r["data_source"] for r in dev} == {"t1_num_hard"} and {r["extra_info"]["split"] for r in dev} == {"dev"}
    held = {r["extra_info"]["uid"] for r in dev}
    train = [r["extra_info"]["uid"] for n in ("rl_main_train", "rl_hard_train") for r in read_jsonl(tmp_path / f"{n}.jsonl")]
    assert len(train) == 30 - 5 and not held & set(train)
    assert json.loads((tmp_path / "uid_family.json").read_text())["v1"] == "v1_fam"
    assert carve_hard_dev(tmp_path, [stage], ["hard_a", "long_b", "missing_c"], per_family=3)["skipped"]
    assert not carve_hard_dev(tmp_path, [stage], ["hard_a"], per_family=2).get("skipped")   # other settings: redo


def test_carve_hard_dev_leaves_two_thirds_of_a_small_family_for_training(tmp_path):
    """2026-11-17: a readmitted family can keep only the few problems the SFT model solved, so the dev split takes at
    most a third of each family, none of one with two problems, and never fails for a small family."""
    stage = tmp_path / "stages" / "05_difficulty.jsonl"
    stage.parent.mkdir()
    sizes = {"big": 40, "eleven": 11, "two": 2}
    fams = {f"{name}{i}": name for name, n in sizes.items() for i in range(n)}
    write_jsonl(stage, [{"uid": u, "task": "t1_num", "source": {"family": f}} for u, f in fams.items()])
    write_jsonl(tmp_path / "rl_main.jsonl", [_row(u, "t1_num", 0.5) for u in fams])
    write_jsonl(tmp_path / "rl_hard.jsonl", [])
    res = carve_hard_dev(tmp_path, [stage], list(sizes), per_family=10)
    assert res["per_family_held"] == {"big": 10, "eleven": 3, "two": 0}
    assert res["per_family_train"] == {"big": 30, "eleven": 8, "two": 2}
    assert res["held_out"] == 13 and res["missing_families"] == []


def test_check_needs_enough_learnable_t1_problems_and_lists_families(tmp_path):
    """2026-11-15: at least MIN_LEARNABLE_T1 T1 main problems graded 1–6 of 8, and each family by grade."""
    from dpt.data.bank_merge import MIN_LEARNABLE_T1

    (tmp_path / "stages").mkdir()
    write_jsonl(tmp_path / "stages" / "05_difficulty.jsonl",
                [{"uid": f"t{i}", "task": "t1_num", "source": {"family": "fam_a" if i < 5 else "fam_b"}} for i in range(12)])
    write_jsonl(tmp_path / "rl_main.jsonl", [_row(f"t{i}", "t1_num", 0.5) for i in range(10)]
                + [_row("t10", "t1_num", 0.875)])
    write_jsonl(tmp_path / "rl_hard.jsonl", [_row("t11", "t1_num", 0.0)])
    summary, problems = check(tmp_path, t3_not_trained=True)
    assert summary["learnable_t1"] == 10 and MIN_LEARNABLE_T1 == 1000
    assert any("graded 1–6 of 8" in p for p in problems)
    assert summary["t1_families"] == {"fam_a": {"3_4_of_8": 5}, "fam_b": {"3_4_of_8": 5, "7_of_8": 1, "hard_pool": 1}}
    assert not any("graded 1–6 of 8" in p for p in check(tmp_path, t3_not_trained=True, min_learnable=10)[1])


def test_check_keeps_the_24k_view_for_24k_runs_and_reports_the_16k_view(tmp_path):
    """2026-11-15: problems solved 8/8 at 24K stay in the bank for the 16K seed; the 24K gates leave them out."""
    rows = [_row(f"a{i}", "t1_num", 0.5) for i in range(20)] + [_row(f"b{i}", "t1_num", 1.0) for i in range(80)]
    for i, r in enumerate(rows):
        r["extra_info"]["pass_at_8_medium_16k"] = 0.25 if i < 60 else 1.0
    write_jsonl(tmp_path / "rl_main.jsonl", rows)
    summary, problems = check(tmp_path, t3_not_trained=True, min_learnable=0)
    assert summary["learnable_t1"] == 20 and summary["expected_mixed_t1_at_step_1"] > 0.9    # the 80 left out
    assert summary["learnable_t1_16k"] == 60 and not any("mixed results" in p for p in problems)
    assert "learnable_t1_20k" not in summary                                     # no 20K rates: no 20K view
    for i, r in enumerate(rows):
        r["extra_info"]["pass_at_8_medium_20k"] = 0.5 if i < 70 else 1.0
    write_jsonl(tmp_path / "rl_main.jsonl", rows)
    summary, _ = check(tmp_path, t3_not_trained=True, min_learnable=0)
    assert summary["learnable_t1_20k"] == 70 and summary["expected_mixed_t1_at_step_1_20k"] > 0.6


def test_merge_readmits_templates_whose_verifiers_were_undecided(tmp_path):
    """2026-11-16: a template rejected only because the verifiers agreed on no answer (all undecided, none contested)
    comes back marked label_check=sft; a contested template does not; other merges readmit nothing."""
    def row(uid, fam):
        return {"uid": uid, "task": "t1_num", "question": f"{uid}?", "answer": {"value": 1.0, "unit": "s"},
                "source": {"family": fam}, "meta": {"family": fam, "template": 0, "generator": {"name": fam}}}
    bank = tmp_path / "v1_long"
    (bank / "stages").mkdir(parents=True)
    (bank / "reports").mkdir()
    decontam = [row(f"k{i}", "kept_fam") for i in range(3)] + [row(f"u{i}", "slow_fam") for i in range(4)] \
        + [row(f"c{i}", "bad_fam") for i in range(2)]
    write_jsonl(bank / "stages" / "03_decontam.jsonl", decontam)
    write_jsonl(bank / "stages" / "04_verify.jsonl", decontam[:3])
    (bank / "reports" / "verify_groups.json").write_text(json.dumps({
        "t1_num:kept_fam:t0": {"audited": 3, "accepted": 3, "contested": 0, "perturbation": 0, "kept": True,
                               "reasons": {"agree": 3}},
        "t1_num:slow_fam:t0": {"audited": 4, "accepted": 0, "contested": 0, "perturbation": 0, "kept": False,
                               "reasons": {"family_undecided": 4}},
        "t1_num:bad_fam:t0": {"audited": 2, "accepted": 0, "contested": 2, "perturbation": 0, "kept": False,
                              "reasons": {"suspect_label": 2}}}))
    out = tmp_path / "merged.jsonl"
    res = merge([bank / "stages" / "04_verify.jsonl"], out)
    rows = {r["uid"]: r for r in read_jsonl(out)}
    assert set(rows) == {"k0", "k1", "k2", "u0", "u1", "u2", "u3"} and res["readmitted"] == {
        str(bank / "stages" / "04_verify.jsonl"): 4}
    assert rows["u0"]["meta"]["label_check"] == "sft" and "label_check" not in rows["k0"]["meta"]
    other = merge([bank / "stages" / "04_verify.jsonl"], tmp_path / "other.jsonl", name="rl_bank.difficulty")
    assert other["rows"] == 3
