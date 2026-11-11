"""verl row export (data_source / prompt / reward_model / extra_info)."""

from __future__ import annotations

import json

from dpt.common.io import read_jsonl
from dpt.data.rl_build import EXTRA_INFO_KEYS, dev_candidates, export_rl_bank, problem_to_verl_row, select_dev
from dpt.data.schema import Problem


def _p(q, pool="main", task="t1_num", answer=None):
    return Problem.from_dict({"task": task, "question": q, "answer": answer or {"value": 2.0, "unit": "s"},
                              "meta": {"pool": pool, "pass_at_8_medium": 0.5}})


def test_row_format():
    row = problem_to_verl_row(_p("q?"), index=3)
    assert row["data_source"] == "t1_num" == row["ability"]
    assert row["prompt"][-1] == {"role": "user", "content": "q?"}
    assert json.loads(row["reward_model"]["ground_truth"])["value"] == 2.0
    assert tuple(row["extra_info"]) == EXTRA_INFO_KEYS and row["extra_info"]["index"] == 3


def test_hidden_tests_never_exported():
    answer = {"entry_point": "f", "visible_tests": "assert f()", "hidden_tests_id": "h-1", "language": "python"}
    p = _p("write f", task="t2_code", answer=answer)
    gt = json.loads(problem_to_verl_row(p, index=0)["reward_model"]["ground_truth"])
    assert gt["hidden_tests_id"] == "h-1" and "hidden_tests" not in gt


def test_export_splits_pools(tmp_path):
    info = export_rl_bank([_p("a?"), _p("b?", pool="hard"), _p("c?")], tmp_path)
    assert info["counts"] == {"rl_main": {"t1_num": 2}, "rl_hard": {"t1_num": 1}}
    hard = list(read_jsonl(tmp_path / "rl_hard.jsonl"))
    assert hard[0]["extra_info"]["pool"] == "hard"


def test_question_and_split_in_extra_info():
    row = problem_to_verl_row(_p("Explain the P-wave.", task="t3_report",
                                 answer={"rubric": [{"criterion": "c", "weight": 1}], "reference_length": 800}),
                              index=0)
    assert row["extra_info"]["question"] == "Explain the P-wave."
    assert row["extra_info"]["split"] == "train"


def test_dev_split_is_deterministic_and_disjoint(tmp_path):
    probs = [_p(f"q{i}?") for i in range(40)] + [_p("h?", pool="hard")]
    a = export_rl_bank(probs, tmp_path / "a", dev_fraction=0.1, seed=1)
    b = export_rl_bank(probs, tmp_path / "b", dev_fraction=0.1, seed=1)
    want = {"rl_main": {"t1_num": 36}, "rl_hard": {"t1_num": 1}, "rl_dev": {"t1_num": 4}}
    assert a["counts"] == b["counts"] == want
    dev = {r["extra_info"]["uid"] for r in read_jsonl(tmp_path / "a" / "rl_dev.jsonl")}
    main = {r["extra_info"]["uid"] for r in read_jsonl(tmp_path / "a" / "rl_main.jsonl")}
    assert dev and not (dev & main)
    assert all(r["extra_info"]["split"] == "dev" for r in read_jsonl(tmp_path / "a" / "rl_dev.jsonl"))


def test_dev_copy_if_empty_relabels_main_for_toy_banks(tmp_path):
    probs = [_p("a?"), _p("b?"), _p("h?", pool="hard")]       # too small: the 1/4 cap gives no dev rows
    assert "rl_dev" not in export_rl_bank(probs, tmp_path / "off", dev_fraction=0.5)["files"]
    info = export_rl_bank(probs, tmp_path / "on", dev_fraction=0.5, dev_copy_if_empty=True)
    assert info["counts"]["rl_dev"] == info["counts"]["rl_main"] == {"t1_num": 2}
    rows = read_jsonl(tmp_path / "on" / "rl_dev.jsonl")
    assert all(r["extra_info"]["split"] == "dev" for r in rows)


def test_dev_split_comes_from_candidates_fixed_before_the_difficulty_pass():
    probs = [_p(f"q{i}?") for i in range(400)]
    cand = dev_candidates(probs, fraction=0.05, seed=1)
    assert len(cand) == 40 and select_dev(probs, fraction=0.05, seed=1) <= cand    # twice as deep
    for drop in range(5):                         # the difficulty pass removes 30% of the problems
        kept = [p for i, p in enumerate(probs) if (i * 7 + drop) % 10 >= 3]
        dev = select_dev(kept, fraction=0.05, seed=1, among=cand)
        assert dev == select_dev(kept, fraction=0.05, seed=1) and len(dev) == 14
    assert select_dev(probs, fraction=0.05, seed=1, among=set()) == set()          # never outside them
