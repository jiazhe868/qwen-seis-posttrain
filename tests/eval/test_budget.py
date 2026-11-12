"""Accuracy under a token budget, regraded from a run's generation cache."""

from __future__ import annotations

from dpt.common.io import write_json, write_jsonl
from dpt.eval.budget import analyze_run, main

GOLD = {"value": 20, "unit": "s", "rel_tol": 0.01, "abs_tol": 0}
RIGHT, WRONG = "t = 20\n</think>\n\n\\boxed{20 s}", "hmm\n</think>\n\n\\boxed{25 s}"


def _sample(text, tokens, finish="stop"):
    return {"text": text, "finish_reason": finish, "n_tokens": tokens}


def _run(tmp_path):
    write_jsonl(tmp_path / "d_num.jsonl", [{"id": "a", "question": "q", "answer": GOLD},
                                           {"id": "b", "question": "q", "answer": GOLD}])
    cfg = {"sets": {"d_num": {"path": str(tmp_path / "d_num.jsonl"), "grader": "numeric", "samples": 2}},
           "general": {}}
    run = tmp_path / "eval_results" / "m"
    write_json(run / "results.json", {"model_name": "m", "max_tokens": 32768,
                                      "sets": {"d_num": {"n_samples": 2, "score": 75.0}}})
    write_jsonl(run / "d_num" / "generations.jsonl", [
        {"item_id": "a", "spec_key": "old", "samples": [_sample(WRONG, 10)] * 2},                    # another spec
        {"item_id": "b", "spec_key": "old", "samples": [_sample(WRONG, 10)] * 2},
        {"item_id": "a", "spec_key": "k", "samples": [_sample(WRONG, 10)] * 2},                      # replaced below
        {"item_id": "a", "spec_key": "k", "samples": [_sample(RIGHT, 3000), _sample(RIGHT, 9000)]},
        {"item_id": "b", "spec_key": "k", "samples": [_sample(WRONG, 1500), _sample(RIGHT, 32771, "length")]},
    ])
    return cfg, run


def test_correct_only_within_the_budget(tmp_path):
    cfg, run = _run(tmp_path)
    c = analyze_run(run, cfg, ["d_num", "d_code"], verify=False)["sets"]["d_num"]
    assert c["n_samples"] == 4 and c["n_correct"] == 3 and c["tokens_total"] == 3000 + 9000 + 1500 + 32771
    within = dict(zip(c["budgets"], c["correct_within"]))
    assert within[2048] == 0 and within[4096] == 1 and within[8192] == 1 and within[12288] == 2
    assert within[28672] == 2 and within[32768] == 3          # the run's own cap cuts nothing more
    assert c["tokens_correct_p50"] == 9000 and c["tokens_wrong_p50"] == 1500


def test_cli_writes_one_json_per_run_and_skips_fresh_ones(tmp_path, monkeypatch):
    cfg, run = _run(tmp_path)
    monkeypatch.setattr("dpt.common.config.load_config", lambda *a, **k: {"eval_suite": cfg})
    monkeypatch.setattr("dpt.common.config.to_container", lambda c: c)
    out = tmp_path / "budget"
    assert main(["--eval-root", str(run.parent), "--out", str(out)]) == 0
    dst = out / "m.json"
    assert dst.exists()
    before = dst.stat().st_mtime_ns
    assert main(["--eval-root", str(run.parent), "--out", str(out)]) == 0
    assert dst.stat().st_mtime_ns == before


# ---- three budgets and the paired comparison with SFT (2026-11-11) -----------------------------------------------
CUT = "t = 120 / 6, and then I need to check the units of the distance again and"
FINISH = " so 20 s.\n</think>\n\n\\boxed{20 s}"


def _three_budget_cfg(tmp_path, ids):
    write_jsonl(tmp_path / "d_fresh.jsonl", [{"id": i, "question": "q", "answer": GOLD} for i in ids])
    return {"sets": {"d_fresh": {"path": str(tmp_path / "d_fresh.jsonl"), "grader": "numeric", "samples": 2,
                                 "grader_kwargs": {"missing_unit": "strict"}}}, "general": {}}


def _row(root, row, gens):
    write_json(root / row / "results.json", {"model_name": row, "max_tokens": 32768,
                                             "sets": {"d_fresh": {"n_samples": 2}}})
    write_jsonl(root / row / "d_fresh" / "generations.jsonl",
                [{"item_id": i, "spec_key": "k", "samples": s} for i, s in gens])


def test_one_32k_run_gives_three_budgets(tmp_path):
    """Item a: a short right answer and one cut at 32K that its continuation finishes right. Item b: a right answer
    of 20K tokens, which a 16K cap would have cut, and a wrong value."""
    from dpt.eval.budget import _set_specs, continuations_path, read_continuations, row_samples, three_budget_scores

    cfg = _three_budget_cfg(tmp_path, ["a", "b"])
    root = tmp_path / "eval_results"
    _row(root, "sft", [("a", [_sample(RIGHT, 900), _sample(CUT, 32768, "length")]),
                       ("b", [_sample(RIGHT, 20000), _sample(WRONG, 800)])])
    write_jsonl(continuations_path(root, "d_fresh", "sft"), [{"key": "a|1", "text": FINISH, "n_tokens": 30}])
    spec = _set_specs(cfg, ["d_fresh"])[0]
    items, gens = row_samples(spec, root / "sft", verify=False)
    sc = three_budget_scores(spec, items, gens, read_continuations(continuations_path(root, "d_fresh", "sft")))
    assert sc["per_item"][16384] == {"a": 50.0, "b": 0.0}
    assert sc["per_item"][32768] == {"a": 50.0, "b": 50.0}
    assert sc["per_item"][65536] == {"a": 100.0, "b": 50.0}
    assert sc["counts"][32768] == {"right": 2, "unfinished": 1, "wrong_value": 1}
    assert sc["counts"][16384]["unfinished"] == 2 and sc["missing_continuations"] == 0
    assert three_budget_scores(spec, items, gens, {})["missing_continuations"] == 1


def test_paired_comparison_with_sft_at_each_budget(tmp_path):
    from dpt.eval.budget import compare, continuations_path, paired_interval

    assert paired_interval({"a": 60.0, "b": 70.0}, {"a": 50.0, "b": 60.0}) == {
        "items": 2, "diff": 10.0, "low": 10.0, "high": 10.0}
    ids = ["a", "b", "c", "d"]
    cfg = _three_budget_cfg(tmp_path, ids)
    root = tmp_path / "eval_results"
    _row(root, "sft", [(i, [_sample(RIGHT, 900), _sample(CUT, 32768, "length")]) for i in ids])
    _row(root, "rl_v2_seed1", [(i, [_sample(RIGHT, 900), _sample(RIGHT if i != "d" else WRONG, 900)]) for i in ids])
    write_jsonl(continuations_path(root, "d_fresh", "sft"), [{"key": f"{i}|1", "text": FINISH} for i in ids])
    res = compare(cfg, root, ["sft", "rl_v2_seed1"], "sft", ["d_fresh"], verify=False)
    rl, sft = res["sets"]["d_fresh"]["rl_v2_seed1"], res["sets"]["d_fresh"]["sft"]
    assert rl["at_32k"]["score"] == 87.5 and rl["at_32k"]["vs_baseline"]["diff"] == 37.5
    assert rl["at_32k"]["vs_baseline"]["above_noise"] is True          # SFT left half its answers unfinished
    assert rl["at_64k"]["vs_baseline"]["diff"] == -12.5 and not rl["at_64k"]["vs_baseline"]["above_noise"]
    assert sft["at_64k"]["score"] == 100.0 and "vs_baseline" not in sft["at_32k"]
    assert sft["at_32k"]["outcomes_pct"] == {"right": 50.0, "wrong_value": 0.0, "unfinished": 50.0, "format": 0.0}
