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
