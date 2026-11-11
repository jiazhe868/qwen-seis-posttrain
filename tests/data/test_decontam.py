"""Decontamination layers, canaries and memorization probes (report §污染检查)."""

from __future__ import annotations

import math

import pytest

from dpt.data.backends import FakeCompleter, FakeEmbedder, FakeParaphraseJudge
from dpt.data.decontam import (
    DecontamConfig,
    Decontaminator,
    RefItem,
    contamination_report,
    make_canary,
    perturbation_gap,
    prefix_completion_probe,
    scan_canaries,
)
from dpt.data.schema import Problem, SFTSample

EVAL_Q1 = ("A P wave travels at 6.0 km/s through the crust. How long does it take to travel from the hypocenter "
           "to a station 120 km away along a straight path? Give the answer in seconds.")
EVAL_Q2 = "An earthquake has a seismic moment of 3.5e19 N·m. What is its moment magnitude Mw?"
EVAL_Q3 = "The S-P time at a station is 12 s. With vp = 6.2 km/s and vs = 3.6 km/s, how far is the earthquake?"
SOLUTION = ("The travel time equals the path length divided by the wave speed, so t = d / v = 120 km / 6.0 km/s = "
            "20 s, which is the time the P wave needs to reach the station.")
REFS = [
    RefItem("d_num/1", EVAL_Q1, {"value": 20.0}, SOLUTION),
    RefItem("d_num/2", EVAL_Q2, {"value": 6.963}),
    RefItem("d_num/3", EVAL_Q3, {"value": 103.0}, params=[12, 6.2, 3.6]),
]


def _dec(**kw):
    return Decontaminator(REFS, DecontamConfig(**kw), embedder=FakeEmbedder(), judge=FakeParaphraseJudge())


def _p(q, value=1.0, **meta):
    return Problem.from_dict({"task": "t1_num", "question": q, "answer": {"value": value, "unit": ""}, "meta": meta})


def test_verbatim_copy_hits_ngram_layer():
    res = _dec().check_problems([_p(EVAL_Q1, 20.0)])
    assert res.removed[0]["layer"] == "ngram_minhash" and res.removed[0]["match"] == "d_num/1"


def test_light_edit_breaking_all_13grams_is_caught_by_minhash():
    edited = EVAL_Q2.replace("3.5e19", "4.1e19")  # every 13-gram spans the number
    res = _dec().check_problems([_p(edited)])
    assert res.removed and res.removed[0]["layer"] == "ngram_minhash"


def test_reworded_with_new_numbers_hits_semantic_layer():
    para = "What is the moment magnitude Mw of an earthquake whose seismic moment is 7.9e18 N·m?"
    res = _dec().check_problems([_p(para)])
    assert res.removed and res.removed[0]["layer"] == "semantic" and res.removed[0]["match"] == "d_num/2"


def test_judge_can_veto_semantic_neighbours():
    class Never:
        def same_problem(self, a, b):
            return False

    para = "What is the moment magnitude Mw of an earthquake whose seismic moment is 7.9e18 N·m?"
    dec = Decontaminator(REFS, DecontamConfig(), embedder=FakeEmbedder(), judge=Never())
    res = dec.check_problems([_p(para)])
    assert not res.removed and res.semantic_judge_rejected >= 1


def test_judge_calls_run_concurrently_and_keep_order():
    import threading

    barrier = threading.Barrier(2, timeout=10)    # sequential calls would time out here

    class Concurrent:
        def same_problem(self, a, b):
            barrier.wait()
            return "7.9e18" in a

    first = "What is the moment magnitude Mw of an earthquake whose seismic moment is 7.9e18 N·m?"
    second = "What is the moment magnitude Mw of an earthquake whose seismic moment is 2.2e17 N·m?"
    dec = Decontaminator(REFS, DecontamConfig(judge_concurrency=4), embedder=FakeEmbedder(), judge=Concurrent())
    assert all(dec.semantic_matches([first, second])), "both texts need an embedding match for this test"
    a, b = _p(first), _p(second)
    res = dec.check_problems([a, b])
    assert [r["uid"] for r in res.removed] == [a.uid] and res.kept == [b]
    assert res.semantic_judge_rejected >= 1


def test_boilerplate_shared_by_many_candidates_is_not_a_leak():
    formula = "using the moment magnitude relation Mw = (2/3) * (log10 M0 - 9.1) with M0 in newton metres"
    ref = RefItem("d_num/9", ("A strike-slip fault near the coast ruptures over a length of 40 km and a depth of 15 km "
                              f"with an average slip of 2 m and a rigidity of 30 GPa. Compute Mw {formula}."),
                  {"value": 7.0})
    dec = Decontaminator([ref], DecontamConfig(ngram_max_df=10), embedder=None, judge=None)
    template = [_p(f"Station {a} records a body-wave amplitude of {a + 1} micrometres at a period of {a % 3 + 1} s, "
                   f"and the analyst first converts the event's seismic moment of {a + 2}e17 N m before reporting "
                   f"Mw {formula}.") for a in range(12)]
    leaked = _p(ref.question)                       # the eval item itself, once
    res = dec.check_problems(template + [leaked])
    assert [r["uid"] for r in res.removed] == [leaked.uid] and res.boilerplate_ngrams > 0
    strict = Decontaminator([ref], DecontamConfig(ngram_max_df=0), embedder=None, judge=None)
    assert len(strict.check_problems(template).removed) == len(template)   # 0 = count every n-gram


def test_answer_level_match_survives_complete_rewording():
    q = ("Station KQ7 logs the shear arrival twelve seconds behind the compressional one. Speeds: "
         "compressional 6.2 km/s, shear 3.6 km/s. Estimate the source range.")
    p = _p(q, 103.0, generator={"name": "sp_distance", "params": {"t": 12, "vp": 6.2, "vs": 3.6}})
    res = _dec().check_problems([p])
    assert res.removed and res.removed[0]["layer"] == "answer_level"
    # same parameters but a different answer is not contamination
    assert not _dec().check_problems([_p(q, 55.0, generator={"name": "x", "params": {"t": 12, "vp": 6.2, "vs": 3.6}})]).removed


def test_unrelated_problem_is_kept_and_output_has_no_eval_text():
    res = _dec().check_problems([_p("Compute the wavelength of a 15 Hz wave at 3000 m/s.", 200.0)])
    assert len(res.kept) == 1
    res2 = _dec().check_problems([_p(EVAL_Q1, 20.0)])
    assert all(EVAL_Q1 not in str(v) for v in res2.removed[0].values())


def test_teacher_recitation_flags_sft_sample_with_different_prompt():
    s = SFTSample.from_dict({"subset": "domain_reasoning", "source": {"origin": "teacher"}, "messages": [
        {"role": "user", "content": "Explain how to turn a distance and a speed into a time."},
        {"role": "assistant", "reasoning_content": "Example: " + SOLUTION, "content": "Divide."}]})
    res = _dec().check_sft([s])
    assert res.removed[0]["layer"] == "teacher_verbatim" and res.removed[0]["source"] == {"origin": "teacher"}


def test_canary_scan_finds_nested_strings():
    c = make_canary("d_num/v1")
    assert c == make_canary("d_num/v1") and c.startswith("DPT-CANARY-")
    hits = scan_canaries([("a", {"x": ["ok", {"y": f"..{c}.."}]}), ("b", "clean")], [c])
    assert hits == [{"key": "a", "canary": c}]
    assert scan_canaries([("a", c)], []) == []


def test_prefix_probe_detects_memorized_benchmark():
    import random

    rng = random.Random(0)
    vocab = [f"term{i}" for i in range(300)]
    texts = [" ".join(rng.choice(vocab) for _ in range(40)) for _ in range(20)]  # diverse continuations
    mem = prefix_completion_probe(texts, FakeCompleter(memorized=texts))
    clean = prefix_completion_probe(texts, FakeCompleter(memorized=[]))
    assert mem["match_rate"] == 1.0 and mem["baseline_rate"] < 0.2 and mem["significant"]
    assert clean["match_rate"] == 0.0 and not clean["significant"]


def test_perturbation_gap_reports_paired_ci():
    out = perturbation_gap([1, 1, 1, 0] * 25, [1, 1, 0, 0] * 25)
    assert out["gap"] == pytest.approx(25.0) and out["ci95"][0] <= 25.0 <= out["ci95"][1]
    with pytest.raises(ValueError):
        perturbation_gap([1], [1, 0])


def test_report_shape():
    res = _dec().check_problems([_p(EVAL_Q1, 20.0), _p("Something else entirely about magnitudes.", 2.0)])
    rep = contamination_report(rl=res, canary_hits=[], canary_scanned=2)
    assert rep["rl"]["total_removed"] == 1 and math.isclose(rep["rl"]["rate"], 0.5)
    assert rep["canary"]["hits"] == 0
