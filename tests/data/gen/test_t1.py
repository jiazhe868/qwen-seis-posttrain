"""T1 seismology families and D-Num: every problem states its inputs, has one parseable answer, and the
held-out families never reach training."""

from __future__ import annotations

import csv
import json
import math

import pytest

from dpt.common.io import read_jsonl, write_jsonl
from dpt.data.answer_verify import perturbation_consistency
from dpt.data.gen.common import rng_for
from dpt.data.gen.families import FAMILIES, Family, Param, by_split, heldout_families_are_present
from dpt.data.gen.t1 import build_dnum_draft, build_train, finalize_dnum, main
from dpt.data.schema import Problem

ALL = sorted(FAMILIES)

# The held-out families behind D-Num are not published (README, Data & licensing). Tests that build D-Num from
# them run only where their private module is present; the rest of the D-Num code is covered either way.
requires_private_heldout_families = pytest.mark.skipif(
    not heldout_families_are_present(), reason="the held-out eval families are not published (README, Data & licensing)")
# A local held-out family (not registered: the registry is global), so the train/held-out guard is always tested.
TOY_HELDOUT = Family("toy_heldout_travel_time", "travel times", lambda p: p["distance_km"] / p["velocity_km_s"],
                     (Param("distance_km", 20, 400), Param("velocity_km_s", 3.0, 8.5, digits=2)), "s",
                     ("A wave travels {distance_km} km at {velocity_km_s} km/s. How many seconds does it take?",),
                     split="heldout")


def test_registry_has_training_families_disjoint_from_heldout():
    train, held = by_split("train"), by_split("heldout")
    assert len(train) >= 45
    assert sum(f.steps >= 2 for f in train) >= 15                      # multi-step problems
    assert not {f.name for f in train} & {f.name for f in held}


@requires_private_heldout_families
def test_registry_has_the_ten_heldout_families():
    assert len(by_split("heldout")) >= 10


@pytest.mark.parametrize("name", ALL)
def test_every_question_states_its_inputs_and_has_a_finite_answer(name):
    f = FAMILIES[name]
    rng = rng_for(3, "test", name)
    for k in range(len(f.templates) * 3):
        p = f.sample(rng)
        q = f.render(p, k)
        assert "{" not in q and "}" not in q
        for prm in f.params:                                         # every input is in the question
            assert prm.show(p[prm.name]) in q, (name, prm.name, q)
        v = f.solve(p)
        assert math.isfinite(v) and (f.unit or f.abs_tol or v != 0)
        Problem.from_dict({"task": "t1_num", "question": q, "answer": f.answer(p)})


@pytest.mark.parametrize("name", ALL)
def test_answers_parse_in_both_graders(name):
    """The label's unit must be readable by the training grader and by the independent eval grader."""
    from dpt.eval.graders.numeric import grade_numeric
    from dpt.rewards.router import compute_score

    f = FAMILIES[name]
    p = f.sample(rng_for(5, "units", name))
    gold = f.answer(p)
    text = f"\\boxed{{{gold['value']:.6g} {f.unit}}}".replace(" }", "}")
    assert grade_numeric(text, gold, missing_unit="strict").correct, (name, text)
    out = compute_score("t1_num", "</think>" + text, json.dumps(gold), {"uid": "x", "task": "t1_num"})
    assert out["score"] == 1.0, (name, text, out)


@pytest.mark.parametrize("name", ALL)
def test_solvers_respond_to_their_inputs_as_the_physics_says(name):
    """The verify stage's perturbation check: declared power laws hold, every input matters, no erratic slopes."""
    f = FAMILIES[name]
    rng = rng_for(7, "pert", name)
    for _ in range(5):
        p = f.sample(rng)
        res = perturbation_consistency(f.generator_meta(p))
        assert res.ok, (name, p, res.issues)


def test_build_train_is_deterministic_distinct_and_excludes_heldout():
    train = by_split("train")[:5]
    a, b = build_train(train, 30, seed=1), build_train(train, 30, seed=1)
    assert a == b and len({r["question"] for r in a}) == len(a)
    assert all(r["meta"]["split"] == "train" and r["source"]["license"] == "internal-generated" for r in a)
    assert all(r["meta"]["generator"]["name"] == r["meta"]["family"] for r in a)
    assert build_train(train, 30, seed=2) != a
    with pytest.raises(ValueError):                                     # a held-out family never trains
        build_train([TOY_HELDOUT], 5, seed=1)


def _flags(path, draft, bad=()):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "flag"])
        w.writeheader()
        w.writerows({"id": r["id"], "flag": "check" if r["id"] in bad else "ok"} for r in draft)


@requires_private_heldout_families
def test_dnum_draft_is_balanced_and_finalize_drops_flagged_items(tmp_path):
    held = by_split("heldout")
    draft = build_dnum_draft(held, 60, seed=1)
    fams = [r["meta"]["family"] for r in draft]
    assert len(draft) == 60 and max(fams.count(x) for x in set(fams)) - min(fams.count(x) for x in set(fams)) <= 1
    bad = {draft[0]["id"], draft[1]["id"]}
    _flags(tmp_path / "w.csv", draft, bad)
    from dpt.data.gen.t1 import read_wording

    items, twins, stats = finalize_dnum(draft, read_wording(tmp_path / "w.csv"), 50, 20, seed=1)
    assert len(items) == 50 and not bad & {r["id"] for r in items}
    assert all(r["canary"].startswith("DPT-CANARY-") for r in items + twins)
    assert stats["flags"] == {"ok": 58, "check": 2}
    for t, o in zip(twins, items):
        assert t["meta"]["twin_of"] == o["id"] and t["meta"]["family"] == o["meta"]["family"]
        assert t["question"] != o["question"]
        f = FAMILIES[t["meta"]["family"]]
        assert t["answer"]["value"] == pytest.approx(f.solve(t["meta"]["params"]))


@requires_private_heldout_families
def test_cli_writes_candidates_and_dnum_with_manifests(tmp_path):
    out, draft = tmp_path / "t1.jsonl", tmp_path / "secure" / "d_num_draft.jsonl"
    assert main(["build", "--out", str(out), "--heldout-out", str(draft), "--per-family", "4",
                 "--n-heldout", "40"]) == 0
    rows = list(read_jsonl(out))
    assert len({r["meta"]["family"] for r in rows}) == len(by_split("train"))
    assert {r["meta"]["family"] for r in rows}.isdisjoint({f.name for f in by_split("heldout")})
    d = list(read_jsonl(draft))
    _flags(tmp_path / "w.csv", d)
    args = ["finalize-dnum", "--draft", str(draft), "--wording", str(tmp_path / "w.csv"),
            "--out", str(tmp_path / "secure" / "d_num.jsonl"), "--perturbed-out",
            str(tmp_path / "secure" / "d_num_perturbed.jsonl"), "--n-items", "30", "--n-perturbed", "10"]
    assert main(args) == 0
    from dpt.eval.datasets import load_eval_items

    items = load_eval_items(tmp_path / "secure" / "d_num.jsonl", "d_num")           # manifest verified
    assert len(items) == 30 and items[0].task == "t1_num"
    assert len(load_eval_items(tmp_path / "secure" / "d_num_perturbed.jsonl", "d_num_perturbed")) == 10
    assert main([*args[:-4], "--n-items", "100", "--n-perturbed", "10"]) == 1        # too few: nothing written


@requires_private_heldout_families
def test_finalize_refuses_items_without_a_wording_verdict(tmp_path):
    draft = build_dnum_draft(by_split("heldout"), 12, seed=1)
    write_jsonl(tmp_path / "d.jsonl", draft)
    with pytest.raises(ValueError, match="no wording verdict"):
        finalize_dnum(draft, {draft[0]["id"]: "ok"}, 5, 2, seed=1)


@pytest.mark.parametrize("name", ALL)
def test_many_draws_never_raise(name):
    """Domain errors inside `valid` (b-value denominator 0, asin > 1) reject the draw instead of crashing."""
    f = FAMILIES[name]
    rng = rng_for(11, "many", name)
    for _ in range(300):
        assert math.isfinite(f.solve(f.sample(rng)))
