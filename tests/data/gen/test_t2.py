"""T2 code tasks and D-Code: specs, hidden tests, mutant checks in the sandbox, split isolation, outputs."""

from __future__ import annotations

import ast
import json
import math
import random
import sys
import zlib

import pytest

from dpt.common.io import read_jsonl
from dpt.data.gen import t2
from dpt.data.gen.families import Family, Param
from dpt.data.gen.t2_tasks import KINDS, heldout_code_kinds_present
from dpt.data.schema import Problem
from dpt.rewards.code_sandbox import HiddenTestStore, LocalSubprocessSandbox, count_tests

# Local families (not registered: the registry is global).
TRAVEL = Family("toy_travel_time", "travel times", lambda p: p["distance_km"] / p["velocity_km_s"],
                (Param("distance_km", 20, 400), Param("velocity_km_s", 3.0, 8.5, digits=2)), "s",
                ("A P wave travels {distance_km} km at {velocity_km_s} km/s. How many seconds does it take?",
                 "How long, in s, does a wave at {velocity_km_s} km/s need for {distance_km} km?"))
HELD = Family("toy_tsunami", "tsunami", lambda p: p["distance_km"] / math.sqrt(9.81 * p["depth_m"] / 1e6) / 3600,
              (Param("distance_km", 200, 4000), Param("depth_m", 500, 5000)), "h",
              ("A tsunami crosses {distance_km} km of ocean {depth_m} m deep. How many hours does it take?",),
              split="heldout")


@pytest.fixture(scope="module")
def sandbox():
    return LocalSubprocessSandbox(python=sys.executable, cpu_s=10, mem_mb=1024)


def _tasks(kind_name, n=4, seed=0):
    rng = random.Random(seed)
    k = KINDS[kind_name]
    out = []
    while len(out) < n:
        try:
            out.append(t2.make_task(k.build(rng), split=k.split, source=k.name, kind=k.name, topic=k.topic))
        except (ValueError, ZeroDivisionError, StopIteration):
            continue
    return out


def _imports(src):
    mods = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add(node.module)
    return mods


@pytest.mark.parametrize("name", sorted(KINDS))
def test_every_kind_builds_sound_self_contained_tasks(name):
    for t in _tasks(name, n=3, seed=zlib.crc32(name.encode()) % 1000):
        assert t2.inprocess_check(t) == []                       # the reference passes its own tests
        assert len(t2.effective_mutants(t)) >= t2.MIN_MUTANTS     # the hidden tests can tell wrong code apart
        assert count_tests(t.hidden_tests) >= 5 and count_tests(t.visible_tests) >= 1
        assert _imports(t.hidden_tests) == {"math", "solution"}  # stdlib only, solution imported as `solution`
        assert f"`{t.entry}(" in t.question and "Example" in t.question
        assert _imports(t.reference) <= {"math", "cmath"}        # references are plain Python


def test_sandbox_runs_references_and_kills_mutants(sandbox):
    sampled_kind_names = ("sta_lta", "rotate_components", "b_value", "catalog_select", "newmark_spectral_acceleration",
                          "window_declustering", "great_circle", "clipping")
    # two of them are held-out D-Code kinds, present only where their private module is (README, Data & licensing)
    sample = [t for name in sampled_kind_names if name in KINDS for t in _tasks(name, n=1, seed=7)]
    rng = random.Random(1)
    sample += [t2.make_task(t2.family_variant(TRAVEL, rng), split="train", source="family:x", kind="physics_function",
                            topic="t") for _ in range(2)]
    for t in sample:
        assert t2.check_task(t, sandbox) == [], t.kind

    t = sample[0]
    wrong = t2.Task(**{**t.__dict__, "reference": t.reference.replace("return", "return 0 *", 1)})
    assert any(i.startswith("reference_hidden") for i in t2.check_task(wrong, sandbox))
    loose = t2.Task(**{**t.__dict__, "hidden_tests": t.hidden_tests.replace("rel, tol):", "rel, tol):\n    return True")})
    assert any(i.startswith("mutant_survives") for i in t2.check_task(loose, sandbox))


def test_family_variant_states_units_and_computes_in_them():
    rng = random.Random(3)
    seen = {}
    for _ in range(300):
        v = t2.family_variant(TRAVEL, rng)
        seen.setdefault(("conv" if "distance_m" in v.entry + v.spec else "plain",
                         "ms" if "in ms (not s)" in v.spec else "s"), v)
    assert set(seen) == {("plain", "s"), ("conv", "s"), ("plain", "ms"), ("conv", "ms")}
    v = seen[("conv", "ms")]
    assert "but passed in m" in v.spec and "Problem:" in v.spec
    args = ("distance_m", "velocity_km_s")
    sig = v.spec.split("`")[1]
    order = [a.strip() for a in sig[sig.index("(") + 1:-1].split(",")]
    call = {"distance_m": 150000.0, "velocity_km_s": 6.0}
    assert set(order) == set(args)
    assert v.fn(*[call[a] for a in order]) == pytest.approx(25000.0)     # 150 km at 6 km/s = 25 s = 25000 ms
    t = t2.make_task(v, split="train", source="family:toy", kind="physics_function", topic="t")
    assert t.reference_is_table and t2.inprocess_check(t) == []


def test_splits_never_mix_and_generation_is_deterministic():
    tr, ho = t2.build(60, 20, seed=5, families=[TRAVEL, HELD])
    train_kinds = {k.name for k in KINDS.values() if k.split == "train"}
    assert {t.source for t in tr.tasks} <= train_kinds | {"family:toy_travel_time"}
    assert {t.source for t in ho.tasks} <= {k.name for k in KINDS.values() if k.split == "heldout"} | {
        "family:toy_tsunami"}
    assert any(t.family == "toy_tsunami" for t in ho.tasks) and any(t.family == "toy_travel_time" for t in tr.tasks)
    assert all(t.uid.startswith("t2-") for t in tr.tasks) and all(t.uid.startswith("dcode-") for t in ho.tasks)
    tr2, ho2 = t2.build(60, 20, seed=5, families=[TRAVEL, HELD])
    assert [t.question for t in tr.tasks] == [t.question for t in tr2.tasks]
    assert [t.uid for t in ho.tasks] == [t.uid for t in ho2.tasks]
    tr3, _ = t2.build(60, 20, seed=6, families=[TRAVEL, HELD])
    assert [t.question for t in tr3.tasks] != [t.question for t in tr.tasks]


def test_cap_per_key_limits_near_copies():
    src = t2.library_sources("train", [KINDS["zero_crossings"]])
    g = t2.generate(src, 100, seed=1, cap_per_key=2)
    keys = [t.key for t in g.tasks]
    assert len(keys) == 2 * len(set(keys)) and g.exhausted == ["zero_crossings"]


def test_comparator_accepts_equivalent_types_and_rejects_wrong_ones():
    same = t2._compiled("_same", t2.COMPARE_SRC, "_same")
    np = pytest.importorskip("numpy")
    assert same(np.array([1.0, 2.0]), [1.0, 2.0000001], 1e-6, 0.0)
    assert same((1.0, 2.0), [1.0, 2.0], 1e-6, 0.0) and same(np.int64(3), 3, 0.0, 0.0)
    assert not same(3.5, 3, 0.0, 0.0) and not same(None, 0.0, 1e-6, 1.0) and not same(0, None, 0.0, 0.0)
    assert not same([1.0], [1.0, 2.0], 1e-6, 0.0) and not same("1.0", 1.0, 1e-6, 0.0)
    assert not same(float("nan"), 1.0, 1e-6, 1e9)


@pytest.mark.skipif(not heldout_code_kinds_present,
                    reason="the held-out D-Code kinds are not published (README, Data & licensing)")
def test_cli_writes_candidates_hidden_tests_and_a_manifested_dcode(tmp_path):
    from dpt.eval.datasets import load_eval_items

    out, ev = tmp_path / "gen", tmp_path / "secure"
    args = ["build", "--out-dir", str(out), "--eval-out-dir", str(ev), "--n-train", "40", "--n-heldout", "12"]
    assert t2.main(args) == 0
    rows = [Problem.from_dict(r) for r in read_jsonl(out / "candidates_t2.jsonl")]
    assert len(rows) == 40 and all(p.task == "t2_code" and p.meta["split"] == "train" for p in rows)
    store = HiddenTestStore(out / "hidden_tests")
    assert all(count_tests(store.get(p.answer["hidden_tests_id"])) >= 5 for p in rows)
    items = load_eval_items(ev / "d_code.jsonl", "d_code")            # manifest verified
    assert len(items) == 12 and all(it.meta["canary"].startswith("DPT-CANARY-") for it in items)
    assert all((ev / "hidden_tests" / f"{it.answer['hidden_tests_id']}.py").exists() for it in items)
    assert not (out / "hidden_tests" / f"{items[0].answer['hidden_tests_id']}.py").exists()
    summary = json.loads((out / "t2_summary.json").read_text())
    assert summary["train"]["tasks"] == 40 and summary["heldout"]["tasks"] == 12

    with pytest.raises(SystemExit, match="exists"):                    # a frozen D-Code is never overwritten
        t2.main(args)
    with pytest.raises(SystemExit, match="must not overlap"):
        t2.main(["build", "--out-dir", str(ev / "x"), "--eval-out-dir", str(ev), "--force"])
