"""General benchmarks and the refusal set: every imported item grades right with its own grader.

The fixtures (tests/eval/fixtures/general_raw/) are in the normalized format ``fetch`` writes: real rows where
they are small (IFEval, BFCL, JBB, XSTest, MATH-500), synthetic rows otherwise.
"""

from __future__ import annotations

import base64
import csv
import datetime
import json
import pickle
import sys
import zlib
from pathlib import Path

import pytest

from dpt.common.io import read_jsonl
from dpt.eval.datasets import get_grader, load_eval_items
from dpt.eval.import_general import (
    build,
    build_gpqa,
    build_livecodebench,
    build_mmlu_pro,
    decode_lcb_tests,
    fetch,
    lcb_set_name,
    main,
)

RAW = Path(__file__).parent / "fixtures" / "general_raw"
LCB = lcb_set_name("2026-08-01")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("secure")
    dec = tmp_path_factory.mktemp("public_benchmarks")
    summary = build(RAW, root, dec)
    return root, dec, summary


def _items(root, rel, name):
    return load_eval_items(root / rel, name)                       # verifies the manifest


# --------------------------------------------------------------------------- choice


@pytest.mark.parametrize("rel, name, n", [("general/gpqa_diamond.jsonl", "gpqa_diamond", 3),
                                          ("general/mmlu_pro_2k.jsonl", "mmlu_pro_2k", 12)])
def test_choice_items_grade_on_their_letter_only(built, rel, name, n):
    root, _, _ = built
    items = _items(root, rel, name)
    grade = get_grader("choice")
    assert len(items) == n
    for it in items:
        k, letter = it.answer["n_options"], it.answer["letter"]
        assert f"{letter}) " in it.question and "\\boxed{X}" in it.question
        assert grade(f"The answer is \\boxed{{{letter}}}.", it.answer).correct
        wrong = "ABCDEFGHIJ"[("ABCDEFGHIJ".index(letter) + 1) % k]
        assert not grade(f"\\boxed{{{wrong}}}", it.answer).correct


def test_gpqa_shuffle_is_deterministic_per_item_and_drops_incomplete_rows():
    rows = list(read_jsonl(RAW / "gpqa_diamond.jsonl"))
    a, b = build_gpqa(rows, seed=1), build_gpqa(rows, seed=1)
    assert a.records == b.records and a.drops == {"missing_field": 1}
    for r, src in zip(a.records, rows):
        letter = r["answer"]["letter"]
        assert f"{letter}) {src['Correct Answer']}" in r["question"]
    orders = {seed: [r["question"] for r in build_gpqa(rows, seed=seed).records] for seed in range(1, 6)}
    assert len({tuple(v) for v in orders.values()}) > 1                 # the seed changes the option order


def test_mmlu_pro_subset_is_stratified_and_deterministic():
    rows = list(read_jsonl(RAW / "mmlu_pro_test.jsonl"))
    b = build_mmlu_pro(rows, n=6, seed=1)
    cats = [r["meta"]["category"] for r in b.records]
    # 6 of 12 valid rows: quotas physics 3, law 1.5, biology 1.5 → the tie on the remainder goes by name
    assert {c: cats.count(c) for c in set(cats)} == {"physics": 3, "biology": 2, "law": 1}
    assert b.drops["answer_mismatch"] == 1
    assert build_mmlu_pro(list(reversed(rows)), n=6, seed=1).records == b.records    # order-free


# --------------------------------------------------------------------------- ifeval

# Hand-written responses per fixture prompt: (one that satisfies every instruction, one that breaks one).
_WORDS_30 = ("Seismology is the *study* of earthquakes and the *propagation* of elastic waves through the Earth "
             "and it uses instruments to record ground motion so that scientists can locate events and image the "
             "deep interior of our planet.")
IFEVAL_RESPONSES = {
    1001: ("Thou shalt sail to Kyoto and then to Nara at dawn.", "Day one, thou shalt sail."),
    1019: ("is the boy holding a toy gun?", "Is the boy holding a toy gun?"),
    1322: ("<<kotlin vs java>>\nKotlin is concise.", "Kotlin vs Java\nKotlin is concise."),
    122: ('"The internet began as a research network."', "The internet began as a research network."),
    2001: ("A seismometer records ground motion. Is there anything else?", "A seismometer records ground motion."),
    2002: ("Faults have *strike* and *dip*.", "Faults have strike and dip."),
    2003: ("* one\n* two\n* three", "* one\n* two"),
    2004: ("SECTION 1\nP waves are fast.\nSECTION 2\nThey are compressional.", "SECTION 1\nP waves are fast."),
    2005: ("Dear [name], see you at [place].", "Dear friend, see you there."),
    2006: ("Magnitude measures size.\nP.S. Ask me more.", "Magnitude measures size."),
    2007: ("Faults have strike and dip.", "Faults have strike only."),
    2008: ("Earthquakes release stress.", "Earthquakes make the ground shake."),
    2009: ("A wave is a wave and a wave.", "A wave is a signal."),
    2010: ("The Moho is the crust-mantle boundary.", " ".join(["word"] * 45)),
    2011: ("Plates move. They collide. They separate.", "Plates move."),
    2012: ("Volcanoes erupt.\n\nThey build mountains.", "Volcanoes erupt."),
    2013: ('{"richter": "a magnitude scale"}', "The Richter scale is a magnitude scale."),
    2014: ("FAULTS ARE FRACTURES.", "Faults are fractures."),
    2015: (_WORDS_30, _WORDS_30.replace(" and it uses", ", and it uses")),
}


def test_ifeval_items_grade_on_their_instructions_and_unsupported_ones_are_dropped(built):
    root, _, summary = built
    items = _items(root, "general/ifeval.jsonl", "ifeval")
    grade = get_grader("ifeval")
    assert {it.id for it in items} == {f"ifeval-{k}" for k in IFEVAL_RESPONSES}
    for it in items:
        good, bad = IFEVAL_RESPONSES[int(it.id.split("-")[1])]
        assert grade(good, it.answer).correct, (it.id, grade(good, it.answer))
        assert not grade(bad, it.answer).correct, it.id
    drops = summary["sets"]["ifeval"]["drops"]
    assert drops == {"unsupported:language:response_language": 1, "unsupported:keywords:letter_frequency": 1}


# --------------------------------------------------------------------------- code

SOLUTIONS = {
    "lcb-9001": ("class Solution:\n    def addTwo(self, a: int, b: int) -> int:\n        return a + b\n",
                 "class Solution:\n    def addTwo(self, a: int, b: int) -> int:\n        return a - b\n"),
    "lcb-9002": ("from typing import List\n\n\nclass Solution:\n    def mean(self, nums: List[int]) -> float:\n"
                 "        return sum(nums) / len(nums)\n",
                 "from typing import List\n\n\nclass Solution:\n    def mean(self, nums: List[int]) -> float:\n"
                 "        return 0.0\n"),
}


def test_livecodebench_keeps_post_cutoff_function_problems_and_grades_them_both_ways(built):
    from dpt.rewards.code_sandbox import HiddenTestStore, LocalSubprocessSandbox, SandboxRequest

    root, _, summary = built
    items = _items(root, f"general/{LCB}.jsonl", LCB)
    assert {it.id for it in items} == set(SOLUTIONS)
    assert summary["sets"][LCB]["drops"] == {"stdin_unsupported": 1, "before_cutoff": 1}
    local = get_grader("code")                                       # dev runner: tests_source
    sandbox = LocalSubprocessSandbox(hidden_store=HiddenTestStore(root / "hidden_tests"), python=sys.executable,
                                     cpu_s=20)
    for it in items:
        good, bad = SOLUTIONS[it.id]
        assert "```python" in it.question and it.answer["entry_point"] in it.question
        assert local(f"```python\n{good}```", it.answer).correct
        assert not local(f"```python\n{bad}```", it.answer).correct
        # production path on the cluster: the sandbox server runs <secure>/hidden_tests/<id>.py
        for code, want in ((good, True), (bad, False)):
            res = sandbox.run(SandboxRequest(code=code, hidden_tests_id=it.answer["hidden_tests_id"], timeout_s=30))
            assert res.status == "ok" and (res.passed == res.total > 0) is want, (it.id, res)


def test_livecodebench_with_no_problems_after_the_cutoff_writes_nothing(tmp_path):
    rows = list(read_jsonl(RAW / "livecodebench.jsonl"))
    assert build_livecodebench(rows, cutoff="2030-01-01").records == []
    rc = main(["build", "--raw-dir", str(RAW), "--out-root", str(tmp_path / "s"), "--decontam-out", str(tmp_path / "d"),
               "--only", "livecodebench", "--lcb-cutoff", "2030-01-01"])
    assert rc == 1 and not (tmp_path / "s" / "general" / f"{lcb_set_name('2030-01-01')}.jsonl").exists()


def test_lcb_private_tests_decode_as_data_and_never_run_a_pickled_object():
    cases = [{"input": "1\n2", "output": "3", "testtype": "functional"}]
    blob = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(cases)))).decode()
    assert decode_lcb_tests(blob) == cases
    evil = base64.b64encode(zlib.compress(pickle.dumps(datetime.date(2020, 1, 1)))).decode()   # needs a global
    with pytest.raises(pickle.UnpicklingError, match="refusing"):
        decode_lcb_tests(evil)


# --------------------------------------------------------------------------- tool calling and refusal


def _xml(calls):
    out = []
    for c in calls:
        params = "".join(f"<parameter={k}>\n{json.dumps(v)}\n</parameter>\n" for k, v in c["arguments"].items())
        out.append(f"<tool_call>\n<function={c['name']}>\n{params}</function>\n</tool_call>")
    return "\n".join(out)


def test_tool_calls_grade_exactly_and_ambiguous_ground_truth_is_dropped(built):
    root, _, summary = built
    items = _items(root, "general/tool_calling.jsonl", "tool_calling")
    grade = get_grader("toolcall")
    assert {it.id for it in items} == {"bfcl-simple_1", "bfcl-multiple_1"}
    assert summary["sets"]["tool_calling"]["drops"] == {"several_acceptable_values": 1}
    for it in items:
        calls = it.answer["calls"]
        assert "<tool_call>" in it.question and calls[0]["name"] in it.question
        assert grade(_xml(calls), it.answer).correct
        arg = next(iter(calls[0]["arguments"]))
        wrong = [{**calls[0], "arguments": {**calls[0]["arguments"], arg: 999}}]
        assert not grade(_xml(wrong), it.answer).correct


def test_refusal_set_labels_harmful_and_benign_prompts(built):
    root, _, _ = built
    items = _items(root, "safety/refusal_regression.jsonl", "refusal")
    grade = get_grader("refusal")
    labels = {it.id: it.answer["should_refuse"] for it in items}
    assert sum(labels.values()) == 4 and len(labels) == 8               # 2 JBB + 2 XSTest each way
    refuse, comply = "I'm sorry, but I can't help with that.", "Sure. Here is an overview of the topic."
    for it in items:
        want = it.answer["should_refuse"]
        assert grade(refuse if want else comply, it.answer).correct
        assert not grade(comply if want else refuse, it.answer).correct


# --------------------------------------------------------------------------- outputs


def test_decontam_files_and_no_overwrite(built, tmp_path):
    root, dec, summary = built
    names = {p.stem for p in dec.glob("*.jsonl")}
    assert names == {"gpqa_diamond", "mmlu_pro_2k", "ifeval", LCB, "tool_calling", "refusal_regression",
                     "gsm8k_test", "math_test", "math500"}
    for p in dec.glob("*.jsonl"):
        rows = list(read_jsonl(p))
        assert rows and all(set(r) == {"uid", "question"} for r in rows)
    m = json.loads((root / "general" / "ifeval.jsonl.manifest.json").read_text())
    assert "google/IFEval" not in m["notes"] and "fixture/ifeval@fixture-rev" in m["notes"]   # from fetch_manifest
    again = build(RAW, root, dec, only=["ifeval"])
    assert again["skipped_existing"] == [str(root / "general" / "ifeval.jsonl")]
    assert not summary["missing"]


def test_fetch_normalizes_sources_and_reports_a_gated_dataset(tmp_path, monkeypatch):
    from dpt.eval import import_general as ig

    src_dir = tmp_path / "src"
    (src_dir / "data").mkdir(parents=True)
    for name, harmful in (("harmful-behaviors.csv", True), ("benign-behaviors.csv", False)):
        rows = [r for r in read_jsonl(RAW / "jbb.jsonl") if r["harmful"] is harmful]
        with open(src_dir / "data" / name, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "harmful"])
            wr.writeheader()
            wr.writerows({k: v for k, v in r.items() if k != "harmful"} for r in rows)
    rows = list(read_jsonl(RAW / "xstest.jsonl"))
    with open(src_dir / "xstest_prompts.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    def fake_download(src, filename, revision=None):
        if src.gated:
            raise ig.GatedError(f"{src.repo} is gated or not accessible: ... `hf auth login` ...")
        return src_dir / filename

    monkeypatch.setattr(ig, "_download", fake_download)
    res = fetch(tmp_path / "raw", ["refusal", "gpqa_diamond"])
    assert set(res["failures"]) == {"gpqa_diamond"} and "hf auth login" in res["failures"]["gpqa_diamond"]
    man = json.loads((tmp_path / "raw" / "fetch_manifest.json").read_text())
    assert man["jbb"]["revision"] == ig.SOURCES["jbb"].revision and man["jbb"]["rows"] == 4
    assert len(man["xstest"]["files"]["xstest_prompts.csv"]) == 64            # sha256 of the source file
    out = build(tmp_path / "raw", tmp_path / "secure", tmp_path / "dec", only=["refusal"])
    assert out["sets"]["refusal_regression"]["n"] == 8
    assert main(["fetch", "--raw-dir", str(tmp_path / "raw2"), "--only", "gpqa_diamond"]) == 1
