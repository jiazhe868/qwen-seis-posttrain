"""Schema validation for RL problems and SFT samples (docs/dev/conventions.md §4)."""

from __future__ import annotations

import pytest

from dpt.data.schema import Problem, SchemaError, SFTSample, problem_uid


def _num(**kw):
    d = {"task": "t1_num", "question": "Travel time for 120 km at 6 km/s?", "answer": {"value": 20.0, "unit": "s"}}
    d.update(kw)
    return d


def test_uid_is_function_of_normalized_question():
    a = Problem.from_dict(_num(question="Travel time for 120 km at 6 km/s?"))
    b = Problem.from_dict(_num(question="  travel TIME for 120 km at 6 km/s ? "))
    assert a.uid == b.uid == problem_uid("Travel time for 120 km at 6 km/s?")
    assert a.uid.startswith("p-") and len(a.uid) == 18


def test_numeric_defaults_filled_so_equal_problems_hash_equal():
    p = Problem.from_dict(_num())
    assert p.answer == {"value": 20.0, "unit": "s", "rel_tol": 1e-2, "abs_tol": 0.0}


@pytest.mark.parametrize("answer", [{"value": float("nan")}, {"value": "20"}, {"value": 1, "rel_tol": -1}])
def test_bad_numeric_answers_rejected(answer):
    with pytest.raises(SchemaError):
        Problem.from_dict(_num(answer=answer))


def test_task_specific_requirements():
    with pytest.raises(SchemaError):
        Problem.from_dict({"task": "t2_code", "question": "write f", "answer": {"entry_point": "f"}})
    with pytest.raises(SchemaError):
        Problem.from_dict({"task": "t3_report", "question": "report", "answer": {"rubric": [], "reference_length": 5}})
    with pytest.raises(SchemaError):
        Problem.from_dict({"task": "t4_general", "question": "q", "answer": {"kind": "essay"}})
    ok = Problem.from_dict({"task": "t4_general", "question": "No commas please.",
                            "answer": {"kind": "ifeval", "constraints": [{"type": "no_commas"}]}})
    assert not ok.is_numeric
    assert Problem.from_dict({"task": "t4_general", "question": "1+1?", "answer": {"kind": "math", "value": 2}}).is_numeric


def test_unknown_task_rejected():
    with pytest.raises(SchemaError):
        Problem.from_dict(_num(task="t9"))


def _sft(**kw):
    d = {"subset": "domain_reasoning", "messages": [
        {"role": "user", "content": "q"}, {"role": "assistant", "content": "a", "reasoning_content": "r"}]}
    d.update(kw)
    return d


def test_sft_roundtrip_and_texts():
    s = SFTSample.from_dict(_sft())
    assert s.uid.startswith("s-")
    assert SFTSample.from_dict(s.to_dict()) == s
    assert s.assistant_text() == "r\na" and s.user_text() == "q"


@pytest.mark.parametrize("bad", [
    {"subset": "pretrain"},
    {"reasoning_effort": "max"},
    {"messages": [{"role": "user", "content": "q"}]},
    {"messages": [{"role": "user", "content": "q"}, {"role": "system", "content": "s"},
                  {"role": "assistant", "content": "a"}]},
    {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"},
                  {"role": "user", "content": "q2"}]},
])
def test_sft_structural_errors(bad):
    with pytest.raises(SchemaError):
        SFTSample.from_dict(_sft(**bad))
