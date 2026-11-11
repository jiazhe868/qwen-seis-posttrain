from dpt.data.schema import Problem


def test_math_and_domain_numeric_get_their_own_default_tolerance():
    t1 = Problem.from_dict({"task": "t1_num", "question": "q1", "answer": {"value": 2.0, "unit": "s"}})
    t4 = Problem.from_dict({"task": "t4_general", "question": "q2", "answer": {"kind": "math", "value": 0.5}})
    assert t1.answer["rel_tol"] == 1e-2
    assert t4.answer["rel_tol"] == 1e-6
