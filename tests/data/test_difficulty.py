"""Difficulty calibration: drop 8/8, route 0/8 to the hard pool (report §处理流水线 step 5)."""

from __future__ import annotations

from dpt.common.config import load_config, to_container
from dpt.data.backends import FakeGenerator, fake_grade
from dpt.data.difficulty import DifficultyConfig, calibrate
from dpt.data.schema import Problem


def _probs(n):
    return [Problem.from_dict({"task": "t1_num", "question": f"Problem number {i} about travel times?",
                               "answer": {"value": 1.0}}) for i in range(n)]


def test_pools_and_pass_rates():
    gen = FakeGenerator()
    res = calibrate(_probs(60), gen, fake_grade, DifficultyConfig(max_concurrency=4))
    st = res.stats()
    assert st["dropped_all_correct"] > 0 and st["hard_pool"] > 0 and st["main_pool"] > 0
    for p in res.kept:
        rate = p.meta["pass_at_8_medium"]
        assert (p.pool == "hard") == (rate == 0.0)
        assert rate < 1.0
    assert all(d["k"] == 8 for d in res.dropped)


def test_calibration_is_deterministic():
    a = calibrate(_probs(20), FakeGenerator(), fake_grade, DifficultyConfig(max_concurrency=8))
    b = calibrate(_probs(20), FakeGenerator(), fake_grade, DifficultyConfig(max_concurrency=1))
    assert [p.uid for p in a.kept] == [p.uid for p in b.kept]
    assert a.histogram == b.histogram


def test_generation_settings_are_passed_through_and_skip_tasks():
    seen = {}

    def gen(messages, *, n, effort, max_tokens, seed):
        seen.update(n=n, effort=effort, max_tokens=max_tokens)
        return ["x"] * n

    cfg = DifficultyConfig(n_samples=4, effort="medium", max_tokens=123, max_concurrency=1)
    res = calibrate(_probs(1), gen, lambda p, r: 0.5, cfg)
    assert seen == {"n": 4, "effort": "medium", "max_tokens": 123}
    assert res.kept[0].pool == "hard"  # 0.5 < threshold 1.0 → never correct
    skip = calibrate(_probs(2), gen, lambda p, r: 1.0, DifficultyConfig(skip_tasks=["t1_num"]))
    assert len(skip.kept) == 2 and "pass_at_8_medium" not in skip.kept[0].meta


def test_continuous_reward_threshold():
    p = Problem.from_dict({"task": "t3_report", "question": "Write a report on site effects.",
                           "answer": {"rubric": [{"criterion": "c", "weight": 1}], "reference_length": 100}})
    res = calibrate([p], lambda m, **k: ["r"] * k["n"], lambda p, r: 0.95, DifficultyConfig())
    assert res.dropped and res.dropped[0]["reason"] == "all_correct"  # 0.95 ≥ 0.9 threshold for t3


def test_audit_calibrates_a_sample_per_template_and_decides_for_the_template():
    from dpt.data.difficulty import DifficultyAuditConfig, DifficultyConfig, calibrate_audit
    from dpt.data.schema import Problem

    def prob(template, d):
        return Problem.from_dict({
            "task": "t1_num", "question": f"T{template}: {d} km at 5 km/s. Time in s?",
            "answer": {"value": d / 5, "unit": "s"},
            "meta": {"family": "travel_time", "template": template,
                     "generator": {"name": "travel_time", "params": {"distance_km": d, "velocity_km_s": 5.0}}}})

    probs = [prob(t, d) for t in (0, 1, 2) for d in range(10, 50)]
    probs.append(Problem.from_dict({"task": "t3_report", "question": "Explain coda waves.",
                                    "answer": {"rubric": [{"criterion": "c", "weight": 1}], "reference_length": 500}}))
    calls = []

    def gen(messages, n, **kw):                     # template 0: always right; 1: never; 2: right iff d is even
        q = messages[-1]["content"]
        calls.append(q)
        t, d = int(q[1]), int(q.split(": ")[1].split(" ")[0])
        ok = {0: True, 1: False, 2: d % 2 == 0}[t]
        return ["ok" if ok else "bad"] * n

    res, groups = calibrate_audit(probs, gen, lambda p, o: 1.0 if o == "ok" else 0.0,
                                  DifficultyConfig(n_samples=4, skip_tasks=["t3_report"], max_concurrency=1),
                                  DifficultyAuditConfig(per_group=10))
    assert len(calls) == 30                                            # 10 audited per template, not 120
    assert groups["t1_num:travel_time:t0"]["decision"] == "drop"
    assert groups["t1_num:travel_time:t1"]["decision"] == "hard"
    assert groups["t1_num:travel_time:t2"]["decision"] == "keep"
    kept = {p.uid: p for p in res.kept}
    t2 = [p for p in probs if p.meta.get("template") == 2]
    for p in t2:
        a = p.meta["difficulty_audit"]["audited"]
        d = p.meta["generator"]["params"]["distance_km"]
        if a and d % 2 == 0:
            assert p.uid not in kept                                   # audited and 4/4: dropped
        elif a:
            assert kept[p.uid].pool == "hard"                          # audited and 0/4: hard pool
        else:
            assert kept[p.uid].pool == "main" and 0 < kept[p.uid].meta["pass_at_4_medium"] < 1
    assert all(kept[p.uid].pool == "hard" for p in probs if p.meta.get("template") == 1)
    assert not any(p.meta.get("template") == 0 for p in res.kept)
    assert any(p.task == "t3_report" and p.pool == "main" for p in res.kept)   # skipped task kept as is


def test_difficulty_requests_fit_in_the_base_timeout():
    # Every problem in flight queues on one base server, and a problem waits for the longest of its replies. Bank
    # replies average ~16K tokens, and one A100 then generates ~1,300 tok/s (measured 2026-10-25); the 4× covers
    # the 24K cap, preemptions and slower stretches. At 64 problems in flight and a 30 min timeout most requests
    # timed out and were re-sent from scratch.
    dp = to_container(load_config("data/pipeline.yaml").data_pipeline)
    d, base = dp["difficulty"], dp["backends"]["base"]
    queue_s = d["max_concurrency"] * d["n_samples"] * 16000 / 1300      # Little's law: in flight / throughput
    assert base["timeout_s"] >= 4 * queue_s


def test_two_stage_samples_every_member_of_surviving_templates_only():
    from dpt.data.difficulty import DifficultyAuditConfig, DifficultyConfig, calibrate_two_stage
    from dpt.data.schema import Problem

    def prob(template, d):
        return Problem.from_dict({
            "task": "t1_num", "question": f"T{template}: {d} km at 5 km/s. Time in s?",
            "answer": {"value": d / 5, "unit": "s"},
            "meta": {"family": "travel_time", "template": template,
                     "generator": {"name": "travel_time", "params": {"distance_km": d, "velocity_km_s": 5.0}}}})

    probs = [prob(t, d) for t in (0, 1, 2) for d in range(10, 50)]       # 40 per template
    calls = []

    def gen(messages, n, **kw):    # template 0: always right; 1: never; 2: 3 of 4 right iff d is odd, else all
        q = messages[-1]["content"]
        calls.append(q)
        t, d = int(q[1]), int(q.split(": ")[1].split(" ")[0])
        if t == 0:
            return ["ok"] * n
        if t == 1:
            return ["bad"] * n
        return (["ok"] * (n - 1) + ["bad"]) if d % 2 else ["ok"] * n

    res, groups = calibrate_two_stage(probs, gen, lambda p, o: 1.0 if o == "ok" else 0.0,
                                      DifficultyConfig(n_samples=4, max_concurrency=1),
                                      DifficultyAuditConfig(per_group=10))
    # stage 1: 10 per template; stage 2: the other 30 of templates 1 and 2 (template 0 is dropped whole)
    assert len(calls) == 30 + 60
    assert groups["t1_num:travel_time:t0"]["decision"] == "drop"
    assert groups["t1_num:travel_time:t1"]["decision"] == "keep"      # no whole-group hard pool: each is measured
    assert groups["t1_num:travel_time:t2"]["decision"] == "keep" and groups["t1_num:travel_time:t2"]["mean_pass"] > 0.8
    kept = {p.uid: p for p in res.kept}
    for p in probs:
        t, d = p.meta["template"], p.meta["generator"]["params"]["distance_km"]
        if t == 0 or (t == 2 and d % 2 == 0):
            assert p.uid not in kept                                     # dropped: template mastered or 4/4
        elif t == 1:
            assert kept[p.uid].pool == "hard" and kept[p.uid].meta["pass_at_4_medium"] == 0.0
        else:
            assert kept[p.uid].pool == "main" and kept[p.uid].meta["pass_at_4_medium"] == 0.75   # its own rate
    assert sum(1 for d in res.dropped if d["reason"] == "group_all_correct") == 30   # unsampled members of t0


def test_pipeline_grades_correctness_not_the_length_penalised_score(monkeypatch):
    import dpt.rewards.router as router
    from dpt.data.pipeline import Context

    seen = {}

    def fake_compute_score(task, response, gt, extra):
        seen.update(extra)
        return {"score": 0.6, "base_score": 1.0, "acc": 1.0}       # right, but a length penalty took 0.4 off

    monkeypatch.setattr(router, "compute_score", fake_compute_score)
    ctx = Context.__new__(Context)
    ctx.offline = False
    grade = ctx.grade_fn()
    p = Problem.from_dict({"task": "t3_report", "question": "Explain coda waves.",
                           "answer": {"rubric": [{"criterion": "c", "weight": 1}], "reference_length": 500}})
    assert grade(p, "answer") == 1.0
    assert seen["question"] == "Explain coda waves." and seen["response_length"] == 0
