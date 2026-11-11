"""CurriculumSampler: dataset rows ↔ curriculum draws, stage mixes, exact resume (Incident 2)."""

from __future__ import annotations

import itertools

import pandas as pd

from dpt.common.config import load_config, to_container
from dpt.verl_ext.sampler import CurriculumSampler


def _dataset(n_per_task=40):
    rows = []
    for t in ("t1_num", "t2_code", "t3_report", "t4_general"):
        for i in range(n_per_task):
            rows.append({"data_source": t, "extra_info": {"uid": f"{t}-{i}", "pool": "hard" if i < 5 else "main"}})
    return type("DS", (), {"dataframe": pd.DataFrame(rows)})()


def _cfg(prompts=16):
    c = to_container(load_config("rl/algorithm.yaml", "rl/curriculum.yaml"))["curriculum"]
    c["prompts_per_step"] = prompts
    return c


def test_stage_a_has_no_t3_and_yields_prompts_per_step():
    ds = _dataset()
    s = CurriculumSampler(ds, _cfg(), total_steps=420)
    first = list(itertools.islice(iter(s), 16))
    tasks = [ds.dataframe.iloc[i]["data_source"] for i in first]
    assert "t3_report" not in tasks
    assert tasks.count("t1_num") == 8                  # 50% of 16 in stage A


def test_state_round_trip_is_exact():
    ds = _dataset()
    a = CurriculumSampler(ds, _cfg(), total_steps=420)
    it = iter(a)
    _ = [next(it) for _ in range(40)]                  # mid-step position
    state = a.state_dict()
    rest_a = [next(it) for _ in range(50)]
    b = CurriculumSampler(ds, _cfg(), total_steps=420)
    b.load_state_dict(state)
    rest_b = list(itertools.islice(iter(b), 50))
    assert rest_a == rest_b
