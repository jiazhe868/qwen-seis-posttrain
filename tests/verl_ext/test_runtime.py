"""Runtime helpers: stage-boundary-safe mix selection and TQ payload conversions."""

from __future__ import annotations

import numpy as np
import torch

from dpt.common.config import to_container
from dpt.launch.common import compose_stage
from dpt.rl.curriculum import Curriculum, CurriculumConfig
from dpt.verl_ext.runtime import build_runtime, row_sums, to_str_list


def _rt():
    cfg = to_container(compose_stage("rl/rl_27b.yaml"))
    rt = build_runtime(cfg)
    pools = {t: {"main": [f"{t}-{i}" for i in range(50)], "hard": []} for t in cfg["curriculum"]["stages"][0]["mix"]}
    rt.curriculum = Curriculum(CurriculumConfig.from_dict(cfg["curriculum"]), pools)
    return rt


def test_stage_boundary_batch_with_t3_does_not_crash():
    rt = _rt()
    for s in range(121):                      # sampler already drew step 120 (stage B, T3 > 0)
        rt.curriculum.sample(s)
    mix = rt.mix_for_batch(119, {"t1_num", "t3_report"})
    assert mix["t3_report"] > 0
    assert rt.mix_for_batch(119, {"t1_num", "t2_code"})["t3_report"] == 0.0   # normal case untouched


def test_row_sums_and_strings():
    nested = torch.nested.nested_tensor([torch.ones(3), torch.ones(5)])
    assert row_sums(nested).tolist() == [3.0, 5.0]
    assert row_sums(torch.ones(2, 4)).tolist() == [4.0, 4.0]
    assert to_str_list(np.array(["a", None], dtype=object)) == ["a", ""]
