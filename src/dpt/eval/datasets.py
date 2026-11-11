"""Eval-set loading and the grader registry.

Storage and isolation (report §数据): eval sets live under
``${project.paths.eval_secure_root}``, a directory that training processes
cannot read. Each set is a JSONL of :class:`EvalItem` records with a
``*.manifest.json`` sidecar. The loader verifies the manifest whenever one
exists, so a silently edited eval file cannot produce a "result". Internal
eval sets embed a canary string. The data pipeline scans training data and
model outputs for it; this module only carries it through.

Every eval set or general benchmark names a grader *kind*:

==============  ===============================================  ==========================
kind            gold fields                                      used by
==============  ===============================================  ==========================
``numeric``     value, unit, rel_tol, abs_tol                    d_num, d_fresh
``code``        hidden_tests_id | tests_source, entry_point      d_code, livecodebench
``pairwise``    rubric, reference_length_words                   d_report (needs base run)
``choice``      letter, n_options                                gpqa_diamond, mmlu_pro_2k
``ifeval``      constraints                                      ifeval
``toolcall``    calls | name+arguments                           tool_calling
``refusal``     should_refuse                                    refusal regression set
==============  ===============================================  ==========================
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from dpt.common.io import read_jsonl
from dpt.common.logging import get_logger
from dpt.eval.graders import GradeResult

log = get_logger(__name__)


@dataclass
class EvalItem:
    id: str
    set: str
    question: str
    answer: dict[str, Any]
    task: str | None = None  # dpt task id, used to build the prompt (t1_num, t2_code, ...)
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_record(cls, rec: dict[str, Any], set_name: str) -> "EvalItem":
        return cls(
            id=str(rec.get("id") or rec.get("uid")),
            set=set_name,
            question=rec["question"],
            answer=dict(rec.get("answer") or {}),
            task=rec.get("task"),
            meta=dict(rec.get("meta") or {}),
        )


def load_eval_items(
    path: str | os.PathLike, set_name: str, *, verify: bool = True, limit: int | None = None
) -> list[EvalItem]:
    """Load a JSONL eval set, verifying its manifest if present."""
    p = Path(path)
    if verify and Path(str(p) + ".manifest.json").exists():
        from dpt.common.manifest import verify_manifest

        verify_manifest(p)
    elif verify:
        log.warning("no manifest for %s; results will not be traceable to a content hash", p)
    items = [EvalItem.from_record(r, set_name) for r in read_jsonl(p)]
    ids = [it.id for it in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{p}: duplicate item ids")
    return items[:limit] if limit else items


# --------------------------------------------------------------------------- grader registry
GraderFn = Callable[[str, dict[str, Any]], GradeResult]


def get_grader(kind: str, **kwargs: Any) -> GraderFn:
    """Return ``grade(answer_text, gold) -> GradeResult`` for a grader kind."""
    if kind == "numeric":
        from dpt.eval.graders.numeric import grade_numeric

        missing_unit = kwargs.get("missing_unit", "strict")      # same policy as training and dev gold
        return lambda a, g: grade_numeric(a, g, missing_unit=missing_unit)
    if kind == "code":
        from dpt.eval.graders.code import HiddenTestClient, LocalTestRunner, grade_code

        endpoint = kwargs.get("sandbox_endpoint")
        runner = HiddenTestClient(endpoint) if endpoint else LocalTestRunner()
        if not endpoint:
            log.warning("code grader: no sandbox_endpoint configured, using LocalTestRunner (dev only)")
        return lambda a, g: grade_code(a, g, runner)
    if kind == "choice":
        from dpt.eval.graders.choice import grade_choice

        return grade_choice
    if kind == "ifeval":
        from dpt.eval.graders.ifeval import grade_ifeval

        return grade_ifeval
    if kind == "toolcall":
        from dpt.eval.graders.toolcall import grade_tool_calls

        return grade_tool_calls
    if kind == "refusal":
        from dpt.eval.graders.refusal import grade_refusal

        return grade_refusal
    raise ValueError(f"unknown grader kind {kind!r} (pairwise is handled by the harness)")
