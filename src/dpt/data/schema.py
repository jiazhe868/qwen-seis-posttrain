"""Record schemas for the RL problem bank and the SFT corpus.

The schemas are deliberately plain dataclasses that round-trip through JSON
(``to_dict`` / ``from_dict``). Every pipeline stage reads and writes JSONL of
these records, and the manifests hash them.

``validate()`` checks *structure* (types, required fields, finite numbers).
*Policy* decisions, such as "a tool trajectory is only admitted if it passed
hidden tests", live in the stage that applies them (``sft_build``), so they are
visible in the funnel statistics instead of being hidden as schema errors.

Stable ids
----------
``Problem.uid = "p-" + sha256(normalize_text(question))[:16]``. Because the id
is a function of the normalized question, re-ingesting the same source gives
the same ids, and exact duplicates collapse to one id before MinHash runs.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from dpt.common.hashing import canonical_json, short_hash
from dpt.common.text import normalize_text
from dpt.common.types import EFFORTS, SFT_SUBSETS, T1_NUM, T2_CODE, T3_REPORT, T4_GENERAL, TASKS

ROLES = ("system", "user", "assistant", "tool")


class SchemaError(ValueError):
    """A record does not match the schema in docs/dev/conventions.md §4."""


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def problem_uid(question: str) -> str:
    return "p-" + short_hash(normalize_text(question))


def sft_uid(messages: list[dict[str, Any]]) -> str:
    return "s-" + short_hash(canonical_json(messages))


# ---------------------------------------------------------------------------------------------
# RL problem
# ---------------------------------------------------------------------------------------------
@dataclass
class Problem:
    uid: str
    task: str
    question: str
    answer: dict[str, Any]
    source: dict[str, Any] = field(default_factory=dict)
    # Free-form metadata. Keys used by the pipeline:
    #   pool: "main" | "hard"              (difficulty stage)
    #   pass_at_8_medium: float            (difficulty stage)
    #   flags: list[str]                   (any stage; informational)
    #   params: dict[str, float]           (numeric input parameters, for answer-level decontam)
    #   generator: {"name", "params", "scaling"?, "invariant"?}  (programmatic problems)
    meta: dict[str, Any] = field(default_factory=dict)

    # -- construction -------------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict[str, Any], *, assign_uid: bool = True) -> "Problem":
        """Build from a dict; ``uid`` is computed from the question when missing."""
        if "question" not in d or "task" not in d:
            raise SchemaError(f"problem record needs 'task' and 'question': {sorted(d)}")
        uid = d.get("uid") or (problem_uid(d["question"]) if assign_uid else "")
        p = cls(
            uid=uid,
            task=d["task"],
            question=d["question"],
            answer=dict(d.get("answer") or {}),
            source=dict(d.get("source") or {}),
            meta=dict(d.get("meta") or {}),
        )
        p.fill_defaults()
        p.validate()
        return p

    def fill_defaults(self) -> None:
        """Fill optional numeric-answer fields so equal problems hash equally (called by from_dict).

        Default tolerances match the graders' (configs/rl/rewards.yaml): T1 domain quantities are
        measured/derived values (1% relative); T4 math answers are exact numbers (1e-6 relative).
        Writing T1's 1% into a math answer would silently loosen the math grader 10,000×.
        """
        if self.is_numeric:
            self.answer.setdefault("unit", "")
            default_rel = 1e-6 if self.task == T4_GENERAL else 1e-2
            self.answer.setdefault("rel_tol", default_rel)
            self.answer.setdefault("abs_tol", 0.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # -- validation ---------------------------------------------------------------------------
    def validate(self) -> None:
        if self.task not in TASKS:
            raise SchemaError(f"{self.uid}: unknown task {self.task!r}")
        if not isinstance(self.question, str) or not self.question.strip():
            raise SchemaError(f"{self.uid}: empty question")
        if not self.uid:
            raise SchemaError("missing uid")
        a = self.answer
        if self.task == T1_NUM or (self.task == T4_GENERAL and a.get("kind") == "math"):
            self._validate_numeric(a)
        elif self.task == T2_CODE:
            for k in ("entry_point", "visible_tests", "hidden_tests_id"):
                if not isinstance(a.get(k), str) or not a.get(k):
                    raise SchemaError(f"{self.uid}: t2_code answer needs non-empty string {k!r}")
            if a.get("language", "python") != "python":
                raise SchemaError(f"{self.uid}: only python code tasks are supported")
        elif self.task == T3_REPORT:
            rubric = a.get("rubric")
            if not isinstance(rubric, list) or not rubric:
                raise SchemaError(f"{self.uid}: t3_report needs a non-empty rubric")
            # Items are either 0–10 criteria {criterion, weight} or atomic yes/no checks {check, weight} (since
            # 2026-11-07, dpt.data.gen.t3_atomic); one rubric uses one kind.
            kinds = {"check" if isinstance(i, dict) and i.get("check") else "criterion" for i in rubric}
            if len(kinds) > 1:
                raise SchemaError(f"{self.uid}: a rubric mixes checks and criteria")
            key = kinds.pop()
            for item in rubric:
                if not isinstance(item, dict) or not item.get(key) or not _is_num(item.get("weight")):
                    raise SchemaError(f"{self.uid}: bad rubric item {item!r}")
                if float(item["weight"]) <= 0:
                    raise SchemaError(f"{self.uid}: rubric weights must be positive")
            if not isinstance(a.get("reference_length"), int) or a["reference_length"] <= 0:
                raise SchemaError(f"{self.uid}: t3_report needs positive int reference_length (tokens)")
        elif self.task == T4_GENERAL:
            if a.get("kind") != "ifeval":
                raise SchemaError(f"{self.uid}: t4_general kind must be 'math' or 'ifeval'")
            cons = a.get("constraints")
            if not isinstance(cons, list) or not cons or not all(isinstance(c, dict) for c in cons):
                raise SchemaError(f"{self.uid}: ifeval needs a non-empty list of constraint dicts")

    def _validate_numeric(self, a: dict[str, Any]) -> None:
        if not _is_num(a.get("value")):
            raise SchemaError(f"{self.uid}: numeric answer needs finite 'value'")
        if not isinstance(a.get("unit", ""), str):
            raise SchemaError(f"{self.uid}: 'unit' must be a string")
        for k in ("rel_tol", "abs_tol"):
            if k in a and (not _is_num(a[k]) or float(a[k]) < 0):
                raise SchemaError(f"{self.uid}: {k} must be a non-negative number")

    # -- helpers ------------------------------------------------------------------------------
    @property
    def is_numeric(self) -> bool:
        return self.task == T1_NUM or (self.task == T4_GENERAL and self.answer.get("kind") == "math")

    @property
    def pool(self) -> str:
        return str(self.meta.get("pool", "main"))


# ---------------------------------------------------------------------------------------------
# SFT sample
# ---------------------------------------------------------------------------------------------
@dataclass
class SFTSample:
    uid: str
    subset: str
    messages: list[dict[str, Any]]
    reasoning_effort: str = "medium"
    tools: list[dict[str, Any]] | None = None
    source: dict[str, Any] = field(default_factory=dict)
    # Keys used by sft_build: verified (domain_reasoning), exec_ok / hidden_tests_passed
    # (tool_traj), generated_by (self_distill), rubric (report), n_tokens / bucket (added).
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SFTSample":
        msgs = d.get("messages")
        if not isinstance(msgs, list):
            raise SchemaError("SFT record needs a 'messages' list")
        s = cls(
            uid=d.get("uid") or sft_uid(msgs),
            subset=d.get("subset", ""),
            messages=[dict(m) for m in msgs],
            reasoning_effort=d.get("reasoning_effort", "medium"),
            tools=d.get("tools"),
            source=dict(d.get("source") or {}),
            meta=dict(d.get("meta") or {}),
        )
        s.validate()
        return s

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.subset not in SFT_SUBSETS:
            raise SchemaError(f"{self.uid}: unknown SFT subset {self.subset!r}")
        if self.reasoning_effort not in EFFORTS:
            raise SchemaError(f"{self.uid}: unknown reasoning effort {self.reasoning_effort!r}")
        if not self.messages:
            raise SchemaError(f"{self.uid}: empty conversation")
        for i, m in enumerate(self.messages):
            if m.get("role") not in ROLES:
                raise SchemaError(f"{self.uid}: message {i} has bad role {m.get('role')!r}")
            if m["role"] == "system" and i != 0:
                raise SchemaError(f"{self.uid}: system message must be first")
        roles = [m["role"] for m in self.messages]
        if "assistant" not in roles:
            raise SchemaError(f"{self.uid}: no assistant turn to train on")
        if "user" not in roles:
            raise SchemaError(f"{self.uid}: no user turn")
        if roles[-1] != "assistant":
            raise SchemaError(f"{self.uid}: conversation must end with an assistant turn")
        if self.tools is not None and not isinstance(self.tools, list):
            raise SchemaError(f"{self.uid}: tools must be a list or null")

    # -- helpers ------------------------------------------------------------------------------
    def user_text(self) -> str:
        return "\n".join(str(m.get("content") or "") for m in self.messages if m["role"] == "user")

    def assistant_text(self, *, include_reasoning: bool = True) -> str:
        parts: list[str] = []
        for m in self.messages:
            if m["role"] != "assistant":
                continue
            if include_reasoning and m.get("reasoning_content"):
                parts.append(str(m["reasoning_content"]))
            parts.append(str(m.get("content") or ""))
        return "\n".join(parts)
