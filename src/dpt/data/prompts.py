"""Task prompts and final-answer formats. These are a contract shared by RL export, graders and eval.

One function, :func:`build_messages`, turns a problem into chat messages. RL
export (``rl_build``) and the eval harness both call it. A model is therefore
trained and evaluated on byte-identical prompt formats, and the graders in
``dpt.rewards`` / ``dpt.eval`` can rely on the answer formats below.

Final-answer contract
---------------------
=============  =================================================================
task           what the final answer (text after ``</think>``) must contain
=============  =================================================================
t1_num         exactly one ``\\boxed{<number> <unit>}``, e.g. ``\\boxed{20 s}``,
               ``\\boxed{3.2 km}``, or ``\\boxed{6.5}`` when dimensionless.
               No ranges, no multiple candidates. Graders score both as 0.
               These are hacks from report §Reward hacking, steps 95 and 140.
t2_code        the complete solution in the **last** ```python fenced block
t3_report      a structured report ending in a ``References`` section, one
               source per line with a DOI or full title (citation verification)
t4 (math)      same as t1_num
t4 (ifeval)    no system prompt. The user instruction is the whole prompt,
               because instruction-following benchmarks are scored that way.
=============  =================================================================

``PROMPT_VERSION`` is written into every exported row's ``extra_info``. If you
change any prompt text, bump it, because results across prompt versions are
not comparable.
"""

from __future__ import annotations

from typing import Any

from dpt.common.types import T1_NUM, T2_CODE, T3_REPORT, T4_GENERAL, check_task

PROMPT_VERSION = "v1"

NUMERIC_FORMAT = (
    "Give exactly one final answer on the last line in the form \\boxed{<number> <unit>}, "
    "for example \\boxed{20 s} or \\boxed{3.2 km}; omit the unit only for dimensionless quantities. "
    "Do not give a range and do not list several candidate answers."
)

SYSTEM_PROMPTS: dict[str, str] = {
    T1_NUM: (
        "You are an expert in the domain. Solve the quantitative problem carefully, keeping track of "
        "units and significant figures. " + NUMERIC_FORMAT
    ),
    T2_CODE: (
        "You are an expert scientific programmer. Write correct, self-contained Python that satisfies "
        "the specification. Your code will be executed against hidden tests in an isolated sandbox. "
        "Put the complete final solution in a single ```python fenced code block at the end of your "
        "answer; that last block is the only code that will be run."
    ),
    T3_REPORT: (
        "You are a domain expert writing a technical report. Be accurate, specific and well organized; "
        "length is not a goal in itself. Cite only sources you are certain exist. End with a section "
        "titled 'References' listing each cited source on its own line with its DOI, or its full title, "
        "authors and year if it has no DOI."
    ),
    "t4_math": "Solve the math problem. " + NUMERIC_FORMAT,
}


def system_prompt(task: str, *, kind: str | None = None) -> str | None:
    """System prompt for ``task`` (``kind`` distinguishes t4 math vs ifeval); ``None`` = no system turn."""
    check_task(task)
    if task == T4_GENERAL:
        if kind == "math":
            return SYSTEM_PROMPTS["t4_math"]
        if kind == "ifeval":
            return None
        raise ValueError("t4_general prompts need kind='math' or kind='ifeval'")
    return SYSTEM_PROMPTS[task]


def build_messages(task: str, question: str, *, extra: dict[str, Any] | None = None) -> list[dict[str, str]]:
    """Build the chat messages for one problem.

    ``extra`` (all optional):
        kind          t4 sub-kind: "math" | "ifeval" (required for t4_general)
        entry_point   t2: function name the tests call (appended to the user turn)
        length_hint   t3: approximate target length in words (appended to the user turn)
    """
    extra = dict(extra or {})
    sys = system_prompt(task, kind=extra.get("kind"))
    user = question.strip()
    if task == T2_CODE and extra.get("entry_point"):
        user += f"\n\nImplement the function `{extra['entry_point']}`."
    if task == T3_REPORT and extra.get("length_hint"):
        user += f"\n\nAim for roughly {int(extra['length_hint'])} words."
    messages: list[dict[str, str]] = []
    if sys:
        messages.append({"role": "system", "content": sys})
    messages.append({"role": "user", "content": user})
    return messages


def messages_for_problem(problem: Any) -> list[dict[str, str]]:
    """Convenience wrapper for a :class:`dpt.data.schema.Problem` (or its dict form)."""
    d = problem if isinstance(problem, dict) else problem.to_dict()
    answer = d.get("answer") or {}
    # Same rule as the eval harness (dpt.eval.harness.build_messages), so a T3 training prompt and a D-Report
    # item with the same answer.prompt_extra.length_hint get the same "Aim for roughly N words." line.
    extra: dict[str, Any] = dict(answer.get("prompt_extra") or {})
    if d["task"] == T4_GENERAL:
        extra["kind"] = answer.get("kind")
    if d["task"] == T2_CODE:
        extra["entry_point"] = answer.get("entry_point")
    return build_messages(d["task"], d["question"], extra=extra)
