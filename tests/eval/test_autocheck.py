"""Automatic checks of the drafted eval sets (no expert review): D-Fresh solver code, Gemma solves and
adjudication; D-Report rubrics checked by answer-first majority votes."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter

import pytest

from dpt.common.io import read_jsonl, write_jsonl
from dpt.eval.autocheck import (
    check_fresh,
    check_report_row,
    fresh_records,
    last_block,
    main,
    parse_last_json,
    report_records,
    select_balanced,
)
from dpt.eval.datasets import EvalItem, load_eval_items
from dpt.eval.drafts import REPORT_COLUMNS, write_csv
from dpt.eval.harness import build_messages
from dpt.eval.hosted import ChatReply
from dpt.rewards.code_sandbox import LocalSubprocessSandbox

Q = ("A P wave crosses a {h} km thick crustal layer at {v} km/s and then travels 60 km horizontally in the mantle "
     "at 8.0 km/s. What is the total travel time along this path, in seconds?")


def _draft(i, h=30.0, v=6.0, code=None, unit="s", value=None, topic="t"):
    value = h / v + 60 / 8.0 if value is None else value
    code = code or f"def solve():\n    return {h} / {v} + 60 / 8.0\n"
    filler = " ".join(f"s{i}x{k}" for k in range(40))           # distinct wording: not near-duplicates
    return {"draft_id": f"d{i}", "topic": topic, "question": Q.format(h=h, v=v) + f" (case {i}) {filler}",
            "answer_value": value, "answer_unit": unit, "rel_tolerance_percent": 1, "solution": "",
            "solve_py": code, "drafted_by": "gpt-6-luna-2026-09-22", "date_written": "2026-10-15"}


class Checker:
    """Fake Gemma: answers come from a callable on the prompt; counts requests."""

    model = "google/gemma-4-26B-A4B-it"

    def __init__(self, answer):
        self.answer, self.calls = answer, 0

    def request(self, messages):
        self.calls += 1
        content, fin = self.answer(messages[-1]["content"])
        return ChatReply(content, fin, self.model)


@pytest.fixture()
def sandbox():
    return LocalSubprocessSandbox(python=sys.executable, cpu_s=5, mem_mb=1024)


def test_parse_last_json_skips_reasoning_with_braces():
    text = 'thinking: maybe {"criteria": "draft"} or {x}... final:\n{"question": {"ok": true}, "criteria": [{"id": 1}]}'
    assert parse_last_json(text, "criteria") == {"question": {"ok": True}, "criteria": [{"id": 1}]}
    with pytest.raises(ValueError):
        parse_last_json("no json here", "criteria")


def test_last_block_takes_the_part_handed_in():
    assert last_block("<think>draft <answer>x</answer></think>\n<answer>final</answer>", "answer") == "final"
    assert last_block("plan <solution>a</solution> then <solution>b</solution>", "solution") == "b"
    assert last_block("no tags " + "y" * 20, "answer", max_chars=5) == "…yyyyy"      # the end of the reply


def test_select_balanced_round_robin_over_topics():
    rows = [{"topic": t, "question": f"{t}{i}"} for t in "ab" for i in range(5)] + [{"topic": "c", "question": "c0"}]
    out = select_balanced(rows, 5)
    assert sorted(r["topic"] for r in out) == ["a", "a", "b", "b", "c"]
    assert out == select_balanced(list(reversed(rows)), 5)            # deterministic, order-free


# --------------------------------------------------------------------------- D-Fresh


def test_fresh_check_runs_the_code_and_counts_independent_solutions(sandbox):
    drafts = [
        _draft(0),                                                        # good
        _draft(1, code="def solve():\n    return 99.0\n"),                # code disagrees with the stated answer
        _draft(2, code="import socket\ndef solve():\n    return 12.5\n"),  # never run
        _draft(3, unit="fortnights-ish"),                                 # unit does not parse
        _draft(4, h=45.0),                                                # Gemma never agrees
    ]

    def answer(prompt):
        if "case 4" in prompt:
            return "\\boxed{3 s}", "stop"
        if "case 0" in prompt:
            answer.n0 += 1
            # 1 of 4 agrees; an earlier boxed attempt in the reasoning does not count against it
            return ("try \\boxed{14 s}... no: \\boxed{12.5 s}" if answer.n0 == 2 else "\\boxed{14 s}"), "stop"
        return "\\boxed{12.5 s}", "stop"
    answer.n0 = 0

    rows = {r["draft_id"]: r for r in check_fresh(drafts, Checker(answer), sandbox, solves=4, adjudicate=False,
                                                  max_concurrency=1)}
    assert rows["d0"]["status"] == "kept" and (rows["d0"]["agree"], rows["d0"]["solves"]) == (1, 4)
    assert rows["d1"]["reason"].startswith("code_mismatch")
    assert rows["d2"]["reason"] == "code_blocked:network_access"
    assert rows["d3"]["reason"].startswith("stated_answer_unparsed")
    assert rows["d4"]["reason"] == "unconfirmed:0/4"


def test_fresh_adjudication_settles_disagreements(sandbox):
    solves = {0: ["12.5 s", "14 s", "14 s", "14 s"],      # 1 vs 3: adjudicated, Gemma upholds the stated answer
              1: ["14 s"] * 4,                            # 0 vs 4: adjudicated, the stated answer is wrong
              2: ["12.5 s", "14 s", "14 s", "14 s"],      # position-biased adjudicator: the orders disagree
              3: ["12.5 s", "12.5 s", "14 s", "14 s"],    # 2 vs 2: adjudicated, ambiguous
              4: ["12.5 s", "14 s", "15 s", "16 s"],      # no rival with 2 votes: kept on the one match
              5: ["12.5 s"] * 3 + ["14 s"],               # clear majority: kept
              6: ["0.23333 min", "14 s", "14 s", "14 s"],  # 14 s in two units is one rival; no wrong step named
              7: [None] * 4}                              # every solve cut off at max_tokens
    judge = {0: "fair", 1: "rival", 2: "always_A", 3: "ambiguous", 6: "no_step"}
    drafts = [{**_draft(i), "solution": f"Luna work for case {i}"} for i in solves]
    seen, adjudications = Counter(), []

    def answer(prompt):
        i = int(re.search(r"\(case (\d+)\)", prompt).group(1))
        if "## Solution A" in prompt:
            adjudications.append(prompt)
            stated = "A" if "Final answer A: 12.5 s" in prompt else "B"
            pick = {"fair": stated, "rival": "AB".replace(stated, ""), "always_A": "A", "ambiguous": stated,
                    "no_step": stated}[judge[i]]
            step = "" if judge[i] == "no_step" else "uses the mantle velocity in the crust"
            return json.dumps({"correct": pick, "ambiguous": judge[i] == "ambiguous", "wrong_step": step}), "stop"
        v = solves[i][seen[i]]
        seen[i] += 1
        if v is None:
            return "still thinking", "length"
        return f"scratch \\boxed{{1 s}}</think><solution>Gemma work {v}\n\\boxed{{{v}}}</solution>", "stop"

    rows = {r["draft_id"]: r for r in check_fresh(drafts, Checker(answer), sandbox, solves=4, max_concurrency=1)}
    got = {k: (r["status"], r["reason"]) for k, r in rows.items()}
    assert got == {"d0": ("kept", "adjudicated"), "d1": ("drop", "adjudicated_refuted"),
                   "d2": ("drop", "adjudicated_split"), "d3": ("drop", "adjudicated_ambiguous"),
                   "d4": ("kept", "ok"), "d5": ("kept", "ok"), "d6": ("drop", "adjudicated_split"),
                   "d7": ("drop", "no_readable_solution")}
    assert rows["d0"]["adjudication"] == "upheld/upheld" and "mantle velocity" in rows["d0"]["adjudication_note"]
    assert rows["d2"]["adjudication"] == "upheld/refuted" and rows["d6"]["adjudication"] == "unsupported/unsupported"
    assert (rows["d6"]["agree"], rows["d6"]["rival"]) == (0, 4) and rows["d4"]["adjudication"] == ""
    assert rows["d7"]["truncated"] == 4 and rows["d0"]["rival_answer"] == "14 s"
    assert len(adjudications) == 10                                       # 5 problems, both orders each
    first = adjudications[0]
    assert "Luna work for case 0" in first and "Gemma work 14 s" in first and "scratch" not in first
    assert first.index("Luna work") < first.index("Gemma work")            # stated first, then swapped
    assert adjudications[1].index("Gemma work") < adjudications[1].index("Luna work")


def test_fresh_check_fails_the_run_when_the_sandbox_is_down():
    class Down:
        def run(self, req):
            from dpt.rewards.code_sandbox import SandboxResult
            return SandboxResult(status="infra_error", detail="no container")

    with pytest.raises(RuntimeError, match="sandbox unavailable"):
        check_fresh([_draft(0)], Checker(lambda p: ("\\boxed{12.5 s}", "stop")), Down(), max_concurrency=1)


def test_fresh_records_dedupe_balance_and_canary():
    drafts = [_draft(i, topic="a" if i < 3 else "b") for i in range(4)]
    drafts.append({**drafts[0], "draft_id": "dup", "question": drafts[0]["question"] + "!"})
    verdicts = [{"draft_id": d["draft_id"], "status": "kept", "reason": "ok", "agree": 2, "solves": 4} for d in drafts]
    records, log_rows, stats = fresh_records(drafts, verdicts, n_keep=3)
    assert len(records) == 3 and {r["meta"]["topic"] for r in records} == {"a", "b"}
    assert all(r["meta"]["canary"].startswith("DPT-CANARY-") and r["task"] == "t1_num" for r in records)
    assert records[0]["answer"]["rel_tol"] == pytest.approx(0.01)
    status = sorted(r["status"] for r in log_rows)
    assert status == ["drop", "kept", "kept", "kept", "spare"]
    assert {"stage": "dropped:near_duplicate", "count": 1} in stats and stats[-1] == {"stage": "kept", "count": 3}
    assert {"stage": "kept_after_adjudication", "count": 0} in stats
    assert records[0]["meta"]["check"] == {"agree": 2, "solves": 4, "adjudication": ""}


# --------------------------------------------------------------------------- D-Report


ROW = {"question": "Explain how pore-pressure diffusion controls the timing of induced earthquakes.",
       "rubric_criteria": "explains diffusion and its time scale | relates injection rate to seismicity rate | "
                          "claims injection lubricates faults",
       "rubric_weights": "3|2|1", "reference_length_words": 550,
       "common_errors_to_penalize": "confuses pore pressure with total stress; says diffusion is instantaneous",
       "author": "", "date_written": "2026-10-15", "topic": "Induced seismicity", "status": "draft",
       "drafted_by": "gpt-6-luna-2026-09-22"}


def _verdict(question_ok=True, marks=("yes", "yes", "yes"), errors=("yes", "yes"), drop_id=None):
    crit = [{"id": i, "correct": m, "checkable": "yes", "issue": "" if m == "yes" else "wrong"}
            for i, m in enumerate(marks, 1) if i != drop_id]
    return json.dumps({"question": {"ok": question_ok, "issue": "" if question_ok else "needs a data lookup"},
                       "criteria": crit, "common_errors": [{"id": i, "really_an_error": e}
                                                           for i, e in enumerate(errors, 1)]})


def _row4():
    return {**ROW, "rubric_criteria": ROW["rubric_criteria"] + " | gives a field example", "rubric_weights": "3|2|1|1"}


def test_report_check_removes_rejected_criteria_and_errors():
    reply = "Let me think {about it}.\n" + _verdict(marks=("yes", "yes", "yes", "no"), errors=("yes", "no"))
    out = check_report_row(_row4(), Checker(lambda p: (reply, "stop")))
    assert out["status"] == "fixed"
    assert out["rubric_criteria"].split(" | ") == ["explains diffusion and its time scale",
                                                   "relates injection rate to seismicity rate",
                                                   "claims injection lubricates faults"]
    assert out["rubric_weights"] == "3|2|1"
    assert out["common_errors_to_penalize"] == "confuses pore pressure with total stress"
    assert "removed criterion: gives a field example" in out["check_issue"]


@pytest.mark.parametrize("reply, why", [
    (_verdict(question_ok=False), "question: needs a data lookup"),
    (_verdict(marks=("no", "yes", "yes")), "essential criterion rejected"),
    (_verdict(marks=("yes", "no", "yes")), "only 2 criteria passed"),
    (_verdict(drop_id=2), "unreadable_check: criteria missing"),
    ("I cannot decide.", "unreadable_check"),
])
def test_report_check_drops(reply, why):
    out = check_report_row(ROW, Checker(lambda p: (reply, "stop")))
    assert out["status"] == "drop" and out["check_issue"].startswith(why)


def _report_checker(verdicts):
    """Fake Gemma: its own answer for an answer prompt, the next verdict for a rubric prompt."""
    replies, prompts = iter(verdicts), []

    def answer(prompt):
        prompts.append(prompt)
        if "## Rubric" not in prompt:
            return "<think>hmm</think>\n<answer>My own answer about pore pressure.</answer>", "stop"
        return next(replies)
    return Checker(answer), prompts


def test_report_check_answers_first_and_the_majority_decides():
    chk, prompts = _report_checker([
        (_verdict(marks=("yes", "yes", "yes", "no"), errors=("yes", "no")), "stop"),   # criterion 4: 1 of 3 reject
        (_verdict(marks=("yes", "yes", "yes", "yes"), errors=("yes", "no")), "stop"),  # error 2: 2 of 3 reject
        (_verdict(marks=("yes", "yes", "yes", "yes"), errors=("yes", "yes")), "stop"),
    ])
    out = check_report_row(_row4(), chk, votes=3)
    assert out["status"] == "fixed" and len(out["rubric_criteria"].split(" | ")) == 4
    assert out["common_errors_to_penalize"] == "confuses pore pressure with total stress"
    assert out["check_votes"] == "3/3 readable; question 3/3; criteria 3,3,3,2; errors 3,1"
    asked = [p for p in prompts if "## Rubric" not in p]
    assert len(asked) == 3 and all("time scale" not in p and "550 words" in p for p in asked)  # blind to the rubric
    checks = [p for p in prompts if "## Rubric" in p]
    assert all("My own answer about pore pressure." in p and "hmm" not in p for p in checks)


def test_report_check_needs_two_readable_votes():
    rejecting = (_verdict(marks=("no", "yes", "yes")), "stop")
    chk, _ = _report_checker([("garbage", "stop"), ("garbage", "stop"), rejecting, rejecting])
    out = check_report_row(ROW, chk, votes=3)                     # first vote unreadable (twice): 2 of 2 decide
    assert out["status"] == "drop" and out["check_issue"].startswith("essential criterion rejected")
    assert out["check_votes"].startswith("2/3 readable")
    chk, _ = _report_checker([("garbage", "stop")] * 4 + [(_verdict(), "stop")])
    out = check_report_row(ROW, chk, votes=3)
    assert out["status"] == "drop" and out["check_issue"].startswith("unreadable_check")


def test_report_check_retries_a_truncated_reply_once():
    chk, _ = _report_checker([("", "length"), (_verdict(), "stop")])
    out = check_report_row(ROW, chk, votes=1)
    assert out["status"] == "ok" and chk.calls == 3 and out["truncated"] == 1     # answer + 2 checks


def test_report_records_carry_the_length_hint_into_the_prompt():
    checked = [{**ROW, "status": "ok", "check_issue": ""},
               {**ROW, "question": "Another question about seismic anisotropy in the mantle wedge.",
                "status": "drop", "check_issue": "question: vague"}]
    records, rows, stats = report_records(checked, n_keep=100)
    assert len(records) == 1
    r = records[0]
    assert r["answer"]["rubric"][0] == {"criterion": "explains diffusion and its time scale", "weight": 3}
    assert r["answer"]["reference_length_words"] == 550 and r["task"] == "t3_report"
    msgs = build_messages(EvalItem.from_record(r, "d_report"))
    assert msgs[-1]["content"].endswith("Aim for roughly 550 words.")
    assert {"stage": "dropped:question", "count": 1} in stats


# --------------------------------------------------------------------------- CLI


def test_cli_writes_a_verified_set_or_nothing(tmp_path, sandbox):
    drafts = [_draft(i, h=20.0 + i, topic=f"t{i % 5}") for i in range(45)]
    write_jsonl(tmp_path / "fresh_drafts.jsonl", drafts)

    def right_answer(prompt):                                       # Gemma solves every problem correctly
        d = next(d for d in drafts if d["question"] in prompt)
        return f"\\boxed{{{d['answer_value']:.6g} s}}", "stop"

    args = ["--model-line", "qwen35_9b", "fresh", "--drafts", str(tmp_path / "fresh_drafts.jsonl"),
            "--out", str(tmp_path / "secure" / "d_fresh.jsonl"), "--log", str(tmp_path / "check.csv")]
    rc = main(args, checker_factory=lambda temperature: Checker(right_answer),
              sandbox_factory=lambda allow_no_container: sandbox)
    assert rc == 0
    items = load_eval_items(tmp_path / "secure" / "d_fresh.jsonl", "d_fresh")      # manifest verified
    assert len(items) == 45 and all(it.answer["unit"] == "s" for it in items)
    assert (tmp_path / "check.csv").exists()

    never = Checker(lambda p: ("\\boxed{1 s}", "stop"))
    rc = main([*args[:-4], "--out", str(tmp_path / "secure" / "none.jsonl")],
              checker_factory=lambda temperature: never, sandbox_factory=lambda allow_no_container: sandbox)
    assert rc == 1 and not (tmp_path / "secure" / "none.jsonl").exists()        # < min_items: nothing written


def test_cli_report(tmp_path):
    rows = [{**ROW, "question": f"Question {i}: explain how {w} affects induced seismicity near injection wells.",
             "topic": f"t{i % 4}"} for i, w in enumerate(
        ["rate", "volume", "depth", "pressure", "permeability", "fault strike", "stress drop", "temperature",
         "salinity", "well spacing"] * 9)]
    rows = [{**r, "question": r["question"] + " " + " ".join(f"aspect{i}x{k}" for k in range(12))}
            for i, r in enumerate(rows)]
    write_csv(tmp_path / "d_report_drafts.csv", rows, REPORT_COLUMNS)
    out = tmp_path / "d_report.jsonl"
    rc = main(["--model-line", "qwen35_9b", "report", "--drafts", str(tmp_path / "d_report_drafts.csv"),
               "--out", str(out)], checker_factory=lambda temperature: Checker(lambda p: (_verdict(), "stop")))
    assert rc == 0
    recs = list(read_jsonl(out))
    assert len(recs) == 90 and (tmp_path / "d_report.jsonl.manifest.json").exists()


def test_cli_refuses_to_run_drafted_code_outside_the_container(tmp_path, monkeypatch):
    monkeypatch.delenv("DPT_SANDBOX_SIF", raising=False)
    write_jsonl(tmp_path / "f.jsonl", [_draft(0)])
    with pytest.raises(SystemExit, match="sandbox container"):
        main(["fresh", "--drafts", str(tmp_path / "f.jsonl"), "--out", str(tmp_path / "o.jsonl")],
             checker_factory=lambda temperature: None)


# --------------------------------------------------------------------------- resuming


def test_fresh_check_resumes_from_its_cache_and_code_checks_can_run_first(tmp_path, sandbox):
    from dpt.eval.autocheck import VerdictCache

    drafts = [_draft(i) for i in range(6)] + [_draft(6, code="def solve():\n    return 99.0\n")]
    right = Checker(lambda p: ("\\boxed{12.5 s}", "stop"))
    clean = {r["draft_id"]: r for r in check_fresh(drafts, right, sandbox, solves=2, max_concurrency=1)}

    cache_path = tmp_path / "c.jsonl"
    code = check_fresh(drafts, None, sandbox, code_only=True, cache=VerdictCache(cache_path, "draft_id"),
                       max_concurrency=1)                                      # on a compute node
    assert {r["status"] for r in code} == {"code_ok", "code_drop"}

    calls = {"n": 0}

    def flaky(prompt):
        calls["n"] += 1
        if calls["n"] > 5:
            raise KeyboardInterrupt                                           # the job is killed mid-run
        return "\\boxed{12.5 s}", "stop"

    with pytest.raises(KeyboardInterrupt):
        check_fresh(drafts, Checker(flaky), None, solves=2, max_concurrency=1,
                    cache=VerdictCache(cache_path, "draft_id"))               # no sandbox: code verdicts cached
    resumed = Checker(lambda p: ("\\boxed{12.5 s}", "stop"))
    out = check_fresh(drafts, resumed, None, solves=2, max_concurrency=1, cache=VerdictCache(cache_path, "draft_id"))
    rows = {r["draft_id"]: r for r in out}
    assert rows == clean
    assert resumed.calls == 2 * 6 - 4                                          # two drafts finished before the kill
    with pytest.raises(RuntimeError, match="no cached code verdict"):
        check_fresh([_draft(9)], resumed, None, cache=VerdictCache(cache_path, "draft_id"))


def test_report_check_resumes_from_its_cache(tmp_path):
    from dpt.eval.autocheck import VerdictCache, check_reports

    rows = [{**ROW, "question": f"{ROW['question']} Case {i}."} for i in range(3)]
    first = Checker(lambda p: (_verdict(), "stop"))
    a = check_reports(rows, first, votes=1, cache=VerdictCache(tmp_path / "r.jsonl", "question"))
    again = Checker(lambda p: (_verdict(), "stop"))
    b = check_reports(rows, again, votes=1, cache=VerdictCache(tmp_path / "r.jsonl", "question"))
    assert a == b and again.calls == 0
