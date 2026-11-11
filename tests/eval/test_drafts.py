"""Eval drafting with the hosted model: topics, D-Report rubrics, D-Fresh problems with solver code, D-Num wording."""

from __future__ import annotations

import csv
import json

import httpx
import pytest

from dpt.common.io import write_jsonl
from dpt.eval.datasets import EvalItem
from dpt.eval.drafts import (
    REPORT_COLUMNS,
    draft_fresh,
    draft_reports,
    draft_topics,
    main,
    near_duplicate,
    parse_json_reply,
    refuse_fresh,
    review_wording,
    write_csv,
)
from dpt.eval.hosted import ChatEndpoint


def _chat(reply_for):
    def handler(req):
        body = json.loads(req.content)
        content, fin = reply_for(body["messages"][-1]["content"])
        return httpx.Response(200, json={"model": "gpt-6-luna-2026-09-22",
                                         "choices": [{"message": {"content": content}, "finish_reason": fin}]})

    return ChatEndpoint("https://api.example.com", "gpt-6-luna", temperature=None,
                        max_tokens_field="max_completion_tokens", extra_body={"reasoning_effort": "xhigh"},
                        client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_parse_json_reply_tolerates_fences_and_text():
    assert parse_json_reply('Here:\n```json\n{"a": 1}\n```\nDone') == {"a": 1}
    assert parse_json_reply('prefix {"b": [1, 2]} suffix') == {"b": [1, 2]}
    with pytest.raises(ValueError):
        parse_json_reply("no json")


def test_topics_deduplicated():
    reply = {"topics": [{"topic": "Induced seismicity from fluid injection", "scope": "s"},
                        {"topic": "Induced seismicity from fluid injection.", "scope": "dup"},
                        {"topic": "Seismic anisotropy in the upper mantle", "scope": "s2"}]}
    topics = draft_topics(_chat(lambda m: (json.dumps(reply), "stop")), "seismology", 10)
    assert [t["topic"] for t in topics] == ["Induced seismicity from fluid injection",
                                            "Seismic anisotropy in the upper mantle"]


def test_report_rows_are_sized_for_short_reports(tmp_path):
    def reply(msg):
        good = {"question": "Write a technical report on how pore pressure diffusion controls the timing "
                            "of induced earthquakes near an injection well.",
                "rubric": [{"criterion": "explains diffusion | time scale", "weight": 3},
                           {"criterion": "relates rate to seismicity", "weight": "2"},
                           {"criterion": "discusses fault stress", "weight": 9},
                           {"criterion": "gives a field example", "weight": "x"}],
                "reference_length_words": 900, "common_errors": ["lubrication myth"]}
        bad = {"question": "Too short?", "rubric": [{"criterion": "c", "weight": 1}]}
        dup = {**good, "question": good["question"] + " Be concise."}
        assert "500-600 words" in msg and "no person edits them" in msg
        return json.dumps({"questions": [good, bad, dup]}), "stop"

    rows, counts = draft_reports(_chat(reply), [{"topic": "Induced seismicity", "scope": "s"}], 3,
                                 date="2026-10-14")
    assert counts == {"kept": 1, "near_duplicate": 1, "malformed": 1, "failed_topics": 0}
    r = rows[0]
    assert r["rubric_criteria"].split(" | ")[0] == "explains diffusion / time scale"   # '|' is the separator
    assert r["rubric_weights"] == "3|2|3|1"                                         # clamped to 1..3
    assert (r["status"], r["drafted_by"], r["author"]) == ("draft", "gpt-6-luna-2026-09-22", "")
    assert r["reference_length_words"] == 550                           # the draft's own length is ignored
    write_csv(tmp_path / "d.csv", rows, REPORT_COLUMNS)
    assert next(csv.DictReader(open(tmp_path / "d.csv")))["topic"] == "Induced seismicity"


def test_failed_topic_does_not_lose_the_others():
    def reply(msg):
        return ("not json", "stop") if "Broken" in msg else (json.dumps({"questions": []}), "stop")

    _, counts = draft_reports(_chat(reply), [{"topic": "Broken"}, {"topic": "Fine"}], 2)
    assert counts["failed_topics"] == 1


def test_wording_review_flags_disagreement_and_ambiguity():
    items = [EvalItem("a", "d_num", "P wave, 120 km at 6 km/s: travel time?", {"value": 20, "unit": "s"}),
             EvalItem("b", "d_num", "Q-wrong", {"value": 20, "unit": "s"}),
             EvalItem("c", "d_num", "Q-ambiguous", {"value": 20, "unit": "s"}),
             EvalItem("d", "d_num", "Q-long", {"value": 20, "unit": "s"})]

    def reply(msg):
        if "Q-wrong" in msg:
            return "t = 25 s\nWORDING: clear\n\\boxed{25 s}", "stop"
        if "Q-ambiguous" in msg:
            return "WORDING: ambiguous - depth not given\n\\boxed{20 s}", "stop"
        if "Q-long" in msg:
            return "", "length"
        return "t = d/v\nWORDING: clear\n\\boxed{20 s}", "stop"

    rows = {r["id"]: r for r in review_wording(_chat(reply), items)}
    assert rows["a"]["flag"] == "ok" and rows["a"]["agrees_with_label"] == "yes"
    assert (rows["b"]["flag"], rows["b"]["agrees_with_label"]) == ("check", "no")
    assert (rows["c"]["wording"], rows["c"]["issue"]) == ("ambiguous", "depth not given")
    assert rows["d"]["flag"] == "unanswered"


def test_fresh_problems_keep_answer_and_solver_code():
    good = {"question": "A P wave crosses a 30 km crust at 6.0 km/s and then 60 km of mantle at 8.0 km/s. "
                        "What is the total travel time in seconds?",
            "answer_value": 12.5, "answer_unit": "s", "rel_tolerance_percent": 0.01, "solution": "5 + 7.5",
            "solve_py": "def solve():\n    return 30 / 6.0 + 60 / 8.0\n"}
    no_code = {**good, "question": good["question"] + " Different.", "solve_py": "print(12.5)"}
    no_unit = {**good, "question": "Another long enough question about mantle travel times for P waves here.",
               "answer_unit": ""}

    def reply(msg):
        assert "solve()" in msg and "at least two dependent steps" in msg
        return json.dumps({"problems": [good, no_code, no_unit]}), "stop"

    rows, counts = draft_fresh(_chat(reply), [{"topic": "Travel times", "scope": "s"}], 3, date="2026-10-15")
    assert counts == {"kept": 1, "near_duplicate": 0, "malformed": 2, "failed_topics": 0}
    r = rows[0]
    assert (r["answer_value"], r["answer_unit"], r["rel_tolerance_percent"]) == (12.5, "s", 0.1)   # tol ≥ 0.1%
    assert r["solve_py"].startswith("def solve") and len(r["draft_id"]) == 12


def test_the_drafting_model_never_checks_d_fresh(tmp_path):
    with pytest.raises(SystemExit, match="autocheck fresh"):
        refuse_fresh(tmp_path / "d_fresh.jsonl", [])
    with pytest.raises(SystemExit):
        refuse_fresh(tmp_path / "x.jsonl", [EvalItem("i", "d_fresh", "q", {})])


def test_cli_refuses_without_a_usable_hosted_model(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    write_jsonl(tmp_path / "d_num.jsonl", [{"id": "a", "question": "q", "answer": {"value": 1}}])
    with pytest.raises(SystemExit, match="not usable here"):
        main(["--model-line", "qwen35_9b", "wording", "--items", str(tmp_path / "d_num.jsonl"),
              "--out", str(tmp_path / "w.csv")])


def test_near_duplicate():
    assert near_duplicate("the quick brown fox", "The quick brown fox!")
    assert not near_duplicate("seismic moment tensor", "gravity anomaly inversion")


def test_regrade_wording_uses_the_saved_answers_without_calling_the_api():
    from dpt.eval.datasets import EvalItem
    from dpt.eval.drafts import regrade_wording

    items = [EvalItem(id=f"i{k}", set="d_num", question="q", answer={"value": v, "unit": u, "rel_tol": 0.01})
             for k, (v, u) in enumerate([(7.97, ""), (1326.5, "yr"), (5.0, "s"), (2.0, "km")])]
    rows = [
        {"id": "i0", "flag": "check", "agrees_with_label": "unit_parse_error", "wording": "clear",
         "hosted_answer": "7.97\\ \\text{unitless}"},                       # grader gap, now fixed
        {"id": "i1", "flag": "check", "agrees_with_label": "parse_error", "wording": "clear",
         "hosted_answer": "1.33\\times10^3\\ \\text{yr}"},                 # env without sympy at the time
        {"id": "i2", "flag": "check", "agrees_with_label": "yes", "wording": "ambiguous", "hosted_answer": "5 s"},
        {"id": "i3", "flag": "error", "agrees_with_label": "", "wording": "", "hosted_answer": ""},
    ]
    out = {r["id"]: r for r in regrade_wording(rows, items)}
    assert out["i0"]["flag"] == "ok" and out["i1"]["flag"] == "ok"
    assert out["i2"]["flag"] == "check"                                     # wording verdict is kept
    assert out["i3"] == rows[3]                                             # nothing to regrade
