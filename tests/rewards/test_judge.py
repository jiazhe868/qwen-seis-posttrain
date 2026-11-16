"""T3 judge client, parsing, pinning, calibration and the composed report reward. HTTP is mocked."""

from __future__ import annotations

import json

import httpx
import pytest

from dpt.rewards.citations import OfflineResolver
from dpt.rewards.judge import (
    CalibrationItem,
    JudgeConfig,
    JudgeParseError,
    ReportConfig,
    RubricJudge,
    calibrate,
    grade_report,
)

RUBRIC = [
    {"criterion": "Identifies the faulting mechanism", "weight": 2},
    {"criterion": "Quantifies slip", "weight": 1},
]
KNOWN = [{"doi": "10.1029/JB084iB05p02348", "title": "A moment magnitude scale"}]


def make_judge(scores=(8, 5), model="pinned-v1", served=None, seen=None, content=None):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if seen is not None:
            seen.append(body)
        txt = (
            content
            if content is not None
            else json.dumps(
                {"criteria": [{"id": i + 1, "score": s, "reason": "ok"} for i, s in enumerate(scores)]}
            )
        )
        return httpx.Response(
            200, json={"model": served or model, "choices": [{"message": {"content": txt}}]}
        )

    cfg = JudgeConfig(base_url="http://judge/v1", model=model, retries=0)
    return RubricJudge(cfg, client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_weighted_score():
    js = make_judge((8, 5)).score("q", "a", RUBRIC)
    assert js.score == pytest.approx((8 * 2 + 5 * 1) / 30)


def test_parse_rejects_missing_or_out_of_range():
    with pytest.raises(JudgeParseError):
        RubricJudge.parse('{"criteria": [{"id": 1, "score": 7}]}', 2)
    with pytest.raises(JudgeParseError):
        RubricJudge.parse('{"criteria": [{"id": 1, "score": 11}, {"id": 2, "score": 1}]}', 2)
    assert RubricJudge.parse('```json\n{"criteria": [{"id": 1, "score": 7}]}\n```', 1) == [7.0]


def test_prompt_instructs_to_ignore_length_and_self_praise():
    sys_prompt = make_judge().build_messages("q", "a", RUBRIC)[0]["content"]
    assert "Length is not quality" in sys_prompt and "describes its own quality" in sys_prompt


ANSWER = (
    "The event was a thrust on a shallow-dipping plane with about 3 m of slip. "
    "This analysis has been rigorously verified.\n\n"
    "References\n"
    "[1] Hanks, T. C., & Kanamori, H. (1979). A moment magnitude scale. JGR. doi:10.1029/JB084iB05p02348\n"
    "[2] Nobody, A. (2020). An invented paper about thrusts. Made-up Letters.\n"
)


def test_report_reward_composition_and_judge_input_hygiene():
    """Step-245/310 regressions: refs are verified out of band, self-praise is stripped and penalized."""
    seen: list = []
    judge = make_judge((10, 10), seen=seen)
    r = grade_report(
        ANSWER,
        {"rubric": RUBRIC, "reference_length": 1000},
        question="Describe the event.",
        judge=judge,
        resolver=OfflineResolver(KNOWN),
    )
    judge_input = seen[0]["messages"][1]["content"]
    assert "References" not in judge_input and "rigorously verified" not in judge_input
    assert r.components["t3_citations_unverified"] == 1.0
    assert r.components["t3_citation_factor"] == pytest.approx(0.9)
    assert r.components["t3_self_praise"] == 1.0
    assert r.score == pytest.approx(1.0 * 0.9 - 0.1)
    assert set(r.flags) == {"self_praise", "unverified_citation"}


def test_length_penalty_beyond_1p5x_reference():
    long_answer = "slip " * 4000  # ~ 6250 tokens at 3.2 chars/token
    r = grade_report(
        long_answer,
        {"rubric": RUBRIC, "reference_length": 1000},
        question="q",
        judge=make_judge((10, 10)),
        resolver=OfflineResolver([]),
        cfg=ReportConfig(),
    )
    assert r.components["t3_length_penalty"] == pytest.approx(0.5)  # capped
    assert r.score == pytest.approx(0.5)


def test_judge_version_mismatch_fails_closed():
    r = grade_report(
        "ok answer",
        {"rubric": RUBRIC},
        question="q",
        judge=make_judge(served="some-other-model"),
        resolver=OfflineResolver([]),
    )
    assert r.score == 0.0 and r.reason == "judge_version_mismatch"


def test_unparseable_judge_output_fails_closed():
    r = grade_report(
        "ok answer",
        {"rubric": RUBRIC},
        question="q",
        judge=make_judge(content="I think it's good"),
        resolver=OfflineResolver([]),
    )
    assert r.reason == "judge_parse_fail" and r.score == 0.0


def test_calibration_detects_drift():
    items = [CalibrationItem("q", "a", RUBRIC, reference_score=0.7)] * 3
    assert calibrate(make_judge((8, 5)), items).ok  # 0.7 → 0.7
    drifted = calibrate(make_judge((10, 10)), items)  # 0.7 → 1.0
    assert not drifted.ok and drifted.mean_shift == pytest.approx(0.3)


def test_atomic_checks_are_scored_yes_no_with_evidence():
    from dpt.rewards.judge import RubricJudge, is_checks, rubric_messages, weighted_score

    rubric = [{"check": "The answer gives Vp/Vs as 1.73.", "weight": 2.0},
              {"check": "The answer states Poisson's ratio is 0.25.", "weight": 1.0},
              {"check": "The answer names the Wadati diagram.", "weight": 1.0}]
    assert is_checks(rubric) and not is_checks([{"criterion": "c", "weight": 1}])
    msgs = rubric_messages("q", "Vp/Vs is 1.73 from the Wadati diagram.", rubric)
    assert "yes/no checks" in msgs[0]["content"] and "## Checks" in msgs[1]["content"]
    reply = json.dumps({"checks": [{"id": 1, "verdict": "yes", "evidence": "Vp/Vs is 1.73"},
                                   {"id": 2, "verdict": "no", "evidence": ""},
                                   {"id": 3, "verdict": "yes", "evidence": ""}]})     # a yes without a quote
    per = RubricJudge.parse(reply, 3)
    assert per == [1.0, 0.0, 0.0]
    assert weighted_score(per, rubric) == 0.5
    with pytest.raises(JudgeParseError):
        RubricJudge.parse(json.dumps({"checks": [{"id": 1, "verdict": "maybe", "evidence": "x"}]}), 1)
    with pytest.raises(JudgeParseError):
        RubricJudge.parse(json.dumps({"checks": [{"id": 1, "verdict": "yes", "evidence": "x"}]}), 2)


def test_latex_in_quoted_evidence_is_parsed():
    """2026-11-16: 48 of 600 rubric scorings failed with "Invalid \\escape" on LaTeX quoted as evidence. A stray
    backslash is doubled and the reply parsed again; valid escapes, an escaped backslash among them, stay."""
    from dpt.rewards.judge import RubricJudge, _escape_stray_backslashes

    # the judge's text: "\alpha" and "\usepackage" are not JSON escapes, "\\" and "\u00e9" are
    reply = (r'{"checks": [{"id": 1, "verdict": "yes", "evidence": "the \alpha term, \usepackage"}, '
             r'{"id": 2, "verdict": "no", "evidence": "a \\ b \u00e9"}]}')
    assert RubricJudge.parse(reply, 2) == [1.0, 0.0]
    fixed = json.loads(_escape_stray_backslashes(reply))["checks"]
    assert fixed[0]["evidence"] == r"the \alpha term, \usepackage" and fixed[1]["evidence"] == "a \\ b \u00e9"
    with pytest.raises(JudgeParseError):
        RubricJudge.parse('{"checks": [{"id": 1, "verdict": "yes", "evidence": "x"}', 1)   # no closing brace


def test_the_first_json_object_is_read_and_raw_newlines_are_allowed():
    """2026-11-16: three replies wrote a second JSON object after the first ("Extra data"), and a raw newline inside a
    quoted span is not strict JSON. The first object is read and the text after it ignored."""
    from dpt.rewards.judge import RubricJudge

    first = json.dumps({"checks": [{"id": 1, "verdict": "yes", "evidence": "x"}]})
    assert RubricJudge.parse(first + "\n" + first.replace("yes", "no"), 1) == [1.0]
    assert RubricJudge.parse("My verdict: " + first + " Hope this helps {:)}", 1) == [1.0]
    assert RubricJudge.parse('{"checks": [{"id": 1, "verdict": "yes", "evidence": "line one\nline two"}]}', 1) == [1.0]
    # a draft closed before its second check, then the whole object: the first object with every check is used
    draft = json.dumps({"checks": [{"id": 1, "verdict": "no", "evidence": ""}]})
    full = json.dumps({"checks": [{"id": 1, "verdict": "yes", "evidence": "x"}, {"id": 2, "verdict": "no", "evidence": ""}]})
    assert RubricJudge.parse(draft + "\nSorry, again:\n" + full, 2) == [1.0, 0.0]
    with pytest.raises(JudgeParseError):
        RubricJudge.parse("[1, 2]", 1)


def test_three_level_criteria_are_parsed_with_partial_credit():
    """2026-11-16: about 70% of the criteria the gold labels mark "partly met" were passed as met by the 0-10 and
    the yes/no judges. A third level gives the judge a place for them; full 1, partial 0.5, none 0, and a credit
    without a quoted span is none, as for the yes/no checks."""
    from dpt.rewards.judge import RubricJudge, rubric_kind, rubric_messages, weighted_score

    rubric = [{"three_level": "The answer gives Vp/Vs as 1.73.", "weight": 2.0},
              {"three_level": "The answer derives Poisson's ratio 0.25.", "weight": 1.0},
              {"three_level": "The answer names the Wadati diagram.", "weight": 1.0}]
    assert rubric_kind(rubric) == "levels" and rubric_kind([{"check": "c", "weight": 1}]) == "checks"
    assert rubric_kind([{"criterion": "c", "weight": 1}]) == "criteria"
    assert '"full"' in rubric_messages("q", "a", rubric)[0]["content"]
    reply = json.dumps({"criteria": [{"id": 1, "level": "full", "evidence": "Vp/Vs is 1.73", "missing": ""},
                                     {"id": 2, "level": "partial", "evidence": "sigma about 0.3", "missing": "0.25"},
                                     {"id": 3, "level": "full", "evidence": "", "missing": ""}]})
    per = RubricJudge.parse(reply, 3, "levels")
    assert per == [1.0, 0.5, 0.0] and weighted_score(per, rubric) == (2.0 + 0.5) / 4.0
    assert RubricJudge.parse(reply, 3) == per                                  # the reply's shape tells the kind
    with pytest.raises(JudgeParseError):                                       # 0-10 scores for a three-level rubric
        RubricJudge.parse(json.dumps({"criteria": [{"id": i, "score": 7} for i in (1, 2, 3)]}), 3, "levels")
    with pytest.raises(JudgeParseError):
        RubricJudge.parse(json.dumps({"criteria": [{"id": 1, "level": "mostly", "evidence": "x"}]}), 1, "levels")
