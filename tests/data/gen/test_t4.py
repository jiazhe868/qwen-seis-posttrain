"""T4 general replay: plain-number math labels both graders accept; constraints both IFEval checkers read alike."""

from __future__ import annotations

import itertools
import json
import os
import re
from pathlib import Path

import pytest

from dpt.common.ifeval_schema import normalize_constraint
from dpt.common.io import read_jsonl
from dpt.data.gen import t4
from dpt.data.schema import Problem
from dpt.eval.graders.ifeval import CHECKS as EVAL_CHECKS
from dpt.eval.graders.ifeval import grade_ifeval
from dpt.eval.graders.numeric import grade_numeric
from dpt.rewards.instruction import CHECKERS as TRAIN_CHECKERS
from dpt.rewards.instruction import content_words, grade_instruction
from dpt.rewards.numeric import grade_math

RAW = Path(__file__).parent / "fixtures" / "t4_raw"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("t4") / "t4.jsonl"
    summary = t4.build(RAW, out, n_math=100, n_if=100, seed=1)
    return list(read_jsonl(out)), summary, out


# --------------------------------------------------------------------------- math labels


@pytest.mark.parametrize("text, value", [
    ("72", 72.0), ("1,200", 1200.0), ("2.5", 2.5), ("-3", -3.0), ("\\frac{3}{4}", 0.75), ("\\dfrac{3}{4}", 0.75),
    ("-\\frac{1}{2}", -0.5), ("\\frac{-1}{2}", -0.5), ("3/4", 0.75), ("10,\\!000", 10000.0), ("$42$", 42.0),
])
def test_plain_numbers_are_kept(text, value):
    assert t4.plain_number(text)[0] == pytest.approx(value)


@pytest.mark.parametrize("text", ["\\sqrt{2}", "(1,2)", "30^\\circ", "5\\text{ cm}", "x=3", "\\pi", "$2", "1\\frac{1}{2}",
                                  "\\frac{1}{0}", "0.\\overline{3}", ".5", "2,50", "1e20", "10000000000000000"])
def test_anything_else_is_dropped(text):
    assert t4.plain_number(text) is None


def test_math_labels_are_accepted_by_both_graders(built):
    rows, _, _ = built
    math_rows = [r for r in rows if r["answer"]["kind"] == "math"]
    assert {r["meta"]["t4_source"] for r in math_rows} == {"gsm8k", "math"}
    for r in math_rows:
        gt = r["answer"]
        assert gt["unit"] == "" and gt["rel_tol"] == 1e-6
        v = gt["value"]
        forms = [r["meta"]["answer_text"], str(int(v)) if v == int(v) else repr(v)]
        for form in forms:
            ans = f"\\boxed{{{form}}}"
            assert grade_math(ans, gt).score == 1.0, (form, gt)
            assert grade_numeric(ans, gt, missing_unit="strict").correct, (form, gt)
        wrong = f"\\boxed{{{v + 1}}}"
        assert grade_math(wrong, gt).score == 0.0 and not grade_numeric(wrong, gt, missing_unit="strict").correct


# --------------------------------------------------------------------------- build


def test_build_is_deterministic_licensed_and_counted(built, tmp_path):
    rows, summary, out = built
    again = t4.build(RAW, tmp_path / "again.jsonl", n_math=100, n_if=100, seed=1)
    assert (tmp_path / "again.jsonl").read_text() == out.read_text() and again["counts"] == summary["counts"]
    assert summary["counts"] == {"gsm8k": 4, "math": 6, "dolly": summary["counts"]["dolly"]}
    assert summary["drops"]["math:not_plain_number"] == 5 and summary["drops"]["dolly:duplicate"] == 1
    assert {"dolly:has_context", "dolly:category", "dolly:length", "dolly:url_or_non_ascii"} <= set(summary["drops"])
    assert Path(f"{out}.summary.json").exists()
    for r in rows:
        Problem.from_dict(r)
        src = r["source"]
        assert src["split"] == "train" and src["id"] and src["revision"] and r["meta"]["split"] == "train"
        assert src["license"] == {"dolly": "CC-BY-SA-3.0"}.get(r["meta"]["t4_source"], "MIT")
    other = t4.build(RAW, tmp_path / "seed2.jsonl", n_math=100, n_if=100, seed=2)
    assert other["counts"]["dolly"] >= summary["counts"]["dolly"] - 2      # a few may hit the relevance-gate drop
    assert (tmp_path / "seed2.jsonl").read_text() != out.read_text()        # other constraints drawn


def test_math_share_and_shortfall(tmp_path):
    s = t4.build(RAW, tmp_path / "m.jsonl", n_math=5, n_if=0, math_share=0.6)
    assert s["counts"] == {"math": 3, "gsm8k": 2}
    s = t4.build(RAW, tmp_path / "m2.jsonl", n_math=9, n_if=0, math_share=0.9)
    assert s["counts"] == {"math": 6, "gsm8k": 3}                              # MATH short: GSM8K fills in


# --------------------------------------------------------------------------- a response that meets the constraints

_LETTERS = "bdfgklmnprstvz"
BANK = ["zq" + "".join(p) for p in itertools.product(_LETTERS, "aeiou", _LETTERS)]  # distinct, never real words


class Writer:
    def __init__(self):
        self.words = iter(BANK)

    def sentence(self, *extra: str, n: int = 6) -> str:
        ws = list(extra) + [next(self.words) for _ in range(n)]
        return " ".join(ws).capitalize() + "."


def _counts(text: str) -> int:
    """The smaller of the two checkers' word counts."""
    from dpt.rewards.instruction import _count_words

    return min(_count_words(text), len(re.findall(r"\b[\w'-]+\b", text)))


def respond(prompt: str, constraints: list[dict]) -> str:
    """A response that satisfies ``constraints`` for base prompt ``prompt`` (test helper)."""
    cs = {c["type"]: normalize_constraint(c) for c in constraints}
    w = Writer()
    topic = sorted(content_words(prompt))
    first = [w.sentence(*topic)]
    if "include_keywords" in cs:
        first.append(w.sentence(*cs["include_keywords"]["value"]))
    if "keyword_frequency" in cs:
        k = cs["keyword_frequency"]
        first.append(w.sentence(*[x for _ in range(int(k["value"]) + 1) for x in (k["keyword"], next(w.words))]))
    if "placeholders" in cs:
        first.append(w.sentence(*[f"[{next(w.words)}]" for _ in range(int(cs["placeholders"]["value"]))]))
    if "highlighted_sections" in cs:
        first.append(w.sentence("note", *[f"*{next(w.words)}*" for _ in range(int(cs["highlighted_sections"]["value"]))]))
    body = list(first)
    need_sent = int(cs["sentences"]["value"]) + 1 if "sentences" in cs else 0
    need_par = int(cs.get("paragraphs", cs.get("sections", {})).get("value", 1))
    while len(body) < max(need_sent, need_par):
        body.append(w.sentence())
    if "min_words" in cs:
        while _counts(" ".join(body)) < int(cs["min_words"]["value"]) + 5:
            body.append(w.sentence())

    if "json_format" in cs:
        text = json.dumps({"answer": " ".join(body)})
    elif "bullets" in cs:
        n = int(cs["bullets"]["value"])
        text = " ".join(body) + "\n\n" + "\n".join(f"* {w.sentence(n=3)}" for _ in range(n))
    elif "sections" in cs:
        n, marker = int(cs["sections"]["value"]), cs["sections"].get("marker", "Section")
        chunks = [body[i::n] for i in range(n)]
        text = "\n\n".join(f"{marker} {i + 1}\n" + " ".join(ch) for i, ch in enumerate(chunks))
    elif "paragraphs" in cs:
        n = int(cs["paragraphs"]["value"])
        chunks = [body[i::n] for i in range(n)]
        text = "\n\n".join(" ".join(ch) for ch in chunks)
    else:
        text = " ".join(body)
    if "title_in_double_angular" in cs:
        text = f"<<{next(w.words)} {next(w.words)}>>\n\n" + text
    if "start_with" in cs:
        sep = "\n" if "sections" in cs else " "
        text = cs["start_with"]["value"] + sep + text
    if "postscript" in cs:
        text += f"\n\n{cs['postscript'].get('marker', 'P.S.')} " + w.sentence(n=4)
    if "end_with" in cs:
        text += " " + cs["end_with"]["value"]
    if "all_lowercase" in cs:
        text = text.lower()
    if "all_uppercase" in cs:
        text = text.upper()
    if "quotation" in cs:
        text = f'"{text}"'
    if "max_words" in cs:
        assert _counts(text) <= int(cs["max_words"]["value"]) - 3, "test helper: too long for max_words"
    return text


def violate(c: dict, prompt: str) -> str:
    """A response that breaks constraint ``c`` (and is otherwise on topic)."""
    c = normalize_constraint(c)
    w = Writer()
    topic = w.sentence(*sorted(content_words(prompt)))
    t, v = c["type"], c.get("value")
    return {
        "min_words": lambda: w.sentence(n=3),
        "max_words": lambda: " ".join([topic] + [w.sentence() for _ in range(int(v) // 4)]),
        "sentences": lambda: topic,
        "paragraphs": lambda: "\n\n".join([topic] + [w.sentence() for _ in range(int(v))]),
        "bullets": lambda: topic + "\n\n" + "\n".join(f"* {w.sentence()}" for _ in range(int(v) - 1)),
        "sections": lambda: "\n\n".join(f"Section {i + 1}\n{w.sentence()}" for i in range(int(v) + 1)),
        "include_keywords": lambda: w.sentence(),
        "keyword_frequency": lambda: w.sentence(c["keyword"]),
        "forbid_keywords": lambda: topic + " " + w.sentence(v[0]),
        "json_format": lambda: topic,
        "end_with": lambda: topic + " " + c["value"] + " " + w.sentence(),
        "start_with": lambda: topic + " " + c["value"],
        "all_lowercase": lambda: topic,
        "all_uppercase": lambda: topic,
        "no_commas": lambda: topic + " " + w.sentence() + ", " + w.sentence(),
        "title_in_double_angular": lambda: topic,
        "placeholders": lambda: topic,
        "quotation": lambda: topic,
        "postscript": lambda: topic,
        "highlighted_sections": lambda: topic,
    }[t]()


def _both(resp: str, question: str, answer: dict) -> tuple[bool, bool]:
    train = grade_instruction(resp, answer, prompt=question)
    return train.score == 1.0, grade_ifeval(resp, answer).correct


# --------------------------------------------------------------------------- constraints

PROMPT = "Describe the water cycle and explain why evaporation matters for farming communities."


@pytest.mark.parametrize("ctype", t4.TYPES)
def test_each_constraint_type_reads_the_same_in_both_checkers(ctype):
    import random

    c = t4.make_constraint(ctype, random.Random(3), PROMPT, [])
    assert c is not None and c["type"] in t4.TYPES and "language" != c["type"]
    q = t4.if_question(PROMPT, [c])
    answer = {"kind": "ifeval", "constraints": [c]}
    assert _both(respond(PROMPT, [c]), q, answer) == (True, True)
    bad = violate(c, PROMPT)
    cn = normalize_constraint(c)
    assert not TRAIN_CHECKERS[ctype](bad, cn) and not EVAL_CHECKS[ctype](bad, cn), (ctype, bad)
    assert _both(bad, q, answer) == (False, False)


def test_language_and_exact_word_counts_are_never_emitted():
    assert "language" not in t4.TYPES and "words" not in t4.TYPES


def test_drawn_combinations_hold_in_both_checkers():
    import random

    prompts = [r["instruction"] for r in read_jsonl(RAW / "dolly.jsonl") if t4.base_prompt_ok(r) is None]
    seen_pairs = set()
    for i in range(400):
        rng = random.Random(i)
        prompt = prompts[i % len(prompts)]
        cs = t4.draw_constraints(prompt, rng, 3)
        types = [c["type"] for c in cs]
        assert len(set(types)) == len(types)
        assert all(frozenset(p) not in t4.CONFLICTS for p in itertools.combinations(types, 2)), types
        if t4._COUNT_OR_FORMAT.search(prompt):
            assert set(types) <= set(t4.SAFE_FOR_FORMATTED), (prompt, types)
        q = t4.if_question(prompt, cs)
        try:
            resp = respond(prompt, cs)
        except AssertionError:                                           # helper cannot fit max_words here
            resp = None
        if resp is not None:
            assert _both(resp, q, {"kind": "ifeval", "constraints": cs}) == (True, True), (prompt, cs, resp)
        seen_pairs |= {frozenset(p) for p in itertools.combinations(types, 2)}
    assert len(seen_pairs) > 40                                          # the test saw many combinations


def test_built_items_can_pass_the_relevance_gate(built):
    rows, summary, _ = built
    ifs = [r for r in rows if r["answer"]["kind"] == "ifeval"]
    assert len(ifs) == summary["counts"]["dolly"] > 15
    for r in ifs:
        base, cs = r["meta"]["base_prompt"], r["answer"]["constraints"]
        assert r["question"].startswith(base)
        if not any(c["type"] == "json_format" for c in cs):
            assert t4.relevance_share(base, r["question"]) >= t4.RELEVANCE_MIN_SHARE
        if not any(c["type"] == "max_words" for c in cs):
            assert _both(respond(base, cs), r["question"], r["answer"]) == (True, True), r


def test_formatted_prompts_only_get_safe_constraints():
    import random

    for i in range(50):
        cs = t4.draw_constraints("Name five birds that live in Antarctica.", random.Random(i), 3)
        assert {c["type"] for c in cs} <= set(t4.SAFE_FOR_FORMATTED)


# --------------------------------------------------------------------------- fetch / CLI


class FakeApi:
    def __init__(self):
        self.files = {"openai/gsm8k": ["main/train-00000-of-00001.parquet", "main/test-00000-of-00001.parquet",
                                       "socratic/train-00000-of-00001.parquet"],
                      "EleutherAI/hendrycks_math": ["algebra/train-00000-of-00001.parquet",
                                                    "algebra/test-00000-of-00001.parquet", "README.md"],
                      "databricks/databricks-dolly-15k": ["databricks-dolly-15k.jsonl", "README.md"]}

    def dataset_info(self, repo):
        return type("Info", (), {"sha": f"sha-{repo.split('/')[0]}"})()

    def list_repo_files(self, repo, repo_type, revision):
        assert repo_type == "dataset" and revision.startswith("sha-")
        return self.files[repo]


def test_fetch_takes_train_splits_and_records_revisions(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    up = tmp_path / "upstream"
    up.mkdir()
    (up / "gsm.jsonl").write_text(json.dumps({"question": "Q?", "answer": "x\n#### 3"}) + "\n")
    (up / "math.jsonl").write_text(json.dumps({"problem": "P?", "solution": "$\\boxed{4}$", "level": "Level 1",
                                               "type": "Algebra"}) + "\n")
    (up / "dolly.jsonl").write_text(json.dumps({"instruction": "Why?", "context": "", "response": "r",
                                                "category": "open_qa"}) + "\n")
    asked = []

    def download(repo, filename, repo_type, revision):
        asked.append((repo, filename, revision))
        return str(up / {"openai": "gsm.jsonl", "EleutherAI": "math.jsonl", "databricks": "dolly.jsonl"}[repo.split("/")[0]])

    raw = tmp_path / "raw"
    man = t4.fetch(raw, api=FakeApi(), download=download)
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"
    assert sorted(f for _, f, _ in asked) == ["algebra/train-00000-of-00001.parquet", "databricks-dolly-15k.jsonl",
                                              "main/train-00000-of-00001.parquet"]          # no test splits
    assert man["gsm8k"]["revision"] == "sha-openai" and man["dolly"]["license"] == "CC-BY-SA-3.0"
    assert next(iter(read_jsonl(raw / "math.jsonl")))["id"] == "math/algebra/train-00000-of-00001/0"
    assert json.loads((raw / "fetch_manifest.json").read_text())["math"]["rows"] == 1


def test_cli_build(tmp_path):
    out = tmp_path / "c.jsonl"
    assert t4.main(["build", "--raw-dir", str(RAW), "--out", str(out), "--n-math", "4", "--n-if", "3"]) == 0
    rows = list(read_jsonl(out))
    assert len(rows) == 7 and json.loads(Path(f"{out}.summary.json").read_text())["total"] == 7
