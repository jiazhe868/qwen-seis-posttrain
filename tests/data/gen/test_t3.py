"""T3 report prompts drafted by the teacher: topics, items with rubrics, dedup, resumability, spot check."""

from __future__ import annotations

import csv
import json
import re

import pytest

from dpt.chat.thinking import THINK_END
from dpt.common.io import read_jsonl
from dpt.data.gen.t3 import build, clean_item, main, parse_items, reference_length
from dpt.data.gen.teacher import Reply, last_json
from dpt.data.prompts import messages_for_problem
from dpt.data.schema import Problem


def _letters(n):
    """A word of letters only (the dedup tokenizer splits letters from digits)."""
    out = ""
    while True:
        n, r = divmod(n, 26)
        out += "abcdefghijklmnopqrstuvwxyz"[r]
        if not n:
            return out + "q"


def _rubric(i):
    return [{"criterion": f"explains mechanism {i} of the process", "weight": 3},
            {"criterion": f"gives the time scale number {i}", "weight": 2},
            {"criterion": f"mentions observational evidence {i}", "weight": 1}]


class FakeTeacher:
    """Answers the topic prompt and the item prompts with thinking first (holding a draft JSON)."""

    model_id = "Qwen/Qwen3.8-27B"
    effort = "medium"
    tokenizer = None

    def __init__(self, n_topics=3, fail=False, start=0):
        self.n_topics, self.calls, self.fail, self.counter = n_topics, 0, fail, start

    def generate(self, messages, *, n=1, seed=None, max_tokens=None, temperature=None):
        if self.fail:
            raise AssertionError("no request expected")
        self.calls += 1
        prompt = messages[-1]["content"]
        if "List" in prompt and "topics" in prompt:
            body = {"topics": [{"topic": f"Topic number {t} of seismology", "scope": "A scope."}
                               for t in range(self.n_topics)]}
        else:
            topic = re.search(r"Topic: (.*)", prompt).group(1)
            k = int(re.search(r"Write (\d+) questions", prompt).group(1))
            items = []
            for _ in range(k):
                self.counter += 1
                c = self.counter
                words = " ".join(_letters(c * 100 + j) for j in range(14))
                items.append({"question": f"Explain for {topic} how {words} controls the outcome?",
                              "rubric": _rubric(c)})
            items.append({"question": "Too short?", "rubric": _rubric(0)})                    # invalid
            items.append({"question": items[0]["question"] + " Please.", "rubric": _rubric(0)})   # near-duplicate
            body = {"items": items}
        text = f'thinking {{"items": "draft"}} {{"topics": []}}\n{THINK_END}\n\n{json.dumps(body)}'
        return [Reply(text, "stop")]


def test_last_json_reads_the_final_answer_not_the_draft():
    text = f'draft {{"items": [1]}}\n{THINK_END}\n\nfinal: {{"items": [2]}}'
    assert last_json(text, "items") == [2]
    with pytest.raises(ValueError):
        last_json("nothing", "items")


def test_clean_item_rules():
    q = "Explain how pore-pressure diffusion controls the delay between injection and induced earthquakes?"
    it = clean_item({"question": q, "rubric": [
        {"criterion": "explains diffusion and its time scale", "weight": 3.0},
        {"criterion": "explains diffusion and its time scale", "weight": 2},        # duplicate
        {"criterion": "too short", "weight": 2},                                     # < 3 words
        {"criterion": "relates the injection rate to seismicity", "weight": "2"},
        {"criterion": "weight out of range criterion here", "weight": 5},
        {"criterion": "notes the role of permeability", "weight": 1}]})
    assert [c["weight"] for c in it["rubric"]] == [3, 2, 1]
    assert clean_item({"question": "Short question?", "rubric": _rubric(1)}) is None
    assert clean_item({"question": q, "rubric": _rubric(1)[:2]}) is None
    many = [{"criterion": f"criterion number {i} here", "weight": 2} for i in range(9)]
    assert len(clean_item({"question": q, "rubric": many})["rubric"]) == 6
    with pytest.raises(ValueError):
        parse_items(f"{THINK_END}\n\nnot json")


def test_build_drafts_dedupes_and_writes_valid_candidates(tmp_path):
    teacher = FakeTeacher(n_topics=3)
    out, spot = tmp_path / "t3.jsonl", tmp_path / "spot.csv"
    stats = build(teacher, tmp_path / "work", out, n_topics=3, per_topic=12, batch=5, concurrency=2,
                  spot_check=spot)
    rows = list(read_jsonl(out))
    assert stats["candidates"] == len(rows) == 36                         # 3 topics × 12
    qs = [r["question"] for r in rows]
    assert len(set(qs)) == len(qs) and not any(q.endswith("Please.") for q in qs)
    p = Problem.from_dict(rows[0])
    assert p.task == "t3_report" and p.answer["reference_length"] == reference_length() == 1031
    assert p.answer["prompt_extra"] == {"length_hint": 550} and 3 <= len(p.answer["rubric"]) <= 6
    assert p.source["license"] == "internal-generated" and p.source["model"] == "Qwen/Qwen3.8-27B"
    assert p.meta["topic"].startswith("Topic number") and p.meta["split"] == "train"
    assert messages_for_problem(p)[-1]["content"].endswith("Aim for roughly 550 words.")
    with open(spot) as f:
        assert len(list(csv.DictReader(f))) == 36                            # fewer than 50 → all
    # resumable: everything is drafted, so a rerun makes no request and writes the same set
    build(FakeTeacher(fail=True), tmp_path / "work", tmp_path / "again.jsonl", n_topics=3, per_topic=12, batch=5)
    assert [r["uid"] for r in read_jsonl(tmp_path / "again.jsonl")] == [r["uid"] for r in rows]


def test_build_continues_a_partial_run(tmp_path):
    build(FakeTeacher(n_topics=2), tmp_path / "w", tmp_path / "a.jsonl", n_topics=2, per_topic=5, batch=5)
    t = FakeTeacher(n_topics=2, start=1000)
    stats = build(t, tmp_path / "w", tmp_path / "b.jsonl", n_topics=2, per_topic=10, batch=5)
    assert t.calls == 2 and stats["candidates"] == 20                    # one more batch per topic, no new topics


def test_cli_with_injected_model(tmp_path):
    rc = main(["build", "--work-dir", str(tmp_path / "w"), "--out", str(tmp_path / "o.jsonl"), "--n-topics", "2",
               "--per-topic", "3", "--batch", "3"], model=FakeTeacher(n_topics=2))
    assert rc == 0 and len(list(read_jsonl(tmp_path / "o.jsonl"))) == 6


def test_a_lost_server_stops_drafting_instead_of_using_up_rounds():
    import httpx

    from dpt.data.gen import t3

    class Gone:
        def generate(self, *a, **k):
            raise httpx.ConnectError("Connection refused")

    class Sink:
        def __init__(self):
            self.rows = []

        def write(self, rows):
            self.rows += list(rows)

    sink = Sink()
    with pytest.raises(httpx.ConnectError):
        t3.draft_items_for_topic(Gone(), {"topic": "Seismic moment", "scope": "s"}, 0, [], per_topic=20, batch=10,
                                 out=sink, seed=1)
    assert sink.rows == []
