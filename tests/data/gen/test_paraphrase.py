"""Teacher paraphrase of T1 questions: numbers and units must survive, the label and the generator stay."""

from __future__ import annotations

from collections import Counter

from dpt.chat.thinking import THINK_END
from dpt.common.io import read_jsonl, write_jsonl
from dpt.data.gen.common import candidate
from dpt.data.gen.paraphrase import check, chosen, main, numbers, units
from dpt.data.gen.teacher import Reply
from dpt.data.schema import Problem


def test_numbers_and_units_are_parsed_as_written():
    assert numbers("M0 = 3.2 × 10^18 N·m over 1,200 s and 4e-3 s, depth −5 km") == Counter(
        {"3.2e+18": 1, "1200": 1, "0.004": 1, "-5": 1})
    assert numbers("an S1 phase on station B7") == Counter()                 # digits inside names do not count
    assert units("6.0 km/s for 120 km, answer in seconds, dip 30°") == {"km/s", "km", "s", "deg"}
    assert units("6.0 kilometres per second") != units("6.0 km/s")             # a rewording of a unit is a change


Q = ("A P wave travels 150 km through the crust at 6.0 km/s before reaching a seismometer. "
     "How long does the trip take, in seconds?")


def test_check_accepts_a_faithful_rewrite_and_rejects_the_rest():
    good = ("Before it reaches a seismometer, a P wave crosses 150 km of crust at 6.0 km/s. "
            "Find the travel time in seconds.")
    assert check(Q, good, 25.0) is None
    assert check(Q, good.replace("150", "160"), 25.0) == "numbers_changed"
    assert check(Q, good.replace("6.0 km/s", "six km/s"), 25.0) == "numbers_changed"   # spelled out
    assert check(Q, good.replace("in seconds", "in minutes"), 25.0) == "units_changed"
    assert check(Q, good + " Note that t = d/v.", 25.0) == "leaks_solution"
    assert check(Q, "Find the time for 150 km at 6.0 km/s in seconds.", 25.0) == "length"
    assert check(Q, None) == "empty"


def _cands(n=40):
    rows = []
    for i in range(n):
        d = 100 + i
        q = (f"A P wave travels {d} km through the crust at 6.0 km/s before reaching a seismometer. "
             "How long does the trip take, in seconds?")
        rows.append(candidate("t1_num", q, {"value": d / 6.0, "unit": "s", "rel_tol": 0.01}, origin="test",
                              family="travel_time",
                              generator={"name": "travel_time", "params": {"distance_km": d, "velocity_km_s": 6.0}}))
    rows.append(candidate("t4_general", "What is 2 + 2 in total, counting carefully?",
                          {"kind": "math", "value": 4, "unit": ""}, origin="test"))
    return rows


class FakeTeacher:
    model_id = "Qwen/Qwen3.8-27B"
    effort = "medium"
    tokenizer = None

    def __init__(self, fail=False):
        self.calls, self.fail = 0, fail

    def generate(self, messages, *, n=1, seed=None, max_tokens=None, temperature=None):
        assert not self.fail, "no request expected"
        self.calls += 1
        q = messages[-1]["content"].split("Problem:\n", 1)[1].split("\n\nReturn only", 1)[0]
        d = int(q.split("travels ")[1].split(" km")[0])
        if d % 3 == 0:     # faithful rewrite
            new = (f"Before it reaches a seismometer, a P wave crosses {d} km of crust at 6.0 km/s. "
                   "Find the travel time in seconds.")
        elif d % 3 == 1:   # changes a number
            new = f"A P wave crosses {d + 1} km of crust at 6.0 km/s. Find the travel time in seconds, please."
        else:              # cut off
            return [Reply("thinking that never ends", "length")]
        return [Reply(f"plan <problem>draft</problem>\n{THINK_END}\n\n<problem>{new}</problem>", "stop")]


def test_run_replaces_only_checked_rewrites_and_resumes(tmp_path):
    rows = _cands()
    write_jsonl(tmp_path / "in.jsonl", rows)
    t = FakeTeacher()
    args = ["--in", str(tmp_path / "in.jsonl"), "--out", str(tmp_path / "out.jsonl"), "--fraction", "0.5"]
    assert main(args, model=t) == 0
    out = list(read_jsonl(tmp_path / "out.jsonl"))
    assert len(out) == len(rows) and t.calls == sum(chosen(r["uid"], 0.5, 1) for r in rows if r["task"] == "t1_num")
    assert 0 < t.calls < 40
    by_status = Counter((r["meta"].get("paraphrase") or {}).get("status", "not_chosen") for r in out)
    assert by_status["ok"] > 0 and by_status["rejected"] > 0 and by_status["not_chosen"] > 0
    for before, after in zip(rows, out):
        p = Problem.from_dict(after)
        assert p.answer == before["answer"] and p.meta.get("generator") == before["meta"].get("generator")
        para = p.meta.get("paraphrase") or {}
        if para.get("status") == "ok":
            assert p.question != before["question"] and p.uid != before["uid"]
            assert para["original"] == before["question"] and para["original_uid"] == before["uid"]
            assert p.question.startswith("Before it reaches")
        else:
            assert p.question == before["question"] and p.uid == before["uid"]
    assert out[-1]["task"] == "t4_general" and "paraphrase" not in out[-1]["meta"]    # other tasks untouched
    # every chosen record was decided (cut-off replies count as rejected), so a rerun asks nothing
    assert main(args, model=FakeTeacher(fail=True)) == 0
    assert [r["uid"] for r in read_jsonl(tmp_path / "out.jsonl")] == [r["uid"] for r in out]
