"""Blind review: 20 pairs first, 10 more only on disagreement; the offline page; the import into results."""

from __future__ import annotations

import csv
import json
import re

import pytest

from dpt.common.io import read_json, write_json, write_jsonl
from dpt.eval.blind_review import main, render_page, summarize
from dpt.eval.graders.pairwise import export_blind_review


def _items(n=40):
    return [{"id": f"r{i:02d}", "question": f"q{i}", "answer": {"rubric": [{"criterion": "c", "weight": 2}]}}
            for i in range(n)]


def _export(n_first=20, n_extra=10):
    items = _items()
    cand = {it["id"]: f"CAND {it['id']}" for it in items}
    base = {it["id"]: f"BASE {it['id']}" for it in items}
    return export_blind_review(items, cand, base, seed=3, n_first=n_first, n_extra=n_extra)


def test_export_takes_a_fixed_count_in_two_batches():
    review, key = _export()
    assert [r["batch"] for r in review] == [1] * 20 + [2] * 10
    assert review == _export()[0]                                       # deterministic
    first20 = {k["item_id"] for k, r in zip(key, review) if r["batch"] == 1}
    assert first20 == {k["item_id"] for k, r in zip(_export(20, 0)[1], _export(20, 0)[0])}   # batch 1 is stable
    sides = [k["candidate_side"] for k in key]
    assert 1 in sides and 2 in sides                                    # shuffled
    assert all("item_id" not in r for r in review)


def test_page_is_self_contained_and_never_runs_response_text():
    review, _ = _export()
    review[0]["response_1"] = 'evil </script><script>alert(1)</script> "quotes"'
    html = render_page(review, 1)
    assert "http://" not in html and "https://" not in html             # works offline, loads nothing
    assert "<script>alert(1)" not in html                               # cannot close the data block
    data = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S).group(1))
    assert len(data["pairs"]) == 20 and data["batch"] == 1
    assert data["pairs"][0]["response_1"].startswith("evil </script>")   # shown as text, intact
    assert "candidate_side" not in html and "item_id" not in html
    with pytest.raises(ValueError):
        render_page(review, 3)


def _results(outcome_by_role):
    """results.json with per-item outcomes of two judges."""
    per_item = [{"id": i, "outcome": outcome_by_role["primary"][i],
                 "by_judge": {r: oc[i] for r, oc in outcome_by_role.items()}} for i in outcome_by_role["primary"]]
    return {"model_name": "rl_seed1", "sets": {"d_report": {"score": 70.0, "score_judge": "second",
                                                            "gate_judge": "min", "per_item": per_item}}}


def _verdicts_for(review, key, want):
    """Reviewer verdict rows that produce candidate outcome ``want(item_id)``."""
    side = {k["blind_id"]: (k["item_id"], k["candidate_side"]) for k in key}
    out = {}
    for r in review:
        iid, cs = side[r["blind_id"]]
        o = want(iid)
        out[r["blind_id"]] = {"verdict": "tie" if o == "tie" else str(cs if o == "win" else 3 - cs), "note": ""}
    return out


def test_disagreement_after_batch_one_asks_for_batch_two():
    review, key = _export()
    ids = [k["item_id"] for k in key]
    judges = {"primary": {i: "win" for i in ids}, "second": {i: "win" for i in ids}}
    b1 = {r["blind_id"] for r in review if r["batch"] == 1}
    v = {b: x for b, x in _verdicts_for(review, key, lambda i: "loss").items() if b in b1}
    s = summarize(review, key, v, _results(judges)["sets"]["d_report"], extra_if_gap=15.0)
    assert s["status"] == "needs_batch_2" and s["n"] == 20 and s["reviewer_win_rate"] == 0.0
    assert s["by_judge"]["primary"] == {"win_rate_same_pairs": 100.0, "agreement": 0.0}

    agree = {b: x for b, x in _verdicts_for(review, key, lambda i: "win").items() if b in b1}
    assert summarize(review, key, agree, _results(judges)["sets"]["d_report"])["status"] == "done"
    part = dict(list(agree.items())[:5])
    assert summarize(review, key, part, _results(judges)["sets"]["d_report"])["status"] == "incomplete"
    with pytest.raises(ValueError, match="unknown pairs"):
        summarize(review, key, {"nope": {"verdict": "1"}}, _results(judges)["sets"]["d_report"])


def test_import_records_the_review_and_the_report_shows_it(tmp_path):
    from dpt.eval.report import build_summary, render_markdown

    review, key = _export()
    ids = [k["item_id"] for k in key]
    row = tmp_path / "rl_seed1"
    write_jsonl(row / "d_report" / "expert_review.jsonl", review)
    write_jsonl(row / "d_report" / "expert_review.KEY.jsonl", key)
    judges = {"primary": {i: "win" for i in ids}, "second": {i: ("win" if i < "r20" else "tie") for i in ids}}
    write_json(row / "results.json", _results(judges))
    html = tmp_path / "page.html"
    assert main(["page", "--row-dir", str(row), "--out", str(html)]) == 0 and html.exists()

    v = _verdicts_for(review, key, lambda i: "win" if int(i[1:]) % 2 else "tie")
    b1 = [(b, x) for b, x in v.items() if any(r["blind_id"] == b and r["batch"] == 1 for r in review)]
    with open(tmp_path / "v1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["blind_id", "verdict", "note"])
        w.writeheader()
        w.writerows({"blind_id": b, **x} for b, x in b1)
    rc = main(["import", "--row-dir", str(row), "--verdicts", str(tmp_path / "v1.csv"), "--extra-if-gap", "50"])
    assert rc == 0
    br = read_json(row / "results.json")["sets"]["d_report"]["blind_review"]
    assert br["status"] == "done" and br["n"] == 20 and set(br["by_judge"]) == {"primary", "second"}
    md = render_markdown(build_summary({"rl_seed1": row}))
    assert "Blind review, rl_seed1 (20 pairs): reviewer win rate" in md and "primary 100.0%" in md
