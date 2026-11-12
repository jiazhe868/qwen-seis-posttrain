"""Knowledge-dense report data from openly licensed papers (dpt.data.gen.knowledge, 2026-11-11)."""

from __future__ import annotations

import json

from dpt.common.io import read_jsonl
from dpt.data.gen.knowledge import (check_answer, fetch_licenses, license_of, openalex_filter, process_chunk,
                                    reference_line, sections, select_chunks, sft_rows)

PROSE = ("Rupture directivity concentrates seismic energy in the direction the rupture propagates, so ground motion "
         "at stations ahead of the rupture front is stronger and richer in high frequencies than behind it. ")
REPORT = " ".join(["Directivity raises peak ground velocity ahead of the rupture by a factor of two to four."] * 40)
PLAN = "Directivity: definition, Doppler-like amplification, frequency dependence, limits for small events."


def _chunk(cid="W1#0", **kw):
    return {"chunk_id": cid, "paper_id": cid.split("#")[0], "title": "Rupture directivity in moderate earthquakes",
            "year": 2021, "license": "cc-by", "doi": "10.1234/abc", "authors": ["Ana Lopez", "Ben Wu", "Cy Dee"],
            "n_authors": 5, "url": "https://example.org/p.pdf", "text": PROSE * 40, **kw}


def test_openalex_lookup_ids_and_the_license_of_the_downloaded_copy():
    assert openalex_filter({"id": "W4384925754"}) == ("openalex", "W4384925754")
    assert openalex_filter({"id": "arxiv-2607.01448v1"}) == ("doi", "10.48550/arxiv.2607.01448")
    assert openalex_filter({"id": "eartharxiv-a9gd6_v1"}) == ("doi", "10.31223/osf.io/a9gd6")
    assert openalex_filter({"id": "10.1038/S41467-022-34955-6"}) == ("doi", "10.1038/s41467-022-34955-6")
    assert openalex_filter({"id": "arxiv-0703249v1", "url": "https://arxiv.org/pdf/physics/0703249v1"}) == (
        "doi", "10.48550/arxiv.physics/0703249")
    assert openalex_filter({"id": "substack-7"}) is None
    work = {"best_oa_location": {"pdf_url": "https://journal.org/x.pdf", "license": "cc-by-nc"},
            "locations": [{"pdf_url": "https://arxiv.org/pdf/2607.01448", "license": "cc-by"}]}
    assert license_of(work, "https://arxiv.org/pdf/2607.01448v1") == "cc-by"          # the copy we downloaded
    assert license_of(work, "https://elsewhere.net/y.pdf") == "cc-by-nc"               # else the best OA location


def test_fetch_licenses_sends_only_ids_and_resumes(tmp_path):
    calls = []

    def get(url):
        calls.append(url)
        return {"results": [{"id": "https://openalex.org/W1", "doi": "https://doi.org/10.1/x", "publication_year": 2020,
                             "authorships": [{"author": {"display_name": "Ana Lopez"}}],
                             "best_oa_location": {"license": "cc-by"}, "open_access": {"oa_status": "gold"}}]}

    out = tmp_path / "licenses.jsonl"
    recs = [{"id": "W1", "url": "u"}, {"id": "W2", "url": "u"}]
    stats = fetch_licenses(recs, out, get=get)
    rows = {r["id"]: r for r in read_jsonl(out)}
    assert stats["found"] == 1 and stats["not_found"] == 1 and rows["W1"]["license"] == "cc-by"
    assert rows["W1"]["doi"] == "10.1/x" and rows["W2"]["found"] is False
    assert "filter=openalex%3AW1%7CW2" in calls[0] and "mailto" not in calls[0]
    fetch_licenses(recs, out, get=get)
    assert len(calls) == 1                                                             # nothing left to ask


def test_sections_drop_line_numbers_and_references_and_split_long_paragraphs():
    numbered = "\n".join(f"{line}\n{i}" for i, line in enumerate((PROSE * 120).split(". ")))
    text = numbered + "\n\nReferences\nAki, K. (1967). Scaling law of seismic spectrum.\n" + "Lopez 2020. " * 400
    secs = sections(text)
    assert secs and all(4000 <= len(s) <= 9000 for s in secs)
    assert not any("Scaling law" in s for s in secs)                                   # the reference list is cut
    assert not any("\n7\n" in s for s in secs)                                         # line numbers removed


def test_select_keeps_open_licenses_on_topic_and_before_d_fresh():
    def rec(i, **kw):
        return {"id": f"W{i}", "title": f"Paper {i} on earthquake rupture", "date": "2020-01-01",
                "text": PROSE * 100, "url": "u", **kw}

    records = [rec(1), rec(2), rec(3, date="2026-09-01"), rec(4, text="Telescope actuators. " * 800),
               rec(5, title="Paper 1 on earthquake rupture")]
    licenses = {f"W{i}": {"license": "cc-by", "year": 2020} for i in range(1, 6)}
    licenses["W2"]["license"] = "cc-by-nc"
    chunks, stats = select_chunks(records, licenses, max_papers=10, chunks_per_paper=2)
    assert {c["paper_id"] for c in chunks} == {"W1"} and len(chunks) == 2
    assert stats["license_not_admitted:cc-by-nc"] == 1 and stats["too_recent"] == 1
    assert stats["off_topic"] == 1 and stats["duplicate_title"] == 1


def _teacher(answer_text):
    def teacher(messages, **kw):
        if "Write 2 questions" in messages[0]["content"]:
            return ['thinking\n</think>\n\n{"questions": [{"question": "Explain how rupture directivity shapes '
                    'ground motion.", "key_facts": ["factor 2-4"]}, {"question": "What does the excerpt say?", '
                    '"key_facts": []}]}']
        return [answer_text]
    return teacher


def _judge(verdict):
    return lambda messages, **kw: ["\n" + json.dumps(verdict)]


OK_VERDICT = {"contradicted": [], "unsupported": [], "mentions_source": False, "knowledge": 4}


def test_one_section_gives_checked_answers_with_reasons():
    good = f"I recall the notes say...\n</think>\n\n<plan>\n{PLAN}\n</plan>\n<report>\n{REPORT}\n</report>"
    rec = process_chunk(_chunk(), _teacher(good), _judge(OK_VERDICT))
    assert [a["status"] for a in rec["answers"]] == ["ok", "question_mentions_source"]
    assert rec["answers"][0]["plan"] == PLAN                        # the teacher's own thinking is not kept
    bad = process_chunk(_chunk(), _teacher(good), _judge({**OK_VERDICT, "contradicted": ["a factor of ten"]}))
    assert bad["answers"][0]["status"] == "contradicted"
    cited = good.replace("Directivity raises", "As the authors show, directivity raises", 1)
    assert process_chunk(_chunk(), _teacher(cited), _judge(OK_VERDICT))["answers"][0]["status"] == "mentions_source"


def test_check_answer_reasons():
    assert check_answer(None, REPORT, OK_VERDICT) == "unparsed_answer"
    assert check_answer(PLAN, "too short", OK_VERDICT) == "length"
    assert check_answer(PLAN, REPORT + "\n\nReferences\nAki 1967", OK_VERDICT) == "has_references"
    assert check_answer(PLAN, REPORT.replace("by a factor", "(Figure 3) by a factor", 1), OK_VERDICT) == "mentions_source"
    assert check_answer(PLAN, REPORT, {**OK_VERDICT, "unsupported": ["a", "b", "c"]}) == "unsupported"
    assert check_answer(PLAN, REPORT, {**OK_VERDICT, "knowledge": 2}) == "low_knowledge"
    assert check_answer(PLAN, REPORT, {"error": "unparsed"}) == "judge_unparsed"
    assert check_answer(PLAN, REPORT, OK_VERDICT) == "ok"


def test_sft_rows_are_closed_book_t3_rows_citing_only_the_source_paper():
    chunk = _chunk()
    rec = {"chunk_id": chunk["chunk_id"], "answers": [
        {"question": "Explain directivity.", "plan": PLAN, "report": REPORT, "verdict": OK_VERDICT, "status": "ok"},
        {"question": "Q2", "status": "length"}]}
    rows, summary = sft_rows([rec], {chunk["chunk_id"]: chunk}, model_id="Qwen/Qwen3.8-27B")
    assert summary == {"rows": 1, "papers": 1, "answer_status": {"ok": 1, "length": 1}, "licenses": {"cc-by": 1}}
    msgs = rows[0]["messages"]
    assert msgs[0]["role"] == "system" and "References" in msgs[0]["content"]          # the T3 system prompt
    assert msgs[1]["content"].endswith("Aim for roughly 550 words.") and "notes" not in msgs[1]["content"]
    assert msgs[2]["reasoning_content"] == PLAN
    assert msgs[2]["content"].endswith("References\nLopez et al. (2021). Rupture directivity in moderate "
                                       "earthquakes. https://doi.org/10.1234/abc")
    assert rows[0]["subset"] == "knowledge_report" and rows[0]["meta"]["knowledge_checked"] is True
    assert reference_line({**chunk, "authors": ["Ana Lopez", "Ben Wu"], "n_authors": 2, "doi": None}).startswith(
        "Lopez and Wu (2021).")
