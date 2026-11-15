"""Knowledge-dense report data from openly licensed papers (dpt.data.gen.knowledge, 2026-11-11)."""

from __future__ import annotations

import json

from dpt.common.io import read_jsonl
from dpt.data.gen.knowledge import (check_answer, fetch_licenses, license_of, openalex_filter, parse_questions,
                                    process_chunk, reference_line, sections, select_chunks, sft_rows)

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


QUESTIONS_REPLY = ("thinking\n</think>\n\n<question>Explain how rupture directivity shapes ground motion "
                   "(approximately 550 words).</question>\n<facts>\n- factor 2-4 ahead of the rupture\n"
                   "- stronger at high frequencies\n</facts>\n<question>What does the excerpt say?</question>\n<facts>\n</facts>")


def _teacher(answer_text):
    def teacher(messages, **kw):
        if "Write 2 questions" in messages[0]["content"]:
            return [QUESTIONS_REPLY]
        return [answer_text]
    return teacher


def _judge(verdict):
    return lambda messages, **kw: ["\n" + json.dumps(verdict)]


OK_VERDICT = {"contradicted": [], "wrong": [], "study_specific": False, "mentions_source": False, "knowledge": 4}


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
    assert check_answer(PLAN, REPORT, {**OK_VERDICT, "wrong": ["moving NW to SE"]}) == "wrong"
    assert check_answer(PLAN, REPORT, {**OK_VERDICT, "study_specific": True}) == "study_specific"
    assert check_answer(PLAN, REPORT, {**OK_VERDICT, "knowledge": 2}) == "low_knowledge"
    assert check_answer(PLAN, REPORT, {"error": "unparsed"}) == "judge_unparsed"
    assert check_answer(PLAN, REPORT, OK_VERDICT) == "ok"


def test_sft_rows_are_closed_book_t3_rows_citing_only_the_source_paper():
    chunk = _chunk()
    rec = {"chunk_id": chunk["chunk_id"], "prompt_version": 2, "answers": [
        {"question": "Explain directivity.", "plan": PLAN, "report": REPORT, "verdict": OK_VERDICT, "status": "ok"},
        {"question": "Q2", "status": "length"}]}
    old_prompts = {**rec, "prompt_version": None}                         # a record of the 11-12 prompts: not used
    rows, summary = sft_rows([old_prompts, rec], {chunk["chunk_id"]: chunk}, model_id="Qwen/Qwen3.8-27B")
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


def test_the_clients_use_the_servers_v1_api(tmp_path):
    """The first knowledge job (2026-11-12) posted to http://127.0.0.1:8010/completions: 404 for all 40 sections."""
    from dpt.data.gen.knowledge import _generator

    for url in ("http://127.0.0.1:8010", "http://127.0.0.1:8010/", "http://127.0.0.1:8010/v1"):
        gen = _generator(url, "teacher", None, tmp_path / "cache.jsonl", "knowledge", 0.0, None, 4, 0.7, 0.95)
        assert gen.gen.client.base_url == "http://127.0.0.1:8010/v1"


def test_questions_come_in_tagged_blocks_without_a_stated_length():
    """2026-11-13: in JSON the teacher closed the brackets wrongly in 17 of the first 40 replies."""
    qs = parse_questions(QUESTIONS_REPLY)
    assert [q["question"] for q in qs] == ["Explain how rupture directivity shapes ground motion.",
                                           "What does the excerpt say?"]
    assert qs[0]["key_facts"] == ["factor 2-4 ahead of the rupture", "stronger at high frequencies"]
    assert qs[1]["key_facts"] == []
    assert parse_questions("still thinking <question>Q?</question>") == []                 # cut off: none
    q = parse_questions("x\n</think>\n\n<question>Write a report of about 550 words on Q.</question>")[0]
    assert q["question"] == "Write a report on Q."
    assert parse_questions('x\n</think>\n\n{"questions": [{"question": "Q", "key_facts": ["a"]}}') == []


def test_the_knowledge_check_sends_no_thinking_switch(tmp_path):
    """2026-11-13: asked to think, Gemma used ~4K tokens of thought and ran out before its JSON (34 of 46 answers)."""
    import httpx

    from dpt.data import backends as B
    from dpt.data.gen.knowledge import _generator

    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"index": 0, "message": {"content": "{}"}}]})

    client = B.OpenAICompatClient("http://judge/v1", "judge_v", transport=httpx.MockTransport(handler))
    B.VLLMGenerator(client, enable_thinking=None)([{"role": "user", "content": "check"}], effort=None)
    B.VLLMGenerator(client)([{"role": "user", "content": "check"}], effort=None)
    assert "chat_template_kwargs" not in sent[0]                                   # as the T3 reward judge
    assert sent[1]["chat_template_kwargs"] == {"enable_thinking": True}            # the default is unchanged
    gen = _generator("http://127.0.0.1:8110", "judge_v", None, tmp_path / "c.jsonl", "knowledge_check_v2", 0.0, None,
                     4, 0.6, 0.95, enable_thinking=None)
    assert gen.gen.enable_thinking is None and gen.label == "knowledge_check_v2:judge_v"


def test_only_records_of_the_current_prompts_count():
    """2026-11-13: the queued jobs' frozen batch scripts all write records.jsonl, so the version is in each record. A
    section done with the older prompts is done again, and the last current record of a section wins."""
    from dpt.data.gen.knowledge import PROMPT_VERSION, current_records

    recs = [{"chunk_id": "a", "status": "done"}, {"chunk_id": "b", "prompt_version": PROMPT_VERSION, "status": "x"},
            {"chunk_id": "b", "prompt_version": PROMPT_VERSION, "status": "y"}, {"chunk_id": "c", "prompt_version": 1}]
    cur = current_records(recs)
    assert sorted(cur) == ["b"] and cur["b"]["status"] == "y"
    rec = process_chunk(_chunk(), _teacher("no tags\n</think>\n\nnothing"), _judge(OK_VERDICT))
    assert rec["prompt_version"] == PROMPT_VERSION


def test_recheck_judges_v2_answers_again_from_their_stored_reports():
    """2026-11-13: the v3 check replaces "unsupported" by "wrong" and adds "study_specific". v2 answers are checked
    again from their stored plan and report, so no teacher work is repeated; v1 records and v3 records are left."""
    from dpt.data.gen.knowledge import JUDGE_VERSION, recheck

    chunk = _chunk()
    ans = {"question": "Q", "plan": PLAN, "report": REPORT, "verdict": {"unsupported": ["x"] * 5}, "status": "unsupported"}
    v2 = {"chunk_id": chunk["chunk_id"], "prompt_version": 2, "status": "done",
          "answers": [{"question": "Q0", "status": "question_mentions_source"}, dict(ans)]}
    v1 = {"chunk_id": chunk["chunk_id"], "status": "done", "answers": [dict(ans)]}
    v3 = {"chunk_id": chunk["chunk_id"], "prompt_version": 3, "judge_version": JUDGE_VERSION, "answers": [dict(ans)]}
    records = [v1, v2, v3]
    assert recheck(records, {chunk["chunk_id"]: chunk}, _judge(OK_VERDICT)) == 1
    assert v2["judge_version"] == JUDGE_VERSION and v2["answers"][1]["status"] == "ok"
    assert v2["answers"][0]["status"] == "question_mentions_source"                 # never reached the teacher
    assert "judge_version" not in v1 and v3["answers"][0]["status"] == "unsupported"

    def broken(messages, **kw):
        raise ConnectionError("judge down")

    v2b = {**v2, "judge_version": None, "answers": [dict(ans)]}
    assert recheck([v2b], {chunk["chunk_id"]: chunk}, broken) == 0 and v2b["judge_version"] is None
