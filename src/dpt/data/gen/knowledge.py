"""Knowledge-dense report data for SFT v2 (2026-11-11): seismology reports written from openly licensed papers.

Why: the D-Report shortfalls are missing science (docs/reports/failure_types_and_tests_1110_en.md). SFT and RL models
write well-organized reports that lack the mechanisms, numbers and limits a specialist would give. RL can only
sharpen what the model already writes, so the knowledge has to come in through SFT.

How: the teacher (Qwen3.8-27B) reads a section of an openly licensed paper, writes report questions on what that
section treats in depth, and answers each one with the section as background notes. The student is trained
closed-book: its SFT row holds only the question, so the facts have to end up in its weights.

    python -m dpt.data.gen.knowledge licenses --corpus fulltext.jsonl preprints.jsonl --out $K/licenses.jsonl
    python -m dpt.data.gen.knowledge select   --corpus fulltext.jsonl preprints.jsonl --licenses $K/licenses.jsonl \\
        --out $K/chunks.jsonl
    python -m dpt.data.gen.knowledge run      --chunks $K/chunks.jsonl --out $K/records.jsonl --cache-dir $K/cache \\
        --teacher-url URL --teacher-model teacher --tokenizer Qwen/Qwen3.8-27B --judge-url URL --judge-model judge_v
    python -m dpt.data.gen.knowledge export   --records $K/records.jsonl --out $SFT/knowledge_report.jsonl

``licenses`` (workstation, needs the internet) asks OpenAlex for the license of every full text, matched to the
location the text was downloaded from. ``select`` keeps CC-BY, CC-BY-SA, CC0 and public-domain texts on a
seismology topic, cuts the references off, and splits the body into sections of 4,000-9,000 characters.
``run`` does, per section:
  1. questions: the teacher writes ``--per-chunk`` (2) report questions a specialist would understand without
     the paper, each with the paper's key facts a good answer needs;
  2. answers: per question, the teacher writes a <plan> and a <report> (~550 words, D-Report's target), with the
     section as notes it must treat as its own knowledge. Its own thinking refers to those notes, so it is
     dropped: the plan, written in the visible answer under the same rules, becomes the row's thinking;
  3. check: plan and report must name no source the reader cannot see (notes, excerpt, the paper, its
     authors, figures or tables; ``SOURCE_MENTION``), the report must be 300-900 words, and Gemma (another family
     than the teacher) must find no statement that contradicts the section, at most 2 specific claims it cannot
     support, and a knowledge score of at least 3 of 5.
``export`` writes the kept answers as SFT rows (subset ``knowledge_report``) in the T3 prompt format, each ending with
a References section that lists the source paper (title, year, DOI), the only source the row cites.

GPT-6 Luna is never used here (eval only). D-Report was drafted from topics, not papers, so what remains to rule
out is wording close to its questions: dpt.data.sft_build decontaminates every row against the eval sets.

Resumable: every model reply is cached in ``--cache-dir`` (dpt.data.backends.CachedGenerator), and a section whose
record is in ``--out`` is skipped.

What the first 40 sections showed (2026-11-13), and what changed:
- the teacher wrote the questions as nested JSON and closed the brackets wrongly in 17 of 40 replies (``]}}`` for
  ``]}]}``), so those sections had no questions. Questions are now tagged blocks (``<question>``, ``<facts>``), as
  the plans and reports already were, which no bracket can break;
- some questions asked about the paper's own study ("a four-day survey of roughly 70 buildings"), which the
  student cannot answer without the paper; the prompt now asks for topics a specialist can answer from the field;
- asked to think, Gemma wrote ~4K tokens of thought and ran out before its JSON on 34 of 46 answers. The check
  now sends no thinking switch, as the T3 reward judge does.
What the next 78 sections showed (2026-11-13, prompts v2), and what changed:
- Gemma listed a median of 4 claims per answer as "unsupported", and with at most 2 allowed it dropped 68% of the
  answers. Most of those claims were true background (the 2011 Tohoku moment, ~4 × 10²² N·m). The check now asks for
  claims Gemma believes are wrong; on 60 of these answers it named such claims in 3;
- 22 of the same 60 still built on one study's set-up (a model's own 10.5 Hz frequency, a test's input levels). The
  check now asks for that ("study_specific"), and the question and answer prompts (v3) say it more firmly.
  With the new check, 36 of the 60 pass, against 12 of 60 before.
Records carry ``prompt_version`` and ``judge_version``. Resuming, ``todo`` and ``export`` count the last record of each
section written with prompts v2 or v3; v1 sections are done again. A record from an older judge is checked again at the
start of ``run`` from its stored plan and report, so no teacher work is repeated. The versions live in the records, not
in the file name: a queued job runs the copy of its batch script taken at submission, and those copies all write
``records.jsonl``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
from urllib.parse import urlencode

from dpt.chat.thinking import split_thinking
from dpt.common.io import read_jsonl, write_json
from dpt.common.logging import get_logger
from dpt.common.types import T3_REPORT
from dpt.data.gen.teacher import _v1, last_block, last_json
from dpt.data.prompts import messages_for_problem
from dpt.data.schema import SFTSample

log = get_logger(__name__)

ADMITTED_LICENSES = ("cc0", "public-domain", "cc-by", "cc-by-sa")
LENGTH_HINT_WORDS = 550                      # D-Report's target (dpt.data.gen.t3)
MIN_REPORT_WORDS, MAX_REPORT_WORDS = 300, 900
MIN_KNOWLEDGE = 3
CHUNK_MIN_CHARS, CHUNK_MAX_CHARS = 4000, 9000
SEISMOLOGY_TERMS = re.compile(
    r"\b(seism\w*|earthquake\w*|fault\w*|ruptur\w*|tomograph\w*|aftershock\w*|tremor\w*|magnitude\w*|"
    r"crust\w*|mantle|moho|subduction|slip|receiver function\w*|attenuation|ground motion\w*|focal mechanism\w*|"
    r"moment tensor\w*|p-wave\w*|s-wave\w*|surface wave\w*|body wave\w*|seismograph\w*|seismogram\w*|"
    r"tsunami\w*|volcan\w*|geophysic\w*)\b", re.I)
REFERENCES_HEADING = re.compile(r"^\s*(references|bibliography|literature cited|acknowledg(e)?ments?)\s*:?\s*$",
                                re.I | re.M)
# A report the student writes closed-book must not point at material the reader cannot see.
SOURCE_MENTION = re.compile(
    r"\b(the|these|those|my|your|provided|given|above|following) (notes?|excerpts?|passages?|sections? above)\b|"
    r"\b(the|this) (excerpt|passage)\b|\bthe (provided|given|above) (text|material|source|paper|article)\b|"
    r"\b(this|the) (paper|article|manuscript|preprint)\b|\bthe authors?\b|\b(Fig\.|Figs\.|Figure|Table)\s*\d",
    re.I)


# ---- 1. licenses (OpenAlex) ---------------------------------------------------------------------------------------
def corpus_records(paths: Iterable[str | Path]) -> Iterable[dict[str, Any]]:
    """Corpus rows with full text: ``{source, id, title, text, url, date, extra}`` (nanogpt-seis crawl, 2026-07)."""
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                extra = r.get("extra") or {}
                if isinstance(extra, str):
                    try:
                        extra = ast.literal_eval(extra)
                    except (ValueError, SyntaxError):
                        extra = {}
                if extra.get("full_text"):
                    yield {**r, "extra": extra}


def openalex_filter(record: dict[str, Any]) -> tuple[str, str] | None:
    """(filter field, value) that finds a corpus row in OpenAlex: its work id or its DOI. Most full texts are keyed
    by DOI; arXiv and EarthArXiv preprints have DOIs of the form 10.48550/arxiv.<id> and 10.31223/osf.io/<id>."""
    rid, url = str(record.get("id", "")), str(record.get("url") or "")
    doi = str((record.get("extra") or {}).get("doi") or "")
    if re.fullmatch(r"W\d+", rid):
        return "openalex", rid
    if rid.startswith("10.") or doi.startswith("10."):
        value = (doi or rid).lower()
        return None if ("," in value or "|" in value) else ("doi", value)     # the filter syntax's separators
    m = re.fullmatch(r"arxiv-(\d{4}\.\d{4,5})(v\d+)?", rid)
    if m:
        return "doi", f"10.48550/arxiv.{m.group(1)}"
    m = re.search(r"arxiv\.org/(?:pdf|abs)/([a-z-]+(?:\.[a-z]{2})?/\d{7})", url, re.I)
    if rid.startswith("arxiv-") and m:                                       # old style, e.g. physics/0703249
        return "doi", f"10.48550/arxiv.{m.group(1).lower()}"
    m = re.fullmatch(r"eartharxiv-([a-z0-9]{5})(_v\d+)?", rid)
    if m:
        return "doi", f"10.31223/osf.io/{m.group(1)}"
    return None


def license_of(work: dict[str, Any], url: str) -> str | None:
    """The license of the location the text came from (matched by URL), else of the best open-access location."""
    locations = [loc for loc in [work.get("best_oa_location"), work.get("primary_location"),
                                 *(work.get("locations") or [])] if loc]
    host = re.sub(r"^https?://(www\.)?", "", url or "").split("/")[0].lower()
    for loc in locations:
        for key in ("pdf_url", "landing_page_url"):
            u = (loc.get(key) or "").lower()
            if u and (u == (url or "").lower() or (host and host in u)):
                return loc.get("license")
    return (work.get("best_oa_location") or {}).get("license")


def fetch_licenses(records: Sequence[dict[str, Any]], out: Path, *, batch: int = 50,
                   get: Callable[[str], dict[str, Any]] | None = None) -> dict[str, int]:
    """Append one row per corpus record to ``out``: license, DOI, year and the first authors. Rows already in ``out``
    are kept; a record OpenAlex does not know gets ``found: false``. Only public work ids are sent."""
    if get is None:
        import httpx

        client = httpx.Client(timeout=60.0, headers={"User-Agent": "dpt-knowledge-licenses"})

        def get(url: str) -> dict[str, Any]:
            for attempt in range(5):
                r = client.get(url)
                if r.status_code == 429 or r.status_code >= 500:
                    time.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                return r.json()
            r.raise_for_status()
            return r.json()

    done = {r["id"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [r for r in records if r["id"] not in done and openalex_filter(r) is not None]
    by_field: dict[str, list[dict[str, Any]]] = {"openalex": [], "doi": []}
    for r in todo:
        by_field[openalex_filter(r)[0]].append(r)
    stats: Counter[str] = Counter()
    out.parent.mkdir(parents=True, exist_ok=True)
    select = "id,doi,publication_year,authorships,best_oa_location,primary_location,locations,open_access"
    with out.open("a", encoding="utf-8") as fh:
        for field, rows in by_field.items():
            for i in range(0, len(rows), batch):
                part = rows[i:i + batch]
                values = [openalex_filter(r)[1] for r in part]
                data = get("https://api.openalex.org/works?" + urlencode(
                    {"filter": f"{field}:{'|'.join(values)}", "select": select, "per-page": batch}))
                found: dict[str, dict[str, Any]] = {}
                for w in data.get("results", []):
                    key = w["id"].rsplit("/", 1)[-1] if field == "openalex" else (w.get("doi") or "").lower()
                    found[key.replace("https://doi.org/", "")] = w
                for r, v in zip(part, values):
                    w = found.get(v)
                    if w is None:
                        row = {"id": r["id"], "found": False}
                    else:
                        authors = [(a.get("author") or {}).get("display_name") for a in (w.get("authorships") or [])]
                        row = {"id": r["id"], "found": True, "license": license_of(w, r.get("url", "")),
                               "doi": (w.get("doi") or "").replace("https://doi.org/", "") or None,
                               "year": w.get("publication_year"), "authors": [a for a in authors if a][:3],
                               "n_authors": len(authors), "oa_status": (w.get("open_access") or {}).get("oa_status")}
                    stats["found" if row["found"] else "not_found"] += 1
                    stats[f"license:{row.get('license')}"] += 1
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return dict(stats)


# ---- 2. selection and sections ------------------------------------------------------------------------------------
def body_of(text: str) -> str:
    """The text before its acknowledgments or references: the first such heading in the second half ends the body."""
    cut = [m.start() for m in REFERENCES_HEADING.finditer(text) if m.start() > len(text) // 2]
    return text[:cut[0]] if cut else text


def sections(text: str, *, min_chars: int = CHUNK_MIN_CHARS, max_chars: int = CHUNK_MAX_CHARS) -> list[str]:
    """The body as sections of ``min_chars``-``max_chars`` characters of mostly prose.

    The texts are PDF extractions (nanogpt-seis crawl): lines are hard-wrapped, paragraphs are separated by blank
    lines only now and then (~3,300 characters apart in an arXiv sample), and manuscripts carry line numbers on
    lines of their own. So line-number lines go, lines are joined, and a paragraph longer than ``max_chars`` is
    split at its last sentence end before the limit."""
    body = "\n".join(ln for ln in body_of(text).split("\n") if not re.fullmatch(r"\s*\d{1,4}\s*", ln))
    pieces = []
    for para in re.split(r"\n\s*\n", body):
        para = re.sub(r"\s*\n\s*", " ", para).strip()
        while len(para) > max_chars:
            cut = para.rfind(". ", 0, max_chars)
            cut = cut + 1 if cut > min_chars // 2 else max_chars
            pieces.append(para[:cut].strip())
            para = para[cut:].strip()
        if para:
            pieces.append(para)
    out, cur = [], ""
    for piece in pieces:
        if cur and len(cur) + len(piece) + 2 > max_chars:
            out.append(cur)
            cur = ""
        cur = f"{cur}\n\n{piece}" if cur else piece
    if cur:
        out.append(cur)
    # prose is ~80% letters; equations, tables and reference lists are far below 60%
    return [s for s in out if len(s) >= min_chars and sum(c.isalpha() for c in s) >= 0.6 * len(s)]


def is_seismology(record: dict[str, Any], *, min_hits: int = 3) -> bool:
    head = f"{record.get('title', '')}\n{(record.get('text') or '')[:4000]}"
    return len(SEISMOLOGY_TERMS.findall(head)) >= min_hits


def _rank(key: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{key}".encode()).hexdigest()


def select_chunks(records: Iterable[dict[str, Any]], licenses: dict[str, dict[str, Any]], *, max_papers: int,
                  chunks_per_paper: int, seed: int = 1,
                  before: str = "2026-08-01") -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Sections of up to ``max_papers`` openly licensed seismology papers dated before ``before`` (D-Fresh is
    newer), in a seeded random order, at most ``chunks_per_paper`` each, spread over the body."""
    stats: Counter[str] = Counter()
    seen_titles: set[str] = set()
    papers = []
    for r in records:
        stats["full_text"] += 1
        lic = licenses.get(r["id"]) or {}
        if lic.get("license") not in ADMITTED_LICENSES:
            stats[f"license_not_admitted:{lic.get('license')}"] += 1
            continue
        if str(r.get("date") or "") >= before:
            stats["too_recent"] += 1
            continue
        if not is_seismology(r):
            stats["off_topic"] += 1
            continue
        title = re.sub(r"\W+", " ", str(r.get("title", "")).lower()).strip()
        if title in seen_titles:
            stats["duplicate_title"] += 1
            continue
        seen_titles.add(title)
        papers.append(r)
    papers.sort(key=lambda r: _rank(r["id"], seed))
    chunks = []
    for r in papers:
        if stats["papers"] >= max_papers:
            break
        secs = sections(r["text"])
        if not secs:
            stats["no_sections"] += 1
            continue
        step = max(1, len(secs) // chunks_per_paper)
        picked = secs[::step][:chunks_per_paper]
        stats["papers"] += 1
        lic = licenses[r["id"]]
        for k, s in enumerate(picked):
            chunks.append({"chunk_id": f"{r['id']}#{k}", "paper_id": r["id"], "title": r.get("title"),
                           "date": r.get("date"), "year": lic.get("year"), "license": lic.get("license"),
                           "doi": lic.get("doi"), "authors": lic.get("authors") or [],
                           "n_authors": lic.get("n_authors") or 0, "url": r.get("url"), "text": s})
        stats["chunks"] += len(picked)
    return chunks, dict(stats)


# ---- 3. questions, answers, checks --------------------------------------------------------------------------------
QUESTIONS_PROMPT = """You are writing questions for seismology specialists. Below is a section of an openly
licensed paper ("{title}", {year}). Write {n} questions, each asking for a short technical report on a topic that
this section treats in depth.

Each question asks about seismology a specialist should know: a mechanism, a method and its limits, a relation
between quantities, or the behaviour of a well-known named earthquake, fault or region. A good answer should need
the specific knowledge in the section (mechanisms, quantities with units, conditions, limits), not general
statements.

Each question must stand on its own: a specialist who never read this paper should understand it and could answer
it fully from knowledge of the field. So the question must not depend on this paper's own study: not on its
experiment, model, survey, data set, stations, input motions or measured values. If the section is mostly about one
study, ask about the general topic behind it (the mechanism, the method, the relation it tests), as a textbook
chapter would. A major named earthquake, fault or region is fine, since its facts belong to the field. Do not
mention the section, the paper, its authors, its figures or its tables, and do not say how long the report should
be. For each question, list the key facts from the section that an excellent answer would contain.

Reply in exactly this form, once per question:
<question>the question</question>
<facts>
- a key fact
- another key fact
</facts>

<section>
{text}
</section>"""

ANSWER_NOTES = """You are a seismologist. The notes below are part of your own knowledge. Use them where they help,
together with what you know, and keep to what you are sure of.

<notes>
{text}

Key facts: {facts}
</notes>

Write as an expert who knows this material. Never mention notes, an excerpt, a paper, a study, its authors, or
figures or tables, and cite nothing: the reader sees only your report. Give mechanisms, quantities with units, and
limits where they matter. Do not present the set-up, measurements or results of the one study behind the notes as
general facts: give what holds in general, with typical values and ranges.

Reply in exactly this form:
<plan>
an outline of the points you will make, in a few lines, as notes to yourself
</plan>
<report>
the report, about {words} words, without a References section
</report>"""

CHECK_PROMPT = """You check a technical report against source notes.

<notes>
{text}
</notes>

<report>
{plan}

{report}
</report>

Answer five questions about the report.
1. "contradicted": every statement in the report that contradicts the notes, quoted. An empty list if none.
2. "wrong": every specific claim (a number, a named result, a causal statement) that you believe is false, judged
   from the notes and from what you know of seismology, quoted. A claim that is true but not in the notes is not
   wrong. An empty list if none.
3. "study_specific": true if the report presents the set-up, measurements or results of one particular study as
   general facts (for example a model's own frequency or a test's own input levels), or refers to "this study",
   "the configuration examined here" or the like. A named, well-known earthquake's facts are not study-specific.
4. "mentions_source": true if the report refers to notes, an excerpt, a paper, a study, its authors, figures or
   tables as if the reader could see them.
5. "knowledge": from 1 to 5, how much specific and correct technical content (mechanisms, quantities with units,
   methods, limits) the report gives beyond textbook generalities.

Reply with JSON only, for example:
{{"contradicted": [], "wrong": [], "study_specific": false, "mentions_source": false, "knowledge": 4}}"""

Gen = Callable[..., list[str]]
PROMPT_VERSION = 3          # 2026-11-13: v2 tagged questions; v3 firmer on one study's set-up (module docstring)
CURRENT_PROMPTS = (2, 3)    # v2 questions and answers are kept: the v3 check filters what v3 prompts now avoid
JUDGE_VERSION = 3           # v3: "wrong" and "study_specific" in place of "unsupported"


def current_records(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The last record of each section written with current prompts (``CURRENT_PROMPTS``)."""
    out: dict[str, dict[str, Any]] = {}
    for r in records:
        if r.get("prompt_version") in CURRENT_PROMPTS:
            out[r["chunk_id"]] = r
    return out


def _seed(*parts: Any) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16) % (2 ** 31)


def t3_messages(question: str) -> list[dict[str, str]]:
    """The question as T3 prompts and D-Report items are rendered (system prompt, the 550-word hint)."""
    return messages_for_problem({"task": T3_REPORT, "question": question,
                                 "answer": {"prompt_extra": {"length_hint": LENGTH_HINT_WORDS}}})


def answer_messages(question: str, chunk: dict[str, Any], key_facts: Sequence[str]) -> list[dict[str, str]]:
    """The T3 prompt with the notes added to its system turn (the SFT row keeps the T3 prompt without them)."""
    msgs = [dict(m) for m in t3_messages(question)]
    notes = ANSWER_NOTES.format(text=chunk["text"], facts="; ".join(key_facts) or "none listed",
                                words=LENGTH_HINT_WORDS)
    if msgs and msgs[0]["role"] == "system":
        msgs[0]["content"] = msgs[0]["content"] + "\n\n" + notes
    else:
        msgs.insert(0, {"role": "system", "content": notes})
    return msgs


def words(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


LENGTH_PHRASE = re.compile(r"\s*\((?:approximately|about|roughly|around|~)?\s*~?\s*\d{3,4}\s*words?\)"
                           r"|\s+of\s+(?:approximately|about|roughly|around)\s+\d{3,4}\s+words", re.I)


def parse_questions(reply: str) -> list[dict[str, Any]]:
    """The ``<question>`` blocks of the teacher's answer, each with the lines of the ``<facts>`` block after it. A
    reply whose thinking never ended (cut off) gives none. A stated report length is removed: the T3 prompt adds its
    own ("Aim for roughly 550 words.")."""
    sp = split_thinking(reply or "")
    if not sp.has_end:
        return []
    out = []
    for m in re.finditer(r"<question>(.*?)</question>\s*(?:<facts>(.*?)</facts>)?", sp.answer, re.S):
        question = " ".join(LENGTH_PHRASE.sub("", m.group(1)).split())
        facts = [f.strip(" -*\t") for f in (m.group(2) or "").splitlines() if f.strip(" -*\t")]
        if question:
            out.append({"question": question, "key_facts": facts})
    return out


VERDICT_KEYS = ("contradicted", "wrong", "study_specific", "mentions_source", "knowledge")


def judge_verdict(reply: str) -> dict[str, Any]:
    try:
        obj = {k: last_json(reply, k) for k in VERDICT_KEYS}
    except ValueError:
        return {"error": "unparsed"}
    try:
        knowledge = int(obj["knowledge"])
    except (TypeError, ValueError):
        return {"error": "no knowledge score"}
    return {"contradicted": list(obj["contradicted"] or []), "wrong": list(obj["wrong"] or []),
            "study_specific": bool(obj["study_specific"]), "mentions_source": bool(obj["mentions_source"]),
            "knowledge": knowledge}


def check_answer(plan: str | None, report: str | None, verdict: dict[str, Any] | None) -> str:
    """"ok" or the first reason the answer is dropped."""
    if not plan or not report:
        return "unparsed_answer"
    if SOURCE_MENTION.search(plan) or SOURCE_MENTION.search(report):
        return "mentions_source"
    if re.search(r"^\s*#*\s*references\b", report, re.I | re.M):
        return "has_references"
    n = words(report)
    if n < MIN_REPORT_WORDS or n > MAX_REPORT_WORDS:
        return "length"
    if verdict is None or "error" in verdict:
        return "judge_unparsed"
    if verdict["contradicted"]:
        return "contradicted"
    if verdict["mentions_source"]:
        return "judge_mentions_source"
    if verdict["study_specific"]:
        return "study_specific"
    if verdict["wrong"]:
        return "wrong"
    if verdict["knowledge"] < MIN_KNOWLEDGE:
        return "low_knowledge"
    return "ok"


PASSES_JUDGE = {"contradicted": [], "wrong": [], "study_specific": False, "mentions_source": False, "knowledge": 5}


def judge_answer(chunk: dict[str, Any], k: int, plan: str | None, report: str | None,
                 judge: Gen) -> tuple[dict[str, Any] | None, str]:
    """Gemma's verdict on answer ``k`` of a section, asked only when the checks that need no model pass, and the
    answer's status."""
    verdict = None
    if plan and report and check_answer(plan, report, PASSES_JUDGE) == "ok":
        verdict = judge_verdict(judge([{"role": "user", "content": CHECK_PROMPT.format(
            text=chunk["text"], plan=plan, report=report)}], n=1, effort=None, max_tokens=4096,
            seed=_seed(chunk["chunk_id"], k, "check"))[0])
    return verdict, check_answer(plan, report, verdict)


def recheck(records: list[dict[str, Any]], chunks: dict[str, dict[str, Any]], judge: Gen, *,
            concurrency: int = 16) -> int:
    """Check again, with the current judge, the answers of current-prompt records that an older judge checked, from
    their stored plans and reports (module docstring). Changes ``records`` in place; returns how many changed. A
    record whose judge call fails keeps its old judge version and is checked again next time."""
    stale = [r for r in records if r.get("prompt_version") in CURRENT_PROMPTS
             and r.get("judge_version") != JUDGE_VERSION and r["chunk_id"] in chunks]

    def one(r: dict[str, Any]) -> bool:
        try:
            for k, a in enumerate(r.get("answers") or []):
                if a.get("plan") is not None or a.get("report") is not None:      # it reached the teacher
                    a["verdict"], a["status"] = judge_answer(chunks[r["chunk_id"]], k, a.get("plan"),
                                                             a.get("report"), judge)
        except Exception as e:
            log.warning("recheck %s: %s: %s", r["chunk_id"], type(e).__name__, e)
            return False
        r["judge_version"] = JUDGE_VERSION
        return True

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        return sum(ex.map(one, stale))


def process_chunk(chunk: dict[str, Any], teacher: Gen, judge: Gen, *, per_chunk: int = 2) -> dict[str, Any]:
    """Questions, answers and checks of one section; one record with every answer and its verdict."""
    cid = chunk["chunk_id"]
    reply = teacher([{"role": "user", "content": QUESTIONS_PROMPT.format(
        title=chunk.get("title") or "untitled", year=chunk.get("year") or "n.d.", n=per_chunk,
        words=LENGTH_HINT_WORDS, text=chunk["text"])}], n=1, effort="medium", max_tokens=8192,
        seed=_seed(cid, "questions"))[0]
    questions = parse_questions(reply)
    versions = {"prompt_version": PROMPT_VERSION, "judge_version": JUDGE_VERSION}
    if not questions:
        return {"chunk_id": cid, **versions, "status": "no_questions", "answers": []}
    answers = []
    for k, q in enumerate(questions[:per_chunk]):
        question = str(q["question"]).strip()
        facts = [str(f) for f in (q.get("key_facts") or [])][:12]
        if SOURCE_MENTION.search(question):
            answers.append({"question": question, "status": "question_mentions_source"})
            continue
        text = teacher(answer_messages(question, chunk, facts), n=1, effort="medium", max_tokens=16384,
                       seed=_seed(cid, k, "answer"))[0]
        plan, report = last_block(text, "plan"), last_block(text, "report")
        verdict, status = judge_answer(chunk, k, plan, report, judge)
        answers.append({"question": question, "key_facts": facts, "plan": plan, "report": report,
                        "verdict": verdict, "status": status})
    return {"chunk_id": cid, **versions, "status": "done", "answers": answers}


# ---- 4. SFT rows --------------------------------------------------------------------------------------------------
def reference_line(chunk: dict[str, Any]) -> str:
    authors = chunk.get("authors") or []
    who = ""
    if authors:
        last = authors[0].split()[-1]
        who = f"{last} et al." if (chunk.get("n_authors") or len(authors)) > 2 else " and ".join(
            a.split()[-1] for a in authors[:2])
    year = chunk.get("year") or str(chunk.get("date") or "")[:4]
    where = f"https://doi.org/{chunk['doi']}" if chunk.get("doi") else chunk.get("url") or ""
    return " ".join(x for x in (f"{who} ({year})." if who else f"({year}).", f"{chunk.get('title', '').strip()}.",
                                where) if x).strip()


def sft_rows(records: Iterable[dict[str, Any]], chunks: dict[str, dict[str, Any]], *, model_id: str) -> tuple[
        list[dict[str, Any]], dict[str, Any]]:
    rows, reasons, licenses = [], Counter(), Counter()
    for rec in current_records(records).values():
        chunk = chunks.get(rec["chunk_id"])
        for k, a in enumerate(rec.get("answers") or []):
            reasons[a["status"]] += 1
            if a["status"] != "ok" or chunk is None:
                continue
            content = a["report"].strip() + "\n\nReferences\n" + reference_line(chunk)
            msgs = t3_messages(a["question"]) + [{"role": "assistant", "reasoning_content": a["plan"].strip(),
                                                  "content": content}]
            meta = {"task": T3_REPORT, "knowledge_checked": True, "knowledge_score": a["verdict"]["knowledge"],
                    "wrong_claims": len(a["verdict"]["wrong"]), "paper_id": chunk["paper_id"],
                    "license": chunk["license"], "doi": chunk.get("doi")}
            rows.append(SFTSample.from_dict({
                "subset": "knowledge_report", "messages": msgs, "reasoning_effort": "medium", "meta": meta,
                "source": {"origin": "dpt.data.gen.knowledge", "model": model_id, "chunk_id": rec["chunk_id"],
                           "answer": k}}).to_dict())
            licenses[chunk["license"]] += 1
    papers = {r["meta"]["paper_id"] for r in rows}
    return rows, {"rows": len(rows), "papers": len(papers), "answer_status": dict(reasons),
                  "licenses": dict(licenses)}


# ---- CLI ----------------------------------------------------------------------------------------------------------
def _generator(url: str, model: str, tokenizer: str | None, cache: Path, label: str, wait_s: float,
               url_file: str | None, concurrency: int, temperature: float, top_p: float,
               enable_thinking: bool | None = True) -> Gen:
    from dpt.data import backends as B

    # the servers' API is under /v1; the first knowledge job posted to /completions and got 404 for every request
    client = B.OpenAICompatClient(_v1(url), model, timeout=7200.0, max_concurrency=concurrency, url_file=url_file,
                                  wait_s=wait_s)
    tok = B.load_hf_tokenizer(tokenizer) if tokenizer else None
    gen = B.VLLMGenerator(client, tokenizer=tok, temperature=temperature, top_p=top_p, enable_thinking=enable_thinking)
    return B.CachedGenerator(gen, cache, f"{label}:{model}")


def cmd_run(a: argparse.Namespace) -> None:
    chunks = list(read_jsonl(a.chunks))
    if a.limit:
        chunks = chunks[:a.limit]
    out = Path(a.out)
    cache = Path(a.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    teacher = _generator(a.teacher_url, a.teacher_model, a.tokenizer, cache / "knowledge_teacher.jsonl", "knowledge",
                         a.wait_s, a.teacher_url_file, a.concurrency, 0.7, 0.95)
    # no thinking switch, and a label per judge version, so no earlier verdict is reused (module docstring)
    judge = _generator(a.judge_url, a.judge_model, None, cache / "knowledge_judge.jsonl",
                       f"knowledge_check_v{JUDGE_VERSION}", a.wait_s, a.judge_url_file, a.concurrency, 0.6, 0.95,
                       enable_thinking=None)
    records = list(read_jsonl(out)) if out.exists() else []
    changed = recheck(records, {c["chunk_id"]: c for c in read_jsonl(a.chunks)}, judge, concurrency=a.concurrency)
    if changed:                                       # one client at a time (knowledge.slurm), so a plain rewrite
        tmp = out.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, out)
        log.info("knowledge: %d records checked again with judge v%d", changed, JUDGE_VERSION)
    done = set(current_records(records))
    todo = [c for c in chunks if c["chunk_id"] not in done]
    log.info("knowledge: %d sections, %d done, %d to go", len(chunks), len(done), len(todo))
    stop_at = time.time() + a.stop_new_after if a.stop_new_after else None

    def one(c: dict[str, Any]) -> dict[str, Any] | None:
        if stop_at and time.time() > stop_at:
            return None
        try:
            return process_chunk(c, teacher, judge, per_chunk=a.per_chunk)
        except Exception as e:                        # an endpoint error: the next run retries this section
            log.warning("knowledge %s: %s: %s", c["chunk_id"], type(e).__name__, e)
            return None

    status: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex, out.open("a", encoding="utf-8") as fh:
        for i, rec in enumerate(ex.map(one, todo), start=1):
            if rec is None:
                status["retry_later"] += 1
                continue
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            status.update(x["status"] for x in rec["answers"])
            if i % 50 == 0:
                log.info("knowledge: %d/%d sections; answers %s", i, len(todo), dict(status))
    log.info("knowledge: done; answers %s", dict(status))
    print(json.dumps(dict(status)))


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    li = sub.add_parser("licenses")
    li.add_argument("--corpus", nargs="+", required=True)
    li.add_argument("--out", required=True, type=Path)
    se = sub.add_parser("select")
    se.add_argument("--corpus", nargs="+", required=True)
    se.add_argument("--licenses", required=True, type=Path)
    se.add_argument("--out", required=True, type=Path)
    se.add_argument("--max-papers", type=int, default=3000)
    se.add_argument("--chunks-per-paper", type=int, default=3)
    se.add_argument("--seed", type=int, default=1)
    ru = sub.add_parser("run")
    ru.add_argument("--chunks", required=True)
    ru.add_argument("--out", required=True)
    ru.add_argument("--cache-dir", required=True)
    ru.add_argument("--teacher-url", required=True)
    ru.add_argument("--teacher-model", required=True)
    ru.add_argument("--teacher-url-file", default=None)
    ru.add_argument("--tokenizer", default="Qwen/Qwen3.8-27B", help="renders the teacher's prompts at medium effort")
    ru.add_argument("--judge-url", required=True)
    ru.add_argument("--judge-model", required=True)
    ru.add_argument("--judge-url-file", default=None)
    ru.add_argument("--wait-s", type=float, default=2700.0, help="how long a request waits for a server to return")
    ru.add_argument("--per-chunk", type=int, default=2)
    ru.add_argument("--concurrency", type=int, default=64)
    ru.add_argument("--limit", type=int, default=0)
    ru.add_argument("--stop-new-after", type=float, default=0, help="seconds after which no new section starts")
    td = sub.add_parser("todo", help="print the number of sections without a record")
    td.add_argument("--chunks", required=True)
    td.add_argument("--records", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("--records", required=True)
    ex.add_argument("--chunks", required=True)
    ex.add_argument("--out", required=True, type=Path)
    ex.add_argument("--model-id", default="Qwen/Qwen3.8-27B")
    a = ap.parse_args(argv)
    if a.cmd == "licenses":
        records = list(corpus_records(a.corpus))
        print(json.dumps(fetch_licenses(records, a.out)))
    elif a.cmd == "select":
        licenses = {r["id"]: r for r in read_jsonl(a.licenses)}
        chunks, stats = select_chunks(corpus_records(a.corpus), licenses, max_papers=a.max_papers,
                                      chunks_per_paper=a.chunks_per_paper, seed=a.seed)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open("w", encoding="utf-8") as fh:
            for c in chunks:
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")
        write_json(a.out.with_suffix(".summary.json"), stats)
        print(json.dumps(stats))
    elif a.cmd == "run":
        cmd_run(a)
    elif a.cmd == "todo":
        done = set(current_records(read_jsonl(a.records))) if Path(a.records).exists() else set()
        print(sum(1 for c in read_jsonl(a.chunks) if c["chunk_id"] not in done))
    else:
        chunks = {c["chunk_id"]: c for c in read_jsonl(a.chunks)}
        rows, summary = sft_rows(read_jsonl(a.records), chunks, model_id=a.model_id)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        write_json(a.out.with_suffix(".summary.json"), summary)
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
