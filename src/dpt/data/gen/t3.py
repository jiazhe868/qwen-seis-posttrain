"""T3 report prompts with rubrics, drafted by the teacher (Qwen3.8-27B). Scope: seismology.

    python -m dpt.data.gen.t3 build --work-dir $GEN/t3 --out $GEN/t3_candidates.jsonl \\
        --spot-check $GEN/t3_spot_check.csv --url http://host:8000 --model teacher --tokenizer Qwen/Qwen3.8-27B

1. ``--n-topics`` (30) distinct seismology topics, each with a one-sentence scope.
2. Per topic, batches of ``--batch`` (10) explanatory questions, each with a rubric of 3–6 criteria (weight
   3 = essential, 2 = important, 1 = useful detail), until ``--per-topic`` (50) valid items exist. Each
   batch sees the questions already written for its topic, so it writes different ones.
3. Near-duplicate questions are removed (word-set Jaccard ≥ 0.6, across all topics), and the rest become
   ``t3_report`` candidates: the answer is ``{rubric, reference_length, prompt_extra: {length_hint: 550}}``,
   the same ~550-word target as D-Report (``dpt.data.prompts`` adds "Aim for roughly 550 words.").
4. ``--spot-check`` writes 50 random items to a CSV for a person to read (plan §6: 50 rubrics spot-checked).

``reference_length`` is in the units the T3 length penalty measures (``dpt.rewards.judge.grade_report``):
the router passes no tokenizer, so a report's length is estimated as characters / 3.2. At ~6 characters
per English word, a 550-word report measures ~1,030, and the penalty starts at 1.5× (~825 words).

This module never reads the D-Report drafts or any eval set. Overlaps with them are removed by the
decontamination stage of ``dpt.data.pipeline``.

Resumable: topics (``topics.json``) and accepted items (``items.jsonl``) are kept in ``--work-dir``; a
rerun continues where the last one stopped.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from dpt.common.io import read_json, write_json
from dpt.common.logging import get_logger
from dpt.common.text import jaccard, normalize_text, tokenize
from dpt.common.types import T3_REPORT
from dpt.data.gen.common import candidate, rng_for, write_candidates
from dpt.data.gen.teacher import Appender, ChatModel, add_model_args, last_json, model_from_args, read_existing

log = get_logger(__name__)

LENGTH_HINT_WORDS = 550
CHARS_PER_WORD = 6.0          # English technical prose, including the space and punctuation


def reference_length(words: int = LENGTH_HINT_WORDS) -> int:
    """``reference_length`` in the length penalty's units (characters / est_chars_per_token)."""
    from dpt.rewards.judge import ReportConfig

    return int(round(words * CHARS_PER_WORD / ReportConfig().est_chars_per_token))


TOPICS_PROMPT = """You are planning training material on seismology for a language model. List {n} distinct
topics within seismology that a specialist could write short technical reports about. Cover the field
broadly: earthquake source physics, seismic wave propagation, Earth structure from seismology, instruments
and data processing, earthquake location and magnitude, seismic hazard and engineering seismology,
induced and volcanic seismicity, and exploration and near-surface seismology. Topics must not overlap.
{avoid}
Return JSON only:
{{"topics": [{{"topic": "<3 to 8 words>", "scope": "<one sentence on what the topic covers>"}}]}}"""

ITEMS_PROMPT = """Write {k} questions for short technical reports on this seismology topic.

Topic: {topic}
Scope: {scope}
{already}
Each question:
- asks for an explanation (mechanisms, quantitative reasoning, methods or trade-offs), not a list of facts,
  and a specialist can answer it well in about {words} words;
- is answerable from established knowledge: no data lookup, no specific recent events or publications,
  no opinions;
- is self-contained and unambiguous, and does not state the length.

For each question write a rubric of 3 to 6 criteria that a correct, complete answer must meet:
- each criterion is one point of established knowledge that a grader can check from the answer;
- weight 3 = essential, 2 = important, 1 = useful detail; at least one criterion has weight 3.

Return JSON only:
{{"items": [{{"question": "...", "rubric": [{{"criterion": "...", "weight": 3}}]}}]}}"""

SPOT_CHECK_COLUMNS = ["uid", "topic", "question", "rubric"]


# --------------------------------------------------------------------------- parsing


def _words(text: str) -> list[str]:
    return [t for t in tokenize(normalize_text(text)) if re.match(r"\w", t)]


def parse_topics(text: str) -> list[dict[str, str]]:
    out = []
    for t in last_json(text, "topics") or []:
        if not isinstance(t, dict):
            continue
        topic, scope = str(t.get("topic", "")).strip(), str(t.get("scope", "")).strip()
        if 1 <= len(topic.split()) <= 12 and scope:
            out.append({"topic": topic, "scope": scope})
    return out


def clean_item(obj: Any) -> dict[str, Any] | None:
    """A valid ``{question, rubric}`` or None. Invalid criteria are dropped; ≥ 3 must remain (≤ 6 kept)."""
    if not isinstance(obj, dict):
        return None
    q = " ".join(str(obj.get("question", "")).split())
    if not (12 <= len(q.split()) <= 150):
        return None
    rubric, seen = [], set()
    for c in obj.get("rubric") or []:
        if not isinstance(c, dict):
            continue
        crit = " ".join(str(c.get("criterion", "")).split())
        try:
            w = int(round(float(c.get("weight"))))
        except (TypeError, ValueError):
            continue
        key = normalize_text(crit)
        if w not in (1, 2, 3) or len(crit.split()) < 3 or key in seen:
            continue
        seen.add(key)
        rubric.append({"criterion": crit, "weight": w})
    if len(rubric) < 3:
        return None
    return {"question": q, "rubric": rubric[:6]}


def parse_items(text: str) -> list[dict[str, Any]]:
    return [it for it in (clean_item(o) for o in (last_json(text, "items") or [])) if it]


# --------------------------------------------------------------------------- drafting


def draft_topics(model: ChatModel, n: int, *, max_rounds: int = 4, seed: int = 1) -> list[dict[str, str]]:
    topics: list[dict[str, str]] = []
    for r in range(max_rounds):
        if len(topics) >= n:
            break
        avoid = ("" if not topics else
                 "Do not repeat any of these topics:\n" + "\n".join(f"- {t['topic']}" for t in topics) + "\n")
        msgs = [{"role": "user", "content": TOPICS_PROMPT.format(n=n - len(topics), avoid=avoid)}]
        for reply in model.generate(msgs, n=1, seed=seed + r):
            try:
                new = parse_topics(reply.text) if not reply.truncated else []
            except ValueError as e:
                log.warning("topics: unreadable reply (%s)", e)
                continue
            known = {normalize_text(t["topic"]) for t in topics}
            topics += [t for t in new if normalize_text(t["topic"]) not in known]
    if len(topics) < n:
        log.warning("topics: only %d of %d after %d rounds", len(topics), n, max_rounds)
    return topics[:n]


def draft_items_for_topic(model: ChatModel, topic: dict[str, str], idx: int, have: list[str], *, per_topic: int,
                          batch: int, out: Appender, seed: int, max_extra_rounds: int = 3) -> int:
    """Draft until ``per_topic`` items exist for this topic; appends each accepted item to ``out``."""
    written = list(have)
    n_have = len(written)
    rounds = 0
    max_rounds = -(-per_topic // batch) + max_extra_rounds
    while len(written) < per_topic and rounds < max_rounds:
        k = min(batch, per_topic - len(written))
        already = ("" if not written else "Questions already written for this topic (write different ones):\n"
                   + "\n".join(f"- {q}" for q in written[-30:]) + "\n")
        prompt = ITEMS_PROMPT.format(k=k, topic=topic["topic"], scope=topic["scope"], already=already,
                                     words=LENGTH_HINT_WORDS)
        rounds += 1
        try:
            reply = model.generate([{"role": "user", "content": prompt}], n=1, seed=seed + 1000 * idx + rounds)[0]
            items = [] if reply.truncated else parse_items(reply.text)
        except (OSError, ConnectionError):
            raise                               # the server is gone: stop, the caller retries (rounds not used up)
        except Exception as e:  # one bad batch must not stop the topic; the round counts
            if _transport_error(e):
                raise
            log.warning("topic %d (%s), round %d: %s", idx, topic["topic"], rounds, e)
            continue
        fresh = []
        for it in items:                               # not a near-duplicate of anything kept so far
            if len(fresh) < k and not _near(it["question"], written):
                fresh.append(it)
                written.append(it["question"])
        out.write({"topic_idx": idx, "topic": topic["topic"], "round": rounds, **it} for it in fresh)
    return len(written) - n_have


def _transport_error(e: BaseException) -> bool:
    """A lost server (httpx transport / 5xx after retries), not a bad reply: must not count as a drafting round."""
    try:
        import httpx
    except ImportError:  # pragma: no cover
        return False
    return isinstance(e, (httpx.TransportError, httpx.HTTPStatusError))


def _near(q: str, others: Sequence[str], threshold: float = 0.6) -> bool:
    w = set(_words(q))
    return any(jaccard(w, set(_words(o))) >= threshold for o in others)


def dedupe(items: Sequence[dict[str, Any]], threshold: float = 0.6) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    sets: list[set[str]] = []
    dropped = 0
    for it in items:
        w = set(_words(it["question"]))
        if any(jaccard(w, s) >= threshold for s in sets):
            dropped += 1
            continue
        kept.append(it)
        sets.append(w)
    return kept, dropped


def to_candidates(items: Sequence[dict[str, Any]], model_id: str) -> list[dict[str, Any]]:
    ref = reference_length()
    return [candidate(T3_REPORT, it["question"],
                      {"rubric": it["rubric"], "reference_length": ref,
                       "prompt_extra": {"length_hint": LENGTH_HINT_WORDS}},
                      origin="dpt.data.gen.t3", source={"model": model_id}, topic=it["topic"],
                      family=f"t3:{_slug(it['topic'])}")
            for it in items]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:48]


def write_spot_check(path: str | Path, cands: Sequence[dict[str, Any]], *, n: int = 50, seed: int = 1) -> int:
    rows = list(cands)
    rng_for(seed, "t3-spot-check").shuffle(rows)
    rows = rows[:n]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SPOT_CHECK_COLUMNS)
        w.writeheader()
        for c in rows:
            w.writerow({"uid": c["uid"], "topic": c["meta"]["topic"], "question": c["question"],
                        "rubric": " | ".join(f"[{r['weight']}] {r['criterion']}" for r in c["answer"]["rubric"])})
    return len(rows)


def build(model: ChatModel, work_dir: str | Path, out: str | Path, *, n_topics: int = 30, per_topic: int = 50,
          batch: int = 10, seed: int = 1, concurrency: int = 16, spot_check: str | Path | None = None
          ) -> dict[str, Any]:
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    tpath = work / "topics.json"
    if tpath.exists():
        topics = read_json(tpath)
    else:
        topics = draft_topics(model, n_topics, seed=seed)
        write_json(tpath, topics)
    log.info("t3: %d topics", len(topics))
    items_path = work / "items.jsonl"
    have: dict[int, list[str]] = {}
    for r in read_existing(items_path):
        have.setdefault(int(r["topic_idx"]), []).append(r["question"])
    app = Appender(items_path)

    def one(i: int) -> int:
        return draft_items_for_topic(model, topics[i], i, have.get(i, []), per_topic=per_topic, batch=batch,
                                     out=app, seed=seed)

    with ThreadPoolExecutor(max(1, concurrency)) as ex:
        added = sum(ex.map(one, range(len(topics))))
    items = [r for r in read_existing(items_path) if int(r["topic_idx"]) < len(topics)]
    per: dict[int, int] = {}
    capped = []
    for it in items:                                   # keep at most per_topic per topic, in drafting order
        if per.get(it["topic_idx"], 0) < per_topic:
            per[it["topic_idx"]] = per.get(it["topic_idx"], 0) + 1
            capped.append(it)
    kept, dups = dedupe(capped)
    cands = to_candidates(kept, model.model_id)
    write_candidates(out, cands)
    n_spot = write_spot_check(spot_check, cands, seed=seed) if spot_check else 0
    stats = {"topics": len(topics), "drafted_this_run": added, "items": len(capped), "near_duplicates": dups,
             "candidates": len(cands), "spot_check": n_spot, "model": model.model_id,
             "reference_length": reference_length(), "length_hint_words": LENGTH_HINT_WORDS}
    write_json(work / "stats.json", stats)
    log.info("t3: %s", json.dumps(stats))
    return stats


def main(argv: Sequence[str] | None = None, *, model: ChatModel | None = None) -> int:
    ap = argparse.ArgumentParser(description="Draft T3 report prompts and rubrics with the teacher")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--work-dir", required=True, help="topics.json + items.jsonl (resumable)")
    b.add_argument("--out", required=True, help="t3_report candidates JSONL")
    b.add_argument("--spot-check", default=None, help="CSV of 50 random items for a person to read")
    b.add_argument("--n-topics", type=int, default=30)
    b.add_argument("--per-topic", type=int, default=50)
    b.add_argument("--batch", type=int, default=10)
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--concurrency", type=int, default=16)
    if model is None:
        add_model_args(b, max_tokens=16384)
    args = ap.parse_args(argv)
    m = model or model_from_args(args, concurrency=args.concurrency)
    stats = build(m, args.work_dir, args.out, n_topics=args.n_topics, per_topic=args.per_topic, batch=args.batch,
                  seed=args.seed, concurrency=args.concurrency, spot_check=args.spot_check)
    return 0 if stats["candidates"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
