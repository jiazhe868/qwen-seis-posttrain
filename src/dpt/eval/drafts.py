"""Eval-side drafting with the hosted model (GPT-6 Luna on the 9B plan, README §3.1).

No person reviews these drafts. A model of another family (the Gemma judge on the cluster) checks every
item before it becomes an eval set (``dpt.eval.autocheck``):

    python -m dpt.eval.drafts topics  --scope "solid-earth geophysics: ..." --n 40 --out topics.csv
    python -m dpt.eval.drafts report  --topics topics.csv --per-topic 4 --out d_report_drafts.csv
    python -m dpt.eval.drafts fresh   --topics topics.csv --per-topic 3 --out fresh_drafts.jsonl
    python -m dpt.eval.drafts wording --items $SECURE/d_num.jsonl --out d_num_wording.csv

* ``topics``: distinct topics for the scope. Trim the list if you like; nothing else needs editing.
* ``report``: D-Report questions sized for ~500–600-word answers, with rubrics, in the columns of
  :data:`REPORT_COLUMNS` (``status`` = ``draft``). ``autocheck report`` sets the status and keeps 100.
* ``fresh``: D-Fresh problems: multi-step numeric problems, each with the stated answer and a
  ``solve()`` function that computes it. ``autocheck fresh`` runs the code in the sandbox and has
  Gemma solve each problem from the text alone.
* ``wording``: the hosted model answers each generated D-Num item from the question alone. The CSV
  flags items where its answer disagrees with the solver's label or where it finds the wording
  ambiguous. Disagreement is a reason to look, not a verdict: fix the template or drop the item.

Rules (``dpt.eval.hosted``): the output is eval material only. OpenAI's terms forbid using it to
develop competing models, so none of it enters training data (decontamination also removes any
training item that resembles an eval item). The model that wrote D-Fresh never reviews it: ``wording``
refuses D-Fresh, whose check is ``autocheck fresh``.

Run from any machine that reaches the API (the workstation, or a cluster login node: this is
light, network-bound work) with the key in the variable named by
``project.judges.second.api_key_env``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from dpt.common.logging import get_logger
from dpt.eval.hosted import ChatEndpoint, endpoint_from_config

log = get_logger(__name__)

REPORT_COLUMNS = [
    "question", "rubric_criteria", "rubric_weights", "reference_length_words", "common_errors_to_penalize",
    "author", "date_written", "topic", "status", "drafted_by",
]
REPORT_WORDS = (500, 600)          # target answer length for D-Report (and T3): the blind review reads them
WORDING_COLUMNS = ["id", "flag", "agrees_with_label", "wording", "issue", "label", "hosted_answer", "question"]

TOPICS_PROMPT = """You are designing a private evaluation set of long-form technical report questions for
graduate-level work in this scope:

{scope}

List {n} distinct topics. Each topic must be specific enough for several different report questions,
cover a different part of the scope from the others, and be answerable from established knowledge
(no breaking news, no questions that need data lookup).
{avoid}
Return only JSON: {{"topics": [{{"topic": "...", "scope": "two or three sentences: what it covers"}}]}}"""

REPORT_PROMPT = """You are drafting questions for a private evaluation set. A model of another family
checks every item; no person edits them, so each must be right as written. Topic: {topic}
Topic scope: {topic_scope}

Write {k} different report questions on this topic. Each question asks for a short technical report a
specialist would write for a technical reader, answerable well in {min_words}-{max_words} words: it must
require explaining mechanisms, quantitative reasoning or trade-offs, not listing facts. Do not copy or
closely paraphrase textbook exercises or well-known exam questions. Do not require a reference list
(named, specific examples are fine). Do not state the length in the question.

For each question give a rubric of 4-6 criteria that a correct, complete answer must meet. Each
criterion is one checkable statement of established knowledge (no contested claims); weight 1-3
(3 = essential). Also give the common errors a grader should penalize: only claims that are wrong.

Return only JSON: {{"questions": [{{"question": "...", "rubric": [{{"criterion": "...", "weight": 2}}],
"common_errors": ["..."]}}]}}"""

FRESH_PROMPT = """You are writing numeric problems for a private evaluation set. Topic: {topic}
Topic scope: {topic_scope}

Write {k} new problems on this topic of the kind a graduate course or a working specialist would pose.
Requirements for each problem:
- at least two dependent steps (e.g. derive an intermediate quantity, then use it; or combine two
  relations; or convert units along the way); not a single plug-in formula;
- every quantity needed is given in the text with its unit (standard physical constants excepted);
- exactly one final numeric answer with a physical unit; one reasonable reading only;
- realistic values; vary the form across problems (word problem, a small table of data, a design or
  inverse question); new numbers and settings, not a copy of a textbook exercise.
For each give the answer, a short worked solution, and a Python function `solve()` that computes the
final answer from the numbers in the text and returns it as a float in `answer_unit`. Use only the
`math` module and numpy; no input, files or printing.

Return only JSON: {{"problems": [{{"question": "...", "answer_value": 1.23, "answer_unit": "km/s",
"rel_tolerance_percent": 1, "solution": "...", "solve_py": "import math\n\ndef solve():\n    ..."}}]}}"""

WORDING_PROMPT = """Solve this problem. Use only the information given and standard physical constants.

{question}

After your solution, write exactly two final lines:
WORDING: clear   (or)   WORDING: ambiguous - <what is missing, or the two readings that give different answers>
\\boxed{{<value> <unit>}}"""

_WORDING_RE = re.compile(r"^\s*WORDING:\s*(clear|ambiguous)\b[\s:\-—]*(.*)$", re.IGNORECASE | re.MULTILINE)
_WORDS = re.compile(r"[a-z0-9]+")


# --------------------------------------------------------------------------- helpers


def parse_json_reply(text: str) -> Any:
    """The JSON object in a model reply (tolerates code fences and text around it)."""
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.DOTALL)
    if fence:
        t = fence.group(1).strip()
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in reply")
    return json.loads(t[start : end + 1])


def _word_set(text: str) -> set[str]:
    return set(_WORDS.findall(text.lower()))


def near_duplicate(a: str, b: str, threshold: float = 0.6) -> bool:
    """Word-set Jaccard similarity at or above ``threshold``."""
    wa, wb = _word_set(a), _word_set(b)
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= threshold


def _dedupe(texts_rows: list[tuple[str, dict[str, Any]]], threshold: float) -> list[dict[str, Any]]:
    kept: list[tuple[str, dict[str, Any]]] = []
    for text, row in texts_rows:
        if any(near_duplicate(text, k, threshold) for k, _ in kept):
            continue
        kept.append((text, row))
    return [r for _, r in kept]


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- drafting


def draft_topics(chat: ChatEndpoint, scope: str, n: int, *, avoid: Sequence[str] = ()) -> list[dict[str, str]]:
    avoid_txt = ("Do not repeat these existing topics: " + "; ".join(avoid) + "\n") if avoid else ""
    reply = chat.request([{"role": "user", "content": TOPICS_PROMPT.format(scope=scope, n=n, avoid=avoid_txt)}])
    topics = parse_json_reply(reply.content).get("topics") or []
    rows = [(str(t.get("topic", "")).strip(), {"topic": str(t.get("topic", "")).strip(),
                                               "scope": str(t.get("scope", "")).strip()})
            for t in topics if isinstance(t, dict) and str(t.get("topic", "")).strip()]
    return _dedupe(rows, 0.7)[:n]


def _report_row(q: dict[str, Any], topic: str, drafted_by: str, date: str) -> dict[str, Any] | None:
    question = str(q.get("question", "")).strip()
    rubric = [r for r in (q.get("rubric") or []) if isinstance(r, dict) and str(r.get("criterion", "")).strip()]
    if len(question.split()) < 8 or not 3 <= len(rubric) <= 8:
        return None
    weights = []
    for r in rubric:
        try:
            weights.append(min(3, max(1, int(r.get("weight", 1)))))
        except (TypeError, ValueError):
            weights.append(1)
    length = sum(REPORT_WORDS) // 2
    errors = q.get("common_errors") or []
    return {
        "question": question,
        "rubric_criteria": " | ".join(str(r["criterion"]).strip().replace("|", "/") for r in rubric),
        "rubric_weights": "|".join(map(str, weights)),
        "reference_length_words": length,
        "common_errors_to_penalize": "; ".join(str(e).strip() for e in errors if str(e).strip()),
        "author": "",
        "date_written": date,
        "topic": topic,
        "status": "draft",
        "drafted_by": drafted_by,
    }


def draft_reports(
    chat: ChatEndpoint,
    topics: Sequence[dict[str, str]],
    per_topic: int,
    *,
    max_concurrency: int = 4,
    date: str | None = None,
    dedupe_threshold: float = 0.6,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Report questions + rubrics for every topic. Returns (rows, counts of kept/rejected/failed)."""
    date = date or _dt.date.today().isoformat()

    def one(t: dict[str, str]) -> tuple[list[dict[str, Any]], int, int]:
        prompt = REPORT_PROMPT.format(topic=t["topic"], topic_scope=t.get("scope", ""), k=per_topic,
                                      min_words=REPORT_WORDS[0], max_words=REPORT_WORDS[1])
        try:
            reply = chat.request([{"role": "user", "content": prompt}])
            qs = parse_json_reply(reply.content).get("questions") or []
        except Exception as e:  # one bad topic must not lose the others
            log.warning("topic %r failed: %s", t["topic"], e)
            return [], 0, 1
        rows = [_report_row(q, t["topic"], reply.model or chat.model, date) for q in qs if isinstance(q, dict)]
        good = [r for r in rows if r is not None]
        return good, len(rows) - len(good), 0

    with ThreadPoolExecutor(max(1, max_concurrency)) as ex:
        results = list(ex.map(one, topics))
    rows = [r for good, _, _ in results for r in good]
    kept = _dedupe([(r["question"], r) for r in rows], dedupe_threshold)
    counts = {"kept": len(kept), "near_duplicate": len(rows) - len(kept),
              "malformed": sum(m for _, m, _ in results), "failed_topics": sum(f for _, _, f in results)}
    return kept, counts


def _fresh_row(q: dict[str, Any], topic: str, drafted_by: str, date: str) -> dict[str, Any] | None:
    question = str(q.get("question", "")).strip()
    unit = str(q.get("answer_unit", "")).strip()
    code = str(q.get("solve_py", "")).strip()
    try:
        value = float(q.get("answer_value"))
    except (TypeError, ValueError):
        return None
    malformed = len(question.split()) < 15 or not unit or "def solve(" not in code
    if malformed or not math.isfinite(value) or value == 0.0:
        return None
    try:
        tol = min(5.0, max(0.1, float(q.get("rel_tolerance_percent") or 1.0)))
    except (TypeError, ValueError):
        tol = 1.0
    return {
        "draft_id": hashlib.sha256(question.encode()).hexdigest()[:12],
        "topic": topic,
        "question": question,
        "answer_value": value,
        "answer_unit": unit,
        "rel_tolerance_percent": tol,
        "solution": str(q.get("solution", "")).strip(),
        "solve_py": code,
        "drafted_by": drafted_by,
        "date_written": date,
    }


def draft_fresh(
    chat: ChatEndpoint,
    topics: Sequence[dict[str, str]],
    per_topic: int,
    *,
    max_concurrency: int = 4,
    date: str | None = None,
    dedupe_threshold: float = 0.6,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Numeric problems with answers and solver code for every topic. Returns (rows, counts)."""
    date = date or _dt.date.today().isoformat()

    def one(t: dict[str, str]) -> tuple[list[dict[str, Any]], int, int]:
        prompt = FRESH_PROMPT.format(topic=t["topic"], topic_scope=t.get("scope", ""), k=per_topic)
        try:
            reply = chat.request([{"role": "user", "content": prompt}])
            qs = parse_json_reply(reply.content).get("problems") or []
        except Exception as e:  # one bad topic must not lose the others
            log.warning("topic %r failed: %s", t["topic"], e)
            return [], 0, 1
        rows = [_fresh_row(q, t["topic"], reply.model or chat.model, date) for q in qs if isinstance(q, dict)]
        good = [r for r in rows if r is not None]
        return good, len(rows) - len(good), 0

    with ThreadPoolExecutor(max(1, max_concurrency)) as ex:
        results = list(ex.map(one, topics))
    rows = [r for good, _, _ in results for r in good]
    kept = _dedupe([(r["question"], r) for r in rows], dedupe_threshold)
    counts = {"kept": len(kept), "near_duplicate": len(rows) - len(kept),
              "malformed": sum(m for _, m, _ in results), "failed_topics": sum(f for _, _, f in results)}
    return kept, counts


# --------------------------------------------------------------------------- wording review


def refuse_fresh(path: str | Path, items: Sequence[Any]) -> None:
    if "fresh" in Path(path).name.lower() or any("fresh" in str(getattr(it, "set", "")).lower() for it in items):
        raise SystemExit("D-Fresh was written by the hosted model, so it cannot check it: the check is "
                         "`python -m dpt.eval.autocheck fresh` (another model family)")


def review_wording(
    chat: ChatEndpoint, items: Sequence[Any], *, max_concurrency: int = 8, missing_unit: str = "strict"
) -> list[dict[str, Any]]:
    """Per item: the hosted model's answer vs the label, and its view of the wording."""
    from dpt.eval.graders.extract import extract_final_answer
    from dpt.eval.graders.numeric import grade_numeric

    def one(it: Any) -> dict[str, Any]:
        row: dict[str, Any] = {"id": it.id, "question": it.question,
                               "label": f"{it.answer.get('value')} {it.answer.get('unit', '')}".strip()}
        try:
            reply = chat.request([{"role": "user", "content": WORDING_PROMPT.format(question=it.question)}])
        except Exception as e:
            return {**row, "flag": "error", "agrees_with_label": "", "wording": "", "issue": str(e)}
        if reply.finish_reason == "length":
            return {**row, "flag": "unanswered", "agrees_with_label": "", "wording": "",
                    "issue": "reply truncated (raise max_tokens)"}
        g = grade_numeric(reply.content, it.answer, missing_unit=missing_unit)
        m = list(_WORDING_RE.finditer(reply.content))
        wording, issue = (m[-1].group(1).lower(), m[-1].group(2).strip()) if m else ("unstated", "")
        agrees = "yes" if g.correct else ("no" if g.reason in ("ok", "wrong_value", "unit_mismatch") else g.reason)
        flag = "ok" if agrees == "yes" and wording == "clear" else "check"
        return {**row, "flag": flag, "agrees_with_label": agrees, "wording": wording, "issue": issue,
                "hosted_answer": extract_final_answer(reply.content).answer or ""}

    with ThreadPoolExecutor(max(1, max_concurrency)) as ex:
        rows = list(ex.map(one, items))
    return sorted(rows, key=lambda r: (r["flag"] == "ok", r["id"]))


def regrade_wording(rows: Sequence[dict[str, Any]], items: Sequence[Any], *,
                    missing_unit: str = "strict") -> list[dict[str, Any]]:
    """Recompute agreement from the saved hosted answers (no API call), e.g. after a grader fix.

    The wording verdict is kept; an item is ``ok`` again only if the regraded answer agrees with the label
    and the wording was ``clear``. Rows without a saved answer (errors, truncations) are left as they were.
    """
    from dpt.eval.graders.numeric import grade_numeric

    by_id = {it.id: it for it in items}
    out = []
    for r in rows:
        it, ans = by_id.get(r["id"]), (r.get("hosted_answer") or "").strip()
        if it is None or not ans:
            out.append(dict(r))
            continue
        g = grade_numeric(f"\\boxed{{{ans}}}", it.answer, missing_unit=missing_unit)
        agrees = "yes" if g.correct else ("no" if g.reason in ("ok", "wrong_value", "unit_mismatch") else g.reason)
        flag = "ok" if agrees == "yes" and r.get("wording") == "clear" else "check"
        out.append({**r, "agrees_with_label": agrees, "flag": flag})
    return sorted(out, key=lambda r: (r["flag"] == "ok", r["id"]))


# --------------------------------------------------------------------------- CLI


def _hosted(model_line: str | None, max_tokens: int | None) -> ChatEndpoint:
    from dpt.common.config import load_config, to_container

    cfg = to_container(load_config("eval/eval_suite.yaml", model_line=model_line))["eval_suite"]
    jcfg = dict(cfg.get("second_judge") or {})
    if max_tokens:
        jcfg["max_tokens"] = max_tokens
    chat = endpoint_from_config(jcfg, ChatEndpoint)
    if chat is None:
        raise SystemExit("hosted model not usable here: set project.judges.second (model line), its API key "
                         "and run where the endpoint is reachable")
    return chat


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Draft eval material with the hosted model for human review")
    ap.add_argument("--model-line", default=None, help="default: $DPT_MODEL_LINE")
    ap.add_argument("--max-tokens", type=int, default=None, help="override the hosted model's budget")
    ap.add_argument("--max-concurrency", type=int, default=4)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("topics")
    t.add_argument("--scope", required=True, help="the domain scope (plan §5 decision 4), a few sentences")
    t.add_argument("--n", type=int, default=40)
    t.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("--topics", required=True, help="CSV with columns topic, scope (from `topics`, edited)")
    r.add_argument("--per-topic", type=int, default=4, help="40 topics × 4 ≈ 160 drafts → 100 kept")
    r.add_argument("--out", required=True)
    f = sub.add_parser("fresh")
    f.add_argument("--topics", required=True, help="CSV with columns topic, scope")
    f.add_argument("--per-topic", type=int, default=3, help="40 topics × 3 ≈ 120 drafts → 60 kept")
    f.add_argument("--out", required=True, help="JSONL (it holds code)")
    w = sub.add_parser("wording")
    w.add_argument("--items", required=True, help="eval JSONL (generated D-Num; never D-Fresh)")
    w.add_argument("--set-name", default="d_num")
    w.add_argument("--out", required=True)
    w.add_argument("--regrade", default=None,
                   help="a wording CSV from an earlier run: regrade its saved answers, no API call")
    args = ap.parse_args(argv)
    if args.cmd == "wording" and args.regrade:
        from dpt.eval.datasets import load_eval_items

        items = load_eval_items(args.items, args.set_name)
        rows = regrade_wording(read_csv(args.regrade), items)
        write_csv(args.out, rows, WORDING_COLUMNS)
        log.info("regraded: %d of %d items to check → %s", sum(r["flag"] != "ok" for r in rows), len(rows), args.out)
        return 0
    chat = _hosted(args.model_line, args.max_tokens)

    if args.cmd == "topics":
        rows = draft_topics(chat, args.scope, args.n)
        write_csv(args.out, rows, ["topic", "scope"])
        log.info("%d topics → %s (edit, then: report --topics %s)", len(rows), args.out, args.out)
    elif args.cmd == "report":
        topics = [row for row in read_csv(args.topics) if (row.get("topic") or "").strip()]
        rows, counts = draft_reports(chat, topics, args.per_topic, max_concurrency=args.max_concurrency)
        write_csv(args.out, rows, REPORT_COLUMNS)
        log.info("%s → %s (next: autocheck report on the cluster)", counts, args.out)
    elif args.cmd == "fresh":
        from dpt.common.io import write_jsonl

        topics = [row for row in read_csv(args.topics) if (row.get("topic") or "").strip()]
        rows, counts = draft_fresh(chat, topics, args.per_topic, max_concurrency=args.max_concurrency)
        write_jsonl(args.out, rows)
        log.info("%s → %s (next: autocheck fresh on the cluster)", counts, args.out)
    else:
        from dpt.eval.datasets import load_eval_items

        items = load_eval_items(args.items, args.set_name)
        refuse_fresh(args.items, items)
        rows = review_wording(chat, items, max_concurrency=args.max_concurrency)
        write_csv(args.out, rows, WORDING_COLUMNS)
        n_check = sum(r["flag"] != "ok" for r in rows)
        log.info("%d of %d items to check → %s", n_check, len(rows), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
