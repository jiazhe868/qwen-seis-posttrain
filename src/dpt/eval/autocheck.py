"""Automatic checks that replace expert review of the drafted eval sets (README §3.1).

GPT-6 Luna drafts the D-Fresh problems and the D-Report questions (``dpt.eval.drafts``). No person
reviews them. A model of another family, the Gemma judge served on the cluster, checks every item, and
only items that pass become eval sets:

    python -m dpt.eval.autocheck fresh  --drafts fresh_drafts.jsonl  --out $SECURE/d_fresh.jsonl
    python -m dpt.eval.autocheck report --drafts d_report_drafts.csv --out $SECURE/d_report.jsonl

``scripts/slurm/eval_sets.slurm`` runs both on a compute node (sandbox container + judge endpoint).
Gemma thinks before every reply, with a budget large enough for long multi-step work
(``eval_suite.autocheck.checker.max_tokens``); a reply cut off at the budget is not used, and the log
counts them.

``fresh``, per problem:

1. The stated answer passes the numeric grader against itself, so its unit parses.
2. The drafter's ``solve()`` reproduces the stated answer to 0.1%. It runs in the T2 sandbox (the
   Apptainer container on the cluster) after the static scan for exits, environment probes and network use.
3. Gemma solves the problem ``solves`` times from the text alone. Matching answers count as agreement;
   the other answers are grouped by value, and the largest group is the rival answer.
4. Adjudication, when the votes do not settle it: no solution matches, or the rival answer has at least
   two votes and at least as many as the stated one. Gemma gets the two worked solutions, the drafter's
   and its own for the rival answer, labelled A and B without saying whose is whose, and must find the
   wrong step. It runs in both orders. The problem is kept only if Gemma upholds the stated answer and
   names the error in the other solution both times; it is dropped if either order finds the problem
   ambiguous, if both reject the stated answer, or if the two orders disagree.
5. Otherwise the problem is kept if at least ``min_agree`` solutions match. One in four is enough: a
   problem is dropped for having no independent confirmation, not for being hard.

``report``, per question, ``votes`` independent passes (majority decides):

1. Gemma writes its own answer to the question, without seeing the rubric.
2. With that answer in front of it, Gemma rules on the question (clear, answerable in the target length,
   no data lookup), on every rubric criterion (correct? checkable?) and on every listed common error
   (really an error?).

A criterion or error is removed, and the question dropped, only when more than half of the readable
votes say so. The question is also dropped if an essential criterion (weight 3) is removed, if fewer
than ``min_criteria`` remain, or if fewer than two votes can be read.

Both: near-duplicates are removed, ``n_keep`` items are chosen round-robin over topics, and the set is
written with ids, a canary and a manifest whose stage statistics record every drop. ``--log`` writes one
CSV row per draft with its status, the reason, the vote counts and the adjudication.

What this cannot catch: an error that Luna and Gemma share. More thinking and more votes make Gemma more
careful, not more knowledgeable. The two-judge D-Report gate and the final blind review of 20 pairs
(``dpt.eval.blind_review``) are the remaining cross-family and human checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from dpt.chat.thinking import THINK_END
from dpt.common.io import read_jsonl, write_jsonl
from dpt.common.logging import get_logger
from dpt.eval.drafts import REPORT_COLUMNS, REPORT_WORDS, near_duplicate, read_csv, write_csv
from dpt.eval.hosted import ChatEndpoint, endpoint_from_config

log = get_logger(__name__)

SOLVE_PROMPT = """Solve this problem. Use only the information given and standard physical constants.

{question}

Write the worked solution you would hand in between the lines <solution> and </solution>. End it with the
final answer on its own line as \\boxed{{<value> <unit>}}."""

ADJUDICATE_PROMPT = """Two solutions of the problem below reach different answers. At most one of them is right,
unless the problem itself is ambiguous.

## Problem
{question}

## Solution A
{a}

Final answer A: {answer_a}

## Solution B
{b}

Final answer B: {answer_b}

Work the problem through yourself first. Then check both solutions step by step against your work: the
reading of the problem, every number and unit taken from it, each formula, and the arithmetic.

Decide:
- correct: "A" or "B" if that solution is right and the other is wrong; "neither" if both are wrong.
- ambiguous: true only if the problem as written honestly supports both readings.
- wrong_step: the first wrong step of the solution you reject, quoted or described exactly.

Return JSON only:
{{"correct": "A", "ambiguous": false, "wrong_step": ""}}"""

REPORT_ANSWER_PROMPT = """Answer this question as a specialist would, in about {words} words. Write the answer you
would hand in between the lines <answer> and </answer>.

{question}"""

REPORT_CHECK_PROMPT = """You are the only check on a question that another model drafted for a private evaluation
set of short technical reports. Nobody else reviews it, so be strict: reject anything you would not
defend to a specialist.

## Question
{question}

Target answer length: about {words} words.

## Your own answer, written before you saw the rubric
{own}

## Rubric: what a correct, complete answer must contain
{criteria}

## Common errors the grader will penalize
{errors}

Compare every rubric criterion and every common error with your own answer and with what you know. Where
they disagree, work out which is right; do not assume the rubric is.

Decide:
- question.ok: false if the question is ambiguous, rests on a false premise, needs data lookup or recent
  events, or cannot be answered well in the target length.
- For every rubric criterion: correct = "yes" if it states established knowledge, "no" if it is wrong or
  contested, "unsure" otherwise; checkable = "yes" if a grader can tell from an answer whether it is met.
- For every common error: really_an_error = "no" if the claim is in fact correct (penalizing it would
  be wrong), otherwise "yes".

Return JSON only:
{{"question": {{"ok": true, "issue": ""}},
 "criteria": [{{"id": 1, "correct": "yes", "checkable": "yes", "issue": ""}}],
 "common_errors": [{{"id": 1, "really_an_error": "yes"}}]}}"""

# Static-scan categories that keep drafted solver code out of the sandbox altogether.
BLOCKING_CODE_HITS = frozenset({"exit_call", "env_probe", "env_access", "test_tamper", "test_dir_read",
                                "introspection", "network_access"})

CODE_TEST = """import math
import solution


def test_reproduces_the_stated_answer():
    value = float(solution.solve())
    assert math.isfinite(value)
    assert abs(value - {stated!r}) <= {rel!r} * abs({stated!r})
"""

FRESH_LOG_COLUMNS = ["draft_id", "status", "reason", "agree", "rival", "readable", "truncated", "solves",
                     "adjudication", "adjudication_note", "topic", "answer", "rival_answer", "question"]
REPORT_LOG_COLUMNS = REPORT_COLUMNS + ["check_issue", "check_votes"]
NOTE_CHARS = 300                     # per adjudication note in the log


# --------------------------------------------------------------------------- helpers


def parse_last_json(text: str, required: str) -> dict[str, Any]:
    """The last JSON object in ``text`` that has the key ``required`` (skips any reasoning before it)."""
    dec, t = json.JSONDecoder(), text or ""
    pos = t.rfind("{")
    while pos >= 0:
        try:
            obj, _ = dec.raw_decode(t, pos)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and required in obj:
            return obj
        pos = t.rfind("{", 0, pos)
    raise ValueError(f"no JSON object with {required!r} in the reply")


def last_block(text: str, tag: str, *, max_chars: int = 6000) -> str:
    """The last ``<tag>…</tag>`` block of a reply, or the reply's last ``max_chars`` characters without one.

    The judge server runs no reasoning parser, so a reply can start with the model's thinking. The block,
    or failing that the end of the reply, is the part the model meant to hand in.
    """
    t = text or ""
    if THINK_END in t:
        t = t[t.rfind(THINK_END) + len(THINK_END):]
    end = t.rfind(f"</{tag}>")
    start = t.rfind(f"<{tag}>", 0, end) if end >= 0 else -1
    body = (t[start + len(tag) + 2:end] if start >= 0 else t).strip()
    return body if len(body) <= max_chars else "…" + body[-max_chars:]


def _ask(checker: ChatEndpoint, prompt: str, parse: Callable[[str], Any], *,
         attempts: int = 2) -> tuple[Any, str, int]:
    """(``parse(reply)`` or None, the last error, replies cut off at max_tokens), retrying a bad reply."""
    last, cut = "", 0
    for _ in range(attempts):
        try:
            reply = checker.request([{"role": "user", "content": prompt}])
            if reply.finish_reason == "length":
                cut += 1
                raise ValueError("reply truncated")
            return parse(reply.content), "", cut
        except Exception as e:  # unreadable or failed request: retry, then report it
            last = str(e)
    return None, last, cut


class VerdictCache:
    """Append-only JSONL of finished per-item verdicts, so a check cut off by a time limit or a crash resumes.

    Rows are keyed by ``key`` (the draft id, or the question for D-Report) and a stage: ``code`` for a D-Fresh
    draft's stated-answer and solver-code checks, ``final`` for the complete verdict. Later rows win.
    """

    def __init__(self, path: str | Path | None, key: str):
        self.path, self.key = (Path(path) if path else None), key
        self._lock = threading.Lock()
        self.rows: dict[tuple[str, str], dict[str, Any]] = {}
        if self.path is not None and self.path.exists():
            for r in read_jsonl(self.path):
                self.rows[(str(r[key]), r.get("stage", "final"))] = r

    def get(self, ident: str, stage: str = "final") -> dict[str, Any] | None:
        r = self.rows.get((str(ident), stage))
        return None if r is None else {k: v for k, v in r.items() if k != "stage"}

    def put(self, row: dict[str, Any], stage: str = "final") -> None:
        if self.path is None:
            return
        rec = {**row, "stage": stage}
        with self._lock:
            self.rows[(str(row[self.key]), stage)] = rec
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _rank(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def select_balanced(rows: Sequence[dict[str, Any]], n: int, *, key: str = "topic",
                    ident: str = "question") -> list[dict[str, Any]]:
    """Up to ``n`` rows, round-robin over ``key`` (topics in sorted order, rows in hash order)."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(str(r.get(key, "")), []).append(r)
    queues = [sorted(g, key=lambda r: _rank(str(r[ident]))) for _, g in sorted(groups.items())]
    out: list[dict[str, Any]] = []
    while len(out) < n and any(queues):
        for q in queues:
            if q and len(out) < n:
                out.append(q.pop(0))
    return out


def dedupe(rows: Sequence[dict[str, Any]], threshold: float = 0.6) -> tuple[list[dict[str, Any]], list[dict]]:
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for r in sorted(rows, key=lambda r: _rank(str(r["question"]))):
        (dropped if any(near_duplicate(r["question"], k["question"], threshold) for k in kept) else kept).append(r)
    return kept, dropped


def write_eval_set(records: list[dict[str, Any]], out: str | Path, name: str, *,
                   stage_stats: list[dict[str, Any]], parent: str | Path, notes: str) -> Path:
    from dpt.common.hashing import sha256_file
    from dpt.common.manifest import build_manifest, write_manifest

    write_jsonl(out, records)
    m = build_manifest(out, name, "v1", parents={Path(parent).name: sha256_file(parent)},
                       stage_stats=stage_stats, notes=notes)
    return write_manifest(m)


def _canary(name: str) -> str:
    from dpt.data.decontam import make_canary

    return make_canary(f"{name}/v1")


# --------------------------------------------------------------------------- D-Fresh


def gold_of(d: dict[str, Any]) -> dict[str, Any]:
    return {"value": float(d["answer_value"]), "unit": str(d["answer_unit"]),
            "rel_tol": float(d.get("rel_tolerance_percent") or 1.0) / 100.0}


def answer_text(d: dict[str, Any]) -> str:
    return f"{float(d['answer_value']):.10g} {d['answer_unit']}"


def check_stated_answer(d: dict[str, Any]) -> str | None:
    """None if the stated answer grades correct against itself (value and unit parse), else the reason."""
    from dpt.eval.graders.numeric import grade_numeric

    g = grade_numeric(f"\\boxed{{{answer_text(d)}}}", gold_of(d), missing_unit="strict")
    return None if g.correct else f"stated_answer_unparsed:{g.reason}"


def check_solver_code(d: dict[str, Any], sandbox: Any, *, rel: float = 1e-3, timeout_s: float = 20.0) -> str | None:
    """None if ``solve()`` reproduces the stated answer (run in the sandbox), else the reason."""
    from dpt.rewards.code_sandbox import SandboxRequest
    from dpt.rewards.hack_patterns import scan_code

    blocked = sorted({h.category for h in scan_code(d["solve_py"])} & BLOCKING_CODE_HITS)
    if blocked:
        return "code_blocked:" + ",".join(blocked)
    tests = CODE_TEST.format(stated=float(d["answer_value"]), rel=float(rel))
    res = sandbox.run(SandboxRequest(code=d["solve_py"], tests=tests, timeout_s=timeout_s, kind="visible"))
    if res.status == "infra_error":
        raise RuntimeError(f"sandbox unavailable: {res.detail}")      # fail the run, not the item
    if res.status != "ok" or res.passed != res.total or res.tamper:
        return f"code_mismatch:{res.status}"
    return None


@dataclass(frozen=True)
class Solve:
    """One independent Gemma solution of a D-Fresh problem."""

    outcome: str             # agree | other | unreadable
    answer: str = ""         # the final answer, as written in its \boxed{}
    solution: str = ""       # the worked solution it handed in
    truncated: bool = False  # cut off at max_tokens


def solve_once(d: dict[str, Any], checker: ChatEndpoint) -> Solve:
    from dpt.eval.graders.numeric import grade_numeric

    try:
        reply = checker.request([{"role": "user", "content": SOLVE_PROMPT.format(question=d["question"])}])
    except Exception as e:  # one failed request must not drop a problem; it just counts as unreadable
        log.warning("solve request failed for %s: %s", d.get("draft_id"), e)
        return Solve("unreadable")
    if reply.finish_reason == "length":
        return Solve("unreadable", truncated=True)
    # Only the final answer: reasoning shown before it may hold earlier boxed attempts. (This checks a
    # label, not the checker, so the eval's multiple-answer rule does not apply.)
    last = reply.content.rfind("\\boxed")
    g = grade_numeric(reply.content[last:] if last >= 0 else reply.content, gold_of(d), missing_unit="strict")
    if g.reason in ("no_answer", "no_final_answer"):
        return Solve("unreadable")
    return Solve("agree" if g.correct else "other", answer=str(g.details.get("answer") or ""),
                 solution=last_block(reply.content, "solution"))


def _parses(answer: str) -> bool:
    from dpt.eval.graders.numeric import ValueParseError, check_single_value, parse_quantity

    try:
        parse_quantity(answer)
    except ValueParseError:
        return False
    return check_single_value(answer) is None


def same_quantity(a: str, b: str, rel: float) -> bool:
    """Whether two final answers state the same quantity (units converted), to ``rel``."""
    from dpt.eval.graders.numeric import ValueParseError, grade_numeric, parse_quantity

    if a.strip() == b.strip():
        return True
    try:
        q = parse_quantity(a)
    except ValueParseError:
        return False
    return grade_numeric(f"\\boxed{{{b}}}", {"value": q.value, "unit": q.unit, "rel_tol": rel},
                         missing_unit="strict").correct


def rival_groups(solves: Sequence[Solve], rel: float) -> list[list[Solve]]:
    """The non-matching answers that parse, grouped by value, largest group first."""
    groups: list[list[Solve]] = []
    for s in solves:
        if s.outcome != "other" or not _parses(s.answer):
            continue
        for g in groups:
            if same_quantity(g[0].answer, s.answer, rel):
                g.append(s)
                break
        else:
            groups.append([s])
    return sorted(groups, key=len, reverse=True)


def needs_adjudication(agree: int, rival: int) -> bool:
    """No solution matches but one parses, or a rival answer has ≥ 2 votes and ≥ the stated answer's."""
    return (agree == 0 and rival >= 1) or (rival >= 2 and rival >= agree)


def _adjudication_outcome(v: dict[str, Any] | None, stated_label: str) -> tuple[str, str]:
    if v is None:
        return "unreadable", ""
    note = str(v.get("wrong_step") or "").strip()
    if v.get("ambiguous") is True:
        return "ambiguous", note
    pick = str(v.get("correct", "")).strip().upper()
    if pick == stated_label:
        return ("upheld" if note else "unsupported"), note     # must name the error in the other solution
    if pick in ("A", "B", "NEITHER"):
        return "refuted", note
    return "unreadable", note


def adjudicate_fresh(d: dict[str, Any], rival: Solve, checker: ChatEndpoint) -> tuple[str, list[str], str]:
    """(decision, the outcome in each order, notes). Decision: upheld | ambiguous | refuted | unreadable | split."""
    # Both solutions are capped (last_block), so the prompt plus the thinking budget fit the judge's context.
    stated = last_block(str(d.get("solution") or ""), "solution") or f"```python\n{str(d['solve_py']).strip()}\n```"
    sides = [(stated, answer_text(d)), (rival.solution or "(no worked solution was handed in)", rival.answer)]
    outcomes, notes = [], []
    for stated_first in (True, False):
        (a, answer_a), (b, answer_b) = sides if stated_first else sides[::-1]
        prompt = ADJUDICATE_PROMPT.format(question=d["question"], a=a, answer_a=answer_a, b=b, answer_b=answer_b)
        v, _, _ = _ask(checker, prompt, lambda c: parse_last_json(c, "correct"))
        outcome, note = _adjudication_outcome(v, "A" if stated_first else "B")
        outcomes.append(outcome)
        if note:
            notes.append(f"{outcome}: {note[:NOTE_CHARS]}")
    if all(o == "upheld" for o in outcomes):
        decision = "upheld"
    elif "ambiguous" in outcomes:
        decision = "ambiguous"
    elif "unreadable" in outcomes:
        decision = "unreadable"
    elif all(o == "refuted" for o in outcomes):
        decision = "refuted"
    else:
        decision = "split"
    return decision, outcomes, " || ".join(notes)


def check_fresh(
    drafts: Sequence[dict[str, Any]],
    checker: ChatEndpoint,
    sandbox: Any,
    *,
    solves: int = 4,
    min_agree: int = 1,
    adjudicate: bool = True,
    code_rel_tol: float = 1e-3,
    max_concurrency: int = 16,
    cache: VerdictCache | None = None,
    code_only: bool = False,
) -> list[dict[str, Any]]:
    """One log row per draft (``status`` kept | drop, ``reason``, vote counts, adjudication).

    With a ``cache``, finished drafts are skipped and code verdicts are reused (``sandbox`` may then be None).
    ``code_only`` runs just the stated-answer and solver-code checks (status ``code_ok`` | ``code_drop``).
    """
    cache = cache or VerdictCache(None, "draft_id")

    def one(d: dict[str, Any]) -> dict[str, Any]:
        did = d.get("draft_id")
        done = cache.get(did)
        if done is not None and not code_only:
            return done
        row = {"draft_id": did, "topic": d.get("topic", ""), "question": d["question"],
               "answer": answer_text(d), **{k: "" for k in ("agree", "rival", "readable", "truncated", "solves",
                                                            "adjudication", "adjudication_note", "rival_answer")}}
        coded = cache.get(did, "code")
        if coded is not None:
            reason = coded["reason"]
        else:
            if sandbox is None:
                raise RuntimeError(f"draft {did}: no cached code verdict and no sandbox (run --code-only on a "
                                   "compute node first)")
            reason = check_stated_answer(d) or check_solver_code(d, sandbox, rel=code_rel_tol) or ""
            cache.put({"draft_id": did, "reason": reason}, "code")
        if code_only:
            return {**row, "status": "code_drop" if reason else "code_ok", "reason": reason or "ok"}
        if reason:
            out = {**row, "status": "drop", "reason": reason}
            cache.put(out)
            return out
        results = [solve_once(d, checker) for _ in range(solves)]
        agree = sum(s.outcome == "agree" for s in results)
        readable = sum(s.outcome != "unreadable" for s in results)
        groups = rival_groups(results, gold_of(d)["rel_tol"])
        rival = groups[0] if groups else []
        row.update(agree=agree, rival=len(rival), readable=readable, solves=solves,
                   truncated=sum(s.truncated for s in results), rival_answer=rival[0].answer if rival else "")
        if adjudicate and needs_adjudication(agree, len(rival)):
            decision, outcomes, notes = adjudicate_fresh(d, rival[0], checker)
            row.update(adjudication="/".join(outcomes), adjudication_note=notes)
            out = ({**row, "status": "kept", "reason": "adjudicated"} if decision == "upheld"
                   else {**row, "status": "drop", "reason": f"adjudicated_{decision}"})
        elif agree < min_agree:
            why = "no_readable_solution" if readable == 0 else f"unconfirmed:{agree}/{solves}"
            out = {**row, "status": "drop", "reason": why}
        else:
            out = {**row, "status": "kept", "reason": "ok"}
        cache.put(out)
        return out

    with ThreadPoolExecutor(max(1, max_concurrency)) as ex:
        return list(ex.map(one, drafts))


def fresh_records(drafts: Sequence[dict[str, Any]], verdicts: Sequence[dict[str, Any]], n_keep: int
                  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """(eval records, updated log rows, stage stats) from the drafts and their check rows."""
    by_id = {v["draft_id"]: v for v in verdicts}
    passed = [d for d in drafts if by_id[d["draft_id"]]["status"] == "kept"]
    kept, dups = dedupe(passed)
    chosen = select_balanced(kept, n_keep)
    chosen_ids = {d["draft_id"] for d in chosen}
    for d in dups:
        by_id[d["draft_id"]].update(status="drop", reason="near_duplicate")
    for d in kept:
        if d["draft_id"] not in chosen_ids:
            by_id[d["draft_id"]].update(status="spare", reason="over n_keep")
    canary = _canary("d_fresh")
    records = [{
        "id": f"d_fresh-{d['draft_id']}",
        "question": d["question"],
        "answer": gold_of(d),
        "task": "t1_num",
        "meta": {"topic": d.get("topic", ""), "drafted_by": d.get("drafted_by", ""),
                 "date_written": d.get("date_written", ""), "canary": canary,
                 "check": {k: by_id[d["draft_id"]].get(k, "") for k in ("agree", "solves", "adjudication")}},
    } for d in chosen]
    reasons = Counter(v["reason"].split(":")[0] for v in verdicts if v["status"] == "drop")
    stats = [{"stage": "drafted", "count": len(drafts)},
             *({"stage": f"dropped:{k}", "count": n} for k, n in sorted(reasons.items())),
             {"stage": "spare", "count": len(kept) - len(chosen)},
             {"stage": "kept_after_adjudication",
              "count": sum(by_id[d["draft_id"]]["reason"] == "adjudicated" for d in chosen)},
             {"stage": "kept", "count": len(records)}]
    return records, list(by_id.values()), stats


# --------------------------------------------------------------------------- D-Report


def _criteria_of(row: dict[str, str]) -> list[dict[str, Any]]:
    crit = [c.strip() for c in str(row.get("rubric_criteria", "")).split("|")]
    weights = [w.strip() for w in str(row.get("rubric_weights", "")).split("|")]
    out = []
    for i, c in enumerate(crit):
        if not c:
            continue
        try:
            w = min(3, max(1, int(weights[i])))
        except (IndexError, ValueError):
            w = 1
        out.append({"criterion": c, "weight": w})
    return out


def _errors_of(row: dict[str, str]) -> list[str]:
    return [e.strip() for e in str(row.get("common_errors_to_penalize", "")).split(";") if e.strip()]


def _parse_verdict(text: str, n_crit: int) -> dict[str, Any]:
    """One readable rubric verdict: question ok, per-criterion (accepted, issue), per-error kept."""
    v = parse_last_json(text, "criteria")
    marks: dict[int, dict[str, Any]] = {}
    for c in v.get("criteria") or []:
        try:
            marks[int(c["id"])] = c
        except (KeyError, TypeError, ValueError):
            continue
    if any(i not in marks for i in range(1, n_crit + 1)):
        raise ValueError("criteria missing from the reply")
    errors: dict[int, bool] = {}
    for e in v.get("common_errors") or []:
        try:
            errors[int(e["id"])] = str(e.get("really_an_error", "")).lower() != "no"
        except (KeyError, TypeError, ValueError):
            continue
    q = v.get("question") or {}
    crit = {}
    for i in range(1, n_crit + 1):
        m = marks[i]
        bad = str(m.get("correct", "")).lower() == "no" or str(m.get("checkable", "")).lower() == "no"
        crit[i] = (not bad, str(m.get("issue") or "").strip())
    return {"question_ok": q.get("ok") is not False, "question_issue": str(q.get("issue") or "").strip(),
            "criteria": crit, "errors": errors}


def _own_answer(text: str) -> str:
    own = last_block(text, "answer")
    if not own:
        raise ValueError("empty answer")
    return own


def report_vote(row: dict[str, str], words: int, crit: list[dict[str, Any]], errors: list[str],
                checker: ChatEndpoint) -> tuple[dict[str, Any] | None, str, int]:
    """One pass: Gemma answers the question, then rules on the rubric with its answer in front of it."""
    own, err, cut = _ask(checker, REPORT_ANSWER_PROMPT.format(words=words, question=row["question"]), _own_answer)
    if own is None:
        return None, f"own answer: {err}", cut
    prompt = REPORT_CHECK_PROMPT.format(
        question=row["question"], words=words, own=own,
        criteria="\n".join(f"{i}. (weight {c['weight']}) {c['criterion']}" for i, c in enumerate(crit, 1)),
        errors="\n".join(f"{i}. {e}" for i, e in enumerate(errors, 1)) or "(none)",
    )
    v, err, cut2 = _ask(checker, prompt, lambda c: _parse_verdict(c, len(crit)))
    return v, err, cut + cut2


def check_report_row(row: dict[str, str], checker: ChatEndpoint, *, min_criteria: int = 3,
                     words: int | None = None, votes: int = 3) -> dict[str, Any]:
    """The row with ``status`` (ok | fixed | drop), ``check_issue``, ``check_votes`` and the surviving rubric."""
    words = int(row.get("reference_length_words") or 0) or words or sum(REPORT_WORDS) // 2
    crit, errors = _criteria_of(row), _errors_of(row)
    out: dict[str, Any] = {**row, "reference_length_words": words}

    readable, last, cut = [], "", 0
    for _ in range(votes):
        v, err, c = report_vote(row, words, crit, errors, checker)
        cut += c
        if v is None:
            last = err
        else:
            readable.append(v)
    n = len(readable)
    out["truncated"] = cut

    def majority(k: int) -> bool:
        return 2 * k > n

    def drop(issue: str) -> dict[str, Any]:
        return {**out, "status": "drop", "check_issue": issue}

    if n < min(2, votes):
        out["check_votes"] = f"{n}/{votes} readable" + (f"; cut {cut}" if cut else "")
        return drop(f"unreadable_check: {last}")
    q_rejects = [v for v in readable if not v["question_ok"]]
    crit_rejects = {i: [v["criteria"][i][1] for v in readable if not v["criteria"][i][0]]
                    for i in range(1, len(crit) + 1)}
    err_rejects = {i: sum(v["errors"].get(i) is False for v in readable) for i in range(1, len(errors) + 1)}
    out["check_votes"] = "; ".join([
        f"{n}/{votes} readable", f"question {n - len(q_rejects)}/{n}",
        "criteria " + ",".join(str(n - len(crit_rejects[i])) for i in crit_rejects),
        "errors " + (",".join(str(n - k) for k in err_rejects.values()) or "-"),
    ] + ([f"cut {cut}"] if cut else []))
    if majority(len(q_rejects)):
        issue = next((v["question_issue"] for v in q_rejects if v["question_issue"]), "rejected")
        return drop(f"question: {issue}")
    keep, rejected = [], []
    for i, c in enumerate(crit, 1):
        issues = crit_rejects[i]
        if majority(len(issues)):
            rejected.append((c, next((w for w in issues if w), "")))
        else:
            keep.append(c)
    if any(c["weight"] == 3 for c, _ in rejected):
        return drop("essential criterion rejected: " + "; ".join(f"{c['criterion']} ({w})" for c, w in rejected
                                                                 if c["weight"] == 3))
    if len(keep) < min_criteria:
        return drop(f"only {len(keep)} criteria passed")
    errors_kept = [e for i, e in enumerate(errors, 1) if not majority(err_rejects[i])]
    fixed = bool(rejected) or len(errors_kept) != len(errors)
    issues = [f"removed criterion: {c['criterion']}" + (f" ({w})" if w else "") for c, w in rejected]
    issues += [f"removed error: {e}" for e in errors if e not in errors_kept]
    return {
        **out,
        "rubric_criteria": " | ".join(c["criterion"] for c in keep),
        "rubric_weights": "|".join(str(c["weight"]) for c in keep),
        "common_errors_to_penalize": "; ".join(errors_kept),
        "status": "fixed" if fixed else "ok",
        "check_issue": "; ".join(issues),
    }


def check_reports(rows: Sequence[dict[str, str]], checker: ChatEndpoint, *, min_criteria: int = 3,
                  votes: int = 3, max_concurrency: int = 16, cache: VerdictCache | None = None) -> list[dict[str, Any]]:
    cache = cache or VerdictCache(None, "question")

    def one(r: dict[str, str]) -> dict[str, Any]:
        done = cache.get(r["question"])
        if done is not None:
            return done
        out = check_report_row(r, checker, min_criteria=min_criteria, votes=votes)
        cache.put(out)
        return out

    with ThreadPoolExecutor(max(1, max_concurrency)) as ex:
        return list(ex.map(one, rows))


def report_records(checked: Sequence[dict[str, Any]], n_keep: int
                   ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """(eval records, updated log rows, stage stats) from the checked rows."""
    rows = [dict(r) for r in checked]
    passed = [r for r in rows if r["status"] in ("ok", "fixed")]
    kept, dups = dedupe(passed)
    chosen = select_balanced(kept, n_keep)
    chosen_q = {r["question"] for r in chosen}
    for r in dups:
        r.update(status="drop", check_issue="near_duplicate")
    for r in kept:
        if r["question"] not in chosen_q:
            r["status"] = "spare"
    canary = _canary("d_report")
    records = []
    for r in chosen:
        words = int(r["reference_length_words"])
        records.append({
            "id": f"d_report-{_rank(r['question'])[:12]}",
            "question": r["question"],
            "answer": {"rubric": _criteria_of(r), "reference_length_words": words, "common_errors": _errors_of(r),
                       "prompt_extra": {"length_hint": words}},
            "task": "t3_report",
            "meta": {"topic": r.get("topic", ""), "drafted_by": r.get("drafted_by", ""),
                     "date_written": r.get("date_written", ""), "check": r["status"], "canary": canary},
        })
    reasons = Counter(str(r.get("check_issue", "")).split(":")[0] for r in rows if r["status"] == "drop")
    stats = [{"stage": "drafted", "count": len(rows)},
             {"stage": "fixed", "count": sum(r["status"] == "fixed" for r in rows)},
             *({"stage": f"dropped:{k}", "count": n} for k, n in sorted(reasons.items())),
             {"stage": "spare", "count": len(kept) - len(chosen)},
             {"stage": "kept", "count": len(records)}]
    return records, rows, stats


# --------------------------------------------------------------------------- CLI


def build_checker(suite: dict[str, Any], *, temperature: float | None) -> ChatEndpoint:
    """The Gemma endpoint (``eval_suite.judge``) with the request settings of ``eval_suite.autocheck``."""
    jcfg = {k: v for k, v in (suite.get("judge") or {}).items()
            if k in ("endpoint", "model", "pinned_model_version", "api_key_env")}
    jcfg.update({k: v for k, v in ((suite.get("autocheck") or {}).get("checker") or {}).items() if v is not None})
    jcfg["temperature"] = temperature
    chat = endpoint_from_config(jcfg, ChatEndpoint)
    if chat is None:
        raise SystemExit("no checker endpoint: serve the judge (serve.slurm judge ...) and set "
                         "eval_suite.judge.endpoint / .model (scripts/slurm/eval_sets.slurm does)")
    return chat


def build_sandbox(*, allow_no_container: bool) -> Any:
    import os
    import sys

    from dpt.rewards.code_sandbox import LocalSubprocessSandbox

    image = os.environ.get("DPT_SANDBOX_SIF") or None
    if image is None and not allow_no_container:
        raise SystemExit("drafted solver code runs only inside the sandbox container: set DPT_SANDBOX_SIF "
                         "(scripts/slurm/env.sh), or pass --allow-no-container on a machine of your own")
    return LocalSubprocessSandbox(python=os.environ.get("DPT_SANDBOX_PYTHON") or sys.executable,
                                  container_image=image, cpu_s=20, mem_mb=2048)


def _summary(rows: Sequence[dict[str, Any]], key: str) -> str:
    return ", ".join(f"{k} {n}" for k, n in sorted(Counter(r[key] for r in rows).items()))


def main(argv: Sequence[str] | None = None, *, checker_factory: Callable[..., Any] | None = None,
         sandbox_factory: Callable[..., Any] | None = None) -> int:
    from dpt.common.config import load_config, to_container

    ap = argparse.ArgumentParser(description="Check drafted D-Fresh / D-Report items with the Gemma judge")
    ap.add_argument("--model-line", default=None, help="default: $DPT_MODEL_LINE")
    ap.add_argument("--max-concurrency", type=int, default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fresh")
    f.add_argument("--drafts", required=True, help="JSONL from `dpt.eval.drafts fresh`")
    f.add_argument("--out", required=True, help="eval set JSONL (a manifest is written next to it)")
    f.add_argument("--log", default=None, help="CSV: one row per draft with its status and reason")
    f.add_argument("--allow-no-container", action="store_true",
                   help="run the drafted code without the Apptainer container (never on a shared account)")
    f.add_argument("--code-only", action="store_true",
                   help="only the stated-answer and solver-code checks (compute node), cached for a later run")
    r = sub.add_parser("report")
    r.add_argument("--drafts", required=True, help="CSV from `dpt.eval.drafts report`")
    r.add_argument("--out", required=True)
    r.add_argument("--log", default=None)
    for p in (f, r):
        p.add_argument("--cache", default=None, help="per-item verdict cache (default: <out>.cache.jsonl)")
        p.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. eval_suite.judge.endpoint=http://h:8000")
    args = ap.parse_args(argv)

    suite = to_container(load_config("eval/eval_suite.yaml", overrides=args.overrides,
                                     model_line=args.model_line))["eval_suite"]
    ac = suite.get("autocheck") or {}
    conc = args.max_concurrency or int(ac.get("max_concurrency", 16))
    make_checker = checker_factory or (lambda temperature: build_checker(suite, temperature=temperature))

    cache_path = args.cache or f"{args.out}.cache.jsonl"
    if args.cmd == "fresh":
        fc = ac.get("fresh") or {}
        drafts = list(read_jsonl(args.drafts))
        cache = VerdictCache(cache_path, "draft_id")
        # The drafted code runs only in the container: build the sandbox only if some draft still needs it.
        need_code = [d for d in drafts if cache.get(d["draft_id"]) is None and cache.get(d["draft_id"], "code") is None]
        sandbox = (sandbox_factory or build_sandbox)(allow_no_container=args.allow_no_container) if need_code else None
        if args.code_only:
            rows = check_fresh(drafts, None, sandbox, code_rel_tol=float(fc.get("code_rel_tol", 1e-3)),
                               max_concurrency=conc, cache=cache, code_only=True)
            log.info("d_fresh code checks: %s → %s", _summary(rows, "status"), cache_path)
            return 0
        verdicts = check_fresh(drafts, make_checker(float(fc.get("solve_temperature", 0.6))), sandbox,
                               solves=int(fc.get("solves", 4)), min_agree=int(fc.get("min_agree", 1)),
                               adjudicate=bool(fc.get("adjudicate", True)),
                               code_rel_tol=float(fc.get("code_rel_tol", 1e-3)), max_concurrency=conc, cache=cache)
        records, log_rows, stats = fresh_records(drafts, verdicts, int(fc.get("n_keep", 60)))
        name, cols, min_n = "d_fresh", FRESH_LOG_COLUMNS, int(fc.get("min_items", 40))
        cut, asked = sum(int(v["truncated"] or 0) for v in verdicts), sum(int(v["solves"] or 0) for v in verdicts)
    else:
        rc = ac.get("report") or {}
        votes = int(rc.get("votes", 3))
        checked = check_reports(read_csv(args.drafts), make_checker(float(rc.get("vote_temperature", 0.6))),
                                min_criteria=int(rc.get("min_criteria", 3)), votes=votes, max_concurrency=conc,
                                cache=VerdictCache(cache_path, "question"))
        records, log_rows, stats = report_records(checked, int(rc.get("n_keep", 100)))
        name, cols, min_n = "d_report", REPORT_LOG_COLUMNS, int(rc.get("min_items", 80))
        cut, asked = sum(int(r.get("truncated") or 0) for r in checked), 2 * votes * len(checked)
    if asked and cut / asked > 0.05:
        log.warning("%s: %d of %d replies were cut off at max_tokens and not used; raise "
                    "eval_suite.autocheck.checker.max_tokens (and the judge's max_model_len)", name, cut, asked)
    if args.log:
        write_csv(args.log, log_rows, cols)
    log.info("%s: %s", name, _summary(log_rows, "status"))
    if len(records) < min_n:
        log.error("%s: only %d items passed (need ≥ %d); draft more topics and rerun. Nothing written.",
                  name, len(records), min_n)
        return 1
    m = write_eval_set(records, args.out, name, stage_stats=stats, parent=args.drafts,
                       notes="drafted by the hosted model, checked by dpt.eval.autocheck; no human review")
    log.info("%d %s items → %s (manifest %s)", len(records), name, args.out, m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
