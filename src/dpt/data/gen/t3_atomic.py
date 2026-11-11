"""Atomic yes/no checks for the T3 report rubrics (2026-11-07).

The v1 rubrics have 4–6 criteria per question. A median criterion is 47 words, and 83% of them join several claims
("and", "or", ";"). The judge scores each one 0–10 as a whole, which leaves room for generous partial credit: on the
same RL dev answers Gemma averaged ~81 and GPT-6 Luna ~48 (the run log, judge-drift audit). This module
rewrites every criterion into 1–6 atomic checks, each one verifiable claim (a fact, a value with its unit, a relation
or a reasoning step) that a judge can answer yes or no from the answer alone. The judge then gives a "yes" only with a
quoted span (dpt.rewards.judge, SYSTEM_PROMPT_CHECKS), and the reward is the weighted share of "yes".

Who writes what (plan §5, model roles):
- the teacher (Qwen3.8-27B, the model that wrote the rubrics) writes the checks;
- Gemma (the verification family, endpoint ``judge_v``) checks that the checks of each criterion cover it and add
  nothing. A failed check sends the rubric back to the teacher once, with Gemma's notes.
GPT-6 Luna is never used here: its output must not enter training data.

Weights: a criterion's weight is split evenly over its checks, so each criterion keeps its share of the score. A
rubric that still fails the structural rules after the retries is dropped by ``apply`` (``--keep-fallback`` keeps it
with its criteria as single checks instead).

    python -m dpt.data.gen.t3_atomic convert --in stages/04_verify.jsonl --out t3_atomic_map.jsonl \\
        --teacher-url URL --teacher-model NAME --tokenizer Qwen/Qwen3.8-27B [--judge-url URL --judge-model NAME]
    python -m dpt.data.gen.t3_atomic apply --map t3_atomic_map.jsonl --in stage.jsonl --out stage_atomic.jsonl

Outputs carry no model text other than the checks themselves. The map has, per T3 problem: ``uid``, ``rubric`` (the
checks), ``rubric_criteria`` (the original rubric), ``status`` and ``coverage``.
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from dpt.common.io import read_jsonl, write_jsonl
from dpt.common.logging import get_logger
from dpt.common.seeding import derive_seed

log = get_logger(__name__)

T3 = "t3_report"
Gen = Callable[..., list[str]]

CONVERT_PROMPT = """You are rewriting the grading rubric of a seismology report question into atomic yes/no checks.

## Question
{question}

## Rubric criteria
{criteria}

## Rules
1. Rewrite every criterion as 1 to {max_checks} checks. Each check states exactly ONE verifiable claim: one fact, one value with its unit, one relation, or one reasoning step.
2. A check must be decidable as yes or no from the answer alone. Write it as a statement the answer must make, starting with "The answer" (for example "The answer states that ...", "The answer gives ... as ...", "The answer derives ... from ...").
3. Never join two claims with "and", "or", ";" or a list. If a criterion names several items, write one check per item.
4. Keep every number, unit, symbol and formula a check needs. Do not drop any requirement of the criterion, and do not add requirements it does not make.
5. No checks about style, clarity, structure, length or citations.
6. Each check has at most {max_words} words.

Respond with JSON only, one entry per criterion, in order:
{{"criteria": [{{"id": 1, "checks": ["The answer states ...", "The answer gives ..."]}}, ...]}}"""

RETRY_NOTE = """

## Your previous attempt was rejected
{problems}
Write the whole JSON again, following every rule."""

COVERAGE_PROMPT = """You check that atomic yes/no checks faithfully replace the criteria of a grading rubric for a seismology report question.

## Question
{question}

## Criteria and their checks
{pairs}

For each criterion decide:
- covers: would an answer that satisfies all of its checks necessarily satisfy everything the criterion requires?
- no_additions: do the checks avoid requirements that the criterion does not make?
Give a short note naming anything missing or added.

Respond with JSON only:
{{"criteria": [{{"id": 1, "covers": true, "no_additions": true, "note": ""}}, ...]}}"""


@dataclass
class AtomicConfig:
    max_checks: int = 6            # per criterion
    max_total: int = 30            # per rubric
    min_words: int = 4
    max_words: int = 40
    retries: int = 1               # extra teacher attempts after a structural or coverage failure
    teacher_max_tokens: int = 12288
    judge_max_tokens: int = 4096
    seed: int = 1


# ---------------------------------------------------------------------------------------------- parsing helpers
def json_object(text: str) -> dict[str, Any]:
    """The JSON object in a model reply: the whole reply, else the span from the first ``{`` to the last ``}``,
    else the last parseable ``{...}`` (a reply may start with plain-text thinking)."""
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    for cand in (t, t[t.find("{"): t.rfind("}") + 1] if "{" in t else ""):
        if cand:
            try:
                obj = json.loads(cand)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
    for i in [m.start() for m in re.finditer(r"\{", t)][::-1]:
        try:
            obj, _ = json.JSONDecoder().raw_decode(t[i:])
            if isinstance(obj, dict) and obj:
                return obj
        except json.JSONDecodeError:
            continue
    raise ValueError("no JSON object in the reply")


def answer_part(reply: str) -> str | None:
    """The teacher's answer after its thinking; None when the thinking never closed (cut off)."""
    from dpt.chat.thinking import split_thinking

    sp = split_thinking(reply)
    return sp.answer if sp.has_end else None


def criteria_of(rubric: Sequence[dict[str, Any]]) -> list[tuple[str, float]]:
    return [(str(r["criterion"]), float(r["weight"])) for r in rubric]


def parse_checks(obj: dict[str, Any], n_criteria: int) -> list[list[str]]:
    crit = obj.get("criteria")
    if not isinstance(crit, list):
        raise ValueError("missing 'criteria' list")
    by_id: dict[int, list[str]] = {}
    for c in crit:
        try:
            cid = int(c["id"])
            checks = [str(x).strip() for x in c["checks"]]
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"bad entry {c!r}") from e
        by_id[cid] = list(dict.fromkeys(x for x in checks if x))         # drop empties and repeats
    missing = [i for i in range(1, n_criteria + 1) if i not in by_id]
    if missing:
        raise ValueError(f"criteria {missing} missing")
    return [by_id[i] for i in range(1, n_criteria + 1)]


def structural_problems(checks: list[list[str]], cfg: AtomicConfig) -> list[str]:
    """Rule violations that send the rubric back to the teacher."""
    out = []
    total = sum(len(c) for c in checks)
    if total > cfg.max_total:
        out.append(f"{total} checks in all; at most {cfg.max_total}")
    for i, cs in enumerate(checks, start=1):
        if not 1 <= len(cs) <= cfg.max_checks:
            out.append(f"criterion {i} has {len(cs)} checks; it needs 1 to {cfg.max_checks}")
        for c in cs:
            n = len(c.split())
            if not cfg.min_words <= n <= cfg.max_words:
                out.append(f"criterion {i}: a check has {n} words (allowed {cfg.min_words}–{cfg.max_words})")
            if ";" in c:
                out.append(f"criterion {i}: a check joins claims with ';'")
    return out


def soft_flags(checks: list[list[str]]) -> dict[str, int]:
    """Possible compound checks, counted for the report (not rejected: "and" also appears inside one claim)."""
    flat = [c for cs in checks for c in cs]
    return {"and": sum(bool(re.search(r"\band\b", c)) for c in flat),
            "or": sum(bool(re.search(r"\bor\b", c)) for c in flat)}


def atomic_rubric(rubric: Sequence[dict[str, Any]], checks: list[list[str]]) -> list[dict[str, Any]]:
    out = []
    for i, ((_, w), cs) in enumerate(zip(criteria_of(rubric), checks), start=1):
        for c in cs:
            out.append({"check": c, "weight": round(w / len(cs), 6), "criterion_id": i})
    return out


# ---------------------------------------------------------------------------------------------- one rubric
def convert_one(uid: str, question: str, rubric: Sequence[dict[str, Any]], teacher: Gen, judge: Gen | None,
                cfg: AtomicConfig) -> dict[str, Any]:
    crits = criteria_of(rubric)
    criteria_txt = "\n".join(f"{i}. (weight {w:g}) {c}" for i, (c, w) in enumerate(crits, start=1))
    base = CONVERT_PROMPT.format(question=question.strip(), criteria=criteria_txt, max_checks=cfg.max_checks,
                                 max_words=cfg.max_words)
    note, last_problems, coverage = "", ["not attempted"], None
    for attempt in range(cfg.retries + 1):
        prompt = base + (RETRY_NOTE.format(problems=note) if note else "")
        seed = derive_seed(cfg.seed, "t3_atomic", uid, attempt)
        reply = teacher([{"role": "user", "content": prompt}], n=1, effort="medium",
                        max_tokens=cfg.teacher_max_tokens, seed=seed)[0]
        ans = answer_part(reply)
        if ans is None:
            note, last_problems = "- The reply was cut off before the JSON.", ["cut_off"]
            continue
        try:
            checks = parse_checks(json_object(ans), len(crits))
        except ValueError as e:
            note, last_problems = f"- {e}", [str(e)]
            continue
        problems = structural_problems(checks, cfg)
        if problems:
            note, last_problems = "\n".join(f"- {p}" for p in problems[:12]), problems
            continue
        if judge is not None:
            coverage = check_coverage(uid, question, crits, checks, judge, cfg, attempt)
            bad = [c for c in coverage.get("criteria", []) if not (c.get("covers") and c.get("no_additions"))]
            if bad and attempt < cfg.retries:
                note = "\n".join(f"- criterion {c.get('id')}: {str(c.get('note') or 'not faithful')[:300]}"
                                 for c in bad)
                last_problems = [f"coverage: {len(bad)} criteria"]
                continue
        return {"uid": uid, "status": "ok", "attempts": attempt + 1,
                "rubric": atomic_rubric(rubric, checks), "rubric_criteria": list(rubric),
                "coverage": _coverage_summary(coverage), "flags": soft_flags(checks)}
    return {"uid": uid, "status": "fallback", "attempts": cfg.retries + 1, "problems": last_problems[:12],
            "rubric": [{"check": c, "weight": w, "criterion_id": i} for i, (c, w) in enumerate(crits, start=1)],
            "rubric_criteria": list(rubric), "coverage": _coverage_summary(coverage), "flags": {}}


def check_coverage(uid: str, question: str, crits: list[tuple[str, float]], checks: list[list[str]], judge: Gen,
                   cfg: AtomicConfig, attempt: int) -> dict[str, Any]:
    from dpt.data.answer_verify import unmarked_reply

    pairs = "\n".join(f"{i}. Criterion: {c}\n" + "\n".join(f"   - {x}" for x in cs)
                      for i, ((c, _), cs) in enumerate(zip(crits, checks), start=1))
    prompt = COVERAGE_PROMPT.format(question=question.strip(), pairs=pairs)
    seed = derive_seed(cfg.seed, "t3_atomic_cov", uid, attempt)
    reply = judge([{"role": "user", "content": prompt}], n=1, effort=None, max_tokens=cfg.judge_max_tokens,
                  seed=seed)[0]
    try:
        obj = json_object(answer_part(unmarked_reply(reply)) or "")
    except ValueError:
        return {"error": "unparsed"}
    return obj if isinstance(obj.get("criteria"), list) else {"error": "no criteria list"}


def _coverage_summary(cov: dict[str, Any] | None) -> dict[str, Any] | None:
    if cov is None:
        return None
    if "error" in cov:
        return {"error": cov["error"]}
    crit = cov.get("criteria") or []
    return {"criteria": len(crit), "covers": sum(bool(c.get("covers")) for c in crit),
            "no_additions": sum(bool(c.get("no_additions")) for c in crit)}


# ---------------------------------------------------------------------------------------------- CLI
def _generator(url: str, model: str, tokenizer: str | None, cache: Path, label: str, wait_s: float,
               url_file: str | None, concurrency: int) -> Gen:
    from dpt.data import backends as B

    client = B.OpenAICompatClient(url, model, timeout=7200.0, max_concurrency=concurrency,
                                  url_file=url_file, wait_s=wait_s)
    tok = B.load_hf_tokenizer(tokenizer) if tokenizer else None
    return B.CachedGenerator(B.VLLMGenerator(client, tokenizer=tok, temperature=0.6, top_p=0.95), cache,
                             f"{label}:{model}")


def cmd_convert(a: argparse.Namespace) -> None:
    cfg = AtomicConfig(retries=a.retries, seed=a.seed)
    rows = [r for r in read_jsonl(a.inp) if r.get("task") == T3]
    if a.limit:
        rows = rows[: a.limit]
    done: set[str] = set()
    out = Path(a.out)
    if out.exists():                                   # resume: keep finished rubrics
        done = {r["uid"] for r in read_jsonl(out)}
    todo = [r for r in rows if r["uid"] not in done]
    log.info("t3_atomic: %d T3 rows, %d already converted, %d to go", len(rows), len(done), len(todo))
    cache = Path(a.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    teacher = _generator(a.teacher_url, a.teacher_model, a.tokenizer, cache / "t3_atomic_teacher.jsonl", "t3_atomic",
                         a.wait_s, a.teacher_url_file, a.concurrency)
    judge = None
    if a.judge_url:
        judge = _generator(a.judge_url, a.judge_model, None, cache / "t3_atomic_judge.jsonl", "t3_atomic_cov",
                           a.wait_s, a.judge_url_file, a.concurrency)

    def one(r: dict[str, Any]) -> dict[str, Any]:
        try:
            return convert_one(r["uid"], r["question"], r["answer"]["rubric"], teacher, judge, cfg)
        except Exception as e:                         # an endpoint error: retried by the next run (not written)
            log.warning("t3_atomic %s: %s", r["uid"], e)
            return {"uid": r["uid"], "status": "error", "error": str(e)[:300]}

    ok = 0
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex, out.open("a", encoding="utf-8") as f:
        for i, res in enumerate(ex.map(one, todo), start=1):
            if res["status"] == "error":
                continue
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
            ok += res["status"] == "ok"
            if i % 50 == 0:
                log.info("t3_atomic: %d/%d done (%d ok)", i, len(todo), ok)
    summary = summarize(list(read_jsonl(out)))
    Path(str(out) + ".summary.json").write_text(json.dumps(summary, indent=1))
    log.info("t3_atomic: %s", summary)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if r["status"] == "ok"]
    n_checks = [len(r["rubric"]) for r in ok]
    words = [len(c["check"].split()) for r in ok for c in r["rubric"]]
    cov = [r["coverage"] for r in ok if r.get("coverage") and "criteria" in r["coverage"]]
    s: dict[str, Any] = {"rubrics": len(rows), "ok": len(ok), "fallback": sum(r["status"] == "fallback" for r in rows)}
    if ok:
        n_checks.sort()
        words.sort()
        s.update({"checks_per_rubric_median": n_checks[len(n_checks) // 2], "checks_total": sum(n_checks),
                  "words_per_check_median": words[len(words) // 2],
                  "words_per_check_p90": words[int(0.9 * (len(words) - 1))],
                  "checks_with_and": sum(r["flags"].get("and", 0) for r in ok),
                  "checks_with_or": sum(r["flags"].get("or", 0) for r in ok)})
    if cov:
        s["coverage_criteria"] = sum(c["criteria"] for c in cov)
        s["coverage_covers"] = sum(c["covers"] for c in cov)
        s["coverage_no_additions"] = sum(c["no_additions"] for c in cov)
    return s


def apply_map(rows: Sequence[dict[str, Any]], mapping: dict[str, dict[str, Any]], keep_fallback: bool = False
              ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Replace the rubric of every T3 row by its checks. A T3 row without a converted rubric (not in the map, or a
    fallback unless ``keep_fallback``) is dropped and counted; other rows pass unchanged."""
    out, dropped = [], {"missing": 0, "fallback": 0}
    for r in rows:
        if r.get("task") != T3:
            out.append(r)
            continue
        c = mapping.get(r["uid"])
        if c is None or (c["status"] != "ok" and not keep_fallback):
            dropped["missing" if c is None else "fallback"] += 1
            continue
        r = dict(r)
        r["answer"] = {**r["answer"], "rubric": c["rubric"]}
        r["meta"] = {**(r.get("meta") or {}), "rubric_criteria": c["rubric_criteria"], "rubric_format": "checks"}
        out.append(r)
    return out, dropped


def cmd_apply(a: argparse.Namespace) -> None:
    """``apply_map`` on a stage or bank file."""
    m = {r["uid"]: r for r in read_jsonl(a.map)}
    out, dropped = apply_map(list(read_jsonl(a.inp)), m, a.keep_fallback)
    write_jsonl(a.out, out)
    log.info("t3_atomic apply: %d rows written, T3 dropped %s", len(out), dropped)
    print(json.dumps({"written": len(out), "t3_dropped": dropped}))


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--in", dest="inp", required=True, help="problem rows (a stage file); only T3 rows are read")
    c.add_argument("--out", required=True, help="the map, one JSON line per T3 problem (appended: resumable)")
    c.add_argument("--teacher-url", required=True)
    c.add_argument("--teacher-model", required=True)
    c.add_argument("--teacher-url-file", default=None)
    c.add_argument("--tokenizer", required=True, help="the teacher's tokenizer (renders its prompt and effort)")
    c.add_argument("--judge-url", default=None, help="Gemma (judge_v): coverage check; omitted = none")
    c.add_argument("--judge-model", default=None)
    c.add_argument("--judge-url-file", default=None)
    c.add_argument("--cache-dir", required=True)
    c.add_argument("--concurrency", type=int, default=32)
    c.add_argument("--retries", type=int, default=1)
    c.add_argument("--wait-s", type=float, default=2700.0)
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("apply")
    p.add_argument("--map", required=True)
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--keep-fallback", action="store_true")
    a = ap.parse_args(argv)
    cmd_convert(a) if a.cmd == "convert" else cmd_apply(a)


if __name__ == "__main__":
    main()
