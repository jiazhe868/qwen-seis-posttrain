"""Failure types on D-Num-hard and D-Report (2026-11-10): are the errors conceptual or in the execution?

    source scripts/slurm/env.sh
    python scripts/talk/failure_types.py rules              # D-Num-hard: every answer of base, SFT and RL, no LLM
    python scripts/talk/failure_types.py label --pilot 2    # GPT-6 Luna on a few answers, to read them first
    python scripts/talk/failure_types.py label              # the D-Num-hard sample and every D-Report answer
    python scripts/talk/failure_types.py summary            # → $WORK/dpt/talk/failure_types.json (aggregates)

**D-Num-hard** (160 items, 8 answers per model at 32K, medium effort).
- `rules` grades every answer the way the harness does: thinking stripped, fail closed, `grade_numeric` with strict
  units.
- A failure is one of:
  - unfinished: the thinking never closed within 32K;
  - an answer-format failure: several answers, a missing unit, a unit of the wrong dimension, unparseable;
  - a wrong value.
- A wrong value gets a numeric signature against the gold value:
  - near miss: within 5× the item's tolerance;
  - off by a power of ten;
  - degrees and radians swapped;
  - off by a simple factor: 2, 3, 4, π, 2π, √2 or their inverses;
  - a wrong sign;
  - other.
- It is "systematic" when at least 3 of the model's 8 answers to that item give the same wrong value (within
  tolerance). A shared wrong value points to a wrong method rather than a slip.
- For an unfinished answer, `rules` also records whether a number matching the correct value (within 1.5× the
  item's tolerance, in the answer's unit) appears in the reasoning, and how far in. Such an answer had the value and
  ran out of room; an answer without it never got there.
- `label` gives GPT-6 Luna a stratified sample of the wrong values: up to 10 per model and family, distinct items
  first. Luna gets the question, the correct value, the solver's code (the reference method) and the model's written
  solution, and decides:
  - **concept:** a wrong formula, a wrong assumption, or a misread question;
  - **execution:** the right method but an arithmetic, unit, input, rounding or sign slip;
  - **unclear.**
- `label` also gives Luna up to 5 unfinished answers per model and family, with the first 6,000 and the last 3,000
  characters of the reasoning. Was the approach right (concept), or was it right but stuck in checks, arithmetic or
  units (execution)?

**D-Report** (100 questions, one report per model). Luna reads each report against:
- its 5 rubric criteria: met, partial or missed;
- the 5 common errors written with the question: present or not;
- and names the main weakness.

A shortfall is typed as one of:
- **concept:** wrong or missing science;
- **quantitative:** wrong numbers, units, magnitudes;
- **evidence:** unsupported claims, made-up or unverifiable references;
- **coverage:** a required part left out;
- **clarity:** organisation, length, format.

GPT-6 Luna is the eval-only judge (`dpt.eval.hosted`, OpenAI terms): nothing it writes enters training data. The
per-answer labels, with short evidence quotes, stay in `$WORK/dpt/talk/failure_types/`. The summary holds counts only.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import datetime as dt
import inspect
import json
import math
import os
import random
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from dpt.chat.thinking import split_thinking
from dpt.common.io import read_jsonl
from dpt.data.gen import families as F
from dpt.data.gen import seismology_hard  # noqa: F401  (registers the hard and held-out families)
from dpt.eval.graders.numeric import extract_final_answer, grade_numeric

WORK = Path(os.environ.get("WORK", "."))
R = WORK / "dpt" / "eval_results"
SECURE = WORK / "dpt" / "eval_secure"
OUT = WORK / "dpt" / "talk" / "failure_types"
MODELS = ("base_medium", "sft", "rl_seed1")
FACTORS = {"2": 2.0, "1/2": 0.5, "3": 3.0, "1/3": 1 / 3, "4": 4.0, "1/4": 0.25, "π": math.pi, "1/π": 1 / math.pi,
           "2π": 2 * math.pi, "1/(2π)": 1 / (2 * math.pi), "√2": math.sqrt(2), "1/√2": 1 / math.sqrt(2)}
DEG = 180 / math.pi


# ---------------------------------------------------------------------------------------------- rules
def signature(pred: float, gold: float, tol: float) -> str:
    if abs(pred - gold) <= 5 * tol:
        return "near_miss"
    if gold == 0 or pred == 0:
        return "other"
    r = pred / gold
    if r < 0:
        return "sign" if abs(abs(r) - 1) < 0.01 else "other"
    k = math.log10(r)
    if round(k) != 0 and abs(k - round(k)) < 0.005:
        return "power_of_ten"
    if abs(r / DEG - 1) < 0.01 or abs(r * DEG - 1) < 0.01:
        return "deg_rad"
    if any(abs(r / f - 1) < 0.01 for f in FACTORS.values()):
        return "simple_factor"
    return "other"


SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
SCI = re.compile(r"(\d+(?:\.\d+)?)\s*(?:×|x|\\times|\*|·)\s*10\s*\^?\s*\{?\s*([-+−]?\d+)\s*\}?")
NUMBER = re.compile(r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def gold_seen(thinking: str, gold: dict[str, Any]) -> float | None:
    """How far into the reasoning (0–1) a number matching the gold value first appears; None if it never does."""
    v = float(gold["value"])
    tol = 1.5 * max(float(gold.get("abs_tol", 0.0)), float(gold.get("rel_tol", 0.01)) * abs(v), 1e-12)
    t = thinking.translate(SUP).replace("−", "-")
    hits = []
    for m in SCI.finditer(t):
        try:
            x = float(m.group(1)) * 10 ** int(m.group(2).replace("−", "-"))
        except (ValueError, OverflowError):
            continue
        if abs(abs(x) - abs(v)) <= tol:
            hits.append(m.start())
    for m in NUMBER.finditer(t):
        try:
            x = float(m.group(0))
        except ValueError:
            continue
        if abs(x - abs(v)) <= tol:
            hits.append(m.start())
            break
    return round(min(hits) / max(1, len(t)), 3) if hits else None


def grade(text: str, gold: dict[str, Any]) -> dict[str, Any]:
    sp = split_thinking(text)
    if not sp.has_end:
        seen = gold_seen(sp.thinking or text, gold)
        return {"kind": "unfinished", "gold_seen": seen is not None, "gold_first_at": seen}
    if not sp.answer:
        return {"kind": "format", "reason": "no_answer"}
    try:
        g = grade_numeric(sp.answer, gold, missing_unit="strict")
    except Exception:
        return {"kind": "format", "reason": "grader_exception"}
    if g.correct:
        return {"kind": "right"}
    d = g.details or {}
    if g.reason != "wrong_value" or "pred_si" not in d:
        return {"kind": "format", "reason": str(g.reason)}
    return {"kind": "wrong_value", "pred_si": float(d["pred_si"]), "gold_si": float(d["gold_si"]),
            "tol": float(d["tol"]), "answer": str(d.get("answer", ""))[:200],
            "signature": signature(float(d["pred_si"]), float(d["gold_si"]), float(d["tol"]))}


def mark_systematic(rows: list[dict[str, Any]]) -> None:
    """Wrong values shared (within tolerance) by >= 3 of a model's answers to one item are systematic."""
    wrong = sorted((r for r in rows if r["kind"] == "wrong_value"), key=lambda r: r["pred_si"])
    clusters: list[list[dict[str, Any]]] = []
    for r in wrong:
        if clusters and abs(r["pred_si"] - clusters[-1][-1]["pred_si"]) <= r["tol"]:
            clusters[-1].append(r)
        else:
            clusters.append([r])
    for c in clusters:
        for r in c:
            r["systematic"] = len(c) >= 3


def rules(_: argparse.Namespace) -> None:
    items = {r["id"]: r for r in read_jsonl(SECURE / "d_num_hard.jsonl")}
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(OUT / "numhard_rules.jsonl", "w") as fh:
        for m in MODELS:
            for g in read_jsonl(R / "d_num_hard" / m / "d_num_hard" / "generations.jsonl"):
                it = items[g["item_id"]]
                rows = []
                for i, s in enumerate(g["samples"]):
                    r = {"model": m, "item": g["item_id"], "sample": i, "family": it["meta"]["family"],
                         "n_tokens": s.get("n_tokens"), **grade(s["text"], it["answer"])}
                    rows.append(r)
                mark_systematic(rows)
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
                    n += 1
    print(f"{n} answers graded → {OUT / 'numhard_rules.jsonl'}")


# ---------------------------------------------------------------------------------------------- the judge
class Luna:
    """GPT-6 Luna through dpt.eval.hosted's request shape, keeping the token usage."""

    def __init__(self, effort: str, max_tokens: int):
        from dpt.common.config import load_config

        j = load_config().project.judges.second
        key = os.environ.get(str(j.api_key_env))
        if not key:
            sys.exit(f"no ${j.api_key_env}: put the key in ~/.config/dpt/openai_key (scripts/slurm/env.sh)")
        self.url = str(j.endpoint).rstrip("/") + "/v1/chat/completions"
        self.model = str(j.model)
        self.body = {"model": self.model, "max_completion_tokens": max_tokens, "reasoning_effort": effort}
        self.client = httpx.Client(timeout=900, headers={"Authorization": f"Bearer {key}"})
        self.usage: collections.Counter[str] = collections.Counter()
        self.lock = threading.Lock()

    def ask(self, system: str, user: str) -> tuple[str, str]:
        body = {**self.body, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        delay = 4.0
        for attempt in range(7):
            try:
                r = self.client.post(self.url, json=body)
                if r.status_code == 429 or r.status_code >= 500:
                    raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
                r.raise_for_status()
                break
            except (httpx.TransportError, httpx.HTTPStatusError):
                if attempt == 6:
                    raise
                time.sleep(delay)
                delay *= 2
        data = r.json()
        with self.lock:
            for k, v in (data.get("usage") or {}).items():
                if isinstance(v, int):
                    self.usage[k] += v
        return (data["choices"][0].get("message") or {}).get("content") or "", data.get("model") or ""


def json_reply(text: str) -> dict[str, Any] | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------------------------- D-Num-hard prompts
NUM_SYSTEM = """You diagnose why a model's answer to a seismology calculation problem is wrong. You get the problem, the correct result, the reference method as Python code, and the model's written solution. The model's final value is wrong; decide the main cause.

Categories:
- "concept": the method is wrong: a wrong physical relation or formula, a missing or extra term, a wrong geometry or assumption (for example flat vs spherical, one-way vs two-way, P vs S), or the question was misread (it computed a different quantity).
- "execution": the method is right but carried out wrongly: an arithmetic slip, a unit conversion slip, a given value misread or mis-copied, rounding or precision beyond the tolerance, or a sign slip.
- "unclear": the text does not show which.

Answer with JSON only:
{"category": "concept" | "execution" | "unclear", "subtype": one of "wrong_formula", "wrong_assumption", "misread_question", "arithmetic", "unit_conversion", "input_misread", "rounding", "sign", "other", "evidence": "<= 25 words on what went wrong"}"""


def reference_code(family: str) -> str:
    fam = F.FAMILIES[family]
    src = inspect.getsource(fam.solver)
    mod = sys.modules[fam.solver.__module__]
    helpers = []
    for name in sorted(set(re.findall(r"\b(_[a-z][a-z0-9_]*)\(", src))):
        fn = getattr(mod, name, None)
        if callable(fn) and fn is not fam.solver:
            try:
                helpers.append(inspect.getsource(fn))
            except (TypeError, OSError):
                pass
    return "\n".join(helpers + [src])


def solution_text(text: str) -> str:
    sp = split_thinking(text)
    ans = (sp.answer or "").strip()
    out = ans[:4000]
    if len(ans) < 800:
        out = "[end of the model's reasoning]\n" + (sp.thinking or "")[-3000:] + "\n[final answer]\n" + out
    return out


def numhard_jobs(rows: list[dict[str, Any]], per: int, seed: int = 1) -> list[dict[str, Any]]:
    """Up to `per` wrong values per (model, family), one per item first, then a second per item, and so on."""
    rng = random.Random(seed)
    by: dict[tuple[str, str], dict[str, list]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["kind"] == "wrong_value":
            by[(r["model"], r["family"])][r["item"]].append(r)
    jobs = []
    for key in sorted(by):
        per_item = {k: rng.sample(v, len(v)) for k, v in sorted(by[key].items())}
        order = rng.sample(sorted(per_item), len(per_item))
        picked: list[dict[str, Any]] = []
        depth = 0
        while len(picked) < per and any(len(per_item[i]) > depth for i in order):
            for i in order:
                if len(per_item[i]) > depth and len(picked) < per:
                    picked.append(per_item[i][depth])
            depth += 1
        jobs += picked
    return jobs


UNF_SYSTEM = """A model was solving a seismology calculation problem and ran out of its 32K-token budget before giving an answer. You get the problem, the correct result, the reference method as Python code, and the start and the end of the model's reasoning. Decide why it did not finish.

Categories:
- "concept": its approach is wrong: a wrong relation or formula, a wrong geometry or assumption, or it is computing a different quantity, so it cannot reach the right answer.
- "execution": its approach is right (or it reached the right value) but it did not finish: it kept re-checking, struggled with the arithmetic or units, or went back and forth between equivalent choices.
- "unclear": the text shown does not tell.

Answer with JSON only:
{"category": "concept" | "execution" | "unclear", "subtype": one of "wrong_formula", "wrong_assumption", "misread_question", "verification_loop", "arithmetic_struggle", "unit_struggle", "indecision", "other", "reached_correct_value": true | false | null, "evidence": "<= 25 words"}"""


# ---------------------------------------------------------------------------------------------- D-Report prompts
REP_SYSTEM = """You are an expert seismologist reviewing a written answer to a seismology question. You get the question, its grading rubric (weighted criteria), a list of common errors for this question, and the answer.

For every rubric criterion, decide "met", "partial" or "missed". For "partial" and "missed", give the type of the shortfall:
- "concept": wrong or missing science: an incorrect mechanism, relation, interpretation or physical claim, or a key concept left out because it is not understood
- "quantitative": wrong numbers, calculations, units or orders of magnitude
- "evidence": claims without support, made-up or unverifiable references
- "coverage": a required part of the question is not addressed at all (an omission, not an error)
- "clarity": disorganised, padded, off-topic or badly formatted, so the content does not come across
For every common error, say whether the answer makes it. Then name the answer's main weakness.

Answer with JSON only:
{"criteria": [{"id": 1, "verdict": "met" | "partial" | "missed", "type": null | "concept" | "quantitative" | "evidence" | "coverage" | "clarity", "evidence": "<= 20 words"}, ...],
 "common_errors": [{"id": 1, "present": true | false}, ...],
 "main_weakness": {"type": "concept" | "quantitative" | "evidence" | "coverage" | "clarity" | "none", "summary": "<= 25 words"}}"""


def report_prompt(item: dict[str, Any], report: str) -> str:
    a = item["answer"]
    crit = "\n".join(f"{i}. (weight {c['weight']}) {c['criterion']}" for i, c in enumerate(a["rubric"], start=1))
    errs = "\n".join(f"{i}. {e}" for i, e in enumerate(a.get("common_errors") or [], start=1))
    return (f"## Question\n{item['question']}\n\n## Rubric criteria\n{crit}\n\n## Common errors\n{errs}\n\n"
            f"## Answer\n{report[:14000]}")


# ---------------------------------------------------------------------------------------------- label
def label(a: argparse.Namespace) -> None:
    luna = Luna(a.effort, a.max_tokens)
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "labels.jsonl"
    done = {json.loads(line)["key"] for line in open(cache)} if cache.exists() else set()
    rows = list(read_jsonl(OUT / "numhard_rules.jsonl"))
    num_items = {r["id"]: r for r in read_jsonl(SECURE / "d_num_hard.jsonl")}
    texts: dict[tuple[str, str], list[str]] = {}
    jobs: list[tuple[str, dict[str, Any]]] = []
    per = a.pilot if a.pilot else a.per
    picked = numhard_jobs(rows, per)
    if a.pilot:
        picked = picked[:: max(1, len(picked) // a.pilot)][: a.pilot]
    need = {(r["model"], r["item"]) for r in picked}
    unf_items = {(r["model"], r["item"]) for r in rows if r["kind"] == "unfinished"}
    for m in MODELS:
        for g in read_jsonl(R / "d_num_hard" / m / "d_num_hard" / "generations.jsonl"):
            if (m, g["item_id"]) in need or (m, g["item_id"]) in unf_items:
                texts[(m, g["item_id"])] = [s["text"] for s in g["samples"]]
    unf = [r for r in rows if r["kind"] == "unfinished"]
    unf_picked = numhard_jobs([{**r, "kind": "wrong_value"} for r in unf], a.pilot if a.pilot else a.per_unfinished,
                              seed=2)
    if a.pilot:
        unf_picked = unf_picked[:: max(1, len(unf_picked) // a.pilot)][: a.pilot]
    need |= {(r["model"], r["item"]) for r in unf_picked}
    for r in picked:
        jobs.append(("num", r))
    for r in unf_picked:
        jobs.append(("unf", r))
    rep_items = {r["id"]: r for r in read_jsonl(SECURE / "d_report.jsonl")}
    reports = []
    for m in MODELS:
        for g in read_jsonl(R / m / "d_report" / "generations.jsonl"):
            reports.append({"model": m, "item": g["item_id"], "text": g["samples"][0]["text"]})
    if a.pilot:
        reports = reports[:: max(1, len(reports) // a.pilot)][: a.pilot]
    jobs += [("rep", r) for r in reports]
    jobs = [(t, r) for t, r in jobs if f"{t}|{r['model']}|{r['item']}|{r.get('sample', 0)}" not in done]
    print(f"{len(jobs)} to label ({len(done)} cached); effort {a.effort}", flush=True)
    write_lock = threading.Lock()

    def one(job: tuple[str, dict[str, Any]]) -> None:
        kind, r = job
        key = f"{kind}|{r['model']}|{r['item']}|{r.get('sample', 0)}"
        out: dict[str, Any] = {"key": key, "task": kind, "model": r["model"], "item": r["item"]}
        if kind == "num":
            it = num_items[r["item"]]
            text = texts[(r["model"], r["item"])][r["sample"]]
            gold = it["answer"]
            ext = extract_final_answer(split_thinking(text).answer or "")
            user = (f"## Problem\n{it['question']}\n\n## Correct result\n{gold['value']:.6g} {gold['unit']} "
                    f"(tolerance {100 * gold.get('rel_tol', 0.01):g}%)\n\n## Model's final answer\n{ext.answer}\n\n"
                    f"## Reference method (Python)\n```python\n{reference_code(it['meta']['family'])}\n```\n\n"
                    f"## Model's written solution\n{solution_text(text)}")
            out.update(family=r["family"], sample=r["sample"], signature=r["signature"],
                       systematic=r.get("systematic", False))
            system = NUM_SYSTEM
        elif kind == "unf":
            it = num_items[r["item"]]
            think = split_thinking(texts[(r["model"], r["item"])][r["sample"]]).thinking or ""
            gold = it["answer"]
            user = (f"## Problem\n{it['question']}\n\n## Correct result\n{gold['value']:.6g} {gold['unit']}\n\n"
                    f"## Reference method (Python)\n```python\n{reference_code(it['meta']['family'])}\n```\n\n"
                    f"## Start of the model's reasoning\n{think[:6000]}\n\n[... {max(0, len(think) - 9000):,} "
                    f"characters left out ...]\n\n## End of the reasoning (cut off by the 32K-token limit)\n"
                    f"{think[-3000:]}")
            out.update(family=r["family"], sample=r["sample"], gold_seen=r.get("gold_seen"),
                       gold_first_at=r.get("gold_first_at"))
            system = UNF_SYSTEM
        else:
            sp = split_thinking(r["text"])
            if not sp.has_end or not (sp.answer or "").strip():
                out.update(unfinished=True)
                with write_lock, open(cache, "a") as fh:
                    fh.write(json.dumps(out) + "\n")
                return
            user, system = report_prompt(rep_items[r["item"]], sp.answer), REP_SYSTEM
        try:
            reply, served = luna.ask(system, user)
            out.update(served=served, label=json_reply(reply))
        except Exception as e:                     # recorded, retried on the next run
            print(f"FAIL {key}: {type(e).__name__}: {e}", flush=True)
            return
        with write_lock, open(cache, "a") as fh:
            fh.write(json.dumps(out) + "\n")

    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, _ in enumerate(ex.map(one, jobs), start=1):
            if i % 20 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} in {time.time() - t0:.0f}s; tokens {dict(luna.usage)}", flush=True)
    (OUT / "usage.json").write_text(json.dumps({"at": dt.datetime.now().isoformat(timespec="minutes"),
                                                "usage": dict(luna.usage), "effort": a.effort}, indent=1))


# ---------------------------------------------------------------------------------------------- summary
def share(c: collections.Counter, n: int) -> dict[str, float]:
    return {k: round(100 * v / n, 1) for k, v in c.most_common()} if n else {}


def summary(_: argparse.Namespace) -> None:
    rows = list(read_jsonl(OUT / "numhard_rules.jsonl"))
    labels = [json.loads(line) for line in open(OUT / "labels.jsonl")] if (OUT / "labels.jsonl").exists() else []
    num: dict[str, Any] = {}
    for m in MODELS:
        mr = [r for r in rows if r["model"] == m]
        kinds = collections.Counter(r["kind"] for r in mr)
        fails = [r for r in mr if r["kind"] != "right"]
        wrong = [r for r in mr if r["kind"] == "wrong_value"]
        lab = [x for x in labels if x["task"] == "num" and x["model"] == m and x.get("label")]
        cats = collections.Counter(str(x["label"].get("category")) for x in lab)
        subs = collections.Counter(f"{x['label'].get('category')}:{x['label'].get('subtype')}" for x in lab)
        by_sys = {s: collections.Counter(str(x["label"].get("category")) for x in lab if x["systematic"] == s)
                  for s in (True, False)}
        fam = {}
        for f in sorted({r["family"] for r in mr}):
            fr = [r for r in mr if r["family"] == f]
            fc = collections.Counter(r["kind"] for r in fr)
            fl = collections.Counter(str(x["label"].get("category")) for x in lab if x["family"] == f)
            fam[f] = {"answers": len(fr), "kinds": share(fc, len(fr)), "labelled": sum(fl.values()),
                      "labels": dict(fl)}
        unf = [r for r in mr if r["kind"] == "unfinished"]
        ul = [x for x in labels if x["task"] == "unf" and x["model"] == m and x.get("label")]
        ucats = collections.Counter(str(x["label"].get("category")) for x in ul)
        usubs = collections.Counter(f"{x['label'].get('category')}:{x['label'].get('subtype')}" for x in ul)
        concept_share = cats["concept"] / max(1, sum(cats.values()))
        u_concept = ucats["concept"] / max(1, sum(ucats.values()))

        def weighted(kind: str, task: str) -> dict[str, float]:
            """Category shares of all `kind` answers, from the per-family label shares weighted by each family's
            count (the sample takes up to N per family, but failures concentrate in a few families). A family
            without labels takes the model's pooled share."""
            pool = collections.Counter(str(x["label"].get("category")) for x in labels
                                       if x["task"] == task and x["model"] == m and x.get("label"))
            out: collections.Counter[str] = collections.Counter()
            for f in {r["family"] for r in mr}:
                n_f = sum(1 for r in mr if r["family"] == f and r["kind"] == kind)
                c = collections.Counter(str(x["label"].get("category")) for x in labels if x["task"] == task
                                        and x["model"] == m and x.get("family") == f and x.get("label")) or pool
                tot = sum(c.values())
                for k, v in c.items():
                    out[k] += n_f * v / max(1, tot)
            return {k: round(v, 1) for k, v in out.most_common()}

        w_wrong, w_unf = weighted("wrong_value", "num"), weighted("unfinished", "unf")
        fam_unf = {f: dict(collections.Counter(str(x["label"].get("category")) for x in ul if x.get("family") == f))
                   for f in sorted({x.get("family") for x in ul})}
        num[m] = {
            "answers": len(mr), "kinds": dict(kinds), "kinds_pct": share(kinds, len(mr)),
            "format_reasons": dict(collections.Counter(r["reason"] for r in mr if r["kind"] == "format")),
            "wrong_signatures": dict(collections.Counter(r["signature"] for r in wrong)),
            "wrong_systematic_share": round(sum(r.get("systematic", False) for r in wrong) / max(1, len(wrong)), 3),
            "labelled": len(lab), "label_categories": dict(cats), "label_subtypes": dict(subs.most_common()),
            "label_by_systematic": {str(k): dict(v) for k, v in by_sys.items()},
            "unfinished_gold_seen_share": round(sum(r.get("gold_seen", False) for r in unf) / max(1, len(unf)), 3),
            "unfinished_labelled": len(ul), "unfinished_label_categories": dict(ucats),
            "unfinished_label_subtypes": dict(usubs.most_common()),
            "unfinished_reached_value": dict(collections.Counter(str(x["label"].get("reached_correct_value"))
                                                                 for x in ul)),
            "est_concept_share_of_failures": round((concept_share * len(wrong) + u_concept * len(unf))
                                                   / max(1, len(fails)), 3),
            "weighted_wrong": w_wrong, "weighted_unfinished": w_unf, "unfinished_labels_by_family": fam_unf,
            "est_concept_share_of_failures_weighted": round((w_wrong.get("concept", 0) + w_unf.get("concept", 0))
                                                            / max(1, len(fails)), 3),
            "families": fam,
        }
    rep: dict[str, Any] = {}
    for m in MODELS:
        lab = [x for x in labels if x["task"] == "rep" and x["model"] == m]
        ok = [x for x in lab if x.get("label") and isinstance(x["label"].get("criteria"), list)]
        verdicts, types, errs, main = (collections.Counter() for _ in range(4))
        weighted_short = collections.Counter()
        for x in ok:
            for c in x["label"]["criteria"]:
                verdicts[str(c.get("verdict"))] += 1
                if c.get("verdict") in ("partial", "missed"):
                    types[str(c.get("type"))] += 1
                    weighted_short[str(c.get("type"))] += 1.0 if c.get("verdict") == "missed" else 0.5
            for e in x["label"].get("common_errors") or []:
                errs["present" if e.get("present") else "absent"] += 1
            main[str((x["label"].get("main_weakness") or {}).get("type"))] += 1
        rep[m] = {"reports": len(lab), "unfinished": sum(1 for x in lab if x.get("unfinished")),
                  "labelled": len(ok), "criteria_verdicts": dict(verdicts),
                  "criteria_verdicts_pct": share(verdicts, sum(verdicts.values())),
                  "shortfall_types": dict(types), "shortfall_types_pct": share(types, sum(types.values())),
                  "shortfall_weighted": {k: round(v, 1) for k, v in weighted_short.most_common()},
                  "common_errors_present_pct": round(100 * errs["present"] / max(1, sum(errs.values())), 1),
                  "main_weakness": dict(main.most_common())}
    trunc: dict[str, Any] = {}                     # answers cut off at 32K, per set (results.json of the final eval)
    for m in MODELS:
        res = json.loads((R / m / "results.json").read_text())
        sets = {**(res.get("general") or {}), **(res.get("sets") or {})}
        hard = R / "d_num_hard" / m / "results.json"
        if hard.exists():
            sets.update(json.loads(hard.read_text()).get("sets") or {})
        trunc[m] = {k: {"score": round(v["score"], 1), "cut_off": round(v.get("truncation_rate") or 0.0, 3),
                        "tokens": round(v.get("mean_output_tokens") or 0)}
                    for k, v in sets.items() if isinstance(v, dict) and "score" in v}
    usage = json.loads((OUT / "usage.json").read_text()) if (OUT / "usage.json").exists() else {}
    out = {"collected_at": dt.datetime.now().strftime("%Y-%m-%dT%H:%M"), "d_num_hard": num, "d_report": rep,
           "cut_off_32k": trunc,
           "judge": sorted({x.get("served") for x in labels if x.get("served")}), "usage": usage}
    p = WORK / "dpt" / "talk" / "failure_types.json"
    p.write_text(json.dumps(out, indent=1))
    for m in MODELS:
        n = num[m]
        print(m, "D-Num-hard", n["kinds_pct"], "| wrong:", n["wrong_signatures"], "| systematic",
              n["wrong_systematic_share"], "| Luna:", n["label_categories"])
        print("   D-Report", rep[m]["criteria_verdicts_pct"], "| shortfalls:", rep[m]["shortfall_types_pct"],
              "| main:", rep[m]["main_weakness"])
    print(p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("rules")
    lb = sub.add_parser("label")
    lb.add_argument("--per", type=int, default=10, help="wrong values per model and family")
    lb.add_argument("--per-unfinished", type=int, default=5, help="unfinished answers per model and family")
    lb.add_argument("--pilot", type=int, default=0, help="label only this many of each kind")
    lb.add_argument("--effort", default="medium")
    lb.add_argument("--max-tokens", type=int, default=12000)
    lb.add_argument("--workers", type=int, default=6)
    sub.add_parser("summary")
    a = ap.parse_args()
    {"rules": rules, "label": label, "summary": summary}[a.cmd](a)


if __name__ == "__main__":
    main()
