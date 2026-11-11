"""Teacher paraphrase of T1 template questions, with a numeric-preservation check.

    python -m dpt.data.gen.paraphrase --in $GEN/t1_candidates.jsonl --out $GEN/t1_paraphrased.jsonl \\
        --fraction 0.5 --url http://host:8000 --model teacher --tokenizer Qwen/Qwen3.8-27B

Templates give every instance of a family the same few wordings, so a policy could learn the wording
instead of the physics (plan §6: "worded by templates + teacher paraphrase"). A chosen ``--fraction`` of
the questions (chosen by a hash of the uid, so reruns choose the same ones) is rewritten by the teacher.
A rewrite replaces the original only if it passes every check below; otherwise the original stays:

* **numbers**: the same multiset of numeric values (``1,200``, ``3.2 × 10^4``, ``4e-3`` are parsed), so
  nothing was dropped, changed or added. Spelled-out numbers count as a change and are rejected.
* **units**: the same set of unit tokens (``km/s``, ``MPa``, "seconds" = ``s`` …), so the quantity asked
  for and the unit it is asked in survive;
* **no leak**: no formula (``=``), no ``\\boxed``, no "answer is", and not the label's value;
* **length**: 0.6–1.8× the original word count.

The label is unchanged (the numbers are), and ``meta.generator`` is kept, so the verify stage re-checks
the paraphrase against the solver like any other candidate. ``meta.paraphrase`` records the status, the
model and the original question; a rewritten candidate gets a new uid (uids hash the question).

Resumable: decisions are appended to ``<out>.progress.jsonl``; ``--out`` is written at the end.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from dpt.common.io import read_jsonl
from dpt.common.logging import get_logger
from dpt.common.seeding import derive_seed
from dpt.common.types import T1_NUM
from dpt.data.gen.common import rng_for, write_candidates
from dpt.data.gen.teacher import Appender, ChatModel, add_model_args, last_block, model_from_args, read_existing
from dpt.data.schema import Problem

log = get_logger(__name__)

PROMPT = """Rewrite the problem below in your own words, as a different author would pose it.

Rules:
- Keep every number exactly as written, with its unit. Add no other numbers, not even spelled-out ones.
- Keep what is given and what is asked exactly the same, including the unit the answer is asked in.
- Do not solve the problem, do not hint at the method, and do not state any result.
- Write one self-contained problem in plain text.

Problem:
{question}

Return only the rewritten problem between <problem> and </problem>."""

_NUM = re.compile(
    r"(?<![\w.^])([-−+]?\d+(?:,\d{3})*(?:\.\d+)?)"
    r"(?:\s*(?:×|x|\*|·)\s*10\s*\^\s*\(?\s*([-−+]?\d+)\s*\)?|[eE]([-−+]?\d+))?"
)

# unit token → canonical form. Words a rewrite may use instead of the symbol map to the same token.
_UNIT_ALIASES = {
    "s": "s", "sec": "s", "secs": "s", "second": "s", "seconds": "s",
    "ms": "ms", "millisecond": "ms", "milliseconds": "ms",
    "min": "min", "minute": "min", "minutes": "min", "h": "h", "hr": "h", "hour": "h", "hours": "h",
    "day": "day", "days": "day", "yr": "yr", "year": "yr", "years": "yr",
    "km": "km", "kilometer": "km", "kilometers": "km", "kilometre": "km", "kilometres": "km",
    "m": "m", "meter": "m", "meters": "m", "metre": "m", "metres": "m",
    "cm": "cm", "centimeter": "cm", "centimeters": "cm", "centimetre": "cm", "centimetres": "cm",
    "mm": "mm", "millimeter": "mm", "millimeters": "mm", "millimetre": "mm", "millimetres": "mm",
    "µm": "µm", "um": "µm", "nm": "nm",
    "km/s": "km/s", "m/s": "m/s", "cm/s": "cm/s", "mm/s": "mm/s", "m/s²": "m/s²", "m/s^2": "m/s²",
    "cm/s²": "cm/s²", "cm/s^2": "cm/s²", "gal": "gal", "g": "g",
    "hz": "Hz", "hertz": "Hz", "khz": "kHz", "mhz": "mHz",
    "pa": "Pa", "kpa": "kPa", "mpa": "MPa", "gpa": "GPa", "bar": "bar", "bars": "bar", "kbar": "kbar",
    "n·m": "N·m", "dyne-cm": "dyne-cm", "dyn·cm": "dyne-cm", "j": "J", "joule": "J", "joules": "J",
    "kg/m³": "kg/m³", "kg/m^3": "kg/m³", "g/cm³": "g/cm³", "g/cm^3": "g/cm³",
    "deg": "deg", "degree": "deg", "degrees": "deg", "rad": "rad", "radian": "rad", "radians": "rad",
    "percent": "%", "db": "dB", "decibel": "dB", "decibels": "dB",
}
# Symbols written right after a number ("30°", "5%"): matched anywhere.
_UNIT_SYMBOLS = {"°": "deg", "%": "%", "‰": "‰"}
_UNIT_RE = re.compile(
    r"(?<![\w/])(" + "|".join(sorted((re.escape(k) for k in _UNIT_ALIASES), key=len, reverse=True))
    + r")(?![\w/²^³])",
    re.IGNORECASE,
)
_LEAK_RE = re.compile(r"\\boxed|=|answer is|the answer:|solution:", re.IGNORECASE)


def numbers(text: str) -> Counter:
    """Multiset of the numeric values in ``text`` (6 significant figures), with ``× 10^k`` and ``e-k`` folded in."""
    out: Counter = Counter()
    for m in _NUM.finditer(text.replace("−", "-")):
        mant = float(m.group(1).replace(",", "").replace("−", "-"))
        exp = m.group(2) or m.group(3)
        v = mant * 10.0 ** int(exp.replace("−", "-")) if exp else mant
        out[_key(v)] += 1
    return out


def _key(v: float) -> str:
    return "0" if v == 0 or not math.isfinite(v) else f"{v:.6g}"


def units(text: str) -> set[str]:
    found = {canon for sym, canon in _UNIT_SYMBOLS.items() if sym in text}
    for m in _UNIT_RE.finditer(text):
        tok = m.group(1)
        canon = _UNIT_ALIASES.get(tok) or _UNIT_ALIASES.get(tok.lower())
        if canon:
            found.add(canon)
    return found


def check(original: str, rewrite: str | None, answer_value: float | None = None) -> str | None:
    """None if ``rewrite`` may replace ``original``, else the reason it may not."""
    if not rewrite:
        return "empty"
    if _LEAK_RE.search(rewrite):
        return "leaks_solution"
    n0, n1 = len(original.split()), len(rewrite.split())
    if not (0.6 * n0 <= n1 <= 1.8 * n0):
        return "length"
    if numbers(original) != numbers(rewrite):
        return "numbers_changed"
    if units(original) != units(rewrite):
        return "units_changed"
    if answer_value is not None:
        a = _key(answer_value)
        if a in numbers(rewrite) and a not in numbers(original):
            return "leaks_answer"
    return None


def chosen(uid: str, fraction: float, seed: int) -> bool:
    return rng_for(seed, "paraphrase", uid).random() < fraction


def paraphrase_one(model: ChatModel, rec: dict[str, Any], seed: int) -> dict[str, Any]:
    q = rec["question"]
    try:
        reply = model.generate([{"role": "user", "content": PROMPT.format(question=q)}], n=1,
                               seed=derive_seed(seed, "paraphrase", rec["uid"]))[0]
    except Exception as e:  # a failed request keeps the original; it is retried on the next run
        log.warning("paraphrase %s: %s", rec["uid"], e)
        return {"uid": rec["uid"], "status": "error", "reason": str(e)[:200]}
    text = None if reply.truncated else last_block(reply.text, "problem")
    text = " ".join(text.split()) if text else None
    reason = check(q, text, (rec.get("answer") or {}).get("value"))
    if reason:
        return {"uid": rec["uid"], "status": "rejected", "reason": reason}
    return {"uid": rec["uid"], "status": "ok", "question": text}


def apply(rec: dict[str, Any], decision: dict[str, Any] | None, model_id: str) -> dict[str, Any]:
    """The record with its paraphrase applied (new uid) or annotated as rejected; unchanged if not chosen."""
    if decision is None:
        return rec
    meta = dict(rec.get("meta") or {})
    if decision["status"] != "ok":
        meta["paraphrase"] = {"status": "rejected", "reason": decision.get("reason", ""), "model": model_id}
        return {**rec, "meta": meta}
    meta["paraphrase"] = {"status": "ok", "model": model_id, "original": rec["question"],
                          "original_uid": rec["uid"]}
    new = {k: v for k, v in rec.items() if k != "uid"}
    return Problem.from_dict({**new, "question": decision["question"], "meta": meta}).to_dict()


def run(model: ChatModel, inp: str | Path, out: str | Path, *, fraction: float = 0.5, seed: int = 1,
        tasks: Sequence[str] = (T1_NUM,), concurrency: int = 32) -> dict[str, Any]:
    recs = [Problem.from_dict(r).to_dict() for r in read_jsonl(inp)]
    progress = Path(f"{out}.progress.jsonl")
    done = {d["uid"]: d for d in read_existing(progress) if d.get("status") in ("ok", "rejected")}
    todo = [r for r in recs if r["task"] in tasks and chosen(r["uid"], fraction, seed) and r["uid"] not in done]
    app = Appender(progress)

    def one(r: dict[str, Any]) -> None:
        d = paraphrase_one(model, r, seed)
        app.write([d])
        if d["status"] in ("ok", "rejected"):
            done[r["uid"]] = d

    with ThreadPoolExecutor(max(1, concurrency)) as ex:
        list(ex.map(one, todo))
    rows = [apply(r, done.get(r["uid"]), model.model_id) for r in recs]
    write_candidates(out, rows)
    st = Counter(d["status"] if d["status"] == "ok" else f"rejected:{d.get('reason')}" for d in done.values())
    chosen_n = sum(1 for r in recs if r["task"] in tasks and chosen(r["uid"], fraction, seed))
    stats = {"records": len(recs), "chosen": chosen_n, "decided": len(done), "outcomes": dict(sorted(st.items())),
             "missing": chosen_n - len(done)}
    log.info("paraphrase: %s", stats)
    return stats


def main(argv: Sequence[str] | None = None, *, model: ChatModel | None = None) -> int:
    ap = argparse.ArgumentParser(description="Teacher paraphrase of T1 questions (numbers and units preserved)")
    ap.add_argument("--in", dest="inp", required=True, help="candidates JSONL (dpt.data.gen.t1)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fraction", type=float, default=0.5)
    ap.add_argument("--tasks", default=T1_NUM, help="comma-separated tasks to paraphrase")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=32)
    if model is None:
        add_model_args(ap, max_tokens=4096)
    args = ap.parse_args(argv)
    m = model or model_from_args(args, concurrency=args.concurrency)
    stats = run(m, args.inp, args.out, fraction=args.fraction, seed=args.seed,
                tasks=[t.strip() for t in args.tasks.split(",") if t.strip()], concurrency=args.concurrency)
    return 0 if stats["missing"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
