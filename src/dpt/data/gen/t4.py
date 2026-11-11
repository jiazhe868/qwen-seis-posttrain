"""T4 general replay: public math with verifiable answers + instruction following with checkable constraints.

T4 guards against forgetting (report §RL 题库; plan §6, ~15k). Two halves:

**Math** comes from the *train* splits of GSM8K (``openai/gsm8k``, MIT) and MATH (``EleutherAI/hendrycks_math``,
MIT). An item is kept only if its final answer is one plain number: an integer, a decimal or a simple fraction.
Anything else (``\\sqrt{2}``, intervals, ``30^\\circ``, ``5\\text{ cm}``, tuples, expressions in ``x``) is
dropped, because a label the graders might read differently is worse than no label. ``tests/data/gen/test_t4.py``
checks that the training grader (``dpt.rewards.numeric.grade_math``) and the eval grader
(``dpt.eval.graders.numeric.grade_numeric``) both accept every kept answer.

**Instruction following** pairs base prompts from ``databricks/databricks-dolly-15k`` (CC-BY-SA-3.0; the
instruction-only categories, rows without a context passage) with 1–3 machine-checkable constraints in the
canonical schema (``dpt.common.ifeval_schema``). The two IFEval checkers (training: ``dpt.rewards.instruction``;
eval: ``dpt.eval.graders.ifeval``) are separate implementations, so only constraints that both read the same
way are emitted:

* no ``language`` (the eval checker has none) and no exact word counts (the two count words differently);
  word limits are ``min_words`` / ``max_words``, sentence counts ``at_least`` only;
* bullet points are requested as markdown ``*`` bullets (the training checker also counts numbered lists);
* forbidden words come from a small bank of adverbs that rarely occur inside longer words (the eval checker
  matches substrings, the training checker whole words);
* combinations that cannot both hold, or that hold in one checker but not the other, are never drawn
  (``CONFLICTS``): lowercase with uppercase, a start/end phrase with a case rule, a title with a paragraph
  count, JSON with anything but keywords, and so on.

The training reward also has a relevance gate: a response must share ≥ 15% of the *whole prompt's* content
words (``dpt.rewards.instruction.relevance``). Constraint sentences add content words of their own, so an item
whose base prompt contributes too few of them could not pass even with a perfect answer. Such items are drawn
again with fewer constraints, and dropped if that fails (``relevance_gate_risk``). ``meta.base_prompt`` keeps the
Dolly instruction alone.

    python -m dpt.data.gen.t4 fetch --raw-dir $WORK/dpt/raw/t4            # login node; HF_HUB_DISABLE_XET=1
    python -m dpt.data.gen.t4 build --raw-dir $WORK/dpt/raw/t4 --out $WORK/dpt/data/candidates/t4.jsonl

``fetch`` writes one normalized JSONL per source (``gsm8k.jsonl``, ``math.jsonl``, ``dolly.jsonl``) plus
``fetch_manifest.json`` (repo, revision, files, license). ``build`` is deterministic for a given seed and writes
``<out>.summary.json`` (counts per source, drops per reason, constraint statistics).

Decontamination: the GSM8K and MATH *test* splits (and MATH-500) and IFEval's prompts belong in
``data_pipeline.inputs.public_benchmarks``, so the decontam stage removes any training item resembling them.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

from dpt.common.io import read_jsonl, write_json, write_jsonl
from dpt.common.logging import get_logger
from dpt.common.text import normalize_text
from dpt.common.types import T4_GENERAL
from dpt.data.gen.common import candidate, rng_for, write_candidates

log = get_logger(__name__)

ORIGIN = "dpt.data.gen.t4"

# --------------------------------------------------------------------------- sources

SOURCES: dict[str, dict[str, Any]] = {
    "gsm8k": {"repo": "openai/gsm8k", "license": "MIT", "files": r"^main/train-.*\.parquet$"},
    "math": {"repo": "EleutherAI/hendrycks_math", "license": "MIT", "files": r"^[a-z_]+/train-.*\.parquet$"},
    "dolly": {"repo": "databricks/databricks-dolly-15k", "license": "CC-BY-SA-3.0",
              "files": r"^databricks-dolly-15k\.jsonl$"},
}

DOLLY_CATEGORIES = ("open_qa", "general_qa", "brainstorming", "creative_writing")

# --------------------------------------------------------------------------- math answers

_PLAIN = re.compile(r"^-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$")
_FRAC = re.compile(r"^(-?)\\[dt]?frac\{(-?\d+)\}\{(\d+)\}$")
_SLASH = re.compile(r"^(-?\d+)/(\d+)$")
MAX_ABS_VALUE = 1e15            # beyond this a float no longer holds every integer exactly


def plain_number(text: str) -> tuple[float, str] | None:
    """``(value, answer text)`` if ``text`` is one plain number (int, decimal, simple fraction), else None."""
    t = text.strip()
    if t.startswith("$") and t.endswith("$"):
        t = t[1:-1].strip()
    t = t.replace("\\!", "").replace("{,}", ",").replace("\\,", ",")
    m = _FRAC.match(t) or _SLASH.match(t)
    if m:
        groups = m.groups()
        sign = -1.0 if (len(groups) == 3 and groups[0] == "-") else 1.0
        num, den = int(groups[-2]), int(groups[-1])
        if den == 0:
            return None
        value = sign * num / den
    elif _PLAIN.match(t):
        value = float(t.replace(",", ""))
    else:
        return None
    if not math.isfinite(value) or abs(value) > MAX_ABS_VALUE:
        return None
    return value, t


def last_boxed(text: str) -> str | None:
    """Content of the last ``\\boxed{...}`` (brace-balanced), or None."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i, depth = start + len("\\boxed{"), 1
    for j in range(i, len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return text[i:j]
    return None


def gsm8k_answer(answer_field: str) -> str | None:
    """The final answer of a GSM8K solution: the text after the last ``####``."""
    if "####" not in answer_field:
        return None
    return answer_field.rsplit("####", 1)[1].strip()


def math_items(raw_dir: Path, revision: dict[str, str]) -> tuple[list[dict[str, Any]], Counter]:
    """Math candidates from ``gsm8k.jsonl`` and ``math.jsonl`` (all that pass), plus drop counts."""
    items: list[dict[str, Any]] = []
    drops: Counter = Counter()
    for name in ("gsm8k", "math"):
        path = raw_dir / f"{name}.jsonl"
        if not path.exists():
            log.warning("no %s; skipping %s", path, name)
            continue
        for r in read_jsonl(path):
            if name == "gsm8k":
                question, final = r.get("question", ""), gsm8k_answer(r.get("answer", ""))
            else:
                question, final = r.get("problem", ""), last_boxed(r.get("solution", ""))
            if not question.strip():
                drops[f"{name}:empty_question"] += 1
                continue
            if final is None:
                drops[f"{name}:no_final_answer"] += 1
                continue
            parsed = plain_number(final)
            if parsed is None:
                drops[f"{name}:not_plain_number"] += 1
                continue
            value, text = parsed
            meta = {"t4_source": name, "answer_text": text}
            if name == "math":
                meta.update(level=r.get("level"), subject=r.get("subject") or r.get("type"))
            items.append(candidate(
                T4_GENERAL, question.strip(), {"kind": "math", "value": value, "unit": ""}, origin=ORIGIN,
                source={"license": SOURCES[name]["license"], "dataset": SOURCES[name]["repo"], "id": r["id"],
                        "split": "train", "revision": revision.get(name)},
                **meta))
    return items, drops


# --------------------------------------------------------------------------- instruction following

FORBID_BANK = ("basically", "literally", "obviously", "essentially", "honestly", "definitely", "furthermore",
               "moreover", "undoubtedly", "remarkably")
START_PHRASES = ("Here is my answer:", "Good question.", "To begin:")
END_PHRASES = ("That is all.", "Any other questions?", "Hope this helps!")

# Types emitted, with their parameter choices. Not emitted: "language" (no eval checker) and "words" (the
# two checkers count words differently, so an exact count can hold in one and fail in the other).
TYPES = ("min_words", "max_words", "sentences", "paragraphs", "bullets", "sections", "include_keywords",
         "keyword_frequency", "forbid_keywords", "json_format", "end_with", "start_with", "all_lowercase",
         "all_uppercase", "no_commas", "title_in_double_angular", "placeholders", "quotation", "postscript",
         "highlighted_sections")

_STRUCTURE = {"paragraphs", "bullets", "sections", "json_format"}
_CASE = {"all_lowercase", "all_uppercase"}
_PAIRS = [
    ("min_words", "max_words"), ("all_lowercase", "all_uppercase"),
    ("all_lowercase", "start_with"), ("all_lowercase", "end_with"), ("all_lowercase", "postscript"),
    ("all_lowercase", "sections"), ("all_uppercase", "start_with"), ("all_uppercase", "end_with"),
    ("all_uppercase", "sections"),
    ("quotation", "bullets"), ("quotation", "sections"), ("quotation", "start_with"), ("quotation", "end_with"),
    ("quotation", "title_in_double_angular"), ("quotation", "postscript"),
    ("title_in_double_angular", "paragraphs"), ("title_in_double_angular", "start_with"),
    ("postscript", "paragraphs"), ("postscript", "end_with"),
    ("sentences", "max_words"), ("max_words", "paragraphs"), ("max_words", "sections"),
]
CONFLICTS: set[frozenset[str]] = {frozenset(p) for p in _PAIRS}
CONFLICTS |= {frozenset({a, b}) for a in _STRUCTURE for b in _STRUCTURE if a != b}
CONFLICTS |= {frozenset({"json_format", t}) for t in TYPES if t not in ("json_format", "include_keywords")}

# Prompts that already ask for a count or a format ("name five birds", "in one sentence", "list ...") only get
# constraints that cannot contradict them.
_COUNT_OR_FORMAT = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|dozen|single|list|bullets?|paragraphs?|"
    r"sentences?|words?|json|lowercase|uppercase|capital|title|commas?|quot\w*|sections?|short|brief\w*|"
    r"summar\w*|yes or no|haiku|poem|limerick|tweet|essay|letter|email)\b", re.I)
SAFE_FOR_FORMATTED = ("include_keywords", "keyword_frequency", "forbid_keywords", "end_with", "postscript",
                      "highlighted_sections", "no_commas", "placeholders")

RELEVANCE_MIN_SHARE = 0.35      # base-prompt content words / whole-prompt content words (gate needs ≥ 0.15)
N_CONSTRAINTS = ((1, 0.35), (2, 0.40), (3, 0.25))


def _content_words(text: str) -> set[str]:
    # The training grader's own definition, so the relevance check here is the one the reward applies.
    from dpt.rewards.instruction import content_words

    return content_words(text)


# Generic or counting words make poor required keywords ("Use the words "three" and "things"").
_KEYWORD_SKIP = frozenset(
    "three four five seven eight nine dozen hundred thousand thing things something anything good best better "
    "great like know want need list name describe explain tell give example examples ideas idea ways kind kinds "
    "type types many much other others people person someone".split())


def keyword_candidates(prompt: str) -> list[str]:
    """Words of the prompt usable as required keywords: alphabetic, ≥ 4 letters, specific content words."""
    words = re.findall(r"[A-Za-z]{4,}", prompt)
    cw = _content_words(prompt)
    seen, out = set(), []
    for w in words:
        lw = w.lower()
        if lw in cw and lw not in seen and lw not in FORBID_BANK and lw not in _KEYWORD_SKIP:
            seen.add(lw)
            out.append(lw)
    return out


def _quoted(words: Sequence[str], joiner: str) -> str:
    q = [f'"{w}"' for w in words]
    return q[0] if len(q) == 1 else ", ".join(q[:-1]) + f" {joiner} " + q[-1]


def make_constraint(t: str, rng: Any, prompt: str, chosen: list[dict[str, Any]]) -> dict[str, Any] | None:
    """One canonical constraint of type ``t`` for ``prompt``, or None if the prompt cannot support it."""
    used = {c["keyword"] for c in chosen if c["type"] == "keyword_frequency"}
    used |= {w for c in chosen if c["type"] == "include_keywords" for w in c["value"]}
    if t == "min_words":
        return {"type": t, "value": rng.choice([50, 100, 150, 200, 300])}
    if t == "max_words":
        return {"type": t, "value": rng.choice([50, 80, 100, 150])}
    if t == "sentences":
        return {"type": t, "value": rng.choice([3, 5, 8]), "relation": "at_least"}
    if t == "paragraphs":
        return {"type": t, "value": rng.choice([2, 3, 4]), "relation": "exactly", "separator": "\n\n"}
    if t == "bullets":
        return {"type": t, "value": rng.choice([3, 4, 5]), "relation": "exactly"}
    if t == "sections":
        return {"type": t, "value": rng.choice([2, 3, 4]), "relation": "exactly", "marker": "Section"}
    if t in ("include_keywords", "keyword_frequency"):
        kws = [w for w in keyword_candidates(prompt) if w not in used]
        if not kws:
            return None
        if t == "include_keywords":
            return {"type": t, "value": rng.sample(kws, min(len(kws), rng.choice([1, 2])))}
        return {"type": t, "keyword": rng.choice(kws), "value": rng.choice([2, 3]), "relation": "at_least"}
    if t == "forbid_keywords":
        low = prompt.lower()
        bank = [w for w in FORBID_BANK if w not in low]
        return {"type": t, "value": rng.sample(bank, 2)}
    if t == "end_with":
        return {"type": t, "value": rng.choice(END_PHRASES)}
    if t == "start_with":
        return {"type": t, "value": rng.choice(START_PHRASES)}
    if t in ("placeholders", "highlighted_sections"):
        return {"type": t, "value": rng.choice([1, 2, 3])}
    if t == "postscript":
        return {"type": t, "marker": "P.S."}
    return {"type": t}                                  # json_format, all_*case, no_commas, title, quotation


def _n(k: int, noun: str) -> str:
    return f"{k} {noun}" if k == 1 else f"{k} {noun}s"


def render_constraint(c: dict[str, Any]) -> str:
    """The instruction sentence. Worded with few content words: they count against the relevance gate."""
    t, v = c["type"], c.get("value")
    return {
        "min_words": lambda: f"Answer with at least {v} words.",
        "max_words": lambda: f"Answer with at most {v} words.",
        "sentences": lambda: f"Write at least {v} sentences.",
        "paragraphs": lambda: f"Write exactly {v} paragraphs, separated by blank lines.",
        "bullets": lambda: f"Use exactly {v} markdown bullet points, such as:\n* This is a point.",
        "sections": lambda: (f"Split your response into {v} sections, each beginning with {c['marker']} X, "
                             f"such as {c['marker']} 1."),
        "include_keywords": lambda: f"Use the words {_quoted(v, 'and')}.",
        "keyword_frequency": lambda: f"Use the word \"{c['keyword']}\" at least {v} times.",
        "forbid_keywords": lambda: f"Do not use the words {_quoted(v, 'or')}.",
        "json_format": lambda: "Answer in JSON format only.",
        "end_with": lambda: f"End your response with exactly these words: \"{v}\"",
        "start_with": lambda: f"Start your response with exactly these words: \"{v}\"",
        "all_lowercase": lambda: "Write in English using only lowercase letters, with no capitals.",
        "all_uppercase": lambda: "Write in English using only capital letters.",
        "no_commas": lambda: "Do not use any commas.",
        "title_in_double_angular": lambda: "Give your answer a title in double angular brackets, such as <<my title>>.",
        "placeholders": lambda: f"Use at least {_n(v, 'placeholder')} in square brackets, such as [name].",
        "quotation": lambda: "Put your whole response in double quotes.",
        "postscript": lambda: f"End with a postscript that starts with {c['marker']}",
        "highlighted_sections": lambda: (f"Highlight at least {_n(v, 'section')} with markdown, "
                                         "i.e. *highlighted section*."),
    }[t]()


def compatible(t: str, chosen_types: Sequence[str]) -> bool:
    return t not in chosen_types and all(frozenset({t, c}) not in CONFLICTS for c in chosen_types)


def draw_constraints(prompt: str, rng: Any, k: int) -> list[dict[str, Any]]:
    """Up to ``k`` mutually compatible constraints suited to ``prompt``."""
    allowed = SAFE_FOR_FORMATTED if _COUNT_OR_FORMAT.search(prompt) else TYPES
    chosen: list[dict[str, Any]] = []
    pool = list(allowed)
    rng.shuffle(pool)
    for t in pool:
        if len(chosen) == k:
            break
        if not compatible(t, [c["type"] for c in chosen]):
            continue
        c = make_constraint(t, rng, prompt, chosen)
        if c is not None:
            chosen.append(c)
    return chosen


def if_question(prompt: str, constraints: Sequence[dict[str, Any]]) -> str:
    return prompt.strip() + "\n\n" + "\n".join(render_constraint(c) for c in constraints)


def relevance_share(prompt: str, question: str) -> float:
    """Share of the whole prompt's content words that come from the base prompt (1.0 if the gate is off)."""
    full = _content_words(question)
    if len(full) < 4:
        return 1.0
    return len(_content_words(prompt) & full) / len(full)


def base_prompt_ok(r: dict[str, Any]) -> str | None:
    """Why a Dolly row is not a usable base prompt, or None."""
    if r.get("category") not in DOLLY_CATEGORIES:
        return "category"
    if str(r.get("context") or "").strip():
        return "has_context"
    p = str(r.get("instruction") or "").strip()
    if not 15 <= len(p) <= 400:
        return "length"
    if "http" in p.lower() or sum(ord(ch) < 128 for ch in p) < 0.97 * len(p):
        return "url_or_non_ascii"
    return None


def if_items(raw_dir: Path, n: int, seed: int, revision: dict[str, str]
             ) -> tuple[list[dict[str, Any]], Counter, Counter]:
    """Up to ``n`` instruction-following candidates, plus drop counts and constraint-type counts."""
    path = raw_dir / "dolly.jsonl"
    drops: Counter = Counter()
    types: Counter = Counter()
    if not path.exists():
        log.warning("no %s; no instruction-following items", path)
        return [], drops, types
    eligible, seen = [], set()
    for r in sorted(read_jsonl(path), key=lambda r: str(r["id"])):
        why = base_prompt_ok(r)
        if why:
            drops[f"dolly:{why}"] += 1
            continue
        key = normalize_text(r["instruction"])
        if key in seen:
            drops["dolly:duplicate"] += 1
            continue
        seen.add(key)
        eligible.append(r)
    order = rng_for(seed, "t4", "if-order")
    order.shuffle(eligible)
    weights = [w for _, w in N_CONSTRAINTS]
    items: list[dict[str, Any]] = []
    for r in eligible:
        if len(items) == n:
            break
        prompt = r["instruction"].strip()
        rng = rng_for(seed, "t4", "if", r["id"])
        k = rng.choices([k for k, _ in N_CONSTRAINTS], weights=weights)[0]
        found = None
        for kk in range(k, 0, -1):          # fewer constraints if the relevance gate would block the item
            for _ in range(4):
                cs = draw_constraints(prompt, rng, kk)
                q = if_question(prompt, cs)
                exempt = any(c["type"] == "json_format" for c in cs)
                if cs and (exempt or relevance_share(prompt, q) >= RELEVANCE_MIN_SHARE):
                    found = (cs, q)
                    break
            if found:
                break
        if found is None:
            drops["dolly:relevance_gate_risk"] += 1
            continue
        cs, q = found
        types.update(c["type"] for c in cs)
        items.append(candidate(
            T4_GENERAL, q, {"kind": "ifeval", "constraints": cs}, origin=ORIGIN,
            source={"license": SOURCES["dolly"]["license"], "dataset": SOURCES["dolly"]["repo"], "id": r["id"],
                    "split": "train", "revision": revision.get("dolly")},
            t4_source="dolly", base_prompt=prompt, dolly_category=r["category"]))
    if len(items) < n:
        log.warning("instruction following: %d of %d requested (eligible base prompts: %d)",
                    len(items), n, len(eligible))
    return items, drops, types


# --------------------------------------------------------------------------- build

def build(raw_dir: str | Path, out: str | Path, *, n_math: int = 7500, n_if: int = 7500, math_share: float = 0.6,
          seed: int = 1) -> dict[str, Any]:
    """Write the T4 candidates to ``out`` and ``<out>.summary.json``; returns the summary."""
    raw_dir, out = Path(raw_dir), Path(out)
    manifest_path = raw_dir / "fetch_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    revision = {k: v.get("revision") for k, v in manifest.items()}

    math_all, math_drops = math_items(raw_dir, revision)
    by_src = {s: [c for c in math_all if c["meta"]["t4_source"] == s] for s in ("math", "gsm8k")}
    for s, rows in by_src.items():
        rng_for(seed, "t4", "math-order", s).shuffle(rows)
    want_math = min(len(by_src["math"]), round(n_math * math_share))
    want_gsm = min(len(by_src["gsm8k"]), n_math - want_math)
    want_math = min(len(by_src["math"]), n_math - want_gsm)       # fill from MATH if GSM8K runs short
    math_sel = by_src["math"][:want_math] + by_src["gsm8k"][:want_gsm]

    if_sel, if_drops, types = if_items(raw_dir, n_if, seed, revision)
    rows = sorted(math_sel + if_sel, key=lambda c: c["uid"])
    n = write_candidates(out, rows)
    summary = {
        "out": str(out), "seed": seed, "total": n,
        "counts": dict(Counter(c["meta"]["t4_source"] for c in rows)),
        "available": {"math": len(by_src["math"]), "gsm8k": len(by_src["gsm8k"])},
        "drops": dict(sorted((math_drops + if_drops).items())),
        "constraint_types": dict(sorted(types.items())),
        "constraints_per_item": dict(sorted(Counter(len(c["answer"]["constraints"]) for c in if_sel).items())),
        "revisions": revision,
    }
    write_json(Path(f"{out}.summary.json"), summary)
    log.info("t4: %d candidates → %s (%s)", n, out, summary["counts"])
    return summary


# --------------------------------------------------------------------------- fetch

def _rows_from_file(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return list(read_jsonl(path))
    try:
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    except ImportError:
        import pandas as pd                                   # raises if neither is installed

        return pd.read_parquet(path).to_dict("records")


def _normalize(name: str, filename: str, rows: list[dict[str, Any]], offset: int) -> list[dict[str, Any]]:
    out = []
    for i, r in enumerate(rows):
        if name == "gsm8k":
            out.append({"id": f"gsm8k/train/{offset + i}", "question": r["question"], "answer": r["answer"]})
        elif name == "math":
            subject = filename.split("/")[0]
            out.append({"id": f"math/{subject}/{Path(filename).stem}/{i}", "subject": subject, "problem": r["problem"],
                        "solution": r["solution"], "level": r.get("level"), "type": r.get("type")})
        else:
            out.append({"id": f"dolly/{offset + i}", "instruction": r["instruction"],
                        "context": r.get("context") or "", "response": r.get("response") or "",
                        "category": r.get("category")})
    return out


def fetch(raw_dir: str | Path, *, sources: Sequence[str] = tuple(SOURCES),
          api: Any = None, download: Callable[..., str] | None = None) -> dict[str, Any]:
    """Download the train splits into ``raw_dir`` as normalized JSONL; returns the fetch manifest."""
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")          # the cluster's login nodes: 8 GB ulimit -v
    from huggingface_hub import HfApi, hf_hub_download

    api = api or HfApi()
    download = download or hf_hub_download
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {}
    for name in sources:
        spec = SOURCES[name]
        info = api.dataset_info(spec["repo"])
        files = sorted(f for f in api.list_repo_files(spec["repo"], repo_type="dataset", revision=info.sha)
                       if re.match(spec["files"], f))
        if not files:
            raise RuntimeError(f"{spec['repo']}@{info.sha}: no files match {spec['files']}; layout changed?")
        rows: list[dict[str, Any]] = []
        for f in files:
            local = Path(download(spec["repo"], f, repo_type="dataset", revision=info.sha))
            rows += _normalize(name, f, _rows_from_file(local), len(rows))
        write_jsonl(raw_dir / f"{name}.jsonl", rows)
        manifest[name] = {"repo": spec["repo"], "revision": info.sha, "files": files, "license": spec["license"],
                          "rows": len(rows)}
        log.info("%s: %d rows from %s@%s", name, len(rows), spec["repo"], info.sha[:12])
    write_json(raw_dir / "fetch_manifest.json", manifest)
    return manifest


# --------------------------------------------------------------------------- CLI

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="T4 general replay: public math + instruction following")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download the train splits (login node)")
    f.add_argument("--raw-dir", required=True)
    f.add_argument("--sources", default=",".join(SOURCES))
    b = sub.add_parser("build", help="write the T4 candidates")
    b.add_argument("--raw-dir", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--n-math", type=int, default=7500)
    b.add_argument("--n-if", type=int, default=7500)
    b.add_argument("--math-share", type=float, default=0.6, help="share of the math half taken from MATH")
    b.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    if args.cmd == "fetch":
        fetch(args.raw_dir, sources=[s for s in args.sources.split(",") if s])
    else:
        build(args.raw_dir, args.out, n_math=args.n_math, n_if=args.n_if, math_share=args.math_share,
              seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
