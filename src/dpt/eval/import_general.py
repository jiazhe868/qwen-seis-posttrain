"""Import the general benchmarks and the safety regression set (eval_suite.yaml ``general:``, ``refusal:``).

    python -m dpt.eval.import_general fetch --raw-dir $WORK/dpt/raw_general            # login node, network
    python -m dpt.eval.import_general build --raw-dir $WORK/dpt/raw_general \\
        --out-root $SECURE --decontam-out $WORK/dpt/public_benchmarks

``fetch`` downloads each source at a pinned revision (Hugging Face, plain HTTP: the cluster's login nodes need
``HF_HUB_DISABLE_XET=1``) and writes one normalized JSONL per source plus ``fetch_manifest.json`` (repo,
revision, license, file hashes). Parquet needs pyarrow (the ``dpt-train`` env). ``build`` reads only the
normalized files, so it is reproducible offline.

=====================  =====================================  ============  ======================================
set (grader)           source                                 license       what is kept
=====================  =====================================  ============  ======================================
gpqa_diamond (choice)  Idavidrein/gpqa, gpqa_diamond.csv      CC-BY-4.0     all 198; options shuffled per item id
                       (gated: ``hf auth login`` + accept)
mmlu_pro_2k (choice)   TIGER-Lab/MMLU-Pro, test               MIT           2,000, stratified by category
ifeval (ifeval)        google/IFEval                          Apache-2.0    prompts whose every instruction the
                                                                            eval checker implements; others dropped
livecodebench_post_*   livecodebench/code_generation_lite     "cc" (card)   LeetCode-style (functional) problems
(code)                                                                      released on/after the cutoff; stdin
                                                                            problems dropped (see below)
tool_calling           gorilla-llm/Berkeley-Function-         Apache-2.0    BFCL v3 simple + multiple items whose
(toolcall)             Calling-Leaderboard                                  ground truth has one value per argument
refusal (refusal)      JailbreakBench/JBB-Behaviors           MIT           100 harmful (refuse) + 100 benign
                       Paul/XSTest                            CC-BY-4.0     200 unsafe (refuse) + 250 safe controls
=====================  =====================================  ============  ======================================

Grader constraints that shape the import, so that no item can be scored wrong for a reason unrelated to the
model:

* No task prompt is added for general sets (``dpt.eval.harness.build_messages``), so every question carries
  its own options and answer-format instruction.
* ``choice`` reads one letter (``\\boxed{C}``, ``Answer: C``).
* ``ifeval`` fails closed on unknown constraint types, so prompts with an instruction it does not implement
  (letter frequency, response language, repeat-the-prompt, two responses, …) are dropped and counted, never
  kept with a constraint silently missing.
* ``code`` on the cluster runs hidden tests through the sandbox server, whose harness imports the solution
  before the tests. A stdin program would read its input at import time, so only function-call (LeetCode)
  problems are kept. Tests are written both as a hidden-test file (``<out-root>/hidden_tests/<id>.py``, the
  directory ``scripts/slurm/eval.slurm`` serves) and inline as ``tests_source`` (the dev-only local runner).
  LiveCodeBench's private tests are pickled; they are decoded with an unpickler that refuses every class, so
  a pickle that is anything but plain data fails instead of running code.
* ``toolcall`` compares the call list exactly, so only BFCL items whose ground truth allows exactly one value
  per argument (no optional "may omit" alternatives) and one call are kept. The harness passes no ``tools``
  to the chat template, so the function schemas and the expected call format are in the question.

``build`` also writes ``<decontam-out>/<name>.jsonl`` ({uid, question}) for every set, plus the GSM8K and MATH
test splits and MATH-500, for ``data_pipeline.inputs.public_benchmarks`` (T4 trains on their train splits).
Existing eval files are never overwritten without ``--force`` (they may be frozen).
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import pickle
import random
import re
import string
import zlib
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from dpt.common.io import read_json, read_jsonl, write_json, write_jsonl
from dpt.common.logging import get_logger

log = get_logger(__name__)

LETTERS = string.ascii_uppercase


# =============================================================================================
# Sources (pinned revisions, checked 2026-10-21)
# =============================================================================================
@dataclass(frozen=True)
class Source:
    repo: str
    revision: str
    license: str
    files: tuple[str, ...] = ()
    gated: bool = False
    note: str = ""


SOURCES: dict[str, Source] = {
    "gpqa_diamond": Source("Idavidrein/gpqa", "83022cefff930aea54f654c0b282e74b9eeda5c6", "CC-BY-4.0",
                           ("gpqa_diamond.csv",), gated=True),
    "mmlu_pro_test": Source("TIGER-Lab/MMLU-Pro", "b189ec765aa7ed75c8acfea42df31fdae71f97be", "MIT",
                            ("data/test-00000-of-00001.parquet",)),
    "ifeval": Source("google/IFEval", "966cd89545d6b6acfd7638bc708b98261ca58e84", "Apache-2.0",
                     ("ifeval_input_data.jsonl",)),
    "livecodebench": Source("livecodebench/code_generation_lite", "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
                            "cc (unspecified variant on the dataset card)",
                            note="test*.jsonl read newest first; release_v6 (test6.jsonl) ends in April 2025"),
    "bfcl": Source("gorilla-llm/Berkeley-Function-Calling-Leaderboard", "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda",
                   "Apache-2.0", ("BFCL_v3_simple.json", "possible_answer/BFCL_v3_simple.json",
                                  "BFCL_v3_multiple.json", "possible_answer/BFCL_v3_multiple.json")),
    "jbb": Source("JailbreakBench/JBB-Behaviors", "886acc352a31533ffbcf4ef22c744658688086fc", "MIT",
                  ("data/harmful-behaviors.csv", "data/benign-behaviors.csv")),
    "xstest": Source("Paul/XSTest", "f600c994b256f12867dfa5b3eb3d545a3e62f8b5", "CC-BY-4.0", ("xstest_prompts.csv",)),
    "gsm8k_test": Source("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "MIT",
                         ("main/test-00000-of-00001.parquet",)),
    "math_test": Source("EleutherAI/hendrycks_math", "21a5633873b6a120296cce3e2df9d5550074f4a3", "MIT",
                        tuple(f"{s}/test-00000-of-00001.parquet" for s in (
                            "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                            "number_theory", "prealgebra", "precalculus"))),
    "math500": Source("HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
                      "MIT (derived from MATH; no license on the card)", ("test.jsonl",)),
}

# eval set → the sources it needs
SETS: dict[str, tuple[str, ...]] = {
    "gpqa_diamond": ("gpqa_diamond",),
    "mmlu_pro_2k": ("mmlu_pro_test",),
    "ifeval": ("ifeval",),
    "livecodebench": ("livecodebench",),
    "tool_calling": ("bfcl",),
    "refusal": ("jbb", "xstest"),
    "decontam_math": ("gsm8k_test", "math_test", "math500"),
}

DEFAULT_LCB_CUTOFF = "2026-08-01"


# =============================================================================================
# Small helpers
# =============================================================================================
def _rank(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _DataOnlyUnpickler(pickle.Unpickler):
    """Plain data only: refuses every global, so a crafted pickle cannot import or call anything."""

    def find_class(self, module: str, name: str):  # noqa: D401 - pickle API
        raise pickle.UnpicklingError(f"refusing to load {module}.{name} from a dataset pickle")


def decode_lcb_tests(blob: str) -> list[dict[str, Any]]:
    """LiveCodeBench private tests: base64 → zlib → pickle(str of JSON) → list of {input, output, testtype}."""
    raw = _DataOnlyUnpickler(io.BytesIO(zlib.decompress(base64.b64decode(blob)))).load()
    if not isinstance(raw, str):
        raise ValueError(f"expected a JSON string inside the pickle, got {type(raw).__name__}")
    return list(json.loads(raw))


def choice_question(question: str, options: Sequence[str]) -> str:
    opts = "\n".join(f"{LETTERS[i]}) {str(o).strip()}" for i, o in enumerate(options))
    last = LETTERS[len(options) - 1]
    return (f"{question.strip()}\n\n{opts}\n\nAnswer with the letter of the correct option (A–{last}) in the form "
            "\\boxed{X}.")


# =============================================================================================
# Builders: normalized source rows → eval records
# =============================================================================================
@dataclass
class Built:
    records: list[dict[str, Any]]
    drops: Counter = field(default_factory=Counter)
    hidden_tests: dict[str, str] = field(default_factory=dict)   # id → test source (code sets)
    sources: tuple[str, ...] = ()


def build_gpqa(rows: Iterable[dict[str, Any]], seed: int = 1) -> Built:
    out, drops = [], Counter()
    for r in rows:
        rid = str(r.get("Record ID") or "").strip()
        q, right = str(r.get("Question") or "").strip(), str(r.get("Correct Answer") or "").strip()
        wrong = [str(r.get(f"Incorrect Answer {i}") or "").strip() for i in (1, 2, 3)]
        if not rid or not q or not right or not all(wrong):
            drops["missing_field"] += 1
            continue
        opts = [right, *wrong]
        random.Random(int(_rank(seed, "gpqa", rid)[:16], 16)).shuffle(opts)
        letter = LETTERS[opts.index(right)]
        out.append({"id": f"gpqa-{rid}", "question": choice_question(q, opts),
                    "answer": {"letter": letter, "n_options": 4},
                    "meta": {"domain": r.get("High-level domain", ""), "subdomain": r.get("Subdomain", "")}})
    return Built(out, drops, sources=("gpqa_diamond",))


def build_mmlu_pro(rows: Iterable[dict[str, Any]], n: int = 2000, seed: int = 1) -> Built:
    """Deterministic subset stratified by category (largest-remainder allocation, hash order within)."""
    drops = Counter()
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        opts = list(r.get("options") or [])
        idx = r.get("answer_index")
        if not opts or idx is None or not 0 <= int(idx) < len(opts) or len(opts) > 10:
            drops["bad_options"] += 1
            continue
        if r.get("answer") and str(r["answer"]).strip().upper() != LETTERS[int(idx)]:
            drops["answer_mismatch"] += 1
            continue
        by_cat.setdefault(str(r.get("category") or ""), []).append(r)
    total = sum(len(v) for v in by_cat.values())
    n = min(n, total)
    quota = {c: n * len(v) / total for c, v in by_cat.items()}
    alloc = {c: int(q) for c, q in quota.items()}
    for c in sorted(by_cat, key=lambda c: (-(quota[c] - alloc[c]), c))[: n - sum(alloc.values())]:
        alloc[c] += 1
    out = []
    for c in sorted(by_cat):
        for r in sorted(by_cat[c], key=lambda r: _rank(seed, "mmlu_pro", r["question_id"]))[: alloc[c]]:
            opts = list(r["options"])
            out.append({"id": f"mmlu_pro-{r['question_id']}", "question": choice_question(r["question"], opts),
                        "answer": {"letter": LETTERS[int(r["answer_index"])], "n_options": len(opts)},
                        "meta": {"category": c, "src": r.get("src", "")}})
    drops["not_sampled"] = total - len(out)
    return Built(out, drops, sources=("mmlu_pro_test",))


_IF_REL = {"at least": "at_least", "less than": "less_than", "at most": "at_most", "exactly": "exactly"}


def _rel(v: Any) -> str:
    r = _IF_REL.get(str(v).strip().lower())
    if r is None:
        raise ValueError(f"unknown relation {v!r}")
    return r


# IFEval instruction id → canonical constraint (dpt.common.ifeval_schema). An id missing here, or one whose
# canonical type dpt.eval.graders.ifeval does not implement, drops the prompt.
IFEVAL_MAP: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "punctuation:no_comma": lambda k: {"type": "no_commas"},
    "change_case:english_lowercase": lambda k: {"type": "all_lowercase"},
    "change_case:english_capital": lambda k: {"type": "all_uppercase"},
    "detectable_format:title": lambda k: {"type": "title_in_double_angular"},
    "detectable_format:json_format": lambda k: {"type": "json_format"},
    "startend:quotation": lambda k: {"type": "quotation"},
    "startend:end_checker": lambda k: {"type": "end_with", "value": k["end_phrase"]},
    "detectable_format:number_highlighted_sections":
        lambda k: {"type": "highlighted_sections", "value": int(k["num_highlights"])},
    "detectable_format:number_bullet_lists":
        lambda k: {"type": "bullets", "value": int(k["num_bullets"]), "relation": "exactly"},
    "detectable_format:multiple_sections":
        lambda k: {"type": "sections", "value": int(k["num_sections"]), "relation": "at_least",
                   "marker": str(k["section_spliter"])},
    "detectable_content:number_placeholders":
        lambda k: {"type": "placeholders", "value": int(k["num_placeholders"])},
    "detectable_content:postscript": lambda k: {"type": "postscript", "marker": str(k["postscript_marker"])},
    "keywords:existence": lambda k: {"type": "include_keywords", "value": list(k["keywords"])},
    "keywords:forbidden_words": lambda k: {"type": "forbid_keywords", "value": list(k["forbidden_words"])},
    "keywords:frequency":
        lambda k: {"type": "keyword_frequency", "keyword": str(k["keyword"]), "value": int(k["frequency"]),
                   "relation": _rel(k["relation"])},
    "length_constraints:number_words":
        lambda k: {"type": "words", "value": int(k["num_words"]), "relation": _rel(k["relation"])},
    "length_constraints:number_sentences":
        lambda k: {"type": "sentences", "value": int(k["num_sentences"]), "relation": _rel(k["relation"])},
    "length_constraints:number_paragraphs":
        lambda k: {"type": "paragraphs", "value": int(k["num_paragraphs"]), "relation": "exactly"},
}


def build_ifeval(rows: Iterable[dict[str, Any]]) -> Built:
    from dpt.eval.graders.ifeval import CHECKS

    out, drops = [], Counter()
    for r in rows:
        cons, bad = [], None
        for iid, kw in zip(r["instruction_id_list"], r["kwargs"]):
            fn = IFEVAL_MAP.get(iid)
            if fn is None:
                bad = f"unsupported:{iid}"
                break
            try:
                c = fn({k: v for k, v in (kw or {}).items() if v is not None})
            except (KeyError, TypeError, ValueError):
                bad = f"bad_kwargs:{iid}"
                break
            if c["type"] not in CHECKS:
                bad = f"unsupported:{iid}"
                break
            cons.append(c)
        if bad or not cons:
            drops[bad or "no_instructions"] += 1
            continue
        out.append({"id": f"ifeval-{r['key']}", "question": str(r["prompt"]),
                    "answer": {"constraints": cons},
                    "meta": {"instruction_id_list": list(r["instruction_id_list"])}})
    return Built(out, drops, sources=("ifeval",))


_LCB_TESTS_HEADER = '''import json
import math

FUNC = {func!r}


def _eq(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    return a == b


def _check(args_text, expected_text):
    from solution import Solution

    args = [json.loads(line) for line in args_text.split("\\n") if line.strip()]
    assert _eq(getattr(Solution(), FUNC)(*args), json.loads(expected_text))
'''

LCB_PROMPT = """You will be given a programming problem. Write a correct, efficient Python solution that passes
all tests.

### {title}

{content}

Use the following starter code. Keep the class and method names, include every import your code needs (for
example `from typing import List`), and put the complete solution in one ```python block.

```python
{starter}
```"""


def lcb_tests_source(func: str, cases: Sequence[dict[str, Any]]) -> str:
    body = [_LCB_TESTS_HEADER.format(func=func)]
    for i, c in enumerate(cases):
        body.append(f"\n\ndef test_{i:03d}():\n    _check({c['input']!r}, {c['output']!r})\n")
    return "".join(body)


def build_livecodebench(rows: Iterable[dict[str, Any]], *, cutoff: str = DEFAULT_LCB_CUTOFF, max_cases: int = 50,
                        max_case_chars: int = 100_000) -> Built:
    out, drops, hidden = [], Counter(), {}
    for r in rows:
        if str(r.get("contest_date", ""))[:10] < cutoff:
            drops["before_cutoff"] += 1
            continue
        meta = r.get("metadata") or {}
        func = (json.loads(meta) if isinstance(meta, str) else dict(meta)).get("func_name")
        cases = list(r.get("public_tests") or []) + list(r.get("private_tests") or [])
        if not func or any(c.get("testtype") != "functional" for c in cases):
            drops["stdin_unsupported"] += 1
            continue
        cases = [c for c in cases if len(c["input"]) + len(c["output"]) <= max_case_chars][:max_cases]
        if not cases:
            drops["no_usable_tests"] += 1
            continue
        iid = f"lcb-{r['question_id']}"
        tests = lcb_tests_source(func, cases)
        hidden[iid] = tests
        prompt = LCB_PROMPT.format(title=r["question_title"], content=r["question_content"],
                                   starter=str(r.get("starter_code") or "").rstrip())
        out.append({"id": iid, "question": prompt,
                    "answer": {"hidden_tests_id": iid, "entry_point": func, "language": "python",
                               "tests_source": tests, "timeout_s": 60.0},
                    "meta": {"platform": r.get("platform"), "contest_date": r.get("contest_date"),
                             "difficulty": r.get("difficulty"), "n_tests": len(cases)}})
    return Built(out, drops, hidden, sources=("livecodebench",))


TOOL_PROMPT = """You can call the following functions. Their parameters are described as JSON schema:

{functions}

To call a function, reply with one block per call in exactly this format:
<tool_call>
<function=FUNCTION_NAME>
<parameter=PARAMETER_NAME>
VALUE
</parameter>
</function>
</tool_call>
Write each VALUE as JSON (numbers, true/false, lists and objects); a string may be written as plain text.

User request: {request}"""


def build_tool_calling(rows: Iterable[dict[str, Any]]) -> Built:
    out, drops = [], Counter()
    for r in rows:
        gt = r["ground_truth"]
        turns = [m for turn in r["question"] for m in turn]
        users = [m for m in turns if m.get("role") == "user"]
        if len(gt) != 1 or len(users) != 1 or len(turns) != 1:
            drops["not_single_call"] += 1
            continue
        (name, args), = gt[0].items()
        if not all(isinstance(v, list) and len(v) == 1 and v[0] != "" for v in args.values()):
            drops["several_acceptable_values"] += 1
            continue
        calls = [{"name": name, "arguments": {k: v[0] for k, v in args.items()}}]
        out.append({"id": f"bfcl-{r['id']}", "question": TOOL_PROMPT.format(
            functions=json.dumps(r["function"], indent=2, ensure_ascii=False), request=users[0]["content"]),
            "answer": {"calls": calls}, "meta": {"category": r.get("category", "")}})
    return Built(out, drops, sources=("bfcl",))


def build_refusal(jbb: Iterable[dict[str, Any]], xstest: Iterable[dict[str, Any]]) -> Built:
    out, drops = [], Counter()
    for r in jbb:
        harmful = bool(r["harmful"])
        out.append({"id": f"jbb-{'h' if harmful else 'b'}-{r['Index']}", "question": str(r["Goal"]).strip(),
                    "answer": {"should_refuse": harmful},
                    "meta": {"source": "jbb", "category": r.get("Category", ""), "behavior": r.get("Behavior", "")}})
    for r in xstest:
        label = str(r.get("label", "")).strip().lower()
        if label not in ("safe", "unsafe"):
            drops["unknown_label"] += 1
            continue
        out.append({"id": f"xstest-{r['id']}", "question": str(r["prompt"]).strip(),
                    "answer": {"should_refuse": label == "unsafe"},
                    "meta": {"source": "xstest", "type": r.get("type", ""), "focus": r.get("focus", "")}})
    return Built(out, drops, sources=("jbb", "xstest"))


def decontam_math(gsm8k: Iterable[dict], math_test: Iterable[dict], math500: Iterable[dict]) -> dict[str, list[dict]]:
    return {
        "gsm8k_test": [{"uid": f"gsm8k-test-{i}", "question": r["question"]} for i, r in enumerate(gsm8k)],
        "math_test": [{"uid": f"math-test-{r.get('subject', '')}-{i}", "question": r["problem"]}
                      for i, r in enumerate(math_test)],
        "math500": [{"uid": f"math500-{r.get('unique_id', i)}", "question": r["problem"]}
                    for i, r in enumerate(math500)],
    }


# =============================================================================================
# fetch
# =============================================================================================
class GatedError(RuntimeError):
    pass


def _download(src: Source, filename: str, revision: str | None = None) -> Path:
    from huggingface_hub import hf_hub_download

    try:
        return Path(hf_hub_download(src.repo, filename, repo_type="dataset", revision=revision or src.revision))
    except Exception as e:
        text = f"{type(e).__name__}: {e}"
        if src.gated or "Gated" in type(e).__name__ or re.search(r"\b(401|403)\b|gated", text, re.I):
            raise GatedError(
                f"{src.repo} is gated or not accessible: accept its terms at https://huggingface.co/datasets/"
                f"{src.repo} and log in (`hf auth login`) on this machine, then rerun fetch ({text[:200]})") from e
        raise


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    except ImportError:
        pass
    try:
        import pandas as pd

        return pd.read_parquet(path).to_dict("records")
    except ImportError as e:  # pragma: no cover - depends on the env
        raise RuntimeError("reading parquet needs pyarrow (activate the dpt-train env)") from e


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _plain(v: Any) -> Any:
    """numpy scalars/arrays from parquet → plain JSON types."""
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def _lcb_files(src: Source, revision: str) -> list[str]:
    from huggingface_hub import HfApi

    names = [f for f in HfApi().list_repo_files(src.repo, repo_type="dataset", revision=revision)
             if re.fullmatch(r"test\d*\.jsonl", f)]
    return sorted(names, key=lambda f: int(re.sub(r"\D", "", f) or 1), reverse=True)


def normalize_lcb_row(r: dict[str, Any]) -> dict[str, Any]:
    pub = json.loads(r.get("public_test_cases") or "[]")
    priv = decode_lcb_tests(r["private_test_cases"]) if r.get("private_test_cases") else []
    keep = ("question_title", "question_content", "platform", "question_id", "contest_id", "contest_date",
            "starter_code", "difficulty", "metadata")
    return {**{k: r.get(k) for k in keep}, "public_tests": pub, "private_tests": priv}


def fetch_livecodebench(src: Source, cutoff: str, revision: str | None = None) -> tuple[list[dict], dict[str, str]]:
    """Rows on/after ``cutoff``, reading the release files newest first; stops at a file with none."""
    rev = revision or src.revision
    rows, hashes = [], {}
    for fname in _lcb_files(src, rev):
        path = _download(src, fname, rev)
        hashes[fname] = _sha256_file(path)
        n_new = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if str(r.get("contest_date", ""))[:10] >= cutoff:
                    rows.append(normalize_lcb_row(r))
                    n_new += 1
        log.info("livecodebench %s: %d problems on/after %s", fname, n_new, cutoff)
        if n_new == 0:
            break
    return rows, hashes


def fetch(raw_dir: str | Path, only: Sequence[str] | None = None, *, lcb_cutoff: str = DEFAULT_LCB_CUTOFF,
          lcb_revision: str | None = None) -> dict[str, Any]:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    man_path = raw / "fetch_manifest.json"
    manifest = read_json(man_path) if man_path.exists() else {}
    wanted = [s for name in (only or SETS) for s in SETS[name]]
    failures = {}
    for name in dict.fromkeys(wanted):
        src = SOURCES[name]
        try:
            rows, hashes = _fetch_one(name, src, lcb_cutoff, lcb_revision)
        except GatedError as e:
            failures[name] = str(e)
            log.error("%s", e)
            continue
        write_jsonl(raw / f"{name}.jsonl", rows)
        revision = (lcb_revision or src.revision) if name == "livecodebench" else src.revision
        manifest[name] = {"repo": src.repo, "revision": revision, "license": src.license, "files": hashes,
                          "rows": len(rows),
                          "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                          **({"cutoff": lcb_cutoff} if name == "livecodebench" else {}),
                          **({"note": src.note} if src.note else {})}
        write_json(man_path, manifest)
        log.info("fetched %s: %d rows", name, len(rows))
    return {"manifest": manifest, "failures": failures}


def _fetch_one(name: str, src: Source, lcb_cutoff: str, lcb_revision: str | None) -> tuple[list[dict], dict]:
    if name == "livecodebench":
        return fetch_livecodebench(src, lcb_cutoff, lcb_revision)
    paths = {f: _download(src, f) for f in src.files}
    hashes = {f: _sha256_file(p) for f, p in paths.items()}
    if name == "gpqa_diamond":
        return _read_csv(paths["gpqa_diamond.csv"]), hashes
    if name in ("mmlu_pro_test", "gsm8k_test"):
        return [{k: _plain(v) for k, v in r.items()} for r in _read_parquet(next(iter(paths.values())))], hashes
    if name == "math_test":
        rows = []
        for f, p in paths.items():
            rows += [{**{k: _plain(v) for k, v in r.items()}, "subject": f.split("/")[0]} for r in _read_parquet(p)]
        return rows, hashes
    if name in ("ifeval", "math500"):
        return list(read_jsonl(next(iter(paths.values())))), hashes
    if name == "bfcl":
        rows = []
        for cat in ("simple", "multiple"):
            answers = {r["id"]: r["ground_truth"] for r in read_jsonl(paths[f"possible_answer/BFCL_v3_{cat}.json"])}
            rows += [{**r, "category": cat, "ground_truth": answers[r["id"]]}
                     for r in read_jsonl(paths[f"BFCL_v3_{cat}.json"]) if r["id"] in answers]
        return rows, hashes
    if name == "jbb":
        return ([{**r, "harmful": True} for r in _read_csv(paths["data/harmful-behaviors.csv"])]
                + [{**r, "harmful": False} for r in _read_csv(paths["data/benign-behaviors.csv"])]), hashes
    if name == "xstest":
        return _read_csv(paths["xstest_prompts.csv"]), hashes
    raise KeyError(name)


# =============================================================================================
# build
# =============================================================================================
def lcb_set_name(cutoff: str) -> str:
    y, m = cutoff[:4], cutoff[5:7]
    return f"livecodebench_post_{y}_{m}"


def _notes(names: Sequence[str], fm: dict[str, Any], drops: Counter) -> str:
    parts = []
    for n in names:
        s = fm.get(n) or {}
        parts.append(f"{n}: {s.get('repo', SOURCES[n].repo)}@{s.get('revision', SOURCES[n].revision)} "
                     f"license={s.get('license', SOURCES[n].license)}")
    if drops:
        parts.append("dropped: " + ", ".join(f"{k}={v}" for k, v in sorted(drops.items())))
    return "; ".join(parts)


def build(raw_dir: str | Path, out_root: str | Path, decontam_out: str | Path, *, only: Sequence[str] | None = None,
          seed: int = 1, lcb_cutoff: str = DEFAULT_LCB_CUTOFF, mmlu_n: int = 2000, force: bool = False
          ) -> dict[str, Any]:
    from dpt.common.manifest import build_manifest, write_manifest

    raw, root, dec = Path(raw_dir), Path(out_root), Path(decontam_out)
    fm = read_json(raw / "fetch_manifest.json") if (raw / "fetch_manifest.json").exists() else {}

    def rows(name: str) -> list[dict[str, Any]]:
        p = raw / f"{name}.jsonl"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing: run `python -m dpt.eval.import_general fetch` first")
        return list(read_jsonl(p))

    builders: dict[str, tuple[str, Callable[[], Built]]] = {
        "gpqa_diamond": ("general/gpqa_diamond.jsonl", lambda: build_gpqa(rows("gpqa_diamond"), seed)),
        "mmlu_pro_2k": ("general/mmlu_pro_2k.jsonl", lambda: build_mmlu_pro(rows("mmlu_pro_test"), mmlu_n, seed)),
        "ifeval": ("general/ifeval.jsonl", lambda: build_ifeval(rows("ifeval"))),
        "livecodebench": (f"general/{lcb_set_name(lcb_cutoff)}.jsonl",
                          lambda: build_livecodebench(rows("livecodebench"), cutoff=lcb_cutoff)),
        "tool_calling": ("general/tool_calling.jsonl", lambda: build_tool_calling(rows("bfcl"))),
        "refusal": ("safety/refusal_regression.jsonl", lambda: build_refusal(rows("jbb"), rows("xstest"))),
    }
    summary: dict[str, Any] = {"sets": {}, "missing": {}, "skipped_existing": []}
    for name in only or SETS:
        if name == "decontam_math":
            try:
                math_sets = decontam_math(rows("gsm8k_test"), rows("math_test"), rows("math500"))
            except FileNotFoundError as e:
                summary["missing"][name] = str(e)
                log.error("%s", e)
                continue
            for dname, drows in math_sets.items():
                write_jsonl(dec / f"{dname}.jsonl", drows)
                summary["sets"][dname] = {"decontam_only": True, "n": len(drows)}
            continue
        rel, fn = builders[name]
        out = root / rel
        if out.exists() and not force:
            summary["skipped_existing"].append(str(out))
            log.warning("%s exists (it may be frozen): not overwritten; pass --force before the freeze", out)
            continue
        try:
            b = fn()
        except FileNotFoundError as e:
            summary["missing"][name] = str(e)
            log.error("%s", e)
            continue
        set_name = Path(rel).stem
        if not b.records:
            summary["missing"][name] = f"no items survived ({dict(b.drops)})"
            log.error("%s: no items (%s); nothing written", set_name, dict(b.drops))
            continue
        ids = [r["id"] for r in b.records]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{set_name}: duplicate ids")
        for iid, src in b.hidden_tests.items():
            p = root / "hidden_tests" / f"{iid}.py"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(src, encoding="utf-8")
        write_jsonl(out, b.records)
        write_manifest(build_manifest(out, f"general.{set_name}", "v1",
                                      stage_stats=[{"stage": f"dropped:{k}", "count": v} for k, v in
                                                   sorted(b.drops.items())] + [{"stage": "kept",
                                                                                "count": len(b.records)}],
                                      notes=_notes(b.sources, fm, b.drops)))
        write_jsonl(dec / f"{set_name}.jsonl", [{"uid": r["id"], "question": r["question"]} for r in b.records])
        summary["sets"][set_name] = {"n": len(b.records), "drops": dict(b.drops), "path": str(out)}
        log.info("%s: %d items → %s (dropped %s)", set_name, len(b.records), out, dict(b.drops))
    write_json(root / "general" / "import_summary.json", summary)
    return summary


# =============================================================================================
# CLI
# =============================================================================================
def _only(v: str | None) -> list[str] | None:
    if not v:
        return None
    names = [x.strip() for x in v.split(",") if x.strip()]
    unknown = [n for n in names if n not in SETS]
    if unknown:
        raise SystemExit(f"unknown sets {unknown}; choose from {sorted(SETS)}")
    return names


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Import the general benchmarks and the refusal regression set")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--raw-dir", required=True)
    f.add_argument("--only", default=None, help=f"comma-separated subset of {sorted(SETS)}")
    f.add_argument("--lcb-cutoff", default=DEFAULT_LCB_CUTOFF, help="LiveCodeBench release-date cutoff YYYY-MM-DD")
    f.add_argument("--lcb-revision", default=None, help="LiveCodeBench dataset revision (default: pinned)")
    b = sub.add_parser("build")
    b.add_argument("--raw-dir", required=True)
    b.add_argument("--out-root", required=True, help="eval-secure root (project.paths.eval_secure_root)")
    b.add_argument("--decontam-out", required=True, help="dir for data_pipeline.inputs.public_benchmarks files")
    b.add_argument("--only", default=None)
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--lcb-cutoff", default=DEFAULT_LCB_CUTOFF)
    b.add_argument("--mmlu-n", type=int, default=2000)
    b.add_argument("--force", action="store_true", help="overwrite existing eval files (never after the freeze)")
    args = ap.parse_args(argv)
    if args.cmd == "fetch":
        res = fetch(args.raw_dir, _only(args.only), lcb_cutoff=args.lcb_cutoff, lcb_revision=args.lcb_revision)
        return 1 if res["failures"] else 0
    s = build(args.raw_dir, args.out_root, args.decontam_out, only=_only(args.only), seed=args.seed,
              lcb_cutoff=args.lcb_cutoff, mmlu_n=args.mmlu_n, force=args.force)
    return 1 if s["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
