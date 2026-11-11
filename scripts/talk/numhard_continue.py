"""D-Num-hard at a 64K budget, by continuing the 32K answers that ran out (2026-11-10).

A sample drawn with a 64K budget is the same as one drawn with a 32K budget up to token 32K, because the budget does
not change sampling. So each answer's 64K outcome is either:
- unchanged, when it finished within 32K; or
- the same model's continuation of its 32K reasoning for the remaining tokens.

That costs about a quarter of a fresh 64K run. It answers the question asked: how many of the unfinished answers
would have come out right with more room?

    python scripts/talk/numhard_continue.py todo --row sft          # how many continuations are still missing
    python scripts/talk/numhard_continue.py run --row sft --endpoint http://127.0.0.1:8000 --served-model sft
    python scripts/talk/numhard_continue.py summary                 # → $WORK/dpt/talk/numhard_64k.json (counts)

The prompt is the harness's own, as in the October final eval:
- `dpt.eval.harness.build_messages`, rendered with the base tokenizer's chat template at medium effort;
- followed by the stored 32K text.

Continuations are sampled at temperature 1 and top-p 1, like the eval. They are appended to
`<eval_root>/d_num_hard_64k/<row>/continuations.jsonl`, so a rerun resumes; `--stop-new-after` stops new requests
before a job's time limit. Grading is the harness's: thinking stripped, fail closed, the numeric grader with strict
units. Rows: base_medium, sft, rl_seed1.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from dpt.chat.thinking import split_thinking
from dpt.common.io import read_jsonl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import failure_types as FT  # noqa: E402  (the same outcome rules as the failure-type analysis)

WORK = Path(os.environ.get("WORK", "."))
R = WORK / "dpt" / "eval_results"
ROWS = ("base_medium", "sft", "rl_seed1")
BUDGET = 65536


def unfinished(row: str) -> list[tuple[str, int, dict[str, Any]]]:
    out = []
    for g in read_jsonl(R / "d_num_hard" / row / "d_num_hard" / "generations.jsonl"):
        for i, s in enumerate(g["samples"]):
            if not split_thinking(s["text"]).has_end:
                out.append((g["item_id"], i, s))
    return out


def cache_path(row: str) -> Path:
    return R / "d_num_hard_64k" / row / "continuations.jsonl"


def cached(row: str) -> dict[str, dict[str, Any]]:
    p = cache_path(row)
    return {r["key"]: r for r in read_jsonl(p)} if p.exists() else {}


def todo(a: argparse.Namespace) -> None:
    done = cached(a.row)
    print(sum(1 for item, i, _ in unfinished(a.row) if f"{item}|{i}" not in done))


async def run_async(a: argparse.Namespace) -> None:
    from dpt.common.config import load_config
    from dpt.eval.datasets import load_eval_items
    from dpt.eval.harness import build_messages, hf_renderer

    cfg = load_config()
    secure = Path(str(cfg.project.paths.eval_secure_root))
    items = {it.id: it for it in load_eval_items(str(secure / "d_num_hard.jsonl"), "d_num_hard")}
    render = hf_renderer(str(cfg.project.model.base))
    done = cached(a.row)
    jobs = [(item, i, s) for item, i, s in unfinished(a.row) if f"{item}|{i}" not in done]
    print(f"{a.row}: {len(jobs)} continuations to do ({len(done)} cached)", flush=True)
    path = cache_path(a.row)
    path.parent.mkdir(parents=True, exist_ok=True)
    t0, stop_at = time.time(), (time.time() + a.stop_new_after) if a.stop_new_after else None
    sem = asyncio.Semaphore(a.concurrency)
    lock = asyncio.Lock()
    stats = collections.Counter()
    async with httpx.AsyncClient(timeout=a.timeout_s) as client:

        async def one(item_id: str, i: int, s: dict[str, Any]) -> None:
            async with sem:
                if stop_at and time.time() > stop_at:
                    stats["skipped_for_time"] += 1
                    return
                key = f"{item_id}|{i}"
                prompt = render(build_messages(items[item_id]), "medium") + s["text"]
                seed = int(hashlib.sha256(f"{a.row}|{key}|64k".encode()).hexdigest()[:8], 16) % (2 ** 31)
                body = {"model": a.served_model, "prompt": prompt, "max_tokens": max(1, BUDGET - int(s["n_tokens"])),
                        "temperature": 1.0, "top_p": 1.0, "n": 1, "seed": seed}
                for attempt in range(4):
                    try:
                        r = await client.post(f"{a.endpoint.rstrip('/')}/v1/completions", json=body)
                        r.raise_for_status()
                        break
                    except (httpx.TransportError, httpx.HTTPStatusError) as e:
                        if attempt == 3:
                            print(f"FAIL {key}: {type(e).__name__}: {e}", flush=True)
                            stats["failed"] += 1
                            return
                        await asyncio.sleep(10 * (attempt + 1))
                data = r.json()
                ch = data["choices"][0]
                rec = {"key": key, "item": item_id, "sample": i, "text": ch.get("text") or "",
                       "finish_reason": ch.get("finish_reason"),
                       "n_tokens": int((data.get("usage") or {}).get("completion_tokens") or 0),
                       "prefix_tokens": int(s["n_tokens"]), "served": data.get("model")}
                async with lock:
                    with open(path, "a") as fh:
                        fh.write(json.dumps(rec) + "\n")
                    stats["done"] += 1
                    if stats["done"] % 20 == 0:
                        print(f"{a.row}: {stats['done']}/{len(jobs)} in {time.time() - t0:.0f}s", flush=True)

        await asyncio.gather(*(one(*j) for j in jobs))
    print(f"{a.row}: finished {dict(stats)} in {time.time() - t0:.0f}s", flush=True)


def summary(_: argparse.Namespace) -> None:
    from dpt.common.config import load_config

    secure = Path(str(load_config().project.paths.eval_secure_root))
    items = {r["id"]: r for r in read_jsonl(secure / "d_num_hard.jsonl")}
    out: dict[str, Any] = {"collected_at": dt.datetime.now().strftime("%Y-%m-%dT%H:%M"), "budget": BUDGET, "rows": {}}
    for row in ROWS:
        cont = cached(row)
        k32: collections.Counter[str] = collections.Counter()
        k64: collections.Counter[str] = collections.Counter()
        moved: collections.Counter[str] = collections.Counter()
        fam: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
        extra_tokens, missing = [], 0
        for g in read_jsonl(R / "d_num_hard" / row / "d_num_hard" / "generations.jsonl"):
            it = items[g["item_id"]]
            for i, s in enumerate(g["samples"]):
                before = FT.grade(s["text"], it["answer"])["kind"]
                after = before
                if before == "unfinished":
                    c = cont.get(f"{g['item_id']}|{i}")
                    if c is None:
                        missing += 1
                    else:
                        after = FT.grade(s["text"] + c["text"], it["answer"])["kind"]
                        moved[after] += 1
                        extra_tokens.append(c["n_tokens"])
                k32[before] += 1
                k64[after] += 1
                fam[it["meta"]["family"]][after] += 1
        n = sum(k32.values())
        out["rows"][row] = {
            "answers": n, "missing_continuations": missing,
            "at_32k_pct": {k: round(100 * v / n, 1) for k, v in k32.most_common()},
            "at_64k_pct": {k: round(100 * v / n, 1) for k, v in k64.most_common()},
            "unfinished_at_32k_became": dict(moved),
            "extra_tokens_median": sorted(extra_tokens)[len(extra_tokens) // 2] if extra_tokens else None,
            "families_64k_pct": {f: {k: round(100 * v / sum(c.values()), 1) for k, v in c.most_common()}
                                 for f, c in sorted(fam.items())},
        }
    p = WORK / "dpt" / "talk" / "numhard_64k.json"
    p.write_text(json.dumps(out, indent=1))
    for row, v in out["rows"].items():
        print(row, "32K", v["at_32k_pct"], "→ 64K", v["at_64k_pct"],
              "| unfinished became", v["unfinished_at_32k_became"], "| missing", v["missing_continuations"])
    print(p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("todo")
    t.add_argument("--row", required=True, choices=ROWS)
    r = sub.add_parser("run")
    r.add_argument("--row", required=True, choices=ROWS)
    r.add_argument("--endpoint", required=True)
    r.add_argument("--served-model", required=True)
    r.add_argument("--concurrency", type=int, default=60)
    r.add_argument("--timeout-s", type=float, default=10800)
    r.add_argument("--stop-new-after", type=float, default=0, help="seconds from start after which no new request")
    sub.add_parser("summary")
    a = ap.parse_args()
    if a.cmd == "todo":
        todo(a)
    elif a.cmd == "run":
        asyncio.run(run_async(a))
    else:
        summary(a)


if __name__ == "__main__":
    main()
