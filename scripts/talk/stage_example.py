"""One question, every stage: the same new problem answered by the base model, SFT, RL run 1 and RL v2 (2026-11-18).

The README's results tables say how often each model is right; this shows what an answer looks like. To keep it from
being a picked example, the question is fixed by a rule before any model answers it: the first problem drawn with
seed 20261118 from three_layer_head_wave_time, one of the public training families, so it is in no test set and new to
every model. Each model answers it 16 times with the evaluation's settings (the prompt the harness renders, medium
effort, temperature 1, top-p 1, at most 32K tokens), graded by the evaluation's numeric grader. The figure shows each
model's first answer (sample 0, seeded) and counts the other 15.

    scripts/serve_vllm.sh <model> <name> <port> 36864      # one server per model (DPT_VLLM_ENV=dpt-train)
    python scripts/talk/stage_example.py --model base=http://localhost:8461/v1 --model sft=... --model rl_seed1=...
    python scripts/talk/stage_example.py --model rl_v2_seed1=http://localhost:8464/v1     # adds a model, keeps the rest

Writes docs/talk/data/stage_example.json: the question and its solver answer; per model and sample the length, whether
the reasoning ended, the final answer and the grade; and an excerpt of sample 0 (the start and the end of its
reasoning, and its final answer).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import dpt.data.gen.seismology_hard  # noqa: F401  (registers the hard training families)
from dpt.chat.thinking import split_thinking
from dpt.common.seeding import derive_seed
from dpt.data.backends import OpenAICompatClient, VLLMGenerator, load_hf_tokenizer
from dpt.data.gen.families import FAMILIES
from dpt.data.prompts import build_messages
from dpt.eval.graders.numeric import grade_numeric

OUT = Path(__file__).resolve().parents[2] / "docs" / "talk" / "data" / "stage_example.json"
FAMILY, SEED, TEMPLATE = "three_layer_head_wave_time", 20261118, 0
# 16 answers per model: with 8, two runs of the same models gave the base 1 and 4 right and SFT 5 and 8 (2026-11-18)
SAMPLES, MAX_TOKENS, EFFORT = 16, 32768, "medium"
HEAD_CHARS, TAIL_CHARS, ANSWER_CHARS = 700, 500, 300


def the_question() -> dict:
    family = FAMILIES[FAMILY]
    params = family.sample(random.Random(SEED))
    return {"family": FAMILY, "seed": SEED, "template": TEMPLATE, "question": family.render(params, TEMPLATE),
            "answer": family.answer(params), "steps": family.steps}


def answer_record(reply: str, tokens: int, gold: dict) -> dict:
    split = split_thinking(reply)
    final = split.answer if split.has_end else ""
    grade = grade_numeric(final, gold) if split.has_end else None
    return {"tokens": tokens, "finished": split.has_end, "right": bool(grade and grade.correct),
            "reason": "unfinished" if grade is None else grade.reason,
            "answer": None if grade is None else (grade.details or {}).get("answer"),
            "final_answer_end": final[-ANSWER_CHARS:]}


def excerpt(reply: str) -> dict:
    """The start and the end of the reasoning, and of the final answer (whose end holds the boxed result)."""
    split = split_thinking(reply)
    thinking, final = split.thinking, split.answer if split.has_end else ""
    long_final = len(final) > 2 * ANSWER_CHARS
    return {"thinking_start": thinking[:HEAD_CHARS],
            "thinking_end": thinking[-TAIL_CHARS:] if len(thinking) > HEAD_CHARS + TAIL_CHARS else "",
            "final_answer_start": final[:ANSWER_CHARS] if long_final else final,
            "final_answer_end": final[-ANSWER_CHARS:] if long_final else ""}


def run_model(name: str, url: str, question: dict, tokenizer_path: str) -> dict:
    tokenizer = load_hf_tokenizer(tokenizer_path)
    generate = VLLMGenerator(OpenAICompatClient(url, name, timeout=7200, max_concurrency=SAMPLES),
                             tokenizer=tokenizer, temperature=1.0, top_p=1.0)
    messages = build_messages("t1_num", question["question"])
    replies = generate(messages, n=SAMPLES, effort=EFFORT, max_tokens=MAX_TOKENS,
                       seed=derive_seed(SEED, "stage_example", name))
    records = [answer_record(r, len(tokenizer(r, add_special_tokens=False).input_ids), question["answer"])
               for r in replies]
    return {"samples": records, "right": sum(r["right"] for r in records),
            "unfinished": sum(not r["finished"] for r in records),
            "median_tokens": round(statistics.median(r["tokens"] for r in records)),
            "example": {"sample": 0, **excerpt(replies[0]),
                        **{k: records[0][k] for k in ("tokens", "right", "reason", "finished", "answer")}}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", action="append", default=[], help="name=url of an OpenAI-compatible server")
    ap.add_argument("--tokenizer", default=None, help="tokenizer for rendering (default: the server's model path)")
    ap.add_argument("--model-path", action="append", default=[], help="name=local path, for the tokenizer")
    a = ap.parse_args()
    question = the_question()
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    if data.get("question") != question:
        data = {"question": question, "models": {}}
    paths = dict(pair.split("=", 1) for pair in a.model_path)
    servers = dict(pair.split("=", 1) for pair in a.model)
    with ThreadPoolExecutor(len(servers) or 1) as pool:          # one server per model: answer them all at once
        futures = {name: pool.submit(run_model, name, url, question, a.tokenizer or paths[name])
                   for name, url in servers.items()}
        for name, future in futures.items():
            data["models"][name] = m = future.result()
            print(f"{name}: {m['right']}/{SAMPLES} right, {m['unfinished']} unfinished, "
                  f"median {m['median_tokens']} tokens")
    data["about"] = ("One new problem from a public training family (seed 20261118), answered 16 times by each model "
                     "with the evaluation's prompt and settings (medium effort, temperature 1, top-p 1, 32K tokens) "
                     "and graded by the evaluation's numeric grader; an excerpt of each model's first answer "
                     "(scripts/talk/stage_example.py).")
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
