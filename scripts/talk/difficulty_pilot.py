"""Pilot: the SFT model's pass rates on bank v2's hard and long T1 families (2026-11-15).

RL learns only from groups whose answers differ: a problem the start model always solves, or never solves, gives
no gradient. Run 1 had almost nothing to learn (85–97% of its T1 groups were solved 8/8). Bank v2 adds 12 hard and 6
long families (10,780 problems) that the SFT model has never been graded on, and RL v2 weights them up. This pilot
grades 10 problems of each family, 8 answers each, at 24K tokens and medium effort (the settings of the bank's
re-grade and of RL v2), with the training grader, on local vLLM servers. Per family it reports the pass-rate
histogram (k of 8), the expected share of groups with mixed results, answer lengths, unfinished answers, and how many
of those already hold the right value.

    python scripts/talk/difficulty_pilot.py --problems problems.jsonl --model <sft export> \\
        --servers http://localhost:8411/v1,http://localhost:8412/v1 --out <dir>
    python scripts/talk/difficulty_pilot.py --summarize <dir>        # the summary of the answers so far

Writes ``<dir>/answers.jsonl`` (one row per problem, resumable) and ``<dir>/summary.json`` (counts only).
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics as st
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dpt.chat.thinking import split_thinking
from dpt.common.seeding import derive_seed
from dpt.data.backends import OpenAICompatClient, VLLMGenerator, load_hf_tokenizer
from dpt.data.prompts import messages_for_problem
from dpt.data.schema import Problem
from dpt.eval.graders.numeric import value_seen
from dpt.rewards.router import compute_score

N_ANSWERS, MAX_TOKENS, EFFORT = 8, 24576, "medium"


def grade(p: Problem, reply: str) -> float:
    """Correctness as RL counts it (``acc``), as dpt.data.pipeline grades the difficulty pass."""
    out = compute_score(p.task, reply, json.dumps(p.answer, sort_keys=True),
                        {"uid": p.uid, "task": p.task, "pool": p.pool, "question": p.question, "response_length": 0})
    return float(out.get("acc", out.get("score", 0.0))) if isinstance(out, dict) else float(out)


def run(problems_path: Path, model: str, servers: list[str], out_dir: Path, in_flight: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    answers_path = out_dir / "answers.jsonl"
    done = {json.loads(line)["uid"] for line in answers_path.open()} if answers_path.exists() else set()
    rows = [json.loads(line) for line in problems_path.open()]
    # families interleaved, so a partial run already covers every family
    rank: dict[str, int] = defaultdict(int)
    order = []
    for r in rows:
        f = r["source"]["family"]
        order.append((rank[f], f, r))
        rank[f] += 1
    todo = [r for _, _, r in sorted(order, key=lambda x: (x[0], x[1])) if r["uid"] not in done]
    print(f"{len(done)} problems done before, {len(todo)} to go", flush=True)
    tokenizer = load_hf_tokenizer(model)
    generators = [VLLMGenerator(OpenAICompatClient(url, "sft_pilot", timeout=7200, max_concurrency=in_flight),
                                tokenizer=tokenizer, temperature=1.0, top_p=1.0) for url in servers]
    next_generator = itertools.cycle(generators)
    lock = threading.Lock()

    def one(row: dict) -> None:
        p = Problem.from_dict(row, assign_uid=False)
        with lock:
            generate = next(next_generator)
        replies = generate(messages_for_problem(p), n=N_ANSWERS, effort=EFFORT, max_tokens=MAX_TOKENS,
                           seed=derive_seed(1, "difficulty", p.uid))
        answers = []
        for reply in replies:
            finished = split_thinking(reply).has_end
            answers.append({"right": grade(p, reply) >= 1.0, "finished": finished,
                            "tokens": len(tokenizer(reply, add_special_tokens=False).input_ids),
                            "value_seen": (not finished) and value_seen(reply, p.answer) is not None})
        rec = {"uid": p.uid, "family": p.source.get("family"), "group": row["meta"]["pilot_group"],
               "k": sum(a["right"] for a in answers), "answers": answers}
        with lock:
            with answers_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            done.add(p.uid)
            print(f"[{len(done)}/{len(rows)}] {rec['family']} k={rec['k']}/8 "
                  f"tokens p50={st.median(a['tokens'] for a in answers):.0f}", flush=True)

    with ThreadPoolExecutor(in_flight) as ex:
        list(ex.map(one, todo))
    summarize(out_dir)


def mixed(k: int, n: int = N_ANSWERS) -> float:
    """Chance that a group of n answers has mixed results when the problem is solved with rate k/n."""
    r = k / n
    return 1.0 - r ** n - (1.0 - r) ** n


BUDGETS = (16384, 20480, 24576)


def by_budget(rs: list[dict], budget: int) -> dict:
    """How learnable the problems are if RL trains at ``budget`` tokens: an answer longer than that is cut off.
    Seeds 1-2 leave cut-off answers out (a group carries gradient only if its finished answers are both right and
    wrong); seed 3 counts them as wrong (a group carries gradient if some but not all answers are right in time)."""
    masked = as_wrong = cut = 0
    for r in rs:
        right = sum(a["right"] and a["finished"] and a["tokens"] <= budget for a in r["answers"])
        wrong = sum(a["finished"] and not a["right"] and a["tokens"] <= budget for a in r["answers"])
        cut += sum(not (a["finished"] and a["tokens"] <= budget) for a in r["answers"])
        masked += right > 0 and wrong > 0
        as_wrong += 0 < right < len(r["answers"])
    n = max(1, len(rs))
    return {"mixed_if_cut_left_out": round(masked / n, 3), "mixed_if_cut_counts_wrong": round(as_wrong / n, 3),
            "cut_off": round(cut / max(1, sum(len(r["answers"]) for r in rs)), 3)}


def summarize(out_dir: Path) -> dict:
    recs = [json.loads(line) for line in (out_dir / "answers.jsonl").open()]
    by: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by[r["family"]].append(r)
        by[f"all {r['group']}"].append(r)
    by["all"] = recs
    summary = {}
    for name, rs in sorted(by.items()):
        answers = [a for r in rs for a in r["answers"]]
        unfinished = [a for a in answers if not a["finished"]]
        ks = Counter(r["k"] for r in rs)
        summary[name] = {
            "problems": len(rs), "answers": len(answers),
            "pass_rate": round(sum(a["right"] for a in answers) / max(1, len(answers)), 3),
            "k_histogram": {str(k): ks.get(k, 0) for k in range(N_ANSWERS + 1)},
            "share_8_of_8": round(ks.get(N_ANSWERS, 0) / len(rs), 3), "share_0_of_8": round(ks.get(0, 0) / len(rs), 3),
            "learnable_1_to_6": round(sum(ks.get(k, 0) for k in range(1, 7)) / len(rs), 3),
            "expected_mixed": round(st.mean(mixed(r["k"]) for r in rs), 3),
            "tokens_p50": round(st.median(a["tokens"] for a in answers)),
            "tokens_p90": round(sorted(a["tokens"] for a in answers)[int(0.9 * (len(answers) - 1))]),
            "unfinished": round(len(unfinished) / max(1, len(answers)), 3),
            "unfinished_with_value": round(sum(a["value_seen"] for a in unfinished) / max(1, len(unfinished)), 3),
            "by_budget": {str(b): by_budget(rs, b) for b in BUDGETS},
        }
    meta = {"about": "SFT model (sft9b_seed1) on 10 problems of each of bank v2's 12 hard and 6 long T1 families, "
                     "8 answers each, 24K tokens, medium effort, temperature 1, the training grader (2026-11-15). "
                     "Counts only.", "settings": {"answers": N_ANSWERS, "max_tokens": MAX_TOKENS, "effort": EFFORT}}
    (out_dir / "summary.json").write_text(json.dumps({**meta, "families": summary}, indent=1))
    print(f"{'family':40s} {'n':>3s} {'pass':>5s} {'8/8':>5s} {'0/8':>5s} {'1-6':>5s} {'mixed':>5s} {'tok50':>6s} "
          f"{'unfin':>5s} {'w/val':>5s}")
    for name, s in summary.items():
        print(f"{name:40s} {s['problems']:3d} {s['pass_rate']:5.2f} {s['share_8_of_8']:5.2f} {s['share_0_of_8']:5.2f} "
              f"{s['learnable_1_to_6']:5.2f} {s['expected_mixed']:5.2f} {s['tokens_p50']:6d} {s['unfinished']:5.2f} "
              f"{s['unfinished_with_value']:5.2f}")
    print("\nmixed groups by training budget (left out = seeds 1-2, counts wrong = seed 3) and the share cut off:")
    for name, s in summary.items():
        cells = [f"{int(b) // 1024}K {v['mixed_if_cut_left_out']:.2f}/{v['mixed_if_cut_counts_wrong']:.2f} "
                 f"cut {v['cut_off']:.2f}" for b, v in s["by_budget"].items()]
        print(f"{name:40s} " + "  ".join(cells))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--problems", type=Path)
    ap.add_argument("--model")
    ap.add_argument("--servers", default="http://localhost:8411/v1,http://localhost:8412/v1")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--in-flight", type=int, default=24, help="problems in flight (each asks for 8 answers)")
    ap.add_argument("--summarize", type=Path, default=None)
    a = ap.parse_args()
    if a.summarize:
        summarize(a.summarize)
        return
    run(a.problems, a.model, [s for s in a.servers.split(",") if s], a.out, a.in_flight)


if __name__ == "__main__":
    main()
