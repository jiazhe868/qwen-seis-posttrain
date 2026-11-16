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

import numpy as np

from dpt.chat.thinking import split_thinking
from dpt.common.seeding import derive_seed
from dpt.data.backends import OpenAICompatClient, VLLMGenerator, load_hf_tokenizer
from dpt.data.prompts import messages_for_problem
from dpt.data.schema import Problem
from dpt.eval.graders.numeric import value_seen
from dpt.rewards.router import compute_score
from dpt.rl.advantage import group_mean_advantage
from dpt.rl.filtering import scores_and_validity, zero_advantage_mask
from dpt.rl.pipeline import within_group_length_gap

N_ANSWERS, MAX_TOKENS, EFFORT = 8, 24576, "medium"
SHORT = 16384                          # seed 3's training budget


def grade(p: Problem, reply: str) -> float:
    """Correctness as RL counts it (``acc``), as dpt.data.pipeline grades the difficulty pass."""
    out = compute_score(p.task, reply, json.dumps(p.answer, sort_keys=True),
                        {"uid": p.uid, "task": p.task, "pool": p.pool, "question": p.question, "response_length": 0})
    return float(out.get("acc", out.get("score", 0.0))) if isinstance(out, dict) else float(out)


def run(problems_path: Path, model: str, servers: list[str], out_dir: Path, in_flight: int, max_tokens: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = {"problems": problems_path.name, "model": Path(model).name, "answers": N_ANSWERS,
                "max_tokens": max_tokens, "effort": EFFORT}
    (out_dir / "settings.json").write_text(json.dumps(settings))
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
        replies = generate(messages_for_problem(p), n=N_ANSWERS, effort=EFFORT, max_tokens=max_tokens,
                           seed=derive_seed(1, "difficulty", p.uid))
        answers = []
        for reply in replies:
            finished = split_thinking(reply).has_end
            ids = tokenizer(reply, add_special_tokens=False).input_ids
            seen = value_seen(reply, p.answer)
            answers.append({
                "right": grade(p, reply) >= 1.0, "finished": finished, "tokens": len(ids),
                "value_seen": (not finished) and seen is not None,
                # The 16K risk check (third review of RL v2): a right answer that finishes between 16K and 24K is a
                # wrong one at 16K. If its value was already written before token 16,384, the cut removes checking;
                # if not, it removes the work that found the value. value_at: the token where a number matching the
                # gold first appears (None = never).
                "value_at": None if seen is None else len(tokenizer(reply[:int(seen * len(reply))],
                                                                    add_special_tokens=False).input_ids),
                "value_by_16k": None if len(ids) <= SHORT else value_seen(tokenizer.decode(ids[:SHORT]),
                                                                          p.answer) is not None})
        rec = {"uid": p.uid, "family": p.source.get("family"), "group": row["meta"]["pilot_group"],
               "k": sum(a["right"] for a in answers), "answers": answers,
               # a gold value that is also a number in the question is "seen" as soon as the reply restates the data
               "value_in_question": value_seen(p.question, p.answer) is not None}
        with lock:
            with answers_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            with (out_dir / "texts.jsonl").open("a") as f:
                f.write(json.dumps({"uid": p.uid, "texts": replies}) + "\n")
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
RAMP, RAMP_PENALTY = 4096, 0.5         # the soft overlong penalty: the last 4K before the cap, down to -0.5
# What a training run does with an answer that runs past its budget. Run 1 (16K) and RL v2 seeds 1-2 (24K) leave it
# out of the loss and of its group's mean, and dock answers that end in the last 4K; seed 3 counts it as wrong.
RULES = {"left out + penalty": (False, RAMP_PENALTY), "left out": (False, 0.0), "counts wrong": (True, 0.0)}


def first_step(rs: list[dict], budget: int, cut_as_wrong: bool, penalty: float) -> dict:
    """The first GRPO step on these problems at ``budget`` tokens under one rule, computed with the trainer's own
    functions: the share of groups that carry gradient, the share whose answers are both right and wrong, and which
    way the step pushes length inside the groups with gradient (dpt.rl.pipeline.within_group_length_gap, logged in
    training as task/t1_num/len_gap_within; the third review of RL v2 asked for it). With the penalty, a group of
    right answers carries gradient once one of them ends in the last 4K, and that gradient is about length only."""
    scores, lens, gids, cut, ok = [], [], [], [], []
    for r in rs:
        for a in r["answers"]:
            # every unfinished answer hit the cap (re-tokenized, 24,565-24,579 tokens), so unfinished = cut off
            over = not a["finished"] or a["tokens"] > budget
            length = min(a["tokens"], budget)
            ramp = min(1.0, max(0.0, (length - budget + RAMP) / RAMP))
            scores.append(float(a["right"] and not over) - penalty * ramp)
            lens.append(length)
            gids.append(r["uid"])
            cut.append(over)
            ok.append(a["right"] and not over)
    tids = np.asarray(["t1_num"] * len(scores), dtype=object)
    r, valid = scores_and_validity(scores, list(tids), cut, ["t1_num"] if cut_as_wrong else [])
    live = valid & ~zero_advantage_mask(r, gids, valid=valid)
    gap = within_group_length_gap(group_mean_advantage(r, gids, valid=valid), np.asarray(lens), gids, tids, live)
    outcomes: dict[str, list[bool]] = defaultdict(list)
    for g, o, v in zip(gids, ok, valid):
        if v:
            outcomes[g].append(o)
    n = max(1, len(rs))
    adv = group_mean_advantage(r, gids, valid=valid)
    with_gradient = {g for g, m in zip(gids, live) if m}
    # groups whose gradient comes from the penalty alone: all answers right (or all wrong), docked by length
    length_only = {g for g in with_gradient if all(outcomes[g]) or not any(outcomes[g])}
    mass = [abs(float(a)) for a, m in zip(adv, live) if m]
    mass_length_only = [abs(float(a)) for a, m, g in zip(adv, live, gids) if m and g in length_only]
    return {"with_gradient": round(len(with_gradient) / n, 3),
            "right_and_wrong": round(sum(any(o) and not all(o) for o in outcomes.values()) / n, 3),
            "length_only": round(len(length_only) / n, 3),
            "length_only_advantage_share": round(sum(mass_length_only) / sum(mass), 3) if mass else None,
            "length_gap_within": round(gap["task/t1_num/len_gap_within"]) if gap else None,
            "below_zero": round(gap["task/t1_num/len_gap_within_below_zero"], 3) if gap else None}


def right_in_time(r: dict, budget: int) -> int:
    return sum(a["right"] and a["finished"] and a["tokens"] <= budget for a in r["answers"])


def budget_moves(rs: list[dict], budget: int, top: int) -> dict:
    """Where groups go when the budget drops from ``top`` to ``budget`` and a cut-off answer counts as wrong (fourth
    review of RL v2): a mixed group whose right answers were all slow becomes 0/8 and loses its gradient; an 8/8
    group with a slow right answer becomes mixed and gains one."""
    lost = sum(0 < right_in_time(r, top) < len(r["answers"]) and right_in_time(r, budget) == 0 for r in rs)
    gained = sum(right_in_time(r, top) == len(r["answers"]) and 0 < right_in_time(r, budget) < len(r["answers"])
                 for r in rs)
    return {"mixed_to_none_right": lost, "all_right_to_mixed": gained}


def by_budget(rs: list[dict], budget: int, top: int) -> dict:
    """Each rule's first step at ``budget``, the share of answers cut off, and the gap of global means that the
    training panel logs (task/t1_num/len_correct minus len_incorrect over the whole batch, cut-off answers at the
    budget): run 1's logs held only this one."""
    answers = [a for r in rs for a in r["answers"]]
    in_time = [a["finished"] and a["tokens"] <= budget for a in answers]
    # right minus wrong length inside each group over finished answers, whatever the rule (third review of RL v2)
    gaps = []
    for r in rs:
        done = [a for a in r["answers"] if a["finished"] and a["tokens"] <= budget]
        right_n, wrong_n = [a["tokens"] for a in done if a["right"]], [a["tokens"] for a in done if not a["right"]]
        if right_n and wrong_n:
            gaps.append(st.mean(right_n) - st.mean(wrong_n))
    right = [a["tokens"] for a, t in zip(answers, in_time) if t and a["right"]]
    other = [min(a["tokens"], budget) for a, t in zip(answers, in_time) if not (t and a["right"])]
    return {"cut_off": round(1 - sum(in_time) / max(1, len(answers)), 3),
            "global_gap": round(st.mean(right) - st.mean(other)) if right and other else None,
            "moves": budget_moves(rs, budget, top),
            "finished_gap": {"groups": len(gaps), "mean": round(st.mean(gaps)) if gaps else None,
                             "median": round(st.median(gaps)) if gaps else None,
                             "below_zero": sum(g < 0 for g in gaps)},
            **{rule: first_step(rs, budget, *args) for rule, args in RULES.items()}}


def late_right_check(rs: list[dict], max_tokens: int) -> dict:
    """The 16K risk check: right answers that finish between 16K and 24K, and whether a number matching the gold was
    already written before token 16,384. Mostly yes: a 16K cut removes checking, and 16K training is safe. Mostly no:
    it removes the work that found the value. Problems whose question holds the gold value are left out."""
    late = [a for r in rs if not r.get("value_in_question") for a in r["answers"]
            if a["right"] and a["finished"] and SHORT < a["tokens"] <= max_tokens and a.get("value_by_16k") is not None]
    at = [a["value_at"] for a in late if a["value_at"] is not None]
    # what the answers that wrote the value before 16K did next: kept it (right), argued it away (finished wrong), or
    # never stopped checking (unfinished at the cap)
    early = [a for r in rs if not r.get("value_in_question") for a in r["answers"]
             if a.get("value_at") is not None and a["value_at"] < SHORT]
    return {"answers": len(late),
            "after_value_by_16k": {"answers": len(early), "right": sum(a["right"] for a in early),
                                   "finished_wrong": sum(a["finished"] and not a["right"] for a in early),
                                   "unfinished": sum(not a["finished"] for a in early)},
            "value_by_16k": round(sum(a["value_by_16k"] for a in late) / len(late), 3) if late else None,
            "value_at_p50": round(st.median(at)) if at else None,
            "tokens_p50": round(st.median(a["tokens"] for a in late)) if late else None}


def summarize(out_dir: Path) -> dict:
    recs = [json.loads(line) for line in (out_dir / "answers.jsonl").open()]
    settings_path = out_dir / "settings.json"
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {"max_tokens": MAX_TOKENS}
    budgets = [b for b in BUDGETS if b <= settings["max_tokens"]]
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
            "right_after_16k": round(sum(a["right"] and a["tokens"] > SHORT for a in answers)
                                     / max(1, sum(a["right"] for a in answers)), 3),
            "late_right_check": late_right_check(rs, settings["max_tokens"]),
            "by_budget": {str(b): by_budget(rs, b, budgets[-1]) for b in budgets},
        }
    meta = {"about": "The SFT model's answers, 8 per problem at temperature 1 and medium effort, graded with the "
                     "training grader (scripts/talk/difficulty_pilot.py). Counts only.", "settings": settings}
    (out_dir / "summary.json").write_text(json.dumps({**meta, "families": summary}, indent=1))
    print(f"{'family':40s} {'n':>3s} {'pass':>5s} {'8/8':>5s} {'0/8':>5s} {'1-6':>5s} {'mixed':>5s} {'tok50':>6s} "
          f"{'unfin':>5s} {'w/val':>5s}")
    for name, s in summary.items():
        print(f"{name:40s} {s['problems']:3d} {s['pass_rate']:5.2f} {s['share_8_of_8']:5.2f} {s['share_0_of_8']:5.2f} "
              f"{s['learnable_1_to_6']:5.2f} {s['expected_mixed']:5.2f} {s['tokens_p50']:6d} {s['unfinished']:5.2f} "
              f"{s['unfinished_with_value']:5.2f}")
    print("\ngroups with gradient at each budget (left out + penalty / left out / counts wrong) and the share cut off:")
    for name, s in summary.items():
        cells = [f"{int(b) // 1024}K " + "/".join(f"{v[rule]['with_gradient']:.2f}" for rule in RULES)
                 + f" cut {v['cut_off']:.2f}" for b, v in s["by_budget"].items()]
        print(f"{name:40s} " + "  ".join(cells))
    print("\nfirst step: groups with gradient, of them right-and-wrong, length gap inside them (share below zero), "
          "and the gap of global means")
    for name in [n for n in summary if n == "all" or n.startswith("all ")]:
        for b, v in summary[name]["by_budget"].items():
            for rule in RULES:
                f = v[rule]
                gap = ("n/a" if f["length_gap_within"] is None
                       else f"{f['length_gap_within']:+d} ({f['below_zero']:.2f})")
                print(f"{name:9s} {int(b) // 1024}K {rule:18s} gradient {f['with_gradient']:.2f} right+wrong "
                      f"{f['right_and_wrong']:.2f} gap {gap:16s} global {v['global_gap']}")
    print("\nfrom the top budget down, cut-off answers counted as wrong: mixed groups that become 0/8 (lost), "
          "8/8 groups that become mixed (gained)")
    for b, v in summary["all"]["by_budget"].items():
        m = v["moves"]
        print(f"  {int(b) // 1024}K: {m['mixed_to_none_right']} lost, {m['all_right_to_mixed']} gained")
    c = summary["all"]["late_right_check"]
    print(f"\nright answers longer than 16K: {summary['all']['right_after_16k']:.2f} of right answers; checked "
          f"{c['answers']}: value written before token 16,384 in {c['value_by_16k']}, first at token "
          f"{c['value_at_p50']} (median), answer ends at {c['tokens_p50']}")
    e = c["after_value_by_16k"]
    print(f"answers with the value before 16K: {e['answers']}; right {e['right']}, finished wrong "
          f"{e['finished_wrong']}, unfinished {e['unfinished']}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--problems", type=Path)
    ap.add_argument("--model")
    ap.add_argument("--servers", default="http://localhost:8411/v1,http://localhost:8412/v1")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--in-flight", type=int, default=24, help="problems in flight (each asks for 8 answers)")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="16384 for run 1's bank (its training budget)")
    ap.add_argument("--summarize", type=Path, default=None)
    a = ap.parse_args()
    if a.summarize:
        summarize(a.summarize)
        return
    run(a.problems, a.model, [s for s in a.servers.split(",") if s], a.out, a.in_flight, a.max_tokens)


if __name__ == "__main__":
    main()
