"""Difficulty calibration: sample the *base* model 8× per problem at medium effort.

Report §处理流水线 step 5. Drop problems the base solves 8/8. Keep the rest,
and route 0/8 problems to a separate *hard pool* that enters RL only after
step 150 (74.1k → 60k, of which 7k hard).

Why this is the right filter for GRPO
-------------------------------------
GRPO's advantage is ``r_i − mean(r_group)``. A prompt the policy always solves
(8/8) has advantage 0 for every sample, so it costs a full rollout and a
forward/backward pass and contributes zero gradient. Dropping these up front
is the static version of the online "drop zero-advantage groups" trick
(report §单步时间预算). A 0/8 prompt is also zero-advantage *today*, but it
becomes useful once the policy improves. That is why it is deferred to the
hard pool and not thrown away.

Calibration settings must match RL rollout settings: same effort (medium),
temperature 1.0, top-p 1.0, the same prompt format and the same training
grader. Otherwise "8/8" means something different from "zero advantage
during RL".

For tasks with continuous rewards (T3), "correct" means score ≥
``correct_threshold[task]``. With continuous scores the group is rarely
exactly zero-advantage, so a high threshold (0.9) only removes items that
are uniformly near-perfect.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from dpt.common.logging import get_logger
from dpt.common.seeding import derive_seed
from dpt.common.types import T2_CODE, T3_REPORT, T4_GENERAL
from dpt.data.prompts import messages_for_problem
from dpt.data.schema import Problem

log = get_logger(__name__)

GradeFn = Callable[[Problem, str], float]


@dataclass
class DifficultyConfig:
    n_samples: int = 8
    effort: str = "medium"
    max_tokens: int = 24576
    correct_threshold: dict[str, float] = field(default_factory=lambda: {
        "t1_num": 1.0, "t2_code": 1.0, "t3_report": 0.9, "t4_general": 1.0})
    drop_all_correct: bool = True
    hard_pool_if_none_correct: bool = True
    skip_tasks: list[str] = field(default_factory=list)   # tasks kept without calibration
    max_concurrency: int = 32
    seed: int = 1
    # A second, shorter budget (tokens; 0 = off), 2026-11-15. A reply counts as right within it only if it is right
    # and at most this long. A problem is dropped only when all n replies are right within it, and a template counts
    # as mastered on the same test, so a problem solved n/n at max_tokens but slowly stays in the bank: a run that
    # trains at the shorter budget, where slow answers are cut off, learns from it (RL v2 seed 3 at 16K).
    short_budget: int = 0
    # More budgets (tokens) whose pass rates are only recorded, pass_at_8_medium_<n>k, for runs that train there; the
    # drop and the mastered test stay on short_budget (2026-11-15: 20K, a candidate budget for RL v2 seed 3).
    extra_budgets: list[int] = field(default_factory=list)
    # At most this many problems of one family are graded and kept (0 = all), 2026-11-16, the user's choice: bank v2's
    # 18 new families have 600 problems each, and grading all 10,800 at 24K would take ~32 hours on RL v2's 24
    # replicas, more than the job has before RL. 200 per family keeps every family and leaves RL ~17 hours.
    max_per_family: int = 0

    def budget_key(self, budget: int) -> str:
        return f"pass_at_{self.n_samples}_{self.effort}_{budget // 1024}k"

    def short_key(self) -> str:
        return self.budget_key(self.short_budget)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "DifficultyConfig":
        d = dict(d or {})
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DifficultyResult:
    kept: list[Problem]
    dropped: list[dict[str, Any]] = field(default_factory=list)   # {"uid", "reason", "k"}
    histogram: dict[str, dict[int, int]] = field(default_factory=dict)  # task -> {k_correct: count}

    def stats(self) -> dict[str, Any]:
        hard = sum(1 for p in self.kept if p.pool == "hard")
        return {"kept": len(self.kept), "hard_pool": hard, "main_pool": len(self.kept) - hard,
                "dropped_all_correct": len(self.dropped),
                "histogram": {t: dict(sorted(h.items())) for t, h in self.histogram.items()}}


LengthFn = Callable[[str], int]


def drop_unconfirmed(kept: list[Problem], n_samples: int, effort: str) -> tuple[list[Problem], list[dict[str, Any]]]:
    """Problems whose label nothing has confirmed yet (``meta.label_check == "sft"``: their template was readmitted
    after the verifiers were undecided, dpt.data.bank_merge.readmit_undecided) stay only if the SFT model reached
    the label in at least one of its answers. Those it never reached are dropped, not sent to the hard pool."""
    key = f"pass_at_{n_samples}_{effort}"
    unconfirmed = {p.uid for p in kept if p.meta.get("label_check") == "sft" and not p.meta.get(key)}
    return ([p for p in kept if p.uid not in unconfirmed],
            [{"uid": u, "reason": "label_unconfirmed", "k": 0} for u in sorted(unconfirmed)])


def cap_per_family(problems: list[Problem], max_per_family: int) -> tuple[list[Problem], list[dict[str, Any]]]:
    """At most ``max_per_family`` problems of each family (``source.family``), the first in a fixed order hashed
    from the uid, so every template of the family is sampled and a rerun keeps the same ones. Problems without a
    family are all kept. Returns (kept, dropped as {"uid", "reason": "family_cap", "k": None})."""
    if max_per_family <= 0:
        return list(problems), []
    by_family: dict[str, list[Problem]] = {}
    for p in problems:
        fam = p.source.get("family") or p.meta.get("family")
        if fam:
            by_family.setdefault(str(fam), []).append(p)
    capped: set[str] = set()
    for members in by_family.values():
        order = sorted(members, key=lambda p: hashlib.sha256(f"family-cap:{p.uid}".encode()).hexdigest())
        capped.update(p.uid for p in order[max_per_family:])
    return ([p for p in problems if p.uid not in capped],
            [{"uid": p.uid, "reason": "family_cap", "k": None} for p in problems if p.uid in capped])


def chars_as_tokens(text: str) -> int:
    """A reply's length in tokens estimated from its characters (rewards.length.est_chars_per_token)."""
    return round(len(text) / 3.2)


def _grade(p: Problem, generate_fn: Callable[..., list[str]], grade_fn: GradeFn, cfg: DifficultyConfig,
           length_fn: LengthFn) -> tuple[int, int]:
    """(replies right, replies right within ``cfg.short_budget`` tokens); the two are equal when it is off. The pass
    rate within each of ``cfg.extra_budgets`` is written on the problem here."""
    seed = derive_seed(cfg.seed, "difficulty", p.uid)
    outs = generate_fn(messages_for_problem(p), n=cfg.n_samples, effort=cfg.effort, max_tokens=cfg.max_tokens,
                       seed=seed)
    thr = cfg.correct_threshold.get(p.task, 1.0)
    right = [grade_fn(p, o) >= thr for o in outs]
    if not (cfg.short_budget or cfg.extra_budgets):
        return sum(right), sum(right)
    lengths = [length_fn(o) if r else 0 for r, o in zip(right, outs)]      # only right replies need a length
    for budget in cfg.extra_budgets:
        p.meta[cfg.budget_key(budget)] = sum(r and n <= budget for r, n in zip(right, lengths)) / cfg.n_samples
    if not cfg.short_budget:
        return sum(right), sum(right)
    return sum(right), sum(r and n <= cfg.short_budget for r, n in zip(right, lengths))


def _rates(p: Problem, k: int, k_short: int, cfg: DifficultyConfig) -> int:
    """Write the pass rate(s) on the problem; return the count that decides "solved n/n" (the short one if on)."""
    n = cfg.n_samples
    p.meta[f"pass_at_{n}_{cfg.effort}"] = k / n
    if cfg.short_budget:
        p.meta[cfg.short_key()] = k_short / n
        return k_short
    return k


def calibrate(
    problems: list[Problem],
    generate_fn: Callable[..., list[str]],
    grade_fn: GradeFn,
    cfg: DifficultyConfig,
    length_fn: LengthFn = chars_as_tokens,
) -> DifficultyResult:
    """Sample ``cfg.n_samples`` responses per problem, grade them, and assign pools.

    Writes ``meta.pass_at_8_medium`` (fraction correct) and ``meta.pool`` on kept problems.
    """

    def _one(p: Problem) -> tuple[Problem, int, int]:
        if p.task in cfg.skip_tasks:
            return p, -1, -1
        return (p, *_grade(p, generate_fn, grade_fn, cfg, length_fn))

    with ThreadPoolExecutor(max_workers=max(1, cfg.max_concurrency)) as ex:
        results = list(ex.map(_one, problems))

    res = DifficultyResult(kept=[])
    hist: dict[str, Counter] = {}
    for p, k, k_short in results:
        if k < 0:  # skipped task: keep as main pool, no pass rate
            p.meta.setdefault("pool", "main")
            res.kept.append(p)
            continue
        hist.setdefault(p.task, Counter())[k] += 1
        solved = _rates(p, k, k_short, cfg)
        if cfg.drop_all_correct and solved == cfg.n_samples:
            res.dropped.append({"uid": p.uid, "reason": "all_correct", "k": k})
            continue
        p.meta["pool"] = "hard" if (cfg.hard_pool_if_none_correct and k == 0) else "main"
        res.kept.append(p)
    res.histogram = {t: dict(c) for t, c in hist.items()}
    log.info("difficulty: %s", res.stats())
    return res


# =============================================================================================
# Audit mode: calibrate a sample of each template, decide for the template (plan §6, 2026-10-23)
# =============================================================================================
def difficulty_group(p: Problem) -> str:
    """Problems expected to be about equally hard: one T1 template, one T2 kind or physics family, one T4
    source (MATH by level, instruction prompts by number of constraints), one T3 topic."""
    m, src = p.meta, p.source
    g = m.get("generator")
    if isinstance(g, dict):
        return f"{p.task}:{m.get('family') or g.get('name')}:t{m.get('template', 0)}"
    if p.task == T2_CODE:
        return f"{p.task}:{m.get('kind') or 'code'}:{m.get('family') or ''}"
    if p.task == T4_GENERAL:
        s = m.get("t4_source") or src.get("dataset") or "t4"
        if p.answer.get("kind") == "ifeval":
            return f"{p.task}:{s}:{len(p.answer.get('constraints') or [])}c"
        return f"{p.task}:{s}:{m['level']}" if m.get("level") else f"{p.task}:{s}"
    if p.task == T3_REPORT:
        return f"{p.task}:{m.get('topic', '')}"
    return f"{p.task}:{src.get('dataset') or src.get('origin') or 'unknown'}"


@dataclass
class DifficultyAuditConfig:
    """Calibrate ``per_group`` instances of each group instead of every problem (~20× less base sampling).

    A group is dropped when at least ``drop_share`` of its audited instances are solved n/n (the template is
    trivial for the base), and goes to the hard pool when at least ``hard_share`` are solved 0/n. Otherwise
    the group stays: audited instances keep their own result (n/n dropped, 0/n hard), the others go to the
    main pool with the group's mean as their estimate. Easy instances that slip through cost rollouts but
    no gradient: the RL loop drops zero-advantage groups online.
    """

    per_group: int = 20
    drop_share: float = 0.9
    hard_share: float = 0.9

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "DifficultyAuditConfig":
        d = dict(d or {})
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def calibrate_audit(
    problems: list[Problem],
    generate_fn: Callable[..., list[str]],
    grade_fn: GradeFn,
    cfg: DifficultyConfig,
    acfg: DifficultyAuditConfig,
    length_fn: LengthFn = chars_as_tokens,
) -> tuple[DifficultyResult, dict[str, dict[str, Any]]]:
    """Like :func:`calibrate`, on a sample per :func:`difficulty_group`. Returns the result and a group report."""
    import hashlib

    groups: dict[str, list[Problem]] = {}
    for p in problems:
        if p.task not in cfg.skip_tasks:
            groups.setdefault(difficulty_group(p), []).append(p)
    rank = lambda p: hashlib.sha256(f"{cfg.seed}:{p.uid}".encode()).hexdigest()  # noqa: E731
    audited = [p for g in groups.values() for p in sorted(g, key=rank)[:acfg.per_group]]
    both = _count_correct(audited, generate_fn, grade_fn, cfg, length_fn)
    ks = {u: k for u, (k, _) in both.items()}
    solved_n = {u: k_short for u, (_, k_short) in both.items()}     # "solved n/n" uses the short budget if on

    n = cfg.n_samples
    report: dict[str, dict[str, Any]] = {}
    for key, members in sorted(groups.items()):
        k = [ks[p.uid] for p in members if p.uid in ks]
        share_all = sum(solved_n[p.uid] == n for p in members if p.uid in ks) / len(k)
        share_none = sum(x == 0 for x in k) / len(k)
        decision = ("drop" if share_all >= acfg.drop_share else "hard" if share_none >= acfg.hard_share else "keep")
        report[key] = {"size": len(members), "audited": len(k), "mean_pass": round(sum(k) / (n * len(k)), 4),
                       "share_all_correct": round(share_all, 4), "share_none_correct": round(share_none, 4),
                       "decision": decision}

    res = DifficultyResult(kept=[])
    hist: dict[str, Counter] = {}
    for p in problems:
        if p.task in cfg.skip_tasks:
            p.meta.setdefault("pool", "main")
            res.kept.append(p)
            continue
        key = difficulty_group(p)
        g = report[key]
        k = ks.get(p.uid)
        solved = None
        if k is not None:
            hist.setdefault(p.task, Counter())[k] += 1
            solved = _rates(p, k, solved_n[p.uid], cfg)
        else:
            p.meta[f"pass_at_{n}_{cfg.effort}"] = g["mean_pass"]          # the group's estimate
        p.meta["difficulty_audit"] = {"group": key, "decision": g["decision"], "audited": k is not None}
        if g["decision"] == "drop" or (cfg.drop_all_correct and solved == n):
            res.dropped.append({"uid": p.uid, "reason": "all_correct" if k == n else "group_all_correct",
                                "group": key, "k": k})
            continue
        hard = g["decision"] == "hard" or (cfg.hard_pool_if_none_correct and k == 0)
        p.meta["pool"] = "hard" if hard else "main"
        res.kept.append(p)
    res.histogram = {t: dict(c) for t, c in hist.items()}
    log.info("difficulty (audit): %d groups, %s; %s", len(report),
             dict(Counter(g["decision"] for g in report.values())), res.stats())
    return res, report


# =============================================================================================
# Two-stage mode (audit_all): audit each group, then sample every problem of the groups that survive
# =============================================================================================
def _count_correct(problems: list[Problem], generate_fn: Callable[..., list[str]], grade_fn: GradeFn,
                   cfg: DifficultyConfig, length_fn: LengthFn = chars_as_tokens) -> dict[str, tuple[int, int]]:
    """uid → (replies graded correct, of those within ``cfg.short_budget``), same seeds and settings everywhere."""
    with ThreadPoolExecutor(max_workers=max(1, cfg.max_concurrency)) as ex:
        return dict(zip((p.uid for p in problems),
                        ex.map(lambda p: _grade(p, generate_fn, grade_fn, cfg, length_fn), problems)))


def calibrate_two_stage(
    problems: list[Problem],
    generate_fn: Callable[..., list[str]],
    grade_fn: GradeFn,
    cfg: DifficultyConfig,
    acfg: DifficultyAuditConfig,
    length_fn: LengthFn = chars_as_tokens,
) -> tuple[DifficultyResult, dict[str, dict[str, Any]]]:
    """Per-problem pass rates without sampling templates the model has already mastered (mode ``audit_all``).

    Stage 1 is :func:`calibrate_audit`'s audit: ``per_group`` instances of each group. A group with at least
    ``drop_share`` of them solved n/n is dropped whole, because sampling its other members would mostly confirm
    "8/8". Stage 2 samples every remaining member of the other groups, so each kept problem carries its *own*
    pass rate (what RL's difficulty weights need) instead of the group mean. A problem solved n/n is dropped,
    one solved 0/n goes to the hard pool, as in :func:`calibrate`. There is no whole-group "hard" decision:
    every member is measured. Replies are cached per uid, so the audited instances are not sampled twice.
    """
    import hashlib

    groups: dict[str, list[Problem]] = {}
    for p in problems:
        if p.task not in cfg.skip_tasks:
            groups.setdefault(difficulty_group(p), []).append(p)
    rank = lambda p: hashlib.sha256(f"{cfg.seed}:{p.uid}".encode()).hexdigest()  # noqa: E731
    audited = [p for g in groups.values() for p in sorted(g, key=rank)[:acfg.per_group]]
    both = _count_correct(audited, generate_fn, grade_fn, cfg, length_fn)

    n = cfg.n_samples
    report: dict[str, dict[str, Any]] = {}
    for key, members in sorted(groups.items()):
        k = [both[p.uid][0] for p in members if p.uid in both]
        share_all = sum(both[p.uid][1] == n for p in members if p.uid in both) / len(k)   # mastered within budget
        share_none = sum(x == 0 for x in k) / len(k)
        report[key] = {"size": len(members), "audited": len(k), "mean_pass_audit": round(sum(k) / (n * len(k)), 4),
                       "share_all_correct": round(share_all, 4), "share_none_correct": round(share_none, 4),
                       "decision": "drop" if share_all >= acfg.drop_share else "keep"}
    rest = [p for key, members in groups.items() if report[key]["decision"] == "keep"
            for p in members if p.uid not in both]
    log.info("difficulty (audit_all): %d groups, %d dropped whole; sampling %d more problems",
             len(report), sum(g["decision"] == "drop" for g in report.values()), len(rest))
    both.update(_count_correct(rest, generate_fn, grade_fn, cfg, length_fn))
    ks = {u: k for u, (k, _) in both.items()}

    res = DifficultyResult(kept=[])
    hist: dict[str, Counter] = {}
    for p in problems:
        if p.task in cfg.skip_tasks:
            p.meta.setdefault("pool", "main")
            res.kept.append(p)
            continue
        key = difficulty_group(p)
        g, k = report[key], ks.get(p.uid)
        p.meta["difficulty_audit"] = {"group": key, "decision": g["decision"], "measured": k is not None}
        if g["decision"] == "drop":
            res.dropped.append({"uid": p.uid, "reason": "all_correct" if k == n else "group_all_correct",
                                "group": key, "k": k})
            continue
        hist.setdefault(p.task, Counter())[k] += 1
        solved = _rates(p, k, both[p.uid][1], cfg)
        if cfg.drop_all_correct and solved == n:
            res.dropped.append({"uid": p.uid, "reason": "all_correct", "group": key, "k": k})
            continue
        p.meta["pool"] = "hard" if (cfg.hard_pool_if_none_correct and k == 0) else "main"
        res.kept.append(p)
    for key, g in report.items():           # the kept groups' pass rate over every member, after stage 2
        if g["decision"] == "keep":
            k_all = [ks[p.uid] for p in groups[key]]
            g["mean_pass"] = round(sum(k_all) / (n * len(k_all)), 4)
    res.histogram = {t: dict(c) for t, c in hist.items()}
    log.info("difficulty (audit_all): %s", res.stats())
    return res, report
