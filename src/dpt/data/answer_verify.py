"""Answer verification. Is the reference answer itself right? (Report §评测体系, 第一层.)

A wrong label is worse than no label. It pushes the policy toward a wrong
answer during RL, and in eval it hides real gains or invents fake ones. Every
numeric problem therefore passes four checks before it enters the bank:

1. **Three-way agreement.** The programmatic solver and two strong models from
   *different families* (8 samples each) must all agree with the label. A
   family "votes" only if at least ``min_family_agreement`` of its samples
   agree with each other (default 5/8). That filters out noise from a single
   lucky or unlucky sample.
2. **Disagreement → expert queue.** Nothing is silently relabeled. In the
   report, about 40% of the 6.8% rejected were label errors and the rest were
   ambiguous wording, so a human must look.
3. **"Everyone against the label."** If both families, and the solver if
   present, agree on one value that differs from the label, the label is the
   prime suspect. Such items get ``reason="suspect_label"`` and high priority
   in the queue.
4. **Perturbation consistency** for programmatic problems. Nudge each input
   and the answer must move as the physics says. If the problem declares
   power-law ``scaling`` exponents, check them. Otherwise check that the
   answer is sensitive to every non-invariant input and that the local slope
   is the same from both sides (smoothness). An answer that does not change,
   or changes erratically, points to a bug in the problem template or the
   solver.

Answer parsing here is deliberately independent of the training graders in
``dpt.rewards``. Verifying labels with the same parser that will later grade
the policy would hide parser bugs in both places.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

from dpt.chat.thinking import THINK_END, strip_thinking
from dpt.common.types import T1_NUM, T4_GENERAL
from dpt.data.schema import Problem
from dpt.data.solvers import solve

# =============================================================================================
# Parsing \boxed{<number> <unit>}
# =============================================================================================
_NUM = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?"
_SCI = re.compile(
    rf"^\s*(?P<mant>{_NUM})\s*(?:\\times|×|x|\*)\s*10\s*\^\s*\{{?\s*(?P<exp>[-+]?\d+)\s*\}}?"
)
_PLAIN = re.compile(rf"^\s*(?P<num>{_NUM})")
_LATEX_WRAP = re.compile(r"\\(?:mathrm|text|textrm|operatorname)\s*\{([^{}]*)\}")


def last_boxed(text: str) -> str | None:
    """Content of the last ``\\boxed{...}`` (brace-balanced), or ``None``."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i, depth = start + len("\\boxed{"), 1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i:j]
    return None


def parse_quantity(s: str) -> tuple[float, str] | None:
    """Parse ``"3.2 km"``, ``"3.2\\times10^{-4} m"``, ``"1,200 s"`` → (value, unit). ``None`` if no number."""
    s = s.strip().replace("$", "")
    m = _SCI.match(s)
    if m:
        value = float(m.group("mant").replace(",", "")) * 10.0 ** int(m.group("exp"))
        rest = s[m.end():]
    else:
        m = _PLAIN.match(s)
        if not m:
            return None
        value = float(m.group("num").replace(",", ""))
        rest = s[m.end():]
    unit = _LATEX_WRAP.sub(r"\1", rest)
    unit = unit.replace("\\,", " ").replace("\\ ", " ").replace("~", " ").strip().strip("{}").strip()
    if not math.isfinite(value):
        return None
    return value, unit


# A small SI table: unit → (dimension, factor to the SI base). Enough to compare model answers
# with labels in verification. The training grader has its own, richer implementation.
_BASE = {"m": "length", "s": "time", "g": "mass", "Hz": "frequency", "Pa": "pressure",
         "N": "force", "J": "energy", "W": "power", "N·m": "moment", "N m": "moment", "m/s": "velocity",
         "m²": "area", "m^2": "area"}
_PREFIX = {"": 1.0, "k": 1e3, "M": 1e6, "G": 1e9, "c": 1e-2, "m": 1e-3, "µ": 1e-6, "u": 1e-6, "n": 1e-9}
_EXTRA = {"min": ("time", 60.0), "h": ("time", 3600.0), "km/s": ("velocity", 1e3),
          "km/h": ("velocity", 1 / 3.6), "kg": ("mass", 1.0), "dyn·cm": ("moment", 1e-7),
          "dyne-cm": ("moment", 1e-7)}


def _unit_info(unit: str) -> tuple[str, float] | None:
    u = unit.strip()
    if u in _EXTRA:
        return _EXTRA[u]
    if u in _BASE:
        dim = _BASE[u]
        return (dim, 1e-3) if u == "g" else (dim, 1.0)
    for pfx, f in _PREFIX.items():
        if pfx and u.startswith(pfx) and u[len(pfx):] in _BASE:
            base = u[len(pfx):]
            return (_BASE[base], f * (1e-3 if base == "g" else 1.0))
    return None


def to_label_unit(value: float, unit: str, label_unit: str) -> float | None:
    """Convert ``value [unit]`` into ``label_unit``; ``None`` if the units are incompatible or unknown.

    An answer with no unit is read as already being in the label unit. If both
    unit strings are equal, no conversion is attempted.
    """
    if not unit or unit == label_unit:
        return value
    a, b = _unit_info(unit), _unit_info(label_unit)
    if a is None or b is None or a[0] != b[0]:
        return None
    return value * a[1] / b[1]


def values_match(a: float, b: float, *, rel_tol: float, abs_tol: float) -> bool:
    return abs(a - b) <= max(abs_tol, rel_tol * abs(b))


def unmarked_reply(reply: str) -> str:
    """A reply from a model family that does not mark the end of its thinking: Gemma, served without a
    reasoning parser, prints its thinking as plain text ("thought\n...") and never writes ``</think>``. All
    of the reply then counts as the answer, so :func:`extract_value` reads its last ``\boxed`` (as
    ``dpt.eval.autocheck`` reads Gemma). For the Qwen teacher a missing ``</think>`` still means cut off."""
    return reply if THINK_END in reply else f"{THINK_END}\n\n{reply}"


def extract_value(response: str, label_unit: str) -> float | None:
    """Final-answer value of a model response in the label unit (``None`` = unparseable / wrong dimension)."""
    final = strip_thinking(response)
    if final is None:
        return None
    boxed = last_boxed(final)
    if boxed is None:
        return None
    q = parse_quantity(boxed)
    if q is None:
        return None
    return to_label_unit(q[0], q[1], label_unit)


# =============================================================================================
# Three-way agreement
# =============================================================================================
@dataclass
class VerifyConfig:
    family_samples: int = 8
    min_family_agreement: float = 0.625
    require_solver: dict[str, bool] = field(default_factory=lambda: {T1_NUM: True, T4_GENERAL: False})

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "VerifyConfig":
        d = dict(d or {})
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class VerifyOutcome:
    uid: str
    status: str                  # "accepted" | "rejected" | "not_applicable"
    reason: str                  # agree | suspect_label | solver_mismatch | family_undecided | disagreement
    votes: dict[str, float | None] = field(default_factory=dict)
    shares: dict[str, float] = field(default_factory=dict)
    priority: str = "normal"     # "high" for suspect_label (label is the likely culprit)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def family_majority(
    values: Sequence[float | None], *, rel_tol: float, abs_tol: float
) -> tuple[float | None, float]:
    """Largest cluster of mutually-matching values and its share of *all* samples (``None`` = miss)."""
    clusters: list[list[float]] = []
    for v in values:
        if v is None:
            continue
        for c in clusters:
            if values_match(v, c[0], rel_tol=rel_tol, abs_tol=abs_tol):
                c.append(v)
                break
        else:
            clusters.append([v])
    if not clusters or not values:
        return None, 0.0
    best = max(clusters, key=len)
    return float(sorted(best)[len(best) // 2]), len(best) / len(values)


def verify_numeric(
    problem: Problem,
    family_responses: dict[str, Sequence[str]],
    cfg: VerifyConfig,
) -> VerifyOutcome:
    """Run solver + family agreement for one numeric problem."""
    a = problem.answer
    label, unit = float(a["value"]), a.get("unit", "")
    rel, abs_ = float(a.get("rel_tol", 1e-2)), float(a.get("abs_tol", 0.0))
    votes: dict[str, float | None] = {}
    shares: dict[str, float] = {}

    gen = problem.meta.get("generator")
    if gen:
        try:
            votes["solver"] = solve(gen)
        except Exception as e:  # a broken solver is a pipeline bug → expert queue, never silent accept
            return VerifyOutcome(problem.uid, "rejected", f"solver_error:{type(e).__name__}", votes, shares)
    elif cfg.require_solver.get(problem.task, False):
        return VerifyOutcome(problem.uid, "rejected", "no_solver", votes, shares)

    for fam, responses in family_responses.items():
        vals = [extract_value(r, unit) for r in responses]
        maj, share = family_majority(vals, rel_tol=rel, abs_tol=abs_)
        shares[fam] = round(share, 4)
        votes[fam] = maj if share >= cfg.min_family_agreement else None

    def ok(v: float | None) -> bool:
        return v is not None and values_match(v, label, rel_tol=rel, abs_tol=abs_)

    if votes and all(ok(v) for v in votes.values()):
        return VerifyOutcome(problem.uid, "accepted", "agree", votes, shares)

    fam_votes = [votes[f] for f in family_responses]
    decided = [v for v in fam_votes if v is not None]
    if fam_votes and len(decided) == len(fam_votes):
        ref = decided[0]
        consensus = all(values_match(v, ref, rel_tol=rel, abs_tol=abs_) for v in decided)
        solver = votes.get("solver")
        if solver is not None:
            consensus = consensus and values_match(solver, ref, rel_tol=rel, abs_tol=abs_)
        if consensus and not ok(ref):
            return VerifyOutcome(problem.uid, "rejected", "suspect_label", votes, shares, priority="high")
    if "solver" in votes and not ok(votes["solver"]):
        return VerifyOutcome(problem.uid, "rejected", "solver_mismatch", votes, shares)
    if len(decided) < len(fam_votes):
        return VerifyOutcome(problem.uid, "rejected", "family_undecided", votes, shares)
    return VerifyOutcome(problem.uid, "rejected", "disagreement", votes, shares)


def verify_problem(
    problem: Problem,
    family_fns: dict[str, Callable[[Problem, int], list[str]]],
    cfg: VerifyConfig,
) -> VerifyOutcome:
    """Dispatch by task. Only numeric problems carry a checkable scalar label.

    T2 labels are hidden tests, validated on the sandbox cluster by running the
    reference solution and mutants. T3 has rubrics, reviewed by experts. T4
    ifeval constraints are executable rules. Those return ``not_applicable``
    here and keep their own validation upstream.
    """
    if not problem.is_numeric:
        return VerifyOutcome(problem.uid, "not_applicable", "non_numeric_task")
    responses = {name: fn(problem, cfg.family_samples) for name, fn in family_fns.items()}
    out = verify_numeric(problem, responses, cfg)
    if out.status == "accepted" and problem.meta.get("generator"):
        pert = perturbation_consistency(problem.meta["generator"])
        if not pert.ok:
            return VerifyOutcome(problem.uid, "rejected", "perturbation:" + ",".join(pert.issues),
                                 out.votes, out.shares)
    return out


# =============================================================================================
# Audit mode: verify a sample of each template, keep or drop the template as a whole
# =============================================================================================
@dataclass
class AuditConfig:
    """Per-template verification (plan §6): ``per_group`` instances of each template, not every problem.

    A generated problem's label comes from its solver, so what can be wrong is the template (ambiguous
    wording, a solver bug), and it is wrong for every instance alike. Verifying 20 instances per template
    finds that at a few percent of the cost of verifying all of them. A template is dropped when the two
    verification families agree with each other on another value too often (``max_contested``: wording or
    solver at fault), when they confirm the label too rarely (``min_accept``; low on purpose, because hard
    multi-step problems are also missed by the verifiers), or when the perturbation check fails.
    """

    per_group: int = 20
    min_accept: float = 0.3
    max_contested: float = 0.15
    max_perturbation: float = 0.0
    seed: int = 1

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "AuditConfig":
        d = dict(d or {})
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def audit_group(problem: Problem) -> str:
    """Template of a generated problem (family + template index); the source dataset otherwise."""
    m, src = problem.meta, problem.source
    if m.get("generator"):
        return f"{problem.task}:{m.get('family') or m['generator'].get('name')}:t{m.get('template', 0)}"
    return f"{problem.task}:{src.get('dataset') or src.get('origin') or 'unknown'}"


def contested(problem: Problem, out: VerifyOutcome) -> bool:
    """Every verification family settled on one value, and it is not the label."""
    a = problem.answer
    rel, abs_ = float(a.get("rel_tol", 1e-2)), float(a.get("abs_tol", 0.0))
    fam = [v for k, v in out.votes.items() if k != "solver"]
    if len(fam) < 2 or any(v is None for v in fam):
        return False
    same = all(values_match(v, fam[0], rel_tol=rel, abs_tol=abs_) for v in fam)
    return same and not values_match(fam[0], float(a["value"]), rel_tol=rel, abs_tol=abs_)


@dataclass
class AuditResult:
    kept: list[Problem]
    removed: list[dict[str, Any]]
    queue: list[dict[str, Any]]
    groups: dict[str, dict[str, Any]]


def audit_verify(problems: Sequence[Problem], family_fns: dict[str, Callable[[Problem, int], list[str]]],
                 cfg: VerifyConfig, acfg: AuditConfig, *, max_workers: int = 16) -> AuditResult:
    import hashlib
    from collections import Counter, defaultdict
    from concurrent.futures import ThreadPoolExecutor

    groups: dict[str, list[Problem]] = defaultdict(list)
    for p in problems:
        if p.is_numeric:
            groups[audit_group(p)].append(p)
    rank = lambda p: hashlib.sha256(f"{acfg.seed}:{p.uid}".encode()).hexdigest()  # noqa: E731
    audited = [p for g in groups.values() for p in sorted(g, key=rank)[:acfg.per_group]]
    with ThreadPoolExecutor(max(1, max_workers)) as ex:
        outcomes = dict(zip((p.uid for p in audited), ex.map(lambda p: verify_problem(p, family_fns, cfg), audited)))

    report: dict[str, dict[str, Any]] = {}
    for key, members in sorted(groups.items()):
        outs = [(p, outcomes[p.uid]) for p in members if p.uid in outcomes]
        n = len(outs)
        acc = sum(o.status == "accepted" for _, o in outs)
        con = sum(contested(p, o) for p, o in outs)
        pert = sum(o.reason.startswith("perturbation") for _, o in outs)
        why = []
        if n and acc / n < acfg.min_accept:
            why.append(f"accept {acc}/{n} < {acfg.min_accept}")
        if n and con / n > acfg.max_contested:
            why.append(f"contested {con}/{n} > {acfg.max_contested}")
        if n and pert / n > acfg.max_perturbation:
            why.append(f"perturbation {pert}/{n}")
        report[key] = {"size": len(members), "audited": n, "accepted": acc, "contested": con,
                       "perturbation": pert, "reasons": dict(Counter(o.reason for _, o in outs)),
                       "kept": not why, "why": "; ".join(why)}

    kept, removed, queue = [], [], []
    for p in problems:
        if not p.is_numeric:
            kept.append(p)
            continue
        key, out = audit_group(p), outcomes.get(p.uid)
        g = report[key]
        if not g["kept"]:
            removed.append({"uid": p.uid, "reason": "verify:template_rejected", "group": key, "why": g["why"]})
            continue
        if out is not None and out.status == "rejected":
            removed.append({"uid": p.uid, "reason": f"verify:{out.reason}", "group": key})
            queue.append(expert_queue_record(p, out))
            continue
        p.meta["verified"] = {"audit": key, "accepted": g["accepted"], "audited": g["audited"]}
        if out is not None:
            p.meta["verified"].update(votes=out.votes, shares=out.shares)
        kept.append(p)
    return AuditResult(kept, removed, queue, report)


# =============================================================================================
# Perturbation consistency
# =============================================================================================
@dataclass
class PerturbationResult:
    ok: bool
    issues: list[str]
    details: dict[str, dict[str, float]]


def perturbation_consistency(
    generator: dict[str, Any],
    *,
    rel_step: float = 0.01,
    exponent_tol: float = 0.05,
    smooth_tol: float = 0.1,
    solver: Callable[[dict[str, Any]], float] = solve,
) -> PerturbationResult:
    """Perturb each numeric input by ±``rel_step`` and check how the solver output responds.

    With ``scaling[p] = k``, the measured elasticity ``ln(f(p(1±δ))/f(p)) / ln(1±δ)``
    must equal ``k`` within ``exponent_tol``. Without it, the answer must change, and the forward and
    backward finite-difference slopes must agree within ``smooth_tol``
    (relative). Parameters listed in ``invariant`` must not change the answer.
    """
    params = dict(generator["params"])
    scaling = dict(generator.get("scaling") or {})
    invariant = set(generator.get("invariant") or [])
    issues: list[str] = []
    details: dict[str, dict[str, float]] = {}
    f0 = solver({**generator, "params": params})
    if not math.isfinite(f0):
        return PerturbationResult(False, ["nonfinite:base"], {})
    for name, val in params.items():
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            continue
        step = rel_step * abs(val) if val != 0 else rel_step
        try:
            f_up = solver({**generator, "params": {**params, name: val + step}})
            f_dn = solver({**generator, "params": {**params, name: val - step}})
        except (ValueError, ZeroDivisionError, OverflowError):
            issues.append(f"solver_error:{name}")
            continue
        details[name] = {"f0": f0, "f_up": f_up, "f_dn": f_dn}
        if not (math.isfinite(f_up) and math.isfinite(f_dn)):
            issues.append(f"nonfinite:{name}")
            continue
        tiny = 1e-12 * max(1.0, abs(f0))
        unchanged = abs(f_up - f0) <= tiny and abs(f_dn - f0) <= tiny
        if name in invariant:
            if not unchanged:
                issues.append(f"depends_on_invariant:{name}")
            continue
        if unchanged:
            issues.append(f"insensitive:{name}")
            continue
        if name in scaling and val != 0 and f0 != 0:
            # Compare *exponents*, not ratios. With δ = 1% a ratio check cannot tell x from x²
            # (1.0100 vs 1.0201), but the measured elasticity ln(f_up/f0)/ln(1+δ) can.
            k = float(scaling[name])
            r_up, r_dn = f_up / f0, f_dn / f0
            if r_up <= 0 or r_dn <= 0:
                issues.append(f"scaling_violation:{name}")
                continue
            k_up = math.log(r_up) / math.log(1 + step / abs(val))
            k_dn = math.log(r_dn) / math.log(1 - step / abs(val))
            details[name].update({"expected_exponent": k, "exponent_up": k_up, "exponent_dn": k_dn})
            if max(abs(k_up - k), abs(k_dn - k)) > exponent_tol * max(1.0, abs(k)):
                issues.append(f"scaling_violation:{name}")
            continue
        s_up, s_dn = (f_up - f0) / step, (f0 - f_dn) / step
        if abs(s_up - s_dn) > smooth_tol * max(abs(s_up), abs(s_dn)):
            # Strong but smooth curvature (b = log10 e / (M̄ − Mc + ΔM/2) near a small denominator) also
            # separates the two slopes. For a smooth function the gap shrinks in proportion to the step; at a
            # kink or jump it does not. Retry with a quarter step before calling it erratic.
            q = step / 4
            try:
                g_up = solver({**generator, "params": {**params, name: val + q}})
                g_dn = solver({**generator, "params": {**params, name: val - q}})
                gap = abs((g_up - f0) / q - (f0 - g_dn) / q)
            except (ValueError, ZeroDivisionError, OverflowError):
                gap = math.inf
            if not gap <= 0.35 * abs(s_up - s_dn):
                issues.append(f"erratic:{name}")
    return PerturbationResult(not issues, issues, details)


def expert_queue_record(problem: Problem, outcome: VerifyOutcome) -> dict[str, Any]:
    """Row for ``expert_queue.jsonl`` (what a domain expert needs to adjudicate one item)."""
    return {
        "uid": problem.uid,
        "task": problem.task,
        "question": problem.question,
        "label": problem.answer,
        "reason": outcome.reason,
        "priority": outcome.priority,
        "votes": outcome.votes,
        "shares": outcome.shares,
        "source": problem.source,
    }
