"""Ship gates and the RL-entry gate, evaluated mechanically from ``configs/project.yaml``.

The gates were fixed before the project started (report §目标与成功标准),
and each must clear the evaluation noise floor. This module never decides
*what* the thresholds are; it only reads them. Changing a gate means
editing project.yaml in its own commit, before any test-set result exists.

Gate rules supported (``rule`` field):

==================  ===========================================================================
rule                meaning
==================  ===========================================================================
``>=`` + ``ref``    ``cand - ref >= delta`` **and** the gain exceeds the combined CI half-width
``>=`` + ``value``  ``cand >= value``; for win rates the lower CI bound must also exceed 50
``gain_ratio>=``    ``(cand - ref) / (cand[of] - ref[of]) >= value`` (D-Fresh generalization)
``abs<=``           ``|cand - ref| <= value`` (refusal-rate change, percentage points)
``<=``              ratio-to-ref or absolute metric ``<= value`` (cost, seed reproducibility)
==================  ===========================================================================

Usage::

    python -m dpt.eval.gates --summary eval_results/summary.json --candidate rl_seed1 --seed2 rl_seed2
    python -m dpt.eval.gates --summary eval_results/summary.json --candidate sft --entry   # RL-entry gate

Exit code 0 means every gate passed, 1 means at least one failed, and 2
means an input is missing. With ``rl_entry_gate.on_fail: hold`` (the 9B line),
a failed entry gate first waits for a person's ``go`` or ``stop`` in
``rl_entry_gate.decision_file`` (``hold_for_decision``).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from dpt.common.io import read_json
from dpt.eval.stats import combined_halfwidth


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    observed: str
    requirement: str
    note: str = ""


class MissingInput(KeyError):
    pass


def _score(rows: dict[str, Any], row: str, key: str) -> tuple[float, float]:
    try:
        r = rows[row][key]
    except KeyError as e:
        raise MissingInput(f"summary has no {key!r} for row {row!r}") from e
    return float(r["score"]), float(r.get("ci_halfwidth") or 0.0)


def evaluate_ship_gates(
    summary: dict[str, Any], gates: dict[str, Any], candidate: str, seed2: str | None = None
) -> list[GateResult]:
    rows = summary["rows"]
    out: list[GateResult] = []
    for name, g in gates.items():
        rule = g["rule"]
        try:
            if name == "general":
                ref = g.get("ref", "base_medium")
                if ref != summary.get("base_row", "base_medium"):
                    raise MissingInput("general_mean_delta is computed against base_medium only")
                v = rows[candidate].get("general_mean_delta")
                if v is None:
                    raise MissingInput("candidate has no general_mean_delta")
                ok = v >= g["value"]
                out.append(
                    GateResult(
                        name,
                        ok,
                        f"{v:+.2f}",
                        f">= {g['value']:+.1f}",
                        f"over {len(rows[candidate].get('general_benchmarks', []))} benchmarks",
                    )
                )
            elif name == "safety":
                c, b = rows[candidate].get("refusal_rate"), rows[g["ref"]].get("refusal_rate")
                if c is None or b is None:
                    raise MissingInput("refusal_rate missing")
                d = c - b
                out.append(GateResult(name, abs(d) <= g["value"], f"{d:+.1f} pp", f"|Δ| <= {g['value']} pp"))
            elif name == "cost":
                c, b = rows[candidate].get("mean_output_tokens"), rows[g["ref"]].get("mean_output_tokens")
                if not c or not b:
                    raise MissingInput("mean_output_tokens missing")
                r = c / b
                out.append(
                    GateResult(
                        name,
                        r <= g["value"],
                        f"{r:.2f}×",
                        f"<= {g['value']}× {g['ref']}",
                        f"{c:.0f} vs {b:.0f} tokens",
                    )
                )
            elif name == "repro":
                if seed2 is None:
                    raise MissingInput("repro gate needs --seed2")
                a, _ = _score(rows, candidate, "d_num")
                b, _ = _score(rows, seed2, "d_num")
                out.append(
                    GateResult(name, abs(a - b) <= g["value"], f"{abs(a - b):.2f}", f"<= {g['value']}")
                )
            elif rule == "gain_ratio>=":
                c, _ = _score(rows, candidate, name)
                r0, _ = _score(rows, g["ref"], name)
                oc, _ = _score(rows, candidate, g["of"])
                or0, _ = _score(rows, g["ref"], g["of"])
                denom = oc - or0
                ratio = (c - r0) / denom if denom > 0 else float("-inf")
                out.append(
                    GateResult(
                        name,
                        ratio >= g["value"],
                        f"{ratio:.0%}",
                        f">= {g['value']:.0%} of {g['of']} gain vs {g['ref']}",
                        f"{c - r0:+.1f} / {denom:+.1f}",
                    )
                )
            elif rule == ">=" and "ref" in g:
                c, hc = _score(rows, candidate, name)
                r0, hr = _score(rows, g["ref"], name)
                gain, band = c - r0, combined_halfwidth(hc, hr)
                ok = gain >= g.get("delta", 0.0) and gain > band
                out.append(
                    GateResult(
                        name,
                        ok,
                        f"{gain:+.1f} ({c:.1f} vs {r0:.1f})",
                        f">= {g['ref']} {g.get('delta', 0.0):+.1f}",
                        f"noise band ±{band:.1f}",
                    )
                )
            elif rule == ">=":
                c, hc = _score(rows, candidate, name)
                beyond = (c - hc) > 50.0 if g.get("metric", "").startswith("winrate") else True
                ok = c >= g["value"] and beyond
                out.append(
                    GateResult(
                        name,
                        ok,
                        f"{c:.1f} ±{hc:.1f}",
                        f">= {g['value']}",
                        "lower CI bound must exceed 50" if g.get("metric", "").startswith("winrate") else "",
                    )
                )
            else:
                out.append(GateResult(name, False, "—", rule, "unsupported rule"))
        except MissingInput as e:
            out.append(GateResult(name, False, "missing", g.get("rule", ""), str(e)))
    return out


def evaluate_entry_gate(
    summary: dict[str, Any], gate: dict[str, Any], candidate: str, base: str = "base_medium"
) -> list[GateResult]:
    """RL-entry gate (report §进入 RL 的门槛): every check must hold to start stage 2.

    ``domain_beats_base`` sets must score strictly above base. ``domain_not_below_base`` sets, for a set the
    base nearly saturates, may not fall below base by more than the combined 95% CI half-width. pass@16 may
    trail base by ``pass_at_16_tolerance`` points (default 0).
    """
    rows = summary["rows"]
    out: list[GateResult] = []
    for key in gate.get("domain_beats_base", []):
        try:
            c, hc = _score(rows, candidate, key)
            b, hb = _score(rows, base, key)
            out.append(
                GateResult(
                    f"entry:{key}",
                    c > b,
                    f"{c:.1f} vs {b:.1f}",
                    f"> {base}",
                    f"{'beyond' if c - b > combined_halfwidth(hc, hb) else 'within'} noise",
                )
            )
        except MissingInput as e:
            out.append(GateResult(f"entry:{key}", False, "missing", f"> {base}", str(e)))
    for key in gate.get("domain_not_below_base", []):
        try:
            c, hc = _score(rows, candidate, key)
            b, hb = _score(rows, base, key)
            hw = combined_halfwidth(hc, hb)
            out.append(
                GateResult(
                    f"entry:{key}", c >= b - hw, f"{c:.1f} vs {b:.1f}", f">= {base} - {hw:.1f}", "base near ceiling"
                )
            )
        except MissingInput as e:
            out.append(GateResult(f"entry:{key}", False, "missing", f">= {base} - noise", str(e)))
    v = rows.get(candidate, {}).get("general_mean_delta")
    lim = gate.get("general_mean_delta_min", -1.5)
    out.append(
        GateResult(
            "entry:general",
            v is not None and v >= lim,
            "missing" if v is None else f"{v:+.2f}",
            f">= {lim:+.1f}",
        )
    )
    if gate.get("pass_at_16_not_below_base", False):
        tol = float(gate.get("pass_at_16_tolerance", 0.0))
        try:
            c = float(rows[candidate]["pass_at_k"]["d_num"]["16"])
            b = float(rows[base]["pass_at_k"]["d_num"]["16"])
            out.append(
                GateResult(
                    "entry:pass@16",
                    c >= b - tol,
                    f"{c:.1f} vs {b:.1f}",
                    f">= {base}" + (f" - {tol:g}" if tol else ""),
                    "RL needs exploration headroom",
                )
            )
        except KeyError:
            out.append(
                GateResult(
                    "entry:pass@16",
                    False,
                    "missing",
                    f">= {base}",
                    "run d_num with --samples-override 16 (or 64) for both rows",
                )
            )
    return out


def hold_for_decision(
    gate: dict[str, Any], *, poll_s: float = 60.0, clock: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> int:
    """After a failed RL-entry gate, wait up to ``hold_hours`` for a person's answer in ``decision_file``.

    The file holds one word: ``go`` starts RL anyway (exit 0) and ``stop`` ends the job (exit 1), as does no
    answer in time. Only a file written after the hold began counts (a minute of slack for clock skew between
    nodes), so an old answer cannot pass a new gate.
    """
    path, hours = Path(gate["decision_file"]), float(gate.get("hold_hours", 8.0))
    clock, sleep = clock or time.time, sleep or time.sleep
    start = clock()
    print(f"\nthe RL-entry gate failed: RL is held for up to {hours:g} h until {path} says go or stop", flush=True)
    while clock() - start < hours * 3600:
        try:
            word = path.read_text().strip().lower() if path.stat().st_mtime >= start - 60 else ""
        except OSError:
            word = ""
        if word in ("go", "stop"):
            print(f"answer in {path}: {word}", flush=True)
            return 0 if word == "go" else 1
        sleep(poll_s)
    print(f"no answer in {path} after {hours:g} h: not starting RL", flush=True)
    return 1


def format_table(results: Sequence[GateResult]) -> str:
    lines = ["| gate | result | observed | requirement | note |", "|---|---|---|---|---|"]
    for r in results:
        lines.append(
            f"| {r.name} | {'PASS' if r.passed else 'FAIL'} | {r.observed} | {r.requirement} | {r.note} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate ship gates / RL-entry gate")
    ap.add_argument("--summary", required=True, help="summary.json from dpt.eval.report")
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--seed2", default=None)
    ap.add_argument("--entry", action="store_true", help="evaluate the RL-entry gate instead of ship gates")
    args = ap.parse_args(argv)

    from dpt.common.config import load_config, to_container

    project = to_container(load_config())["project"]
    summary = read_json(args.summary)
    if args.candidate not in summary["rows"]:
        print(f"candidate row {args.candidate!r} not in summary")
        return 2
    res = (
        evaluate_entry_gate(summary, project["rl_entry_gate"], args.candidate)
        if args.entry
        else evaluate_ship_gates(summary, project["gates"], args.candidate, args.seed2)
    )
    print(format_table(res), flush=True)        # flushed: campaign.sh tees it, and a hold may follow
    n_fail = sum(not r.passed for r in res)
    print(f"\n{len(res) - n_fail}/{len(res)} gates passed", flush=True)
    if n_fail and args.entry and project["rl_entry_gate"].get("on_fail") == "hold":
        return hold_for_decision(project["rl_entry_gate"])
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
