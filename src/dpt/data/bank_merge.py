"""Bank v2 (2026-11-07): merge verified stage files, apply the atomic T3 rubrics, and check the published bank.

    python -m dpt.data.bank_merge merge --out $V2/stages/04_verify.jsonl --t3-map $V2/t3_atomic_map.jsonl \\
        $V2/stages/04_verify.jsonl $DATA/v1_hard/stages/04_verify.jsonl $DATA/v1_long/stages/04_verify.jsonl
    python -m dpt.data.bank_merge check $V2          # after the difficulty and publish stages

``merge`` reads every input before writing (the output may be one of them), keeps the first row of each uid, applies
the T3 map (dpt.data.gen.t3_atomic.apply_map) and writes a manifest whose parents are the inputs' content hashes.

``check`` prints counts only (tasks, pools, pass rates, rubric formats) and writes ``<bank>/bank_check.json``. It
exits 1 when the bank cannot train: no T1 rows, a T3 rubric that is not atomic, a dev split different from v1's, or
a main pool whose T1 problems are nearly all easy. ``--t3-not-trained`` (RL v2, 2026-11-11) still counts the
non-atomic T3 rubrics but does not fail on them: that run trains no T3 and keeps its rubrics as they are.

The check also estimates the share of T1 groups with mixed results at step 1 (2026-11-11). A problem graded r = k/8
by the SFT model gives a group of 8 with mixed results with probability 1 − r⁸ − (1 − r)⁸, and only such groups carry
gradient. In run 1's steps 1–20, 13% of T1 groups were mixed. Below 40% the bank fails, so an RL job never spends its
nodes on a bank that is too easy. The estimate is optimistic: a problem graded 7/8 may really be at 0.95.

``pin-dev`` (2026-11-11) makes ``<bank>/rl_dev`` the reference split (v1's) and removes its problems from
``rl_main`` and ``rl_hard``. Publish draws the dev split from the problems that survive the difficulty pass. Bank
v2 re-grades them with the SFT model, which drops every problem it solves 8/8 (88% of run 1's T1 groups), so the
drawn split would hold only hard problems, differ from run 1's, and leave the rest of run 1's dev problems in
training. Run 1's dev curve and the pause rules calibrated on it compare only on the same split::

    python -m dpt.data.bank_merge pin-dev --reference-dev $DATA/v1/rl_dev.jsonl $V2

``without-task`` writes a published split without one task's rows, with a manifest, e.g. the dev split without
T3 for a run that has no judge::

    python -m dpt.data.bank_merge without-task --task t3_report --out $V2/rl_dev_no_t3 $V2/rl_dev.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from dpt.common.io import read_jsonl, write_jsonl
from dpt.common.logging import get_logger
from dpt.common.manifest import build_manifest, load_manifest, manifest_path_for, write_manifest
from dpt.data.rl_build import write_rows

log = get_logger(__name__)


def merge(inputs: Sequence[Path], out: Path, t3_map: Path | None = None, version: str = "v2",
          name: str = "rl_bank.verify") -> dict[str, Any]:
    rows, seen, per_input = [], set(), {}
    parents = {str(p): load_manifest(p).content_hash for p in inputs if manifest_path_for(p).exists()}
    for p in inputs:
        n = 0
        for r in read_jsonl(p):
            if r["uid"] in seen:
                continue
            seen.add(r["uid"])
            rows.append(r)
            n += 1
        per_input[str(p)] = n
    dropped: dict[str, int] = {}
    if t3_map is not None:
        from dpt.data.gen.t3_atomic import apply_map

        rows, dropped = apply_map(rows, {r["uid"]: r for r in read_jsonl(t3_map)})
    write_jsonl(out, rows)
    write_manifest(build_manifest(out, name=name, version=version, parents=parents,
                                  notes="bank v2: verified v1 + hard + long problems; T3 rubrics as atomic checks"))
    res = {"rows": len(rows), "per_input": per_input, "t3_dropped": dropped,
           "tasks": dict(Counter(r["task"] for r in rows))}
    log.info("bank merge: %s", res)
    return res


def _rows(path: Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path)) if path.exists() else []


def task_of(row: dict[str, Any]) -> str | None:
    """The task of a published RL row (``extra_info.task``, else verl's ``data_source``)."""
    return (row.get("extra_info") or {}).get("task") or row.get("data_source")


MIN_EXPECTED_MIXED_T1 = 0.40


def expected_mixed_share(pass_rates: Sequence[float], group_size: int = 8) -> float:
    """Expected share of groups with mixed results when each problem is solved with its graded rate r."""
    if not pass_rates:
        return 0.0
    return sum(1.0 - r ** group_size - (1.0 - r) ** group_size for r in pass_rates) / len(pass_rates)


def check(bank: Path, reference_dev: Path | None = None,
          t3_not_trained: bool = False) -> tuple[dict[str, Any], list[str]]:
    main, hard, dev = (_rows(bank / f"rl_{n}.jsonl") for n in ("main", "hard", "dev"))
    problems: list[str] = []

    def rate_of(r):
        return (r.get("extra_info") or {}).get("pass_at_8_medium")

    summary: dict[str, Any] = {
        "main": dict(Counter(task_of(r) for r in main)), "hard": dict(Counter(task_of(r) for r in hard)),
        "dev": dict(Counter(task_of(r) for r in dev)), "pass_rate_bands": {}}
    for t in sorted({task_of(r) for r in main}):
        rates = [rate_of(r) for r in main if task_of(r) == t]
        known = [x for x in rates if isinstance(x, (int, float))]
        summary["pass_rate_bands"][t] = {
            "unknown": len(rates) - len(known), "le_3_of_8": sum(x <= 0.375 for x in known),
            "4_5_of_8": sum(0.375 < x <= 0.625 for x in known), "6_7_of_8": sum(0.625 < x < 1 for x in known),
            "8_of_8": sum(x >= 1 for x in known)}
    t3 = [r for r in main + hard + dev if task_of(r) == "t3_report"]
    not_atomic = 0
    for r in t3:
        gt = (r.get("reward_model") or {}).get("ground_truth")
        gt = json.loads(gt) if isinstance(gt, str) else (gt or {})
        rub = gt.get("rubric") or []
        not_atomic += not (rub and all(isinstance(c, dict) and c.get("check") for c in rub))
    summary["t3_rows"], summary["t3_not_atomic"] = len(t3), not_atomic

    summary["t3_trained"] = not t3_not_trained
    t1_rates = [x for x in (rate_of(r) for r in main if task_of(r) == "t1_num") if isinstance(x, (int, float))]
    summary["expected_mixed_t1_at_step_1"] = round(expected_mixed_share(t1_rates), 3)

    if not summary["main"].get("t1_num"):
        problems.append("no T1 rows in rl_main")
    if not_atomic and not t3_not_trained:
        problems.append(f"{not_atomic} T3 rubrics are not atomic checks")
    if t1_rates and summary["expected_mixed_t1_at_step_1"] < MIN_EXPECTED_MIXED_T1:
        problems.append(f"only {100 * summary['expected_mixed_t1_at_step_1']:.0f}% of T1 groups are expected to have "
                        f"mixed results at step 1 (at least {100 * MIN_EXPECTED_MIXED_T1:.0f}%)")
    if reference_dev is not None and reference_dev.exists():
        ref = {(r.get("extra_info") or {}).get("uid") for r in read_jsonl(reference_dev)}
        mine = {(r.get("extra_info") or {}).get("uid") for r in dev}
        summary["dev_same_as_reference"] = ref == mine
        if ref != mine:
            problems.append(f"dev split differs from {reference_dev} ({len(mine)} vs {len(ref)} rows)")
    b = summary["pass_rate_bands"].get("t1_num", {})
    hardish = b.get("le_3_of_8", 0) + b.get("4_5_of_8", 0)
    if b and hardish < 0.1 * max(1, sum(b.values())):
        problems.append(f"only {hardish} T1 main problems at 5/8 or below")
    summary["problems"] = problems
    return summary, problems


def uid_of(row: dict[str, Any]) -> str | None:
    return (row.get("extra_info") or {}).get("uid")


def _rewrite_split(bank: Path, name: str, rows: list[dict[str, Any]], extra_parents: dict[str, str], notes: str):
    """Rewrite a published split (``<bank>/<name>.jsonl`` and ``.parquet``), keeping its manifest's name and version."""
    path = bank / f"{name}.jsonl"
    old = load_manifest(path) if manifest_path_for(path).exists() else None
    files = write_rows(rows, bank / name)
    write_manifest(build_manifest(files[0], name=old.name if old else name, version=old.version if old else "v2",
                                  parents={**(old.parents if old else {}), **extra_parents}, notes=notes))


def pin_dev(bank: Path, reference_dev: Path, t3_map: Path | None = None) -> dict[str, Any]:
    """``<bank>/rl_dev`` := the reference split's rows; its uids leave ``rl_main`` and ``rl_hard`` (module docstring).
    With ``t3_map``, a T3 dev row whose rubric was converted (status ok) gets the atomic checks, as in training."""
    reference = list(read_jsonl(reference_dev))
    converted = 0
    if t3_map is not None:
        mapping = {r["uid"]: r for r in read_jsonl(t3_map)}
        for r in reference:
            entry = mapping.get(uid_of(r)) if task_of(r) == "t3_report" else None
            if entry is None or entry.get("status") != "ok":
                continue
            gt = r["reward_model"]["ground_truth"]
            gt = json.loads(gt) if isinstance(gt, str) else dict(gt)
            gt["rubric"] = entry["rubric"]
            gt_text = json.dumps(gt, sort_keys=True, ensure_ascii=False)
            r["reward_model"] = {**r["reward_model"], "ground_truth": gt_text}
            converted += 1
    pinned = {uid_of(r) for r in reference}
    ref_hash = load_manifest(reference_dev).content_hash if manifest_path_for(reference_dev).exists() else "unknown"
    removed: dict[str, int] = {}
    for name in ("rl_main", "rl_hard"):
        rows = _rows(bank / f"{name}.jsonl")
        kept = [r for r in rows if uid_of(r) not in pinned]
        removed[name] = len(rows) - len(kept)
        if removed[name]:
            _rewrite_split(bank, name, kept, {"pinned_dev": ref_hash},
                           f"{name} without the {len(pinned)} problems of {reference_dev}")
    _rewrite_split(bank, "rl_dev", reference, {"pinned_dev": ref_hash}, f"the dev split of {reference_dev}")
    res = {"dev_rows": len(reference), "removed_from_training": removed, "t3_rubrics_converted": converted,
           "dev_tasks": dict(Counter(task_of(r) for r in reference))}
    log.info("pin dev: %s", res)
    return res


def without_task(source: Path, task: str, out_stem: Path) -> dict[str, Any]:
    """Write ``<out_stem>.jsonl`` (and ``.parquet``) with the rows of ``source`` that are not ``task``. The manifest's
    parent is the source's content hash, so the run's launch record traces the file back to the published split."""
    rows = list(read_jsonl(source))
    kept = [r for r in rows if task_of(r) != task]
    files = write_rows(kept, out_stem)
    parent = load_manifest(source)
    write_manifest(build_manifest(files[0], name=f"{parent.name}_without_{task}", version=parent.version,
                                  parents={str(source): parent.content_hash},
                                  notes=f"{source.name} without its {task} rows"))
    res = {"rows": len(kept), "dropped": len(rows) - len(kept), "files": [str(f) for f in files]}
    log.info("without %s: %s", task, res)
    return res


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--out", required=True, type=Path)
    m.add_argument("--t3-map", type=Path, default=None)
    m.add_argument("--name", default="rl_bank.verify", help="manifest name: rl_bank.<stage> of the merged stage")
    m.add_argument("inputs", nargs="+", type=Path)
    c = sub.add_parser("check")
    c.add_argument("bank", type=Path)
    c.add_argument("--reference-dev", type=Path, default=None, help="e.g. $DATA/v1/rl_dev.jsonl")
    c.add_argument("--t3-not-trained", action="store_true",
                   help="the run trains no T3: count non-atomic T3 rubrics, but do not fail on them")
    d = sub.add_parser("pin-dev")
    d.add_argument("bank", type=Path)
    d.add_argument("--reference-dev", required=True, type=Path, help="e.g. $DATA/v1/rl_dev.jsonl")
    d.add_argument("--t3-map", type=Path, default=None, help="convert the dev T3 rubrics too (t3_atomic map)")
    w = sub.add_parser("without-task")
    w.add_argument("--task", required=True)
    w.add_argument("--out", required=True, type=Path, help="path stem: writes <out>.jsonl and <out>.parquet")
    w.add_argument("source", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "merge":
        print(json.dumps(merge(a.inputs, a.out, a.t3_map, name=a.name)))
        return 0
    if a.cmd == "pin-dev":
        print(json.dumps(pin_dev(a.bank, a.reference_dev, a.t3_map)))
        return 0
    if a.cmd == "without-task":
        print(json.dumps(without_task(a.source, a.task, a.out)))
        return 0
    summary, problems = check(a.bank, a.reference_dev, t3_not_trained=a.t3_not_trained)
    (a.bank / "bank_check.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
