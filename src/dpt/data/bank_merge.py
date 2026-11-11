"""Bank v2 (2026-11-07): merge verified stage files, apply the atomic T3 rubrics, and check the published bank.

    python -m dpt.data.bank_merge merge --out $V2/stages/04_verify.jsonl --t3-map $V2/t3_atomic_map.jsonl \\
        $V2/stages/04_verify.jsonl $DATA/v1_hard/stages/04_verify.jsonl $DATA/v1_long/stages/04_verify.jsonl
    python -m dpt.data.bank_merge check $V2          # after the difficulty and publish stages

``merge`` reads every input before writing (the output may be one of them), keeps the first row of each uid, applies
the T3 map (dpt.data.gen.t3_atomic.apply_map) and writes a manifest whose parents are the inputs' content hashes.

``check`` prints counts only (tasks, pools, pass rates, rubric formats) and writes ``<bank>/bank_check.json``. It
exits 1 when the bank cannot train: no T1 rows, a T3 rubric that is not atomic, a dev split different from v1's, or
a main pool whose T1 problems are nearly all easy.
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


def check(bank: Path, reference_dev: Path | None = None) -> tuple[dict[str, Any], list[str]]:
    main, hard, dev = (_rows(bank / f"rl_{n}.jsonl") for n in ("main", "hard", "dev"))
    problems: list[str] = []

    def task_of(r):
        return (r.get("extra_info") or {}).get("task") or r.get("data_source")

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

    if not summary["main"].get("t1_num"):
        problems.append("no T1 rows in rl_main")
    if not_atomic:
        problems.append(f"{not_atomic} T3 rubrics are not atomic checks")
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
    a = ap.parse_args(argv)
    if a.cmd == "merge":
        print(json.dumps(merge(a.inputs, a.out, a.t3_map, name=a.name)))
        return 0
    summary, problems = check(a.bank, a.reference_dev)
    (a.bank / "bank_check.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
