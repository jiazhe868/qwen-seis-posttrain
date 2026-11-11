"""Preliminary RL bank for the 9B timing run only (configs/rl/rl_9b_timing.yaml).

    python -m dpt.data.gen.prelim --t1 $G/t1_candidates.jsonl --t2 $DATA/prelim/t2/candidates_t2.jsonl \\
        --t4 $G/t4_candidates.jsonl --out-dir $DATA/prelim --n 3000

The timing run has to start before the real bank exists, so this takes generated candidates as they are:
no verification, no decontamination, no difficulty filter. Shares follow curriculum stage A (T1 50%,
T2 25%, T4 25%; no T3). Never train a real run on it: it is written to its own directory and every row
is marked ``meta.prelim``.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path
from typing import Sequence

from dpt.common.io import read_jsonl, write_json
from dpt.common.logging import get_logger
from dpt.data.rl_build import export_rl_bank
from dpt.data.schema import Problem

log = get_logger(__name__)

SHARES = {"t1": 0.50, "t2": 0.25, "t4": 0.25}


def take(path: str | Path, n: int, seed: int) -> list[Problem]:
    rows = [r for r in read_jsonl(path) if (r.get("meta") or {}).get("split", "train") == "train"]
    rows.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['question']}".encode()).hexdigest())
    out = []
    for r in rows[:n]:
        p = Problem.from_dict(r)
        p.meta.update(pool="main", prelim=True)
        out.append(p)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Preliminary RL bank for the timing run (not for training)")
    ap.add_argument("--t1", required=True)
    ap.add_argument("--t2", required=True)
    ap.add_argument("--t4", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    if out.resolve().name in ("v1", "v2") or (out / "stages").exists():
        raise SystemExit(f"{out} looks like a real bank directory; the preliminary bank goes elsewhere")
    probs = []
    for key, path in (("t1", args.t1), ("t2", args.t2), ("t4", args.t4)):
        probs += take(path, int(args.n * SHARES[key]), args.seed)
    info = export_rl_bank(probs, out, dev_fraction=0.01, dev_min_per_task=8, seed=args.seed)
    note = "timing run only: unverified, not decontaminated, no difficulty filter"
    # Manifested like a published bank (the launcher refuses unmanifested data), and named as prelim.
    from dpt.common.manifest import build_manifest, write_manifest

    for name, files in info["files"].items():
        write_manifest(build_manifest(files[0], name=f"prelim.{name}", version="prelim", notes=note,
                                      stage_stats=[{"stage": "prelim", "count": sum(info["counts"][name].values())}]))
    write_json(out / "prelim_summary.json",
               {"counts": info["counts"], "by_task": dict(Counter(p.task for p in probs)), "note": note})
    log.info("preliminary bank → %s: %s", out, info["counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
