"""Noise floor from repeated evaluations of one checkpoint (report §harness 规则).

    python -m dpt.eval.noise_floor --set d_num --out eval_results/noise_floor.json \
        eval_results/base_medium_seed{0..9}

Each run directory holds a ``results.json`` from ``dpt.eval.harness`` produced
with a different ``--seed`` on the *same* checkpoint. The report found a mean
pairwise D-Num difference of 0.9 and a maximum of 2.1 over 10 repeats. Any
change smaller than the maximum is treated as noise, and
``dpt.eval.stats.is_significant_improvement`` uses it as a floor.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dpt.common.io import read_json, write_json
from dpt.common.logging import get_logger
from dpt.eval.report import row_from_results
from dpt.eval.stats import noise_floor

log = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="run directories (same checkpoint, different seeds)")
    ap.add_argument("--set", default="d_num")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    scores = [row_from_results(read_json(Path(r) / "results.json"))[a.set]["score"] for r in a.runs]
    nf = noise_floor(scores)
    write_json(a.out, {"set": a.set, "runs": list(a.runs), "scores": scores, **nf.to_dict()})
    log.info("noise floor on %s over %d runs: %s", a.set, len(scores), nf.to_dict())


if __name__ == "__main__":
    main()
