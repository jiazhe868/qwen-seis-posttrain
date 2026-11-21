"""Collect the comparison of RL v2 with two larger Qwen models on D-Num-hard into one public data file.

Why this file exists. A 9B result invites one question first: how does it stand against larger models of the same
family? Two of them were run on D-Num-hard with the harness, prompts, effort, sampling and 8 samples per item of every
other model here, with 4-bit weights on a workstation's A100s: Qwen3.8-27B (NVFP4), which is also the teacher that
wrote the SFT solutions, and Qwen3.5-122B-A10B (GPTQ-Int4), the 9B's own generation at 13 times its size. The paired
comparison (`dpt.eval.budget compare`, the same code as the three-budget test of §8.3) wrote one file per large model
as baseline, next to the private eval results. This script keeps what may be published, the scores, outcome shares
and paired differences, and writes them to docs/talk/data/large_models_1120.json for the README, the report and the
deck.

Only the 16K and 32K budgets are kept. RL v2 trained with a 24K response cap, so a 64K comparison asks it to use room
it never trained for; and the 122B's 64K step was not run, so its 64K entry in the source file is a copy of its 32K
entry.

    python scripts/talk/large_models.py --results <folder holding budgets_large_models_vs_*.json>
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

repository_root = Path(__file__).resolve().parents[2]
output_path = repository_root / "docs" / "talk" / "data" / "large_models_1120.json"
large_model_names = {"qwen35_122b_int4": "Qwen3.5-122B-A10B (GPTQ-Int4)", "qwen38_27b_nvfp4": "Qwen3.8-27B (NVFP4)"}
kept_rows = ["qwen35_122b_int4", "qwen38_27b_nvfp4", "sft", "rl_seed1", "rl_v2_seed1", "rl_v2_seed2", "rl_v2"]
kept_budgets = ["at_16k", "at_32k"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", required=True, help="folder with budgets_large_models_vs_<model>.json")
    arguments = parser.parse_args()
    collected: dict = {"written_at": dt.datetime.now().isoformat(timespec="minutes"), "set": "d_num_hard",
                       "samples_per_item": 8, "large_models": large_model_names, "rows": {}, "versus": {}}
    for baseline_name in large_model_names:
        source = json.loads((Path(arguments.results) / f"budgets_large_models_vs_{baseline_name}.json").read_text())
        assert source["baseline"] == baseline_name, f"{baseline_name}: the file names another baseline"
        rows = source["sets"]["d_num_hard"]
        assert all(row in rows for row in kept_rows), f"{baseline_name}: a row is missing"
        for row_name in kept_rows:
            row = rows[row_name]
            scores = {budget: {"score": row[budget]["score"], "outcomes_pct": row[budget]["outcomes_pct"]}
                      for budget in kept_budgets}
            collected["rows"].setdefault(row_name, {"items": row["items"], **scores})
            if row_name != baseline_name:
                # the paired difference row − baseline over the same 160 items, with its 95% interval
                collected["versus"].setdefault(baseline_name, {})[row_name] = {
                    budget: row[budget]["vs_baseline"] for budget in kept_budgets}
        collected.setdefault("collected_at", {})[baseline_name] = source["collected_at"]
    output_path.write_text(json.dumps(collected, indent=1) + "\n")
    print(f"wrote {output_path}")
    for row_name in kept_rows:
        print(f"  {row_name:18s} 16K {collected['rows'][row_name]['at_16k']['score']:5.1f}   "
              f"32K {collected['rows'][row_name]['at_32k']['score']:5.1f}")


if __name__ == "__main__":
    main()
