"""Collect the RL run's training dynamics for the talk on a cluster login node: aggregate metrics only, never the
text of a prompt, an answer or a reference.

    python3 scripts/talk/collect_rl.py --out $WORK/dpt/talk/rl.json      # ~30 s, standard library only

Sources (campaign 3532660, RL from step 0 at 2026-10-29 22:50):
- `<run>/metrics.jsonl`: the trainer's per-step metrics; the keys the figures read are kept.
- `<run>/val_generations/<step>.jsonl`: the dev passes. Per task: accuracy, failure reasons and response lengths
  (quantiles and a log-spaced histogram, split by correct and wrong).
- the SFT run's step lines (`campaign/logs/3532660/sft_node7.log`): loss, MTP loss, memory per step.
- `$WORK/dpt/logs/rl_stage8_host_mem.csv` (host memory of the last stage's node) and `rl_stage7_mem.csv` /
  `sft_stage7_mem.csv` (nvidia-smi on stage 7), each reduced to one row a minute.
scripts/talk/deck_figures.py draws them on the workstation.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

WORK = Path(os.environ.get("WORK", "."))
SCRATCH = Path(os.environ.get("SCRATCH", "."))
RUN = SCRATCH / "dpt" / "checkpoints" / "rl" / "rl9b_seed1"
LOGS = WORK / "dpt" / "logs"
TASKS = ["t1_num", "t2_code", "t3_report", "t4_general"]
KEEP = ("task/", "filter/", "curriculum/", "dev/", "alert/", "actor/entropy", "actor/pg_clipfrac", "actor/grad_norm",
        "actor/lr", "actor/ppo_kl", "actor/mtp_losses/", "actor/perf/", "mismatch/log_ppl_diff",
        "mismatch/log_ppl_abs_diff", "mismatch/kl_k3", "drift/log_ppl_diff", "drift/kl_k3", "critic/score/mean",
        "batch/", "response_length/", "timing_s/step", "timing_s/gen", "perf/")
LEN_BINS = [2 ** (k / 2) for k in range(12, 30)]          # 64 .. 16K tokens, half-octave edges


def metrics() -> list[dict[str, Any]]:
    rows = []
    with (RUN / "metrics.jsonl").open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows.append({k: v for k, v in r.items() if k == "step" or (k.startswith(KEEP) and
                                                                            isinstance(v, (int, float)))})
    return rows


def quantiles(xs: list[float]) -> dict[str, float]:
    xs = sorted(xs)
    return {f"p{int(100 * p)}": xs[min(len(xs) - 1, int(p * len(xs)))] for p in (0.1, 0.25, 0.5, 0.75, 0.9)} if xs else {}


def histogram(xs: list[float]) -> list[int]:
    counts = [0] * (len(LEN_BINS) + 1)
    for x in xs:
        counts[sum(1 for e in LEN_BINS if x >= e)] += 1
    return counts


def dev() -> dict[str, Any]:
    out: dict[str, Any] = {"len_bins": LEN_BINS, "steps": {}}
    for p in sorted((RUN / "val_generations").glob("*.jsonl"), key=lambda p: int(p.stem)):
        by_task: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        with p.open() as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    by_task[r.get("task", "?")].append({k: r.get(k) for k in ("acc", "reason", "response_length",
                                                                               "truncated", "overlong")})
        step = {}
        for t, rs in by_task.items():
            ok = [r["response_length"] or 0 for r in rs if (r["acc"] or 0) >= 1]
            bad = [r["response_length"] or 0 for r in rs if (r["acc"] or 0) < 1]
            step[t] = {"n": len(rs), "acc": sum(r["acc"] or 0 for r in rs) / len(rs),
                       "reasons": dict(collections.Counter(r["reason"] for r in rs if (r["acc"] or 0) < 1)),
                       "truncated": sum(1 for r in rs if r["truncated"]), "overlong": sum(1 for r in rs if r["overlong"]),
                       "len": quantiles(ok + bad), "len_correct": quantiles(ok), "len_wrong": quantiles(bad),
                       "hist_correct": histogram(ok), "hist_wrong": histogram(bad)}
        out["steps"][p.stem] = step
    return out


def per_minute(rows: list[tuple[str, dict[str, float]]], how: str = "max") -> list[dict[str, Any]]:
    """Rows keyed by 'YYYY-MM-DD HH:MM'; one row a minute with each field's max (or last)."""
    acc: dict[str, dict[str, float]] = {}
    for minute, vals in rows:
        cur = acc.setdefault(minute, {})
        for k, v in vals.items():
            cur[k] = max(cur.get(k, v), v) if how == "max" else v
    return [{"t": m, **v} for m, v in sorted(acc.items())]


def host_memory() -> list[dict[str, Any]]:
    p = LOGS / "rl_stage8_host_mem.csv"
    rows = []
    with p.open() as f:
        head = f.readline().strip().split(",")
        for line in f:
            v = dict(zip(head, line.strip().split(",")))
            try:
                rows.append((v["time"][:16], {"used_mb": float(v["used_mb"]), "avail_mb": float(v["avail_mb"]),
                                              "vllm_gb": float(v["vllm_worker_gb"]) + float(v["vllm_server_gb"]),
                                              "workerdict_gb": float(v["workerdict_gb"])}))
            except (KeyError, ValueError):
                continue
    # "in use" is total - available, as Ray's memory monitor counts it; keep the minute's peak
    return per_minute([(m, {**v, "in_use_mb": max(0.0, 256872 - v["avail_mb"])}) for m, v in rows])


def gpu_memory(name: str) -> list[dict[str, Any]]:
    """nvidia-smi CSV rows `YYYY/MM/DD HH:MM:SS.mmm, gpu, N MiB, U %`: per minute, the peak over the node's GPUs."""
    rows = []
    with (LOGS / name).open() as f:
        for line in f:
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 4 or not parts[2].endswith("MiB"):
                continue
            rows.append((parts[0][:16].replace("/", "-"), {"mib": float(parts[2].split()[0]),
                                                            "util": float(parts[3].split()[0])}))
    return per_minute(rows)


def sft_steps() -> list[dict[str, float]]:
    """verl's `step:N - key:value - …` console lines of the SFT run (last stage's log); the final launch only."""
    import re

    p = WORK / "dpt" / "campaign" / "logs" / "3532660" / "sft_node7.log"
    run: list[dict[str, float]] = []
    pat = re.compile(r"step:(\d+) - (.*)")
    with p.open(errors="replace") as f:
        for line in f:
            m = pat.search(re.sub(r"\x1b\[[0-9;]*m", "", line))
            if not m:
                continue
            row: dict[str, float] = {"step": int(m.group(1))}
            for kv in m.group(2).split(" - "):
                k, _, v = kv.partition(":")
                try:
                    row[k.strip()] = float(v)
                except ValueError:
                    pass
            if run and row["step"] <= run[-1]["step"]:
                run = []                                  # a relaunch from step 1 starts the list again
            run.append(row)
    return run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=WORK / "dpt" / "talk" / "rl.json")
    a = ap.parse_args()
    data = {"collected_at": dt.datetime.now().isoformat(timespec="minutes"), "run": str(RUN),
            "metrics": metrics(), "dev": dev(), "host_memory_stage8": host_memory(),
            "gpu_memory_stage7_rl": gpu_memory("rl_stage7_mem.csv"),
            "gpu_memory_stage7_sft": gpu_memory("sft_stage7_mem.csv"), "sft_steps": sft_steps(),
            "paused": json.loads((RUN / "PAUSED.json").read_text()) if (RUN / "PAUSED.json").exists() else None}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(data, separators=(",", ":")))
    print(f"{a.out}: {a.out.stat().st_size / 1e3:.0f} KB, {len(data['metrics'])} steps, dev {list(data['dev']['steps'])}")


if __name__ == "__main__":
    main()
