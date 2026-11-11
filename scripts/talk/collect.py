"""Collect the talk's numbers on a cluster login node: aggregate counts and metrics only, never the text of a
problem, an answer or an eval item.

    python3 scripts/talk/collect.py --out $WORK/dpt/talk/data.json      # ~1 min, standard library only

Run `python -m dpt.eval.budget --eval-root $WORK/dpt/eval_results --out $WORK/dpt/talk/budget` (dpt-train env)
first: it regrades the cached generations into token-budget counts, which this script only reads.

scripts/talk/render.py turns the JSON into docs/talk/numbers.md and docs/talk/figures/ on the workstation.
A source that does not exist yet (a stage not run) is recorded as null and rendered as pending.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

WORK = Path(os.environ.get("WORK", "."))
DATA = WORK / "dpt" / "data"
BANK = DATA / "v1"
SECURE = WORK / "dpt" / "eval_secure"
CAMPAIGN = WORK / "dpt" / "campaign"
REPO = Path(__file__).resolve().parents[2]              # job logs (<name>-<id>.out) land in the submit dir
STAGES = ["ingest", "quality", "dedup", "decontam", "verify", "difficulty"]
TASKS = ["t1_num", "t2_code", "t3_report", "t4_general"]
ALLOCATION = "PROJ"
SINCE = "2026-10-13"                     # the month on the cluster starts (README)
# SU per node-hour (configs/hardware/slurm_3xa100_40g.yaml; the cluster user guide)
SU_RATE = {"gpu-a100": 3.0, "gpu-a100-dev": 3.0, "gpu-a100-small": 1.5, "gpu-h100": 6.0,
           "normal": 1.0, "development": 1.0, "large": 1.0}


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def drop_keys(obj: Any, keys: set[str]) -> Any:
    if isinstance(obj, dict):
        return {k: drop_keys(v, keys) for k, v in obj.items() if k not in keys}
    if isinstance(obj, list):
        return [drop_keys(v, keys) for v in obj]
    return obj


def pipeline() -> dict[str, Any] | None:
    """The RL bank's funnel, per stage and per task, with every removal reason."""
    funnel = read_json(BANK / "funnel.json")
    ingest = BANK / "stages" / "00_ingest.jsonl"
    if funnel is None or not ingest.exists():
        return None
    task_of = {r["uid"]: r["task"] for r in jsonl(ingest)}
    entering = collections.Counter(task_of.values())        # after ingest's own removal
    removed: dict[str, dict[str, collections.Counter]] = {}
    for st in STAGES:
        per = collections.defaultdict(collections.Counter)
        p = BANK / "removed" / f"{st}.jsonl"
        if p.exists():
            for r in jsonl(p):
                t = task_of.get(r.get("uid")) or str(r.get("group") or "?").split(":")[0]
                per[t][r.get("reason", "?")] += 1
        removed[st] = per
    by_task: dict[str, dict[str, int]] = {}
    for t in TASKS:
        n = entering[t] + sum(removed["ingest"][t].values())
        row = {"ingest": n}
        for st in STAGES:
            n -= sum(removed[st][t].values())
            row[f"after_{st}"] = n
        by_task[t] = row
    published = (read_json(BANK / "publish_summary.json") or {}).get("counts", {})
    kept = {t: sum(published.get(s, {}).get(t, 0) for s in ("rl_main", "rl_hard", "rl_dev")) for t in TASKS}
    return {"stages": funnel, "by_task": by_task,
            "removed_by_task": {st: {t: dict(c) for t, c in per.items()} for st, per in removed.items()},
            "published": published, "check_after_difficulty_equals_published":
                all(by_task[t]["after_difficulty"] == kept[t] for t in TASKS)}


def groups() -> dict[str, Any]:
    verify = read_json(BANK / "reports" / "verify_groups.json") or {}
    for g in verify.values():
        if isinstance(g.get("why"), str):
            g["why"] = g["why"][:200]
    return {"verify": verify, "difficulty": read_json(BANK / "reports" / "difficulty_groups.json")}


def base_replies() -> dict[str, Any] | None:
    """Length of the base model's replies in the difficulty audit (8 per audited problem), in characters."""
    p = BANK / "cache" / "difficulty_base.jsonl"
    if not p.exists():
        return None
    lengths, closed, problems = [], 0, 0
    for r in jsonl(p):
        problems += 1
        for out in r.get("out") or []:
            out = out or ""
            lengths.append(len(out))
            closed += "</think>" in out
    lengths.sort()
    q = (lambda f: lengths[min(len(lengths) - 1, int(f * len(lengths)))]) if lengths else (lambda f: None)
    edges = [0, 2000, 5000, 10000, 20000, 40000, 60000, 80000, 100000, 10 ** 9]
    hist = [sum(1 for x in lengths if lo <= x < hi) for lo, hi in zip(edges, edges[1:])]
    return {"problems": problems, "replies": len(lengths), "closed_think": closed, "chars_p10": q(0.1),
            "chars_p50": q(0.5), "chars_p90": q(0.9), "chars_max": lengths[-1] if lengths else None,
            "hist_edges": edges, "hist": hist}


def sft() -> dict[str, Any] | None:
    report = read_json(BANK / "sft" / "sft_report.json")
    train = BANK / "sft" / "sft_train.jsonl"
    if report is None or not train.exists():
        return report
    by_subset, by_task = collections.Counter(), collections.defaultdict(collections.Counter)
    for r in jsonl(train):
        s = r.get("subset", "?")
        by_subset[s] += 1
        by_task[s][(r.get("meta") or {}).get("task") or "?"] += 1
    return {**report, "conversations_by_subset": dict(by_subset),
            "conversations_by_subset_task": {s: dict(c) for s, c in by_task.items()}}


def eval_sets() -> dict[str, Any]:
    out: dict[str, Any] = {"sets": {}, "extra_files": {}}
    for m in sorted(SECURE.glob("*.jsonl.manifest.json")):
        j = read_json(m) or {}
        out["sets"][m.name.replace(".jsonl.manifest.json", "")] = {
            "records": j.get("num_records"), "created": (j.get("created_at") or "")[:10]}
    frozen = read_json(SECURE / "FROZEN.json") or {}
    out["frozen_at"] = frozen.get("frozen_at")
    for sub in ("general", "safety"):
        for f in sorted((SECURE / sub).glob("*.jsonl")):
            with f.open(encoding="utf-8") as fh:
                out["extra_files"][f"{sub}/{f.name}"] = sum(1 for line in fh if line.strip())
    return out


def jobs() -> dict[str, Any]:
    cols = "JobID,JobName,Partition,NNodes,Submit,Start,End,ElapsedRaw,State"
    try:
        txt = subprocess.run(["sacct", "-u", os.environ.get("USER", ""), "-S", SINCE, "-X", "-n", "-P",
                              "-o", cols], capture_output=True, text=True, timeout=120).stdout
    except (OSError, subprocess.SubprocessError):
        txt = ""
    rows = []
    for line in txt.splitlines():
        f = line.split("|")
        if len(f) != 9:
            continue
        jid, name, part, nodes, submit, start, end, elapsed, state = f
        nodes, elapsed = int(nodes or 0), int(elapsed or 0)
        rate = SU_RATE.get(part)
        row = {"id": jid, "name": name, "partition": part, "nodes": nodes, "submit": submit, "start": start,
               "end": end, "elapsed_s": elapsed, "state": state.split()[0],
               "su": round(nodes * elapsed / 3600 * rate, 2) if rate is not None else None}
        log = REPO / f"{name}-{jid}.out"                    # serve.slurm: "serving <model> as '<endpoint>'"
        if name == "dpt-serve" and log.exists():
            with log.open(encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    m = re.search(r"serving (\S+) as '([\w-]+)'", line)
                    if m:
                        row["model"], row["endpoint"] = m.group(1), m.group(2)
                        break
        rows.append(row)
    balance = None
    try:
        info = subprocess.run(["/usr/local/etc/allocation_report"], capture_output=True, text=True, timeout=60).stdout
        m = re.search(rf"{ALLOCATION}\s+(-?[\d,]+)\s+(\S+)", info)
        if m:
            balance = {"available_su": int(m.group(1).replace(",", "")), "expires": m.group(2)}
    except (OSError, subprocess.SubprocessError):
        pass
    return {"since": SINCE, "jobs": rows, "balance": balance,
            "unknown_partitions": sorted({r["partition"] for r in rows if r["su"] is None})}


SET_KEYS = ("metric", "score", "ci", "pass_at_k", "truncation_rate", "mean_output_tokens", "reasons", "outcomes",
            "length_matched", "refusal_rate", "over_refusal_rate", "score_by_max_tokens", "by_judge", "gate_judge")


def eval_results() -> dict[str, Any]:
    """Every harness run under eval_results/: scores, CIs, pass@k and counts. Per-item lists are left out."""
    root = WORK / "dpt" / "eval_results"
    runs = {}
    for p in sorted(root.glob("**/results.json")):
        j = read_json(p) or {}
        runs[str(p.parent.relative_to(root))] = {
            "model_name": j.get("model_name"), "effort": j.get("effort"), "max_tokens": j.get("max_tokens"),
            "seed": j.get("seed"), "mean_output_tokens": j.get("mean_output_tokens"),
            "written": dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="minutes"),
            "sets": {n: {k: v[k] for k in SET_KEYS if k in v} for n, v in (j.get("sets") or {}).items()},
            "general": {n: {k: v[k] for k in ("metric", "score", "ci", "relative_only") if k in v}
                        for n, v in (j.get("general") or {}).items()}}
    return {"runs": runs, "noise_floor": read_json(root / "noise" / "noise_floor.json")}


def budget() -> dict[str, Any]:
    """Accuracy under a token budget per run: counts that `python -m dpt.eval.budget` (dpt-train env) writes."""
    return {p.stem: read_json(p) for p in sorted((WORK / "dpt" / "talk" / "budget").glob("*.json"))}


def family_scores() -> dict[str, Any]:
    """Per run and numeric set, the mean score of each problem family (family names come from the code)."""
    family = {}
    for name in ("d_num", "d_num_hard"):
        p = SECURE / f"{name}.jsonl"
        if p.exists():
            family[name] = {r["id"]: r["meta"]["family"] for r in jsonl(p)}
    out: dict[str, Any] = {}
    root = WORK / "dpt" / "eval_results"
    for p in sorted(root.glob("**/results.json")):
        sets = (read_json(p) or {}).get("sets") or {}
        for name, fam in family.items():
            items = (sets.get(name) or {}).get("per_item") or []
            if not items:
                continue
            acc: dict[str, list[float]] = collections.defaultdict(list)
            for it in items:
                acc[fam.get(it["id"], "?")].append(it["mean"])
            out.setdefault(str(p.parent.relative_to(root)), {})[name] = {
                f: {"score": 100 * sum(v) / len(v), "items": len(v), "never_solved": sum(1 for x in v if x == 0)}
                for f, v in acc.items()}
    return out


def _boxed_number(text: str) -> float | None:
    i = text.rfind("\\boxed{")
    m = re.match(r"\s*([-+]?\d+(?:\.\d+)?)", text[i + 7:]) if i >= 0 else None
    return float(m.group(1)) if m else None


def diagnostics() -> dict[str, Any]:
    """Aggregates behind the report's diagnosis figures: how long each noise seed ran, when pass@64 requests
    failed (minutes after its start), and how the base's finished Moho-depth answers compare with the plane-wave
    and the ray-path formulas (ratios to the label only, no item text)."""
    out: dict[str, Any] = {}
    logs = sorted(CAMPAIGN.glob("logs/*/baseline_eval.log"), key=lambda q: q.stat().st_mtime)
    if logs:
        ts = lambda line: dt.datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")  # noqa: E731
        seeds, start, k64, fails = [], None, None, []
        with logs[-1].open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if "generating 300/300 items (n=8," in line:
                    start = ts(line)
                elif "/noise/base_medium_s" in line and "wrote" in line and start:
                    seed = int(line.rsplit("_s", 1)[1].split("/")[0])
                    seeds.append({"seed": seed, "minutes": round((ts(line) - start).total_seconds() / 60, 2)})
                    start = None
                elif "generating 300/300 items (n=64," in line:
                    k64 = ts(line)
                elif "request failed" in line and k64:
                    fails.append({"minutes": round((ts(line) - k64).total_seconds() / 60, 2),
                                  "kind": "timeout" if "server error" in line else "disconnect"})
        out["seed_minutes"], out["k64_failures"] = seeds, fails
    items, gens = SECURE / "d_num_hard.jsonl", WORK / "dpt" / "eval_results" / "d_num_hard" / "base_medium"
    gens = gens / "d_num_hard" / "generations.jsonl"
    if items.exists() and gens.exists():
        moho = {r["id"]: r for r in jsonl(items) if r["meta"]["family"] == "moho_depth_from_ps_delay"}
        ratios, finished, total = [], 0, 0
        for rec in jsonl(gens):
            it = moho.get(rec["item_id"])
            if it is None:
                continue
            q, true = it["meta"]["params"], it["answer"]["value"]
            ci = (1 - (q["p_s_km"] * q["vp_km_s"]) ** 2) ** 0.5
            cj = (1 - (q["p_s_km"] * q["vs_km_s"]) ** 2) ** 0.5
            ray = q["dt_s"] / (1 / (q["vs_km_s"] * cj) - 1 / (q["vp_km_s"] * ci))
            for smp in rec["samples"]:
                total += 1
                if "</think>" not in smp["text"]:
                    continue
                finished += 1
                v = _boxed_number(smp["text"].split("</think>")[-1])
                if v:
                    ratios.append({"model": round(v / true, 4), "ray_path": round(ray / true, 4)})
        out["receiver"] = {"samples": total, "finished": finished, "ratios": ratios}
    return out


def campaign() -> dict[str, Any]:
    st = CAMPAIGN / "state"
    marks = {}
    if st.exists():
        for p in sorted(st.iterdir()):
            marks[p.name] = dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="minutes")
    return {"finished_steps": marks}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(WORK / "dpt" / "talk" / "data.json"))
    args = ap.parse_args()
    data = {"collected_at": dt.datetime.now().isoformat(timespec="minutes"), "pipeline": pipeline(),
            "groups": groups(), "base_replies": base_replies(), "sft_mix": sft(),
            "contamination": drop_keys(read_json(BANK / "reports" / "contamination_report.json"), {"path"}),
            "eval_sets": eval_sets(), "compute": jobs(), "campaign": campaign(), "eval": eval_results(),
            "budget": budget(), "family_scores": family_scores(), "diagnostics": diagnostics()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1, sort_keys=True))
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
