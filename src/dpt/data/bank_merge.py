"""Bank v2 (2026-11-07): merge verified stage files, apply the atomic T3 rubrics, and check the published bank.

    python -m dpt.data.bank_merge merge --out $V2/stages/04_verify.jsonl --t3-map $V2/t3_atomic_map.jsonl \\
        $V2/stages/04_verify.jsonl $DATA/v1_hard/stages/04_verify.jsonl $DATA/v1_long/stages/04_verify.jsonl
    python -m dpt.data.bank_merge check $V2          # after the difficulty and publish stages

``merge`` reads every input before writing (the output may be one of them), keeps the first row of each uid, leaves
out the problems of ``FAULTY_FAMILIES``, applies the T3 map (dpt.data.gen.t3_atomic.apply_map) and writes a manifest
whose parents are the inputs' content hashes.

``check`` prints counts only (tasks, pools, pass rates, rubric formats) and writes ``<bank>/bank_check.json``. It
exits 1 when the bank cannot train: no T1 rows, a T3 rubric that is not atomic, a dev split different from v1's, or
a main pool whose T1 problems are nearly all easy. ``--t3-not-trained`` (RL v2, 2026-11-11) still counts the
non-atomic T3 rubrics but does not fail on them: that run trains no T3 and keeps its rubrics as they are.

The check also estimates the share of T1 groups with mixed results at step 1 (2026-11-11). A problem graded r = k/8
by the SFT model gives a group of 8 with mixed results with probability 1 − r⁸ − (1 − r)⁸, and only such groups carry
gradient. In run 1's steps 1–20, 13% of T1 groups were mixed. Below 40% the bank fails, so an RL job never spends its
nodes on a bank that is too easy. The estimate is optimistic: a problem graded 7/8 may really be at 0.95.

The check also fails a bank with fewer than ``MIN_LEARNABLE_T1`` T1 main problems graded 1–6 of 8, and it
lists every T1 family's problems by grade (``t1_families``, from the stage file), so a bank whose hard
families turned out easy shows it (2026-11-15).

``pin-dev`` (2026-11-11) makes ``<bank>/rl_dev`` the reference split (v1's) and removes its problems from
``rl_main`` and ``rl_hard``. Publish draws the dev split from the problems that survive the difficulty pass. Bank
v2 re-grades them with the SFT model, which drops every problem it solves 8/8 (88% of run 1's T1 groups), so the
drawn split would hold only hard problems, differ from run 1's, and leave the rest of run 1's dev problems in
training. Run 1's dev curve and the pause rules calibrated on it compare only on the same split::

    python -m dpt.data.bank_merge pin-dev --reference-dev $DATA/v1/rl_dev.jsonl $V2

``without-task`` writes a published split without one task's rows, with a manifest, e.g. the dev split without
T3 for a run that has no judge::

    python -m dpt.data.bank_merge without-task --task t3_report --out $V2/rl_dev_no_t3 $V2/rl_dev.jsonl

``carve-hard-dev`` (2026-11-15) holds out a few problems of each hard and long family as a second dev split
(``rl_dev_hard``, its own pause rules) and writes the uid → family map the curriculum's family weights
need::

    python -m dpt.data.bank_merge carve-hard-dev --stages $V2/stages/05_difficulty.jsonl \\
        --families epicentre_from_three_sp,... --per-family 10 $V2
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
from dpt.common.seeding import derive_seed
from dpt.data.rl_build import write_rows

log = get_logger(__name__)

# Families whose labels are wrong, found by the 2026-11-14 screening of bank v1's expert queue. local_magnitude
# printed ML = log10 A + 1.11 log10 r + 0.00189 r − 2.09 with A in mm and called it Hutton–Boore: −2.09 is the
# IASPEI constant for A in nm, and with a Wood–Anderson amplitude in mm Hutton–Boore has +0.591 (1 mm at 100 km must
# give ML 3.0; the printed relation gave 0.32). Every label was 2.681 too low and the problems taught a false relation.
# Bank v1 holds 795 such T1 problems and 41 T2 code tasks (network_local_magnitude); the generators are fixed for new
# problems, and bank v2 leaves the old ones out. Run 1's dev split keeps its 6 (pin-dev): they are not trained on,
# and they grade answers against the relation they print.
FAULTY_FAMILIES = frozenset({"local_magnitude", "network_local_magnitude"})


def family_of(row: dict[str, Any]) -> str | None:
    """The generator family of a problem. Code tasks built from a T1 family carry it as ``family``; the other code
    tasks carry only their ``kind`` (12 network_local_magnitude tasks in bank v1 have family null)."""
    source, meta = row.get("source") or {}, row.get("meta") or {}
    return (source.get("family") or (meta.get("generator") or {}).get("name") or meta.get("family")
            or source.get("kind") or meta.get("kind"))


READMIT_REASONS = frozenset({"family_undecided", "agree"})


def readmit_undecided(verify_path: Path) -> list[dict[str, Any]]:
    """Problems of the templates that the verification audit rejected only because its verifiers could not agree on
    an answer: every audited instance undecided or agreeing, none contested, no perturbation (2026-11-16, the user's
    choice). The verifiers answer within 16K tokens. On bank v2's long families they were undecided on 10 of 12
    templates, with nothing contested, while the SFT model reached the label on 29-70% of its answers at 24K in
    several of them. Each readmitted problem carries ``meta.label_check = "sft"``: the difficulty pass keeps it only
    if the SFT model reaches its label at least once (dpt.data.difficulty.drop_unconfirmed). Reads the stage's
    ``../reports/verify_groups.json`` and ``03_decontam.jsonl`` next to ``verify_path``; none there, none readmitted."""
    from dpt.data.answer_verify import audit_group
    from dpt.data.schema import Problem

    report = verify_path.parent.parent / "reports" / "verify_groups.json"
    source = verify_path.parent / "03_decontam.jsonl"
    if not (report.exists() and source.exists()):
        return []
    groups = json.loads(report.read_text())
    readmit = {key for key, g in groups.items()
               if not g.get("kept") and g.get("audited") and not g.get("contested") and not g.get("perturbation")
               and set(g.get("reasons") or {}) <= READMIT_REASONS}
    out = []
    for r in read_jsonl(source):
        key = audit_group(Problem.from_dict(r, assign_uid=False))
        if key in readmit:
            g = groups[key]
            meta = {**(r.get("meta") or {}), "label_check": "sft",
                    "verified": {"audit": key, "accepted": g["accepted"], "audited": g["audited"],
                                 "readmitted": "verifiers undecided"}}
            out.append({**r, "meta": meta})
    return out


def merge(inputs: Sequence[Path], out: Path, t3_map: Path | None = None, version: str = "v2",
          name: str = "rl_bank.verify") -> dict[str, Any]:
    rows, seen, per_input, readmitted = [], set(), {}, {}
    parents = {str(p): load_manifest(p).content_hash for p in inputs if manifest_path_for(p).exists()}
    for p in inputs:
        n = 0
        extra = readmit_undecided(p) if name == "rl_bank.verify" else []
        for r in list(read_jsonl(p)) + extra:
            if r["uid"] in seen:
                continue
            seen.add(r["uid"])
            rows.append(r)
            n += 1
        per_input[str(p)] = n
        readmitted[str(p)] = len(extra)
    faulty = Counter(family_of(r) for r in rows if family_of(r) in FAULTY_FAMILIES)
    rows = [r for r in rows if family_of(r) not in FAULTY_FAMILIES]
    dropped: dict[str, int] = {}
    if t3_map is not None:
        from dpt.data.gen.t3_atomic import apply_map

        rows, dropped = apply_map(rows, {r["uid"]: r for r in read_jsonl(t3_map)})
    write_jsonl(out, rows)
    write_manifest(build_manifest(out, name=name, version=version, parents=parents,
                                  notes="bank v2: verified v1 + hard + long problems; T3 rubrics as atomic checks; "
                                        f"faulty families left out: {dict(faulty)}; readmitted (verifiers "
                                        f"undecided, label to be confirmed by the SFT model): {readmitted}"))
    res = {"rows": len(rows), "per_input": per_input, "readmitted": readmitted, "t3_dropped": dropped,
           "faulty_dropped": dict(faulty),
           "tasks": dict(Counter(r["task"] for r in rows))}
    log.info("bank merge: %s", res)
    return res


def _rows(path: Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path)) if path.exists() else []


def task_of(row: dict[str, Any]) -> str | None:
    """The task of a published RL row (``extra_info.task``, else verl's ``data_source``)."""
    return (row.get("extra_info") or {}).get("task") or row.get("data_source")


MIN_EXPECTED_MIXED_T1 = 0.40
# RL v2 draws ~8,400 T1 problems from the main pool in 120 steps (70% of 126 prompts, 80% of them from the main pool
# after step 20). With fewer learnable problems (graded 1–6 of 8) each would be seen 8+ times, and the run would be
# back to run 1's position of little to learn (2026-11-15).
MIN_LEARNABLE_T1 = 1000


def expected_mixed_share(pass_rates: Sequence[float], group_size: int = 8) -> float:
    """Expected share of groups with mixed results when each problem is solved with its graded rate r."""
    if not pass_rates:
        return 0.0
    return sum(1.0 - r ** group_size - (1.0 - r) ** group_size for r in pass_rates) / len(pass_rates)


def check(bank: Path, reference_dev: Path | None = None, t3_not_trained: bool = False,
          min_learnable: int = MIN_LEARNABLE_T1) -> tuple[dict[str, Any], list[str]]:
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
    # The 24K view, for the runs at 24K. Problems solved 8/8 at 24K stay in the bank since the 16K re-grade
    # (2026-11-15) only for a run at 16K; a 24K run gives them no copies, so they are left out of these estimates.
    t1_rates = [x for x in (rate_of(r) for r in main if task_of(r) == "t1_num")
                if isinstance(x, (int, float)) and x < 1]
    summary["expected_mixed_t1_at_step_1"] = round(expected_mixed_share(t1_rates), 3)
    summary["learnable_t1"] = sum(0.1 < x < 0.8 for x in t1_rates)          # graded 1–6 of 8
    # The 16K and 20K views, for a run there that counts a cut-off answer as wrong: mixed when 1–7 of 8 are right in
    # time (the 20K rate since 2026-11-15, difficulty extra_budgets).
    for budget in ("16k", "20k"):
        rates = [x for x in ((r.get("extra_info") or {}).get(f"pass_at_8_medium_{budget}") for r in main
                             if task_of(r) == "t1_num") if isinstance(x, (int, float))]
        if rates or budget == "16k":
            summary[f"learnable_t1_{budget}"] = sum(0 < x < 1 for x in rates)
            summary[f"expected_mixed_t1_at_step_1_{budget}"] = round(expected_mixed_share(rates), 3)
    summary["t1_families"] = t1_family_bands(bank, main, hard)

    if not summary["main"].get("t1_num"):
        problems.append("no T1 rows in rl_main")
    if not_atomic and not t3_not_trained:
        problems.append(f"{not_atomic} T3 rubrics are not atomic checks")
    if t1_rates and summary["learnable_t1"] < min_learnable:
        problems.append(f"only {summary['learnable_t1']} T1 main problems graded 1–6 of 8 (at least {min_learnable})")
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
    if b and hardish < 0.1 * max(1, sum(v for k, v in b.items() if k != "8_of_8")):
        problems.append(f"only {hardish} T1 main problems at 5/8 or below")
    summary["problems"] = problems
    return summary, problems


def uid_of(row: dict[str, Any]) -> str | None:
    return (row.get("extra_info") or {}).get("uid")


def t1_family_bands(bank: Path, main: list[dict[str, Any]], hard: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Per T1 family: problems in the main pool by grade (0/8 sits in the hard pool, 8/8 was dropped), from the
    bank's merged stage file (the published rows carry no family). Empty when the stage file is missing."""
    stage = bank / "stages" / "05_difficulty.jsonl"
    if not stage.exists():
        return {}
    fam = {str(r["uid"]): family_of(r) for r in read_jsonl(stage) if r.get("uid")}
    out: dict[str, dict[str, int]] = {}
    for pool, rows in (("main", main), ("hard", hard)):
        for r in rows:
            if task_of(r) != "t1_num":
                continue
            f = fam.get(str(uid_of(r))) or "unknown"
            band = "hard_pool" if pool == "hard" else _band((r.get("extra_info") or {}).get("pass_at_8_medium"))
            out.setdefault(f, {}).setdefault(band, 0)
            out[f][band] += 1
    return dict(sorted(out.items()))


def _band(rate: float | None) -> str:
    if not isinstance(rate, (int, float)):
        return "unknown"
    return "1_2_of_8" if rate <= 0.25 else "3_4_of_8" if rate <= 0.5 else "5_6_of_8" if rate <= 0.75 else "7_of_8"


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


HARD_DEV_SOURCE = "t1_num_hard"     # the hard dev rows' data_source: verl reports their dev metrics on their own


def carve_hard_dev(bank: Path, stages: Sequence[Path], families: Sequence[str], per_family: int,
                   seed: int = 1) -> dict[str, Any]:
    """Hold out ``per_family`` T1 problems of each of ``families`` as a second dev split, and write the uid → family
    map of the bank (2026-11-15).

    Run 1 got shorter on the hard problems and turned unfinished answers into wrong ones, while its dev split (run
    1's, pinned) held almost only easy problems, so the pause rules saw it late. This split gives the hard and long
    families their own dev numbers (``dev/t1_hard*``, ``dpt.monitoring.bridge``). Its rows keep the T1 grader but
    get ``data_source`` ``t1_num_hard`` (``dpt.verl_ext.reward_fn`` grades it as T1). The published rows carry no
    family, so the family comes from the bank's stage files (``family_of``).

    A family gives at most a third of its problems (2026-11-17, a review of RL v2): a readmitted family keeps only the
    problems the SFT model solved at least once (dpt.data.difficulty.drop_unconfirmed), which on the hardest families
    may be a dozen, and holding out 10 of them would leave training almost none. A family with fewer than three
    problems gives none.

    Writes ``rl_dev_hard``, ``rl_main_train`` and ``rl_hard_train`` (.jsonl, .parquet: rl_main and rl_hard without
    the held-out problems) and ``uid_family.json``; ``hard_dev.json`` records the inputs, and a later call on the
    same bank with the same settings only reads it."""
    fam: dict[str, str] = {}
    for stage in stages:
        for row in read_jsonl(stage):
            f = family_of(row)
            if row.get("uid") and f:
                fam[str(row["uid"])] = str(f)
    main, hard = _rows(bank / "rl_main.jsonl"), _rows(bank / "rl_hard.jsonl")
    inputs = {name: (load_manifest(bank / f"{name}.jsonl").content_hash
                     if manifest_path_for(bank / f"{name}.jsonl").exists() else f"{len(rows)} rows")
              for name, rows in (("rl_main", main), ("rl_hard", hard))}
    settings = {"inputs": inputs, "families": sorted(families), "per_family": per_family, "seed": seed,
                "max_dev_share": "1/3"}
    record = bank / "hard_dev.json"
    if record.exists() and {k: v for k, v in json.loads(record.read_text()).items() if k in settings} == settings:
        log.info("hard dev split already carved for these inputs (%s)", record)
        return {"skipped": True, **json.loads(record.read_text())}

    wanted = set(families)
    candidates: dict[str, list[dict[str, Any]]] = {f: [] for f in sorted(wanted)}
    for r in main + hard:
        f = fam.get(str(uid_of(r)))
        if f in wanted and task_of(r) == "t1_num":
            candidates[f].append(r)
    held: list[dict[str, Any]] = []
    per_family_held = {f: min(per_family, len(rows) // 3) for f, rows in candidates.items()}
    for f, rows in candidates.items():   # the same problems for a given bank and seed
        rows.sort(key=lambda r: derive_seed(seed, "hard_dev", uid_of(r)))
        for r in rows[:per_family_held[f]]:
            held.append({**r, "data_source": HARD_DEV_SOURCE, "extra_info": {**r["extra_info"], "split": "dev"}})
        if 0 < len(rows) < 3 * per_family:
            log.warning("hard dev split: %s has %d problems, so it gives %d and keeps %d for training", f, len(rows),
                        per_family_held[f], len(rows) - per_family_held[f])
    if not held:
        raise ValueError(f"no problem of {sorted(wanted)} in {bank}: wrong family names or stage files {stages}")
    held_uids = {uid_of(r) for r in held}
    for name, rows in (("rl_main_train", main), ("rl_hard_train", hard)):
        _rewrite_split(bank, name, [r for r in rows if uid_of(r) not in held_uids], inputs,
                       f"{name[:-6]} without the {len(held)} problems of rl_dev_hard")
    _rewrite_split(bank, "rl_dev_hard", held, inputs,
                   f"{per_family} T1 problems of each of {len(wanted)} hard and long families, held out")
    uid_family = {str(uid_of(r)): fam[str(uid_of(r))] for r in main + hard if str(uid_of(r)) in fam}
    (bank / "uid_family.json").write_text(json.dumps(uid_family, sort_keys=True))
    res = {**settings, "held_out": len(held), "per_family_held": per_family_held,
           "per_family_train": {f: len(v) - per_family_held[f] for f, v in candidates.items()},
           "missing_families": sorted(f for f, v in candidates.items() if not v), "family_map": len(uid_family)}
    record.write_text(json.dumps(res, indent=1))
    log.info("hard dev split: %s", res)
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
    c.add_argument("--min-learnable", type=int, default=MIN_LEARNABLE_T1,
                   help="fewest T1 main problems graded 1-6 of 8 for the bank to pass (2026-11-15)")
    d = sub.add_parser("pin-dev")
    d.add_argument("bank", type=Path)
    d.add_argument("--reference-dev", required=True, type=Path, help="e.g. $DATA/v1/rl_dev.jsonl")
    d.add_argument("--t3-map", type=Path, default=None, help="convert the dev T3 rubrics too (t3_atomic map)")
    w = sub.add_parser("without-task")
    w.add_argument("--task", required=True)
    w.add_argument("--out", required=True, type=Path, help="path stem: writes <out>.jsonl and <out>.parquet")
    w.add_argument("source", type=Path)
    h = sub.add_parser("carve-hard-dev")
    h.add_argument("--stages", nargs="+", required=True, type=Path, help="the bank's stage files, for the families")
    h.add_argument("--families", required=True, help="comma-separated family names")
    h.add_argument("--per-family", type=int, default=10)
    h.add_argument("--seed", type=int, default=1)
    h.add_argument("bank", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "carve-hard-dev":
        families = [f for f in a.families.split(",") if f]
        print(json.dumps(carve_hard_dev(a.bank, a.stages, families, a.per_family, a.seed)))
        return 0
    if a.cmd == "merge":
        print(json.dumps(merge(a.inputs, a.out, a.t3_map, name=a.name)))
        return 0
    if a.cmd == "pin-dev":
        print(json.dumps(pin_dev(a.bank, a.reference_dev, a.t3_map)))
        return 0
    if a.cmd == "without-task":
        print(json.dumps(without_task(a.source, a.task, a.out)))
        return 0
    summary, problems = check(a.bank, a.reference_dev, t3_not_trained=a.t3_not_trained, min_learnable=a.min_learnable)
    (a.bank / "bank_check.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
