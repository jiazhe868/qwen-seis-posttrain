"""T1 instances from the seismology families, and the D-Num eval set from the held-out families.

    python -m dpt.data.gen.t1 build --out candidates_t1.jsonl --heldout-out $SECURE/d_num_draft.jsonl --seed 1
    python -m dpt.eval.drafts wording --items $SECURE/d_num_draft.jsonl --out d_num_wording.csv   # login node
    python -m dpt.data.gen.t1 finalize-dnum --draft $SECURE/d_num_draft.jsonl --wording d_num_wording.csv \\
        --out $SECURE/d_num.jsonl --perturbed-out $SECURE/d_num_perturbed.jsonl

The hard bank (``dpt.data.gen.seismology_hard``, 2026-10-27) builds the same way into its own files::

    python -m dpt.data.gen.t1 build --bank hard --set-name d_num_hard --out candidates_t1_hard.jsonl \\
        --heldout-out $SECURE/d_num_hard_draft.jsonl --per-family 600 --n-heldout 210

``build`` writes training candidates (every ``split="train"`` family, up to ``--per-family`` distinct
questions each, templates in turn) and a D-Num *draft*: ``--n-heldout`` items, more than the set needs,
balanced over the held-out families.

``finalize-dnum`` drops every draft item the hosted model's wording check did not mark ``ok`` (its answer
disagreed with the solver, or it found the wording unclear; nobody adjudicates), keeps ``--n-items``
balanced over families, and writes D-Num plus 200 perturbed twins for the layer-3 check
(``dpt.eval.consistency``): the same questions with every input moved by 5–15% and the answer recomputed.
Both files get ids, a canary and a manifest. The wording CSV is read as plain data; nothing here imports
the hosted-model client, and its text never enters training data.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from dpt.common.hashing import short_hash
from dpt.common.io import read_jsonl, write_json, write_jsonl
from dpt.common.logging import get_logger
from dpt.common.types import T1_NUM
from dpt.data.gen.common import candidate, rng_for, write_candidates
from dpt.data.gen.families import FAMILIES, Family, by_split

log = get_logger(__name__)

ORIGIN = "dpt.data.gen.t1"


def instances(f: Family, n: int, seed: int, *, max_draws_factor: int = 20) -> list[tuple[dict[str, float], int, str]]:
    """Up to ``n`` (params, template index, question) with distinct questions, templates used in turn."""
    rng = rng_for(seed, "t1", f.name)
    out, seen = [], set()
    for i in range(n * max_draws_factor):
        if len(out) >= n:
            break
        params = f.sample(rng)
        k = i % len(f.templates)
        q = f.render(params, k)
        if q in seen:
            continue
        seen.add(q)
        out.append((params, k, q))
    return out


def family_candidates(f: Family, n: int, seed: int) -> list[dict[str, Any]]:
    rows = []
    for params, k, q in instances(f, n, seed):
        rows.append(candidate(
            T1_NUM, q, f.answer(params), origin=ORIGIN, split=f.split,
            source={"family": f.name, "template": k},
            family=f.name, topic=f.topic, steps=f.steps, template=k,
            generator=f.generator_meta(params), params=dict(params),
        ))
    return rows


def build_train(families: Sequence[Family], per_family: int, seed: int) -> list[dict[str, Any]]:
    if any(f.split != "train" for f in families):
        raise ValueError("held-out families never produce training candidates")
    return [r for f in families for r in family_candidates(f, per_family, seed)]


def build_dnum_draft(families: Sequence[Family], n_items: int, seed: int,
                     set_name: str = "d_num") -> list[dict[str, Any]]:
    """Eval records, balanced over the held-out families (round-robin)."""
    if any(f.split != "heldout" for f in families):
        raise ValueError("D-Num comes from held-out families only")
    per = -(-n_items // max(1, len(families)))
    pools = {f.name: family_candidates(f, per, seed + 7919) for f in families}
    out: list[dict[str, Any]] = []
    while len(out) < n_items and any(pools.values()):
        for name in sorted(pools):
            if pools[name] and len(out) < n_items:
                out.append(_eval_record(pools[name].pop(0), set_name))
    return out


def _eval_record(c: dict[str, Any], set_name: str = "d_num") -> dict[str, Any]:
    m = c["meta"]
    return {"id": f"{set_name}-" + short_hash(c["question"], 12), "set": set_name, "task": T1_NUM,
            "question": c["question"], "answer": c["answer"],
            "meta": {k: m[k] for k in ("family", "topic", "steps", "template", "generator", "params")}}


# --------------------------------------------------------------------------- D-Num finalization


def read_wording(path: str | Path) -> dict[str, str]:
    with open(path, newline="", encoding="utf-8") as f:
        return {r["id"]: (r.get("flag") or "").strip() for r in csv.DictReader(f)}


def perturbed_twin(rec: dict[str, Any], seed: int) -> dict[str, Any]:
    """The same question with every input moved by ±5–15% (re-rounded), the answer recomputed by the solver."""
    f = FAMILIES[rec["meta"]["family"]]
    rng = rng_for(seed, "perturb", rec["id"])
    base = rec["meta"]["params"]
    for _ in range(200):
        if f.make is not None:            # consistent draws: take a fresh draw from the family instead
            params = f.sample(rng)
        else:
            params = {q.name: q.round(base[q.name] * (1 + rng.choice((-1, 1)) * rng.uniform(0.05, 0.15)))
                      for q in f.params}
        value = f.solve(params)
        if params != base and f.is_valid(params) and value == value:
            break
    else:
        raise RuntimeError(f"{rec['id']}: no valid perturbation")
    q = f.render(params, rec["meta"]["template"])
    return {"id": rec["id"] + "-p", "set": rec["set"] + "_perturbed", "task": T1_NUM, "question": q,
            "answer": f.answer(params),
            "meta": {**rec["meta"], "params": params, "generator": f.generator_meta(params), "twin_of": rec["id"]}}


def finalize_dnum(draft: Sequence[dict[str, Any]], flags: dict[str, str], n_items: int, n_perturbed: int,
                  seed: int, set_name: str = "d_num") -> tuple[list[dict], list[dict], dict[str, Any]]:
    from dpt.data.decontam import make_canary

    missing = [r["id"] for r in draft if r["id"] not in flags]
    if missing:
        raise ValueError(f"{len(missing)} draft items have no wording verdict (e.g. {missing[:3]}); rerun wording")
    ok = [r for r in draft if flags[r["id"]] == "ok"]
    by_fam: dict[str, list[dict]] = {}
    for r in ok:
        by_fam.setdefault(r["meta"]["family"], []).append(r)
    chosen: list[dict] = []
    while len(chosen) < n_items and any(by_fam.values()):
        for name in sorted(by_fam):
            if by_fam[name] and len(chosen) < n_items:
                chosen.append(by_fam[name].pop(0))
    canary = make_canary(f"{set_name}/v1")
    items = [{**r, "canary": canary} for r in chosen]
    pcanary = make_canary(f"{set_name}_perturbed/v1")
    twins = [{**perturbed_twin(r, seed), "canary": pcanary} for r in chosen[:n_perturbed]]
    stats = {"draft": len(draft), "flags": dict(Counter(flags[r["id"]] for r in draft)),
             "kept": len(items), "per_family": dict(Counter(r["meta"]["family"] for r in items)),
             "perturbed": len(twins)}
    return items, twins, stats


def _write_eval(path: str | Path, rows: list[dict], name: str, stats: dict[str, Any], parent: str | Path) -> None:
    from dpt.common.hashing import sha256_file
    from dpt.common.manifest import build_manifest, write_manifest

    write_jsonl(path, rows)
    write_manifest(build_manifest(path, name, "v1", parents={Path(parent).name: sha256_file(parent)},
                                  stage_stats=[{"stage": k, "count": v} for k, v in stats.items()
                                               if isinstance(v, int)],
                                  notes="generated from held-out solver families (dpt.data.gen.t1); wording "
                                        "checked by the hosted model, flagged items dropped"))


# --------------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="T1 seismology instances and the D-Num eval set")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True, help="training candidates JSONL")
    b.add_argument("--heldout-out", default=None, help="D-Num draft JSONL (eval-secure storage)")
    b.add_argument("--per-family", type=int, default=800)
    b.add_argument("--n-heldout", type=int, default=360, help="draft size (> D-Num's 300: flagged items drop)")
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--bank", default="v1",
                   help="family bank: v1 (2026-10), hard (dpt.data.gen.seismology_hard) "
                        "or long (dpt.data.gen.seismology_long, 2026-11-07)")
    b.add_argument("--set-name", default="d_num", help="eval set of the held-out draft (hard bank: d_num_hard)")
    fz = sub.add_parser("finalize-dnum")
    fz.add_argument("--draft", required=True)
    fz.add_argument("--wording", required=True, help="CSV from `python -m dpt.eval.drafts wording`")
    fz.add_argument("--out", required=True)
    fz.add_argument("--perturbed-out", required=True)
    fz.add_argument("--n-items", type=int, default=300)
    fz.add_argument("--n-perturbed", type=int, default=200)
    fz.add_argument("--seed", type=int, default=1)
    fz.add_argument("--set-name", default="d_num")
    args = ap.parse_args(argv)

    if args.cmd == "build":
        rows = build_train(by_split("train", args.bank), args.per_family, args.seed)
        n = write_candidates(args.out, rows)
        per = Counter(r["meta"]["family"] for r in rows)
        short = {k: v for k, v in per.items() if v < args.per_family}
        write_json(str(args.out) + ".summary.json", {"candidates": n, "families": len(per), "per_family": per,
                                                     "below_per_family": short, "seed": args.seed})
        log.info("%d T1 candidates from %d families → %s (%d families have fewer distinct questions: %s)",
                 n, len(per), args.out, len(short), sorted(short))
        if args.heldout_out:
            draft = build_dnum_draft(by_split("heldout", args.bank), args.n_heldout, args.seed, args.set_name)
            write_jsonl(args.heldout_out, draft)
            log.info("%d D-Num draft items from %d held-out families → %s", len(draft),
                     len({r['meta']['family'] for r in draft}), args.heldout_out)
        return 0

    draft = list(read_jsonl(args.draft))
    items, twins, stats = finalize_dnum(draft, read_wording(args.wording), args.n_items, args.n_perturbed, args.seed,
                                        args.set_name)
    if len(items) < args.n_items:
        log.error("only %d D-Num items passed the wording check (need %d): build a larger draft "
                  "(--n-heldout) and rerun the wording check. Nothing written.", len(items), args.n_items)
        return 1
    _write_eval(args.out, items, args.set_name, stats, args.draft)
    _write_eval(args.perturbed_out, twins, f"{args.set_name}_perturbed", {"perturbed": len(twins)}, args.draft)
    log.info("D-Num: %s", stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
