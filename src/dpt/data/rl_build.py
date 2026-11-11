"""Export the verified RL problem bank as verl training rows.

Each :class:`Problem` becomes one row in verl's RL dataset format::

    {
      "data_source": "t1_num",                       # routes to the task grader in dpt.rewards
      "prompt": [{"role": "system", ...}, {"role": "user", ...}],   # dpt.data.prompts.build_messages
      "ability": "t1_num",
      "reward_model": {"style": "rule", "ground_truth": "<json of Problem.answer>"},
      "extra_info": {"uid", "task", "pool", "split", "effort", "reference_length", "pass_at_8_medium",
                     "prompt_version", "index", "question"}
    }

Notes:

* ``ground_truth`` is a JSON *string*. Answer payloads differ by task, and a
  string column keeps the parquet schema uniform. Graders ``json.loads`` it.
* For T2, ``ground_truth`` carries only ``hidden_tests_id``. The hidden tests
  live on the sandbox cluster and never enter the training data or the
  policy's context (report §防范原则).
* ``extra_info`` has the same keys in every row (missing values are ``None``),
  so parquet schema inference stays stable.
* The main pool and the hard pool (0/8 at calibration) go to separate files.
  The curriculum adds the hard pool after step 150 (report §三段课程).
* ``extra_info.question`` is the raw problem text. The T3 judge needs it to
  grade a report against its prompt, and the T4 instruction grader uses it for
  the relevance gate that defeats filler text (report hack at step 60).
* A small **dev split** (``rl_dev``, ``split="dev"``) is carved deterministically
  out of the main pool. verl validates on it every 20 steps, and checkpoints are
  selected on dev only, never on the test sets (report §评测体系). Dev rows are
  also gold-graded by the independent eval graders (``dpt.verl_ext.reward_fn``),
  which feeds the proxy/gold divergence alert.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from dpt.common.hashing import short_hash
from dpt.common.io import write_jsonl
from dpt.common.logging import get_logger
from dpt.data.prompts import PROMPT_VERSION, messages_for_problem
from dpt.data.schema import Problem

log = get_logger(__name__)

EXTRA_INFO_KEYS = ("uid", "task", "pool", "split", "effort", "reference_length", "pass_at_8_medium",
                   "prompt_version", "index", "question")


def problem_to_verl_row(p: Problem, *, index: int, effort: str = "medium", split: str = "train") -> dict[str, Any]:
    extra = {
        "uid": p.uid,
        "task": p.task,
        "pool": p.pool,
        "split": split,
        "effort": effort,
        "reference_length": p.answer.get("reference_length"),
        "pass_at_8_medium": p.meta.get("pass_at_8_medium"),
        "prompt_version": PROMPT_VERSION,
        "index": index,
        "question": p.question,
    }
    assert tuple(extra) == EXTRA_INFO_KEYS
    return {
        "data_source": p.task,
        "prompt": messages_for_problem(p),
        "ability": p.task,
        "reward_model": {"style": "rule",
                         "ground_truth": json.dumps(p.answer, sort_keys=True, ensure_ascii=False)},
        "extra_info": extra,
    }


def write_rows(rows: Sequence[dict[str, Any]], path_stem: Path) -> list[Path]:
    """Write ``<stem>.jsonl`` always, and ``<stem>.parquet`` if pandas + pyarrow are installed."""
    out = [path_stem.with_suffix(".jsonl")]
    write_jsonl(out[0], rows)
    try:
        import pandas as pd
        import pyarrow  # noqa: F401

        pq = path_stem.with_suffix(".parquet")
        pd.DataFrame(list(rows)).to_parquet(pq, index=False)
        out.append(pq)
    except ImportError:
        log.warning("pyarrow not installed: wrote %s only (verl needs parquet: install the 'data' extra)",
                    out[0])
    return out


def select_dev(problems: Sequence[Problem], *, fraction: float, min_per_task: int = 1,
               seed: int = 0, among: set[str] | None = None) -> set[str]:
    """Deterministic, task-stratified dev split by hashed uid. Returns the dev uids.

    Never takes more than a quarter of a task's main-pool problems, so tiny banks (toy data)
    still keep training data for every task. With ``among`` (:func:`dev_candidates`), each task's
    split keeps its size but is drawn only from those uids, so it is never anything else.
    """
    by_task: dict[str, list[Problem]] = {}
    for p in problems:
        by_task.setdefault(p.task, []).append(p)
    dev: set[str] = set()
    for _, group in sorted(by_task.items()):
        ranked = sorted(group, key=lambda p: short_hash(f"{seed}|dev|{p.uid}"))
        k = min(max(min_per_task, round(fraction * len(group))), len(group) // 4)
        if among is not None:
            ranked = [p for p in ranked if p.uid in among]
        dev.update(p.uid for p in ranked[:k])
    return dev


def dev_candidates(problems: Sequence[Problem], *, fraction: float, min_per_task: int = 1,
                   seed: int = 0, factor: int = 2) -> set[str]:
    """The uids a later :func:`select_dev` (``among=``) draws the dev split from: the same hash order, to
    ``factor`` × the depth.

    Fixed right after verification. The SFT mix then excludes every candidate and need not wait for the
    difficulty pass, which takes about a day of base-model time on the cluster (2026-10-24), while the dev
    split is still chosen after it. The difficulty pass drops problems independently of the hash, so the
    split it leaves reaches about ``fraction`` deep into this order, well inside the candidates.
    """
    return select_dev(problems, fraction=fraction * factor, min_per_task=min_per_task * factor, seed=seed)


def export_rl_bank(
    problems: Sequence[Problem], out_dir: str | Path, *, effort: str = "medium",
    dev_fraction: float = 0.0, dev_min_per_task: int = 1, seed: int = 0, dev_copy_if_empty: bool = False,
    dev_among: set[str] | None = None,
) -> dict[str, Any]:
    """Write ``rl_main``, ``rl_hard`` (and ``rl_dev`` when ``dev_fraction > 0``); return files and counts.

    ``dev_among``: draw the dev split only from these uids (:func:`dev_candidates`).

    ``dev_copy_if_empty`` is for the toy bank only. It is too small for a held-out split, so
    ``rl_dev`` becomes a relabeled copy (``split="dev"``) of ``rl_main``. That exercises the dev-only
    code path (gold grader → ``gold/<task>``, ``dev/d_num``) in the smoke test. The scores are
    meaningless because dev overlaps training; never set this for real data.
    """
    out_dir = Path(out_dir)
    main_all = [p for p in problems if p.pool != "hard"]
    hard = [p for p in problems if p.pool == "hard"]
    dev_uids = select_dev(main_all, fraction=dev_fraction, min_per_task=dev_min_per_task, seed=seed,
                          among=dev_among) if dev_fraction > 0 else set()
    main = [p for p in main_all if p.uid not in dev_uids]
    dev = [p for p in main_all if p.uid in dev_uids]
    if dev_copy_if_empty and dev_fraction > 0 and not dev:
        log.warning("dev split empty (bank too small); rl_dev = relabeled copy of rl_main (smoke only)")
        dev = list(main)
    files: dict[str, list[str]] = {}
    counts: dict[str, dict[str, int]] = {}
    for name, group, split in (("rl_main", main, "train"), ("rl_hard", hard, "train"), ("rl_dev", dev, "dev")):
        if name == "rl_dev" and not dev:
            continue
        rows = [problem_to_verl_row(p, index=i, effort=effort, split=split) for i, p in enumerate(group)]
        files[name] = [str(x) for x in write_rows(rows, out_dir / name)]
        c: dict[str, int] = {}
        for p in group:
            c[p.task] = c.get(p.task, 0) + 1
        counts[name] = dict(sorted(c.items()))
    return {"files": files, "counts": counts}
