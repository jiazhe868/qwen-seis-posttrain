"""CLI that runs the RL problem-bank pipeline and the SFT mixture build, stage by stage.

    python -m dpt.data.pipeline --config data/pipeline.yaml [--stages a,b,...] [--offline] [key=value ...]

Stages (default: all except ``probe``)::

    ingest → quality → dedup → decontam → verify → difficulty → publish      (RL problem bank)
    sft                                                                    (SFT mixture)
    probe                                                                  (base memorization probes)

Every RL stage reads the previous stage's output from ``<output_dir>/stages/``.
It writes its own JSONL with a manifest (whose parent is the previous stage's
content hash), a ``removed/<stage>.jsonl`` with per-item reasons, and a row
in ``funnel.json``. Any stage can therefore be re-run alone (``--stages
verify,difficulty,publish``), and the final manifest carries the whole funnel
(report §处理流水线 step 6).

``--offline`` swaps every model call for the deterministic fakes in
``dpt.data.backends``. It is a full plumbing test on a laptop. The *numbers*
it produces are meaningless.

Where to run what (README §3):
    workstation (4×A100-40GB): all stages. Serve Qwen3.8-27B with vLLM TP4 for
        ``difficulty``. Point ``verify.families`` / ``backends.embed`` /
        ``backends.judge`` at the external inference services.
    cluster: not needed for data. Keep GPU time for SFT and RL.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

from dpt.common.config import REPO_ROOT, config_hash, load_config, to_container
from dpt.common.io import read_json, read_jsonl, write_json, write_jsonl
from dpt.common.logging import get_logger
from dpt.common.manifest import build_manifest, load_manifest, manifest_path_for, write_manifest
from dpt.data import backends as B
from dpt.data.answer_verify import (
    AuditConfig,
    VerifyConfig,
    audit_verify,
    expert_queue_record,
    unmarked_reply,
    verify_problem,
)
from dpt.data.decontam import (
    DecontamConfig,
    Decontaminator,
    contamination_report,
    load_ref_items,
    prefix_completion_probe,
    scan_canaries,
)
from dpt.data.dedup import cross_dedup, instance_dedup, near_duplicate_dedup
from dpt.data.difficulty import (DifficultyAuditConfig, DifficultyConfig, calibrate, calibrate_audit,
                                 calibrate_two_stage)
from dpt.data.ingest import ingest_files
from dpt.data.quality import QualityConfig, filter_by_quality
from dpt.data.rl_build import dev_candidates, export_rl_bank
from dpt.data.schema import Problem, SFTSample
from dpt.data.sft_build import SFTBuildConfig, build_sft, load_subsets
from dpt.data.solvers import solve

log = get_logger(__name__)

RL_STAGES = ["ingest", "quality", "dedup", "decontam", "verify", "difficulty", "publish"]
ALL_STAGES = RL_STAGES + ["sft", "probe"]
DEFAULT_STAGES = RL_STAGES + ["sft"]


def _repo_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


class Context:
    """Resolved config, output layout and lazily constructed backends for one pipeline run."""

    def __init__(self, cfg: dict[str, Any], *, offline: bool, cfg_hash: str):
        self.cfg = cfg
        self.dp = cfg["data_pipeline"]
        self.offline = offline
        self.cfg_hash = cfg_hash
        self.out = _repo_path(self.dp["output_dir"])
        self.stage_dir = self.out / "stages"
        self.removed_dir = self.out / "removed"
        self.report_dir = self.out / "reports"
        self._decontaminator: Decontaminator | None = None

    # -- stage files --------------------------------------------------------------------------
    def stage_path(self, name: str) -> Path:
        return self.stage_dir / f"{RL_STAGES.index(name):02d}_{name}.jsonl"

    def previous(self, name: str) -> str:
        return RL_STAGES[RL_STAGES.index(name) - 1]

    def load_stage(self, name: str) -> list[Problem]:
        path = self.stage_path(name)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing — run stage {name!r} first")
        return [Problem.from_dict(r) for r in read_jsonl(path)]

    def save_stage(self, name: str, problems: list[Problem], removed: list[dict[str, Any]],
                   reasons: dict[str, int], n_in: int, extra: dict[str, Any] | None = None) -> None:
        path = self.stage_path(name)
        write_jsonl(path, [p.to_dict() for p in problems])
        write_jsonl(self.removed_dir / f"{name}.jsonl", removed)
        row = {"stage": name, "in": n_in, "out": len(problems), "removed": n_in - len(problems),
               "reasons": dict(reasons), **(extra or {})}
        funnel = self.update_funnel(row)
        parents = {}
        if name != RL_STAGES[0]:
            prev = self.stage_path(self.previous(name))
            if manifest_path_for(prev).exists():
                parents[self.previous(name)] = load_manifest(prev).content_hash
        write_manifest(build_manifest(path, name=f"rl_bank.{name}", version=self.dp["version"],
                                      config_hash=self.cfg_hash, parents=parents, stage_stats=funnel))
        log.info("stage %-10s %6d → %6d  %s", name, n_in, len(problems), dict(reasons))

    def update_funnel(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        path = self.out / "funnel.json"
        funnel = [r for r in (read_json(path) if path.exists() else []) if r["stage"] != row["stage"]]
        funnel.append(row)
        order = {s: i for i, s in enumerate(ALL_STAGES)}
        funnel.sort(key=lambda r: order.get(r["stage"], 99))
        write_json(path, funnel)
        return funnel

    # -- backends -----------------------------------------------------------------------------
    def client(self, spec: dict[str, Any]) -> B.OpenAICompatClient:
        # ``url_file`` + ``wait_s``: follow a serve.slurm endpoint across server windows (OpenAICompatClient).
        # ``timeout_s``: a request may wait that long in the server's queue before it counts as failed.
        return B.OpenAICompatClient(spec["base_url"], spec["model"],
                                    max_concurrency=int(self.dp["backends"].get("max_concurrency", 64)),
                                    timeout=float(spec.get("timeout_s", 1800)),
                                    url_file=spec.get("url_file"), wait_s=float(spec.get("wait_s", 0)))

    def cached(self, gen: Callable[..., list[str]], stage: str, spec: dict[str, Any]) -> Callable[..., list[str]]:
        # Model replies survive a crash or a closed server window: a rerun of the stage reuses them
        # (backends.CachedGenerator; delete <output_dir>/cache to sample afresh).
        label = f"{stage}:{spec.get('name', 'base')}:{spec['model']}"
        return B.CachedGenerator(gen, self.out / "cache" / f"{stage}_{spec.get('name', 'base')}.jsonl", label)

    def base_generator(self) -> Callable[..., list[str]]:
        if self.offline:
            return B.FakeGenerator()
        bk = self.dp["backends"]
        tok = B.load_hf_tokenizer(bk["tokenizer"]) if bk.get("tokenizer") else None
        return self.cached(B.VLLMGenerator(self.client(bk["base"]), tokenizer=tok), "difficulty", bk["base"])

    def grade_fn(self) -> Callable[[Problem, str], float]:
        if self.offline:
            return B.fake_grade
        try:
            from dpt.rewards.router import compute_score  # the *training* grader (verl signature)
        except ImportError as e:  # pragma: no cover - depends on dpt.rewards being installed
            raise RuntimeError("difficulty calibration needs dpt.rewards.router.compute_score") from e

        def _grade(p: Problem, response: str) -> float:
            # Correctness as RL counts it (``acc``), not the final score. That score subtracted a length penalty
            # estimated from the characters of the whole reply, so a long right answer could count as a failure;
            # response_length 0 turns the penalty off, as dpt.rewards.validate does. ``question`` is what the T3
            # judge needs (without it a report scores 0 with bad_ground_truth).
            out = compute_score(p.task, response, json.dumps(p.answer, sort_keys=True),
                                {"uid": p.uid, "task": p.task, "pool": p.pool, "question": p.question,
                                 "response_length": 0})
            if not isinstance(out, dict):
                return float(out)
            for key in ("acc", "base_score", "score"):
                if isinstance(out.get(key), (int, float)):
                    return float(out[key])
            return 0.0

        return _grade

    def family_fns(self) -> dict[str, Callable[[Problem, int], list[str]]]:
        from dpt.data.prompts import messages_for_problem

        families = self.dp["verify"]["families"]
        if self.offline:
            def truth(p: Problem) -> float | None:
                gen = p.meta.get("generator")
                return solve(gen) if gen else float(p.answer["value"])

            rates = [0.1, 0.15, 0.2]
            return {f["name"]: B.FakeSolverFamily(f["name"], truth, rates[i % 3])
                    for i, f in enumerate(families)}
        max_tokens = int(self.dp["verify"].get("max_tokens", 16384))

        def make_fn(f: dict[str, Any]) -> Callable[[Problem, int], list[str]]:
            # A family with ``tokenizer`` + ``effort`` (the Qwen teacher) is rendered locally at that effort;
            # without them (another model family) no Qwen-specific effort field is sent.
            tok = B.load_hf_tokenizer(f["tokenizer"]) if f.get("tokenizer") else None
            gen = self.cached(B.VLLMGenerator(self.client(f), tokenizer=tok), "verify", f)
            effort = f.get("effort") if tok is not None else None
            if tok is not None:                # rendered here: </think> marks the end of thinking
                return lambda p, n: gen(messages_for_problem(p), n=n, effort=effort, max_tokens=max_tokens)
            # another family's chat replies may not mark it (Gemma): read them whole (answer_verify.unmarked_reply)

            def read_whole(p: Problem, n: int) -> list[str]:
                return [unmarked_reply(r) for r in gen(messages_for_problem(p), n=n, effort=effort,
                                                       max_tokens=max_tokens)]
            return read_whole

        return {f["name"]: make_fn(f) for f in families}

    def decontaminator(self) -> Decontaminator:
        if self._decontaminator is None:
            inp = self.dp["inputs"]
            protected = {**inp["eval_sets"], **public_benchmarks(inp)}
            colls = {k: str(_repo_path(v)) for k, v in protected.items()}
            refs = load_ref_items(colls)
            if self.offline:
                emb, judge = B.FakeEmbedder(), B.FakeParaphraseJudge()
            else:
                ecfg = self.dp["backends"]["embed"]
                emb = (B.LocalEmbedder(ecfg["model"], pooling=ecfg.get("pooling", "last"),
                                       batch_size=int(ecfg.get("batch_size", 64)))
                       if ecfg.get("local") else self.client(ecfg))
                judge = B.LLMParaphraseJudge(self.client(self.dp["backends"]["judge"]))
            self._decontaminator = Decontaminator(refs, DecontamConfig.from_dict(self.dp["decontam"]),
                                                  embedder=emb, judge=judge)
        return self._decontaminator


def public_benchmarks(inp: dict[str, Any]) -> dict[str, str]:
    """Protected public collections: ``public_benchmarks_dir`` (every ``*.jsonl`` in it, named by file stem)
    replaces the ``public_benchmarks`` map when set; the map is the toy default."""
    d = inp.get("public_benchmarks_dir")
    if d:
        files = sorted(_repo_path(d).glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"no *.jsonl in public_benchmarks_dir {d}")
        return {f.stem: str(f) for f in files}
    return dict(inp.get("public_benchmarks") or {})


# =============================================================================================
# Stages
# =============================================================================================
def stage_ingest(ctx: Context) -> None:
    ing = ctx.dp["ingest"]
    paths = [str(_repo_path(p)) for p in ctx.dp["inputs"]["rl_candidates"]]
    res = ingest_files(paths, parser=ing["parser"], license_allowlist=ing["license_allowlist"])
    n_in = len(res.problems) + len(res.dropped)
    removed = [{"uid": (d["record"].get("uid") or ""), "reason": d["reason"]} for d in res.dropped]
    ctx.save_stage("ingest", res.problems, removed, res.reasons, n_in)


def stage_quality(ctx: Context) -> None:
    probs = ctx.load_stage("ingest")
    res = filter_by_quality(probs, lambda p: p.question, QualityConfig.from_dict(ctx.dp["quality"]))
    ctx.save_stage("quality", res.kept, res.dropped, res.reasons, len(probs))


def stage_dedup(ctx: Context) -> None:
    probs = ctx.load_stage("quality")
    d = ctx.dp["dedup"]
    kw = dict(threshold=d["threshold"], num_perm=d["num_perm"], shingle_k=d["shingle_k"], seed=d["seed"])
    removed: list[dict[str, Any]] = []
    reasons: Counter = Counter()
    refs: list[tuple[str, str]] = []
    for name, path in (ctx.dp["inputs"].get("reference_banks") or {}).items():
        for i, r in enumerate(read_jsonl(_repo_path(path))):
            refs.append((f"{name}/{r.get('uid', i)}", r["question"]))
    if refs:  # across banks first: other banks have priority
        cres = cross_dedup([p.question for p in probs], refs, **kw)
        for r in cres.removed:
            removed.append({"uid": probs[r["index"]].uid, "reason": "cross_bank_duplicate",
                            "duplicate_of": r["duplicate_of"], "jaccard": r["jaccard"]})
        reasons["cross_bank_duplicate"] = len(cres.removed)
        probs_kept = [probs[i] for i in cres.kept]
    else:
        probs_kept = probs
    # Generated problems are deduplicated by instance, not by text (dpt.data.dedup.instance_dedup): T1 instances
    # of one template differ only in their numbers (identity: solver + inputs), and T2 copies of one spec
    # differ only in their examples and hidden tests (identity: the exact question).
    by_instance = bool(d.get("skip_programmatic", True))

    def identity(p: Problem) -> dict[str, Any] | None:
        g = p.meta.get("generator")
        if isinstance(g, dict):
            return g
        if p.meta.get("variant_key"):
            return {"name": f"text:{p.uid}", "params": {}}
        return None

    prog = [p for p in probs_kept if by_instance and identity(p) is not None]
    rest = [p for p in probs_kept if not (by_instance and identity(p) is not None)]
    ires = instance_dedup([identity(p) for p in prog])
    for r in ires.removed:
        removed.append({"uid": prog[r["index"]].uid, "reason": "near_duplicate",
                        "duplicate_of": prog[r["duplicate_of"]].uid, "match": r["match"]})
    prog_kept = [prog[i] for i in ires.kept]
    wres = near_duplicate_dedup([p.question for p in rest], **kw)
    for r in wres.removed:
        removed.append({"uid": rest[r["index"]].uid, "reason": "near_duplicate",
                        "duplicate_of": rest[r["duplicate_of"]].uid, "jaccard": r["jaccard"]})
    reasons["near_duplicate"] = len(wres.removed) + len(ires.removed)
    ctx.save_stage("dedup", prog_kept + [rest[i] for i in wres.kept], removed, reasons, len(probs))


def stage_decontam(ctx: Context) -> None:
    probs = ctx.load_stage("dedup")
    dc = ctx.dp["decontam"]
    canary_hits = scan_canaries(((p.uid, p.to_dict()) for p in probs), dc.get("canaries", []))
    leaked = {h["key"] for h in canary_hits}
    if leaked:
        log.error("CANARY found in %d candidates: an eval file leaked into a source. Investigate!",
                  len(leaked))
    clean = [p for p in probs if p.uid not in leaked]
    dec = ctx.decontaminator()
    res = dec.check_problems(clean)
    removed = [{"uid": u, "reason": "canary"} for u in sorted(leaked)]
    removed += [{"uid": r["uid"], "reason": f"decontam:{r['layer']}", "match": r["match"],
                 "score": r["score"]} for r in res.removed]
    reasons = Counter({f"decontam:{k}": v for k, v in res.layer_counts.items()})
    if leaked:
        reasons["canary"] = len(leaked)
    audit = dec.audit_sample(res, {p.uid: p.question for p in clean})
    audit_path = _repo_path(dc["audit_dir"]) / "semantic_audit.jsonl"
    write_jsonl(audit_path, audit)  # contains eval text: keep on secure storage
    rep = contamination_report(rl=res, canary_hits=canary_hits, canary_scanned=len(probs),
                               semantic_audit={"sampled": len(audit), "path": str(audit_path),
                                               "confirmed_rate": None})
    write_json(ctx.report_dir / "contamination_rl.json", {**rep["rl"], "canary": rep["canary"]})
    ctx.save_stage("decontam", res.kept, removed, reasons, len(probs))


def stage_verify(ctx: Context) -> None:
    probs = ctx.load_stage("decontam")
    vcfg = VerifyConfig.from_dict(ctx.dp["verify"])
    fams = ctx.family_fns()
    if ctx.dp["verify"].get("mode", "all") == "audit":   # per template (answer_verify.AuditConfig)
        acfg = AuditConfig.from_dict({**(ctx.dp["verify"].get("audit") or {}), "seed": ctx.dp["seed"]})
        res = audit_verify(probs, fams, vcfg, acfg,
                           max_workers=int(ctx.dp["backends"].get("max_concurrency", 16)))
        res.queue.sort(key=lambda r: (r["priority"] != "high", r["uid"]))
        write_jsonl(ctx.out / "expert_queue.jsonl", res.queue)
        write_json(ctx.report_dir / "verify_groups.json", res.groups)
        reasons = Counter(r["reason"].split(":")[0] + ":" + r["reason"].split(":")[1] for r in res.removed)
        dropped = [k for k, g in res.groups.items() if not g["kept"]]
        ctx.save_stage("verify", res.kept, res.removed, reasons, len(probs),
                       {"expert_queue": len(res.queue), "groups": len(res.groups), "groups_dropped": dropped})
        return
    kept, removed, queue = [], [], []
    reasons: Counter = Counter()
    for p in probs:
        out = verify_problem(p, fams, vcfg)
        if out.status == "rejected":
            removed.append({"uid": p.uid, "reason": f"verify:{out.reason}"})
            queue.append(expert_queue_record(p, out))
            reasons[f"verify:{out.reason.split(':')[0]}"] += 1
        else:
            if out.status == "accepted":
                p.meta["verified"] = {"votes": out.votes, "shares": out.shares}
            kept.append(p)
    queue.sort(key=lambda r: (r["priority"] != "high", r["uid"]))
    write_jsonl(ctx.out / "expert_queue.jsonl", queue)
    ctx.save_stage("verify", kept, removed, reasons, len(probs), {"expert_queue": len(queue)})


def stage_difficulty(ctx: Context) -> None:
    probs = ctx.load_stage("verify")
    dcfg = DifficultyConfig.from_dict(ctx.dp["difficulty"])
    extra: dict[str, Any] = {}
    mode = ctx.dp["difficulty"].get("mode", "all")
    if mode in ("audit", "audit_all"):   # per group (difficulty.DifficultyAuditConfig); audit_all then every problem
        acfg = DifficultyAuditConfig.from_dict(ctx.dp["difficulty"].get("audit"))
        fn = calibrate_audit if mode == "audit" else calibrate_two_stage
        res, groups = fn(probs, ctx.base_generator(), ctx.grade_fn(), dcfg, acfg)
        write_json(ctx.report_dir / "difficulty_groups.json", groups)
        extra["groups"] = dict(Counter(g["decision"] for g in groups.values()))
    else:
        res = calibrate(probs, ctx.base_generator(), ctx.grade_fn(), dcfg)
    st = res.stats()
    reasons = Counter(d["reason"] for d in res.dropped)
    ctx.save_stage("difficulty", res.kept, res.dropped, reasons, len(probs),
                   {"hard_pool": st["hard_pool"], "histogram": st["histogram"], **extra})


DEV_CANDIDATES = "rl_dev_candidates.jsonl"


def write_dev_candidates(ctx: Context) -> Path:
    """``<output_dir>/rl_dev_candidates.jsonl``: the uids publish draws the RL dev split from
    (``publish.dev_candidates``, rl_build.dev_candidates), fixed from the verify stage's output so the SFT
    mix can exclude them before the difficulty pass has run."""
    pub = ctx.dp["publish"]
    probs = [p for p in ctx.load_stage("verify") if p.pool != "hard"]
    uids = dev_candidates(probs, fraction=float(pub.get("dev_fraction", 0.0)),
                          min_per_task=int(pub.get("dev_min_per_task", 1)), seed=int(ctx.dp["seed"]))
    path = ctx.out / DEV_CANDIDATES
    write_jsonl(path, [{"uid": u} for u in sorted(uids)])
    log.info("RL dev candidates: %d of %d problems -> %s", len(uids), len(probs), path)
    return path


def stage_publish(ctx: Context) -> None:
    probs = ctx.load_stage("difficulty")
    pub = ctx.dp["publish"]
    among = None
    if pub.get("dev_candidates"):        # the dev split comes only from the candidates the SFT mix excluded
        among = {r["uid"] for r in read_jsonl(ctx.out / DEV_CANDIDATES)}
    info = export_rl_bank(probs, ctx.out, effort=pub["effort"], dev_fraction=float(pub.get("dev_fraction", 0.0)),
                          dev_min_per_task=int(pub.get("dev_min_per_task", 1)), seed=int(ctx.dp["seed"]),
                          dev_copy_if_empty=bool(pub.get("dev_copy_if_empty", False)), dev_among=among)
    rows = []
    for files in info["files"].values():
        rows.extend(read_jsonl(files[0]))
    hits = scan_canaries(((r["extra_info"]["uid"], r) for r in rows), ctx.dp["decontam"].get("canaries", []))
    if hits:  # should be impossible after the decontam stage; never publish a leak
        raise RuntimeError(f"canary found in published rows: {hits[:3]}")
    funnel = read_json(ctx.out / "funnel.json")
    for name, files in info["files"].items():
        src = ctx.stage_path("difficulty")
        write_manifest(build_manifest(files[0], name=name, version=ctx.dp["version"],
                                      config_hash=ctx.cfg_hash,
                                      parents={"difficulty": load_manifest(src).content_hash},
                                      stage_stats=funnel))
    assemble_contamination_report(ctx)
    write_json(ctx.out / "publish_summary.json", {"counts": info["counts"], "files": info["files"]})
    log.info("published %s", info["counts"])


def assemble_contamination_report(ctx: Context) -> None:
    """Merge the per-stage contamination outputs into ``reports/contamination_report.json``.

    Called by every stage that contributes (publish, sft, probe), so the merged
    report is current no matter which subset of stages ran last.
    """
    merged: dict[str, Any] = {}
    for key, name in (("rl", "contamination_rl.json"), ("sft", "contamination_sft.json"),
                      ("memorization_probes", "probes.json")):
        p = ctx.report_dir / name
        if p.exists():
            merged[key] = read_json(p)
    write_json(ctx.report_dir / "contamination_report.json", merged)


def filter_by_problem(samples: list[SFTSample], keep_from: str | None,
                      exclude_from: Sequence[str]) -> tuple[list[SFTSample], dict[str, int]]:
    """Keep samples of bank problems that survived verification and are not in the RL dev split.

    Teacher sampling may start from the decontam stage before verification has finished (2026-10-24), so
    a sample's ``source.problem_uid`` must be in ``keep_from`` (the verify stage's output) and in none of
    ``exclude_from`` (``rl_dev.jsonl``). Samples without a source problem pass unchanged.
    """
    if not keep_from and not exclude_from:
        return samples, {"kept": len(samples)}
    uid = lambda r: r.get("uid") or (r.get("extra_info") or {}).get("uid")  # noqa: E731
    keep = {uid(r) for r in read_jsonl(_repo_path(keep_from))} if keep_from else None
    drop = {uid(r) for f in exclude_from for r in read_jsonl(_repo_path(f))}
    out, removed = [], Counter()
    for s in samples:
        pu = s.source.get("problem_uid")
        if pu is not None and keep is not None and pu not in keep:
            removed["problem_not_verified"] += 1
        elif pu is not None and pu in drop:
            removed["rl_dev_problem"] += 1
        else:
            out.append(s)
    return out, {"kept": len(out), **removed}


def stage_sft(ctx: Context) -> None:
    sc = ctx.dp["sft"]
    paths = {k: str(_repo_path(v)) for k, v in ctx.dp["inputs"]["sft_subsets"].items()}
    samples, schema_errors = load_subsets(paths)
    samples, problem_filter = filter_by_problem(samples, sc.get("keep_problems_from"),
                                                sc.get("exclude_problems_from") or [])
    if ctx.offline:
        tok = B.JinjaCharTokenizer(_repo_path(sc["offline_chat_template"]))
    else:
        tok = B.load_hf_tokenizer(sc["tokenizer"])
    res = build_sft(samples, tok, SFTBuildConfig.from_dict(sc),
                    quality=QualityConfig.from_dict(ctx.dp["quality"]), decontaminator=ctx.decontaminator())
    out = ctx.out / "sft" / "sft_train.jsonl"
    write_jsonl(out, [s.to_dict() for s in res.samples])
    write_jsonl(ctx.out / "sft" / "dropped.jsonl", res.dropped)
    stats = [{"stage": "schema", "errors": dict(schema_errors)}, {"stage": "problem_filter", **problem_filter}] \
        + res.stage_stats
    write_manifest(build_manifest(out, name="sft_train", version=ctx.dp["version"], config_hash=ctx.cfg_hash,
                                  stage_stats=stats))
    write_json(ctx.out / "sft" / "sft_report.json", {"funnel": stats, "mixture": res.mixture,
                                                     "padding": res.padding})
    canaries = ctx.dp["decontam"].get("canaries", [])
    canary_hits = scan_canaries(((s.uid, s.to_dict()) for s in res.samples), canaries)
    if canary_hits:
        raise RuntimeError(f"canary found in SFT output: {canary_hits[:3]}")
    rep = contamination_report(sft=res.decontam, canary_hits=canary_hits, canary_scanned=len(res.samples))
    write_json(ctx.report_dir / "contamination_sft.json", {**rep.get("sft", {}), "canary": rep["canary"]})
    assemble_contamination_report(ctx)
    log.info("sft: %d samples, mixture=%s, padding=%s", len(res.samples), res.mixture, res.padding)


def _greedy_completion_fn(client: B.OpenAICompatClient) -> Callable[[str], str]:
    """Raw (no chat template) greedy continuation, which is how a memorization probe should query a model."""

    def complete(prefix: str) -> str:
        return client.complete(prefix, temperature=0.0, max_tokens=64)[0]

    return complete


def stage_probe(ctx: Context) -> None:
    pc = ctx.dp["probe"]
    pubs = public_benchmarks(ctx.dp["inputs"])
    results = {}
    for coll in pc["collections"]:
        texts = [r["question"] for r in read_jsonl(_repo_path(pubs[coll]))]
        if ctx.offline:
            complete = B.FakeCompleter(memorized=texts[:max(1, len(texts) // 2)])
        else:
            complete = _greedy_completion_fn(ctx.client(ctx.dp["backends"]["base"]))
        results[coll] = prefix_completion_probe(texts, complete, prefix_frac=pc["prefix_frac"],
                                                k_words=pc["k_words"], match_frac=pc["match_frac"],
                                                seed=ctx.dp["seed"])
    write_json(ctx.report_dir / "probes.json", results)
    assemble_contamination_report(ctx)
    log.info("probes: %s", results)


STAGE_FNS: dict[str, Callable[[Context], None]] = {
    "ingest": stage_ingest, "quality": stage_quality, "dedup": stage_dedup, "decontam": stage_decontam,
    "verify": stage_verify, "difficulty": stage_difficulty, "publish": stage_publish,
    "sft": stage_sft, "probe": stage_probe,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="data/pipeline.yaml")
    ap.add_argument("--stages", default=",".join(DEFAULT_STAGES),
                    help=f"comma-separated subset of {ALL_STAGES}")
    ap.add_argument("--offline", action="store_true", help="use deterministic fakes for all model calls")
    ap.add_argument("--dev-candidates", action="store_true",
                    help=f"write <output_dir>/{DEV_CANDIDATES} from the verify stage's output and exit")
    ap.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. data_pipeline.output_dir=/tmp/x")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, overrides=args.overrides)
    resolved = to_container(cfg)
    ctx = Context(resolved, offline=args.offline, cfg_hash=config_hash(cfg))
    ctx.out.mkdir(parents=True, exist_ok=True)
    write_json(ctx.out / "resolved_config.json",
               {"config_hash": ctx.cfg_hash, "offline": args.offline, "config": resolved})
    if args.dev_candidates:
        write_dev_candidates(ctx)
        return 0
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in STAGE_FNS]
    if unknown:
        ap.error(f"unknown stages {unknown}; choose from {ALL_STAGES}")
    for s in sorted(stages, key=ALL_STAGES.index):
        STAGE_FNS[s](ctx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
