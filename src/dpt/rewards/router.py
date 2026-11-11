"""verl entry points: dispatch each response to its task grader and apply length control.

verl calls (``reward.custom_reward_function``):

    compute_score(data_source, solution_str, ground_truth, extra_info=None, **kw) -> dict
    compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos, **kw) -> list[dict]

Per-response pipeline (each step fails closed):

1. **Truncated?** (``finish_reason == "length"`` or length ≥ the 24K cap) →
   score 0 with ``truncated=True``. The trainer drops these from the loss.
2. **Split thinking.** No ``</think>`` → 0 (``no_think_end``). Graders and the
   judge only ever see the final answer.
3. **Dispatch** by task: T1 numeric, T2 sandbox, T3 judge + citations, T4
   math or instruction following.
4. **Scan** the answer for known hack patterns (flags for monitoring).
5. **Overlong penalty**: linear from 20K to 24K tokens, down to −0.5
   (``rewards.length.apply_penalty``). Disable it when the framework already
   applies a token-exact penalty, as verl's ``reward_manager=dapo`` does, so
   it is not applied twice.

Output contract: every dict has *exactly* the keys in :data:`VERL_KEYS`,
whatever the task. verl turns per-sample extra-info dicts into columns. A key
present for one task and missing for another would make the columns ragged
and crash the step, or silently misalign them. Missing components are 0.0;
monitoring slices by ``task`` before aggregating.

Inputs from the data contract (``dpt.data.rl_build``): ``data_source`` is the
task id, ``ground_truth`` is a JSON string of ``Problem.answer``, and
``extra_info`` carries ``uid, task, …``. If the rollout adapter adds
``response_length`` (tokens) and ``finish_reason``, they are used. Otherwise
the length is estimated from characters, and that estimate flags truncation
only when ``rewards.length.truncation_from_estimate`` is true. T3/T4-ifeval also need
the prompt text (``extra_info["question"]``) for the judge and the relevance
gate.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from dpt.chat.thinking import split_thinking
from dpt.common.logging import get_logger
from dpt.common.types import T1_NUM, T2_CODE, T3_REPORT, T4_GENERAL, TASKS
from dpt.rewards.base import RewardResult, zero
from dpt.rewards.code_sandbox import (
    CodeConfig,
    HiddenTestStore,
    LocalSubprocessSandbox,
    RemoteSandboxClient,
    SandboxClient,
    SandboxPool,
    grade_code,
)
from dpt.rewards.citations import CitationResolver, CrossrefResolver, OfflineResolver
from dpt.rewards.hack_patterns import scan_text
from dpt.rewards.instruction import InstructionConfig, grade_instruction
from dpt.rewards.judge import JudgeConfig, ReportConfig, RubricJudge, grade_report
from dpt.rewards.length import LengthConfig, estimate_tokens, is_truncated, overlong_penalty
from dpt.rewards.numeric import NumericConfig, grade_math, grade_numeric

log = get_logger(__name__)

# Fixed output schema (see module docstring). Order is irrelevant; the set is the contract.
VERL_KEYS: tuple[str, ...] = (
    "score",
    "base_score",
    "acc",
    "reason",
    "truncated",
    "task",
    "flags",
    "response_length",
    "length_penalty",
    "grader_ms",
    "t1_rel_err",
    "t2_hidden_pass",
    "t2_visible_pass",
    "t2_visible_hidden_gap",
    "t2_timeout",
    "t2_infra_error",
    "t3_rubric",
    "t3_citation_factor",
    "t3_citations_total",
    "t3_citations_unverified",
    "t3_citation_errors",
    "t3_self_praise",
    "t3_length_ratio",
    "t3_length_penalty",
    "t4_if_frac",
    "t4_relevance",
)
_STR_KEYS = {"reason", "task", "flags"}
_BOOL_KEYS = {"truncated"}


def _resolve_task(data_source: str | None, extra_info: dict[str, Any]) -> str | None:
    for cand in (extra_info.get("task"), data_source):
        if not cand:
            continue
        cand = str(cand)
        if cand in TASKS:
            return cand
        for t in TASKS:  # tolerate prefixed sources like "dpt/t1_num"
            if cand.endswith(t):
                return t
    return None


def _as_dict(ground_truth: Any) -> dict[str, Any]:
    if isinstance(ground_truth, dict):
        return ground_truth
    if isinstance(ground_truth, (str, bytes)):
        obj = json.loads(ground_truth)
        if not isinstance(obj, dict):
            raise ValueError("ground_truth JSON is not an object")
        return obj
    raise ValueError(f"unsupported ground_truth type {type(ground_truth).__name__}")


class RewardRouter:
    """Holds the configured graders and their (lazily built) backends."""

    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        *,
        sandbox: SandboxClient | None = None,
        judge: RubricJudge | None = None,
        resolver: CitationResolver | None = None,
    ):
        self.cfg = dict(cfg or {})
        self.length_cfg = LengthConfig.from_dict(self.cfg.get("length"))
        self.numeric_cfg = NumericConfig.from_dict(self.cfg.get("t1_num"))
        self.code_cfg = CodeConfig.from_dict(self.cfg.get("t2_code"))
        self.report_cfg = ReportConfig.from_dict(self.cfg.get("t3_report"))
        self.instr_cfg = InstructionConfig.from_dict(self.cfg.get("t4_ifeval"))
        self.math_cfg = NumericConfig.from_dict(
            {**(self.cfg.get("t1_num") or {}), **(self.cfg.get("t4_math") or {})}
        )
        self._sandbox, self._judge, self._resolver = sandbox, judge, resolver
        self._lock = threading.Lock()

    # -- lazy backends --------------------------------------------------------------------------
    @property
    def sandbox(self) -> SandboxClient:
        with self._lock:
            if self._sandbox is None:
                sc = self.cfg.get("t2_code", {}).get("sandbox", {})
                if sc.get("mode", "local") == "remote":
                    token = os.environ.get(sc.get("token_env", "DPT_SANDBOX_TOKEN"))
                    urls = sc.get("urls") or [sc["url"]]
                    self._sandbox = SandboxPool([RemoteSandboxClient(u, token=token) for u in urls])
                else:
                    hd = sc.get("hidden_tests_dir")
                    self._sandbox = LocalSubprocessSandbox(
                        hidden_store=HiddenTestStore(hd) if hd else None,
                        cpu_s=int(sc.get("cpu_s", 10)),
                        mem_mb=int(sc.get("mem_mb", 2048)),
                        network_isolation=sc.get("network_isolation", "auto"),
                        python=sc.get("python") or sys.executable,
                        container_image=sc.get("container_image"),
                        container_runtime=sc.get("container_runtime", "apptainer"),
                        container_network=sc.get("container_network", "none"),
                    )
            return self._sandbox

    @property
    def judge(self) -> RubricJudge:
        with self._lock:
            if self._judge is None:
                self._judge = RubricJudge(JudgeConfig.from_dict(self.cfg.get("t3_report", {}).get("judge")))
            return self._judge

    @property
    def resolver(self) -> CitationResolver:
        with self._lock:
            if self._resolver is None:
                cc = self.cfg.get("t3_report", {}).get("citations", {})
                if cc.get("mode", "crossref") == "offline":
                    known = json.loads(Path(cc["known_works"]).read_text()) if cc.get("known_works") else []
                    self._resolver = OfflineResolver(known)
                else:
                    self._resolver = CrossrefResolver(
                        mailto=cc.get("mailto"),
                        cache_path=cc.get("cache_path"),
                        min_interval_s=float(cc.get("min_interval_s", 0.1)),
                        timeout_s=float(cc.get("timeout_s", 10)),
                    )
            return self._resolver

    # -- grading --------------------------------------------------------------------------------
    def grade(
        self,
        data_source: str | None,
        solution_str: str,
        ground_truth: Any,
        extra_info: dict[str, Any] | None = None,
        **kw: Any,
    ) -> RewardResult:
        extra_info = dict(extra_info or {})
        task = _resolve_task(data_source, extra_info)
        if task is None:
            return zero("unknown", "unknown_task")
        n_tokens = kw.get("response_length", extra_info.get("response_length"))
        finish_reason = kw.get("finish_reason", extra_info.get("finish_reason"))
        estimated = n_tokens is None
        if estimated:
            n_tokens = estimate_tokens(solution_str, self.length_cfg)
        n_tokens = int(n_tokens)

        # A char-based estimate may only flag truncation when explicitly allowed; with verl the
        # trainer derives truncation from the response mask instead (rewards.length.truncation_from_estimate).
        may_flag = finish_reason is not None or not estimated or self.length_cfg.truncation_from_estimate
        if may_flag and is_truncated(n_tokens, self.length_cfg, finish_reason=finish_reason):
            res = zero(task, "truncated", truncated=True)
            res.components["response_length"] = float(n_tokens)
            return res

        split = split_thinking(solution_str)
        if not split.has_end:
            res = zero(task, "no_think_end")
        else:
            try:
                gt = _as_dict(ground_truth)
            except (ValueError, json.JSONDecodeError):
                return zero(task, "bad_ground_truth")
            res = self._dispatch(task, split.answer, gt, extra_info)
            res.flags = sorted(set(res.flags) | set(scan_text(split.answer, task=task)))

        # rewards.length.apply_penalty=false when verl's dapo reward manager already applies the
        # token-exact overlong penalty; applying it here too would double-penalize.
        pen = overlong_penalty(n_tokens, self.length_cfg) if self.length_cfg.apply_penalty else 0.0
        res.components["base_score"] = res.score
        res.components["length_penalty"] = pen
        res.components["response_length"] = float(n_tokens)
        res.score = res.score + pen
        return res

    def _dispatch(
        self, task: str, answer: str, gt: dict[str, Any], extra_info: dict[str, Any]
    ) -> RewardResult:
        if task == T1_NUM:
            return grade_numeric(answer, gt, self.numeric_cfg)
        if task == T2_CODE:
            return grade_code(answer, gt, sandbox=self.sandbox, cfg=self.code_cfg, extra_info=extra_info)
        if task == T3_REPORT:
            question = extra_info.get("question") or gt.get("question")
            if not question:
                return zero(task, "bad_ground_truth")
            return grade_report(
                answer, gt, question=question, judge=self.judge, resolver=self.resolver, cfg=self.report_cfg
            )
        if task == T4_GENERAL:
            kind = gt.get("kind", "math")
            if kind == "ifeval":
                return grade_instruction(
                    answer, gt, prompt=extra_info.get("question") or "", cfg=self.instr_cfg
                )
            return grade_math(answer, gt, self.math_cfg)
        return zero(task, "unknown_task")


def to_verl_output(res: RewardResult) -> dict[str, Any]:
    """Project a RewardResult onto the fixed :data:`VERL_KEYS` schema."""
    d = res.to_verl_dict()
    out: dict[str, Any] = {}
    for k in VERL_KEYS:
        if k == "task":
            out[k] = res.task
        elif k == "flags":
            out[k] = ",".join(sorted(set(res.flags)))
        elif k == "base_score":
            out[k] = float(d.get("base_score", res.score))
        elif k in _STR_KEYS:
            out[k] = str(d.get(k, ""))
        elif k in _BOOL_KEYS:
            out[k] = bool(d.get(k, False))
        else:
            out[k] = float(d.get(k, 0.0))
    return out


# ---------------------------------------------------------------------------------------------
# Process-wide router (one per Ray reward worker) and the verl-facing functions
# ---------------------------------------------------------------------------------------------
_ROUTER: RewardRouter | None = None
_ROUTER_LOCK = threading.Lock()
CONFIG_ENV = "DPT_REWARDS_CONFIG"


def load_rewards_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load the ``rewards`` subtree, resolving ``${project.*}`` interpolations against project.yaml."""
    from omegaconf import OmegaConf

    from dpt.common.config import load_config

    path = path or os.environ.get(CONFIG_ENV) or "rl/rewards.yaml"
    cfg = load_config(path)
    sub = cfg.get("rewards")
    if sub is None:
        raise KeyError(f"{path} has no top-level 'rewards' section")
    out = OmegaConf.to_container(sub, resolve=True)
    assert isinstance(out, dict)
    return out


def get_router() -> RewardRouter:
    global _ROUTER
    with _ROUTER_LOCK:
        if _ROUTER is None:
            _ROUTER = RewardRouter(load_rewards_config())
        return _ROUTER


def set_router(router: RewardRouter | None) -> None:
    """Install a router explicitly (tests, or adapters that build backends themselves)."""
    global _ROUTER
    with _ROUTER_LOCK:
        _ROUTER = router


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **kw: Any,
) -> dict[str, Any]:
    """verl single-sample reward entry point. Never raises."""
    try:
        res = get_router().grade(data_source, solution_str, ground_truth, extra_info, **kw)
    except Exception as e:  # last-resort fail closed (router construction errors etc.)
        log.error("compute_score failed closed: %s: %s", type(e).__name__, e)
        res = zero(str(data_source), f"exception:{type(e).__name__}")
    return to_verl_output(res)


def compute_score_batch(
    data_sources: Sequence[str],
    solution_strs: Sequence[str],
    ground_truths: Sequence[Any],
    extra_infos: Sequence[dict[str, Any] | None] | None = None,
    **kw: Any,
) -> list[dict[str, Any]]:
    """verl batch reward entry point; sandbox/judge I/O overlaps in a thread pool. Never raises."""
    n = len(solution_strs)
    extra_infos = list(extra_infos) if extra_infos is not None else [None] * n
    try:
        workers = int(get_router().cfg.get("max_workers", 32))
    except Exception:
        workers = 32
    per_sample_kw = [
        {k: v[i] for k, v in kw.items() if isinstance(v, (list, tuple)) and len(v) == n} for i in range(n)
    ]
    with ThreadPoolExecutor(max_workers=max(1, min(workers, n))) as pool:
        futs = [
            pool.submit(
                compute_score,
                data_sources[i],
                solution_strs[i],
                ground_truths[i],
                extra_infos[i],
                **per_sample_kw[i],
            )
            for i in range(n)
        ]
        return [f.result() for f in futs]
