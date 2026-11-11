"""Pairwise judging for D-Report: candidate vs base, both orders, plus expert blind review.

Protocol (report §评测协议, §多任务 RL 难调):

* Every item is judged **twice**, once with the candidate in position A and
  once in position B. A win counts only when both orders prefer the
  candidate, a loss only when both prefer the base. Everything else is a tie.
  This cancels position bias. Win rate = (wins + ½·ties) / n, so base vs
  itself scores exactly 50%.
* The judge sees only final answers (thinking stripped by the harness). It
  is told to ignore length and self-assessment ("this analysis has been
  rigorously verified" was the step-310 hack).
* A subset of items also gets a **human blind review** (30% of items, or on
  the 9B plan 20 pairs plus 10 more if needed). The subset is chosen
  deterministically by hashing, sides are shuffled, and model identities are
  removed. Verdicts are imported back (``dpt.eval.blind_review``) and reported
  next to the judge win rates, together with reviewer-judge agreement.
* A **length-matched** win rate (pairs whose lengths differ by at most
  ×1.2) is reported as well. In the direct-mixing run, half of T3's judge
  "gain" disappeared once length was matched.
* The judge version is pinned. A response whose ``model`` field differs
  from the pinned version raises, because a silent server-side model change
  shifts the score distribution (report §风险).
* An optional **second judge** of another family (the 9B plan: GPT-6
  Luna through the OpenAI API) judges the same pairs. The primary judge is
  also the T3 reward during RL, so the policy was optimized against it; the
  second judge never touched training. Both win rates and their agreement
  are reported (``dpt.eval.harness.score_pairwise``).
* An answer with no parseable verdict counts as a tie, which pulls the win
  rate toward 50%. ``JudgeCall.parsed`` records it, and the harness refuses to
  score a judge whose unparsed rate is too high (a reasoning model that runs
  out of tokens before the verdict looks exactly like this).
* Every verdict is cached per judge (``CachedJudge``): reruns cost nothing, the
  cache is the audit trail, and a judge reachable from only one machine (an
  API from a node with internet) can be run as a separate ``--regrade-only``
  pass while the other judge's verdicts come from its cache.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

from dpt.eval.hosted import ChatEndpoint, JudgeVersionError  # noqa: F401 (re-exported)

Verdict = str  # "A" | "B" | "tie"
Outcome = str  # "win" | "loss" | "tie"   (from the candidate's point of view)

JUDGE_SYSTEM_PROMPT = """You are an impartial domain expert comparing two answers to the same question.
Judge ONLY on: factual and technical correctness, sound reasoning, correct use of units and numbers,
coverage of the rubric criteria, and clarity.
Rules:
- Do NOT prefer an answer because it is longer. Extra length that adds no correct content is a negative.
- IGNORE any self-assessment inside an answer (e.g. "this analysis has been rigorously verified").
- Citations count only if they are specific and relevant; do not reward citation-looking text by itself.
- The order of the answers is random; do not let position influence you.
Finish with exactly one verdict on its own line: [[A]] if answer A is better, [[B]] if answer B is better,
or [[TIE]] if they are equally good."""

_VERDICT_RE = re.compile(r"\[\[(A|B|TIE)\]\]", re.IGNORECASE)


class PairwiseJudge(Protocol):
    def compare(
        self, question: str, answer_a: str, answer_b: str, rubric: Sequence[Any] | None = None
    ) -> Verdict: ...


class JudgeUnavailable(RuntimeError):
    """A verdict is needed that is neither cached nor obtainable (no endpoint configured)."""


@dataclass(frozen=True)
class JudgeCall:
    verdict: Verdict
    parsed: bool = True          # False: no [[A]]/[[B]]/[[TIE]] found (or output truncated) → counted as tie
    served_model: str | None = None


def parse_verdict_or_none(text: str) -> Verdict | None:
    """Last ``[[A]]``/``[[B]]``/``[[TIE]]`` in the judge output, or None if there is none."""
    found = _VERDICT_RE.findall(text or "")
    if not found:
        return None
    v = found[-1].upper()
    return "tie" if v == "TIE" else v


def parse_verdict(text: str) -> Verdict:
    """Last ``[[A]]``/``[[B]]``/``[[TIE]]`` in the judge output; unparseable → ``"tie"``."""
    return parse_verdict_or_none(text) or "tie"


def build_judge_user_prompt(question: str, answer_a: str, answer_b: str, rubric: Sequence[Any] | None) -> str:
    rub = ""
    if rubric:
        lines = [
            f"- {r['criterion']} (weight {r.get('weight', 1)})" if isinstance(r, dict) else f"- {r}"
            for r in rubric
        ]
        rub = "Rubric:\n" + "\n".join(lines) + "\n\n"
    return (
        f"Question:\n{question}\n\n{rub}"
        f"[Answer A]\n{answer_a}\n[End of A]\n\n[Answer B]\n{answer_b}\n[End of B]"
    )


class HttpPairwiseJudge(ChatEndpoint):
    """Pairwise judge behind an OpenAI-compatible ``/v1/chat/completions`` endpoint.

    Defaults suit a vLLM-served open model (temperature 0, ``max_tokens``). A hosted reasoning model
    such as GPT-6 Luna needs ``temperature=None``, ``max_tokens_field="max_completion_tokens"`` (the
    budget includes hidden reasoning, so make it large) and ``extra_body={"reasoning_effort": "xhigh"}``;
    the eval suite config sets these (``dpt.eval.hosted``).
    """

    def call(
        self, question: str, answer_a: str, answer_b: str, rubric: Sequence[Any] | None = None
    ) -> JudgeCall:
        reply = self.request([
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": build_judge_user_prompt(question, answer_a, answer_b, rubric)},
        ])
        v = parse_verdict_or_none(reply.content)
        truncated = reply.finish_reason == "length"
        return JudgeCall(v or "tie", parsed=v is not None and not truncated, served_model=reply.model)

    def compare(
        self, question: str, answer_a: str, answer_b: str, rubric: Sequence[Any] | None = None
    ) -> Verdict:
        return self.call(question, answer_a, answer_b, rubric).verdict


def _prompt_version() -> str:
    return hashlib.sha256(JUDGE_SYSTEM_PROMPT.encode()).hexdigest()[:12]


class CachedJudge:
    """Persist every verdict of one judge to a JSONL file and reuse it.

    The key covers the judge prompt version, question, rubric and both answers in order, so a changed
    answer or prompt is judged again. Records made by a different ``model`` than the configured one
    are ignored. ``inner=None`` replays the cache only: a miss raises ``JudgeUnavailable``.
    ``stats`` counts calls, cache hits and unparsed verdicts (the harness reports them).
    """

    def __init__(self, inner: Any, path: Path | str, *, model: str | None = None):
        self.inner, self.path, self.model = inner, Path(path), model
        self._lock = threading.Lock()
        self.stats: Counter = Counter()
        self.served_models: set[str] = set()
        self._cache: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if self.model is None or rec.get("model") == self.model:
                    self._cache[rec["key"]] = rec

    @staticmethod
    def key(question: str, answer_a: str, answer_b: str, rubric: Sequence[Any] | None) -> str:
        blob = json.dumps([_prompt_version(), question, rubric, answer_a, answer_b], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def compare(
        self, question: str, answer_a: str, answer_b: str, rubric: Sequence[Any] | None = None
    ) -> Verdict:
        k = self.key(question, answer_a, answer_b, rubric)
        with self._lock:
            rec = self._cache.get(k)
        if rec is not None:
            hit = True
        else:
            if self.inner is None:
                raise JudgeUnavailable(f"no cached verdict in {self.path} and no endpoint configured")
            call = (
                self.inner.call(question, answer_a, answer_b, rubric)
                if hasattr(self.inner, "call")
                else JudgeCall(self.inner.compare(question, answer_a, answer_b, rubric))
            )
            rec = {"key": k, "verdict": call.verdict, "parsed": call.parsed,
                   "model": self.model, "served_model": call.served_model}
            hit = False
        with self._lock:
            self.stats["calls"] += 1
            self.stats["cached"] += int(hit)
            self.stats["unparsed"] += int(not rec.get("parsed", True))
            if rec.get("served_model"):
                self.served_models.add(str(rec["served_model"]))
            if not hit:
                self._cache[k] = rec
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
        return rec["verdict"]


@dataclass(frozen=True)
class PairResult:
    item_id: str
    outcome: Outcome
    verdict_cand_first: Verdict  # candidate shown as A
    verdict_base_first: Verdict  # base shown as A


def judge_both_orders(
    judge: PairwiseJudge,
    item_id: str,
    question: str,
    candidate: str,
    base: str,
    rubric: Sequence[Any] | None = None,
) -> PairResult:
    v1 = judge.compare(question, candidate, base, rubric)  # candidate = A
    v2 = judge.compare(question, base, candidate, rubric)  # candidate = B
    if v1 == "A" and v2 == "B":
        outcome = "win"
    elif v1 == "B" and v2 == "A":
        outcome = "loss"
    else:
        outcome = "tie"
    return PairResult(item_id, outcome, v1, v2)


def length_matched(
    results: Sequence[PairResult],
    cand_lens: dict[str, int],
    base_lens: dict[str, int],
    max_ratio: float = 1.2,
) -> list[PairResult]:
    """Keep pairs whose output lengths differ by at most ``max_ratio`` (either direction)."""
    out = []
    for r in results:
        a, b = max(1, cand_lens[r.item_id]), max(1, base_lens[r.item_id])
        if max(a, b) / min(a, b) <= max_ratio:
            out.append(r)
    return out


# --------------------------------------------------------------------------- expert blind review
def _in_subset(item_id: str, fraction: float, seed: int) -> bool:
    h = int(hashlib.sha256(f"{seed}|{item_id}".encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < fraction


def export_blind_review(
    items: Iterable[dict[str, Any]],
    candidate: dict[str, str],
    base: dict[str, str],
    *,
    fraction: float = 0.30,
    seed: int = 0,
    n_first: int | None = None,
    n_extra: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build (review_rows, key_rows) for the reviewed subset.

    ``review_rows`` go to the reviewer: blind id, review ``batch``, question, rubric, two anonymous
    responses, and an empty verdict. ``key_rows`` stay with the eval owner and record which side was
    the candidate. With ``n_first`` set, the subset is the first ``n_first + n_extra`` items in a
    seeded hash order: batch 1 is reviewed always, batch 2 only when batch 1 disagrees with the judges
    (``dpt.eval.blind_review``). Otherwise a hash-chosen ``fraction`` of the items, all in batch 1.
    """
    rng = random.Random(seed)
    items = list(items)
    if n_first is not None:
        order = sorted(items, key=lambda it: hashlib.sha256(f"{seed}|order|{it['id']}".encode()).hexdigest())
        chosen = [(it, 1 if i < n_first else 2) for i, it in enumerate(order[: n_first + n_extra])]
    else:
        chosen = [(it, 1) for it in items if _in_subset(str(it["id"]), fraction, seed)]
    review, key = [], []
    for it, batch in chosen:
        iid = str(it["id"])
        cand_first = rng.random() < 0.5
        blind_id = hashlib.sha256(f"blind|{seed}|{iid}".encode()).hexdigest()[:12]
        r1, r2 = (candidate[iid], base[iid]) if cand_first else (base[iid], candidate[iid])
        review.append(
            {
                "blind_id": blind_id,
                "batch": batch,
                "question": it["question"],
                "rubric": it.get("answer", {}).get("rubric"),
                "response_1": r1,
                "response_2": r2,
                "verdict": None,
            }
        )
        key.append({"blind_id": blind_id, "item_id": iid, "candidate_side": 1 if cand_first else 2})
    return review, key


def import_blind_verdicts(
    review_rows: Iterable[dict[str, Any]], key_rows: Iterable[dict[str, Any]]
) -> dict[str, Outcome]:
    """Map expert verdicts ("1" | "2" | "tie") back to candidate outcomes keyed by item id."""
    key = {k["blind_id"]: k for k in key_rows}
    out: dict[str, Outcome] = {}
    for r in review_rows:
        v = r.get("verdict")
        if v is None:
            continue
        k = key[r["blind_id"]]
        v = str(v).strip().lower()
        if v == "tie":
            out[k["item_id"]] = "tie"
        elif v in ("1", "2"):
            out[k["item_id"]] = "win" if int(v) == k["candidate_side"] else "loss"
        else:
            raise ValueError(f"bad expert verdict {r.get('verdict')!r} for {r['blind_id']}")
    return out


def judge_expert_agreement(judge: dict[str, Outcome], expert: dict[str, Outcome]) -> float:
    """Fraction of expert-reviewed items where the judge outcome equals the expert outcome."""
    common = [i for i in expert if i in judge]
    return sum(judge[i] == expert[i] for i in common) / max(1, len(common))
