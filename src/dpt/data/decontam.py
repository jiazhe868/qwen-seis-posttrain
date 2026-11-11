"""Decontamination: keep evaluation content out of training data (report §污染检查).

Three directions are covered: training data ↔ eval sets, indirect leakage
through the distillation *teacher*, and memorization already present in the
*base model*.

Layers, applied in order. An item is attributed to the first layer that fires,
which gives the funnel rows "13-gram+MinHash 1,190" and then "embedding+LLM
1,350":

1. ``ngram_minhash``: any shared 13-gram (normalized word tokens), **or**
   MinHash Jaccard ≥ 0.5 on character 5-grams. 13 words is long enough that a
   shared span is almost never coincidental. The 0.5 MinHash threshold catches
   light edits that break every 13-gram (for example a changed number every
   few words).
2. ``semantic``: embedding nearest neighbours with cosine ≥ 0.85, then an LLM
   judge decides "same problem?". The judge counts a rewording with changed
   numbers as the same problem.
3. ``answer_level``: for numeric problems, the eval item's input parameters
   (multiset, ≥ 2 values) all appear in the candidate **and** the answers
   match. Rewording cannot hide a problem if its numbers and its result
   travel together.
4. ``teacher_verbatim`` (SFT only): assistant text that reproduces 13-grams of
   eval questions/solutions not present in its own prompt. The teacher was
   reciting eval content. The report found 46 such samples and traced their
   prompt sources.
5. ``canary``: unique canary strings embedded in internal eval files are
   scanned for in all training data and in model outputs.

Asymmetric costs drive the thresholds. The semantic layer's audited
precision was 87%, so about 13% of its removals were not real contamination.
A false positive costs one training problem. A false negative makes an eval
conclusion invalid. The thresholds are therefore set for recall.

Access control: this stage runs under the eval-authorized account, because
training jobs cannot read eval sets. Its outputs in the training data area
contain only candidate uids and the *ids* of matched eval items, never eval
text. The human-audit sample, which needs both texts, is written to
``audit_dir`` on the secure side.
"""

from __future__ import annotations

import math
import random
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol, Sequence

import numpy as np

from dpt.common.io import read_jsonl
from dpt.common.logging import get_logger
from dpt.common.text import normalize_text, tokenize, word_ngrams
from dpt.data.dedup import MinHashIndex
from dpt.data.schema import Problem, SFTSample

log = get_logger(__name__)


# =============================================================================================
# Protocols and records
# =============================================================================================
class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray:  # (n, d), L2-normalized rows
        ...


class ParaphraseJudge(Protocol):
    def same_problem(self, a: str, b: str) -> bool:
        ...


@dataclass
class RefItem:
    """A protected item: an eval-set item or a public-benchmark item."""

    key: str                     # "<collection>/<uid>", e.g. "d_num/e-001" or "gsm8k/1234"
    question: str
    answer: dict[str, Any] | None = None
    solution: str | None = None  # reference derivation, if any (used by the teacher check)
    params: list[float] | None = None


def load_ref_items(collections: dict[str, str]) -> list[RefItem]:
    """Load ``{collection_name: jsonl_path}``. Each line needs ``uid`` and ``question``."""
    items: list[RefItem] = []
    for name, path in collections.items():
        for i, r in enumerate(read_jsonl(path)):
            items.append(RefItem(
                key=f"{name}/{r.get('uid', i)}",
                question=str(r["question"]),
                answer=r.get("answer"),
                solution=r.get("solution"),
                params=(list((r.get("meta") or {}).get("params", {}).values()) or None),
            ))
    return items


@dataclass
class DecontamConfig:
    ngram_n: int = 13
    minhash_threshold: float = 0.5
    num_perm: int = 128
    shingle_k: int = 5
    embed_threshold: float = 0.85
    embed_top_k: int = 5
    answer_min_params: int = 2
    answer_sig_digits: int = 4
    teacher_min_hits: int = 1
    # An n-gram found in more than this many of the texts checked is template or instruction boilerplate
    # (a generator's formula sentence, IFEval's constraint wording, the code-task scaffold), not a sign that
    # one eval item leaked: a leaked item shows up in a few texts. Such n-grams are not counted; 0 counts all.
    ngram_max_df: int = 10
    judge_concurrency: int = 32   # paraphrase-judge calls in flight (one at a time took ~20 s each)
    canaries: list[str] = field(default_factory=list)
    audit_sample: int = 200
    seed: int = 1

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "DecontamConfig":
        d = dict(d or {})
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DecontamResult:
    kept: list[Any]
    removed: list[dict[str, Any]] = field(default_factory=list)  # {"uid","layer","match","score"}
    layer_counts: Counter = field(default_factory=Counter)
    semantic_judge_rejected: int = 0
    scanned: int = 0
    boilerplate_ngrams: int = 0

    def stats(self) -> dict[str, Any]:
        return {"scanned": self.scanned, "removed": len(self.removed),
                "rate": round(len(self.removed) / max(1, self.scanned), 5),
                "by_layer": dict(self.layer_counts),
                "semantic_judge_rejected": self.semantic_judge_rejected,
                "boilerplate_ngrams": self.boilerplate_ngrams}


# =============================================================================================
# Building blocks
# =============================================================================================
def _grams(text: str, n: int) -> set[tuple[str, ...]]:
    toks = tokenize(normalize_text(text))
    if not toks:
        return set()
    if len(toks) < n:  # short texts: the whole normalized sequence acts as one "n-gram" (exact match)
        return {tuple(toks)}
    return set(word_ngrams(toks, n))


class NgramIndex:
    """Exact n-gram overlap index (normalized word tokens)."""

    def __init__(self, n: int = 13):
        self.n = n
        self._map: dict[tuple[str, ...], set[str]] = defaultdict(set)

    def add(self, key: str, text: str) -> None:
        for g in _grams(text, self.n):
            self._map[g].add(key)

    def hits(self, text: str, exclude: set[tuple[str, ...]] | None = None) -> Counter:
        """``Counter{ref_key: number of shared n-grams}``; n-grams in ``exclude`` are ignored."""
        c: Counter = Counter()
        for g in _grams(text, self.n):
            if exclude and g in exclude:
                continue
            for k in self._map.get(g, ()):
                c[k] += 1
        return c


_NUMBER = __import__("re").compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _round_sig(x: float, digits: int) -> float:
    if x == 0 or not math.isfinite(x):
        return x
    return round(x, digits - 1 - int(math.floor(math.log10(abs(x)))))


def extract_numbers(text: str) -> list[float]:
    return [float(m.group(0)) for m in _NUMBER.finditer(text.replace(",", ""))]


def problem_params(p: Problem) -> list[float]:
    """Numeric inputs of a problem: ``meta.params`` / generator params, else the numbers in its text."""
    declared = p.meta.get("params") or (p.meta.get("generator") or {}).get("params")
    if declared:
        return [float(v) for v in declared.values()
                if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return extract_numbers(p.question)


def _multiset_contains(big: Sequence[float], small: Sequence[float]) -> bool:
    cb, cs = Counter(big), Counter(small)
    return all(cb[k] >= v for k, v in cs.items())


# =============================================================================================
# Decontaminator
# =============================================================================================
class Decontaminator:
    """Indexes protected items once, then checks candidate problems / SFT samples against them."""

    def __init__(
        self,
        refs: Sequence[RefItem],
        cfg: DecontamConfig,
        *,
        embedder: Embedder | None = None,
        judge: ParaphraseJudge | None = None,
    ):
        self.cfg = cfg
        self.refs = list(refs)
        self.embedder, self.judge = embedder, judge
        self.ngram = NgramIndex(cfg.ngram_n)
        self.minhash = MinHashIndex(cfg.minhash_threshold, num_perm=cfg.num_perm,
                                    shingle_k=cfg.shingle_k, seed=cfg.seed)
        self.teacher_index = NgramIndex(cfg.ngram_n)   # eval questions + solutions
        self._ref_text = {r.key: r.question for r in self.refs}
        self._answer_index: dict[float, list[tuple[str, list[float]]]] = defaultdict(list)
        for r in self.refs:
            self.ngram.add(r.key, r.question)
            self.minhash.add(r.key, r.question)
            self.teacher_index.add(r.key, r.question)
            if r.solution:
                self.teacher_index.add(r.key, r.solution)
            self._index_answer(r)
        self._ref_keys = [r.key for r in self.refs]
        self._ref_emb: np.ndarray | None = None
        if embedder is not None and self.refs:
            self._ref_emb = embedder.embed([r.question for r in self.refs])

    def _index_answer(self, r: RefItem) -> None:
        a = r.answer or {}
        v = a.get("value")
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return
        params = r.params if r.params is not None else extract_numbers(r.question)
        d = self.cfg.answer_sig_digits
        params = [_round_sig(float(x), d) for x in params]
        if len(params) >= self.cfg.answer_min_params:
            self._answer_index[_round_sig(float(v), d)].append((r.key, params))

    # -- individual layers --------------------------------------------------------------------
    def boilerplate(self, index: NgramIndex, texts: Iterable[str]) -> set[tuple[str, ...]]:
        """Indexed n-grams that occur in more than ``ngram_max_df`` of ``texts`` (see the config field)."""
        if self.cfg.ngram_max_df <= 0:
            return set()
        df: Counter = Counter()
        for t in texts:
            df.update(g for g in _grams(t, index.n) if g in index._map)
        return {g for g, c in df.items() if c > self.cfg.ngram_max_df}

    def layer_ngram_minhash(self, text: str, exclude: set[tuple[str, ...]] | None = None) -> tuple[str, float] | None:
        hits = self.ngram.hits(text, exclude=exclude)
        if hits:
            key, cnt = hits.most_common(1)[0]
            return key, float(cnt)
        mh = self.minhash.query(text)
        if mh:
            return str(mh[0][0]), float(mh[0][1])
        return None

    def layer_answer(self, p: Problem) -> tuple[str, float] | None:
        if not p.is_numeric:
            return None
        d = self.cfg.answer_sig_digits
        key = _round_sig(float(p.answer["value"]), d)
        cand = [_round_sig(x, d) for x in problem_params(p)]
        for ref_key, ref_params in self._answer_index.get(key, ()):
            if _multiset_contains(cand, ref_params):
                return ref_key, 1.0
        return None

    def semantic_matches(self, texts: Sequence[str]) -> list[list[tuple[str, float]]]:
        """For each text, refs with cosine ≥ threshold (top-k, best first). Empty if no embedder."""
        if self.embedder is None or self._ref_emb is None or not texts:
            return [[] for _ in texts]
        out: list[list[tuple[str, float]]] = []
        for i in range(0, len(texts), 1024):
            emb = self.embedder.embed(list(texts[i:i + 1024]))
            sims = emb @ self._ref_emb.T
            k = min(self.cfg.embed_top_k, sims.shape[1])
            top = np.argsort(-sims, axis=1)[:, :k]
            for row, idxs in zip(sims, top):
                out.append([(self._ref_keys[j], float(row[j])) for j in idxs
                            if row[j] >= self.cfg.embed_threshold])
        return out

    # -- full checks --------------------------------------------------------------------------
    def check_problems(self, problems: Sequence[Problem]) -> DecontamResult:
        res = DecontamResult(kept=[], scanned=len(problems))
        common = self.boilerplate(self.ngram, (p.question for p in problems))
        res.boilerplate_ngrams = len(common)
        survivors: list[Problem] = []
        for p in problems:  # layer 1 first (cheap); embeddings only for the survivors
            hit = self.layer_ngram_minhash(p.question, exclude=common)
            if hit:
                self._remove(res, p.uid, "ngram_minhash", hit)
            else:
                survivors.append(p)
        sem = self.semantic_matches([p.question for p in survivors])
        for p, hit in zip(survivors, self._judge_all([p.question for p in survivors], sem, res)):
            if hit:
                self._remove(res, p.uid, "semantic", hit)
                continue
            hit = self.layer_answer(p)
            if hit:
                self._remove(res, p.uid, "answer_level", hit)
                continue
            res.kept.append(p)
        log.info("decontam(problems): %s", res.stats())
        return res

    def check_sft(self, samples: Sequence[SFTSample]) -> DecontamResult:
        """Prompt-side layers (1, 2) on user turns, then the teacher-recitation layer on assistant turns."""
        res = DecontamResult(kept=[], scanned=len(samples))
        common = self.boilerplate(self.ngram, (s.user_text() for s in samples))
        recited = self.boilerplate(self.teacher_index, (s.assistant_text() for s in samples))
        res.boilerplate_ngrams = len(common | recited)
        survivors: list[SFTSample] = []
        for s in samples:
            hit = self.layer_ngram_minhash(s.user_text(), exclude=common)
            if hit:
                self._remove(res, s.uid, "ngram_minhash", hit)
            else:
                survivors.append(s)
        sem = self.semantic_matches([s.user_text() for s in survivors])
        for s, hit in zip(survivors, self._judge_all([s.user_text() for s in survivors], sem, res)):
            if hit:
                self._remove(res, s.uid, "semantic", hit)
                continue
            t = self.teacher_recitation(s, exclude=recited)
            if t:
                self._remove(res, s.uid, "teacher_verbatim", t, source=s.source)
                continue
            res.kept.append(s)
        log.info("decontam(sft): %s", res.stats())
        return res

    def teacher_recitation(self, s: SFTSample, exclude: set[tuple[str, ...]] | None = None
                           ) -> tuple[str, float] | None:
        own = _grams(s.user_text(), self.cfg.ngram_n)
        hits = self.teacher_index.hits(s.assistant_text(), exclude=own | (exclude or set()))
        if hits:
            key, cnt = hits.most_common(1)[0]
            if cnt >= self.cfg.teacher_min_hits:
                return key, float(cnt)
        return None

    def _judge(self, text: str, matches: list[tuple[str, float]]) -> tuple[tuple[str, float] | None, int]:
        """The first embedding match the judge calls the same problem, and how many it rejected before."""
        rejected = 0
        for key, sim in matches:
            if self.judge is None or self.judge.same_problem(text, self._ref_text[key]):
                return (key, sim), rejected
            rejected += 1
        return None, rejected

    def _judge_all(self, texts: Sequence[str], sem: Sequence[list[tuple[str, float]]],
                   res: DecontamResult) -> list[tuple[str, float] | None]:
        """:meth:`_judge` for every text with embedding matches, ``judge_concurrency`` calls in flight;
        the verdicts do not depend on the order (the judge samples at temperature 0)."""
        from concurrent.futures import ThreadPoolExecutor

        todo = [i for i, m in enumerate(sem) if m]
        hits: list[tuple[str, float] | None] = [None] * len(texts)
        workers = max(1, min(self.cfg.judge_concurrency, len(todo))) if self.judge is not None else 1
        with ThreadPoolExecutor(workers) as ex:
            for i, (hit, rejected) in zip(todo, ex.map(lambda i: self._judge(texts[i], sem[i]), todo)):
                hits[i] = hit
                res.semantic_judge_rejected += rejected
        return hits

    @staticmethod
    def _remove(res: DecontamResult, uid: str, layer: str, hit: tuple[str, float], **extra: Any) -> None:
        res.removed.append({"uid": uid, "layer": layer, "match": hit[0], "score": round(hit[1], 4), **extra})
        res.layer_counts[layer] += 1

    def audit_sample(
        self, res: DecontamResult, texts_by_uid: dict[str, str], layer: str = "semantic"
    ) -> list[dict]:
        """Seeded sample of removals for human confirmation. Contains eval text: secure storage only."""
        rows = [r for r in res.removed if r["layer"] == layer]
        rng = random.Random(self.cfg.seed)
        rng.shuffle(rows)
        return [{**r, "candidate_text": texts_by_uid.get(r["uid"], ""),
                 "ref_text": self._ref_text.get(r["match"], ""), "human_label": None}
                for r in rows[:self.cfg.audit_sample]]


# =============================================================================================
# Canaries
# =============================================================================================
def make_canary(namespace: str) -> str:
    """Deterministic canary GUID for an internal eval file, e.g. ``make_canary("d_num/v1")``."""
    return "DPT-CANARY-" + str(uuid.uuid5(uuid.NAMESPACE_URL, f"dpt-eval-canary:{namespace}"))


def _strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)


def scan_canaries(records: Iterable[tuple[str, Any]], canaries: Sequence[str]) -> list[dict[str, str]]:
    """Scan ``(key, record)`` pairs (every nested string) for any canary. Any hit is a leak and must block."""
    hits = []
    if not canaries:
        return hits
    for key, rec in records:
        for s in _strings(rec):
            for c in canaries:
                if c in s:
                    hits.append({"key": key, "canary": c})
    return hits


# =============================================================================================
# Base-model memorization probes
# =============================================================================================
def _norm_words(text: str) -> list[str]:
    return normalize_text(text).split()


def prefix_completion_probe(
    texts: Sequence[str],
    complete_fn: Callable[[str], str],
    *,
    prefix_frac: float = 0.5,
    k_words: int = 20,
    match_frac: float = 0.8,
    seed: int = 1,
) -> dict[str, Any]:
    """Give the base model the first half of each item; does it continue with the true second half?

    A match means at least ``match_frac`` of the next ``k_words`` normalized
    words agree position by position. The **chance baseline** compares each
    generation with a *different* item's continuation, using a seeded
    derangement. This measures how often generic phrasing matches by
    accident, without extra generations. The report found MMLU-Pro 6.2% vs
    0.3% baseline (memorized: use deltas only) and GPQA-Diamond 0.4%
    (no evidence).
    """
    gens, refs = [], []
    for t in texts:
        words = t.split()
        if len(words) < 8:
            continue
        m = max(1, int(len(words) * prefix_frac))
        gens.append(_norm_words(complete_fn(" ".join(words[:m])))[:k_words])
        refs.append(_norm_words(" ".join(words[m:]))[:k_words])

    def _match(g: list[str], r: list[str]) -> bool:
        k = min(k_words, len(r))
        if k == 0:
            return False
        same = sum(1 for a, b in zip(g[:k], r[:k]) if a == b)
        return same / k >= match_frac

    n = len(gens)
    if n == 0:
        return {"n": 0, "match_rate": 0.0, "baseline_rate": 0.0, "z": 0.0, "significant": False}
    hits = sum(_match(g, r) for g, r in zip(gens, refs))
    perm = list(range(n))
    random.Random(seed).shuffle(perm)
    shifted = perm[1:] + perm[:1]  # derangement: perm[i] is paired with shifted[i] ≠ perm[i]
    base_hits = sum(_match(gens[a], refs[b]) for a, b in zip(perm, shifted)) if n > 1 else 0
    p1, p0 = hits / n, base_hits / n
    pooled = (hits + base_hits) / (2 * n)
    se = math.sqrt(max(pooled * (1 - pooled) * 2 / n, 1e-12))
    z = (p1 - p0) / se
    return {"n": n, "match_rate": round(p1, 4), "baseline_rate": round(p0, 4), "z": round(z, 2),
            "significant": bool(z > 3.0 and p1 - p0 > 0.01)}


def perturbation_gap(
    original: Sequence[float], perturbed: Sequence[float], *, n_boot: int = 2000, seed: int = 1
) -> dict[str, Any]:
    """Accuracy on original vs numerically perturbed versions of the same items, with a paired bootstrap CI.

    A model that *memorized* answers does much worse on perturbed numbers. In
    the report: final model 71.4 vs 68.9 (gap 2.5), base gap 2.1. The gap did
    not widen, so the RL gains are not memorization.
    """
    if len(original) != len(perturbed) or not original:
        raise ValueError("original and perturbed must be non-empty and paired")
    o, p = np.asarray(original, dtype=float), np.asarray(perturbed, dtype=float)
    diffs = o - p
    rng = np.random.RandomState(seed)
    boots = diffs[rng.randint(0, len(diffs), size=(n_boot, len(diffs)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": len(o), "original": round(100 * o.mean(), 2), "perturbed": round(100 * p.mean(), 2),
            "gap": round(100 * diffs.mean(), 2), "ci95": [round(100 * lo, 2), round(100 * hi, 2)]}


# =============================================================================================
# Report
# =============================================================================================
def contamination_report(
    *,
    rl: DecontamResult | None = None,
    sft: DecontamResult | None = None,
    probes: dict[str, dict[str, Any]] | None = None,
    canary_hits: list[dict[str, str]] | None = None,
    canary_scanned: int = 0,
    semantic_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the report-shaped summary (the "设想结果" table of §污染检查)."""
    rep: dict[str, Any] = {}
    if rl is not None:
        rep["rl"] = {
            "scanned": rl.scanned,
            "ngram_minhash": rl.layer_counts.get("ngram_minhash", 0),
            "semantic": rl.layer_counts.get("semantic", 0),
            "answer_level": rl.layer_counts.get("answer_level", 0),
            "semantic_judge_rejected": rl.semantic_judge_rejected,
            "total_removed": len(rl.removed),
            "rate": round(len(rl.removed) / max(1, rl.scanned), 5),
        }
        if semantic_audit:
            rep["rl"]["semantic_audit"] = semantic_audit
    if sft is not None:
        rep["sft"] = {
            "scanned": sft.scanned,
            "prompt_side": sft.layer_counts.get("ngram_minhash", 0) + sft.layer_counts.get("semantic", 0),
            "teacher_verbatim": sft.layer_counts.get("teacher_verbatim", 0),
            "total_removed": len(sft.removed),
            "rate": round(len(sft.removed) / max(1, sft.scanned), 5),
            "teacher_sources": sorted({str((r.get("source") or {}).get("origin", "?"))
                                       for r in sft.removed if r["layer"] == "teacher_verbatim"}),
        }
    if probes:
        rep["memorization_probes"] = probes
    rep["canary"] = {"scanned": canary_scanned, "hits": len(canary_hits or []), "details": canary_hits or []}
    return rep
