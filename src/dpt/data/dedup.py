"""Near-duplicate detection with MinHash + LSH. Self-contained (numpy only) and deterministic.

Report §处理流水线 step 2: MinHash over 5-gram shingles, Jaccard threshold 0.8,
run *within* the bank and *across* banks. 91k → 82k.

Design
------
* **Shingles**: character 5-grams of the normalized text, with spaces removed
  (``dpt.common.text.char_shingles``). Character shingles survive tokenization
  differences, LaTeX spacing and OCR word-splitting better than word shingles.
* **Hashing**: each shingle is hashed with blake2b to 32 bits. Each of the
  ``num_perm`` permutations is then simulated with Dietzfelbinger's
  multiply-add-shift hash ``h(x) = ((a·x + b) mod 2^64) >> 32``, where a and
  b are random 64-bit words. For 32-bit keys this family is strongly
  universal. ``mod 2^64`` is exactly numpy's uint64 wrap-around, so
  signatures are bit-identical on every machine. (A tempting alternative,
  ``(a·x + b) mod p`` with *small* a, is nearly monotone in x. Its
  "permutations" are then highly correlated and the Jaccard estimate
  collapses. The unit tests check estimate quality for that reason.)
* **LSH banding**: (bands, rows) is picked by numerically minimizing the
  weighted false-positive + false-negative area around the threshold (the
  standard S-curve analysis). False negatives are weighted 0.98 and false
  positives 0.02. Every candidate pair is verified exactly afterwards, so a
  false positive costs only a cheap set intersection, while a false negative
  is a duplicate that silently survives. With equal weights, 128 permutations
  give about (9 bands × 13 rows). That misses roughly 30% of pairs at
  Jaccard 0.84. The recall-weighted choice (17 × 7 at 0.8, 34 × 3 at 0.5)
  reaches about 98% recall at the threshold and 99.7% at +0.04.
* **Verification**: every LSH candidate pair is checked with the *exact*
  Jaccard of the hashed shingle sets, so the threshold is applied exactly and
  not as a ±0.04 estimate. LSH only decides which pairs are worth checking.
* **Representative**: in a duplicate cluster we keep the item with the
  smallest ``priority`` (default: input order). Feed higher-quality sources
  first, or pass explicit priorities.

Why 0.8 for dedup but 0.5 for decontamination (``decontam.py``): the costs are
asymmetric. A missed duplicate only skews the training mix slightly. A missed
contaminated item invalidates an eval conclusion. Dedup can afford to be
conservative about what it merges, and decontam must be aggressive.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterable, Sequence

import numpy as np

from dpt.common.text import char_shingles

_SHIFT32 = np.uint64(32)


def _hash32(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=4).digest(), "little")


def _integrate(f: Callable[[np.ndarray], np.ndarray], lo: float, hi: float, n: int = 401) -> float:
    if hi <= lo:
        return 0.0
    xs = np.linspace(lo, hi, n)
    ys = f(xs)
    # Trapezoidal rule written out (np.trapz is deprecated/renamed across numpy versions).
    return float(np.sum((ys[1:] + ys[:-1]) * np.diff(xs)) / 2.0)


def optimal_lsh_params(
    threshold: float, num_perm: int, fp_weight: float = 0.02, fn_weight: float = 0.98
) -> tuple[int, int]:
    """Return (bands, rows) minimizing weighted FP/FN probability mass for a Jaccard threshold."""
    best, best_err = (1, num_perm), float("inf")
    for b in range(1, num_perm + 1):
        for r in range(1, num_perm // b + 1):
            fp = _integrate(lambda s: 1 - (1 - s**r) ** b, 0.0, threshold)
            fn = _integrate(lambda s: (1 - s**r) ** b, threshold, 1.0)
            err = fp_weight * fp + fn_weight * fn
            if err < best_err:
                best, best_err = (b, r), err
    return best


class MinHasher:
    """Computes shingle hash sets and MinHash signatures with a fixed seed."""

    def __init__(self, num_perm: int = 128, shingle_k: int = 5, seed: int = 1):
        self.num_perm = num_perm
        self.shingle_k = shingle_k
        rng = np.random.RandomState(seed)
        # Full 64-bit random words; a is forced odd (standard for multiply-shift hashing).
        self.a = rng.randint(0, 2**63, size=num_perm, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
        self.b = rng.randint(0, 2**63, size=num_perm, dtype=np.uint64) * np.uint64(2) + np.uint64(
            rng.randint(0, 2))

    def shingle_hashes(self, text: str) -> np.ndarray:
        """Sorted unique uint64 array of 32-bit shingle hashes (empty for very short texts)."""
        sh = char_shingles(text, self.shingle_k)
        if not sh:
            return np.zeros(0, dtype=np.uint64)
        return np.unique(np.fromiter((_hash32(s) for s in sh), dtype=np.uint64, count=len(sh)))

    def signature(self, hashes: np.ndarray) -> np.ndarray:
        if hashes.size == 0:
            return np.full(self.num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
        # (m, 1) * (1, P) with uint64 wrap-around == multiplication mod 2^64 (multiply-add-shift).
        with np.errstate(over="ignore"):
            phv = (hashes[:, None] * self.a[None, :] + self.b[None, :]) >> _SHIFT32
        return phv.min(axis=0)


def exact_jaccard(h1: np.ndarray, h2: np.ndarray) -> float:
    if h1.size == 0 and h2.size == 0:
        return 0.0  # two empty texts are not "duplicates" in any useful sense
    inter = np.intersect1d(h1, h2, assume_unique=True).size
    return inter / float(h1.size + h2.size - inter)


class MinHashIndex:
    """LSH index over texts. ``query`` returns exact-Jaccard-verified matches ≥ threshold."""

    def __init__(self, threshold: float, num_perm: int = 128, shingle_k: int = 5, seed: int = 1):
        self.threshold = threshold
        self.hasher = MinHasher(num_perm=num_perm, shingle_k=shingle_k, seed=seed)
        self.bands, self.rows = optimal_lsh_params(threshold, num_perm)
        self._buckets: list[dict[bytes, list[Hashable]]] = [defaultdict(list) for _ in range(self.bands)]
        self._hashes: dict[Hashable, np.ndarray] = {}

    def _band_keys(self, sig: np.ndarray) -> list[bytes]:
        r = self.rows
        return [sig[i * r:(i + 1) * r].tobytes() for i in range(self.bands)]

    def add(self, key: Hashable, text: str) -> None:
        h = self.hasher.shingle_hashes(text)
        self._hashes[key] = h
        if h.size == 0:
            return
        for band, bk in zip(self._buckets, self._band_keys(self.hasher.signature(h))):
            band[bk].append(key)

    def candidates(self, text: str) -> tuple[np.ndarray, set[Hashable]]:
        h = self.hasher.shingle_hashes(text)
        if h.size == 0:
            return h, set()
        out: set[Hashable] = set()
        for band, bk in zip(self._buckets, self._band_keys(self.hasher.signature(h))):
            out.update(band.get(bk, ()))
        return h, out

    def query(self, text: str) -> list[tuple[Hashable, float]]:
        h, cands = self.candidates(text)
        hits = []
        for k in cands:
            j = exact_jaccard(h, self._hashes[k])
            if j >= self.threshold:
                hits.append((k, j))
        return sorted(hits, key=lambda kv: (-kv[1], str(kv[0])))

    def __len__(self) -> int:
        return len(self._hashes)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


@dataclass
class DedupResult:
    kept: list[int]                                              # indices into the input
    removed: list[dict[str, Any]] = field(default_factory=list)  # {"index", "duplicate_of", "jaccard"}

    @property
    def num_removed(self) -> int:
        return len(self.removed)


def near_duplicate_dedup(
    texts: Sequence[str],
    *,
    threshold: float = 0.8,
    num_perm: int = 128,
    shingle_k: int = 5,
    seed: int = 1,
    priority: Sequence[float] | None = None,
) -> DedupResult:
    """Cluster near-duplicates (transitively) and keep one representative per cluster.

    The representative is the member with the lowest ``priority`` (ties → lower index).
    Each removed item records the representative it duplicates and the
    best verified Jaccard against any cluster member it was linked to.
    """
    n = len(texts)
    index = MinHashIndex(threshold, num_perm=num_perm, shingle_k=shingle_k, seed=seed)
    uf = _UnionFind(n)
    best_j: dict[int, float] = {}
    for i, t in enumerate(texts):
        for j, jac in index.query(t):  # only earlier items are in the index → each pair checked once
            uf.union(i, int(j))
            best_j[i] = max(best_j.get(i, 0.0), jac)
            best_j[int(j)] = max(best_j.get(int(j), 0.0), jac)
        index.add(i, t)

    prio = list(priority) if priority is not None else list(range(n))
    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(i)
    kept, removed = [], []
    for members in clusters.values():
        rep = min(members, key=lambda i: (prio[i], i))
        kept.append(rep)
        for m in members:
            if m != rep:
                removed.append({"index": m, "duplicate_of": rep, "jaccard": round(best_j.get(m, 0.0), 4)})
    kept.sort()
    removed.sort(key=lambda r: r["index"])
    return DedupResult(kept=kept, removed=removed)


def cross_dedup(
    texts: Sequence[str],
    reference_texts: Iterable[tuple[Hashable, str]],
    *,
    threshold: float = 0.8,
    num_perm: int = 128,
    shingle_k: int = 5,
    seed: int = 1,
) -> DedupResult:
    """Remove items of ``texts`` that near-duplicate any item of another bank (``reference_texts``)."""
    index = MinHashIndex(threshold, num_perm=num_perm, shingle_k=shingle_k, seed=seed)
    for key, t in reference_texts:
        index.add(key, t)
    kept, removed = [], []
    for i, t in enumerate(texts):
        hits = index.query(t)
        if hits:
            removed.append({"index": i, "duplicate_of": hits[0][0], "jaccard": round(hits[0][1], 4)})
        else:
            kept.append(i)
    return DedupResult(kept=kept, removed=removed)


def instance_key(generator: dict[str, Any]) -> str:
    """Identity of a programmatic problem: its solver and its inputs, whatever the wording."""
    import json

    params = sorted((k, float(v)) for k, v in (generator.get("params") or {}).items())
    return json.dumps([generator.get("name"), params])


def instance_dedup(generators: Sequence[dict[str, Any]]) -> DedupResult:
    """Keep the first of every group of programmatic problems with the same solver and inputs.

    Instances of one template differ only in their numbers, so MinHash on their text would keep one per
    template although every instance has a different answer. Rewordings of one instance (a paraphrase
    next to its original) are still duplicates and are removed here.
    """
    first: dict[str, int] = {}
    kept, removed = [], []
    for i, g in enumerate(generators):
        k = instance_key(g)
        if k in first:
            removed.append({"index": i, "duplicate_of": first[k], "match": "same_instance"})
        else:
            first[k] = i
            kept.append(i)
    return DedupResult(kept=kept, removed=removed)
