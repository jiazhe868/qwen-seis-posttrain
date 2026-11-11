"""Text normalization and n-gram utilities shared by dedup, decontamination and quality filters.

The same normalization must be used on both sides of any overlap test
(training candidate vs. eval item); otherwise a trivial whitespace or
full-width/half-width difference hides contamination. That is why all
overlap-based modules import these functions instead of rolling their own.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable, Sequence

_WS = re.compile(r"\s+")
# Keep letters, digits, CJK, and math-relevant symbols; drop other punctuation.
_PUNCT = re.compile(r"[^\w\s一-鿿\.\+\-\*/=\^<>\(\)]", re.UNICODE)
# Word tokens; each CJK character counts as its own token so n-grams work for Chinese text too.
_TOKEN = re.compile(r"[一-鿿]|[A-Za-z]+|\d+(?:\.\d+)?|[^\sA-Za-z\d一-鿿]", re.UNICODE)


def normalize_text(text: str, *, lowercase: bool = True, strip_punct: bool = True) -> str:
    """NFKC-normalize, optionally lowercase / strip punctuation, collapse whitespace."""
    t = unicodedata.normalize("NFKC", text)
    if lowercase:
        t = t.lower()
    if strip_punct:
        t = _PUNCT.sub(" ", t)
    return _WS.sub(" ", t).strip()


def tokenize(text: str) -> list[str]:
    """Cheap language-agnostic tokenizer (words, numbers, single CJK chars, symbols)."""
    return _TOKEN.findall(text)


def word_ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0:
        raise ValueError("n must be positive")
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def ngram_set(text: str, n: int, *, normalize: bool = True) -> set[tuple[str, ...]]:
    t = normalize_text(text) if normalize else text
    return set(word_ngrams(tokenize(t), n))


def char_shingles(text: str, k: int = 5, *, normalize: bool = True) -> set[str]:
    """Character k-shingles (used by MinHash; robust to tokenization differences)."""
    t = normalize_text(text) if normalize else text
    t = t.replace(" ", "")
    if len(t) < k:
        return {t} if t else set()
    return {t[i : i + k] for i in range(len(t) - k + 1)}


def ngram_repetition_ratio(text: str, n: int = 4) -> float:
    """Fraction of n-gram occurrences that are repeats of an earlier n-gram.

    ``1 - unique/total``. OCR failures that loop the same line hundreds of
    times score close to 1; normal prose scores < 0.1. The SFT loss spike at
    step ~400 (report §阶段一踩到的坑) came from a few dozen samples above 0.3.
    """
    grams = word_ngrams(tokenize(normalize_text(text)), n)
    if not grams:
        return 0.0
    counts = Counter(grams)
    return 1.0 - len(counts) / len(grams)


def jaccard(a: Iterable, b: Iterable) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))
