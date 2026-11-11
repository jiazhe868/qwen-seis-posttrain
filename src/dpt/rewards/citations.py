"""Citation parsing and verification for T3 (open-ended domain reports).

The most serious reward hack of the project (report §案例记录, step 245): the
policy invented realistic-looking references, with authors, year and journal
all present. The judge could not check them and rewarded the apparent
scholarship. Proxy reward rose while expert blind win rate fell 4 points, and
31% of sampled citations did not exist.

The fix, implemented here:

1. The reference list is **removed from the judge's input**. The judge grades
   substance only, and citations are verified out of band.
2. Every citation is looked up by DOI, or by title when there is no DOI,
   against Crossref (:class:`CrossrefResolver`, cached, rate limited).
3. Each citation that cannot be verified costs 0.1 of the reward
   (``citation_factor = max(0, 1 - 0.1 * n_unverified)``, applied in ``judge.py``).

Fail closed: a lookup *error* (network, 5xx) counts as unverifiable. Errors
are reported separately (``t3_citation_errors``) so an infrastructure outage
shows up as an alert and is not mistaken for a model regression.

A DOI that resolves to a paper whose title does not match the cited title
counts as unverified. That is the "real DOI, invented paper" variant.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

from dpt.common.io import atomic_write_text
from dpt.common.logging import get_logger
from dpt.common.text import normalize_text

log = get_logger(__name__)

DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9<>]+)")
_YEAR_RE = re.compile(r"\((?P<y1>(?:19|20)\d{2})[a-z]?\)|(?<!\d)(?P<y2>(?:19|20)\d{2})[a-z]?(?=[.,;)\s])")
_REF_HEADING = re.compile(
    r"(?im)^\s*(?:#+\s*|\*\*)?(?:references|bibliography|works cited|sources|参考文献|引用文献)(?:\*\*)?\s*[:：]?\s*$"
)
_ENTRY_START = re.compile(r"^\s*(?:\[\d+\]|\d+[.)]|[-*•])\s+")
_INLINE_AUTHOR_YEAR = re.compile(
    r"\((?P<a>[A-Z][A-Za-z\-']+(?: et al\.?| (?:and|&) [A-Z][A-Za-z\-']+)?),?\s+(?P<y>(?:19|20)\d{2})[a-z]?\)"
)
_QUOTED_TITLE = re.compile(r"[\"“](.+?)[\"”]")


@dataclass
class Citation:
    raw: str
    doi: str | None = None
    title: str | None = None
    year: int | None = None
    authors: str | None = None


@dataclass
class VerifyResult:
    status: str  # "verified" | "not_found" | "error"
    matched_title: str | None = None
    similarity: float = 0.0
    detail: str = ""


@dataclass
class CitationReport:
    n_total: int = 0
    n_verified: int = 0
    n_unverified: int = 0  # not_found + error (fail closed)
    n_errors: int = 0
    details: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------------
def split_references(text: str) -> tuple[str, str]:
    """Split ``text`` at the last reference-list heading → ``(body, references_block)``."""
    matches = list(_REF_HEADING.finditer(text))
    if not matches:
        return text, ""
    m = matches[-1]
    return text[: m.start()].rstrip(), text[m.end() :].strip()


def _split_entries(block: str) -> list[str]:
    lines = [ln.rstrip() for ln in block.splitlines()]
    has_markers = any(_ENTRY_START.match(ln) for ln in lines)
    entries: list[str] = []
    for ln in lines:
        if not ln.strip():
            continue
        if not has_markers or _ENTRY_START.match(ln) or not entries:
            entries.append(_ENTRY_START.sub("", ln).strip())
        else:
            entries[-1] += " " + ln.strip()
    return [e for e in entries if len(e) > 8]


def parse_entry(raw: str) -> Citation:
    """Best-effort field extraction from one reference entry (APA-ish, numbered, or free form)."""
    doi_m = DOI_RE.search(raw)
    doi = doi_m.group(1).rstrip(".,;)]") if doi_m else None
    year_m = _YEAR_RE.search(raw)
    year = int(year_m.group("y1") or year_m.group("y2")) if year_m else None
    title = None
    q = _QUOTED_TITLE.search(raw)
    if q:
        title = q.group(1).strip().rstrip(".,")
    elif year_m and year_m.group("y1"):
        # APA: "Authors (2019). Title. Journal, 12(3), 45-67."
        after = raw[year_m.end() :].lstrip(". ")
        title = re.split(r"\.\s", after, maxsplit=1)[0].strip()
    else:
        # "Authors. Title. Journal, vol, pages, 2019."
        parts = re.split(r"\.\s", raw)
        if len(parts) >= 2:
            title = parts[1].strip()
    if title:
        title = re.sub(r"\s*https?://\S+", "", title).strip()
        if len(title) < 8:
            title = None
    authors = raw[: year_m.start()].strip(" ,.(") if year_m else raw.split(".")[0].strip()
    return Citation(raw=raw, doi=doi, title=title, year=year, authors=authors or None)


def parse_references(block: str) -> list[Citation]:
    return [parse_entry(e) for e in _split_entries(block)]


def parse_inline_citations(body: str) -> list[Citation]:
    """Author-year mentions in running text, e.g. "(Aki and Richards, 2002)", de-duplicated."""
    seen: dict[tuple[str, int], Citation] = {}
    for m in _INLINE_AUTHOR_YEAR.finditer(body):
        key = (m.group("a"), int(m.group("y")))
        seen.setdefault(key, Citation(raw=m.group(0), authors=key[0], year=key[1]))
    return list(seen.values())


def extract_citations(text: str) -> tuple[str, list[Citation]]:
    """Return ``(body_without_reference_list, citations)``.

    With a reference list, its entries are the citations. Without one, inline
    author-year mentions are the citations. They carry no title or DOI, so
    they are unverifiable by construction. That is deliberate: a claim of
    support that a reader cannot follow up should not earn credit.
    """
    body, block = split_references(text)
    if block:
        return body, parse_references(block)
    return body, parse_inline_citations(body)


# ---------------------------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------------------------
def title_similarity(a: str, b: str) -> float:
    """max(sequence ratio, token Jaccard) on normalized titles, in [0, 1]."""
    na, nb = normalize_text(a), normalize_text(b)
    if not na or not nb:
        return 0.0
    seq = difflib.SequenceMatcher(None, na, nb).ratio()
    ta, tb = set(na.split()), set(nb.split())
    jac = len(ta & tb) / max(1, len(ta | tb))
    return max(seq, jac)


class CitationResolver(Protocol):
    def verify(self, c: Citation) -> VerifyResult: ...


class OfflineResolver:
    """Resolver over a fixed list of known works (tests; air-gapped dry runs)."""

    def __init__(self, known: Iterable[dict[str, Any]], title_threshold: float = 0.85):
        self.by_doi = {str(k["doi"]).lower(): k for k in known if k.get("doi")}
        self.known = list(known) if not isinstance(known, list) else known
        self.title_threshold = title_threshold

    def verify(self, c: Citation) -> VerifyResult:
        if c.doi:
            k = self.by_doi.get(c.doi.lower())
            if not k:
                return VerifyResult("not_found", detail="unknown DOI")
            if (
                c.title
                and k.get("title")
                and title_similarity(c.title, k["title"]) < self.title_threshold * 0.7
            ):
                return VerifyResult("not_found", matched_title=k.get("title"), detail="DOI/title mismatch")
            return VerifyResult("verified", matched_title=k.get("title"), similarity=1.0)
        if c.title:
            best = max(
                ((title_similarity(c.title, k.get("title", "")), k) for k in self.known),
                key=lambda x: x[0],
                default=(0.0, None),
            )
            if best[1] is not None and best[0] >= self.title_threshold:
                return VerifyResult("verified", matched_title=best[1].get("title"), similarity=best[0])
            return VerifyResult("not_found", similarity=best[0])
        return VerifyResult("not_found", detail="no DOI or title to look up")


class CrossrefResolver:
    """Crossref REST API resolver with an on-disk JSON cache and a polite rate limit.

    Crossref asks API users to identify themselves (``mailto``) and to keep
    request rates modest. Lookups are cached on disk, so the same citation is
    never fetched twice across RL steps or runs.
    """

    def __init__(
        self,
        *,
        mailto: str | None = None,
        cache_path: str | os.PathLike | None = None,
        base_url: str = "https://api.crossref.org",
        min_interval_s: float = 0.1,
        timeout_s: float = 10.0,
        title_threshold: float = 0.85,
        year_slack: int = 1,
        client: Any | None = None,
    ):
        import httpx

        self.base_url = base_url.rstrip("/")
        self.mailto = mailto
        self.min_interval_s = min_interval_s
        self.timeout_s = timeout_s
        self.title_threshold = title_threshold
        self.year_slack = year_slack
        self._client = client or httpx.Client(
            headers={"User-Agent": f"dpt-citation-check (mailto:{mailto or 'n/a'})"}
        )
        self._lock = threading.Lock()
        self._last = 0.0
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, dict[str, Any]] = {}
        if self.cache_path and self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text())
            except json.JSONDecodeError:
                log.warning("citation cache %s unreadable; starting empty", self.cache_path)

    # -- infra --------------------------------------------------------------------------------
    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        with self._lock:
            wait = self.min_interval_s - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
        params = dict(params or {})
        if self.mailto:
            params["mailto"] = self.mailto
        return self._client.get(f"{self.base_url}{path}", params=params, timeout=self.timeout_s)

    def _cache_put(self, key: str, res: VerifyResult) -> None:
        if res.status == "error":
            return  # never cache transient failures
        with self._lock:
            self._cache[key] = res.__dict__
            if self.cache_path:
                atomic_write_text(self.cache_path, json.dumps(self._cache, ensure_ascii=False))

    # -- lookup -------------------------------------------------------------------------------
    def verify(self, c: Citation) -> VerifyResult:
        key = (
            f"doi:{c.doi.lower()}|t:{normalize_text(c.title or '')}"
            if c.doi
            else f"t:{normalize_text(c.title or '')}|y:{c.year}"
        )
        if key in self._cache:
            return VerifyResult(**self._cache[key])
        try:
            res = self._verify_uncached(c)
        except Exception as e:  # fail closed: counted unverifiable, reported as error
            return VerifyResult("error", detail=f"{type(e).__name__}: {e}")
        self._cache_put(key, res)
        return res

    def _verify_uncached(self, c: Citation) -> VerifyResult:
        if c.doi:
            r = self._get(f"/works/{c.doi}")
            if r.status_code == 404:
                return VerifyResult("not_found", detail="DOI not registered")
            if r.status_code >= 400:
                return VerifyResult("error", detail=f"HTTP {r.status_code}")
            item = r.json().get("message", {})
            found_title = (item.get("title") or [""])[0]
            if c.title and found_title:
                sim = title_similarity(c.title, found_title)
                if sim < self.title_threshold * 0.7:
                    return VerifyResult(
                        "not_found",
                        matched_title=found_title,
                        similarity=sim,
                        detail="DOI resolves to a different work",
                    )
                return VerifyResult("verified", matched_title=found_title, similarity=sim)
            return VerifyResult("verified", matched_title=found_title, similarity=1.0)
        if c.title:
            r = self._get("/works", {"query.bibliographic": c.title, "rows": 5})
            if r.status_code >= 400:
                return VerifyResult("error", detail=f"HTTP {r.status_code}")
            best_sim, best_title = 0.0, None
            for item in r.json().get("message", {}).get("items", []):
                t = (item.get("title") or [""])[0]
                sim = title_similarity(c.title, t)
                year = None
                for k in ("published-print", "published-online", "issued"):
                    parts = item.get(k, {}).get("date-parts") or [[None]]
                    if parts and parts[0] and parts[0][0]:
                        year = int(parts[0][0])
                        break
                if c.year and year and abs(year - c.year) > self.year_slack:
                    continue
                if sim > best_sim:
                    best_sim, best_title = sim, t
            if best_sim >= self.title_threshold:
                return VerifyResult("verified", matched_title=best_title, similarity=best_sim)
            return VerifyResult("not_found", matched_title=best_title, similarity=best_sim)
        return VerifyResult("not_found", detail="no DOI or title to look up")


def verify_citations(citations: list[Citation], resolver: CitationResolver) -> CitationReport:
    rep = CitationReport(n_total=len(citations))
    for c in citations:
        v = resolver.verify(c)
        if v.status == "verified":
            rep.n_verified += 1
        else:
            rep.n_unverified += 1
            if v.status == "error":
                rep.n_errors += 1
        rep.details.append(
            {
                "raw": c.raw[:200],
                "doi": c.doi,
                "title": c.title,
                "status": v.status,
                "matched_title": v.matched_title,
                "similarity": round(v.similarity, 3),
            }
        )
    return rep
