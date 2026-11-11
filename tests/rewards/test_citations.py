"""Citation parsing and verification (step-245 fabricated-citation hack). HTTP is mocked."""

from __future__ import annotations

import json

import httpx

from dpt.rewards.citations import (
    Citation,
    CrossrefResolver,
    OfflineResolver,
    extract_citations,
    parse_entry,
    split_references,
    title_similarity,
    verify_citations,
)

REPORT = """Coseismic slip concentrated at 10-20 km depth (Aki and Richards, 2002).

References
[1] Aki, K., & Richards, P. G. (2002). Quantitative Seismology. University Science Books.
[2] Hanks, T. C., & Kanamori, H. (1979). A moment magnitude scale. Journal of Geophysical Research, 84(B5), 2348-2350. doi:10.1029/JB084iB05p02348
[3] Smith, J. (2021). Imaginary faults of the outer ring. Fake Journal, 1, 1-2.
"""


def test_split_and_parse():
    body, block = split_references(REPORT)
    assert "References" not in body and block.startswith("[1]")
    _, cits = extract_citations(REPORT)
    assert len(cits) == 3
    assert cits[1].doi == "10.1029/JB084iB05p02348" and cits[1].year == 1979
    assert cits[1].title == "A moment magnitude scale"


def test_inline_only_citations_are_unverifiable_by_construction():
    body, cits = extract_citations("Slip was shallow (Smith et al., 2019), as noted (Lee and Kim, 2020).")
    assert len(cits) == 2 and all(c.title is None and c.doi is None for c in cits)
    rep = verify_citations(cits, OfflineResolver([]))
    assert rep.n_unverified == 2


def test_parse_entry_quoted_title():
    c = parse_entry('J. Doe, "Crustal anisotropy beneath Tibet," Geophys. J. Int., 2015.')
    assert c.title == "Crustal anisotropy beneath Tibet" and c.year == 2015


def test_title_similarity():
    assert title_similarity("A moment magnitude scale", "A Moment Magnitude Scale.") > 0.95
    assert title_similarity("A moment magnitude scale", "Imaginary faults") < 0.4


def test_offline_resolver_doi_title_mismatch():
    known = [{"doi": "10.1029/JB084iB05p02348", "title": "A moment magnitude scale"}]
    r = OfflineResolver(known)
    assert (
        r.verify(Citation(raw="", doi="10.1029/JB084iB05p02348", title="A moment magnitude scale")).status
        == "verified"
    )
    # real DOI attached to an invented paper
    assert (
        r.verify(
            Citation(raw="", doi="10.1029/JB084iB05p02348", title="Neural nets predict all quakes")
        ).status
        == "not_found"
    )


def _crossref_transport(calls):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.path.startswith("/works/10.1029"):
            return httpx.Response(200, json={"message": {"title": ["A moment magnitude scale"]}})
        if request.url.path.startswith("/works/"):
            return httpx.Response(404, json={})
        q = request.url.params.get("query.bibliographic", "")
        if "Quantitative Seismology" in q:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "items": [{"title": ["Quantitative Seismology"], "issued": {"date-parts": [[2002]]}}]
                    }
                },
            )
        if "boom" in q:
            return httpx.Response(503, json={})
        return httpx.Response(200, json={"message": {"items": [{"title": ["Something else entirely"]}]}})

    return httpx.MockTransport(handler)


def test_crossref_resolver_with_cache(tmp_path):
    calls: list[str] = []
    client = httpx.Client(transport=_crossref_transport(calls))
    cache = tmp_path / "cache.json"
    res = CrossrefResolver(mailto="t@example.org", cache_path=cache, min_interval_s=0, client=client)
    _, cits = extract_citations(REPORT)
    rep = verify_citations(cits, res)
    assert (rep.n_verified, rep.n_unverified, rep.n_errors) == (2, 1, 0)
    n_calls = len(calls)
    verify_citations(cits, res)  # second pass is fully cached
    assert len(calls) == n_calls
    assert len(json.loads(cache.read_text())) == 3


def test_crossref_errors_fail_closed_and_are_not_cached(tmp_path):
    calls: list[str] = []
    res = CrossrefResolver(
        cache_path=tmp_path / "c.json",
        min_interval_s=0,
        client=httpx.Client(transport=_crossref_transport(calls)),
    )
    rep = verify_citations([Citation(raw="x", title="boom boom paper title")], res)
    assert rep.n_unverified == 1 and rep.n_errors == 1
    verify_citations([Citation(raw="x", title="boom boom paper title")], res)
    assert len(calls) == 2  # retried, not served from cache
