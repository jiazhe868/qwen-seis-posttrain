"""Ingestion: parsed documents / candidate problems → validated :class:`Problem` records.

Report §处理流水线 step 1: "PDFs parsed with layout-aware OCR, formulas converted
to LaTeX, filtered by license. 91k candidates."

What lives here and what does not
---------------------------------
OCR itself is not reimplemented. Layout-aware PDF OCR is a separate GPU
service (for example a Nougat/olmOCR-class model) that is run once over the
raw corpus. It writes one JSON object per document or per extracted problem.
This module defines the :class:`DocumentParser` protocol that the service
output must satisfy and ships :class:`PreParsedJSONLParser` for that output.
Adding a new parser means implementing ``parse(path) -> Iterable[dict]`` and
registering it in :data:`PARSERS`.

Ingestion then
1. normalizes LaTeX delimiters, so the same formula written as ``\\(..\\)`` or
   ``$..$`` hashes identically for dedup/decontam,
2. applies a license **allow-list**. Unknown or missing license means drop.
   That is the fail-closed choice, because a problem we may not train on
   cannot be put back after a model ships.
3. validates the schema and assigns stable uids.

Each dropped record keeps a machine-readable reason. The counts become the
first row of the funnel.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

from dpt.common.io import read_jsonl
from dpt.common.logging import get_logger
from dpt.data.schema import Problem, SchemaError

log = get_logger(__name__)


class DocumentParser(Protocol):
    """Anything that turns a raw input file into candidate-problem dicts."""

    name: str

    def parse(self, path: str) -> Iterable[dict[str, Any]]:  # pragma: no cover - protocol
        ...


class PreParsedJSONLParser:
    """Reads JSONL written by the external OCR/extraction service (one candidate per line).

    Expected fields per line: ``task``, ``question``, ``answer`` and ``source``
    (with ``license``, ``origin``, ``doc_id``), plus an optional ``meta``.
    """

    name = "preparsed_jsonl"

    def parse(self, path: str) -> Iterable[dict[str, Any]]:
        yield from read_jsonl(path)


class ExternalOCRParser:
    """Placeholder for a direct PDF → problem parser. It deliberately raises.

    Running OCR inline in the pipeline would couple a GPU model to a CPU job
    and make reruns non-deterministic. Run the OCR service separately, then
    ingest its JSONL with :class:`PreParsedJSONLParser`.
    """

    name = "external_ocr"

    def parse(self, path: str) -> Iterable[dict[str, Any]]:
        raise NotImplementedError(
            "Run the layout-aware OCR service over the PDFs first and ingest its JSONL output with "
            "parser=preparsed_jsonl (see docs/data_pipeline.md §1)."
        )


PARSERS: dict[str, type] = {
    PreParsedJSONLParser.name: PreParsedJSONLParser,
    ExternalOCRParser.name: ExternalOCRParser,
}

_INLINE_PAREN = re.compile(r"\\\((.+?)\\\)", re.S)
_DISPLAY_BRACKET = re.compile(r"\\\[(.+?)\\\]", re.S)
_SPACES = re.compile(r"[ \t]+")


def normalize_latex(text: str) -> str:
    """Canonicalize math delimiters and whitespace: ``\\(x\\)`` → ``$x$``, ``\\[x\\]`` → ``$$x$$``."""
    t = _DISPLAY_BRACKET.sub(lambda m: "$$" + m.group(1).strip() + "$$", text)
    t = _INLINE_PAREN.sub(lambda m: "$" + m.group(1).strip() + "$", t)
    t = _SPACES.sub(" ", t)
    return "\n".join(line.rstrip() for line in t.splitlines()).strip()


@dataclass
class IngestResult:
    problems: list[Problem]
    dropped: list[dict[str, Any]] = field(default_factory=list)   # {"record": ..., "reason": ...}
    reasons: Counter = field(default_factory=Counter)


def license_ok(source: dict[str, Any], allowlist: Iterable[str]) -> bool:
    lic = str(source.get("license") or "").strip()
    return bool(lic) and lic in set(allowlist)


def ingest_candidates(
    records: Iterable[dict[str, Any]],
    *,
    license_allowlist: Iterable[str],
) -> IngestResult:
    """Validate, normalize and license-filter raw candidate dicts. Exact-duplicate uids are dropped here."""
    allow = list(license_allowlist)
    out = IngestResult(problems=[])
    seen: set[str] = set()
    for rec in records:
        src = rec.get("source") or {}
        if not license_ok(src, allow):
            out.dropped.append({"record": rec, "reason": f"license:{src.get('license') or 'missing'}"})
            out.reasons["license"] += 1
            continue
        rec = dict(rec)
        rec["question"] = normalize_latex(str(rec.get("question", "")))
        try:
            p = Problem.from_dict(rec)
        except SchemaError as e:
            out.dropped.append({"record": rec, "reason": f"schema:{e}"})
            out.reasons["schema"] += 1
            continue
        if p.uid in seen:  # identical normalized question; MinHash would catch it too, but cheaper here
            out.dropped.append({"record": rec, "reason": "exact_duplicate"})
            out.reasons["exact_duplicate"] += 1
            continue
        seen.add(p.uid)
        out.problems.append(p)
    log.info("ingest: kept %d, dropped %s", len(out.problems), dict(out.reasons))
    return out


def ingest_files(paths: Iterable[str], *, parser: str, license_allowlist: Iterable[str]) -> IngestResult:
    """Parse every input file with the named parser, then :func:`ingest_candidates`."""
    if parser not in PARSERS:
        raise ValueError(f"unknown parser {parser!r}; available: {sorted(PARSERS)}")
    p = PARSERS[parser]()

    def _all() -> Iterable[dict[str, Any]]:
        for path in paths:
            yield from p.parse(path)

    return ingest_candidates(_all(), license_allowlist=license_allowlist)
