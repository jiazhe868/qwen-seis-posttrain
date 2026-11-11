"""Ingestion: LaTeX normalization, license allow-list (fail closed), schema validation."""

from __future__ import annotations

import pytest

from dpt.common.io import write_jsonl
from dpt.data.ingest import ingest_candidates, ingest_files, normalize_latex

ALLOW = ["CC-BY-4.0"]


def _rec(q="Travel time for 120 km at 6 km/s?", lic="CC-BY-4.0", **kw):
    r = {"task": "t1_num", "question": q, "answer": {"value": 20, "unit": "s"}, "source": {"license": lic}}
    r.update(kw)
    return r


def test_normalize_latex_delimiters_are_canonical():
    assert normalize_latex(r"Let \( v = d/t \) and \[ E = mc^2 \]") == "Let $v = d/t$ and $$E = mc^2$$"


def test_license_allowlist_fails_closed():
    res = ingest_candidates([_rec(lic="proprietary"), _rec(lic=None), _rec()], license_allowlist=ALLOW)
    assert len(res.problems) == 1
    assert res.reasons["license"] == 2


def test_schema_errors_and_exact_duplicates_are_counted():
    res = ingest_candidates([_rec(), _rec(q="travel time for 120 km at 6 km/s ?"),
                             _rec(q="bad", answer={"value": "x"})], license_allowlist=ALLOW)
    assert len(res.problems) == 1
    assert res.reasons["exact_duplicate"] == 1 and res.reasons["schema"] == 1


def test_ingest_files_and_external_parser_guard(tmp_path):
    path = tmp_path / "c.jsonl"
    write_jsonl(path, [_rec()])
    assert len(ingest_files([str(path)], parser="preparsed_jsonl", license_allowlist=ALLOW).problems) == 1
    with pytest.raises(NotImplementedError):
        ingest_files([str(path)], parser="external_ocr", license_allowlist=ALLOW)
    with pytest.raises(ValueError):
        ingest_files([str(path)], parser="nope", license_allowlist=ALLOW)
