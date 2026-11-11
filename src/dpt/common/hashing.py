"""Content hashing for data versioning and provenance.

Every dataset written by the pipeline carries a content hash, and the RL/SFT
training logs record the hashes of everything they read. "Which exact data
produced this checkpoint?" must always have a one-line answer
(report §数据 处理流水线 step 6).

The hash is over a *canonical* serialization (sorted keys, no whitespace
variance, records sorted by ``uid`` when requested), so semantically identical
datasets hash identically regardless of write order or JSON formatting.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Iterable


def canonical_json(obj: Any) -> str:
    """Deterministic JSON string: sorted keys, compact separators, UTF-8 preserved."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_obj(obj: Any) -> str:
    """Hash of an arbitrary JSON-serializable object (e.g. a resolved config)."""
    return sha256_text(canonical_json(obj))


def sha256_file(path: str | os.PathLike, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_records(records: Iterable[dict[str, Any]], sort_by: str | None = "uid") -> str:
    """Order-independent (when ``sort_by`` is set) hash of a collection of records.

    Each record is canonicalized and hashed; the per-record digests are then
    sorted (by ``sort_by`` value, falling back to digest) and hashed together.
    """
    items: list[tuple[str, str]] = []
    for r in records:
        digest = sha256_text(canonical_json(r))
        key = str(r.get(sort_by, digest)) if sort_by else ""
        items.append((key, digest))
    if sort_by:
        items.sort()
    h = hashlib.sha256()
    for _, digest in items:
        h.update(digest.encode("ascii"))
    return h.hexdigest()


def short_hash(text: str, n: int = 16) -> str:
    """First ``n`` hex chars of sha256 — used for stable record ids (e.g. ``p-<hash>``)."""
    return sha256_text(text)[:n]
