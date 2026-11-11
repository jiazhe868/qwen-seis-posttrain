"""Dataset manifests: every artifact the pipeline writes gets a sidecar ``*.manifest.json``.

A manifest records *what* the file is (name, version, record count, content
hash), *where it came from* (parent manifests' hashes, the hash of the resolved
config that produced it, git revision) and *what each stage removed* (stage
statistics such as "decontam removed 2,540"). Training jobs call
:func:`verify_manifest` on every input and log the content hashes, so any
reported number can be traced back to exact bytes (report §数据-6).

The funnel numbers in the report (91k → 82k → 79.5k → 74.1k → 60k) are exactly
the ``stage_stats`` of the RL problem-bank manifest.
"""

from __future__ import annotations

import datetime as _dt
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dpt.common.hashing import sha256_file, sha256_records
from dpt.common.io import read_json, read_jsonl, write_json

MANIFEST_SUFFIX = ".manifest.json"


def _git_revision() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


@dataclass
class Manifest:
    name: str                                   # e.g. "rl_bank"
    version: str                                # e.g. "v3"
    path: str                                   # data file this manifest describes
    num_records: int
    content_hash: str                           # order-independent hash of the records
    file_sha256: str                            # byte-level hash of the file
    created_at: str = field(default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat())
    git_revision: str = field(default_factory=_git_revision)
    config_hash: str | None = None              # sha256 of the resolved stage config
    parents: dict[str, str] = field(default_factory=dict)        # parent name -> content_hash
    stage_stats: list[dict[str, Any]] = field(default_factory=list)  # ordered funnel
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def manifest_path_for(data_path: str | os.PathLike) -> Path:
    return Path(str(data_path) + MANIFEST_SUFFIX)


def build_manifest(
    data_path: str | os.PathLike,
    name: str,
    version: str,
    *,
    config_hash: str | None = None,
    parents: dict[str, str] | None = None,
    stage_stats: list[dict[str, Any]] | None = None,
    notes: str = "",
) -> Manifest:
    """Compute hashes for an existing JSONL file and return (not write) its manifest."""
    records = list(read_jsonl(data_path))
    return Manifest(
        name=name,
        version=version,
        path=str(data_path),
        num_records=len(records),
        content_hash=sha256_records(records),
        file_sha256=sha256_file(data_path),
        config_hash=config_hash,
        parents=dict(parents or {}),
        stage_stats=list(stage_stats or []),
        notes=notes,
    )


def write_manifest(manifest: Manifest) -> Path:
    out = manifest_path_for(manifest.path)
    write_json(out, manifest.to_dict())
    return out


def load_manifest(data_path: str | os.PathLike) -> Manifest:
    return Manifest(**read_json(manifest_path_for(data_path)))


def verify_manifest(data_path: str | os.PathLike) -> Manifest:
    """Re-hash ``data_path`` and compare with its manifest. Raises ``ValueError`` on mismatch.

    Training entry points call this before reading any dataset; a mismatch
    means the file was modified after the pipeline produced it.
    """
    m = load_manifest(data_path)
    actual = sha256_file(data_path)
    if actual != m.file_sha256:
        raise ValueError(
            f"{data_path}: file hash {actual[:12]} != manifest {m.file_sha256[:12]}; "
            "the data changed after its manifest was written — rerun the pipeline stage."
        )
    return m
