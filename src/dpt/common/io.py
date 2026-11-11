"""JSONL / YAML I/O helpers with atomic writes.

Why atomic writes: pipeline stages and checkpoints are written to shared NFS.
A job killed mid-write must never leave a truncated file that the next stage
silently reads as "fewer records" — that is exactly the kind of quiet data
change the manifest/hash system (report §数据-6 "版本化") is meant to prevent.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml


def ensure_parent(path: str | os.PathLike) -> Path:
    """Create the parent directory of ``path`` if needed and return ``Path(path)``."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_text(path: str | os.PathLike, text: str) -> None:
    """Write ``text`` to ``path`` via a temp file + ``os.replace`` (atomic on POSIX)."""
    p = ensure_parent(path)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=f".{p.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read_jsonl(path: str | os.PathLike) -> Iterator[dict[str, Any]]:
    """Yield one dict per non-empty line. Raises with the line number on malformed JSON."""
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:  # pragma: no cover - message path
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e


def write_jsonl(path: str | os.PathLike, records: Iterable[dict[str, Any]]) -> int:
    """Atomically write ``records`` as JSONL; returns the number of records written."""
    lines = [json.dumps(r, ensure_ascii=False, sort_keys=True) for r in records]
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def read_yaml(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: str | os.PathLike, obj: Any) -> None:
    atomic_write_text(path, yaml.safe_dump(obj, sort_keys=False, allow_unicode=True))


def write_json(path: str | os.PathLike, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def read_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
