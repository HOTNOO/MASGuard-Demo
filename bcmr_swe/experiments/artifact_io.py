"""Shared artifact I/O helpers for MASGuard experiment runners."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_object(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def load_optional_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return load_json_object(path)


def sha256_path(path: Path) -> str:
    hasher = hashlib.sha256()
    if path.is_dir():
        for child in sorted(path.iterdir(), key=lambda item: item.name):
            stat = child.stat()
            hasher.update(child.name.encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(b"dir" if child.is_dir() else b"file")
            hasher.update(b"\0")
            hasher.update(str(stat.st_size).encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
            hasher.update(b"\0")
        return hasher.hexdigest()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
