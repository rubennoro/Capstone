"""Process-specific file helpers for ``process`` / ``status``: JSON written atomically (via
:func:`dataproc.fsutil.write_atomic`), canonical hashing of stage inputs, ``status.json`` read/write.

Nothing here knows about stages or episodes. Generic filesystem plumbing (atomic writes, renames, locks) lives in
:mod:`dataproc.fsutil`.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from dataproc.fsutil import write_atomic

__all__ = ["STATUS_FILE", "ERROR_FILE", "jsonable", "canonical_json", "sha256_hex", "hash_of", "write_json",
           "read_json", "iso_utc", "load_status", "save_status"]

STATUS_FILE = "status.json"
ERROR_FILE = "error.txt"


def jsonable(x) -> Any:
    """``x`` with numpy arrays / scalars and paths converted to plain JSON types (recursively)."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x


def canonical_json(x) -> str:
    """Sorted keys, no whitespace: equal values give equal strings (the input of :func:`hash_of`)."""
    return json.dumps(jsonable(x), sort_keys=True, separators=(",", ":"), allow_nan=True)


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def hash_of(x) -> str:
    """sha256 of :func:`canonical_json` of ``x``."""
    return sha256_hex(canonical_json(x).encode("utf-8"))


def write_json(path, obj) -> None:
    """``obj`` as indented JSON, written atomically (dot temp file + fsync + rename)."""
    write_atomic(Path(path), json.dumps(jsonable(obj), indent=2) + "\n")


def read_json(path, default=None):
    """Parsed JSON, or ``default`` if the file is missing or not valid JSON."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def iso_utc(t: float) -> str:
    return _dt.datetime.fromtimestamp(t, _dt.timezone.utc).isoformat(timespec="seconds")


def load_status(processed_dir) -> dict:
    """``<processed_dir>/status.json`` as ``{stage: record}`` (``{}`` if missing or unreadable)."""
    d = read_json(Path(processed_dir) / STATUS_FILE, {})
    return d if isinstance(d, dict) else {}


def save_status(processed_dir, status: dict) -> None:
    write_json(Path(processed_dir) / STATUS_FILE, status)
