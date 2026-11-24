# adapted from rnoro/deployment:deployment/contracts/contracts/limits.py @ 91b178b
# changes: only the three C4 tolerances (anchor_row_tol, rot6d_norm_tol, rot6d_dot_max), from c4_limits.yaml next to
# this file; the shared limits are deploy_contracts.ContractLimits (deployment/contracts/limits.yaml).
"""C4Limits: the three C4 tolerances, loaded once from c4_limits.yaml (STRUCTURE.md 3.3). Strict like ContractLimits:
a missing, unknown, non-numeric or non-positive key is refused."""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import yaml

C4_LIMITS_FILE = Path(__file__).with_name("c4_limits.yaml")


@dataclass(frozen=True)
class C4Limits:
    anchor_row_tol: float      # Observation: last rows equal 0 / identity rot6d within this
    rot6d_norm_tol: float      # rot6d vectors of length 1 +- this
    rot6d_dot_max: float       # rot6d vectors perpendicular within this (|cos angle| <= this)

    _cached: ClassVar[C4Limits | None] = None

    @classmethod
    def load(cls, path: Path | str | None = None) -> C4Limits:
        """The C4 limits. With no path: c4_limits.yaml next to this file, read once and cached.

        Raises ValueError naming the file and the key if one is missing, unknown or has a bad value.
        """
        if path is None:
            if cls._cached is None:
                cls._cached = cls._read(C4_LIMITS_FILE)
            return cls._cached
        return cls._read(Path(path))

    @classmethod
    def _read(cls, path: Path) -> C4Limits:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a map of limits")
        names = [f.name for f in dataclasses.fields(cls)]
        for key in raw:
            if key not in names:
                raise ValueError(f"{path}: unknown key {key!r}")
        values = {}
        for name in names:
            if name not in raw:
                raise ValueError(f"{path}: missing key {name!r}")
            v = raw[name]
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v <= 0:
                raise ValueError(f"{path}: {name} must be a finite number > 0, got {v!r}")
            values[name] = float(v)
        return cls(**values)


C4Limits.load()  # a broken c4_limits.yaml fails at import time, so it fails every test
