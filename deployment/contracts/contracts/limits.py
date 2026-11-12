"""ContractLimits: the numbers the validators use, loaded once from limits.yaml (STRUCTURE.md section 4.7)."""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import yaml

LIMITS_FILE = Path(__file__).with_name("limits.yaml")


@dataclass(frozen=True)
class ContractLimits:
    rotation_tol: float
    anchor_row_tol: float
    rot6d_norm_tol: float
    rot6d_dot_max: float
    gripper_width_max_m: float
    max_waypoints: int

    _cached: ClassVar[ContractLimits | None] = None

    @classmethod
    def load(cls, path: Path | str | None = None) -> ContractLimits:
        """The limits. With no path: limits.yaml next to this file, read once and cached.

        Raises ValueError naming the key if one is missing, unknown or has a bad value.
        """
        if path is None:
            if cls._cached is None:
                cls._cached = cls._read(LIMITS_FILE)
            return cls._cached
        return cls._read(Path(path))

    @classmethod
    def _read(cls, path: Path) -> ContractLimits:
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
            if name == "max_waypoints":
                if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                    raise ValueError(f"{path}: {name} must be an integer >= 1, got {v!r}")
            elif not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v <= 0:
                raise ValueError(f"{path}: {name} must be a finite number > 0, got {v!r}")
            else:
                v = float(v)
            values[name] = v
        return cls(**values)


ContractLimits.load()  # a broken limits.yaml fails at import time, so it fails every test
