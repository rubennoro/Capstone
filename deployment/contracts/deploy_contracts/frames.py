"""TcpFrame: frames.yaml, the one definition of the TCP (STRUCTURE.md 4.2, 4.4).

franka_control computes O_T_TCP = O_T_F x F_T_TCP for C2 and its gate; the laptop compares the PC's hello F_T_TCP
with this file (compare_hello) and refuses a trained policy on the real arm while `settled` is false.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import yaml

from deploy_contracts.c2_state import rigid_problem
from deploy_contracts.limits import ContractLimits
from deploy_contracts.paths import frames_file

_KEYS = ("F_T_TCP", "settled")


@dataclass(frozen=True, eq=False)
class TcpFrame:
    F_T_TCP: np.ndarray    # float64 (4, 4), row-major, metres; read-only
    settled: bool          # laptop only: false -> no trained policy on the real arm (PLAN section 9)

    _cached: ClassVar[TcpFrame | None] = None

    @classmethod
    def load(cls, path: Path | str | None = None) -> TcpFrame:
        """The TCP frame. With no path: deployment/contracts/frames.yaml, read once and cached.

        Raises ValueError naming the file and the key: missing or unknown key, not 4 rows of 4 finite numbers, not
        rigid within limits.yaml rotation_tol, settled not a bool.
        """
        if path is None:
            if cls._cached is None:
                cls._cached = cls._read(frames_file())
            return cls._cached
        return cls._read(Path(path))

    @classmethod
    def _read(cls, path: Path) -> TcpFrame:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a map with the keys {', '.join(_KEYS)}")
        for key in raw:
            if key not in _KEYS:
                raise ValueError(f"{path}: unknown key {key!r}")
        for key in _KEYS:
            if key not in raw:
                raise ValueError(f"{path}: missing key {key!r}")
        rows = raw["F_T_TCP"]
        if not isinstance(rows, list) or len(rows) != 4 or not all(isinstance(r, list) and len(r) == 4 for r in rows):
            raise ValueError(f"{path}: F_T_TCP must be 4 rows of 4 numbers (row-major), got {rows!r}")
        for i, row in enumerate(rows):
            for j, v in enumerate(row):
                if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
                    raise ValueError(f"{path}: F_T_TCP[{i}][{j}] must be a finite number, got {v!r}")
        matrix = np.array(rows, dtype=np.float64)
        problem = rigid_problem(matrix, ContractLimits.load().rotation_tol)
        if problem is not None:
            raise ValueError(f"{path}: F_T_TCP is not a rigid transform: {problem}")
        matrix.flags.writeable = False
        settled = raw["settled"]
        if not isinstance(settled, bool):
            raise ValueError(f"{path}: settled must be true or false, got {settled!r}")
        return cls(F_T_TCP=matrix, settled=settled)
