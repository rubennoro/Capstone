"""GripperFile: loads and checks deployment/gripper.yaml (STRUCTURE.md section 6.5)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from deploy.errors import GeometryError

UP_SIDES = {"+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0), "+y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0)}
ULTRAWIDE_KEYS = ("position_mm", "up_side", "tilt_deg")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass(frozen=True, eq=False)
class GripperFile:
    position_m: np.ndarray      # (3,), the ultrawide lens centre in the hand frame, metres
    up_axis: np.ndarray         # (3,), hand-frame unit vector of up_side
    tilt_deg: float
    path: Path

    @classmethod
    def load(cls, path: Path) -> "GripperFile":
        """Read and check the file; every problem is collected into one GeometryError."""
        path = Path(path).expanduser().absolute()
        name = path.name
        try:
            data = yaml.safe_load(path.read_text())
        except FileNotFoundError:
            raise GeometryError(f"{name}: not found: {path}") from None
        except (OSError, yaml.YAMLError) as exc:
            raise GeometryError(f"{name}: cannot read {path}: {exc}") from None
        if not isinstance(data, dict):
            raise GeometryError(f"{name}: expected a map with the key ultrawide")
        problems = [f"{name}: {key}: unknown key" for key in data if key != "ultrawide"]
        cam = data.get("ultrawide")
        if not isinstance(cam, dict):
            problems.append(f"{name}: ultrawide: required, a map of {', '.join(ULTRAWIDE_KEYS)}")
            raise GeometryError(problems)
        problems += [f"{name}: ultrawide.{key}: unknown key" for key in cam if key not in ULTRAWIDE_KEYS]
        problems += [f"{name}: ultrawide.{key}: required" for key in ULTRAWIDE_KEYS if key not in cam]

        position = cam.get("position_mm")
        if "position_mm" in cam and not (isinstance(position, list) and len(position) == 3
                                         and all(_is_number(v) and math.isfinite(v) for v in position)):
            problems.append(f"{name}: ultrawide.position_mm: expected 3 finite numbers (mm), got {position!r}")
        up_side = cam.get("up_side")
        if "up_side" in cam and not (isinstance(up_side, str) and up_side in UP_SIDES):
            problems.append(f"{name}: ultrawide.up_side: expected one of {', '.join(UP_SIDES)}, got {up_side!r}")
        tilt = cam.get("tilt_deg")
        if "tilt_deg" in cam and not (_is_number(tilt) and math.isfinite(tilt) and 0.0 <= tilt < 90.0):
            problems.append(f"{name}: ultrawide.tilt_deg: expected a number in [0, 90), got {tilt!r}")
        if problems:
            raise GeometryError(problems)
        return cls(position_m=np.array(position, dtype=float) / 1000.0, up_axis=np.array(UP_SIDES[up_side]),
                   tilt_deg=float(tilt), path=path)
