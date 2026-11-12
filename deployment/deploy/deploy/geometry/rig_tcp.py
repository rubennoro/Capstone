"""RigTcp: reads and writes tcp.T_cam_tcp in a rig file (STRUCTURE.md section 6.6, step 3)."""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from deploy.errors import GeometryError
from deploy.geometry.frames import as_rigid

_TCP_LINE = re.compile(r"^tcp:(\s|#|$)")
_HEADER_1 = "# T_cam_tcp COMPUTED by `python -m deploy.geometry`"
_HEADER_2 = "# and franka_control tcp.F_T_TCP."
_ROUND_TRIP_TOL = 1e-9       # written with 9 decimals, so read-back differs by at most 5e-10


class RigTcp:
    @staticmethod
    def read(rig_file: Path) -> np.ndarray:
        """tcp.T_cam_tcp as a 4x4: either `matrix`, or rotation{euler_xyz_deg} + translation_m (as dataproc)."""
        rig_file = Path(rig_file)
        label = f"{rig_file.name}: tcp.T_cam_tcp"
        try:
            data = yaml.safe_load(rig_file.read_text())
        except FileNotFoundError:
            raise GeometryError(f"rig file not found: {rig_file}") from None
        except (OSError, yaml.YAMLError) as exc:
            raise GeometryError(f"cannot read rig file {rig_file}: {exc}") from None
        tcp = data.get("tcp") if isinstance(data, dict) else None
        block = tcp.get("T_cam_tcp") if isinstance(tcp, dict) else None
        if not isinstance(block, dict):
            raise GeometryError(f"{label}: missing (expected tcp: T_cam_tcp: matrix or rotation + translation_m)")
        if "matrix" in block:
            return as_rigid(block["matrix"], label)
        try:
            euler = block["rotation"]["euler_xyz_deg"]
            T = np.eye(4)
            T[:3, :3] = Rotation.from_euler("xyz", np.array(euler, dtype=float), degrees=True).as_matrix()
            T[:3, 3] = np.array(block["translation_m"], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise GeometryError(f"{label}: expected matrix, or rotation: {{euler_xyz_deg: [3]}} and "
                                f"translation_m: [3] ({type(exc).__name__}: {exc})") from None
        return as_rigid(T, label)

    @staticmethod
    def write(rig_file: Path, T_cam_tcp: np.ndarray, gripper_file: Path) -> None:
        """Replace only the top-level `tcp:` block (and an earlier COMPUTED header) as text, keeping every other
        line, then read the file back and check the matrix round-trips."""
        rig_file = Path(rig_file)
        T = as_rigid(T_cam_tcp, "T_cam_tcp")
        lines = rig_file.read_text().splitlines(keepends=True)
        starts = [i for i, line in enumerate(lines) if _TCP_LINE.match(line)]
        if len(starts) != 1:
            raise GeometryError(f"{rig_file}: expected one top-level `tcp:` block, found {len(starts)}")
        start = end = starts[0]
        end += 1
        while end < len(lines) and (not lines[end].strip() or lines[end][0] in " \t"):
            end += 1                                  # the block: indented (or blank) lines after `tcp:`
        while end > start + 1 and not lines[end - 1].strip():
            end -= 1                                  # keep blank lines that separate it from the next key
        if start >= 2 and lines[start - 2].startswith(_HEADER_1) and lines[start - 1].startswith(_HEADER_2):
            start -= 2                                # the header of an earlier run
        rows = ", ".join("[" + ", ".join(_number(v) for v in row) + "]" for row in T)
        block = [
            f"{_HEADER_1} on {date.today().isoformat()} from {Path(gripper_file).expanduser().absolute()}\n",
            f"{_HEADER_2} Do not edit by hand: change gripper.yaml and rerun the tool.\n",
            "tcp:\n",
            "  T_cam_tcp:\n",
            f"    matrix: [{rows}]   # 4x4, row-major\n",
            "  provenance: nominal\n",
        ]
        rig_file.write_text("".join(lines[:start] + block + lines[end:]))
        back = RigTcp.read(rig_file)
        if not np.allclose(back, T, atol=_ROUND_TRIP_TOL, rtol=0.0):
            raise GeometryError(f"{rig_file}: T_cam_tcp read back differs from what was written")


def _number(value: float) -> str:
    text = f"{value:.9f}"
    return "0.000000000" if text == "-0.000000000" else text
