"""The sim eraser's width: ONE value for every sim eraser that is built or rendered (2026-11-21, the user).

WIDTH_M = W0's 0.030 m (upstream f632576 policy.yaml gripper_map.robot_closed_m: the 0.040 m stroke straddles it).
stations/sim_rig.yaml and every generated look station (look/make_stations.py) carry it in scene.eraser.size_m[1];
`python -m sim_deploy.look.make_stations --check` fails if any station's eraser width differs. Code that projects or
renders an eraser box of its own (camera_study/robot_trials.py, corrections/geometry*.py) takes its width from here
and keeps its own length / thickness. The user's REAL eraser is 127 x 51 x 12 mm: only measurements of real frames
(look/make_assets.py ERASER_MM, the fits to real images) use that."""

from __future__ import annotations

from pathlib import Path

import yaml

WIDTH_M = 0.030
STATIONS = Path(__file__).resolve().parent / "stations"


def size_m(length_m: float, thickness_m: float) -> list[float]:
    """[length, WIDTH_M, thickness] (the scene.eraser.size_m order)."""
    return [float(length_m), WIDTH_M, float(thickness_m)]


def station_width_errors(paths=None, tol_m: float = 1e-9) -> list[str]:
    """One line per station YAML (default: stations/*.yaml) whose scene.eraser.size_m width is not WIDTH_M."""
    errs = []
    for p in sorted(paths if paths is not None else STATIONS.glob("*.yaml")):
        doc = yaml.safe_load(Path(p).read_text()) or {}
        er = (doc.get("scene") or {}).get("eraser") if isinstance(doc, dict) else None
        if not isinstance(er, dict) or "size_m" not in er:
            continue
        w = float(er["size_m"][1])
        if abs(w - WIDTH_M) > tol_m:
            errs.append(f"{Path(p).name}: eraser width {w * 1e3:.1f} mm, must be {WIDTH_M * 1e3:.1f} mm "
                        f"(sim_deploy/eraser.py WIDTH_M)")
    return errs
