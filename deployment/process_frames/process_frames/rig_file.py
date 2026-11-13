"""The rig file for live use (STRUCTURE.md §5.3): the copied dataproc loader plus the checks live use needs.

``python -m process_frames.rig_file FILE [--json]`` prints the rig summary (or one JSON line) and exits 0 if the file
is valid, else 1. ``launch.py`` runs it in a subprocess, so the rig is parsed by exactly one loader everywhere.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from process_frames._dataproc.config import ConfigError, RigConfig, load_rig

#: The sensors contract C0 allows (contracts' FrameInfo.sensor).
C0_SENSORS = ("imx519", "imx219")
ROLES = ("main", "ultrawide")


def load_live_rig(path: Path) -> RigConfig:
    """``load_rig(path)``, then: both cameras' sensor is one C0 allows. Raises ConfigError naming the file and key."""
    rig = load_rig(path)
    for role in ROLES:
        sensor = rig.cameras[role].sensor
        if sensor not in C0_SENSORS:
            raise ConfigError(f"{path}: cameras.{role}.sensor: {sensor!r} is not a sensor C0 allows "
                              f"({', '.join(C0_SENSORS)})")
    return rig


def camera_summary(rig: RigConfig) -> dict[str, dict]:
    out = {}
    for role in ROLES:
        cam = rig.cameras[role]
        out[role] = {"sensor": cam.sensor, "width": int(cam.resolution[0]), "height": int(cam.resolution[1]),
                     "fps": float(cam.fps)}
    return out


def rig_summary(rig: RigConfig) -> str:
    """e.g. ``rig_pi_v2: main imx519 1600x1200@30, ultrawide imx219 1280x960@30``."""
    parts = [f"{role} {c['sensor']} {c['width']}x{c['height']}@{c['fps']:g}" for role, c in camera_summary(rig).items()]
    return f"{rig.rig_id}: {', '.join(parts)}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m process_frames.rig_file", description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--json", action="store_true", help="print one JSON line")
    args = parser.parse_args(argv)
    try:
        rig = load_live_rig(args.file)
    except (ConfigError, OSError) as e:
        if args.json:
            print(json.dumps({"ok": False, "error": str(e)}))
        else:
            print(f"rig file invalid: {e}")
        return 1
    if args.json:
        print(json.dumps({"ok": True, "rig_id": rig.rig_id, "cameras": camera_summary(rig)}))
    else:
        print(rig_summary(rig))
    return 0


if __name__ == "__main__":
    sys.exit(main())
