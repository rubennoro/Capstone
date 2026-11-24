"""The sim's run looks (2026-11-21): which station the sim Pi and the sim PC load for a closed-loop run.

    python -m sim_deploy.looks robot        prints: <station yaml> <build dir>   (run.sh reads these two words)

robot (default) = stations/sim_rig_look_robot.yaml (the real robot cell, W2), train = stations/sim_rig_look_train.yaml
(the training scene, W2), current = stations/sim_rig.yaml (E7's look). All three have W0's 30 mm wide eraser
(look/make_stations.py RUN_LOOKS). The model is built from the station into build/<station id>/; the sim PC and the
sim Pi must load the same build (the sim Pi refuses another model sha).
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

from sim_deploy import ROOT

LOOKS = {"robot": "sim_rig_look_robot.yaml", "train": "sim_rig_look_train.yaml", "current": "sim_rig.yaml"}
DEFAULT_LOOK = "robot"


def station(look: str) -> Path:
    if look not in LOOKS:
        raise ValueError(f"look {look!r}: expected one of {', '.join(LOOKS)}")
    return ROOT / "stations" / LOOKS[look]


def build_dir(look: str) -> Path:
    """build/<station id>/ (the builder's output folder for that station)."""
    sid = (yaml.safe_load(station(look).read_text()) or {}).get("id")
    if not sid:
        raise ValueError(f"{station(look)}: no id")
    return ROOT / "build" / sid


def look_of_model(model_json: Path | str) -> str | None:
    """The look whose build holds this model.json, or None."""
    d = Path(model_json).expanduser().resolve().parent
    for look in LOOKS:
        if build_dir(look).resolve() == d:
            return look
    return None


def main(argv: list[str]) -> int:
    if len(argv) != 1 or argv[0] not in LOOKS:
        print(f"usage: python -m sim_deploy.looks {{{'|'.join(LOOKS)}}}", file=sys.stderr)
        return 2
    print(station(argv[0]), build_dir(argv[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
