"""Sim Pi settings: defaults, then the station file's ``sim_pi:`` block, then the command line.

Station file (``--station``, e.g. ``sim_deploy/stations/sim_rig.yaml``): keys of an optional ``sim_pi:`` block
override the defaults below (same names). Without ``model``, the model is the station's build,
``<sim_deploy>/build/<station id>/scene.xml`` (DESIGN.md section 2). Relative paths are taken from the station
file's folder.

Timing defaults copy the live Pi (PI_PAYLOAD.md 0.6, packet 800 of 2026-11-12): exposures 25 640 / 33 287 us, the
ultrawide 16.3 ms newer than its main frame, each packet sent 45.8 ms after its main frame's ``t_sensor_ns``. No
motion blur is rendered, so the exposure only sets ``t_sensor_ns`` = ``t_mid_ns`` + exposure / 2.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sim_deploy.sim_pi._deploy import DEFAULT_RIG_FILE, REPO


@dataclass
class SimPiSettings:
    model: Path | None = None                       # scene.xml (model.json beside it)
    model_wait_s: float = 600.0                     # poll this long for model.json
    state: str = "tcp://127.0.0.1:5650"             # sim.state/v1 SUB (DESIGN.md section 4)
    bind: str = "tcp://127.0.0.1:5555"              # pi_stream/v1 PUB (PI_PAYLOAD.md 0.1: port 5555)
    rig_file: Path = DEFAULT_RIG_FILE               # the copy's deployed rig file: every camera constant
    exposure_us: tuple[int, int] = (25640, 33287)   # main, ultrawide (live Pi)
    uw_phase_ms: float = 16.3                       # ultrawide mid-exposure minus main mid-exposure (live: +16.3)
    send_delay_ms: float = 45.8                     # packet leaves this long after main t_sensor_ns (live Pi)
    pi_clock_offset_s: float = 1000.0               # Pi clock = CLOCK_MONOTONIC + offset + drift (as FakePiSender)
    pi_clock_drift_ppm: float = 20.0
    jpeg_quality: int = 90
    max_hold_ms: float = 10.0                       # newest state older than the frame time by more: "stale"
    density: float = 1.0                            # rendered px per native px at the image centre (both cameras)
    keep_model_lights: bool = True                  # the model's lights (made directional); false: render.Lighting
    shadows: bool = True
    send_hwm: int = 10
    stats_s: float = 1.0                            # one status line per this many seconds (0: none)
    save_dir: Path | None = None                    # save the raw 640x480 frames (PNG) of every save_every-th packet
    save_every: int = 30
    stats_json: Path | None = None                  # write the running totals here (at every status line and exit)
    seconds: float = 0.0                            # stop after this long (0: until SIGINT/SIGTERM)
    camera_look: Path | None = None                 # opt-in camera model (look/camera_look.py YAML) applied to each
                                                    # 640x480 frame before JPEG; None: frames exactly as before

    @property
    def fields(self) -> list[str]:
        return [f.name for f in dataclasses.fields(self)]


PATH_KEYS = ("model", "rig_file", "save_dir", "stats_json", "camera_look")


def _coerce(name: str, value: Any, base: Path) -> Any:
    if value is None:
        return None
    if name in PATH_KEYS:
        p = Path(str(value)).expanduser()
        return p if p.is_absolute() else (base / p).resolve()
    if name == "exposure_us":
        a, b = (int(v) for v in value)
        return (a, b)
    return value


def load(station: Path | None, overrides: dict[str, Any]) -> SimPiSettings:
    s = SimPiSettings()
    if station is not None:
        station = Path(station).expanduser().resolve()
        doc = yaml.safe_load(station.read_text()) or {}
        block = doc.get("sim_pi") or {}
        unknown = sorted(set(block) - set(s.fields))
        if unknown:
            raise ValueError(f"{station}: sim_pi.{unknown[0]}: unknown key (known: {', '.join(s.fields)})")
        for k, v in block.items():
            setattr(s, k, _coerce(k, v, station.parent))
        if s.model is None and doc.get("id"):
            s.model = REPO / "sim_deploy" / "build" / str(doc["id"]) / "scene.xml"
    for k, v in overrides.items():
        if v is not None:
            setattr(s, k, _coerce(k, v, Path.cwd()))
    if s.model is None:
        raise ValueError("no model: give --model scene.xml or a --station with an id (sim_deploy/build/<id>/)")
    return s
