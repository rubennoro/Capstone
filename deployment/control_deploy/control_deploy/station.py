"""Station: this PC's settings, control_deploy/station.yaml (control_deploy/STRUCTURE.md 6.2). Strict.

The binary reads only session_root and the franka_control section (and refuses endpoints); the wrapper reads
station_version, session_root and console. Addresses and ports are in deployment/contracts/link.yaml, never here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

KEYS = ("station_version", "session_root", "franka_control", "console")
# Keys of the three contract files: no station may override them (deployment/STRUCTURE.md 1.3 rule 4).
CONTRACT_KEYS = {
    "rotation_tol": "limits.yaml", "gripper_width_max_m": "limits.yaml", "max_waypoints": "limits.yaml",
    "F_T_TCP": "frames.yaml", "settled": "frames.yaml", "tcp": "frames.yaml",
    "pc": "link.yaml", "laptop": "link.yaml", "ports": "link.yaml", "timing": "link.yaml", "clock": "link.yaml",
    "watchdog_limit_s": "link.yaml",
}


@dataclass(frozen=True)
class ConsoleSettings:
    """The bring-up console's step sizes (6.5). Floats > 0; the moves stay small."""

    hold_s: float = 3.0
    move_m: float = 0.015
    rotate_deg: float = 2.0
    gripper_delta_m: float = 0.02
    segment_s: float = 2.0
    out_of_box_m: float = 0.03

    LIMITS = {"move_m": (None, 0.03), "rotate_deg": (None, 10.0), "out_of_box_m": (0.02, 0.05)}


@dataclass(frozen=True)
class Station:
    path: Path
    session_root: Path
    franka_control: dict[str, Any] = field(default_factory=dict)
    console: ConsoleSettings = field(default_factory=ConsoleSettings)

    @classmethod
    def load(cls, path: Path | str) -> Station:
        """Raises ValueError naming the file and the key for anything wrong."""
        path = Path(path).expanduser().resolve()
        name = path.name

        def err(key: str, text: str) -> ValueError:
            return ValueError(f"{name}: {key}: {text}")

        try:
            with open(path, encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        except OSError as exc:
            raise ValueError(f"{name}: cannot read {path} ({exc.strerror or exc})") from None
        except yaml.YAMLError as exc:
            raise ValueError(f"{name}: not valid YAML ({exc})") from None
        if not isinstance(raw, dict):
            raise ValueError(f"{name}: expected a map with the keys {', '.join(KEYS)}")
        for key in raw:
            if key == "endpoints":
                raise err("endpoints", "no longer used: franka_control binds contracts/link.yaml pc.address and ports")
            if key in CONTRACT_KEYS:
                raise err(key, f"is a contracts/{CONTRACT_KEYS[key]} key: change it there (no station can override "
                               f"the contract files)")
            if key not in KEYS:
                raise err(key, f"unknown key (expected {', '.join(KEYS)})")
        version = raw.get("station_version")
        if version != 1 or isinstance(version, bool):
            raise err("station_version", f"must be 1, got {version!r}")
        root = raw.get("session_root")
        if not isinstance(root, str) or not root:
            raise err("session_root", f"expected a folder, got {root!r}")
        session_root = Path(root).expanduser()
        if not session_root.is_absolute():
            session_root = path.parent / session_root
        overrides = raw.get("franka_control") or {}
        if not isinstance(overrides, dict):
            raise err("franka_control", f"expected a map of config.yaml overrides, got {overrides!r}")
        if "libfranka_version" in overrides:
            raise err("franka_control.libfranka_version", "set it in franka_control/config.yaml only (build.sh "
                                                          "reads it there)")
        for key, home in (("tcp", "contracts/frames.yaml"), ("F_T_TCP", "contracts/frames.yaml")):
            if key in overrides:
                raise err(f"franka_control.{key}", f"moved to {home}")
        if isinstance(overrides.get("watchdog"), dict) and "limit_s" in overrides["watchdog"]:
            raise err("franka_control.watchdog.limit_s", "moved to contracts/link.yaml timing.watchdog_limit_s")
        console = _console(raw.get("console"), err)
        return cls(path=path, session_root=Path(session_root).resolve(), franka_control=overrides, console=console)


def _console(raw: Any, err) -> ConsoleSettings:
    if raw is None:
        return ConsoleSettings()
    if not isinstance(raw, dict):
        raise err("console", f"expected a map, got {raw!r}")
    names = [f for f in ConsoleSettings.__dataclass_fields__]
    values: dict[str, float] = {}
    for key, value in raw.items():
        if key not in names:
            raise err(f"console.{key}", f"unknown key (expected {', '.join(names)})")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise err(f"console.{key}", f"must be a number > 0, got {value!r}")
        low, high = ConsoleSettings.LIMITS.get(key, (None, None))
        if (low is not None and value < low) or (high is not None and value > high):
            span = f"in [{low}, {high}]" if low is not None else f"<= {high}"
            raise err(f"console.{key}", f"must be {span}, got {value!r}")
        values[key] = float(value)
    return ConsoleSettings(**values)
