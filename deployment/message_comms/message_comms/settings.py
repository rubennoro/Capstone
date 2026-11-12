"""Settings: message_comms/config.yaml merged with station.yaml's `message_comms:` section (STRUCTURE.md 4.1, 8)."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from contracts import ACTION_STEPS

from message_comms.errors import SettingsError

PACKAGE_FOLDER = Path(__file__).resolve().parents[1]      # deployment/message_comms/
CONFIG_FILE = PACKAGE_FOLDER / "config.yaml"
NAME = "message_comms"
ENDPOINT_KEYS = ("frames", "state", "commands", "policy")
POSE_SOURCES = ("target", "measured")
EXEC_H_RANGE = (1, ACTION_STEPS)


@dataclass(frozen=True)
class Endpoints:
    frames: str          # station endpoints.frames    (C1)
    state: str           # station endpoints.state     (C2)
    commands: str        # station endpoints.commands  (C3)
    policy: str          # station endpoints.policy    (C4)


@dataclass(frozen=True)
class Timeouts:
    commands_s: float
    health_s: float
    frames_s: float
    policy_hang_s: float
    config_s: float
    reset_s: float
    prompt_s: float
    home_s: float
    startup_s: float


@dataclass(frozen=True)
class Settings:
    exec_h: int
    lookahead_ms: int
    pose_source: str
    odd_chunk_max_pos_m: float
    odd_chunk_max_rot_deg: float
    anchor_fallback_m: float
    loop_ms: int
    status_ms: int
    health_poll_ms: int
    history_s: float
    max_state_age_ms: int
    state_lost_s: float
    max_image_age_ms: int
    state_recv_hwm: int
    timeouts: Timeouts
    task_name: str | None
    prompt_index: int
    operator_script: Path | None
    skip_warn_s: float
    log_steps: bool
    save_tap: bool
    tap_images_every: int
    endpoints: Endpoints
    session_dir: Path    # <session>/message_comms/
    mock: bool
    station_file: Path

    def to_dict(self) -> dict[str, Any]:
        """Plain values (for session.yaml)."""
        out: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (Timeouts, Endpoints)):
                out[key] = dict(value.__dict__)
            elif isinstance(value, Path):
                out[key] = str(value)
            else:
                out[key] = value
        return out


# key -> (kind, rule). kinds: int, float, bool, str, path. rules: "pos" (> 0), "nonneg" (>= 0), "exec_h", "ge1",
# "pose_source", "optional" (may be null).
_RULES: dict[str, tuple[str, str]] = {
    "exec_h": ("int", "exec_h"),
    "lookahead_ms": ("int", "nonneg"),
    "pose_source": ("str", "pose_source"),
    "odd_chunk_max_pos_m": ("float", "pos"),
    "odd_chunk_max_rot_deg": ("float", "pos"),
    "anchor_fallback_m": ("float", "pos"),
    "loop_ms": ("int", "pos"),
    "status_ms": ("int", "pos"),
    "health_poll_ms": ("int", "pos"),
    "history_s": ("float", "pos"),
    "max_state_age_ms": ("int", "pos"),
    "state_lost_s": ("float", "pos"),
    "max_image_age_ms": ("int", "pos"),
    "state_recv_hwm": ("int", "pos"),
    "task_name": ("str", "optional"),
    "prompt_index": ("int", "nonneg"),
    "operator_script": ("path", "optional"),
    "skip_warn_s": ("float", "pos"),
    "log_steps": ("bool", ""),
    "save_tap": ("bool", ""),
    "tap_images_every": ("int", "ge1"),
}
_TIMEOUT_KEYS = tuple(Timeouts.__dataclass_fields__)


def merge(defaults: dict[str, Any], overrides: Any, path: str, problems: list[str]) -> dict[str, Any]:
    """deploy/STRUCTURE.md 6.3: maps merge key by key; other values are replaced; an unknown override key is an
    error."""
    out = copy.deepcopy(defaults)
    if not isinstance(overrides, dict):
        problems.append(f"station.yaml: {path}: expected a map, got {type(overrides).__name__}")
        return out
    for key, value in overrides.items():
        key_path = f"{path}.{key}"
        if key not in defaults:
            problems.append(f"station.yaml: {key_path}: not a key of message_comms/config.yaml")
        elif isinstance(defaults[key], dict):
            out[key] = merge(defaults[key], value, key_path, problems)
        else:
            out[key] = value
    return out


def _check(key: str, value: Any, kind: str, rule: str, problems: list[str]) -> None:
    where = f"message_comms.{key}"
    if value is None:
        if rule != "optional":
            problems.append(f"{where}: must be set")
        return
    if kind == "int":
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif kind == "float":
        ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    elif kind == "bool":
        ok = isinstance(value, bool)
    else:                                   # str, path
        ok = isinstance(value, str) and value != ""
    if not ok:
        problems.append(f"{where}: expected {kind}, got {value!r}")
        return
    if rule == "pos" and not value > 0:
        problems.append(f"{where}: must be > 0, got {value!r}")
    elif rule == "nonneg" and not value >= 0:
        problems.append(f"{where}: must be >= 0, got {value!r}")
    elif rule == "ge1" and not value >= 1:
        problems.append(f"{where}: must be >= 1, got {value!r}")
    elif rule == "exec_h" and not EXEC_H_RANGE[0] <= value <= EXEC_H_RANGE[1]:
        problems.append(f"{where}: must be between {EXEC_H_RANGE[0]} and {EXEC_H_RANGE[1]}, got {value!r}")
    elif rule == "pose_source" and value not in POSE_SOURCES:
        problems.append(f"{where}: must be one of {', '.join(POSE_SOURCES)}, got {value!r}")


def _resolve(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _read_yaml(path: Path, label: str, problems: list[str]) -> Any:
    try:
        return yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        problems.append(f"{label}: cannot read {path}: {exc}")
        return None


def load_settings(station_file: Path, session: Path | None, mock: bool) -> Settings:
    """Raises SettingsError listing every problem."""
    problems: list[str] = []
    station_file = Path(station_file).expanduser().resolve()
    defaults = _read_yaml(CONFIG_FILE, "message_comms/config.yaml", problems)
    station = _read_yaml(station_file, "station.yaml", problems)
    if problems:
        raise SettingsError(problems)
    if not isinstance(defaults, dict) or not isinstance(station, dict):
        raise SettingsError(["station.yaml and message_comms/config.yaml must be maps"])

    section = station.get(NAME) or {}
    merged = merge(defaults, section, NAME, problems)

    for key, (kind, rule) in _RULES.items():
        _check(key, merged.get(key), kind, rule, problems)
    timeouts = merged.get("timeouts")
    if not isinstance(timeouts, dict):
        problems.append("message_comms.timeouts: expected a map")
        timeouts = {}
    for key in _TIMEOUT_KEYS:
        _check(f"timeouts.{key}", timeouts.get(key), "float", "pos", problems)

    endpoints = station.get("endpoints")
    if not isinstance(endpoints, dict):
        problems.append("station.yaml: endpoints: required map")
        endpoints = {}
    for key in ENDPOINT_KEYS:
        value = endpoints.get(key)
        if not isinstance(value, str) or not value.startswith("tcp://"):
            problems.append(f"station.yaml: endpoints.{key}: expected \"tcp://host:port\", got {value!r}")
    session_root = station.get("session_root")
    if not isinstance(session_root, str) or not session_root:
        problems.append("station.yaml: session_root: required")
    if problems:
        raise SettingsError(problems)

    # a station-set path resolves against the station file's folder, a config.yaml default against this folder
    script = merged["operator_script"]
    if script is not None:
        base = station_file.parent if "operator_script" in section else PACKAGE_FOLDER
        script = _resolve(script, base)

    if session is None:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        session = _resolve(session_root, station_file.parent) / f"{stamp}_{NAME}_alone"
    session_dir = Path(session).expanduser().resolve() / NAME
    try:
        session_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SettingsError([f"session folder {session_dir}: cannot create: {exc}"]) from exc

    values = {key: merged[key] for key in _RULES}
    for key in ("odd_chunk_max_pos_m", "odd_chunk_max_rot_deg", "anchor_fallback_m", "history_s",
                "state_lost_s", "skip_warn_s"):
        values[key] = float(values[key])
    values["operator_script"] = script
    return Settings(
        **values,
        timeouts=Timeouts(**{key: float(timeouts[key]) for key in _TIMEOUT_KEYS}),
        endpoints=Endpoints(**{key: endpoints[key] for key in ENDPOINT_KEYS}),
        session_dir=session_dir,
        mock=bool(mock),
        station_file=station_file,
    )
