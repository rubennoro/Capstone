"""Settings: config.yaml + the station's process_frames section + the station's shared keys (STRUCTURE.md §5.2, §10).

Merge rule (deploy/STRUCTURE.md §6.3): maps merge key by key, other values are replaced, and an override key that is
not in config.yaml is an error. Relative paths: config.yaml defaults from the process_frames/ folder, station
overrides from the station file's folder.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from process_frames import FOLDER, NAME
from process_frames.errors import SettingsError


@dataclass(frozen=True)
class ReceiveSettings:
    timeout_s: float
    recv_hwm: int


@dataclass(frozen=True)
class BufferSettings:
    keep_s: float


@dataclass(frozen=True)
class SelectSettings:
    main_spacing_ms: float
    ultrawide_spacing_ms: float
    tolerance_frames: float


@dataclass(frozen=True)
class HealthSettings:
    stale_s: float
    down_s: float
    fps_window_s: float


@dataclass(frozen=True)
class ClockSettings:
    interval_s: float
    timeout_s: float
    startup_exchanges: int
    window_s: float
    best_fraction: float
    drift_min_span_s: float
    max_drift_ppm: float
    max_sync_age_s: float


@dataclass(frozen=True)
class H264Settings:
    thread_type: str


@dataclass(frozen=True)
class ServerSettings:
    poll_s: float
    join_timeout_s: float


@dataclass(frozen=True)
class LogSettings:
    stats_every_s: float


@dataclass(frozen=True)
class MockSettings:
    max_seconds: float
    keyframe_interval: int
    jpeg_quality: int
    pi_clock_offset_s: float
    drift_ppm: float
    send_hwm: int


@dataclass(frozen=True)
class CheckSettings:
    reachable_timeout_s: float
    subprocess_timeout_s: float


@dataclass(frozen=True)
class ProbeSettings:
    timeout_s: float


# ------------------------------------------------------------------------------------------------ value checks

def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _float_gt0(v: Any) -> str | None:
    return None if _is_number(v) and v > 0 else f"expected a number > 0, got {v!r}"


def _float_any(v: Any) -> str | None:
    return None if _is_number(v) else f"expected a number, got {v!r}"


def _fraction(v: Any) -> str | None:
    return None if _is_number(v) and 0 < v <= 1 else f"expected a number in (0, 1], got {v!r}"


def _int_min(lo: int) -> Callable[[Any], str | None]:
    def check(v: Any) -> str | None:
        ok = isinstance(v, int) and not isinstance(v, bool) and v >= lo
        return None if ok else f"expected an integer >= {lo}, got {v!r}"
    return check


def _int_range(lo: int, hi: int) -> Callable[[Any], str | None]:
    def check(v: Any) -> str | None:
        ok = isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi
        return None if ok else f"expected an integer in [{lo}, {hi}], got {v!r}"
    return check


def _choice(*choices: str) -> Callable[[Any], str | None]:
    def check(v: Any) -> str | None:
        return None if v in choices else f"expected one of {list(choices)}, got {v!r}"
    return check


def _path(v: Any) -> str | None:
    return None if isinstance(v, str) and v else f"expected a path, got {v!r}"


def _path_or_null(v: Any) -> str | None:
    return None if v is None else _path(v)


#: Every key of config.yaml and its check (STRUCTURE.md §10).
SCHEMA: dict[str, Callable[[Any], str | None]] = {
    "rig_file": _path,
    "decoder": _choice("h264", "jpeg"),
    "mock_episode": _path_or_null,
    "default_exposure_ns": _int_min(0),
    "receive.timeout_s": _float_gt0,
    "receive.recv_hwm": _int_min(1),
    "buffer.keep_s": _float_gt0,
    "select.main_spacing_ms": _float_gt0,
    "select.ultrawide_spacing_ms": _float_gt0,
    "select.tolerance_frames": _float_gt0,
    "health.stale_s": _float_gt0,
    "health.down_s": _float_gt0,
    "health.fps_window_s": _float_gt0,
    "clock.interval_s": _float_gt0,
    "clock.timeout_s": _float_gt0,
    "clock.startup_exchanges": _int_min(1),
    "clock.window_s": _float_gt0,
    "clock.best_fraction": _fraction,
    "clock.drift_min_span_s": _float_gt0,
    "clock.max_drift_ppm": _float_gt0,
    "clock.max_sync_age_s": _float_gt0,
    "h264.thread_type": _choice("SLICE", "NONE"),
    "server.poll_s": _float_gt0,
    "server.join_timeout_s": _float_gt0,
    "log.stats_every_s": _float_gt0,
    "mock.max_seconds": _float_gt0,
    "mock.keyframe_interval": _int_min(1),
    "mock.jpeg_quality": _int_range(1, 100),
    "mock.pi_clock_offset_s": _float_any,
    "mock.drift_ppm": _float_any,
    "mock.send_hwm": _int_min(1),
    "check.reachable_timeout_s": _float_gt0,
    "check.subprocess_timeout_s": _float_gt0,
    "probe.timeout_s": _float_gt0,
}

STATION_NAME = "station.yaml"


def _read_yaml(path: Path, what: str) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise SettingsError([f"{what}: cannot read {path} ({e})"]) from None
    except yaml.YAMLError as e:
        raise SettingsError([f"{what}: invalid YAML in {path} ({e})"]) from None


def merge_overrides(base: dict, overrides: Mapping[str, Any], prefix: str, problems: list[str]) -> dict:
    """``base`` with ``overrides`` merged over it (maps key by key, other values replaced). An override key that is
    not in ``base`` is added to ``problems`` with its path, e.g. ``process_frames.clock.intervl_s``."""
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        path = f"{prefix}.{key}"
        if key not in base:
            problems.append(f"{STATION_NAME}: {path}: not a key of {NAME}/config.yaml")
        elif isinstance(base[key], dict) and isinstance(value, dict):
            out[key] = merge_overrides(base[key], value, path, problems)
        elif isinstance(base[key], dict):
            problems.append(f"{STATION_NAME}: {path}: expected a map, got {value!r}")
        else:
            out[key] = value
    return out


def _flatten(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _endpoint(endpoints: Any, key: str, problems: list[str]) -> str:
    value = endpoints.get(key) if isinstance(endpoints, dict) else None
    if not isinstance(value, str) or not value.startswith("tcp://") or ":" not in value[len("tcp://"):]:
        problems.append(f"{STATION_NAME}: endpoints.{key}: expected \"tcp://host:port\", got {value!r}")
        return ""
    return value


@dataclass(frozen=True)
class Settings:
    rig_file: Path
    decoder: str
    mock: bool
    mock_episode: Path | None
    pi_frames: str
    pi_time_sync: str
    frames: str
    session_root: Path
    default_exposure_ns: int
    receive: ReceiveSettings
    buffer: BufferSettings
    select: SelectSettings
    health: HealthSettings
    clock: ClockSettings
    h264: H264Settings
    server: ServerSettings
    log: LogSettings
    mock_settings: MockSettings
    check: CheckSettings
    probe: ProbeSettings

    @classmethod
    def load(cls, station_path: Path, mock: bool, config_path: Path | None = None) -> "Settings":
        """Load and check everything; raises SettingsError listing every problem. ``config_path`` is for tests
        (default: ``FOLDER / "config.yaml"``)."""
        station_path = Path(station_path).expanduser().absolute()
        config_path = Path(config_path) if config_path is not None else FOLDER / "config.yaml"
        problems: list[str] = []
        config = _read_yaml(config_path, f"{NAME}/config.yaml")
        if not isinstance(config, dict):
            raise SettingsError([f"{NAME}/config.yaml: expected a map"])
        station = _read_yaml(station_path, STATION_NAME)
        if not isinstance(station, dict):
            raise SettingsError([f"{STATION_NAME}: expected a map at the top level ({station_path})"])
        section = station.get(NAME) or {}
        if not isinstance(section, dict):
            problems.append(f"{STATION_NAME}: {NAME}: expected a map, got {section!r}")
            section = {}
        merged = merge_overrides(config, section, NAME, problems)

        flat = _flatten(merged)
        for key in flat:
            if key not in SCHEMA:
                problems.append(f"{NAME}/config.yaml: {key}: unknown key")
        for key, check in SCHEMA.items():
            if key not in flat:
                problems.append(f"{NAME}/config.yaml: {key}: missing")
                continue
            msg = check(flat[key])
            if msg:
                problems.append(f"{NAME}.{key}: {msg}")
        if not problems and flat["health.down_s"] <= flat["health.stale_s"]:
            problems.append(f"{NAME}.health.down_s: must be > health.stale_s ({flat['health.stale_s']}), "
                            f"got {flat['health.down_s']}")

        def resolve(key: str) -> Path | None:
            value = flat.get(key)
            if not isinstance(value, str) or not value:
                return None
            p = Path(value).expanduser()
            if p.is_absolute():
                return p
            from_station = key in section
            return (station_path.parent if from_station else config_path.parent) / p

        rig_file = resolve("rig_file")
        mock_episode = resolve("mock_episode")
        if mock:
            if mock_episode is None:
                problems.append(f"{NAME}.mock_episode: required with --mock (a recorded Pi episode folder)")
            elif not mock_episode.is_dir():
                problems.append(f"{NAME}.mock_episode: not found: {mock_episode}")

        endpoints = station.get("endpoints")
        frames = _endpoint(endpoints, "frames", problems)
        prefix = "mock_pi_" if mock else "pi_"
        pi_frames = _endpoint(endpoints, prefix + "frames", problems)
        pi_time_sync = _endpoint(endpoints, prefix + "time_sync", problems)
        root = station.get("session_root")
        if not isinstance(root, str) or not root:
            problems.append(f"{STATION_NAME}: session_root: required, got {root!r}")
            session_root = Path()
        else:
            session_root = Path(root).expanduser()
            if not session_root.is_absolute():
                session_root = station_path.parent / session_root
        if problems:
            raise SettingsError(problems)

        f = flat
        return cls(
            rig_file=rig_file.absolute(), decoder=f["decoder"], mock=mock,
            mock_episode=mock_episode.absolute() if mock_episode is not None else None,
            pi_frames=pi_frames, pi_time_sync=pi_time_sync, frames=frames, session_root=session_root,
            default_exposure_ns=f["default_exposure_ns"],
            receive=ReceiveSettings(float(f["receive.timeout_s"]), f["receive.recv_hwm"]),
            buffer=BufferSettings(float(f["buffer.keep_s"])),
            select=SelectSettings(float(f["select.main_spacing_ms"]), float(f["select.ultrawide_spacing_ms"]),
                                  float(f["select.tolerance_frames"])),
            health=HealthSettings(float(f["health.stale_s"]), float(f["health.down_s"]),
                                  float(f["health.fps_window_s"])),
            clock=ClockSettings(float(f["clock.interval_s"]), float(f["clock.timeout_s"]),
                                f["clock.startup_exchanges"], float(f["clock.window_s"]),
                                float(f["clock.best_fraction"]), float(f["clock.drift_min_span_s"]),
                                float(f["clock.max_drift_ppm"]), float(f["clock.max_sync_age_s"])),
            h264=H264Settings(f["h264.thread_type"]),
            server=ServerSettings(float(f["server.poll_s"]), float(f["server.join_timeout_s"])),
            log=LogSettings(float(f["log.stats_every_s"])),
            mock_settings=MockSettings(float(f["mock.max_seconds"]), f["mock.keyframe_interval"],
                                       f["mock.jpeg_quality"], float(f["mock.pi_clock_offset_s"]),
                                       float(f["mock.drift_ppm"]), f["mock.send_hwm"]),
            check=CheckSettings(float(f["check.reachable_timeout_s"]), float(f["check.subprocess_timeout_s"])),
            probe=ProbeSettings(float(f["probe.timeout_s"])),
        )
