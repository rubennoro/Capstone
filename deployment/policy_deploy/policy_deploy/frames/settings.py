# adapted from rnoro/deployment:deployment/process_frames/process_frames/settings.py @ 91b178b
# changes: the station schema of STRUCTURE.md 2.4 (Station.load + config_merge.load_config("frames", station): the
# merge rule lives in policy_deploy.contracts now); endpoints from station ports (C1 frames_endpoint, the Pi or the fake
# Pi pi_endpoint, the fake Pi's bind mock_pi_endpoint); drops stream, decoder, clock.interval_s/timeout_s/
# startup_exchanges, h264.*, mock.keyframe_interval and every pi_time_sync key; adds select.pair_max_age_ms,
# select.pair_max_lead_ms, select.pair_max_reuse.
# copied from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/frames/settings.py @ f632576
# changes: Settings.fake_pi from the station's launcher.fake_pi (config/launcher.yaml); under --mock with fake_pi false,
# pi_frames is tcp://<pi.address>:<pi.port> as in a real run (pi.address then required, mock_episode not) (sim_deploy)
"""Settings: config/frames_process.yaml merged with station.yaml's `frames:` section, validated (STRUCTURE.md 4.4).

Merge rule (STRUCTURE.md 2.5): maps merge key by key, other values are replaced, and an override key that is not in
the config file is an error. Relative paths: defaults from deployment/policy_deploy/, station overrides from the
station file's folder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from policy_deploy.contracts import Station, StationError, load_config, resolve_path

from policy_deploy.frames import NAME
from policy_deploy.frames.errors import SettingsError


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
    pair_max_age_ms: float
    pair_max_lead_ms: float
    pair_max_reuse: int


@dataclass(frozen=True)
class HealthSettings:
    stale_s: float
    down_s: float
    fps_window_s: float


@dataclass(frozen=True)
class ClockSettings:
    window_s: float
    best_fraction: float
    drift_min_span_s: float
    max_drift_ppm: float
    max_sync_age_s: float


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


def _path(v: Any) -> str | None:
    return None if isinstance(v, str) and v else f"expected a path, got {v!r}"


def _path_or_null(v: Any) -> str | None:
    return None if v is None else _path(v)


#: Every key of config/frames_process.yaml and its check (STRUCTURE.md 4.4).
SCHEMA: dict[str, Callable[[Any], str | None]] = {
    "rig_file": _path,
    "mock_episode": _path_or_null,
    "default_exposure_ns": _int_min(0),
    "receive.timeout_s": _float_gt0,
    "receive.recv_hwm": _int_min(1),
    "buffer.keep_s": _float_gt0,
    "select.main_spacing_ms": _float_gt0,
    "select.ultrawide_spacing_ms": _float_gt0,
    "select.tolerance_frames": _float_gt0,
    "select.pair_max_age_ms": _float_gt0,
    "select.pair_max_lead_ms": _float_gt0,
    "select.pair_max_reuse": _int_min(1),
    "health.stale_s": _float_gt0,
    "health.down_s": _float_gt0,
    "health.fps_window_s": _float_gt0,
    "clock.window_s": _float_gt0,
    "clock.best_fraction": _fraction,
    "clock.drift_min_span_s": _float_gt0,
    "clock.max_drift_ppm": _float_gt0,
    "clock.max_sync_age_s": _float_gt0,
    "server.poll_s": _float_gt0,
    "server.join_timeout_s": _float_gt0,
    "log.stats_every_s": _float_gt0,
    "mock.max_seconds": _float_gt0,
    "mock.jpeg_quality": _int_range(1, 100),
    "mock.pi_clock_offset_s": _float_any,
    "mock.drift_ppm": _float_any,
    "mock.send_hwm": _int_min(1),
    "check.reachable_timeout_s": _float_gt0,
    "check.subprocess_timeout_s": _float_gt0,
    "probe.timeout_s": _float_gt0,
}

CONFIG_LABEL = "config/frames_process.yaml"


def _flatten(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


@dataclass(frozen=True)
class Settings:
    rig_file: Path
    mock: bool
    fake_pi: bool               # --mock: start the fake Pi (launcher.fake_pi); false: connect to pi.address as a real run
    mock_episode: Path | None
    pi_frames: str              # where frames connects for C0: the Pi, or the fake Pi under --mock with fake_pi
    mock_pi_frames: str         # where the fake Pi binds (--mock)
    frames: str                 # C1, on 127.0.0.1
    pi_address: str | None      # the Pi's address (None under --mock with fake_pi), for the reachability check
    pi_port: int
    session_root: Path
    station_file: Path
    default_exposure_ns: int
    receive: ReceiveSettings
    buffer: BufferSettings
    select: SelectSettings
    health: HealthSettings
    clock: ClockSettings
    server: ServerSettings
    log: LogSettings
    mock_settings: MockSettings
    check: CheckSettings
    probe: ProbeSettings

    @classmethod
    def load(cls, station_path: Path | None, mock: bool, config_path: Path | None = None) -> "Settings":
        """Load the station and config/frames_process.yaml and check everything; raises SettingsError listing every
        problem. ``config_path`` replaces config/frames_process.yaml (tests)."""
        try:
            station = Station.load(station_path, mock=mock)
            merged = load_config(NAME, station, config_path)
        except StationError as e:
            raise SettingsError(e.problems) from None
        return cls.from_config(station, merged)

    @classmethod
    def from_config(cls, station: Station, merged: Mapping[str, Any]) -> "Settings":
        problems: list[str] = []
        try:
            fake_pi = load_config("launcher", station).get("fake_pi")
        except StationError as e:
            raise SettingsError(e.problems) from None
        if not isinstance(fake_pi, bool):
            raise SettingsError([f"launcher.fake_pi: expected true or false, got {fake_pi!r}"])
        flat = _flatten(merged)
        for key in flat:
            if key not in SCHEMA:
                problems.append(f"{CONFIG_LABEL}: {key}: unknown key")
        for key, check in SCHEMA.items():
            if key not in flat:
                problems.append(f"{CONFIG_LABEL}: {key}: missing")
                continue
            msg = check(flat[key])
            if msg:
                problems.append(f"{NAME}.{key}: {msg}")
        if not problems and flat["health.down_s"] <= flat["health.stale_s"]:
            problems.append(f"{NAME}.health.down_s: must be > health.stale_s ({flat['health.stale_s']}), "
                            f"got {flat['health.down_s']}")
        if problems:
            raise SettingsError(problems)
        mock = station.mock
        rig_file = resolve_path(station, NAME, merged, "rig_file")
        mock_episode = resolve_path(station, NAME, merged, "mock_episode")
        if mock and fake_pi:
            if mock_episode is None:
                problems.append(f"{NAME}.mock_episode: required with --mock (a recorded Pi episode folder)")
            elif not mock_episode.is_dir():
                problems.append(f"{NAME}.mock_episode: not found: {mock_episode}")
        real_pi = not mock or not fake_pi                  # frames connects to pi.address:pi.port
        if mock and not fake_pi and not station.pi_address:
            problems.append(f"{station.path.name}: pi.address: required with --mock and launcher.fake_pi false")
        if problems:
            raise SettingsError(problems)
        pi_frames = (f"tcp://{station.pi_address}:{station.pi_port}" if mock and not fake_pi
                     else station.pi_endpoint())
        f = flat
        return cls(
            rig_file=rig_file.absolute(), mock=mock, fake_pi=fake_pi,
            mock_episode=mock_episode.absolute() if mock_episode is not None else None,
            pi_frames=pi_frames, mock_pi_frames=station.mock_pi_endpoint(),
            frames=station.frames_endpoint(), pi_address=station.pi_address if real_pi else None,
            pi_port=station.pi_port, session_root=station.session_root, station_file=station.path,
            default_exposure_ns=f["default_exposure_ns"],
            receive=ReceiveSettings(float(f["receive.timeout_s"]), f["receive.recv_hwm"]),
            buffer=BufferSettings(float(f["buffer.keep_s"])),
            select=SelectSettings(float(f["select.main_spacing_ms"]), float(f["select.ultrawide_spacing_ms"]),
                                  float(f["select.tolerance_frames"]), float(f["select.pair_max_age_ms"]),
                                  float(f["select.pair_max_lead_ms"]), f["select.pair_max_reuse"]),
            health=HealthSettings(float(f["health.stale_s"]), float(f["health.down_s"]),
                                  float(f["health.fps_window_s"])),
            clock=ClockSettings(float(f["clock.window_s"]), float(f["clock.best_fraction"]),
                                float(f["clock.drift_min_span_s"]), float(f["clock.max_drift_ppm"]),
                                float(f["clock.max_sync_age_s"])),
            server=ServerSettings(float(f["server.poll_s"]), float(f["server.join_timeout_s"])),
            log=LogSettings(float(f["log.stats_every_s"])),
            mock_settings=MockSettings(float(f["mock.max_seconds"]), f["mock.jpeg_quality"],
                                       float(f["mock.pi_clock_offset_s"]), float(f["mock.drift_ppm"]),
                                       f["mock.send_hwm"]),
            check=CheckSettings(float(f["check.reachable_timeout_s"]), float(f["check.subprocess_timeout_s"])),
            probe=ProbeSettings(float(f["probe.timeout_s"])),
        )
