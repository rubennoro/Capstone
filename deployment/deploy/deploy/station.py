"""Station: loads and checks station.yaml (STRUCTURE.md sections 3.1 and 6.1)."""
from __future__ import annotations

import math
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

import deploy
from deploy.errors import StationError

PROCESS_NAMES = ("franka_control", "process_frames", "policy", "message_comms")
ENDPOINTS = ("pi_frames", "pi_time_sync", "mock_pi_frames", "mock_pi_time_sync", "frames", "state", "commands",
             "policy")
REAL_ONLY_ENDPOINTS = ("pi_frames", "pi_time_sync")       # not needed with --mock
LAUNCHER_DEFAULTS = {
    "stop_timeout_s": 5.0,
    "poll_s": 0.1,
    "tcp_check_max_mm": 2.0,
    "tcp_check_max_deg": 0.5,
}
READY_TIMEOUT_DEFAULTS = {"franka_control": 30.0, "process_frames": 30.0, "policy": 300.0, "message_comms": 30.0}
TOP_LEVEL_KEYS = ("station_version", "session_root", "python", "endpoints", "launcher", *PROCESS_NAMES)
PYTHON_KEYS = ("light", "policy")
STATION_VERSION = 1

_ENDPOINT_RE = re.compile(r"^tcp://([^:/\s]+):(\d+)$")


def parse_endpoint(endpoint: str) -> tuple[str, int]:
    """`tcp://host:port` -> (host, port). Raises ValueError for anything else."""
    match = _ENDPOINT_RE.match(endpoint) if isinstance(endpoint, str) else None
    if match is None or not 0 < int(match.group(2)) < 65536:
        raise ValueError(f'expected "tcp://host:port", got {endpoint!r}')
    return match.group(1), int(match.group(2))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class Station:
    """The checked shared settings of station.yaml, plus each process's raw section of overrides."""

    def __init__(self, path: Path, data: Mapping[str, Any], mock: bool) -> None:
        # Use Station.load: this constructor trusts that `data` was checked.
        self.path = path
        self.mock = mock
        self._data = data
        self.session_root = self.resolve(data["session_root"])
        launcher = data.get("launcher") or {}
        self.ready_timeout_s: Mapping[str, float] = {
            **READY_TIMEOUT_DEFAULTS, **{k: float(v) for k, v in (launcher.get("ready_timeout_s") or {}).items()}}
        values = {**LAUNCHER_DEFAULTS, **{k: v for k, v in launcher.items() if k != "ready_timeout_s"}}
        self.stop_timeout_s = float(values["stop_timeout_s"])
        self.poll_s = float(values["poll_s"])
        self.tcp_check_max_mm = float(values["tcp_check_max_mm"])
        self.tcp_check_max_deg = float(values["tcp_check_max_deg"])

    @classmethod
    def load(cls, path: Path | None, mock: bool) -> "Station":
        """Read and check the station file; every problem is collected into one StationError."""
        path = Path(deploy.DEPLOYMENT_ROOT / "station.yaml" if path is None else path).expanduser().absolute()
        name = path.name
        try:
            data = yaml.safe_load(path.read_text())
        except FileNotFoundError:
            raise StationError(f"{name}: not found: {path}") from None
        except (OSError, yaml.YAMLError) as exc:
            raise StationError(f"{name}: cannot read {path}: {exc}") from None
        if not isinstance(data, dict):
            raise StationError(f"{name}: expected a map of settings at the top level")
        problems = _check(data, mock)
        if problems:
            raise StationError([f"{name}: {p}" for p in problems])
        return cls(path, data, mock)

    def python(self, env: str) -> str:
        """The interpreter for "light" (default: the Python running deploy) or "policy"."""
        if env not in PYTHON_KEYS:
            raise ValueError(f"python environment must be one of {PYTHON_KEYS}, got {env!r}")
        value = (self._data.get("python") or {}).get(env)
        if value is None:
            if env == "light":
                return sys.executable
            raise StationError(f"{self.path.name}: python.policy is not set (needed without --mock)")
        return str(self.resolve(value))

    def endpoint(self, name: str) -> str:
        """e.g. "commands" -> "tcp://127.0.0.1:5630"."""
        if name not in ENDPOINTS:
            raise ValueError(f"unknown endpoint {name!r}; known: {', '.join(ENDPOINTS)}")
        value = self._data["endpoints"].get(name)
        if value is None:
            raise StationError(f"{self.path.name}: endpoints.{name} is not set (needed without --mock)")
        return value

    def section(self, process: str) -> Mapping[str, Any]:
        """That process's overrides of its config.yaml ({} if absent)."""
        if process not in PROCESS_NAMES:
            raise ValueError(f"unknown process {process!r}; known: {', '.join(PROCESS_NAMES)}")
        return self._data.get(process) or {}

    def resolve(self, value: str | Path) -> Path:
        """`~` expanded; a relative path is taken from the station file's folder."""
        p = Path(value).expanduser()
        return p if p.is_absolute() else self.path.parent / p


def _check(data: Mapping[str, Any], mock: bool) -> list[str]:
    problems: list[str] = []
    for key in data:
        if key not in TOP_LEVEL_KEYS:
            problems.append(f"{key}: unknown key (known: {', '.join(TOP_LEVEL_KEYS)})")

    version = data.get("station_version")
    if version is None:
        problems.append("station_version: required")
    elif isinstance(version, bool) or version != STATION_VERSION:
        problems.append(f"station_version: expected {STATION_VERSION}, got {version!r}")

    root = data.get("session_root")
    if root is None:
        problems.append("session_root: required")
    elif not isinstance(root, str) or not root:
        problems.append(f"session_root: expected a path, got {root!r}")

    python = _map(data, "python", problems)
    for key in python:
        if key not in PYTHON_KEYS:
            problems.append(f"python.{key}: unknown key (known: {', '.join(PYTHON_KEYS)})")
    for key in PYTHON_KEYS:
        value = python.get(key)
        if value is None:
            if key == "policy" and not mock:
                problems.append("python.policy: required (except with --mock)")
        elif not isinstance(value, str) or not value:
            problems.append(f"python.{key}: expected a path, got {value!r}")

    endpoints = _map(data, "endpoints", problems)
    for key in endpoints:
        if key not in ENDPOINTS:
            problems.append(f"endpoints.{key}: unknown key (known: {', '.join(ENDPOINTS)})")
    for key in ENDPOINTS:
        value = endpoints.get(key)
        if value is None:
            if not (mock and key in REAL_ONLY_ENDPOINTS):
                problems.append(f"endpoints.{key}: required" + (" (except with --mock)"
                                                                 if key in REAL_ONLY_ENDPOINTS else ""))
            continue
        try:
            parse_endpoint(value)
        except ValueError as exc:
            problems.append(f"endpoints.{key}: {exc}")

    launcher = _map(data, "launcher", problems)
    for key, value in launcher.items():
        if key == "ready_timeout_s":
            timeouts = value if isinstance(value, dict) else None
            if timeouts is None and value is not None:
                problems.append(f"launcher.ready_timeout_s: expected a map of process name to seconds, got {value!r}")
            for name, seconds in (timeouts or {}).items():
                if name not in PROCESS_NAMES:
                    problems.append(f"launcher.ready_timeout_s.{name}: unknown process "
                                    f"(known: {', '.join(PROCESS_NAMES)})")
                elif not _is_number(seconds) or seconds <= 0:
                    problems.append(f"launcher.ready_timeout_s.{name}: expected a number > 0, got {seconds!r}")
        elif key not in LAUNCHER_DEFAULTS:
            problems.append(f"launcher.{key}: unknown key (known: ready_timeout_s, {', '.join(LAUNCHER_DEFAULTS)})")
        elif not _is_number(value) or value <= 0:
            problems.append(f"launcher.{key}: expected a number > 0, got {value!r}")

    for name in PROCESS_NAMES:
        _map(data, name, problems)
    return problems


def _map(data: Mapping[str, Any], key: str, problems: list[str]) -> Mapping[str, Any]:
    """data[key] as a map: absent or empty -> {}; anything else is a problem."""
    value = data.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        problems.append(f"{key}: expected a map, got {value!r}")
        return {}
    return value
