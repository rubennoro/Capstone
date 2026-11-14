# adapted from rnoro/deployment:deployment/deploy/deploy/station.py @ 91b178b
# changes: the laptop station schema of STRUCTURE.md 2.4 (python light/torch, pi, wifi_ssid, ports, one section per
# laptop process); refuses the contract key names of deployment/contracts/*.yaml by an explicit list; endpoints are
# built from ports on 127.0.0.1 (and the Pi's address); the process sections are checked against config/*.yaml
# (config_merge) so every problem is in one StationError.
"""Station: loads and checks station.yaml, the laptop station file (STRUCTURE.md 2.4).

One file for real and --mock runs. It may not override any key of the three contract yaml files (the PC's address,
ports and timing come only from deployment/contracts/link.yaml); those names are refused wherever they appear.
"""
from __future__ import annotations

import ipaddress
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from policy_deploy.contracts import paths

STATION_VERSION = 1
PROCESS_SECTIONS = ("frames", "policy", "manager", "fake_pc", "launcher")
TOP_LEVEL_KEYS = ("station_version", "session_root", "python", "pi", "wifi_ssid", "ports", *PROCESS_SECTIONS)
PYTHON_KEYS = ("light", "torch")
PI_KEYS = ("address", "port")
PORT_KEYS = ("frames", "policy", "mock_pi")
LOCAL_HOST = "127.0.0.1"
DEFAULT_PI_PORT = 5555

_LINK = "deployment/contracts/link.yaml"
_FRAMES = "deployment/contracts/frames.yaml"
_LIMITS = "deployment/contracts/limits.yaml"
#: Contract key names refused at any level of station.yaml and of every config/*.yaml (TOP 1.3 rule 4).
REFUSED_ANYWHERE: dict[str, str] = {
    "watchdog_limit_s": _LINK, "heartbeat_period_s": _LINK, "c3_timeout_s": _LINK, "state_stale_s": _LINK,
    "state_lost_s": _LINK, "estimator": _LINK,
    "F_T_TCP": _FRAMES, "settled": _FRAMES,
    "rotation_tol": _LIMITS, "gripper_width_max_m": _LIMITS, "max_waypoints": _LIMITS,
}
#: Contract section names refused at the top level of station.yaml.
REFUSED_TOP: dict[str, str] = {"pc": _LINK, "laptop": _LINK, "timing": _LINK, "clock": _LINK}


class StationError(Exception):
    """Every problem of a station or config file, each one line with its key path."""

    def __init__(self, problems: str | list[str]) -> None:
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        super().__init__("; ".join(self.problems))


def refused_keys(data: Any, prefix: str = "") -> list[tuple[str, str]]:
    """(key path, contract file) for every REFUSED_ANYWHERE name at any level of `data`."""
    out: list[tuple[str, str]] = []
    if isinstance(data, Mapping):
        for key, value in data.items():
            path = f"{prefix}{key}"
            if key in REFUSED_ANYWHERE:
                out.append((path, REFUSED_ANYWHERE[key]))
            out.extend(refused_keys(value, path + "."))
    elif isinstance(data, list):
        for i, value in enumerate(data):
            out.extend(refused_keys(value, f"{prefix.rstrip('.')}[{i}]."))
    return out


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


class Station:
    """The checked settings of station.yaml, plus each process's raw section of overrides. Use Station.load."""

    def __init__(self, path: Path, data: Mapping[str, Any], mock: bool) -> None:
        self.path = path
        self.mock = mock
        self._data = data
        self.session_root = self.resolve(data["session_root"])
        pi = data.get("pi") or {}
        self.pi_address: str | None = pi.get("address")
        self.pi_port: int = int(pi.get("port", DEFAULT_PI_PORT))
        self.wifi_ssid: str | None = data.get("wifi_ssid")
        self.ports: dict[str, int] = {k: int(v) for k, v in (data.get("ports") or {}).items()}

    @classmethod
    def load(cls, path: Path | str | None = None, mock: bool = False) -> Station:
        """Read and check the station file (default: deployment/policy_deploy/station.yaml); every problem, with its
        key path, is collected into one StationError."""
        path = Path(paths.default_station() if path is None else path).expanduser().absolute()
        name = path.name
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise StationError(f"{name}: not found: {path}") from None
        except (OSError, yaml.YAMLError) as exc:
            raise StationError(f"{name}: cannot read {path}: {exc}") from None
        if not isinstance(data, dict):
            raise StationError(f"{name}: expected a map of settings at the top level")
        problems = [f"{name}: {p}" for p in _check(data, mock)]
        if problems:
            raise StationError(problems)
        station = cls(path, data, mock)
        # the process sections against config/*.yaml, and python.torch for the kind that runs
        from policy_deploy.contracts.config_merge import load_config
        for section in PROCESS_SECTIONS:
            try:
                merged = load_config(section, station)
            except StationError as e:
                problems.extend(e.problems)
                continue
            if section == "policy":
                kind = merged.get("mock_kind") if mock else merged.get("kind")
                if kind == "diffusion" and (data.get("python") or {}).get("torch") is None:
                    problems.append(f"{name}: python.torch: required for policy kind diffusion"
                                    + (" (policy.mock_kind under --mock)" if mock else ""))
        if problems:
            raise StationError(problems)
        return station

    # ---------------------------------------------------------------------------------------------------- values

    def python(self, env: str) -> str:
        """The interpreter for "light" (default: the Python running this) or "torch" (required for diffusion)."""
        if env not in PYTHON_KEYS:
            raise ValueError(f"python environment must be one of {PYTHON_KEYS}, got {env!r}")
        value = (self._data.get("python") or {}).get(env)
        if value is None:
            if env == "light":
                return sys.executable
            raise StationError(f"{self.path.name}: python.torch is not set (needed for policy kind diffusion)")
        return str(self.resolve(value))

    def section(self, name: str) -> Mapping[str, Any]:
        """That process's (or the launcher's) overrides of its config file ({} if absent)."""
        if name not in PROCESS_SECTIONS:
            raise ValueError(f"unknown section {name!r}; known: {', '.join(PROCESS_SECTIONS)}")
        return self._data.get(name) or {}

    def resolve(self, value: str | Path) -> Path:
        """`~` expanded; a relative path is taken from the station file's folder."""
        p = Path(value).expanduser()
        return p if p.is_absolute() else self.path.parent / p

    def raw(self) -> Mapping[str, Any]:
        return self._data

    # ------------------------------------------------------------------------------------------------- endpoints

    def pi_endpoint(self) -> str:
        """Where frames connects for C0: tcp://<pi.address>:<pi.port>, or the fake Pi under --mock."""
        if self.mock:
            return self.mock_pi_endpoint()
        if not self.pi_address:
            raise StationError(f"{self.path.name}: pi.address is not set (needed without --mock)")
        return f"tcp://{self.pi_address}:{self.pi_port}"

    def mock_pi_endpoint(self) -> str:
        return f"tcp://{LOCAL_HOST}:{self.ports['mock_pi']}"

    def frames_endpoint(self) -> str:
        return f"tcp://{LOCAL_HOST}:{self.ports['frames']}"

    def policy_endpoint(self) -> str:
        return f"tcp://{LOCAL_HOST}:{self.ports['policy']}"


def _check(data: Mapping[str, Any], mock: bool) -> list[str]:
    problems: list[str] = []
    for key in data:
        if key in REFUSED_TOP:
            problems.append(f"{key}: belongs to {REFUSED_TOP[key]}")
        elif key in REFUSED_ANYWHERE:
            pass                                          # reported by refused_keys below
        elif key not in TOP_LEVEL_KEYS:
            problems.append(f"{key}: unknown key (known: {', '.join(TOP_LEVEL_KEYS)})")
    for path, file in refused_keys(data):
        problems.append(f"{path}: belongs to {file}")

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
    for key, value in python.items():
        if key not in PYTHON_KEYS:
            problems.append(f"python.{key}: unknown key (known: {', '.join(PYTHON_KEYS)})")
        elif value is not None and (not isinstance(value, str) or not value):
            problems.append(f"python.{key}: expected a path, got {value!r}")

    pi = _map(data, "pi", problems)
    for key in pi:
        if key not in PI_KEYS:
            problems.append(f"pi.{key}: unknown key (known: {', '.join(PI_KEYS)})")
    address = pi.get("address")
    if address is None:
        if not mock:
            problems.append("pi.address: required (except with --mock): the Pi's address on ASUS_5g")
    else:
        try:
            ipaddress.IPv4Address(address if isinstance(address, str) else "")
        except ValueError:
            problems.append(f"pi.address: expected an IPv4 address, got {address!r}")
    port = pi.get("port", DEFAULT_PI_PORT)
    if not _is_int(port) or not 1 <= port <= 65535:
        problems.append(f"pi.port: expected a port 1-65535, got {port!r}")

    ssid = data.get("wifi_ssid")
    if ssid is not None and (not isinstance(ssid, str) or not ssid):
        problems.append(f"wifi_ssid: expected a network name, got {ssid!r}")

    ports = _map(data, "ports", problems)
    for key in ports:
        if key not in PORT_KEYS:
            problems.append(f"ports.{key}: unknown key (known: {', '.join(PORT_KEYS)})")
    seen: dict[int, str] = {}
    for key in PORT_KEYS:
        value = ports.get(key)
        if value is None:
            problems.append(f"ports.{key}: required")
        elif not _is_int(value) or not 1024 <= value <= 65535:
            problems.append(f"ports.{key}: expected a port 1024-65535, got {value!r}")
        elif value in seen:
            problems.append(f"ports.{key}: {value} is also ports.{seen[value]}")
        else:
            seen[value] = key

    for name in PROCESS_SECTIONS:
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


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
