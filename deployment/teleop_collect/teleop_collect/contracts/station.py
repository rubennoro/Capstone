# adapted from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/contracts/station.py @ 2ffe641
# changes: the teleop_collect station of STRUCTURE.md 3.2: PROCESS_SECTIONS frames, teleop, ui, fake_pc, launcher; no
# python section (one environment) and no policy-kind check; data_root required (a Path attribute, `~` expanded);
# PORT_KEYS mock_pi, r1, preview, u1, http (distinct); endpoints r1_endpoint, preview_endpoint, u1_endpoint,
# http_url and http_port instead of frames_endpoint and policy_endpoint; imports; mock_data_root required (wave-3
# review): data_root is it when loaded with mock=True, so --mock episodes never mix with the real ones, and the two
# are refused when equal. + the remembered storage folder (2026-11-22): storage.local.yaml next to the station file
# overrides data_root / mock_data_root (storage_file, read_storage, remember_data_root), and load(data_root=...) sets
# this run's. The source's own header follows.
# adapted from rnoro/deployment:deployment/deploy/deploy/station.py @ 91b178b
# changes: the laptop station schema of STRUCTURE.md 2.4 (python light/torch, pi, wifi_ssid, ports, one section per
# laptop process); refuses the contract key names of deployment/contracts/*.yaml by an explicit list; endpoints are
# built from ports on 127.0.0.1 (and the Pi's address); the process sections are checked against config/*.yaml
# (config_merge) so every problem is in one StationError.
"""Station: loads and checks station.yaml, the laptop station file (STRUCTURE.md 3.2).

One file for real and --mock runs. It may not override any key of the three contract yaml files (the PC's address,
ports and timing come only from deployment/contracts/link.yaml); those names are refused wherever they appear.

The storage folder chosen in the page (or with `teleop_collect run --data-root`) is remembered in storage.local.yaml
next to the station file (gitignored): its `data_root` / `mock_data_root` replace station.yaml's. Delete the file to go
back to station.yaml's folders.
"""
from __future__ import annotations

import ipaddress
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from teleop_collect.contracts import paths

STATION_VERSION = 1
PROCESS_SECTIONS = ("frames", "teleop", "ui", "fake_pc", "launcher")
TOP_LEVEL_KEYS = ("station_version", "session_root", "data_root", "mock_data_root", "pi", "wifi_ssid", "ports",
                  *PROCESS_SECTIONS)
PI_KEYS = ("address", "port")
PORT_KEYS = ("mock_pi", "r1", "preview", "u1", "http")
LOCAL_HOST = "127.0.0.1"
DEFAULT_PI_PORT = 5555
#: The remembered storage folders, next to the station file (gitignored); keys STORAGE_KEYS, each optional.
STORAGE_FILE = "storage.local.yaml"
STORAGE_KEYS = ("data_root", "mock_data_root")

_LINK = "deployment/contracts/link.yaml"
_FRAMES = "deployment/contracts/frames.yaml"
_LIMITS = "deployment/contracts/limits.yaml"
#: Contract key names refused at any level of station.yaml and of every config/*.yaml (PLAN.md rule 5).
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

    def __init__(self, path: Path, data: Mapping[str, Any], mock: bool,
                 storage: Mapping[str, str] | None = None) -> None:
        """storage: the remembered folders (read_storage); each key present replaces station.yaml's."""
        self.path = path
        self.mock = mock
        self._data = data
        self.storage = dict(storage or {})
        roots = {**{k: data[k] for k in STORAGE_KEYS}, **self.storage}
        self.session_root = self.resolve(data["session_root"])
        self.real_data_root = self.resolve(roots["data_root"])
        self.mock_data_root = self.resolve(roots["mock_data_root"])
        # where this run's episodes go: --mock episodes are kept apart from the real ones (PLAN 6.1)
        self.data_root = self.mock_data_root if mock else self.real_data_root
        key = STORAGE_KEYS[1] if mock else STORAGE_KEYS[0]
        #: where this run's data_root comes from: "station.yaml", "storage.local.yaml" or "--data-root" (load)
        self.data_root_source = STORAGE_FILE if key in self.storage else path.name
        pi = data.get("pi") or {}
        self.pi_address: str | None = pi.get("address")
        self.pi_port: int = int(pi.get("port", DEFAULT_PI_PORT))
        self.wifi_ssid: str | None = data.get("wifi_ssid")
        self.ports: dict[str, int] = {k: int(v) for k, v in (data.get("ports") or {}).items()}

    @classmethod
    def load(cls, path: Path | str | None = None, mock: bool = False, data_root: Path | str | None = None) -> Station:
        """Read and check the station file (default: deployment/teleop_collect/station.yaml) and the remembered
        storage folders next to it; every problem, with its key path, is collected into one StationError.
        data_root: this run's storage folder instead (data_root, or mock_data_root under --mock); not remembered."""
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
        storage = read_storage(path)
        if data_root is not None:
            storage[STORAGE_KEYS[1] if mock else STORAGE_KEYS[0]] = str(data_root)
        station = cls(path, data, mock, storage)
        if data_root is not None:
            station.data_root_source = "--data-root"
        where = f" (the storage folder: {STORAGE_FILE} or --data-root)" if storage else ""
        if same_folder(station.real_data_root, station.mock_data_root):
            raise StationError(f"{name}: mock_data_root: must differ from data_root ({station.real_data_root}): "
                               f"--mock episodes are kept apart from the real ones{where}")
        if roots_overlap(station.real_data_root, station.mock_data_root):
            raise StationError(f"{name}: mock_data_root: must not be inside data_root or contain it "
                               f"({station.real_data_root}): --mock episodes are kept apart from the real ones{where}")
        # the process sections against config/*.yaml
        from teleop_collect.contracts.config_merge import load_config
        for section in PROCESS_SECTIONS:
            try:
                load_config(section, station)
            except StationError as e:
                problems.extend(e.problems)
        if problems:
            raise StationError(problems)
        return station

    # ---------------------------------------------------------------------------------------------------- values

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
        """--mock: the fake Pi inside frames (PUB bound here)."""
        return f"tcp://{LOCAL_HOST}:{self.ports['mock_pi']}"

    def r1_endpoint(self) -> str:
        """R1, teleop -> frames (frames' REP)."""
        return f"tcp://{LOCAL_HOST}:{self.ports['r1']}"

    def preview_endpoint(self) -> str:
        """U2, ui -> frames (frames' REP)."""
        return f"tcp://{LOCAL_HOST}:{self.ports['preview']}"

    def u1_endpoint(self) -> str:
        """U1, ui -> teleop (teleop's REP)."""
        return f"tcp://{LOCAL_HOST}:{self.ports['u1']}"

    @property
    def http_port(self) -> int:
        """The ui's page port (bound on 127.0.0.1 only)."""
        return self.ports["http"]

    def http_url(self) -> str:
        """The page: http://127.0.0.1:<ports.http>."""
        return f"http://{LOCAL_HOST}:{self.http_port}"


def same_folder(a: Path, b: Path) -> bool:
    """The same folder, after resolving symlinks and `..` (neither needs to exist)."""
    return Path(a).resolve(strict=False) == Path(b).resolve(strict=False)


def roots_overlap(a: Path, b: Path) -> bool:
    """The same folder, or one inside the other (resolved as same_folder)."""
    ra, rb = Path(a).resolve(strict=False), Path(b).resolve(strict=False)
    return ra.is_relative_to(rb) or rb.is_relative_to(ra)


# ------------------------------------------------------------------------------------- the remembered storage folder

def storage_file(station_path: Path | str) -> Path:
    """storage.local.yaml, next to the station file."""
    return Path(station_path).expanduser().absolute().with_name(STORAGE_FILE)


def read_storage(station_path: Path | str) -> dict[str, str]:
    """The remembered folders ({} when there is no file). StationError naming the file when it is malformed."""
    file = storage_file(station_path)
    fix = "fix it or delete it (it only remembers the storage folder)"
    try:
        data = yaml.safe_load(file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, yaml.YAMLError) as exc:
        raise StationError(f"{file}: cannot read: {exc}: {fix}") from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise StationError(f"{file}: expected a map of {' / '.join(STORAGE_KEYS)}, got {data!r}: {fix}")
    problems = []
    for key, value in data.items():
        if key not in STORAGE_KEYS:
            problems.append(f"{file}: {key}: unknown key (known: {', '.join(STORAGE_KEYS)}): {fix}")
        elif not isinstance(value, str) or not value:
            problems.append(f"{file}: {key}: expected a path, got {value!r}: {fix}")
    if problems:
        raise StationError(problems)
    return dict(data)


def remember_data_root(station_path: Path | str, mock: bool, root: Path | str) -> Path:
    """Remember root as the storage folder of real runs (mock False) or --mock runs, keeping the other key; written
    atomically (a temp file, then renamed). Returns the file. StationError when it is malformed or cannot be written."""
    file = storage_file(station_path)
    data = read_storage(station_path)
    data[STORAGE_KEYS[1] if mock else STORAGE_KEYS[0]] = str(root)
    text = ("# The storage folder chosen in the page or with --data-root (teleop_collect; not in git).\n"
            "# Delete this file to go back to station.yaml's data_root / mock_data_root.\n"
            + yaml.safe_dump({k: data[k] for k in STORAGE_KEYS if k in data}, sort_keys=False))
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(prefix=f".{file.name}.", suffix=".tmp", dir=file.parent)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, file)
    except OSError as exc:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        raise StationError(f"{file}: cannot write: {exc}") from None
    return file


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

    for key in ("session_root", "data_root", "mock_data_root"):
        root = data.get(key)
        if root is None:
            problems.append(f"{key}: required")
        elif not isinstance(root, str) or not root:
            problems.append(f"{key}: expected a path, got {root!r}")

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
