"""Temporary station files for tests: free ports on 127.0.0.1, session_root under the test's tmp_path."""
from __future__ import annotations

import copy
import socket
from pathlib import Path
from typing import Any

import yaml


def free_port() -> int:
    """A TCP port on 127.0.0.1 that was free a moment ago."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def station_dict(tmp_path: Path, **sections: Any) -> dict[str, Any]:
    """A valid station as a dict; keyword arguments replace top-level keys (e.g. manager={...})."""
    ports = set()
    while len(ports) < 3:
        ports.add(free_port())
    frames, policy, mock_pi = sorted(ports)
    d: dict[str, Any] = {
        "station_version": 1,
        "session_root": str(tmp_path / "sessions"),
        "python": {"light": None, "torch": None},
        "pi": {"address": "192.168.50.30", "port": 5555},
        "wifi_ssid": "ASUS_5g",
        "ports": {"frames": frames, "policy": policy, "mock_pi": mock_pi},
        "frames": {}, "policy": {"kind": "hold_still"}, "manager": {}, "fake_pc": {}, "launcher": {},
    }
    import sys
    d["python"] = {"light": sys.executable}
    for key, value in sections.items():
        d[key] = copy.deepcopy(value)
    return d


def write_station(tmp_path: Path, name: str = "station.yaml", **sections: Any) -> Path:
    path = Path(tmp_path) / name
    path.write_text(yaml.safe_dump(station_dict(tmp_path, **sections), sort_keys=False), encoding="utf-8")
    return path
