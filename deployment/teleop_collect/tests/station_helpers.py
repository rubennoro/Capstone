# adapted from rnoro/final-deployment:deployment/policy_deploy/tests/station_helpers.py @ ea9f7fb
# changes: the teleop_collect station (data_root, mock_data_root, ports mock_pi/r1/preview/u1/http, sections frames/teleop/ui/fake_pc/
# launcher, no python section); write_station(tmp_path, *, sections, ports, pi_address) with the Pi on 127.0.0.1 by
# default (never the real Pi); + contract_error (from policy_deploy/tests/contract_helpers.py, unchanged); RECORDED_EPISODE
# reads this package's config/frames_process.yaml, override TELEOP_COLLECT_TEST_EPISODE.
"""Temporary station files for tests: distinct free ports on 127.0.0.1, session_root, data_root and mock_data_root under the test's
tmp_path. Plus contract_error, the contract tests' one-line check."""
from __future__ import annotations

import copy
import os
import socket
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml
from deploy_contracts import ContractError

from teleop_collect.contracts import paths

# The recorded Pi episode the real-recording tests play: the --mock default (config/frames_process.yaml
# mock_episode), or TELEOP_COLLECT_TEST_EPISODE. Tests that need it skip when it is not on this machine.
RECORDED_EPISODE = Path(os.environ.get("TELEOP_COLLECT_TEST_EPISODE") or yaml.safe_load(
    (paths.folder() / "config" / "frames_process.yaml").read_text())["mock_episode"]).expanduser()

PORT_KEYS = ("mock_pi", "r1", "preview", "u1", "http")
SECTIONS = ("frames", "teleop", "ui", "fake_pc", "launcher")


def free_port() -> int:
    """A TCP port on 127.0.0.1 that was free a moment ago."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def free_ports(n: int) -> list[int]:
    """n distinct free ports."""
    ports: set[int] = set()
    while len(ports) < n:
        ports.add(free_port())
    return sorted(ports)


def station_dict(tmp_path: Path, *, sections: dict | None = None, ports: dict | None = None,
                 pi_address: str | None = "127.0.0.1") -> dict[str, Any]:
    """A valid station as a dict.

    sections: top-level keys to replace, usually process sections ({"teleop": {"motion": {...}}}); any other
    top-level key (data_root, wifi_ssid, ...) works too. ports: replaces some of the free ports. pi_address: None
    leaves pi.address out (valid only with --mock).
    """
    tmp_path = Path(tmp_path)
    d: dict[str, Any] = {
        "station_version": 1,
        "session_root": str(tmp_path / "sessions"),
        "data_root": str(tmp_path / "data"),
        "mock_data_root": str(tmp_path / "data_mock"),
        "pi": {"address": pi_address, "port": 5555},
        "wifi_ssid": "ASUS_5g",
        "ports": dict(zip(PORT_KEYS, free_ports(len(PORT_KEYS)))),
        **{name: {} for name in SECTIONS},
    }
    if pi_address is None:
        del d["pi"]["address"]
    if ports:
        d["ports"].update(ports)
    for key, value in (sections or {}).items():
        d[key] = copy.deepcopy(value)
    return d


def write_station(tmp_path: Path, *, sections: dict | None = None, ports: dict | None = None,
                  pi_address: str | None = "127.0.0.1") -> Path:
    """station_dict() written to <tmp_path>/station.yaml; returns its path."""
    path = Path(tmp_path) / "station.yaml"
    d = station_dict(tmp_path, sections=sections, ports=ports, pi_address=pi_address)
    path.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    return path


def contract_error(fn: Callable[[], Any], field: str, problem: str) -> ContractError:
    """Run fn, expect a ContractError with exactly this field and problem, and return it."""
    with pytest.raises(ContractError) as info:
        fn()
    e = info.value
    assert (e.field, e.problem) == (field, problem), str(e)
    return e
