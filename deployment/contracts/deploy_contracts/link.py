"""LinkSettings: link.yaml, the addresses, ports and timing of the laptop <-> PC link (STRUCTURE.md 4.3, 4.4)."""
from __future__ import annotations

import ipaddress
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import yaml

from deploy_contracts.paths import link_file

ESTIMATORS = ("round_trip", "one_way")
LOCAL_HOST = "127.0.0.1"

# section -> its keys, in file order. Every key is required and no other key is allowed.
_LAYOUT: dict[str, tuple[str, ...]] = {
    "pc": ("address",),
    "laptop": ("addresses",),
    "ports": ("state", "commands"),
    "timing": ("heartbeat_period_s", "watchdog_limit_s", "c3_timeout_s", "state_stale_s", "state_lost_s"),
    "clock": ("estimator",),
}


@dataclass(frozen=True)
class LinkSettings:
    pc_address: str                      # pc.address: the IPv4 address franka_control binds and the laptop connects to
    laptop_addresses: tuple[str, ...]    # laptop.addresses: the PC's firewall lets these (and loopback) in
    state_port: int                      # ports.state (C2)
    commands_port: int                   # ports.commands (C3)
    heartbeat_period_s: float            # timing.*
    watchdog_limit_s: float
    c3_timeout_s: float
    state_stale_s: float
    state_lost_s: float
    clock_estimator: str                 # clock.estimator: round_trip | one_way

    _cached: ClassVar[LinkSettings | None] = None

    def state_endpoint(self, local: bool = False) -> str:
        """tcp://<pc.address>:<ports.state>, or tcp://127.0.0.1:<ports.state> with local."""
        return f"tcp://{LOCAL_HOST if local else self.pc_address}:{self.state_port}"

    def commands_endpoint(self, local: bool = False) -> str:
        """tcp://<pc.address>:<ports.commands>, or tcp://127.0.0.1:<ports.commands> with local."""
        return f"tcp://{LOCAL_HOST if local else self.pc_address}:{self.commands_port}"

    @classmethod
    def load(cls, path: Path | str | None = None) -> LinkSettings:
        """The link settings. With no path: deployment/contracts/link.yaml, read once and cached.

        Raises ValueError naming the file and the key (dotted, e.g. timing.c3_timeout_s): missing or unknown key,
        wrong type, value out of range, or a broken cross-key rule.
        """
        if path is None:
            if cls._cached is None:
                cls._cached = cls._read(link_file())
            return cls._cached
        return cls._read(Path(path))

    @classmethod
    def _read(cls, path: Path) -> LinkSettings:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        values = _flat(path, raw)

        def err(key: str, text: str) -> ValueError:
            return ValueError(f"{path}: {key}: {text}")

        address = _ipv4(path, "pc.address", values["pc.address"])
        if address == "0.0.0.0":
            raise err("pc.address", "must be the PC's own address, not 0.0.0.0 (never bind every interface)")

        laptops = values["laptop.addresses"]
        if not isinstance(laptops, list) or not laptops:
            raise err("laptop.addresses", f"must be a list of at least one IPv4 address, got {laptops!r}")
        laptop_addresses = tuple(_ipv4(path, f"laptop.addresses[{i}]", a) for i, a in enumerate(laptops))
        if len(set(laptop_addresses)) != len(laptop_addresses):
            raise err("laptop.addresses", f"addresses must be unique, got {list(laptop_addresses)}")

        ports = {}
        for key in ("ports.state", "ports.commands"):
            v = values[key]
            if not isinstance(v, int) or isinstance(v, bool) or not 1024 <= v <= 65535:
                raise err(key, f"must be an integer port 1024-65535, got {v!r}")
            ports[key] = v
        if ports["ports.state"] == ports["ports.commands"]:
            raise err("ports.commands", f"must differ from ports.state ({ports['ports.state']})")

        timing = {}
        for name in _LAYOUT["timing"]:
            key = f"timing.{name}"
            v = values[key]
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v <= 0:
                raise err(key, f"must be a finite number of seconds > 0, got {v!r}")
            timing[name] = float(v)
        for lower, upper in (("heartbeat_period_s", "watchdog_limit_s"), ("c3_timeout_s", "watchdog_limit_s"),
                             ("state_stale_s", "state_lost_s")):
            if not timing[lower] < timing[upper]:
                raise err(f"timing.{lower}", f"{timing[lower]} must be below timing.{upper} ({timing[upper]})")

        estimator = values["clock.estimator"]
        if estimator not in ESTIMATORS:
            raise err("clock.estimator", f"must be one of {', '.join(ESTIMATORS)}, got {estimator!r}")

        return cls(pc_address=address, laptop_addresses=laptop_addresses, state_port=ports["ports.state"],
                   commands_port=ports["ports.commands"], clock_estimator=estimator, **timing)


def _flat(path: Path, raw: Any) -> dict[str, Any]:
    """{"timing.c3_timeout_s": 0.3, ...}; refuses a missing or unknown section or key."""
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a map with the sections {', '.join(_LAYOUT)}")
    for section in raw:
        if section not in _LAYOUT:
            raise ValueError(f"{path}: unknown key {section!r}")
    out: dict[str, Any] = {}
    for section, keys in _LAYOUT.items():
        if section not in raw:
            raise ValueError(f"{path}: missing key {section!r}")
        body = raw[section]
        if not isinstance(body, dict):
            raise ValueError(f"{path}: {section}: expected a map with the keys {', '.join(keys)}, got {body!r}")
        for key in body:
            if key not in keys:
                raise ValueError(f"{path}: unknown key '{section}.{key}'")
        for key in keys:
            if key not in body:
                raise ValueError(f"{path}: missing key '{section}.{key}'")
            out[f"{section}.{key}"] = body[key]
    return out


def _ipv4(path: Path, key: str, value: Any) -> str:
    """The value if it is an IPv4 dotted literal (no hostnames, no IPv6), else ValueError naming the key."""
    if not isinstance(value, str):
        raise ValueError(f"{path}: {key}: expected an IPv4 address, got {value!r}")
    try:
        ipaddress.IPv4Address(value)
    except ValueError:
        raise ValueError(f"{path}: {key}: expected an IPv4 address, got {value!r}") from None
    return value
