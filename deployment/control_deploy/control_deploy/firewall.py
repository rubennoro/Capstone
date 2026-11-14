"""FirewallRule: the nftables table that lets only link.yaml laptop.addresses and the PC itself reach C2/C3
(control_deploy/STRUCTURE.md 6.6, decision 4). Prints the text; reads and compares the live table. Never changes it."""
from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from deploy_contracts import LinkSettings

TABLE = "inet control_deploy"
LIST_CMD = ["nft", "-j", "list", "table", "inet", "control_deploy"]
UFW_CONF = Path("/etc/ufw/ufw.conf")


def rule_text(link: LinkSettings) -> str:
    """The table the operator installs once (DIRECTIONS): loopback and the laptop addresses accepted, the rest dropped."""
    ports = f"{{ {link.state_port}, {link.commands_port} }}"
    addresses = ", ".join(link.laptop_addresses)
    return (f"table {TABLE} {{\n"
            f"  chain input {{\n"
            f"    type filter hook input priority filter; policy accept;\n"
            f"    tcp dport {ports} iif \"lo\" accept                       # the PC itself: status line, console\n"
            f"    tcp dport {ports} ip saddr {{ {addresses} }} accept     # link.yaml laptop.addresses\n"
            f"    tcp dport {ports} drop\n"
            f"  }}\n"
            f"}}\n")


@dataclass
class LiveRules:
    """What the live table does for the C2/C3 ports."""

    loopback_ports: set[int] = field(default_factory=set)      # ports accepted from iif lo
    accepted: dict[str, set[int]] = field(default_factory=dict)   # source address -> ports accepted
    dropped_ports: set[int] = field(default_factory=set)       # ports dropped for everyone else
    other: list[str] = field(default_factory=list)             # rules this reader does not understand


def _values(right: Any) -> list[Any]:
    if isinstance(right, dict) and "set" in right:
        return list(right["set"])
    return [right]


def parse_nft_json(text: str) -> LiveRules:
    """`nft -j list table inet control_deploy` -> LiveRules (only the rules' matches and verdicts are read)."""
    live = LiveRules()
    for item in json.loads(text).get("nftables", []):
        rule = item.get("rule")
        if not rule:
            continue
        ports: set[int] = set()
        saddrs: list[Any] = []
        loopback = False
        verdict = None
        unknown = False
        for expr in rule.get("expr", []):
            if "accept" in expr:
                verdict = "accept"
            elif "drop" in expr:
                verdict = "drop"
            elif "match" in expr:
                m = expr["match"]
                left, right = m.get("left", {}), m.get("right")
                if m.get("op", "==") not in ("==", "in"):
                    unknown = True
                elif left.get("payload", {}).get("field") == "dport":
                    ports.update(int(v) for v in _values(right))
                elif left.get("payload", {}) == {"protocol": "ip", "field": "saddr"}:
                    saddrs.extend(_values(right))
                elif left.get("meta", {}).get("key") in ("iif", "iifname") and right == "lo":
                    loopback = True
                else:
                    unknown = True
            elif "counter" in expr:
                pass
            else:
                unknown = True
        if unknown or verdict is None or not ports or any(not isinstance(a, str) for a in saddrs):
            live.other.append(json.dumps(rule.get("expr", [])))
        elif verdict == "accept" and loopback and not saddrs:
            live.loopback_ports |= ports
        elif verdict == "accept" and saddrs and not loopback:
            for a in saddrs:
                live.accepted.setdefault(a, set()).update(ports)
        elif verdict == "drop" and not saddrs and not loopback:
            live.dropped_ports |= ports
        else:
            live.other.append(json.dumps(rule.get("expr", [])))
    return live


def compare(live: LiveRules, link: LinkSettings) -> list[str]:
    """Every difference from link.yaml (empty: the table matches)."""
    ports = {link.state_port, link.commands_port}
    out: list[str] = []
    if not ports <= live.loopback_ports:
        out.append(f"no loopback accept for ports {sorted(ports - live.loopback_ports)}")
    want = set(link.laptop_addresses)
    have = {a for a, p in live.accepted.items() if ports <= p}
    if have != want:
        missing, extra = sorted(want - have), sorted(set(live.accepted) - want)
        if missing:
            out.append(f"laptop address(es) not accepted: {', '.join(missing)}")
        if extra:
            out.append(f"address(es) accepted that link.yaml does not list: {', '.join(extra)}")
    if not ports <= live.dropped_ports:
        out.append(f"no final drop for ports {sorted(ports - live.dropped_ports)}")
    return out


@dataclass(frozen=True)
class LiveTable:
    state: str            # "ok" (read), "absent", "unreadable"
    rules: LiveRules | None
    detail: str


def read_live(run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> LiveTable:
    """Without sudo first, then `sudo -n` (an optional sudoers line allows exactly LIST_CMD)."""
    errors: list[str] = []
    for cmd in (LIST_CMD, ["sudo", "-n", *LIST_CMD]):
        try:
            done = run(cmd, capture_output=True, text=True, timeout=5.0, stdin=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{cmd[0]}: {exc}")
            continue
        if done.returncode == 0:
            return LiveTable("ok", parse_nft_json(done.stdout), "")
        err = (done.stderr or done.stdout).strip()
        if "No such file or directory" in err or "does not exist" in err:
            return LiveTable("absent", None, err.splitlines()[0] if err else "no such table")
        errors.append(err.splitlines()[0] if err else f"exit {done.returncode}")
    return LiveTable("unreadable", None, "; ".join(errors))


def ufw_enabled(conf: Path = UFW_CONF) -> bool:
    try:
        text = conf.read_text()
    except OSError:
        return False
    return any(line.strip().replace(" ", "") == "ENABLED=yes" for line in text.splitlines())
