"""Local IPv4 addresses per interface (`ip -j -4 addr`) and the C3 peers (`ss`) (control_deploy/STRUCTURE.md 6.3, 6.4).
Parsing is separate from running the tools, so the tests feed samples."""
from __future__ import annotations

import json
import subprocess
from collections.abc import Callable

Runner = Callable[..., subprocess.CompletedProcess]


def parse_ip_addr(text: str) -> dict[str, list[str]]:
    """{"wlp2s0": ["192.168.50.37"], "lo": ["127.0.0.1"], ...} from `ip -j -4 addr` output."""
    out: dict[str, list[str]] = {}
    for link in json.loads(text or "[]"):
        name = link.get("ifname", "?")
        for a in link.get("addr_info", []):
            if a.get("family", "inet") == "inet" and "local" in a:
                out.setdefault(name, []).append(a["local"])
    return out


def local_addresses(run: Runner = subprocess.run) -> dict[str, list[str]]:
    done = run(["ip", "-j", "-4", "addr"], capture_output=True, text=True, timeout=5.0)
    if done.returncode != 0:
        raise RuntimeError(f"ip -j -4 addr failed: {done.stderr.strip()}")
    return parse_ip_addr(done.stdout)


def _host(addr_port: str) -> str:
    host = addr_port.rsplit(":", 1)[0].strip("[]")
    return host[7:] if host.startswith("::ffff:") else host


def parse_ss_peers(text: str) -> list[str]:
    """Peer addresses from `ss -Htn state established '( sport = :PORT )'` (columns: Recv-Q Send-Q Local Peer ...)."""
    peers: list[str] = []
    for line in text.splitlines():
        cols = line.split()
        if len(cols) < 4:
            continue
        local, peer = (cols[3], cols[4]) if not cols[0].isdigit() else (cols[2], cols[3])
        del local
        host = _host(peer)
        if host not in peers:
            peers.append(host)
    return peers


def c3_peers(port: int, run: Runner = subprocess.run) -> list[str]:
    """The addresses of the clients connected to the C3 port now (empty if ss is missing)."""
    try:
        done = run(["ss", "-Htn", "state", "established", f"( sport = :{port} )"], capture_output=True, text=True,
                   timeout=2.0)
    except (OSError, subprocess.TimeoutExpired):
        return []
    return parse_ss_peers(done.stdout) if done.returncode == 0 else []
