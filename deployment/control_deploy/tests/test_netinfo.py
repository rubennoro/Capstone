"""Parsing `ip -j -4 addr` and `ss -Htn` samples (control_deploy/STRUCTURE.md 9.2)."""
from __future__ import annotations

import json
import subprocess

from control_deploy import netinfo

IP_SAMPLE = json.dumps([
    {"ifindex": 1, "ifname": "lo", "addr_info": [{"family": "inet", "local": "127.0.0.1", "prefixlen": 8}]},
    {"ifindex": 2, "ifname": "enp3s0", "addr_info": [{"family": "inet", "local": "172.16.0.1", "prefixlen": 24}]},
    {"ifindex": 3, "ifname": "wlp2s0", "addr_info": [{"family": "inet", "local": "192.168.50.10", "prefixlen": 24},
                                                    {"family": "inet", "local": "192.168.50.11", "prefixlen": 24}]},
    {"ifindex": 4, "ifname": "docker0", "addr_info": []},
])


def test_ip_addr():
    assert netinfo.parse_ip_addr(IP_SAMPLE) == {"lo": ["127.0.0.1"], "enp3s0": ["172.16.0.1"],
                                                "wlp2s0": ["192.168.50.10", "192.168.50.11"]}


def test_ss_peers_with_and_without_the_state_column():
    filtered = ("0      0      192.168.50.10:5630   192.168.50.20:41234\n"
                "0      0          127.0.0.1:5630       127.0.0.1:50000 \n"
                "0      0      192.168.50.10:5630   192.168.50.20:41235\n")
    assert netinfo.parse_ss_peers(filtered) == ["192.168.50.20", "127.0.0.1"]
    full = "ESTAB 0      0      [::ffff:192.168.50.10]:5630   [::ffff:192.168.50.20]:41234\n"
    assert netinfo.parse_ss_peers(full) == ["192.168.50.20"]
    assert netinfo.parse_ss_peers("") == []


def test_c3_peers_runs_ss_with_the_port_filter():
    seen = []

    def run(cmd, **kw):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "0 0 192.168.50.10:5630 192.168.50.20:1\n", "")
    assert netinfo.c3_peers(5630, run) == ["192.168.50.20"]
    assert seen[0] == ["ss", "-Htn", "state", "established", "( sport = :5630 )"]
