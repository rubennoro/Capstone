"""The nftables table: printing it, parsing `nft -j` (sample JSON), comparing with link.yaml (STRUCTURE.md 6.6)."""
from __future__ import annotations

import json
import subprocess
from dataclasses import replace

from deploy_contracts import LinkSettings

from control_deploy import firewall

LINK = LinkSettings.load()


def nft_json(addresses=("192.168.50.20",), ports=(5620, 5630), drop=True, loopback=True) -> str:
    dport = {"match": {"op": "==", "left": {"payload": {"protocol": "tcp", "field": "dport"}},
                       "right": {"set": list(ports)}}}
    saddr = list(addresses)
    rules = []
    if loopback:
        rules.append([dport, {"match": {"op": "==", "left": {"meta": {"key": "iif"}}, "right": "lo"}},
                      {"accept": None}])
    rules.append([dport, {"match": {"op": "==", "left": {"payload": {"protocol": "ip", "field": "saddr"}},
                                    "right": saddr[0] if len(saddr) == 1 else {"set": saddr}}}, {"accept": None}])
    if drop:
        rules.append([dport, {"drop": None}])
    items = [{"metainfo": {"version": "1.0.2", "json_schema_version": 1}},
             {"table": {"family": "inet", "name": "control_deploy", "handle": 7}},
             {"chain": {"family": "inet", "table": "control_deploy", "name": "input", "handle": 1, "type": "filter",
                        "hook": "input", "prio": 0, "policy": "accept"}}]
    for i, expr in enumerate(rules):
        items.append({"rule": {"family": "inet", "table": "control_deploy", "chain": "input", "handle": 2 + i,
                               "expr": expr}})
    return json.dumps({"nftables": items})


def test_the_printed_table_for_one_address():
    text = firewall.rule_text(LINK)
    assert text.startswith("table inet control_deploy {\n  chain input {\n")
    assert 'tcp dport { 5620, 5630 } iif "lo" accept' in text
    assert "tcp dport { 5620, 5630 } ip saddr { 192.168.50.20 } accept" in text
    assert "tcp dport { 5620, 5630 } drop\n" in text


def test_the_printed_table_for_two_addresses():
    link = replace(LINK, laptop_addresses=("192.168.50.20", "192.168.50.21"))
    assert "ip saddr { 192.168.50.20, 192.168.50.21 } accept" in firewall.rule_text(link)


def test_parsing_a_matching_table():
    live = firewall.parse_nft_json(nft_json())
    assert live.loopback_ports == {5620, 5630}
    assert live.accepted == {"192.168.50.20": {5620, 5630}}
    assert live.dropped_ports == {5620, 5630}
    assert firewall.compare(live, LINK) == []


def test_two_addresses_as_a_set():
    link = replace(LINK, laptop_addresses=("192.168.50.20", "192.168.50.21"))
    assert firewall.compare(firewall.parse_nft_json(nft_json(("192.168.50.20", "192.168.50.21"))), link) == []


def test_a_missing_drop_rule_is_a_mismatch():
    diffs = firewall.compare(firewall.parse_nft_json(nft_json(drop=False)), LINK)
    assert diffs == ["no final drop for ports [5620, 5630]"]


def test_a_missing_loopback_rule_and_other_addresses_are_mismatches():
    diffs = firewall.compare(firewall.parse_nft_json(nft_json(("10.0.0.5",), loopback=False)), LINK)
    assert any("loopback" in d for d in diffs)
    assert any("not accepted: 192.168.50.20" in d for d in diffs)
    assert any("does not list: 10.0.0.5" in d for d in diffs)


def test_other_ports_are_a_mismatch():
    assert firewall.compare(firewall.parse_nft_json(nft_json(ports=(5620,))), LINK)


def fake_run(results):
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        rc, out, err = results.pop(0)
        return subprocess.CompletedProcess(cmd, rc, out, err)
    return run, calls


def test_read_live_without_sudo_then_with_sudo_n():
    run, calls = fake_run([(1, "", "Operation not permitted (you must be root)"), (0, nft_json(), "")])
    t = firewall.read_live(run)
    assert t.state == "ok" and calls[1][:2] == ["sudo", "-n"]
    run, _ = fake_run([(1, "", "Operation not permitted"), (1, "", "sudo: a password is required")])
    assert firewall.read_live(run).state == "unreadable"
    run, _ = fake_run([(1, "", "Error: No such file or directory\nlist table inet control_deploy")])
    assert firewall.read_live(run).state == "absent"


def test_ufw_conf(tmp_path):
    conf = tmp_path / "ufw.conf"
    conf.write_text("# x\nENABLED=yes\nLOGLEVEL=low\n")
    assert firewall.ufw_enabled(conf)
    conf.write_text("ENABLED=no\n")
    assert not firewall.ufw_enabled(conf)
    assert not firewall.ufw_enabled(tmp_path / "missing")
