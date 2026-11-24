"""link.yaml and LinkSettings (STRUCTURE.md 4.3, 4.4)."""
from __future__ import annotations

import pytest
import yaml

from deploy_contracts import LinkSettings, link_file


def _good() -> dict:
    return yaml.safe_load(link_file().read_text())


def _write(tmp_path, d):
    p = tmp_path / "link.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


def _with(tmp_path, section, key, value):
    d = _good()
    d[section][key] = value
    return _write(tmp_path, d)


def test_loads_the_real_file():
    link = LinkSettings.load()
    assert link.pc_address == "192.168.50.10"
    assert link.laptop_addresses == ("192.168.50.20",)
    assert (link.state_port, link.commands_port) == (5620, 5630)
    assert link.heartbeat_period_s == 0.02 and link.watchdog_limit_s == 1.0 and link.c3_timeout_s == 0.3
    assert link.state_stale_s == 0.1 and link.state_lost_s == 2.0
    assert link.clock_estimator == "round_trip"
    assert LinkSettings.load() is link  # cached
    assert LinkSettings.load(link_file()) == link


def test_file_layout():
    assert {k: list(v) for k, v in _good().items()} == {
        "pc": ["address"], "laptop": ["addresses"], "ports": ["state", "commands"],
        "timing": ["heartbeat_period_s", "watchdog_limit_s", "c3_timeout_s", "state_stale_s", "state_lost_s"],
        "clock": ["estimator"]}


def test_endpoints():
    link = LinkSettings.load()
    assert link.state_endpoint() == "tcp://192.168.50.10:5620"
    assert link.commands_endpoint() == "tcp://192.168.50.10:5630"
    assert link.state_endpoint(local=True) == "tcp://127.0.0.1:5620"
    assert link.commands_endpoint(local=True) == "tcp://127.0.0.1:5630"


def test_two_laptop_addresses(tmp_path):
    link = LinkSettings.load(_with(tmp_path, "laptop", "addresses", ["192.168.50.20", "192.168.50.21"]))
    assert link.laptop_addresses == ("192.168.50.20", "192.168.50.21")


def test_integer_seconds_are_accepted_as_floats(tmp_path):
    link = LinkSettings.load(_with(tmp_path, "timing", "state_lost_s", 3))
    assert link.state_lost_s == 3.0 and isinstance(link.state_lost_s, float)


@pytest.mark.parametrize("value", ["0.0.0.0", "pc.local", "localhost", "192.168.50", "192.168.50.256",
                                   "192.168.050.10", "::1", "", 3232248330, None, ["192.168.50.10"]])
def test_pc_address_refused(tmp_path, value):
    with pytest.raises(ValueError, match="pc.address"):
        LinkSettings.load(_with(tmp_path, "pc", "address", value))


@pytest.mark.parametrize("value", [[], "192.168.50.20", ["laptop.local"], ["192.168.50.20", "192.168.50.20"],
                                   [7], None])
def test_laptop_addresses_refused(tmp_path, value):
    with pytest.raises(ValueError, match="laptop.addresses"):
        LinkSettings.load(_with(tmp_path, "laptop", "addresses", value))


@pytest.mark.parametrize("key", ["state", "commands"])
@pytest.mark.parametrize("value", [1023, 65536, 0, -1, 5620.0, "5620", True, None])
def test_port_refused(tmp_path, key, value):
    with pytest.raises(ValueError, match=f"ports.{key}"):
        LinkSettings.load(_with(tmp_path, "ports", key, value))


def test_port_range_ends_accepted(tmp_path):
    d = _good()
    d["ports"] = {"state": 1024, "commands": 65535}
    link = LinkSettings.load(_write(tmp_path, d))
    assert (link.state_port, link.commands_port) == (1024, 65535)


def test_equal_ports_refused(tmp_path):
    with pytest.raises(ValueError, match="ports.commands"):
        LinkSettings.load(_with(tmp_path, "ports", "commands", 5620))


@pytest.mark.parametrize("key", ["heartbeat_period_s", "watchdog_limit_s", "c3_timeout_s", "state_stale_s",
                                 "state_lost_s"])
@pytest.mark.parametrize("value", [0.0, -0.1, float("inf"), float("nan"), "1.0", True, None])
def test_timing_type_and_range(tmp_path, key, value):
    with pytest.raises(ValueError, match=f"timing.{key}"):
        LinkSettings.load(_with(tmp_path, "timing", key, value))


@pytest.mark.parametrize("change,key", [
    ({"heartbeat_period_s": 1.0}, "timing.heartbeat_period_s"),       # equal to the watchdog
    ({"heartbeat_period_s": 1.5}, "timing.heartbeat_period_s"),
    ({"c3_timeout_s": 1.0}, "timing.c3_timeout_s"),
    ({"watchdog_limit_s": 0.25}, "timing.c3_timeout_s"),             # now c3_timeout_s 0.3 is not below it
    ({"state_stale_s": 2.0}, "timing.state_stale_s"),
    ({"state_lost_s": 0.05}, "timing.state_stale_s"),
])
def test_cross_key_rules(tmp_path, change, key):
    d = _good()
    d["timing"].update(change)
    with pytest.raises(ValueError, match=key):
        LinkSettings.load(_write(tmp_path, d))


@pytest.mark.parametrize("value", ["kalman", "", None, "ROUND_TRIP"])
def test_unknown_estimator_refused(tmp_path, value):
    with pytest.raises(ValueError, match="clock.estimator"):
        LinkSettings.load(_with(tmp_path, "clock", "estimator", value))


def test_one_way_accepted(tmp_path):
    assert LinkSettings.load(_with(tmp_path, "clock", "estimator", "one_way")).clock_estimator == "one_way"


@pytest.mark.parametrize("section", ["pc", "laptop", "ports", "timing", "clock"])
def test_missing_section_refused(tmp_path, section):
    d = _good()
    del d[section]
    with pytest.raises(ValueError, match=section):
        LinkSettings.load(_write(tmp_path, d))


@pytest.mark.parametrize("section,key", [("pc", "address"), ("laptop", "addresses"), ("ports", "state"),
                                         ("timing", "watchdog_limit_s"), ("clock", "estimator")])
def test_missing_key_refused(tmp_path, section, key):
    d = _good()
    del d[section][key]
    with pytest.raises(ValueError, match=f"{section}.{key}"):
        LinkSettings.load(_write(tmp_path, d))


def test_unknown_keys_refused(tmp_path):
    d = _good()
    d["timing"]["max_lead_s"] = 1.5  # stays in franka_control/config.yaml, never here
    with pytest.raises(ValueError, match="timing.max_lead_s"):
        LinkSettings.load(_write(tmp_path, d))
    d = _good()
    d["watchdog"] = {"limit_s": 1.0}
    with pytest.raises(ValueError, match="watchdog"):
        LinkSettings.load(_write(tmp_path, d))
    d = _good()
    d["pc"] = "192.168.50.10"
    with pytest.raises(ValueError, match="pc"):
        LinkSettings.load(_write(tmp_path, d))


def test_errors_name_the_file(tmp_path):
    p = _with(tmp_path, "ports", "state", 80)
    with pytest.raises(ValueError) as info:
        LinkSettings.load(p)
    assert str(info.value).startswith(f"{p}: ports.state:")
