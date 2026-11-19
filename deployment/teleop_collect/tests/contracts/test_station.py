# adapted from rnoro/final-deployment:deployment/policy_deploy/tests/contracts/test_station.py @ 2ffe641
# changes: the teleop_collect station (STRUCTURE.md 3.2): sections frames/teleop/ui/fake_pc/launcher, ports
# mock_pi/r1/preview/u1/http and their endpoints, data_root and mock_data_root required (data_root follows --mock,
# the two differ); the python/policy-kind test dropped; + the station_helpers contract (distinct free ports, roots under tmp_path, the Pi on 127.0.0.1).
"""station.yaml (STRUCTURE.md 3.2, 12 row E1): the example loads; contract key names refused by an explicit list."""
from __future__ import annotations

import pytest
import yaml
from station_helpers import station_dict, write_station

from teleop_collect.contracts import Station, StationError, paths
from teleop_collect.contracts.station import PORT_KEYS, PROCESS_SECTIONS, REFUSED_ANYWHERE, REFUSED_TOP


def load(tmp_path, mock=False, **kw):
    return Station.load(write_station(tmp_path, **kw), mock=mock)


def problems(tmp_path, mock=False, **kw):
    with pytest.raises(StationError) as e:
        load(tmp_path, mock, **kw)
    return e.value.problems


def test_the_example_station_loads_real_and_mock():
    for mock in (False, True):
        s = Station.load(paths.default_station(), mock=mock)
        assert s.ports == {"mock_pi": 5650, "r1": 5651, "preview": 5652, "u1": 5653, "http": 8765}
        assert s.r1_endpoint() == "tcp://127.0.0.1:5651" and s.preview_endpoint() == "tcp://127.0.0.1:5652"
        assert s.u1_endpoint() == "tcp://127.0.0.1:5653" and s.mock_pi_endpoint() == "tcp://127.0.0.1:5650"
        assert s.http_url() == "http://127.0.0.1:8765" and s.http_port == 8765
        assert s.pi_endpoint() == ("tcp://127.0.0.1:5650" if mock else "tcp://192.168.50.30:5555")
        assert s.data_root.name == ("CapstoneTeleop_mock" if mock else "CapstoneTeleop") and s.data_root.is_absolute()
        assert s.real_data_root.name == "CapstoneTeleop" and s.mock_data_root.name == "CapstoneTeleop_mock"
        assert s.session_root.name == "teleop_collect_sessions" and s.session_root.is_absolute()
        assert s.wifi_ssid == "ASUS_5g" and s.mock is mock
        for name in PROCESS_SECTIONS:
            assert s.section(name) == {}


def test_process_sections():
    assert PROCESS_SECTIONS == ("frames", "teleop", "ui", "fake_pc", "launcher")
    assert PORT_KEYS == ("mock_pi", "r1", "preview", "u1", "http")


def test_the_helper_writes_a_valid_station(tmp_path):
    s = load(tmp_path)
    assert s.session_root == tmp_path / "sessions" and s.data_root == tmp_path / "data"
    assert s.mock_data_root == tmp_path / "data_mock"
    assert set(s.ports) == set(PORT_KEYS) and len(set(s.ports.values())) == len(PORT_KEYS)
    assert s.pi_endpoint() == "tcp://127.0.0.1:5555"
    s = load(tmp_path, ports={"http": 18765}, sections={"teleop": {"history_s": 2.0}})
    assert s.http_port == 18765 and s.section("teleop") == {"history_s": 2.0}
    s = load(tmp_path, mock=True, pi_address=None)
    assert s.pi_address is None and s.pi_endpoint() == s.mock_pi_endpoint()
    assert s.data_root == tmp_path / "data_mock" and s.real_data_root == tmp_path / "data"


def test_mock_runs_use_mock_data_root_and_the_two_roots_must_differ(tmp_path):
    assert load(tmp_path).data_root == tmp_path / "data"
    assert load(tmp_path, mock=True).data_root == tmp_path / "data_mock"
    for same in (str(tmp_path / "data"), str(tmp_path / "data_mock" / ".." / "data")):
        with pytest.raises(StationError) as e:
            load(tmp_path, sections={"mock_data_root": same})
        assert "station.yaml: mock_data_root: must differ from data_root" in str(e.value)


def test_station_frames_section_and_pi_are_legal(tmp_path):
    s = load(tmp_path, sections={"frames": {"buffer": {"keep_s": 2.0}}}, pi_address="10.0.0.5")
    assert s.section("frames") == {"buffer": {"keep_s": 2.0}}
    assert s.pi_endpoint() == "tcp://10.0.0.5:5555"


@pytest.mark.parametrize("name", sorted(REFUSED_ANYWHERE))
@pytest.mark.parametrize("section", PROCESS_SECTIONS)
def test_contract_leaf_names_refused_in_any_section(tmp_path, name, section):
    found = problems(tmp_path, sections={section: {name: 1.0}})
    assert f"station.yaml: {section}.{name}: belongs to {REFUSED_ANYWHERE[name]}" in found


def test_refused_at_any_depth_and_at_the_top(tmp_path):
    found = problems(tmp_path, sections={"teleop": {"clock": {"estimator": "one_way"}}})
    assert "station.yaml: teleop.clock.estimator: belongs to deployment/contracts/link.yaml" in found
    found = problems(tmp_path, sections={"teleop": {"motion": {"max_waypoints": 3}}})
    assert "station.yaml: teleop.motion.max_waypoints: belongs to deployment/contracts/limits.yaml" in found
    d = station_dict(tmp_path)
    d["settled"] = True
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump(d))
    with pytest.raises(StationError) as e:
        Station.load(path)
    assert "s.yaml: settled: belongs to deployment/contracts/frames.yaml" in e.value.problems


def test_examples_from_the_structure_text(tmp_path):
    assert "station.yaml: teleop.c3_timeout_s: belongs to deployment/contracts/link.yaml" in problems(
        tmp_path, sections={"teleop": {"c3_timeout_s": 0.5}})
    assert "station.yaml: frames.F_T_TCP: belongs to deployment/contracts/frames.yaml" in problems(
        tmp_path, sections={"frames": {"F_T_TCP": [[1, 0, 0, 0]]}})
    assert "station.yaml: teleop.motion.gripper_width_max_m: belongs to deployment/contracts/limits.yaml" in problems(
        tmp_path, sections={"teleop": {"motion": {"gripper_width_max_m": 0.1}}})


@pytest.mark.parametrize("name", sorted(REFUSED_TOP))
def test_contract_sections_refused_at_the_top(tmp_path, name):
    found = problems(tmp_path, sections={name: {"x": 1}})
    assert f"station.yaml: {name}: belongs to {REFUSED_TOP[name]}" in found


def test_teleop_clock_stays_legal(tmp_path):
    s = load(tmp_path, sections={"teleop": {"clock": {"hello_period_s": 0.25}}})
    assert s.section("teleop")["clock"]["hello_period_s"] == 0.25


def test_pi_address_required_except_mock(tmp_path):
    path = write_station(tmp_path, pi_address=None)
    with pytest.raises(StationError) as e:
        Station.load(path, mock=False)
    assert any("pi.address: required" in p for p in e.value.problems)
    s = Station.load(path, mock=True)
    assert s.pi_endpoint().startswith("tcp://127.0.0.1:")
    assert "pi.address: expected an IPv4 address, got 'pi.local'" in "\n".join(
        problems(tmp_path, pi_address="pi.local"))


@pytest.mark.parametrize("key", ["data_root", "mock_data_root", "session_root"])
def test_roots_required(tmp_path, key):
    d = station_dict(tmp_path)
    del d[key]
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump(d))
    with pytest.raises(StationError) as e:
        Station.load(path, mock=True)
    assert f"s.yaml: {key}: required" in e.value.problems
    assert f"station.yaml: {key}: expected a path, got 3" in problems(tmp_path, sections={key: 3})


def test_python_section_is_gone(tmp_path):
    found = problems(tmp_path, sections={"python": {"light": "/usr/bin/python3"}})
    assert any(p.startswith("station.yaml: python: unknown key") for p in found)


def test_unknown_keys_and_bad_values_all_listed(tmp_path):
    found = problems(tmp_path, ports={"r1": 80, "u1": 5640, "preview": 5640, "extra": 1},
                     sections={"teleop": {"exec_h": 8}, "bogus": 1})
    text = "\n".join(found)
    for want in ("bogus: unknown key", "ports.extra: unknown key", "ports.r1: expected a port 1024-65535",
                 "ports.u1: 5640 is also ports.preview"):
        assert want in text, want


def test_every_port_required(tmp_path):
    d = station_dict(tmp_path)
    del d["ports"]["http"]
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump(d))
    with pytest.raises(StationError) as e:
        Station.load(path)
    assert "s.yaml: ports.http: required" in e.value.problems


def test_process_sections_checked_against_config_files(tmp_path):
    found = problems(tmp_path, sections={"teleop": {"exec_h": 8}, "frames": {"select": 3},
                                         "ui": {"status_hz": {"a": 1}}, "launcher": {"policy": 1}})
    assert "station.yaml: teleop.exec_h: not a key of config/teleop.yaml" in found
    assert "station.yaml: frames.select: expected a map (as in config/frames_process.yaml), got 3" in found
    assert "station.yaml: launcher.policy: not a key of config/launcher.yaml" in found
    s = load(tmp_path, sections={"ui": {"status_hz": {"a": 1}}})          # a value replaced by a map: the
    assert s.section("ui") == {"status_hz": {"a": 1}}                     # process's settings.py refuses it


def test_dropped_frames_keys_refused(tmp_path):
    for key in ("main_spacing_ms", "ultrawide_spacing_ms", "tolerance_frames"):
        assert f"station.yaml: frames.select.{key}: not a key of config/frames_process.yaml" in problems(
            tmp_path, sections={"frames": {"select": {key: 1.0}}})
    assert "station.yaml: frames.probe: not a key of config/frames_process.yaml" in problems(
        tmp_path, sections={"frames": {"probe": {}}})


def test_config_file_names():
    assert paths.config_file("frames").name == "frames_process.yaml"
    assert paths.config_file("frames").parent.name == "config"
    for name in ("teleop", "ui", "fake_pc", "launcher"):
        assert paths.config_file(name).name == f"{name}.yaml"
    for name in ("franka_control", "policy", "manager"):
        with pytest.raises(ValueError):
            paths.config_file(name)


def test_unknown_section_name(tmp_path):
    with pytest.raises(ValueError, match="unknown section"):
        load(tmp_path).section("manager")


def test_missing_and_unreadable_files(tmp_path):
    with pytest.raises(StationError, match="not found"):
        Station.load(tmp_path / "nope.yaml")
    (tmp_path / "bad.yaml").write_text("- a list\n")
    with pytest.raises(StationError, match="expected a map"):
        Station.load(tmp_path / "bad.yaml")


def test_resolve_and_roots(tmp_path):
    s = load(tmp_path, sections={"session_root": "rel/sessions", "data_root": "~/teleop_data"})
    assert s.session_root == tmp_path / "rel/sessions"
    assert s.data_root.is_absolute() and "~" not in str(s.data_root) and s.data_root.name == "teleop_data"
    assert s.resolve("~/x").is_absolute()
    assert s.raw()["data_root"] == "~/teleop_data"
