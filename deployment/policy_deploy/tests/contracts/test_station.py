"""station.yaml (STRUCTURE.md 2.4, 12.2): the example loads; contract key names refused by an explicit list."""
from __future__ import annotations

import pytest
import yaml
from station_helpers import station_dict, write_station

from policy_deploy.contracts import Station, StationError, paths
from policy_deploy.contracts.station import REFUSED_ANYWHERE, REFUSED_TOP


def load(tmp_path, mock=False, **sections):
    return Station.load(write_station(tmp_path, **sections), mock=mock)


def problems(tmp_path, mock=False, **sections):
    with pytest.raises(StationError) as e:
        load(tmp_path, mock, **sections)
    return e.value.problems


def test_the_example_station_loads_real_and_mock():
    for mock in (False, True):
        s = Station.load(paths.default_station(), mock=mock)
        assert s.ports == {"frames": 5610, "policy": 5640, "mock_pi": 5600}
        assert s.frames_endpoint() == "tcp://127.0.0.1:5610" and s.policy_endpoint() == "tcp://127.0.0.1:5640"
        assert s.pi_endpoint() == ("tcp://127.0.0.1:5600" if mock else "tcp://192.168.50.30:5555")
        assert s.section("policy")["kind"] == "hold_still" and s.section("frames") == {}


def test_station_frames_section_and_ports_are_legal(tmp_path):
    s = load(tmp_path, frames={"buffer": {"keep_s": 2.0}}, pi={"address": "10.0.0.5", "port": 5555})
    assert s.section("frames") == {"buffer": {"keep_s": 2.0}}
    assert s.pi_endpoint() == "tcp://10.0.0.5:5555"


@pytest.mark.parametrize("name", sorted(REFUSED_ANYWHERE))
@pytest.mark.parametrize("section", ["manager", "frames", "policy", "fake_pc", "launcher"])
def test_contract_leaf_names_refused_in_any_section(tmp_path, name, section):
    found = problems(tmp_path, **{section: {name: 1.0}})
    assert f"station.yaml: {section}.{name}: belongs to {REFUSED_ANYWHERE[name]}" in found


def test_refused_at_any_depth_and_at_the_top(tmp_path):
    found = problems(tmp_path, manager={"clock": {"estimator": "one_way"}})
    assert "station.yaml: manager.clock.estimator: belongs to deployment/contracts/link.yaml" in found
    d = station_dict(tmp_path)
    d["settled"] = True
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump(d))
    with pytest.raises(StationError) as e:
        Station.load(path)
    assert "s.yaml: settled: belongs to deployment/contracts/frames.yaml" in e.value.problems


def test_examples_from_the_structure_text(tmp_path):
    assert "station.yaml: manager.c3_timeout_s: belongs to deployment/contracts/link.yaml" in problems(
        tmp_path, manager={"c3_timeout_s": 0.5})
    assert "station.yaml: frames.F_T_TCP: belongs to deployment/contracts/frames.yaml" in problems(
        tmp_path, frames={"F_T_TCP": [[1, 0, 0, 0]]})


@pytest.mark.parametrize("name", sorted(REFUSED_TOP))
def test_contract_sections_refused_at_the_top(tmp_path, name):
    found = problems(tmp_path, **{name: {"x": 1}})
    assert f"station.yaml: {name}: belongs to {REFUSED_TOP[name]}" in found


def test_manager_clock_stays_legal(tmp_path):
    s = load(tmp_path, manager={"clock": {"hello_period_s": 0.25}})
    assert s.section("manager")["clock"]["hello_period_s"] == 0.25


def test_pi_address_required_except_mock(tmp_path):
    d = station_dict(tmp_path)
    del d["pi"]["address"]
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump(d))
    with pytest.raises(StationError) as e:
        Station.load(path, mock=False)
    assert any("pi.address: required" in p for p in e.value.problems)
    s = Station.load(path, mock=True)
    assert s.pi_endpoint().startswith("tcp://127.0.0.1:")


def test_python_torch_only_for_diffusion(tmp_path):
    load(tmp_path, python={"light": "/usr/bin/python3"})                            # hold_still: no torch needed
    found = problems(tmp_path, python={"light": "/usr/bin/python3"},
                     policy={"kind": "diffusion", "checkpoint": "/x.ckpt"})
    assert any("python.torch: required for policy kind diffusion" in p for p in found)
    load(tmp_path, mock=True, python={"light": "/usr/bin/python3"},                 # --mock runs mock_kind
         policy={"kind": "diffusion"})
    found = problems(tmp_path, mock=True, python={"light": "/usr/bin/python3"}, policy={"mock_kind": "diffusion"})
    assert any("python.torch" in p for p in found)
    s = load(tmp_path, python={"light": "/usr/bin/python3", "torch": "envs/t/bin/python"},
             policy={"kind": "diffusion"})
    assert s.python("torch") == str(tmp_path / "envs/t/bin/python")


def test_unknown_keys_and_bad_values_all_listed(tmp_path):
    found = problems(tmp_path, ports={"frames": 80, "policy": 5640, "mock_pi": 5640, "extra": 1},
                     manager={"exec_hh": 8}, bogus=1)
    text = "\n".join(found)
    for want in ("bogus: unknown key", "ports.extra: unknown key", "ports.frames: expected a port 1024-65535",
                 "ports.mock_pi: 5640 is also ports.policy"):
        assert want in text, want


def test_process_sections_checked_against_config_files(tmp_path):
    found = problems(tmp_path, manager={"exec_hh": 8}, frames={"select": 3})
    assert "station.yaml: manager.exec_hh: not a key of config/manager.yaml" in found
    assert "station.yaml: frames.select: expected a map (as in config/frames_process.yaml), got 3" in found


def test_config_file_names():
    assert paths.config_file("frames").name == "frames_process.yaml"
    assert paths.config_file("frames").parent.name == "config"
    for name in ("policy", "manager", "fake_pc", "launcher"):
        assert paths.config_file(name).name == f"{name}.yaml"
    with pytest.raises(ValueError):
        paths.config_file("franka_control")


def test_missing_and_unreadable_files(tmp_path):
    with pytest.raises(StationError, match="not found"):
        Station.load(tmp_path / "nope.yaml")
    (tmp_path / "bad.yaml").write_text("- a list\n")
    with pytest.raises(StationError, match="expected a map"):
        Station.load(tmp_path / "bad.yaml")


def test_resolve_and_session_root(tmp_path):
    s = load(tmp_path, session_root="rel/sessions")
    assert s.session_root == tmp_path / "rel/sessions"
    assert s.resolve("~/x").is_absolute()
