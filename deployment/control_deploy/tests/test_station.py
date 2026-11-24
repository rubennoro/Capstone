"""Station: control_deploy/station.yaml (control_deploy/STRUCTURE.md 6.2)."""
from __future__ import annotations

import pytest
from helpers import write_station

from control_deploy import paths
from control_deploy.station import ConsoleSettings, Station


def test_the_real_station_loads():
    s = Station.load(paths.default_station())
    assert s.franka_control == {"realtime": "ignore", "homing": {"q_start_rad": [-0.5928536, 0.5219462, -0.0897168,
                                -1.5930073, 0.6595725, 1.5068439, -1.5301472]}}
    assert s.console == ConsoleSettings(hold_s=3.0, move_m=0.015, rotate_deg=2.0, gripper_delta_m=0.02, segment_s=2.0,
                                        out_of_box_m=0.03)
    assert s.session_root.name == "control_deploy_sessions" and s.session_root.is_absolute()


def test_session_root_relative_to_the_station_folder(tmp_path):
    path = write_station(tmp_path)
    (tmp_path / "station.yaml").write_text("station_version: 1\nsession_root: runs\n")
    assert Station.load(path).session_root == (tmp_path / "runs").resolve()


@pytest.mark.parametrize("extra, key", [
    ({"whatever": 1}, "whatever"),
    ({"endpoints": {"state": "tcp://*:5620"}}, "endpoints"),
    ({"pc": {"address": "1.2.3.4"}}, "pc"),
    ({"max_waypoints": 8}, "max_waypoints"),
    ({"F_T_TCP": [[1]]}, "F_T_TCP"),
])
def test_unknown_and_contract_keys_are_refused_with_the_key(tmp_path, extra, key):
    with pytest.raises(ValueError, match=rf"station.yaml: {key}: "):
        Station.load(write_station(tmp_path, extra=extra))


def test_endpoints_hint(tmp_path):
    with pytest.raises(ValueError, match="no longer used: franka_control binds contracts/link.yaml"):
        Station.load(write_station(tmp_path, extra={"endpoints": {}}))


@pytest.mark.parametrize("version", [2, 0, None, "1", True])
def test_a_bad_station_version_is_refused(tmp_path, version):
    path = write_station(tmp_path)
    text = path.read_text().replace("station_version: 1", f"station_version: {version!r}" if version is not None
                                    else "station_version: null")
    path.write_text(text)
    with pytest.raises(ValueError, match="station_version"):
        Station.load(path)


@pytest.mark.parametrize("console, key", [
    ({"move_m": 0.05}, "console.move_m"),
    ({"rotate_deg": 12}, "console.rotate_deg"),
    ({"out_of_box_m": 0.01}, "console.out_of_box_m"),
    ({"out_of_box_m": 0.06}, "console.out_of_box_m"),
    ({"hold_s": 0}, "console.hold_s"),
    ({"segment_s": -1.0}, "console.segment_s"),
    ({"gripper_delta_m": True}, "console.gripper_delta_m"),
    ({"speed": 1.0}, "console.speed"),
])
def test_out_of_range_console_values_are_refused(tmp_path, console, key):
    with pytest.raises(ValueError, match=rf"station.yaml: {key}: "):
        Station.load(write_station(tmp_path, console=console))


@pytest.mark.parametrize("overrides, key", [
    ({"libfranka_version": "0.10.0"}, "franka_control.libfranka_version"),
    ({"tcp": {"F_T_TCP": []}}, "franka_control.tcp"),
    ({"watchdog": {"limit_s": 2.0}}, "franka_control.watchdog.limit_s"),
])
def test_overrides_the_binary_refuses_are_refused_here_too(tmp_path, overrides, key):
    with pytest.raises(ValueError, match=rf"{key}: "):
        Station.load(write_station(tmp_path, franka_control=overrides))


def test_missing_console_keys_take_the_defaults(tmp_path):
    s = Station.load(write_station(tmp_path, console={"rotate_deg": 3}))
    assert s.console.rotate_deg == 3.0 and s.console.move_m == 0.015
