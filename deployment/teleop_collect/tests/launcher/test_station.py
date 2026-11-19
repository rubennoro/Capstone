# adapted from rnoro/final-deployment:deployment/policy_deploy/tests/launcher/test_station.py @ 2ffe641
# changes: the teleop_collect processes and timeouts (teleop 150 s above its longest legal start-up, from
# config/teleop.yaml and config/fake_pc.yaml); no python section (one environment); imports. The source's own header
# follows.
# adapted from rnoro/deployment:deployment/deploy/tests/test_station.py @ 91b178b
# changes: only what tests/contracts/test_station.py does not cover: the launcher's settings (config/launcher.yaml +
# station launcher:, the old LAUNCHER_DEFAULTS / ready_timeout_s cases) and python.light's default.
from __future__ import annotations

import pytest

from teleop_collect.contracts import Station, StationError
from teleop_collect.launcher.launcher import LauncherSettings


def test_defaults_from_config_launcher_yaml(make_station):
    s = LauncherSettings.load(Station.load(make_station(), False))
    assert (s.stop_timeout_s, s.poll_s, s.fake_pc) == (5.0, 0.1, True)
    assert s.ready_timeout_s == {"fake_pc": 10.0, "frames": 30.0, "teleop": 150.0, "ui": 15.0}


def test_teleop_timeout_above_the_longest_legal_startup(make_station):
    """150 s > startup_s 20 + pc_homing_wait_s 60 + homing_timeout_s 30 + 5 (STRUCTURE.md 3.3)."""
    from teleop_collect.contracts.config_merge import load_config
    station = Station.load(make_station(), False)
    teleop, fake_pc = load_config("teleop", station), load_config("fake_pc", station)
    longest = teleop["timeouts"]["startup_s"] + teleop["timeouts"]["pc_homing_wait_s"] + \
        fake_pc["homing_timeout_s"] + 5.0
    assert LauncherSettings.load(station).ready_timeout_s["teleop"] > longest == 115.0


def test_station_overrides(make_station):
    s = LauncherSettings.load(Station.load(make_station(launcher={"poll_s": 0.5, "fake_pc": False,
                                                                  "ready_timeout_s": {"frames": 3}}), True))
    assert (s.poll_s, s.fake_pc, s.ready_timeout_s["frames"]) == (0.5, False, 3.0)


@pytest.mark.parametrize("section,match", [
    ({"poll_s": 0}, "launcher.poll_s: expected a number > 0"),
    ({"stop_timeout_s": "x"}, "launcher.stop_timeout_s"),
    ({"fake_pc": 1}, "launcher.fake_pc: expected true or false"),
    ({"ready_timeout_s": {"teleop": -1}}, "launcher.ready_timeout_s.teleop"),
])
def test_bad_values_refused(make_station, section, match):
    with pytest.raises(StationError, match=match):
        LauncherSettings.load(Station.load(make_station(launcher=section), False))


def test_unknown_launcher_keys_refused_by_the_station(make_station):
    with pytest.raises(StationError, match="launcher.tcp_check_max_mm: not a key of config/launcher.yaml"):
        Station.load(make_station(launcher={"tcp_check_max_mm": 2.0}), False)
    with pytest.raises(StationError, match="launcher.ready_timeout_s.franka_control: not a key"):
        Station.load(make_station(launcher={"ready_timeout_s": {"franka_control": 30}}), False)
