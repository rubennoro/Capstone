"""teleop Settings (STRUCTURE.md 3.3, 6.2): config/teleop.yaml merged with station.yaml teleop:, link.yaml values,
E4's MotionSettings / HomeSettings and their refusals."""
from __future__ import annotations

import pytest
import yaml
from deploy_contracts import LinkSettings
from station_helpers import write_station

from teleop_collect.contracts import StationError
from teleop_collect.teleop.settings import Settings


def test_defaults(settings):
    link = LinkSettings.load()
    assert settings.heartbeat_period_s == link.heartbeat_period_s and settings.c3_timeout_s == link.c3_timeout_s
    assert settings.worst_gap_s == pytest.approx(0.57)
    assert settings.waypoint_lead_ms == 100.0
    assert settings.motion.v_max_m_s == 0.05 and settings.motion.waypoints == 2
    assert settings.home.on_start is True and settings.home.tolerance_rad == 0.01
    assert settings.u1_max_per_cycle == 4 and settings.at_limit_hold_ms == 500 and settings.messages_kept == 200
    assert settings.timeouts.frames_s == 4.0 and settings.clock.mismatch_min_lead_ms == 60.0
    assert settings.camera_down_s == 2.0
    assert settings.gripper_width_max_m == 0.08
    assert settings.rig_file.name == "rig_pi_v3_640.yaml" and settings.rig_file.is_file()
    assert settings.data_root.name == ("data_mock" if settings.mock else "data")     # --mock: mock_data_root


def test_mock_endpoints_are_local(make_settings, tmp_path):
    s = make_settings()
    assert s.state_endpoint.startswith("tcp://127.0.0.1:") and s.commands_endpoint.startswith("tcp://127.0.0.1:")
    (tmp_path / "real").mkdir()
    real = Settings.load(write_station(tmp_path / "real", sections={}), mock=False)
    assert LinkSettings.load().pc_address in real.state_endpoint
    assert s.u1_endpoint.startswith("tcp://127.0.0.1:") and s.r1_endpoint.startswith("tcp://127.0.0.1:")


def test_overrides_merge(make_settings):
    s = make_settings(motion={"v_max_m_s": 0.02}, timeouts={"home_s": 100.0}, u1_max_per_cycle=2)
    assert s.motion.v_max_m_s == 0.02 and s.motion.w_max_deg_s == 10.0
    assert s.timeouts.home_s == 100.0 and s.u1_max_per_cycle == 2


@pytest.mark.parametrize("overrides, text", [
    ({"history_s": 0}, "teleop.history_s: must be > 0"),
    ({"u1_max_per_cycle": 0}, "teleop.u1_max_per_cycle: must be >= 1"),
    ({"health_poll_ms": 0.5}, "teleop.health_poll_ms: expected int"),
    ({"clock": {"round_trip": {"best_fraction": 2.0}}}, "must be in (0, 1]"),
    ({"motion": {"v_max_m_s": 0.2}}, "teleop.motion:"),               # the margin rule (E4's MotionSettings)
    ({"home": {"tolerance_rad": -1.0}}, "teleop.home:"),
])
def test_refusals(make_settings, overrides, text):
    with pytest.raises(StationError) as e:
        make_settings(**overrides)
    assert text in str(e.value), str(e.value)


def test_unknown_key_is_refused_by_the_merge(make_settings):
    with pytest.raises(StationError) as e:
        make_settings(nonsense=1)
    assert "nonsense" in str(e.value)


def test_to_dict_and_task_settings_are_plain(settings):
    for doc in (settings.to_dict(), settings.task_settings()):
        assert yaml.safe_load(yaml.safe_dump(doc)) == doc
    assert settings.task_settings()["motion"]["waypoints"] == 2
