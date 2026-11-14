# adapted from rnoro/deployment:deployment/message_comms/tests/test_settings.py @ 91b178b
# changes: config/manager.yaml + station.yaml manager: + link.yaml + frames.yaml; the link values come only from
# link.yaml (contract key names refused in the station); loop_ms, max_state_age_ms, state_lost_s and
# timeouts.commands_s are gone; the new keys of STRUCTURE.md 6.6; endpoints from the station and link.yaml.
import pytest
import yaml
from deploy_contracts import LinkSettings, TcpFrame
from manager_harness import load_settings
from station_helpers import write_station

from policy_deploy.contracts import StationError, paths
from policy_deploy.manager.settings import Settings


def test_config_loads_with_defaults(settings):
    assert settings.exec_h == 8 and settings.pose_source == "target" and settings.lookahead_ms == 0
    assert settings.timeouts.home_s == 90.0 and settings.timeouts.pc_homing_wait_s == 60.0
    assert settings.timeouts.health_s == 0.05 and settings.link_jitter_allowance_s == 0.2
    assert settings.operator_script is None and settings.operator_script_wait_s == 30.0
    c = settings.clock
    assert (c.hello_period_s, c.startup_hello_period_s, c.round_trip_window_s, c.one_way_window_s) == \
        (0.5, 0.02, 10.0, 5.0)
    assert (c.min_samples_s, c.min_samples, c.max_sample_age_s, c.startup_agree_ms) == (2.0, 10, 5.0, 5.0)
    assert (c.mismatch_sets, c.mismatch_min_lead_ms, c.future_state_ms) == (3, 100.0, 20.0)
    keys = set(yaml.safe_load(paths.config_file("manager").read_text()))
    renamed = {"clock", "link_stats"}
    assert keys - renamed <= set(Settings.__dataclass_fields__)


def test_link_values_come_from_link_yaml(settings):
    link = LinkSettings.load()
    assert settings.heartbeat_period_s == link.heartbeat_period_s
    assert settings.c3_timeout_s == link.c3_timeout_s and settings.state_stale_s == link.state_stale_s
    assert settings.state_lost_s == link.state_lost_s and settings.watchdog_limit_s == link.watchdog_limit_s
    assert settings.clock.estimator == link.clock_estimator
    assert settings.frame.F_T_TCP.shape == (4, 4) and settings.frame is TcpFrame.load()


def test_worst_gap_is_0_57(settings):
    assert settings.worst_gap_s == pytest.approx(0.57)


def test_endpoints_real_and_mock(tmp_path):
    link = LinkSettings.load()
    real = load_settings(tmp_path, mock=False)
    assert real.commands_endpoint == f"tcp://{link.pc_address}:{link.commands_port}"
    assert real.state_endpoint == f"tcp://{link.pc_address}:{link.state_port}"
    mock = load_settings(tmp_path, mock=True)
    assert mock.commands_endpoint == f"tcp://127.0.0.1:{link.commands_port}" and mock.mock is True
    assert mock.frames_endpoint.startswith("tcp://127.0.0.1:") and mock.policy_endpoint.startswith("tcp://127.0.0.1:")


def test_station_override_merges_maps_key_by_key(make_settings):
    s = make_settings(exec_h=12, timeouts={"policy_hang_s": 2.5}, clock={"round_trip": {"window_s": 20.0}})
    assert s.exec_h == 12 and s.timeouts.policy_hang_s == 2.5 and s.clock.round_trip_window_s == 20.0
    assert s.timeouts.home_s == 90.0 and s.clock.round_trip_best_fraction == 0.3      # the rest of the map kept


def test_unknown_key_refused_with_its_path(tmp_path):
    with pytest.raises(StationError) as e:
        load_settings(tmp_path, exec_hh=3, timeouts={"home": 1})
    text = str(e.value)
    assert "manager.exec_hh: not a key of config/manager.yaml" in text
    assert "manager.timeouts.home: not a key" in text


@pytest.mark.parametrize("override", [{"c3_timeout_s": 0.5}, {"heartbeat_period_s": 0.01},
                                      {"timeouts": {"state_lost_s": 1.0}}, {"clock": {"estimator": "one_way"}},
                                      {"watchdog_limit_s": 2.0}])
def test_link_keys_refused_in_the_station(tmp_path, override):
    with pytest.raises(StationError) as e:
        load_settings(tmp_path, **override)
    assert "belongs to deployment/contracts/link.yaml" in str(e.value)


@pytest.mark.parametrize("old_key", ["loop_ms", "max_state_age_ms", "state_lost_s"])
def test_old_keys_are_gone(tmp_path, old_key):
    with pytest.raises(StationError):
        load_settings(tmp_path, **{old_key: 1})


@pytest.mark.parametrize("override, words", [
    ({"exec_h": 0}, "exec_h: must be between 1 and 16"),
    ({"exec_h": 17}, "exec_h: must be between 1 and 16"),
    ({"exec_h": 2.5}, "exec_h: expected int"),
    ({"pose_source": "both"}, "pose_source: must be one of target, measured"),
    ({"lookahead_ms": -1}, "lookahead_ms: must be >= 0"),
    ({"odd_chunk_max_pos_m": 0}, "odd_chunk_max_pos_m: must be > 0"),
    ({"odd_chunk_max_rot_deg": -1.0}, "odd_chunk_max_rot_deg: must be > 0"),
    ({"anchor_fallback_m": 0}, "anchor_fallback_m: must be > 0"),
    ({"status_ms": 0}, "status_ms: must be > 0"),
    ({"health_poll_ms": 0}, "health_poll_ms: must be > 0"),
    ({"history_s": 0}, "history_s: must be > 0"),
    ({"max_image_age_ms": 0}, "max_image_age_ms: must be > 0"),
    ({"state_recv_hwm": 0}, "state_recv_hwm: must be > 0"),
    ({"link_jitter_allowance_s": -0.1}, "link_jitter_allowance_s: must be >= 0"),
    ({"listener_poll_s": 0}, "listener_poll_s: must be > 0"),
    ({"prompt_index": -1}, "prompt_index: must be >= 0"),
    ({"skip_warn_s": 0}, "skip_warn_s: must be > 0"),
    ({"operator_script_wait_s": 0}, "operator_script_wait_s: must be > 0"),
    ({"log_steps": "yes"}, "log_steps: expected bool"),
    ({"save_tap": 1}, "save_tap: expected bool"),
    ({"tap_images_every": 0}, "tap_images_every: must be >= 1"),
    ({"task_name": 5}, "task_name: expected str"),
    ({"timeouts": {"health_s": 0}}, "timeouts.health_s: must be > 0"),
    ({"timeouts": {"home_s": "long"}}, "timeouts.home_s: expected float"),
    ({"timeouts": {"pc_homing_wait_s": -1}}, "timeouts.pc_homing_wait_s: must be > 0"),
    ({"clock": {"round_trip": {"best_fraction": 1.5}}}, "clock.round_trip.best_fraction: must be in (0, 1]"),
    ({"clock": {"min_samples": 0}}, "clock.min_samples: must be >= 1"),
    ({"clock": {"mismatch_sets": 2.5}}, "clock.mismatch_sets: expected int"),
    ({"link_stats": {"window_s": 0}}, "link_stats.window_s: must be > 0"),
])
def test_range_checks(tmp_path, override, words):
    with pytest.raises(StationError) as e:
        load_settings(tmp_path, **override)
    assert words in str(e.value)


def test_every_problem_at_once(tmp_path):
    with pytest.raises(StationError) as e:
        load_settings(tmp_path, exec_h=0, status_ms=0, history_s=0)
    assert len(e.value.problems) >= 3


def test_operator_script_resolves_against_the_station_folder(tmp_path):
    s = load_settings(tmp_path, operator_script="scripts/run.yaml")
    assert s.operator_script == tmp_path / "scripts/run.yaml"


def test_to_dict_is_plain(settings):
    d = settings.to_dict()
    yaml.safe_dump(d)                                       # plain values only
    assert d["timeouts"]["home_s"] == 90.0 and d["clock"]["estimator"] == settings.clock.estimator
    assert isinstance(d["link"]["laptop_addresses"], list) and d["frame"]["settled"] is False


def test_from_station_endpoint_overrides(tmp_path):
    from policy_deploy.contracts import Station
    st = Station.load(write_station(tmp_path), mock=True)
    s = Settings.from_station(st, state_endpoint="tcp://127.0.0.1:1", commands_endpoint="tcp://127.0.0.1:2")
    assert (s.state_endpoint, s.commands_endpoint) == ("tcp://127.0.0.1:1", "tcp://127.0.0.1:2")
