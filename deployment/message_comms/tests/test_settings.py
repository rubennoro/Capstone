import re

import pytest
import yaml

from conftest import write_station
from message_comms.errors import SettingsError
from message_comms.settings import CONFIG_FILE, load_settings


def test_config_yaml_loads_with_defaults(settings):
    assert settings.exec_h == 8 and settings.pose_source == "target" and settings.lookahead_ms == 0
    assert settings.timeouts.home_s == 90.0 and settings.operator_script is None
    assert settings.operator_script_wait_s == 30.0
    assert settings.endpoints.commands.startswith("tcp://")
    assert settings.session_dir.name == "message_comms" and settings.session_dir.is_dir()
    keys = set(yaml.safe_load(CONFIG_FILE.read_text()))
    assert keys <= set(settings.__dataclass_fields__)


def test_station_override_merges_maps_key_by_key(make_settings):
    s = make_settings(exec_h=12, timeouts={"policy_hang_s": 2.5})
    assert s.exec_h == 12 and s.timeouts.policy_hang_s == 2.5
    assert s.timeouts.home_s == 90.0          # the rest of the map kept


def test_unknown_key_refused_with_its_path(tmp_path):
    station = write_station(tmp_path, {"exec_hh": 3, "timeouts": {"home": 1}})
    with pytest.raises(SettingsError) as e:
        load_settings(station, tmp_path / "s", False)
    text = str(e.value)
    assert "message_comms.exec_hh: not a key of message_comms/config.yaml" in text
    assert "message_comms.timeouts.home: not a key" in text


@pytest.mark.parametrize("override, words", [
    ({"exec_h": 0}, "exec_h: must be between 1 and 16"),
    ({"exec_h": 17}, "exec_h: must be between 1 and 16"),
    ({"exec_h": 2.5}, "exec_h: expected int"),
    ({"pose_source": "both"}, "pose_source: must be one of target, measured"),
    ({"lookahead_ms": -1}, "lookahead_ms: must be >= 0"),
    ({"odd_chunk_max_pos_m": 0}, "odd_chunk_max_pos_m: must be > 0"),
    ({"odd_chunk_max_rot_deg": -1.0}, "odd_chunk_max_rot_deg: must be > 0"),
    ({"anchor_fallback_m": 0}, "anchor_fallback_m: must be > 0"),
    ({"loop_ms": 0}, "loop_ms: must be > 0"),
    ({"status_ms": 0}, "status_ms: must be > 0"),
    ({"health_poll_ms": 0}, "health_poll_ms: must be > 0"),
    ({"history_s": 0}, "history_s: must be > 0"),
    ({"max_state_age_ms": 0}, "max_state_age_ms: must be > 0"),
    ({"state_lost_s": 0}, "state_lost_s: must be > 0"),
    ({"max_image_age_ms": 0}, "max_image_age_ms: must be > 0"),
    ({"state_recv_hwm": 0}, "state_recv_hwm: must be > 0"),
    ({"prompt_index": -1}, "prompt_index: must be >= 0"),
    ({"skip_warn_s": 0}, "skip_warn_s: must be > 0"),
    ({"operator_script_wait_s": 0}, "operator_script_wait_s: must be > 0"),
    ({"log_steps": "yes"}, "log_steps: expected bool"),
    ({"save_tap": 1}, "save_tap: expected bool"),
    ({"tap_images_every": 0}, "tap_images_every: must be >= 1"),
    ({"task_name": 5}, "task_name: expected str"),
    ({"timeouts": {"commands_s": 0}}, "timeouts.commands_s: must be > 0"),
    ({"timeouts": {"home_s": "long"}}, "timeouts.home_s: expected float"),
])
def test_range_checks(tmp_path, override, words):
    with pytest.raises(SettingsError) as e:
        load_settings(write_station(tmp_path, override), tmp_path / "s", False)
    assert words in str(e.value)


def test_every_problem_at_once(tmp_path):
    with pytest.raises(SettingsError) as e:
        load_settings(write_station(tmp_path, {"exec_h": 0, "loop_ms": 0}, endpoints={"frames": "x"}), None, False)
    assert len(e.value.problems) >= 6     # two values, three missing endpoints, a bad one


def test_no_session_makes_the_alone_folder(tmp_path):
    s = load_settings(write_station(tmp_path), None, True)
    assert s.mock is True
    assert s.session_dir.name == "message_comms"
    assert re.fullmatch(r"\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d_message_comms_alone", s.session_dir.parent.name)
    assert s.session_dir.parent.parent == (tmp_path / "sessions").resolve()


def test_operator_script_resolves_against_the_station_folder(tmp_path):
    s = load_settings(write_station(tmp_path, {"operator_script": "scripts/run.yaml"}), tmp_path / "s", False)
    assert s.operator_script == (tmp_path / "scripts/run.yaml").resolve()
