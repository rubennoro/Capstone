"""Settings: config.yaml + the station's process_frames section (STRUCTURE.md §5.2, §10)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from conftest import write_station
from process_frames import FOLDER
from process_frames.errors import SettingsError
from process_frames.settings import SCHEMA, Settings

# Every key of STRUCTURE.md §10 with its default.
DEFAULTS = {
    "rig_file": "rigs/rig_pi_v2.yaml", "decoder": "h264", "mock_episode": None,
    "default_exposure_ns": 5000000, "receive.timeout_s": 0.1, "receive.recv_hwm": 60, "buffer.keep_s": 1.0,
    "select.main_spacing_ms": 50, "select.ultrawide_spacing_ms": 100, "select.tolerance_frames": 2.0,
    "health.stale_s": 0.35, "health.down_s": 2.0, "health.fps_window_s": 1.0, "clock.interval_s": 1.0,
    "clock.timeout_s": 0.5, "clock.startup_exchanges": 10, "clock.window_s": 60.0, "clock.best_fraction": 0.3,
    "clock.drift_min_span_s": 10.0, "clock.max_drift_ppm": 500.0, "clock.max_sync_age_s": 10.0,
    "h264.thread_type": "SLICE", "server.poll_s": 0.05, "server.join_timeout_s": 2.0, "log.stats_every_s": 10.0,
    "mock.max_seconds": 10.0, "mock.keyframe_interval": 30, "mock.jpeg_quality": 90,
    "mock.pi_clock_offset_s": 1000.0, "mock.drift_ppm": 20.0, "mock.send_hwm": 30,
    "check.reachable_timeout_s": 1.0, "check.subprocess_timeout_s": 10.0, "probe.timeout_s": 2.0,
}


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def test_config_yaml_has_every_key_with_its_default():
    flat = _flat(yaml.safe_load((FOLDER / "config.yaml").read_text()))
    assert flat == DEFAULTS
    assert set(SCHEMA) == set(DEFAULTS)


def test_defaults_load(tmp_path):
    s = Settings.load(write_station(tmp_path / "station.yaml"), mock=False)
    assert s.rig_file == FOLDER / "rigs" / "rig_pi_v2.yaml"
    assert s.decoder == "h264" and s.mock is False and s.mock_episode is None
    assert s.clock.startup_exchanges == 10 and isinstance(s.clock.interval_s, float)
    assert s.select.tolerance_frames == 2.0 and s.h264.thread_type == "SLICE"
    assert s.frames == "tcp://127.0.0.1:*" and s.session_root == tmp_path / "sessions"


def test_override_merges_key_by_key(tmp_path):
    st = write_station(tmp_path / "station.yaml", process_frames={"clock": {"interval_s": 2.5}, "decoder": "jpeg"})
    s = Settings.load(st, mock=False)
    assert s.clock.interval_s == 2.5 and s.clock.timeout_s == 0.5 and s.decoder == "jpeg"


def test_unknown_override_key_is_refused_with_its_path(tmp_path):
    st = write_station(tmp_path / "station.yaml", process_frames={"clock": {"intervl_s": 2.5}})
    with pytest.raises(SettingsError) as e:
        Settings.load(st, mock=False)
    assert any("process_frames.clock.intervl_s" in p and "not a key" in p for p in e.value.problems)


@pytest.mark.parametrize("override,key", [
    ({"decoder": "vp9"}, "decoder"), ({"clock": {"best_fraction": 1.5}}, "clock.best_fraction"),
    ({"receive": {"recv_hwm": "many"}}, "receive.recv_hwm"), ({"mock": {"jpeg_quality": 0}}, "mock.jpeg_quality"),
    ({"health": {"stale_s": True}}, "health.stale_s"), ({"health": {"down_s": 0.1}}, "health.down_s"),
    ({"h264": {"thread_type": "FRAME"}}, "h264.thread_type"), ({"clock": 3}, "clock"),
])
def test_wrong_type_or_range_is_refused(tmp_path, override, key):
    st = write_station(tmp_path / "station.yaml", process_frames=override)
    with pytest.raises(SettingsError) as e:
        Settings.load(st, mock=False)
    assert any(key in p for p in e.value.problems), e.value.problems


def test_relative_paths_resolve_from_the_right_folder(tmp_path):
    (tmp_path / "eps" / "ep1").mkdir(parents=True)
    st = write_station(tmp_path / "station.yaml", process_frames={"mock_episode": "eps/ep1"})
    s = Settings.load(st, mock=True)
    assert s.mock_episode == tmp_path / "eps" / "ep1"                      # station value: station's folder
    assert s.rig_file == FOLDER / "rigs" / "rig_pi_v2.yaml"                # default: process_frames/
    st = write_station(tmp_path / "station.yaml", process_frames={"rig_file": "my_rigs/r.yaml"})
    assert Settings.load(st, mock=False).rig_file == tmp_path / "my_rigs" / "r.yaml"


def test_mock_swaps_in_the_mock_endpoints_and_requires_mock_episode(tmp_path):
    ep = {"pi_frames": "tcp://192.168.1.50:5600", "pi_time_sync": "tcp://192.168.1.50:5601",
          "mock_pi_frames": "tcp://127.0.0.1:5700", "mock_pi_time_sync": "tcp://127.0.0.1:5701"}
    st = write_station(tmp_path / "station.yaml", endpoints=ep)
    s = Settings.load(st, mock=False)
    assert (s.pi_frames, s.pi_time_sync) == ("tcp://192.168.1.50:5600", "tcp://192.168.1.50:5601")
    with pytest.raises(SettingsError) as e:
        Settings.load(st, mock=True)
    assert any("mock_episode" in p and "required" in p for p in e.value.problems)
    (tmp_path / "ep").mkdir()
    st = write_station(tmp_path / "station.yaml", endpoints=ep, process_frames={"mock_episode": str(tmp_path / "ep")})
    s = Settings.load(st, mock=True)
    assert (s.pi_frames, s.pi_time_sync) == ("tcp://127.0.0.1:5700", "tcp://127.0.0.1:5701")
    st = write_station(tmp_path / "station.yaml", endpoints=ep, process_frames={"mock_episode": "nope"})
    with pytest.raises(SettingsError, match="not found"):
        Settings.load(st, mock=True)


def test_every_problem_is_listed_at_once(tmp_path):
    st = write_station(tmp_path / "station.yaml", endpoints={"frames": "localhost:5610"},
                       process_frames={"decoder": "vp9", "bogus": 1, "buffer": {"keep_s": -1}})
    with pytest.raises(SettingsError) as e:
        Settings.load(st, mock=False)
    text = "\n".join(e.value.problems)
    for key in ("process_frames.bogus", "decoder", "buffer.keep_s", "endpoints.frames"):
        assert key in text
    assert len(e.value.problems) >= 4
