# adapted from rnoro/final-deployment:deployment/policy_deploy/tests/frames/test_settings.py @ 2ffe641
# changes: imports; config/frames_process.yaml of STRUCTURE.md 3.3 (no select spacing/tolerance keys, no probe, + record; server.poll_s 0.02); the R1 and U2 endpoints instead of C1; the test station's Pi is 127.0.0.1; dropped-key and record cases added
# adapted from rnoro/deployment:deployment/process_frames/tests/test_settings.py @ 91b178b
# changes: config/frames_process.yaml and the laptop station (station_helpers); the dropped keys (stream, decoder,
# clock.interval_s/timeout_s/startup_exchanges, h264.*, mock.keyframe_interval) are gone; the pair keys added; the Pi
# and C1 endpoints come from station ports.
"""Settings: config/frames_process.yaml + the station's frames section (STRUCTURE.md 4.4)."""

from __future__ import annotations

import pytest
import yaml
from frames_helpers import FOLDER, write_station

from teleop_collect.frames.errors import SettingsError
from teleop_collect.frames.settings import SCHEMA, Settings

# Every key of STRUCTURE.md 4.4 with its default.
DEFAULTS = {
    "rig_file": "rigs/rig_pi_v2_640.yaml", "mock_episode": "~/Downloads/episode_000045",
    "default_exposure_ns": 5000000, "receive.timeout_s": 0.1, "receive.recv_hwm": 60, "buffer.keep_s": 1.0,
    "select.pair_max_age_ms": 60, "select.pair_max_lead_ms": 25, "select.pair_max_reuse": 2,
    "health.stale_s": 0.35, "health.down_s": 2.0, "health.fps_window_s": 1.0,
    "clock.window_s": 60.0, "clock.best_fraction": 0.3, "clock.drift_min_span_s": 10.0, "clock.max_drift_ppm": 500.0,
    "clock.max_sync_age_s": 10.0, "server.poll_s": 0.02, "server.join_timeout_s": 2.0, "log.stats_every_s": 10.0,
    "record.queue_max_packets": 90, "record.clock_log_every_s": 1.0, "record.playback_fps": 30,
    "mock.max_seconds": 10.0, "mock.jpeg_quality": 90, "mock.pi_clock_offset_s": 1000.0, "mock.drift_ppm": 20.0,
    "mock.send_hwm": 30, "check.reachable_timeout_s": 1.0, "check.subprocess_timeout_s": 10.0,
}


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def test_config_file_has_every_key_with_its_default():
    flat = _flat(yaml.safe_load((FOLDER / "config" / "frames_process.yaml").read_text()))
    assert flat == DEFAULTS
    assert set(SCHEMA) == set(DEFAULTS)


def test_defaults_load(tmp_path):
    st = write_station(tmp_path / "station.yaml")
    s = Settings.load(st, mock=False)
    ports = yaml.safe_load(st.read_text())["ports"]
    assert s.rig_file == FOLDER / "rigs" / "rig_pi_v2_640.yaml"
    assert s.mock is False and s.record.queue_max_packets == 90 and s.record.playback_fps == 30
    assert (s.select.pair_max_age_ms, s.select.pair_max_lead_ms, s.select.pair_max_reuse) == (60.0, 25.0, 2)
    assert s.r1 == f"tcp://127.0.0.1:{ports['r1']}" and s.preview == f"tcp://127.0.0.1:{ports['preview']}"
    assert s.session_root == tmp_path / "sessions"
    assert s.pi_frames == "tcp://127.0.0.1:5555" and s.pi_address == "127.0.0.1"
    assert s.mock_pi_frames == f"tcp://127.0.0.1:{ports['mock_pi']}"


def test_override_merges_key_by_key(tmp_path):
    st = write_station(tmp_path / "station.yaml", frames={"clock": {"window_s": 30.0},
                                                          "select": {"pair_max_lead_ms": 20}})
    s = Settings.load(st, mock=False)
    assert s.clock.window_s == 30.0 and s.clock.best_fraction == 0.3
    assert s.select.pair_max_lead_ms == 20.0 and s.select.pair_max_age_ms == 60.0


def test_unknown_override_key_is_refused_with_its_path(tmp_path):
    st = write_station(tmp_path / "station.yaml", frames={"clock": {"intervl_s": 2.5}})
    with pytest.raises(SettingsError) as e:
        Settings.load(st, mock=False)
    assert any("frames.clock.intervl_s" in p and "not a key" in p for p in e.value.problems)


@pytest.mark.parametrize("override,key", [
    ({"clock": {"best_fraction": 1.5}}, "clock.best_fraction"),
    ({"receive": {"recv_hwm": "many"}}, "receive.recv_hwm"), ({"mock": {"jpeg_quality": 0}}, "mock.jpeg_quality"),
    ({"health": {"stale_s": True}}, "health.stale_s"), ({"health": {"down_s": 0.1}}, "health.down_s"),
    ({"select": {"pair_max_reuse": 0}}, "select.pair_max_reuse"),
    ({"select": {"pair_max_lead_ms": -1}}, "select.pair_max_lead_ms"),
    ({"select": {"pair_max_age_ms": "x"}}, "select.pair_max_age_ms"), ({"clock": 3}, "clock"),
    ({"record": {"queue_max_packets": 0}}, "record.queue_max_packets"),
    ({"record": {"clock_log_every_s": 0}}, "record.clock_log_every_s"),
    ({"record": {"playback_fps": 29.97}}, "record.playback_fps"),
])
def test_wrong_type_or_range_is_refused(tmp_path, override, key):
    st = write_station(tmp_path / "station.yaml", frames=override)
    with pytest.raises(SettingsError) as e:
        Settings.load(st, mock=False)
    assert any(key in p for p in e.value.problems), e.value.problems


@pytest.mark.parametrize("dropped", ["stream", "decoder", "h264", "pi_time_sync", "probe"])
def test_dropped_keys_are_refused(tmp_path, dropped):
    st = write_station(tmp_path / "station.yaml", frames={dropped: "x"})
    with pytest.raises(SettingsError) as e:
        Settings.load(st, mock=False)
    assert any(f"frames.{dropped}" in p for p in e.value.problems)


def test_relative_paths_resolve_from_the_right_folder(tmp_path):
    (tmp_path / "eps" / "ep1").mkdir(parents=True)
    st = write_station(tmp_path / "station.yaml", frames={"mock_episode": "eps/ep1"})
    s = Settings.load(st, mock=True)
    assert s.mock_episode == tmp_path / "eps" / "ep1"                      # station value: station's folder
    assert s.rig_file == FOLDER / "rigs" / "rig_pi_v2_640.yaml"            # default: deployment/teleop_collect/
    st = write_station(tmp_path / "station.yaml", frames={"rig_file": "my_rigs/r.yaml"})
    assert Settings.load(st, mock=False).rig_file == tmp_path / "my_rigs" / "r.yaml"


def test_mock_swaps_in_the_fake_pi_and_requires_mock_episode(tmp_path):
    st = write_station(tmp_path / "station.yaml", frames={"mock_episode": None})
    s = Settings.load(st, mock=False)
    assert s.pi_frames == "tcp://127.0.0.1:5555"
    with pytest.raises(SettingsError) as e:
        Settings.load(st, mock=True)
    assert any("mock_episode" in p and "required" in p for p in e.value.problems)
    (tmp_path / "ep").mkdir()
    st = write_station(tmp_path / "station.yaml", frames={"mock_episode": str(tmp_path / "ep")})
    s = Settings.load(st, mock=True)
    assert s.pi_frames == s.mock_pi_frames and s.pi_frames.startswith("tcp://127.0.0.1:") and s.pi_address is None
    st = write_station(tmp_path / "station.yaml", frames={"mock_episode": "nope"})
    with pytest.raises(SettingsError, match="not found"):
        Settings.load(st, mock=True)


def test_every_problem_is_listed_at_once(tmp_path):
    st = write_station(tmp_path / "station.yaml", frames={"bogus": 1, "buffer": {"keep_s": -1, "x": 2}})
    with pytest.raises(SettingsError) as e:
        Settings.load(st, mock=False)
    text = "\n".join(e.value.problems)
    for key in ("frames.bogus", "frames.buffer.x"):
        assert key in text
    assert len(e.value.problems) >= 2
