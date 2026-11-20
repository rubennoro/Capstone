# adapted from rnoro/final-deployment:deployment/policy_deploy/tests/contracts/test_config_files.py @ ea9f7fb
# changes: this package's five config files and their documented defaults (STRUCTURE.md 3.3); the C4 limits test
# dropped; + every key of STRUCTURE.md 3.3 present, the merge rule, resolve_path, fake_pc.yaml and the contracts rig
# file byte-identical to the source, ProcessArgs. + start pose (2026-11-20): fake_pc.yaml is the source's bytes plus
# an appended start pose block (p_pose, at_pose_tol_m, auto_h_p_after_stop_s); teleop.yaml start_pose: instead of
# home:.
"""The config files: every config/*.yaml loads, defines no contract key, holds exactly the keys of STRUCTURE.md 3.3,
and merges with a station section as config_merge says."""
from __future__ import annotations

from pathlib import Path

import pytest
from station_helpers import write_station

from teleop_collect.contracts import ProcessArgs, Station, StationError, fatal_line, load_config, paths, resolve_path
from teleop_collect.contracts.config_merge import read_config
from teleop_collect.contracts.process_args import PROCESS_NAMES
from teleop_collect.contracts.station import refused_keys

SOURCE = Path("~/Capstone-final-deployment/deployment/policy_deploy").expanduser()

#: Every key of STRUCTURE.md 3.3, as dotted paths (maps listed by their leaves).
KEYS = {
    "frames": """rig_file mock_episode default_exposure_ns receive.timeout_s receive.recv_hwm buffer.keep_s
        select.pair_max_age_ms select.pair_max_lead_ms select.pair_max_reuse health.stale_s health.down_s
        health.fps_window_s clock.window_s clock.best_fraction clock.drift_min_span_s clock.max_drift_ppm
        clock.max_sync_age_s server.poll_s server.join_timeout_s log.stats_every_s record.queue_max_packets
        record.clock_log_every_s record.playback_fps mock.max_seconds mock.jpeg_quality mock.pi_clock_offset_s
        mock.drift_ppm mock.send_hwm check.reachable_timeout_s check.subprocess_timeout_s""",
    "teleop": """motion.v_max_m_s motion.w_max_deg_s motion.slow_factor motion.waypoints motion.spacing_ms
        motion.key_stale_ms motion.state_max_age_ms motion.assumed_pc_box_margin_m motion.assumed_pc_cone_margin_deg
        motion.margin_use start_pose.tolerance_rad start_pose.tolerance_m history_s state_recv_hwm listener_poll_s
        link_jitter_allowance_s health_poll_ms u1_max_per_cycle at_limit_hold_ms messages_kept timeouts.health_s
        timeouts.frames_s timeouts.home_s timeouts.startup_s timeouts.pc_homing_wait_s clock.hello_period_s
        clock.startup_hello_period_s clock.round_trip.window_s clock.round_trip.best_fraction clock.one_way.window_s
        clock.one_way.best_fraction clock.drift_min_span_s clock.max_drift_ppm clock.min_samples_s clock.min_samples
        clock.max_sample_age_s clock.startup_agree_ms clock.jump_log_ms clock.mismatch_sets
        clock.mismatch_min_lead_ms clock.future_state_ms clock.log_every_s link_stats.window_s""",
    "ui": "status_hz preview_hz keys_resend_ms keys_stale_ms rpc_timeout_s open_browser",
    "launcher": """stop_timeout_s poll_s fake_pc ready_timeout_s.fake_pc ready_timeout_s.frames
        ready_timeout_s.teleop ready_timeout_s.ui""",
}


def leaves(d, prefix=""):
    out = set()
    for key, value in d.items():
        if isinstance(value, dict):
            out |= leaves(value, f"{prefix}{key}.")
        else:
            out.add(f"{prefix}{key}")
    return out


@pytest.mark.parametrize("name", paths.CONFIG_NAMES)
def test_every_config_loads_without_contract_keys(name):
    assert isinstance(load_config(name, None), dict)
    assert refused_keys(read_config(name)) == []


@pytest.mark.parametrize("name", sorted(KEYS))
def test_exactly_the_documented_keys(name):
    assert leaves(read_config(name)) == set(KEYS[name].split())


def test_paths_need_an_editable_install(monkeypatch, tmp_path):
    (tmp_path / "pkg").mkdir()
    monkeypatch.setattr(paths, "_package_file", lambda: tmp_path / "pkg" / "__init__.py")
    with pytest.raises(FileNotFoundError, match="installed editable from deployment/teleop_collect"):
        paths.folder()


def test_folders():
    assert paths.folder().name == "teleop_collect" and (paths.folder() / "teleop_collect").is_dir()
    assert paths.deployment_dir() == paths.folder().parent
    assert (paths.deployment_dir() / "contracts" / "link.yaml").is_file()
    assert (paths.deployment_dir() / "contracts" / "rigs" / "rig_pi_v3_640.yaml").is_file()
    assert paths.default_station() == paths.folder() / "station.yaml"


def test_documented_defaults():
    frames = load_config("frames", None)
    assert frames["select"] == {"pair_max_age_ms": 60.0, "pair_max_lead_ms": 25.0, "pair_max_reuse": 2}
    assert frames["record"] == {"queue_max_packets": 90, "clock_log_every_s": 1.0, "playback_fps": 30}
    assert frames["server"]["poll_s"] == 0.02 and frames["rig_file"] == "../contracts/rigs/rig_pi_v3_640.yaml"
    teleop = load_config("teleop", None)
    m = teleop["motion"]
    assert (m["v_max_m_s"], m["w_max_deg_s"], m["slow_factor"], m["waypoints"], m["spacing_ms"]) == (
        0.05, 10.0, 0.25, 6, 50)                   # wave 3: lead 300 ms (was 2 x 50 ms)
    assert (m["key_stale_ms"], m["state_max_age_ms"], m["margin_use"]) == (150, 250, 0.8)   # wave 3: 250 (was 50)
    # wave 3: the PC's teleop block (control_deploy/DIRECTIONS.md) sets the same margins (were 0.01 m, 2 deg)
    assert (m["assumed_pc_box_margin_m"], m["assumed_pc_cone_margin_deg"]) == (0.05, 10.0)
    assert teleop["start_pose"] == {"tolerance_rad": 0.01, "tolerance_m": 0.005} and "home" not in teleop
    assert teleop["timeouts"]["pc_homing_wait_s"] == 60.0 and teleop["timeouts"]["home_s"] == 90.0
    assert teleop["timeouts"]["frames_s"] == 4.0   # raised from 2.0 in the final review: frames' stop drains up to 1 + 2 s
    assert teleop["clock"]["min_samples"] == 10 and teleop["clock"]["mismatch_min_lead_ms"] == 60.0
    ui = load_config("ui", None)
    assert ui == {"status_hz": 10, "preview_hz": 15, "keys_resend_ms": 50, "keys_stale_ms": 150,
                  "rpc_timeout_s": 0.5, "open_browser": True}
    launcher = load_config("launcher", None)
    assert launcher["ready_timeout_s"] == {"fake_pc": 10.0, "frames": 30.0, "teleop": 150.0, "ui": 15.0}
    assert launcher["fake_pc"] is True


def test_copied_files_are_byte_identical():
    if not SOURCE.is_dir():
        pytest.skip("rnoro/final-deployment checkout not here")
    ours, source = paths.config_file("fake_pc").read_bytes(), (SOURCE / "config" / "fake_pc.yaml").read_bytes()
    assert ours.startswith(source)          # the source's file, then the start pose block appended (2026-11-20)
    assert ours[len(source):].startswith(b"# + start pose")
    rig = Path("contracts") / "rigs" / "rig_pi_v3_640.yaml"
    assert (paths.deployment_dir() / rig).read_bytes() == (SOURCE.parent / rig).read_bytes()


def test_fake_pc_start_pose_keys():
    fake_pc = load_config("fake_pc", None)
    assert fake_pc["p_pose"] == fake_pc["start_pose"]     # a mock session starts at the start pose
    assert fake_pc["at_pose_tol_m"] == 0.002 and fake_pc["auto_h_p_after_stop_s"] == 1.0


def test_merge_with_the_station(tmp_path):
    s = Station.load(write_station(tmp_path, sections={
        "teleop": {"motion": {"v_max_m_s": 0.02}, "history_s": 2.0},
        "frames": {"rig_file": "my_rig.yaml", "mock_episode": "/abs/episode"}}))
    teleop = load_config("teleop", s)
    assert teleop["motion"]["v_max_m_s"] == 0.02 and teleop["motion"]["w_max_deg_s"] == 10.0
    assert teleop["history_s"] == 2.0
    frames = load_config("frames", s)
    assert resolve_path(s, "frames", frames, "rig_file") == tmp_path / "my_rig.yaml"     # station: its folder
    assert resolve_path(s, "frames", frames, "mock_episode") == Path("/abs/episode")
    default = load_config("frames", None)
    assert resolve_path(None, "frames", default, "rig_file") == paths.folder() / ".." / "contracts" / "rigs" / "rig_pi_v3_640.yaml"
    assert resolve_path(None, "frames", default, "mock_episode") == Path("~/Downloads/data/incoming/episode_000152").expanduser()


def test_merge_problems_name_the_station_and_the_file(tmp_path):
    with pytest.raises(StationError) as e:
        Station.load(write_station(tmp_path, sections={"ui": {"status": 1}, "teleop": {"motion": 3}}))
    assert "station.yaml: ui.status: not a key of config/ui.yaml" in e.value.problems
    assert "station.yaml: teleop.motion: expected a map (as in config/teleop.yaml), got 3" in e.value.problems


def test_process_args(tmp_path):
    assert PROCESS_NAMES == ("frames", "teleop", "ui", "fake_pc")
    args, _ = ProcessArgs.parse("teleop", ["--mock", "--session", str(tmp_path / "run")])
    assert args.mock and args.station == paths.default_station() and args.name == "teleop"
    assert args.out_dir(tmp_path) == tmp_path / "run" / "teleop" and (tmp_path / "run" / "teleop").is_dir()
    alone, _ = ProcessArgs.parse("ui", ["--station", str(tmp_path / "s.yaml"), "--ready-file", str(tmp_path / "r")])
    assert alone.out_dir(tmp_path).name == "ui" and alone.out_dir(tmp_path).parent.name.endswith("_ui_alone")
    alone.write_ready()
    assert (tmp_path / "r").read_text() == "ready\n"
    for name in ("manager", "policy", "launcher"):
        with pytest.raises(ValueError):
            ProcessArgs.parse(name, [])
    assert fatal_line("frames", "main camera down", "check the Pi\n and ASUS_5g") == (
        "FATAL frames: main camera down: check the Pi and ASUS_5g")
