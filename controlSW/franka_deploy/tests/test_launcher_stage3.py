"""B9 launcher items: task --v-max / --w-max from criteria.limits (one source with the bridge clamp), the sim state endpoint in the
snapshot, render sender args, the iPhone capture-latency sender flags, the sim_reset timeout. Station-less goldens stay in
test_launcher_station.py (unchanged)."""
import sys
from pathlib import Path

import pytest
import yaml

from franka_deploy.launch import launcher as lm
from franka_deploy.launch.launcher import Launcher, load_config

PKG = Path(__file__).resolve().parents[1]


def _cfg(tmp_path, profile, sets=(), **top):
    cfg = load_config(PKG / "configs" / "deploy.yaml", profile)
    cfg.update(work_root=str(tmp_path / "runs"), python={"bp": sys.executable, "realenv": sys.executable}, **top)
    cfg["station_set"] = list(cfg.get("station_set", [])) + list(sets)
    return cfg


def _task_cmd(L):
    c = L.cfg
    return L.task_command(sys.executable, c["camera"], c["spacemouse"], L.session / "logs", L.session / "task_ready.json")


def _flag(cmd, name):
    return cmd[cmd.index(name) + 1]


def test_task_vmax_follows_the_criteria_limits_set(tmp_path):
    L = Launcher(_cfg(tmp_path, "sim-iphone-fake", ["criteria.limits.v_max_m_s=0.5", "criteria.limits.w_max_rad_s=0.8"]), interactive=False)
    cmd = _task_cmd(L)
    assert _flag(cmd, "--v-max") == "0.5" and _flag(cmd, "--w-max") == "0.8"
    # the bridge clamp comes from the same value: no split between the task and the bridge
    assert L.station.bridge_fragment()["safety"]["v_max"] == 0.5
    v, w, contradictions = lm.task_limits(L.cfg, L.station)
    assert (v, w) == (0.5, 0.8)
    assert contradictions == []                                  # sim profiles leave task.v_max null: the station decides
    L.snapshot()
    assert yaml.safe_load((L.session / "config" / "bridge.yaml").read_text())["safety"]["v_max"] == 0.5


def test_limits_are_the_minimum_of_their_sources_and_mismatches_warn(tmp_path, capsys):
    # real-* keep 0.25 / 1.0
    R = Launcher(_cfg(tmp_path / "r", "real-iphone"), interactive=False)
    assert lm.task_limits(R.cfg, R.station) == (0.25, 1.0, [])
    # deploy task.v_max lower than the station: the lower one wins, loudly
    L = Launcher(_cfg(tmp_path, "sim-iphone-fake"), interactive=False)
    L.cfg["task"]["v_max"] = 0.1
    v, w, c = lm.task_limits(L.cfg, L.station)
    assert (v, w) == (0.1, 1.0) and c[0]["used"] == 0.1 and _flag(_task_cmd(L), "--v-max") == "0.1"
    L.snapshot()
    out = capsys.readouterr().out
    assert "WARNING: limit mismatch task v_max" in out and any(e["kind"] == "warning" for e in L.events)
    # station lower than the deploy value: the station wins
    L2 = Launcher(_cfg(tmp_path / "b", "sim-iphone-fake", ["criteria.limits.v_max_m_s=0.1"]), interactive=False)
    L2.cfg["task"]["v_max"] = 0.25
    assert lm.task_limits(L2.cfg, L2.station)[0] == 0.1
    # the bridge clamp: min(station, bridge.override.safety)
    L3 = Launcher(_cfg(tmp_path / "c", "sim-iphone-fake"), interactive=False)
    L3.cfg["bridge"]["override"] = {**L3.cfg["bridge"]["override"], "safety": {"v_max": 0.1}}
    b, ov = lm.build_bridge_cfg(L3.cfg, L3.station)
    assert b["safety"]["v_max"] == 0.1 and any(o.get("used") == 0.1 and o["override"] == 0.1 for o in ov)
    b2, _ = lm.build_bridge_cfg(L.cfg, L.station)
    assert b2["safety"]["v_max"] == 0.25


def test_task_vmax_default_station_equals_deploy_value_and_stationless_is_untouched(tmp_path):
    L = Launcher(_cfg(tmp_path, "sim-iphone-fake"), interactive=False)
    assert _flag(_task_cmd(L), "--v-max") == "0.25" and lm.task_limits(L.cfg, L.station)[2] == []     # null task.v_max -> station
    M = Launcher(_cfg(tmp_path / "m", "mock"), interactive=False)          # no station: deploy.yaml task.v_max, verbatim
    M.cfg["task"]["v_max"], M.cfg["task"]["w_max"] = 1, 2
    assert lm.task_limits(M.cfg, None) == (1, 2, []) and _flag(_task_cmd(M), "--v-max") == "1"


def test_state_endpoint_in_the_snapshot_for_a_mujoco_profile(tmp_path):
    L = Launcher(_cfg(tmp_path, "sim-iphone", sim_state_endpoint="tcp://127.0.0.1:7570"), interactive=False)
    L.snapshot()
    b = yaml.safe_load((L.session / "config" / "bridge.yaml").read_text())
    assert b["backend"]["type"] == "mujoco" and b["backend"]["mujoco"]["state_endpoint"] == "tcp://127.0.0.1:7570"
    assert b["backend"]["mujoco"]["station"].endswith("station_resolved.yaml")
    assert "tcp://127.0.0.1:7570" in L.endpoints()                         # preflight checks / teardown see the offset port
    D = Launcher(_cfg(tmp_path / "d", "sim-iphone"), interactive=False)
    D.snapshot()
    assert yaml.safe_load((D.session / "config" / "bridge.yaml").read_text())["backend"]["mujoco"]["state_endpoint"] == "tcp://127.0.0.1:5570"
    F = Launcher(_cfg(tmp_path / "f", "sim-iphone-fake"), interactive=False)      # a fake backend has no state PUB entry
    F.snapshot()
    assert "state_endpoint" not in (yaml.safe_load((F.session / "config" / "bridge.yaml").read_text()).get("backend", {}).get("mujoco") or {})


def test_render_sender_gets_state_endpoint_and_station(tmp_path, monkeypatch):
    L = Launcher(_cfg(tmp_path, "sim-iphone", sim_state_endpoint="tcp://127.0.0.1:7570"), interactive=False)
    L.snapshot()
    fake = tmp_path / "render_sender.py"
    fake.write_text("")
    monkeypatch.setitem(lm.SENDERS, "render", ("franka_sim.senders.render_sender", fake, "A7/B8"))
    cmd = L.sender_command()
    assert L.image_source() == "render" and cmd[cmd.index("-m") + 1] == "franka_sim.senders.render_sender"
    assert _flag(cmd, "--state-endpoint") == "tcp://127.0.0.1:7570"
    assert _flag(cmd, "--station") == str(L.session / "config" / "station_resolved.yaml")
    assert "--config" in cmd and "--capture-latency-s" not in cmd


def test_iphone_sender_models_the_capture_latency_instead_of_the_link(tmp_path):
    L = Launcher(_cfg(tmp_path, "sim-iphone-fake"), interactive=False)
    L.snapshot()
    cmd = L.sender_command()
    assert L.image_source() == "video_replay"
    assert _flag(cmd, "--capture-latency-s") == "0.08" and _flag(cmd, "--capture-latency-uw-s") == "0.08"
    assert L.station.get("link.latency_s") < 0.08                            # the link keeps its own (Wi-Fi) latency
    # the rig is stamped (v2): nothing to model
    R = Launcher(_cfg(tmp_path / "r", "sim-rig-fake"), interactive=False)
    R.snapshot()
    assert "--capture-latency-s" not in R.sender_command() if R.image_source() == "rig_raw_replay" else True


def test_sim_reset_timeout_is_configurable_and_longer_than_before(tmp_path):
    L = Launcher(_cfg(tmp_path, "sim-iphone"), interactive=False)
    assert L.sim_reset_timeout_s() == 2.0
    L.cfg["bridge"]["sim_reset_timeout_s"] = 4
    assert L.sim_reset_timeout_s() == 4.0


def test_sim_profiles_declare_the_state_endpoint_and_allow_reset():
    cfg = yaml.safe_load((PKG / "configs" / "deploy.yaml").read_text())
    for name in ("sim-iphone", "sim-rig"):
        p = load_config(PKG / "configs" / "deploy.yaml", name)
        assert p["sim_state_endpoint"] == "tcp://127.0.0.1:5570" and p["bridge"]["allow_sim_reset"] is True
        assert p["run"]["sim"] == "render"
    assert load_config(PKG / "configs" / "deploy.yaml", "sim-rig-fake")["run"]["sim"] == "rig_raw_replay"
