"""B6: launcher wireless integration (receiver + sim sender), sim/real profiles, task image-age guard, Track A needs.

Process tests (the sim-iphone-fake end-to-end run and the sender kill) need real_env, flask_socketio, sensor_rig and
data/raw/2026-10-01. Run under the suite lock with FD_TEST_PORT_OFFSET=2000:
    flock $LOCK env -u PYTHONPATH PYTHONPATH=. FD_TEST_PORT_OFFSET=2000 $RE -m pytest -q -s tests/test_launcher_wireless.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from franka_deploy.launch import launcher as lm
from franka_deploy.launch import preflight as pf
from franka_deploy.launch.launcher import Launcher, check_config, load_config

PKG = Path(__file__).resolve().parents[1]
OFFSET = int(os.environ.get("FD_TEST_PORT_OFFSET", "0"))
RAW = PKG.parents[1] / "data" / "raw" / "2026-10-01"


def _has_raw():
    try:
        lm.resolve_replay_episode({"episode_index": 0})
        return True
    except (OSError, ValueError):
        return False


def cfg_for(tmp_path, profile, ports=False, **over):
    """A profile with laptop python (this interpreter) and tmp work dirs; ports=True moves every port by +OFFSET+20
    (receiver ports via station_set) so a process test never collides with a running stack."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    ck = tmp_path / "m.ckpt"
    ck.write_bytes(b"x")
    z = tmp_path / "p.zarr.zip"
    z.write_bytes(b"x")
    cfg = load_config(PKG / "configs" / "deploy.yaml", profile)
    cfg.update(work_root=str(tmp_path / "runs"), python={"bp": sys.executable, "realenv": sys.executable})
    cfg.update(checkpoint=str(ck), prompt_zarr=str(z))
    if ports:
        o = OFFSET + 20
        cfg["station_set"] = list(cfg.get("station_set", [])) + [f"sensor.transport.ports.main={5555 + o}"] + (
            [f"sensor.transport.ports.ultrawide={5556 + o}"] if "iphone" in profile else [])
        cfg["camera"].update(endpoint=f"tcp://localhost:{16702 + o}", logger_endpoint=f"tcp://localhost:{25556 + o}")
        cfg["bridge"]["endpoint"] = f"tcp://127.0.0.1:{5565 + o}"
        cfg["controller"]["endpoint"] = f"tcp://localhost:{26010 + o}"
        cfg["inference"]["endpoint"] = f"tcp://0.0.0.0:{18765 + o}"
        cfg["spacemouse"]["endpoint"] = f"tcp://localhost:{15557 + o}"
    cfg.update(over)
    return cfg


# ------------------------------------------------------------------------------------------------ static / unit


def test_teardown_order_and_child_pythonpath(tmp_path):
    assert lm.TEARDOWN_ORDER == ("task", "controller", "sender", "camera", "inference", "spacemouse", "hand", "bridge")
    L = Launcher(cfg_for(tmp_path, "sim-iphone-fake"), interactive=False)
    pp = L.env()["PYTHONPATH"].split(os.pathsep)
    assert pp[0] == str(PKG) and (str(lm.SIM_ROOT) in pp) == lm.SIM_ROOT.is_dir()
    assert L.env()["PYTHONNOUSERSITE"] == "1" and L.env("task")["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONNOUSERSITE" not in L.env("camera") and "PYTHONNOUSERSITE" not in L.env("sender")
    M = Launcher(cfg_for(tmp_path / "m", "mock"), interactive=False)
    assert M.env("camera")["PYTHONNOUSERSITE"] == "1"          # no user_site_children: as before


def test_sim_iphone_fake_snapshot_camera_yaml_sender_and_ports(tmp_path):
    L = Launcher(cfg_for(tmp_path, "sim-iphone-fake"), interactive=False)
    L.snapshot()
    cfgd = L.session / "config"
    cam = yaml.safe_load((cfgd / "camera.yaml").read_text())
    assert cam == L.station.camera_config(L.cfg["camera"]["endpoint"], L.cfg["camera"]["logger_endpoint"])
    assert cam["bind_host"] == "127.0.0.1" and cam["camera_latency_s"] == 0.0
    assert yaml.safe_load((cfgd / "link.yaml").read_text())["id"] == "wifi_good"
    assert (cfgd / "sender.yaml").exists() and not (cfgd / "rig.yaml").exists()
    eps = L.endpoints()
    assert "tcp://127.0.0.1:5555" in eps and "tcp://127.0.0.1:5556" in eps and lm.SIM_STATE_ENDPOINT not in eps
    assert L.image_source() == "video_replay" and L.hand_endpoint() is None
    if _has_raw():
        cmd = L.sender_command()
        assert cmd[1:4] == ["-u", "-m", "franka_deploy.stream.iphone_replay_sender"]
        assert Path(cmd[cmd.index("--episode") + 1]) == lm.resolve_replay_episode(L.station.get("sim"))
        assert cmd[cmd.index("--ports") + 1] == "5555,5556" and cmd[cmd.index("--protocol") + 1] == "v1"
        assert cmd[cmd.index("--fps") + 1] == "30" and cmd[cmd.index("--size") + 1] == "960x720"
        assert cmd[cmd.index("--jpeg-quality") + 1] == "90" and "--loop" in cmd
        assert cmd[cmd.index("--link") + 1] == str(cfgd / "link.yaml")
        assert cmd[cmd.index("--ready-file") + 1] == str(L.session / "sender_ready.json")


def test_replay_episode_resolution(tmp_path):
    root = tmp_path / "raw"
    for n in ("b_firstreal_demonstration_left", "a_firstreal_demonstration_left", "c_firstreal_grippercalibration_left"):
        (root / n).mkdir(parents=True)
        (root / n / f"{n}_rgb.mp4").write_bytes(b"")
    assert lm.resolve_replay_episode({"raw_root": str(root)}).name == "a_firstreal_demonstration_left"
    assert lm.resolve_replay_episode({"raw_root": str(root), "episode_index": 1}).name.startswith("b_")
    assert lm.resolve_replay_episode({"raw_root": str(root), "episode_dir": "c_firstreal_grippercalibration_left"}) == \
        root / "c_firstreal_grippercalibration_left"
    with pytest.raises(FileNotFoundError):
        lm.resolve_replay_episode({"raw_root": str(root), "episode_index": 2})


def test_station_home_matches_the_fake_backend_start(tmp_path):
    """A station profile with a FakeBackend starts at the station home: FK(bridge start) . F_T_TCP = home.pose."""
    L = Launcher(cfg_for(tmp_path, "station-mock"), interactive=False)
    L.snapshot()
    bridge = yaml.safe_load((L.session / "config" / "bridge.yaml").read_text())
    ctrl = yaml.safe_load((L.session / "config" / "franka_cartesian.yaml").read_text())
    E = np.eye(4)
    E[:3, :3] = lm._quat_wxyz_to_R(bridge["home_O_T_EE_xyz_wxyz"][3:])
    E[:3, 3] = bridge["home_O_T_EE_xyz_wxyz"][:3]
    T = E @ np.asarray(ctrl["F_T_TCP"])
    home = L.cfg["home"]
    np.testing.assert_allclose(T[:3, 3], home["xyz"], atol=1e-5)
    np.testing.assert_allclose(T[:3, 3], L.station.get("robot.home_tcp.xyz_m"), atol=1e-5)
    np.testing.assert_allclose(lm._quat_wxyz_to_R(home["pose"][3:]), T[:3, :3], atol=5e-4)   # F_T_TCP R is rounded
    assert ctrl["home_tcp_xyz_wxyz"] == home["pose"] and L.summary["station_home"]["xyz"] == home["xyz"]
    R = Launcher(cfg_for(tmp_path / "r", "real-iphone"), interactive=False)   # hardware: xyz only, pose untouched
    assert R.cfg["home"]["xyz"] == [0.45, 0.0, 0.25] and R.cfg["home"]["pose"] is None
    R.snapshot()
    rb = yaml.safe_load((R.session / "config" / "bridge.yaml").read_text())
    assert rb["home_O_T_EE_xyz_wxyz"] == R.station.get("robot.bridge.home_O_T_EE_xyz_wxyz")


def test_hand_child_and_mujoco_ports(tmp_path):
    R = Launcher(cfg_for(tmp_path / "r", "real-iphone"), interactive=False)
    assert R.hand_endpoint() == "tcp://localhost:5566" and "tcp://localhost:5566" in R.endpoints()
    assert R.image_source() is None                                     # real: no sender
    S = Launcher(cfg_for(tmp_path / "s", "sim-iphone"), interactive=False)
    assert S.hand_endpoint() is None                                    # hand served by the bridge
    assert S.backend_type() == "mujoco" and lm.SIM_STATE_ENDPOINT in S.endpoints()
    assert S.image_source() == "render"
    if not lm.SENDERS["render"][1].is_file():
        with pytest.raises(lm.Abort, match="not available yet"):
            S.sender_command()


def _stats(t, transport="websocket", fps=30.0, age=0.12, reconnects=0, size=0, uw_transport=None):
    s = lambda tr: {"fps_in": fps, "frames": int(30 * t), "age_s": {"p50": age, "p95": age}, "reconnects": reconnects,  # noqa: E731
                    "size_mismatches": size, "transport": tr}
    return {"t": t, "fatal": None, "clock": {"mode": "arrival_minus_latency"},
            "streams": {"main": s(transport), "ultrawide": s(uw_transport or transport)}}


def _health(tmp_path, first, later):
    p = tmp_path / "camera_stats.json"
    p.write_text(json.dumps(first))
    seq = iter(later)

    def sleep(_):
        nxt = next(seq, None)
        if nxt is not None:
            p.write_text(json.dumps(nxt))
    return pf.wireless_health_ok(p, {"main": 30.0, "ultrawide": 30.0}, 0.3, ["main", "ultrawide"],
                                 upgrade_timeout_s=0.3, window_s=0.2, stats_period_s=0.05, sleep=lambda s: (sleep(s), time.sleep(0.01)))


def test_wireless_health_ok_passes_and_fails(tmp_path):
    ok, d = _health(tmp_path, _stats(1.0), [_stats(1.0 + 0.1 * i) for i in range(1, 50)])
    assert ok, d
    assert "transport websocket/websocket" in d and "age p95 120 ms" in d
    ok, d = _health(tmp_path, _stats(1.0, transport="polling"), [_stats(1.0 + 0.1 * i, transport="polling") for i in range(50)])
    assert not ok and "transport stuck on polling: install simple-websocket" in d
    ok, d = _health(tmp_path, _stats(1.0), [_stats(1.0 + 0.1 * i, fps=20.0) for i in range(1, 50)])
    assert not ok and "fps_in 20.0 < 0.9 x 30" in d
    ok, d = _health(tmp_path, _stats(1.0), [_stats(1.0 + 0.1 * i, reconnects=1) for i in range(1, 50)])
    assert not ok and "reconnected during the check" in d
    ok, d = _health(tmp_path, _stats(1.0), [_stats(1.0 + 0.1 * i, age=0.5) for i in range(1, 50)])
    assert not ok and "age p95" in d
    ok, d = _health(tmp_path, _stats(1.0), [_stats(1.0 + 0.1 * i, size=2) for i in range(1, 50)])
    assert not ok and "size mismatch" in d
    ok, d = _health(tmp_path, _stats(1.0), [])                          # file never updated again
    assert not ok and "stopped updating" in d
    assert not pf.wireless_health_ok(tmp_path / "none.json", {"main": 30.0}, 0.3, ["main"], upgrade_timeout_s=0.2)[0]


def test_clock_and_hello_checks():
    c = lambda off, rtt=3.0: {"clock": {"mode": "sender_stamp_ping", "offset_s": off, "rtt_ms_min": rtt, "drift_ppm": 1.0}}  # noqa: E731
    assert pf.clock_eval(c(12.3450), c(12.3455))[0]
    assert not pf.clock_eval(c(12.345), c(12.350))[0]                  # 5 ms offset jump
    assert not pf.clock_eval(c(12.345), c(12.345, rtt=25.0))[0]
    assert not pf.clock_eval({"clock": {"mode": "sender_stamp_ping"}}, {"clock": {"mode": "sender_stamp_ping"}})[0]
    assert pf.clock_eval({}, {"clock": {"mode": "arrival_minus_latency"}})[0]
    assert pf.hello_check([], "real")[0]
    ok, d = pf.hello_check(["main: K differs"], "sim")
    assert ok and "WARNING" in d
    assert not pf.hello_check(["main: K differs"], "real")[0]


def test_task_image_age_guard_raises_frame_selection_error(monkeypatch):
    pytest.importorskip("real_env")
    from franka_deploy.tasks import franka_task as ft
    from franka_deploy.tasks.frame_select import FrameSelectionError
    from franka_deploy.tasks.run_task import parser

    assert parser().parse_args(["--work-dir", "w"]).max_image_age_s is None
    assert parser().parse_args(["--work-dir", "w", "--max-image-age-s", "0.35"]).max_image_age_s == 0.35

    class Past(Exception):
        pass

    class Arx:
        def get_eef_xyz_wxyz(self, timestamp):
            raise Past()

    class Stub:
        policy_agent = type("PA", (), {"image_history_len": 2, "agent_update_freq_hz": 20.0, "cam_obs_freq_hz": None,
                                       "proprio_history_len": 2})()
        wrist_camera_client = None
        match_camera_latency = False
        arx5_client = Arx()

    def frames(age):
        t = time.monotonic() - age
        return lambda *a, **k: ({"main": np.zeros((2, 4, 4, 3)), "ultrawide": np.zeros((2, 4, 4, 3))},
                                {"main": np.array([t - 0.05, t]), "ultrawide": np.array([t - 0.1, t])})

    s = Stub()
    s.max_image_age_s, s.counters = 0.35, {"stale_image": 0}
    monkeypatch.setattr(ft, "get_downsampled_camera_images", frames(0.6))
    with pytest.raises(FrameSelectionError, match=r"image \d+ ms old \(max 350 ms\)"):
        ft.FrankaPromptTask.get_observations(s)
    assert s.counters["stale_image"] == 1
    monkeypatch.setattr(ft, "get_downsampled_camera_images", frames(0.05))
    with pytest.raises(Past):                                          # fresh: past the guard
        ft.FrankaPromptTask.get_observations(s)
    s.max_image_age_s = None                                           # default: no check at all
    monkeypatch.setattr(ft, "get_downsampled_camera_images", frames(5.0))
    with pytest.raises(Past):
        ft.FrankaPromptTask.get_observations(s)
    assert s.counters["stale_image"] == 1


def test_launcher_passes_max_image_age_from_the_station(tmp_path, monkeypatch):
    L = Launcher(cfg_for(tmp_path, "station-mock"), interactive=False)
    ta = L.station.task_args()
    assert ta["max_image_age_s"] == L.station.get("sensor.task.max_image_age_s") == 0.35


def _check(tmp_path, capsys, profile, **over):
    rc = check_config(cfg_for(tmp_path / profile, profile, **over))
    return rc, capsys.readouterr().out


def _line(out, name):
    hits = [ln for ln in out.splitlines() if ln[5:].startswith(name + ":")]
    assert hits, (name, out)
    return hits[0]


def test_check_config_new_profiles(tmp_path, capsys):
    if _has_raw():
        rc, out = _check(tmp_path, capsys, "sim-iphone-fake")
        assert rc == 0, out
        assert _line(out, "sender").startswith("OK") and "episode" in _line(out, "sender")
        assert _line(out, "camera_kind").startswith("OK") and "wireless" in out

    rc, out = _check(tmp_path, capsys, "sim-iphone")
    assert rc == 1
    mj = _line(out, "mujoco_backend")
    assert "franka_sim present" in mj and mj.startswith("OK" if lm.MUJOCO_BACKEND.is_file() else "FAIL")
    assert ("missing (A4)" in mj) != lm.MUJOCO_BACKEND.is_file()
    assert _line(out, "mujoco_import").startswith("OK")
    assert _line(out, "bridge_backend").startswith("OK   bridge_backend: mujoco")
    snd = _line(out, "sender")
    assert "render" in snd and ("not available yet (A7/B8)" in snd) != lm.SENDERS["render"][1].is_file()
    assert "hand: endpoint tcp://127.0.0.1:5565 = the bridge" in out and "hand_server:" not in out

    for prof in ("sim-rig-fake", "sim-rig"):
        rc, out = _check(tmp_path, capsys, prof)
        snd = _line(out, "sender")
        if prof == "sim-rig":    # B9: closed loop = rendered images (Stage 2b swaps in rig_raw_replay)
            assert "render" in snd and ("not available yet (A7/B8)" in snd) != lm.SENDERS["render"][1].is_file()
        else:
            assert "rig_raw_replay" in snd and ("not available yet (B7)" in snd) != lm.SENDERS["rig_raw_replay"][1].is_file()
        assert "iphumi_socketio_v2" in out and "sender_stamp_ping" in out

    for prof in ("real-iphone", "real-rig"):
        rc, out = _check(tmp_path, capsys, prof)
        assert rc == 1
        assert _line(out, "station_placeholders").startswith("FAIL") and "real mode" in out
        hs = _line(out, "hand_server")
        assert "tcp://localhost:5566" in hs and ("missing (A5)" in hs) != lm.HAND_SERVER.is_file()
        assert "real mode: no sender" in out and "sender:" not in out
        assert _line(out, "panda_py_versions_pinned").startswith("FAIL")


def test_check_config_wireless_without_station_fails(tmp_path, capsys):
    cfg = cfg_for(tmp_path, "mock")
    cfg["camera"]["kind"] = "wireless"
    assert check_config(cfg) == 1
    assert "FAIL camera_kind: wireless needs a station" in capsys.readouterr().out
    cfg["camera"]["kind"] = "usb"
    assert check_config(cfg) == 1
    assert "FAIL camera_kind" in capsys.readouterr().out


def test_sim_reset_script_action_and_flag_probe(tmp_path, monkeypatch):
    sent = []

    class FakeBridgeClient:
        def __init__(self, endpoint, timeout_s=0.005, **kw):
            self.endpoint = endpoint

        def request(self, topic, data=None, timeout_s=None):
            sent.append((self.endpoint, topic, data))
            return {"accepted": True}

    import franka_deploy.bridge.link as link

    monkeypatch.setattr(link, "BridgeClient", FakeBridgeClient)
    L = Launcher(cfg_for(tmp_path, "sim-iphone"), interactive=False,
                 script='0:sim_reset:{"eraser_xy_yaw": [0.5, 0.0, 0.3], "hand_gap_m": 0.08},0.05:sim_reset')
    L._done = False
    L._script_thread()
    assert sent == [("tcp://127.0.0.1:5565", "sim_reset", {"eraser_xy_yaw": [0.5, 0.0, 0.3], "hand_gap_m": 0.08}),
                    ("tcp://127.0.0.1:5565", "sim_reset", {})]
    assert [e["kind"] for e in L.events].count("sim_reset") == 2
    # --allow-sim-reset only when franka_bridge --help lists it (A6); otherwise skipped with a debug log
    flag = L.allow_sim_reset_flag()
    help_text = (PKG / "franka_deploy" / "bridge" / "franka_bridge.py").read_text()
    assert flag == (["--allow-sim-reset"] if "--allow-sim-reset" in help_text else [])
    if not flag:
        assert any(e["kind"] == "debug" and "allow-sim-reset" in e["msg"] for e in L.events)


# ------------------------------------------------------------------------------------------------ process tests


def _proc_skip():
    pytest.importorskip("real_env")
    pytest.importorskip("flask_socketio")
    pytest.importorskip("sensor_rig")
    if not _has_raw():
        pytest.skip("data/raw/2026-10-01 firstreal episodes missing")


def _no_strays(L):
    for ch in L.children.values():
        assert not ch.alive(), ch.name
        assert not os.path.exists(f"/proc/{ch.p.pid}") or open(f"/proc/{ch.p.pid}/stat").read().split()[2] == "Z"


def _summary_numbers(L):
    st = L.summary.get("camera_stats") or {}
    ts = json.loads((L.session / "task" / "task_summary.json").read_text())
    m, u = st["streams"]["main"], st["streams"]["ultrawide"]
    return {"fps_in": [round(m["fps_in"], 1), round(u["fps_in"], 1)], "publish_fps": round(st["publish_fps"], 1),
            "age_p95_ms": round(1000 * m["age_s"]["p95"], 1), "clock": st["clock"]["mode"],
            "transport": [m["transport"], u["transport"]], "reconnects": [m["reconnects"], u["reconnects"]],
            "stale_ticks": st["stale_ticks"], "policy_calls": ts["counters"]["policy_calls"],
            "holds": ts["counters"]["holds"], "stale_image": ts["counters"]["stale_image"],
            "obs_retries": ts["counters"]["obs_retries"], "ready_s": L.summary.get("ready_s"),
            "wireless_health_ok": L.summary["preflight"]["wireless_health_ok"]["detail"]}


def test_sim_iphone_fake_end_to_end(tmp_path, monkeypatch):
    """iPhone replay -> emulated Wi-Fi (wifi_good) -> receiver -> task -> fake policy -> FakeBackend, clean teardown."""
    _proc_skip()
    L = Launcher(cfg_for(tmp_path, "sim-iphone-fake", ports=True), interactive=False, script="6:c,20:s,24:q")
    rc = L.run()
    kinds = [e["kind"] for e in L.events]
    assert "READY" in kinds, [e for e in L.events if e["kind"] in ("abort", "exit", "preflight")][-6:]
    pre = L.summary["preflight"]
    assert pre["wireless_health_ok"]["ok"] and pre["websocket_available"]["ok"] and pre["sender_available"]["ok"]
    assert pre["camera_rate_and_selection"]["ok"] and pre["fk_near_home"]["ok"]
    assert rc == 0 and L.exit_reason == "quit", L.exit_reason
    nums = _summary_numbers(L)
    print("B6_E2E", json.dumps(nums))
    (tmp_path / "e2e_summary.json").write_text(json.dumps(nums, indent=1))
    assert nums["policy_calls"] > 0 and nums["transport"] == ["websocket", "websocket"]
    task_cmd = (L.session / "logs" / "task.log").read_text().splitlines()[0]
    assert "--max-image-age-s 0.35" in task_cmd                       # station sensor.task.max_image_age_s
    run = L.session / "task" / "episodes" / "franka_deploy" / "block_raise" / "run"
    eps = sorted(run.iterdir())
    assert eps and list((eps[0] / "tap").glob("*.npz")), eps
    ready = json.loads((L.session / "sender_ready.json").read_text())
    assert ready["first_tick_ns"] > ready["start_now_ns"] and ready["first_slot"] >= 0
    assert (L.session / "logs" / "camera_truth.jsonl").stat().st_size > 0
    assert (L.session / "logs" / "sender_truth.jsonl").stat().st_size > 0
    assert L.summary["ports_free_after"][0], L.summary["ports_free_after"]
    assert L.summary["shm_clean_after"][0], L.summary["shm_clean_after"]
    assert set(L.summary["child_exit_codes"]) == {"task", "controller", "sender", "camera", "inference", "spacemouse", "bridge"}
    _no_strays(L)


def test_kill_sender_mid_episode_holds_without_crash(tmp_path):
    _proc_skip()
    L = Launcher(cfg_for(tmp_path, "sim-iphone-fake", ports=True), interactive=False, script="6:c,11:kill:sender,17:q")
    rc = L.run()
    kinds = [e["kind"] for e in L.events]
    assert "READY" in kinds and "sender_died" in kinds and "launcher_error" not in kinds, kinds
    assert any(e["kind"] == "hold_sent" and e["reason"] == "sender died" for e in L.events)
    assert rc == 0 and L.exit_reason == "quit", L.exit_reason
    nums = _summary_numbers(L)
    print("B6_KILL", json.dumps(nums))
    assert nums["stale_image"] > 0 or nums["stale_ticks"] > 0, nums
    assert nums["holds"] > 0, nums
    ts = json.loads((L.session / "task" / "task_summary.json").read_text())
    assert not any(e["kind"] == "policy_exception" and "Traceback" in str(e.get("error")) for e in ts["events"])
    assert L.summary["ports_free_after"][0] and L.summary["shm_clean_after"][0]
    _no_strays(L)


def test_split_script_keeps_json_actions():
    assert lm.split_script("6:c,20:s,24:q") == ["6:c", "20:s", "24:q"]
    assert lm.split_script("1.5:kill:sender,2:note:a b") == ["1.5:kill:sender", "2:note:a b"]
    assert lm.split_script('0:sim_reset:{"a": [1, 2], "b": 3},4:q') == ['0:sim_reset:{"a": [1, 2], "b": 3}', "4:q"]
