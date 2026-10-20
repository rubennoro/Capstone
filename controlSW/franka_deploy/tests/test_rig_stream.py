"""B7: the rig's wireless stream. rig_replay_sender (sensor_rig RigStreamSender, mock backend, replaying a simulated raw
rig episode) -> LinkEmulator (wifi_good_skew: sender clock +12.345 s, 30 ppm drift) -> WirelessCamera (v2,
sender_stamp_ping, 60 s ping window); plus `sensor-rig stream-send` (the default socket.io transport) and the
sim-rig-fake launcher profile end to end.

The raw rig episode is simulated once from one short firstreal demo and cached (FD_TEST_CACHE, default
<tempdir>/franka_deploy_test_cache). Process tests need real_env, flask_socketio, sensor_rig and data/raw/2026-10-01.
Run under the suite lock with FD_TEST_PORT_OFFSET=2000:
    flock $LOCK env -u PYTHONPATH PYTHONPATH=. FD_TEST_PORT_OFFSET=2000 $RE -m pytest -q -s tests/test_rig_stream.py
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("real_env")
pytest.importorskip("flask_socketio")
pytest.importorskip("sensor_rig")

from test_launcher_wireless import PKG, _no_strays, _proc_skip, _summary_numbers, cfg_for  # noqa: E402
from test_wireless_camera import PORTS, Receiver, _env, _join, _ms, camera_yaml  # noqa: E402

from franka_deploy.launch.launcher import Launcher  # noqa: E402
from franka_deploy.stream import rig_replay_sender as rrs  # noqa: E402

CACHE = Path(os.environ.get("FD_TEST_CACHE") or Path(tempfile.gettempdir()) / "franka_deploy_test_cache")
SENSOR_RIG = PKG.parents[1] / "sensor_rig"


def _sim_cfg():
    return yaml.safe_load((PKG.parents[1] / "controlSW" / "configs" / "run" / "sim" / "rig_raw_replay.yaml").read_text())


def _camera_yaml_with_sender_K(tmp, raw_episode, **kw):
    """The receiver config of sim-rig-fake. Its K is the station yaml's (= the iPhone deploy K = what the rig sends), so
    the receiver's hello check passes without overriding anything (see test_hello_K_vs_station_yaml)."""
    return camera_yaml(tmp, profile="sim-rig-fake", **kw)


@pytest.fixture(scope="module")
def raw_episode():
    """A short simulated raw rig episode (cached); skipped without the iPhone data."""
    gen = {**rrs.DEFAULT_GENERATE, **_sim_cfg()["generate"], "n_frames": 240}   # run/sim/rig_raw_replay.yaml, shorter
    try:
        t0 = time.monotonic()
        ep = rrs.generate_raw_episode(gen, CACHE / "rig_episodes")
    except (FileNotFoundError, OSError) as e:
        pytest.skip(f"iPhone data for the simulated rig episode missing: {e}")
    print(f"\nrig episode {ep} ({time.monotonic() - t0:.1f} s)")
    return ep


def test_episode_generation_is_cached_and_resolution_order(raw_episode, tmp_path):
    from sensor_rig.raw_episode import RawEpisode

    t0 = time.monotonic()
    again = rrs.generate_raw_episode({**rrs.DEFAULT_GENERATE, **_sim_cfg()["generate"], "n_frames": 240},
                                     CACHE / "rig_episodes")
    assert again == raw_episode and time.monotonic() - t0 < 2.0
    ep = RawEpisode(raw_episode)
    assert len(ep.camera_times("main")) > 100 and len(ep.camera_times("ultrawide")) > 20
    # an explicit --episode wins over the config; a config raw_episode_dir wins over generation
    assert rrs.resolve_episode(str(raw_episode), {"sim": {"raw_episode_dir": "/nonexistent"}}, None) == raw_episode
    assert rrs.resolve_episode("None", {"sim": {"raw_episode_dir": str(raw_episode)}}, None) == raw_episode
    with pytest.raises(FileNotFoundError):
        rrs.resolve_episode(str(tmp_path), {}, None)


def _run_sender(tmp, tag, args, seconds):
    log = open(Path(tmp) / f"tx_{tag}.log", "w")
    p = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=_env(), cwd=str(SENSOR_RIG))
    return p, log


def _finish(p, log, timeout=60):
    try:
        p.wait(timeout)
    finally:
        if p.poll() is None:
            p.kill()
            p.wait(5)
        log.close()
    text = Path(log.name).read_text()
    assert p.returncode == 0, text[-3000:]
    return text


def test_rig_stamp_error_over_skewed_wifi(raw_episode, tmp_path):
    """wifi_good_skew: after warm-up the receiver's stamp error against the sender's truth is <= 2 ms (p95); health
    shows protocol v2, hello OK, websocket, no sequence gaps."""
    cfg_path, cfg = _camera_yaml_with_sender_K(tmp_path, raw_episode, clock={"ping_window_s": 60.0})
    assert cfg["protocol"] == "iphumi_socketio_v2" and cfg["clock"]["mode"] == "sender_stamp_ping"
    assert cfg["clock"]["ping_window_s"] == 60.0
    rx = Receiver(tmp_path, cfg_path, [PORTS["main"]])
    truth, ready = tmp_path / "tx_truth.jsonl", tmp_path / "ready.json"
    tx = log = None
    try:
        tx, log = _run_sender(tmp_path, "rig", [
            sys.executable, "-u", "-m", "franka_deploy.stream.rig_replay_sender", "--episode", str(raw_episode),
            "--host", "127.0.0.1", "--ports", str(PORTS["main"]), "--fps", "30", "--link", "run/link/wifi_good_skew.yaml",
            "--truth", str(truth), "--ready-file", str(ready), "--loop", "--max-s", "16"], 16)
        text = _finish(tx, log)
        time.sleep(0.8)
        st = rx.stats()
        rxt, txt = rx.truth(), [json.loads(x) for x in truth.read_text().splitlines() if x.strip()]
    finally:
        if tx is not None and tx.poll() is None:
            tx.kill()
        rx.stop()
    done = json.loads(text.rsplit("[rig-replay] done ", 1)[1])
    rd = json.loads(ready.read_text())
    assert rd["protocol"] == "v2" and rd["fps"] == 30.0 and rd["first_tick_ns"] > rd["start_now_ns"] - 1_000_000_000
    assert st["hellos"] == 1 and st["hello_problems"] == [], st["hello_problems"]
    assert st["streams"]["main"]["transport"] == "websocket" and st["streams"]["main"]["seq_gaps"] == 0
    assert st["packet_errors"] == 0 and st["fatal"] is None
    c = st["clock"]
    assert c["n_pings"] >= 15 and abs(c["offset_s"] - 12.345) < 0.01, c
    pairs = _join(rxt, txt)
    t0 = min(t for _, t in pairs)
    res = {}
    for s in ("main", "ultrawide"):
        e = np.array([r["t_est"] - t for r, t in pairs if r["stream"] == s and t >= t0 + 5.0])
        p50, p95 = _ms(np.median(np.abs(e))), _ms(np.percentile(np.abs(e), 95))
        res[s] = (p50, p95, len(e))
        print(f"\nB7_STAMP {s}: stamp err after 5 s p50 {p50:.3f} / p95 {p95:.3f} ms (n={len(e)}); drift est "
              f"{c['drift_ppm']:.1f} ppm, rtt_min {c['rtt_ms_min']:.2f} ms")
        assert len(e) >= 100
        assert p95 < 2.0
    print("B7_SENDER", json.dumps({k: done[k] for k in ("captured", "sent", "dropped", "max_queue_len", "queue_wait_ms",
                                                         "send_fps", "decimated", "repeats_skipped")}))
    assert done["sent"] > 300 and done["max_queue_len"] <= 4


def test_hello_K_vs_station_yaml(raw_episode):
    """The rig streams VIRTUAL-iPhone images, so the sender hello K == station rig1_wifi.yaml K == the iPhone deploy K
    (iphone16pro_wifi.yaml), for both streams, and the receiver's hello check finds no problem."""
    from sensor_rig.raw_episode import RawEpisode
    from sensor_rig.rig_config import load_rig
    from sensor_rig.stream.sender import RigStreamSender

    sdir = PKG.parents[1] / "controlSW" / "configs" / "station" / "sensor"
    rig_y = yaml.safe_load((sdir / "rig1_wifi.yaml").read_text())["streams"]
    iph_y = yaml.safe_load((sdir / "iphone16pro_wifi.yaml").read_text())["streams"]
    for rig in (RawEpisode(raw_episode).rig(), load_rig("rig_default.yaml")):
        h = RigStreamSender(rig, "127.0.0.1", 1, transport=object()).hello()
        assert h["protocol"] == "v2"
        for name in ("main", "ultrawide"):
            assert h["streams"][name] == {"w": 960, "h": 720}
            np.testing.assert_allclose(h["K"][name], rig.virtual[name].camera((960, 720)).K)
            np.testing.assert_allclose(h["K"][name], rig_y[name]["K"], atol=1e-3)
            np.testing.assert_allclose(rig_y[name]["K"], iph_y[name]["K"], atol=1e-9)
    np.testing.assert_allclose(rig_y["main"]["K"][0][0], 703.1714)
    np.testing.assert_allclose(rig_y["ultrawide"]["K"][0][0], 388.5231)


def test_sensor_rig_stream_send_cli_default_transport(raw_episode, tmp_path):
    """`sensor-rig stream-send --backend mock --episode` (the real socket.io transport, same host: no skew) reaches the
    receiver: v2 hello, 30 fps, no gaps, clock offset ~0."""
    cfg_path, cfg = _camera_yaml_with_sender_K(tmp_path, raw_episode)
    rx = Receiver(tmp_path, cfg_path, [PORTS["main"]])
    tx = log = None
    try:
        tx, log = _run_sender(tmp_path, "cli", [
            sys.executable, "-u", "-m", "sensor_rig", "stream-send", "--rig", str(raw_episode / "rig.yaml"),
            "--backend", "mock", "--episode", str(raw_episode), "--loop", "--host", "127.0.0.1",
            "--port", str(PORTS["main"]), "--fps", "30", "--duration", "8"], 8)
        text = _finish(tx, log)
        time.sleep(0.8)
        st = rx.stats()
    finally:
        if tx is not None and tx.poll() is None:
            tx.kill()
        rx.stop()
    out = json.loads(text.strip().splitlines()[-1])
    assert out["sent"] >= 150 and out["transport"]["connected"] in (True, False) and "capture_error" not in out
    assert st["hellos"] >= 1 and st["hello_problems"] == [] and st["packet_errors"] == 0
    m = st["streams"]["main"]
    assert m["transport"] == "websocket" and m["seq_gaps"] == 0 and m["frames"] >= 200, m
    assert 30.0 < m["inter_arrival_ms"]["p50"] < 36.0, m      # 30 fps (fps_in is a sliding window: 0 after the sender ended)
    assert abs(st["clock"]["offset_s"]) < 0.005, st["clock"]


def test_sim_rig_fake_end_to_end(raw_episode, tmp_path, monkeypatch):
    """rig replay -> wifi_good_skew -> receiver -> task -> fake policy -> FakeBackend; READY, health and clock checks pass,
    an episode is recorded, teardown is clean."""
    _proc_skip()
    L = Launcher(cfg_for(tmp_path, "sim-rig-fake", ports=True), interactive=False, script="8:c,22:s,26:q")
    rc = L.run()
    kinds = [e["kind"] for e in L.events]
    assert "READY" in kinds, [e for e in L.events if e["kind"] in ("abort", "exit", "preflight")][-6:]
    pre = L.summary["preflight"]
    assert pre["wireless_health_ok"]["ok"] and pre["clock_ok"]["ok"] and pre["websocket_available"]["ok"], pre
    assert pre["sender_available"]["ok"] and pre["camera_rate_and_selection"]["ok"]
    assert rc == 0 and L.exit_reason == "quit", L.exit_reason
    nums = _summary_numbers(L)
    cs = L.summary["camera_stats"]["clock"]
    nums["clock_offset_s"], nums["rtt_ms_min"] = round(cs["offset_s"], 3), round(cs["rtt_ms_min"], 2)
    nums["hello"] = pre.get("hello_problems", {}).get("detail")
    print("B7_E2E", json.dumps(nums))
    (tmp_path / "e2e_summary.json").write_text(json.dumps(nums, indent=1))
    assert nums["policy_calls"] > 0 and nums["transport"][0] == "websocket"
    assert abs(cs["offset_s"] - 12.345) < 0.05, cs
    run = L.session / "task" / "episodes" / "franka_deploy" / "block_raise" / "run"
    eps = sorted(run.iterdir())
    assert eps and list((eps[0] / "tap").glob("*.npz")), eps
    ready = json.loads((L.session / "sender_ready.json").read_text())
    assert ready["protocol"] == "v2" and ready["start_now_ns"] > 0
    assert (L.session / "logs" / "sender_truth.jsonl").stat().st_size > 0
    assert (L.session / "logs" / "camera_truth.jsonl").stat().st_size > 0
    assert L.summary["ports_free_after"][0], L.summary["ports_free_after"]
    assert L.summary["shm_clean_after"][0], L.summary["shm_clean_after"]
    assert set(L.summary["child_exit_codes"]) == {"task", "controller", "sender", "camera", "inference", "spacemouse", "bridge"}
    _no_strays(L)
