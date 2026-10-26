"""A7 render_sender against the real B5 receiver (WirelessCamera subprocess, the station's camera config) and the A6
sim-state PUB (an in-process MuJoCoBackend). The sender runs with the launcher's own argv (Launcher.sender_command of
the sim-iphone / sim-rig profiles), only the ports, endpoints, link (ideal) and output paths swapped for test ones.

- v1: 960x720 frames, no stamps; receiver stamp (arrival - 0.08) vs the truth state time |err| p95 <= 10 ms (ideal).
- v2: stamps = the state time (<= 1 ms after the clock sync), hello K = the station K, no hello problems.
- SIGTERM -> exit 0; the ready file; the launcher argv parses (--help and a dry parse).

    cd controlSW/franka_sim && flock $LOCK env -u PYTHONPATH MUJOCO_GL=egl FD_TEST_PORT_OFFSET=1000 $RE -m pytest -q -s tests/test_render_sender.py
Ports: 5555/5556 + OFFSET + 20 (receiver), 16702 / 55556 + OFFSET + 20, sim state 5570 + OFFSET + 30.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("mujoco")
pytest.importorskip("zmq")
pytest.importorskip("robotmq")
pytest.importorskip("socketio")
pytest.importorskip("flask_socketio")
pytest.importorskip("real_env")
if os.environ.get("MUJOCO_GL") != "egl":
    pytest.skip("set MUJOCO_GL=egl", allow_module_level=True)

from conftest import DEPLOY_ROOT, SIM_ROOT  # noqa: E402
from franka_deploy.harness.dryrun import cleanup_shm  # noqa: E402
from franka_deploy.launch import launcher as lm  # noqa: E402
from franka_deploy.launch.launcher import Launcher, load_config  # noqa: E402
from franka_sim.senders import render_sender as RS  # noqa: E402

OFFSET = int(os.environ.get("FD_TEST_PORT_OFFSET", "0"))
PORTS = {"main": 5555 + OFFSET + 20, "ultrawide": 5556 + OFFSET + 20}
CAM_EP = f"tcp://localhost:{16702 + OFFSET + 20}"
LOG_EP = f"tcp://localhost:{55556 + OFFSET + 20}"
STATE_EP = f"tcp://127.0.0.1:{5570 + OFFSET + 30}"
IDEAL = SIM_ROOT.parent / "configs" / "run" / "link" / "ideal.yaml"
HOME = np.array([[1.0, 0, 0, 0.45], [0, -1.0, 0, 0.0], [0, 0, -1.0, 0.45], [0, 0, 0, 1]])


def _env():
    e = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    e.update(PYTHONPATH=f"{DEPLOY_ROOT}:{SIM_ROOT}", PYTHONDONTWRITEBYTECODE="1", MUJOCO_GL="egl")
    return e


def _port_open(port):
    with socket.socket() as s:
        s.settimeout(0.2)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _launcher(tmp_path, profile):
    cfg = load_config(DEPLOY_ROOT / "configs" / "deploy.yaml", profile)
    cfg.update(work_root=str(tmp_path / "runs"), python={"bp": sys.executable, "realenv": sys.executable})
    L = Launcher(cfg, interactive=False)
    L.snapshot()
    return L


def _flag(cmd, name):
    return cmd[cmd.index(name) + 1]


def _swap(cmd, name, value):
    cmd = list(cmd)
    if name in cmd:
        cmd[cmd.index(name) + 1] = value
    else:
        cmd += [name, value]
    return cmd


@pytest.mark.parametrize("profile", ["sim-iphone", "sim-rig"])
def test_launcher_argv_parses(tmp_path, profile):
    L = _launcher(tmp_path, profile)
    cmd = L.sender_command()
    assert L.image_source() == "render" and cmd[cmd.index("-m") + 1] == "franka_sim.senders.render_sender"
    args = RS.build_parser().parse_args(cmd[cmd.index("-m") + 2:])
    assert args.loop and args.state_endpoint == lm.SIM_STATE_ENDPOINT and args.station.endswith("station_resolved.yaml")
    assert args.protocol == ("v1" if profile == "sim-iphone" else "v2") and args.fps == 30.0
    assert Path(args.config).is_file() and Path(args.station).is_file()
    h = subprocess.run([sys.executable, "-m", "franka_sim.senders.render_sender", "--help"], env=_env(), capture_output=True,
                       text=True, timeout=60)
    assert h.returncode == 0 and "--state-endpoint" in h.stdout and "--loop" in h.stdout


@pytest.fixture(scope="module")
def backend(tmp_path_factory):
    from franka_sim.backend import MuJoCoBackend

    tmp = tmp_path_factory.mktemp("rs_st")
    L = _launcher(tmp, "sim-iphone")
    b = MuJoCoBackend.from_config({"type": "mujoco", "mujoco": {"station": str(L.session / "config" / "station_resolved.yaml"),
                                                                "state_endpoint": STATE_EP}}, HOME)
    b.start()
    yield b
    b.close()


class Receiver:
    def __init__(self, tmp: Path, profile: str, L):
        cfg = L.station.camera_config(CAM_EP, LOG_EP)
        cfg["ports"] = {s: PORTS[s] for s in cfg["ports"]}
        cfg["bind_host"] = "127.0.0.1"
        cfg["health"]["stats_period_s"] = 0.5
        p = tmp / f"camera_{profile}.yaml"
        p.write_text(yaml.safe_dump(cfg))
        self.cfg = cfg
        self.stats_path, self.truth_path, self.log_path = tmp / "rx_stats.json", tmp / "rx_truth.jsonl", tmp / "rx.log"
        self.log = open(self.log_path, "w")
        self.p = subprocess.Popen([sys.executable, "-u", "-m", "franka_deploy.peripherals.wireless_camera", "--config", str(p),
                                   "--stats-path", str(self.stats_path), "--truth-path", str(self.truth_path)],
                                  stdout=self.log, stderr=subprocess.STDOUT, env=_env(), cwd=str(tmp), start_new_session=True)
        end = time.monotonic() + 30
        while not all(_port_open(PORTS[s]) for s in cfg["ports"]):
            if self.p.poll() is not None or time.monotonic() > end:
                raise AssertionError("receiver did not start:\n" + self.log_path.read_text()[-3000:])
            time.sleep(0.1)

    def stats(self):
        try:
            return json.loads(self.stats_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def truth(self):
        return [json.loads(x) for x in self.truth_path.read_text().splitlines() if x.strip()]

    def stop(self):
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(5)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait(5)
        self.log.close()
        cleanup_shm(self.p.pid)


def _run(tmp_path, profile, seconds=7.0):
    """Receiver + the launcher-argv render sender for ``seconds``, then SIGTERM. Returns everything measured."""
    L = _launcher(tmp_path, profile)
    cmd = L.sender_command()
    ports = ",".join(str(PORTS[s]) for s in ("main", "ultrawide") if s in L.station.get("sensor.transport.ports"))
    cmd = _swap(cmd, "--ports", ports)
    cmd = _swap(cmd, "--state-endpoint", STATE_EP)
    cmd = _swap(cmd, "--link", str(IDEAL))
    cmd = _swap(cmd, "--truth", str(tmp_path / "tx_truth.jsonl"))
    cmd = _swap(cmd, "--ready-file", str(tmp_path / "ready.json"))
    cmd[0] = sys.executable
    rx = Receiver(tmp_path, profile, L)
    log = open(tmp_path / "tx.log", "w")
    tx = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=_env(), cwd=str(tmp_path))
    out = {"L": L, "cmd": cmd}
    try:
        end = time.monotonic() + 60
        while not (tmp_path / "ready.json").exists():
            assert tx.poll() is None and time.monotonic() < end, (tmp_path / "tx.log").read_text()[-3000:]
            time.sleep(0.1)
        time.sleep(seconds)
        out["stats"] = rx.stats()
        tx.send_signal(signal.SIGTERM)
        out["rc"] = tx.wait(30)
        time.sleep(0.6)
        out["rx_truth"], out["ready"] = rx.truth(), json.loads((tmp_path / "ready.json").read_text())
        out["tx_truth"] = [json.loads(x) for x in (tmp_path / "tx_truth.jsonl").read_text().splitlines() if x.strip()]
    finally:
        if tx.poll() is None:
            tx.kill()
            tx.wait(5)
        log.close()
        rx.stop()
    out["tx_log"] = (tmp_path / "tx.log").read_text()
    return out


def _errors(r, t_from_s=2.0):
    t = {(x["seq"], x["stream"]): x["t_true"] for x in r["tx_truth"]}
    pairs = [(x, t[(x["seq"], x["stream"])]) for x in r["rx_truth"] if (x["seq"], x["stream"]) in t]
    t0 = min(tt for _, tt in pairs)
    return {s: np.array([x["t_est"] - tt for x, tt in pairs if x["stream"] == s and tt >= t0 + t_from_s]) for s in ("main", "ultrawide")}


def test_v1_render_stream(tmp_path, backend):
    r = _run(tmp_path, "sim-iphone")
    assert r["rc"] == 0, r["tx_log"][-3000:]
    rd = r["ready"]
    assert rd["sender"] == "render_sender" and rd["protocol"] == "v1" and rd["size"] == [960, 720]
    assert np.allclose(rd["K_main"], r["L"].station.get("sensor.streams.main.K"))
    st = r["stats"]
    for s in ("main", "ultrawide"):
        m = st["streams"][s]
        assert m["frames"] > 100 and m["size_mismatches"] == 0, m
        # the 30 Hz render tick must not be throttled by the v1 latency delay (was ~14 fps when it slept inline)
        assert m["fps_in"] >= 27 and m["inter_arrival_ms"]["p95"] <= 50, m
        assert abs(rd["fps"] - m["fps_in"]) <= 0.10 * m["fps_in"], (rd["fps"], m["fps_in"])
    assert st["protocol"] in ("v1", "iphumi_socketio_v1") and not st["packet_errors"]
    e = _errors(r)
    msg = "; ".join(f"{s} p50 {np.median(np.abs(v)) * 1e3:.2f} / p95 {np.percentile(np.abs(v), 95) * 1e3:.2f} ms (n {len(v)})" for s, v in e.items())
    print(f"\n[v1] receiver stamp (arrival - 0.08) - state time: {msg}; render p95 {rd['render_ms_p95']} ms; "
          f"receiver fps_in {st['streams']['main']['fps_in']:.1f}/{st['streams']['ultrawide']['fps_in']:.1f}, "
          f"inter-arrival p95 {st['streams']['main']['inter_arrival_ms']['p95']} ms; done: {r['tx_log'].strip().splitlines()[-1][:300]}")
    for v in e.values():
        assert len(v) > 60 and np.percentile(np.abs(v), 95) <= 0.010


def test_v2_render_stream(tmp_path, backend):
    r = _run(tmp_path, "sim-rig")
    assert r["rc"] == 0, r["tx_log"][-3000:]
    st = r["stats"]
    assert st["hellos"] >= 1 and st["hello_problems"] == [], st["hello_problems"]
    assert st["streams"]["main"]["frames"] > 100 and st["streams"]["main"]["size_mismatches"] == 0
    e = _errors(r, t_from_s=3.0)
    msg = "; ".join(f"{s} p50 {np.median(np.abs(v)) * 1e3:.3f} / p95 {np.percentile(np.abs(v), 95) * 1e3:.3f} ms (n {len(v)})" for s, v in e.items())
    print(f"\n[v2] receiver stamp - state time: {msg}; render p95 {r['ready']['render_ms_p95']} ms")
    for v in e.values():
        # p95 <= 2 ms is the B7 rig criterion (drifts under suite load); p50 is the tight bound
        assert len(v) > 60 and np.percentile(np.abs(v), 95) <= 0.002 and np.median(np.abs(v)) <= 0.0005
    assert np.allclose(r["ready"]["K_uw"], r["L"].station.get("sensor.streams.ultrawide.K"))
