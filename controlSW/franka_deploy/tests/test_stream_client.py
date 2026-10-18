"""B4: StreamClient against a minimal Flask-SocketIO test server (threading + simple-websocket, like the B5 receiver).

The server runs in its own process (logs every event as JSON lines to a file). Ports: 5555/5556 + FD_TEST_PORT_OFFSET.
"""
import base64
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

from franka_deploy.stream.client import StreamClient
from franka_deploy.stream.link import LinkConfig
from sensor_rig.stream import protocol as P

OFFSET = int(os.environ.get("FD_TEST_PORT_OFFSET", "0"))
PORTS = {"main": 5555 + OFFSET, "ultrawide": 5556 + OFFSET}

SERVER = textwrap.dedent('''
    import base64, json, os, sys, threading, time, logging
    import cv2, numpy as np
    from flask import Flask
    from flask_socketio import SocketIO
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    log_path, ports, ctl = sys.argv[1], [int(p) for p in sys.argv[2].split(",")], sys.argv[3]
    lock = threading.Lock()
    def rec(**kw):
        kw["t"] = time.monotonic()
        with lock, open(log_path, "a") as f:
            f.write(json.dumps(kw) + "\\n")
    def make(port, v2):
        app = Flask(f"s{port}")
        sio = SocketIO(app, async_mode="threading", max_http_buffer_size=10 * 1024 * 1024)
        @sio.on("connect")
        def c():
            rec(ev="connect", port=port)
        @sio.on("disconnect")
        def d(*a):
            rec(ev="disconnect", port=port)
        @sio.on("rgb")
        def rgb(data):
            t_rx = time.monotonic()
            im = cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_COLOR)
            rec(ev="rgb", port=port, type=type(data).__name__, n=len(data), head=data[:8], shape=list(im.shape), t_rx=t_rx)
        @sio.on("validate")
        def val(data):
            rec(ev="validate", port=port, data=data)
            if os.path.exists(ctl + ".noack"):
                return None
            return "server got it!"
        @sio.on("hello")
        def hello(d):
            from flask import request
            rec(ev="hello", port=port, hello=d)
            sid = request.sid
            def ping():
                t0 = time.monotonic_ns()
                r = sio.call("clock_ping", {"t0_ns": t0}, to=sid, timeout=3)
                rec(ev="pong", port=port, t0=t0, t3=time.monotonic_ns(), r=r)
            threading.Thread(target=ping, daemon=True).start()
        @sio.on("packet")
        def packet(d):
            rec(ev="packet", port=port, types={k: type(v).__name__ for k, v in d.items()},
                lens={k: len(v) for k, v in d.items() if isinstance(v, (bytes, bytearray))},
                v=d["v"], seq=d["seq"], t_main_ns=d["t_main_ns"], t_uw_ns=d["t_uw_ns"], clock=d["clock"])
        threading.Thread(target=lambda: sio.run(app, host="127.0.0.1", port=port, allow_unsafe_werkzeug=True), daemon=True).start()
    for p in ports:
        make(p, False)
    rec(ev="ready")
    while True:
        time.sleep(0.2)
''')


class Server:
    def __init__(self, tmp, ports):
        self.log = Path(tmp) / "server.jsonl"
        self.ctl = str(Path(tmp) / "ctl")
        script = Path(tmp) / "server.py"
        script.write_text(SERVER)
        self.proc = subprocess.Popen([sys.executable, str(script), str(self.log), ",".join(map(str, ports)), self.ctl],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.wait(lambda r: r["ev"] == "ready", 1, 15)

    def events(self, pred=None):
        if not self.log.exists():
            return []
        out = [json.loads(l) for l in self.log.read_text().splitlines() if l.strip()]
        return [r for r in out if pred is None or pred(r)]

    def wait(self, pred, n, timeout=10.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            ev = self.events(pred)
            if len(ev) >= n:
                return ev
            time.sleep(0.02)
        raise AssertionError(f"timeout waiting for {n} events; have {len(self.events(pred))}; all={self.events()[-5:]}")

    def stop(self):
        self.proc.kill()
        self.proc.wait(timeout=5)


@pytest.fixture
def server(tmp_path):
    s = Server(tmp_path, list(PORTS.values()))
    try:
        yield s
    finally:
        s.stop()


@pytest.fixture(autouse=True)
def _lock_note():
    # process tests are meant to run under `flock $LOCK` (see the B4 spec); nothing to do here
    yield


import cv2  # noqa: E402

JPEG = cv2.imencode(".jpg", (np.random.default_rng(0).random((48, 64, 3)) * 255).astype(np.uint8))[1].tobytes()


def test_v1_rgb_validate_websocket(server, tmp_path):
    c = StreamClient("127.0.0.1", PORTS, "v1", truth_path=str(tmp_path / "truth.jsonl"), validate_period_s=0.2)
    try:
        c.connect(5.0)
        t0 = time.monotonic()
        c.send_v1("main", JPEG, t0, 0)
        c.send_v1("ultrawide", JPEG, t0, 0)
        c.send_v1("main", JPEG, t0 + 0.03, 1)
        ev = server.wait(lambda r: r["ev"] == "rgb", 3)
        assert {(r["port"], r["type"]) for r in ev} == {(PORTS["main"], "str"), (PORTS["ultrawide"], "str")}
        assert sum(r["port"] == PORTS["main"] for r in ev) == 2
        assert P.decode_v1(base64.b64encode(JPEG).decode()) == JPEG
        assert ev[0]["shape"] == [48, 64, 3] and ev[0]["n"] == len(base64.b64encode(JPEG)) and ev[0]["head"] == base64.b64encode(JPEG)[:8].decode()
        v = server.wait(lambda r: r["ev"] == "validate", 4, 5)
        assert all(r["data"] == "ping" for r in v)
        assert {r["port"] for r in v} == set(PORTS.values())
        time.sleep(0.1)
        st = c.stats()
        assert st["transport"] == "websocket" and st["reconnects"] == 0 and st["validate_ok"] >= 2
        assert c.stats()["sent"] == 3
    finally:
        c.close()
    truth = [json.loads(l) for l in (tmp_path / "truth.jsonl").read_text().splitlines()]
    assert [(t["seq"], t["stream"]) for t in truth] == [(0, "main"), (0, "ultrawide"), (1, "main")]


def test_v1_reconnects_when_no_ack(server):
    c = StreamClient("127.0.0.1", PORTS, "v1", validate_period_s=0.3)
    try:
        c.connect(5.0)
        Path(server.ctl + ".noack").write_text("1")
        end = time.monotonic() + 8
        while c.stats()["reconnects"] < 1 and time.monotonic() < end:
            time.sleep(0.05)
        assert c.stats()["reconnects"] >= 1
        assert c.stats()["validate_timeouts"] >= 1
        os.remove(server.ctl + ".noack")
        end = time.monotonic() + 6
        while not c.connected("main") and time.monotonic() < end:
            time.sleep(0.05)
        assert c.connected("main")
        time.sleep(0.5)
        before = len(server.events(lambda r: r["ev"] == "rgb"))
        c.send_v1("main", JPEG, time.monotonic(), 9)
        server.wait(lambda r: r["ev"] == "rgb", before + 1, 5)
    finally:
        c.close()


def test_v1_skips_while_disconnected_and_connect_error(tmp_path):
    c = StreamClient("127.0.0.1", {"main": 5500 + OFFSET + 90}, "v1")
    with pytest.raises(ConnectionError):
        c.connect(0.6)
    c.send_v1("main", JPEG, time.monotonic(), 0)  # no crash, nothing emitted
    c.close()


def test_v2_packet_hello_clock_ping(server):
    link = LinkConfig(clock_offset_s=12.345, clock_drift_ppm=30.0)
    c = StreamClient("127.0.0.1", {"main": PORTS["main"]}, "v2", link=link,
                     v2_meta={"fps_hz": 30.0, "streams": {"main": {"w": 960, "h": 720}}, "K": {}})
    try:
        c.connect(5.0)
        h = server.wait(lambda r: r["ev"] == "hello", 1)[0]
        assert h["hello"]["v"] == 2 and h["hello"]["fps_hz"] == 30.0
        t = time.monotonic()
        c.send_v2(JPEG, JPEG[:500], t, t - 0.1, 7)
        p = server.wait(lambda r: r["ev"] == "packet", 1)[0]
        assert p["types"]["main_jpeg"] == "bytes" and p["lens"] == {"main_jpeg": len(JPEG), "uw_jpeg": 500}
        assert p["seq"] == 7 and p["clock"] == "sender_monotonic"
        # stamps are in the sender clock: true + offset (+ tiny drift)
        assert abs(p["t_main_ns"] - (t * 1e9 + 12.345e9)) < 5e6
        assert abs((p["t_main_ns"] - p["t_uw_ns"]) - 0.1e9) < 1e3 * 100
        pong = server.wait(lambda r: r["ev"] == "pong", 1)[0]
        assert abs(pong["r"]["t1_ns"] - (pong["t0"] + 12.345e9)) < 2e6
        assert pong["r"]["t2_ns"] >= pong["r"]["t1_ns"]
        assert c.stats()["transport"] == "websocket"
    finally:
        c.close()


def test_link_delays_delivery(server):
    link = LinkConfig(latency_s=0.2)
    c = StreamClient("127.0.0.1", PORTS, "v1", link=link, validate_period_s=0)
    try:
        c.connect(5.0)
        t0 = time.monotonic()
        c.send_v1("main", JPEG, t0, 0)
        r = server.wait(lambda r: r["ev"] == "rgb", 1)[0]
        assert 0.18 < r["t"] - t0 < 0.5
        assert c.stats()["backlog_s"] == 0.0
    finally:
        c.close()


# ---------------------------------------------------------------------------------------------- replay sender
RAW = Path(__file__).resolve().parents[3] / "data" / "raw" / "2026-10-01"


def _episode():
    if not RAW.is_dir():
        return None
    eps = sorted(p for p in RAW.iterdir() if "firstreal_demonstration" in p.name and (p / (p.name + "_rgb.mp4")).is_file())
    return eps[0] if eps else None


@pytest.mark.skipif(_episode() is None, reason="data/raw/2026-10-01 missing")
def test_replay_sender_v1_pacing_size_and_K(server, tmp_path):
    from franka_deploy.stream.iphone_replay_sender import scale_K
    from sensor_rig.sim.iphone_reader import IphoneEpisode
    ep_dir = _episode()
    ready, truth = tmp_path / "ready.json", tmp_path / "truth.jsonl"
    cmd = [sys.executable, "-m", "franka_deploy.stream.iphone_replay_sender", "--episode", str(ep_dir),
           "--host", "127.0.0.1", "--ports", f"{PORTS['main']},{PORTS['ultrawide']}", "--protocol", "v1", "--fps", "30",
           "--link", "run/link/ideal.yaml", "--truth", str(truth), "--ready-file", str(ready), "--max-s", "3"]
    t_launch = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = proc.communicate(timeout=90)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 0, err
    prev = -1
    while len(server.events(lambda r: r["ev"] == "rgb")) != prev:     # let the server drain
        prev = len(server.events(lambda r: r["ev"] == "rgb"))
        time.sleep(0.5)
    ev = [r for r in server.events(lambda r: r["ev"] == "rgb")]
    main = [r for r in ev if r["port"] == PORTS["main"]]
    uw = [r for r in ev if r["port"] == PORTS["ultrawide"]]
    assert 88 <= len(main) <= 92 and len(uw) == len(main)
    assert all(r["shape"] == [720, 960, 3] and r["type"] == "str" for r in ev)
    span_wire = main[-1]["t_rx"] - main[0]["t_rx"]
    tr = [json.loads(l) for l in truth.read_text().splitlines()]
    tm = [r["t_true"] for r in tr if r["stream"] == "main"]
    span_true = tm[-1] - tm[0]
    assert 2.9 < span_true <= 3.0   # first tick is slot 2 (first UW frame exists), last tick <= 3 s
    assert abs(span_wire - span_true) / span_true < 0.02, (span_wire, span_true)
    # uw content times never run ahead of the tick and lag by less than one UW period (+ frame)
    tu = [r["t_true"] for r in tr if r["stream"] == "ultrawide"]
    lag = np.array(tm) - np.array(tu)
    assert lag.min() >= -1e-6 and lag.max() < 0.25
    d = json.loads(ready.read_text())
    ep = IphoneEpisode(ep_dir)
    assert d["episode"] == ep.name and d["fps"] == 30 and d["protocol"] == "v1" and d["size"] == [960, 720]
    assert np.allclose(d["K_main"], scale_K(ep.K_main, (1920, 1440), (960, 720)))
    assert np.allclose(d["K_uw"], scale_K(ep.K_uw, (640, 480), (960, 720)))
    assert d["K_main"][0][0] == pytest.approx(ep.K_main[0, 0] * 0.5)
    assert d["K_main"][0][2] == pytest.approx((ep.K_main[0, 2] + 0.5) * 0.5 - 0.5)
    assert d["K_uw"][0][0] == pytest.approx(ep.K_uw[0, 0] * 1.5)
    assert d["start_now_ns"] / 1e9 > t_launch - 1


def test_scale_K_half_pixel():
    from franka_deploy.stream.iphone_replay_sender import scale_K
    K = [[100, 0, 9.5], [0, 100, 9.5], [0, 0, 1]]      # image centre of a 20x20 image
    K2 = scale_K(K, (20, 20), (40, 40))
    assert K2[0][2] == pytest.approx(19.5) and K2[0][0] == 200
