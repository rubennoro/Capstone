"""RigStreamSender: v2 packets from RigFrameSource (stamps, seq, JPEG size, hello, ping) and bounded-queue backpressure."""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import pytest

from sensor_rig.drivers.base import Frame
from sensor_rig.rig_config import load_rig
from sensor_rig.stream import protocol as P
from sensor_rig.stream.sender import RigStreamSender
from test_frame_source import FakeSource, T0, frame, main_t, uw_t

MS = 1_000_000


@pytest.fixture(scope="module")
def rig():
    return load_rig("rig_default.yaml")


@pytest.fixture(scope="module")
def phys_imgs(rig):
    cache = {}

    def get(cam, seq):
        w, h = rig.cameras[cam].resolution
        key = (cam, seq % 200)
        if key not in cache:
            cache[key] = np.full((h, w, 3), 10 + seq % 200, np.uint8)
        return cache[key]

    return get


class RecordingTransport:
    """In-memory transport; `delay_s` makes send_packet slow (a congested link)."""

    def __init__(self, delay_s: float = 0.0):
        self.delay_s = delay_s
        self.hello = None
        self.packets: list[dict] = []
        self.sent_at: list[int] = []
        self.closed = False

    def connect(self, hello, timeout_s=10.0):
        self.hello = hello

    def send_packet(self, pkt, size):
        if self.delay_s:
            time.sleep(self.delay_s)
        self.packets.append(pkt)
        self.sent_at.append(time.monotonic_ns())

    def answer_ping(self, data=None):
        t = time.monotonic_ns()
        return {"t1_ns": t, "t2_ns": t}

    def stats(self):
        return {"n": len(self.packets)}

    def close(self):
        self.closed = True


def make_sender(rig, main, uw, transport, **kw):
    kw.setdefault("sync", "lockstep")
    kw.setdefault("max_queue", 1000)          # finite offline sources: no backpressure drops
    kw.setdefault("startup_timeout_s", 0.5)
    return RigStreamSender(rig, "127.0.0.1", 5555, sources={"main": main, "ultrawide": uw}, transport=transport, **kw)


def finite_sources(phys_imgs, n_main=30, n_uw=30):
    main, uw = FakeSource("main"), FakeSource("ultrawide")
    for k in range(n_main):
        main.push(frame(phys_imgs("main", k), main_t(k), k))
    for j in range(n_uw):
        uw.push(frame(phys_imgs("ultrawide", j), uw_t(j), j))
    main.finished = uw.finished = True
    return main, uw


def test_packets_carry_capture_stamps_seq_and_960x720_jpegs(rig, phys_imgs):
    main, uw = finite_sources(phys_imgs)
    tr = RecordingTransport()
    s = make_sender(rig, main, uw, tr, fps_hz=60.0)
    stats = s.run()
    assert "capture_error" not in stats
    assert stats["sent"] == len(tr.packets) == 30 and stats["dropped"] == 0
    off_m = rig.capture_time_ns(rig.virtual["main"].source, 0)
    off_u = rig.capture_time_ns(rig.virtual["ultrawide"].source, 0)
    for k, d in enumerate(tr.packets):
        assert P.check_packet(d) == []
        assert d["seq"] == k                                              # increments by one
        assert d["t_main_ns"] == main_t(k) + off_m                        # = Packet.t_capture_ns["main"], monotonic
        assert d["t_uw_ns"] % 1 == 0 and abs(d["t_main_ns"] - d["t_uw_ns"]) <= 20 * MS
        assert d["t_uw_ns"] - off_u in {uw_t(j) for j in range(30)}
        assert (d["w"], d["h"], d["clock"]) == (960, 720, "sender_monotonic")
        for key, cam in (("main_jpeg", "main"), ("uw_jpeg", "ultrawide")):
            img = cv2.imdecode(np.frombuffer(d[key], np.uint8), cv2.IMREAD_COLOR)
            assert img.shape == (720, 960, 3)
    # same stamp stream as RigFrameSource's own packets
    ref = make_sender(rig, *finite_sources(phys_imgs), RecordingTransport(), fps_hz=60.0)
    ref.frame_source = ref._make_source()
    ref.frame_source.start()
    pk = ref.frame_source.next_packet()
    assert tr.packets[0]["t_main_ns"] == pk.t_capture_ns["main"]
    assert tr.packets[0]["t_uw_ns"] == pk.t_capture_ns["ultrawide"]
    ref.frame_source.stop()


def test_hello_carries_virtual_K_and_ping_answers(rig, phys_imgs):
    tr = RecordingTransport()
    s = make_sender(rig, *finite_sources(phys_imgs, 3, 12), tr, fps_hz=30.0)
    s.run()
    assert P.check_hello(tr.hello) == []
    assert tr.hello["protocol"] == "v2" and tr.hello["fps_hz"] == 30.0
    assert tr.hello["streams"]["main"] == {"w": 960, "h": 720}
    np.testing.assert_allclose(tr.hello["K"]["main"], rig.virtual["main"].camera((960, 720)).K)
    np.testing.assert_allclose(tr.hello["K"]["ultrawide"], rig.virtual["ultrawide"].camera((960, 720)).K)
    t0 = time.monotonic_ns()
    a = tr.answer_ping()
    assert t0 <= a["t1_ns"] <= a["t2_ns"] <= time.monotonic_ns()
    assert abs(s.clock_ns() - time.monotonic_ns()) < 50 * MS


def test_realtime_clock_shifts_stamps(rig, phys_imgs):
    tr = RecordingTransport()
    s = make_sender(rig, *finite_sources(phys_imgs, 4, 12), tr, fps_hz=60.0, clock="sender_realtime")
    s.run()
    off = rig.capture_time_ns(rig.virtual["main"].source, 0)
    shift = tr.packets[0]["t_main_ns"] - (main_t(0) + off)
    assert abs(shift - (time.time_ns() - time.monotonic_ns())) < 50 * MS
    assert tr.packets[0]["clock"] == "sender_realtime"
    assert abs(s.clock_ns() - time.time_ns()) < 50 * MS


def test_decimates_to_target_fps(rig, phys_imgs):
    tr = RecordingTransport()
    s = make_sender(rig, *finite_sources(phys_imgs, 40, 40), tr, fps_hz=30.0)      # 60 fps cameras -> every 2nd
    st = s.run()
    assert "capture_error" not in st
    assert st["sent"] == 20 and st["decimated"] == 20
    gaps = np.diff([d["t_main_ns"] for d in tr.packets]) / 1e6
    assert np.all(np.abs(gaps - 1e3 / 30) < 1.0)


class BlockingSource(FakeSource):
    """FakeSource whose read() waits up to timeout_s for a frame (like a camera queue)."""

    def read(self, timeout_s: float = 1.0):
        end = time.monotonic() + timeout_s
        while not self.queue and time.monotonic() < end:
            time.sleep(0.001)
        return super().read(0.0)


def feed_live(main, uw, rig, duration_s, period_s, stop):
    """Frames stamped with the live monotonic clock (minus the rig offset so capture time == now) at `period_s`."""
    off_m = rig.capture_time_ns(rig.virtual["main"].source, 0)
    off_u = rig.capture_time_ns(rig.virtual["ultrawide"].source, 0)
    wm, hm = rig.cameras[rig.virtual["main"].source].resolution
    wu, hu = rig.cameras[rig.virtual["ultrawide"].source].resolution
    imgm, imgu = np.full((hm, wm, 3), 90, np.uint8), np.full((hu, wu, 3), 160, np.uint8)
    t_end = time.monotonic() + duration_s
    k = 0
    while time.monotonic() < t_end and not stop.is_set():
        now = time.monotonic_ns()
        uw.push(Frame(image=imgu, t_ns=now - off_u, seq=k, t_arrival_ns=now))
        main.push(Frame(image=imgm, t_ns=now - off_m, seq=k, t_arrival_ns=now))
        k += 1
        time.sleep(period_s)


@pytest.mark.parametrize("policy", ["drop_oldest", "drop_newest"])
def test_slow_link_drops_and_latency_stays_bounded(rig, policy):
    """Cameras 50 fps, link takes 60 ms per packet (~17 fps): the queue stays <= max_queue, drops are counted, and the
    capture-to-send latency of late packets does not grow beyond the early ones (the app would build a backlog)."""
    main, uw = BlockingSource("main"), BlockingSource("ultrawide")
    tr = RecordingTransport(delay_s=0.06)
    s = RigStreamSender(rig, "127.0.0.1", 5555, sources={"main": main, "ultrawide": uw}, transport=tr, fps_hz=60.0,
                        max_queue=3, drop_policy=policy, sync="live", startup_timeout_s=2.0)
    stop = threading.Event()
    th = threading.Thread(target=feed_live, args=(main, uw, rig, 2.0, 0.02, stop), daemon=True)
    th.start()
    time.sleep(0.05)
    st = s.run(duration_s=2.0)
    stop.set()
    th.join()
    assert st["dropped"] > 5 and st["max_queue_len"] <= 3
    assert st["sent"] >= 10
    assert st["captured"] >= st["sent"] + st["dropped"] - 3
    seqs = [d["seq"] for d in tr.packets]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs) and seqs[-1] - seqs[0] + 1 > len(seqs)   # gaps = drops
    lat = np.array([(t - d["t_main_ns"]) / 1e6 for t, d in zip(tr.sent_at, tr.packets)])
    bound = (3 + 2) * 60 + 150.0
    assert lat.max() < bound, lat
    assert lat[-4:].mean() <= lat[:4].mean() + 120.0, (lat[:4], lat[-4:])
    assert st["queue_wait_ms"]["max"] < bound


def test_validation_and_stop(rig, phys_imgs):
    with pytest.raises(ValueError):
        RigStreamSender(rig, "h", 1, drop_policy="block", transport=RecordingTransport())
    with pytest.raises(ValueError):
        RigStreamSender(rig, "h", 1, max_queue=0, transport=RecordingTransport())
    with pytest.raises(ValueError):
        RigStreamSender(rig, "h", 1, clock="wall", transport=RecordingTransport())
    main, uw = BlockingSource("main"), BlockingSource("ultrawide")
    stop = threading.Event()
    th = threading.Thread(target=feed_live, args=(main, uw, rig, 5.0, 0.02, stop), daemon=True)
    th.start()
    time.sleep(0.05)
    tr = RecordingTransport()
    s = RigStreamSender(rig, "h", 1, sources={"main": main, "ultrawide": uw}, transport=tr, sync="live",
                        startup_timeout_s=2.0)
    threading.Timer(0.5, s.stop).start()
    t0 = time.monotonic()
    s.run()
    stop.set()
    th.join()
    assert time.monotonic() - t0 < 2.5 and len(tr.packets) > 3


def test_cli_registered_and_help():
    from sensor_rig.cli import build_parser

    parser, unavailable = build_parser()
    assert not [u for u in unavailable if "stream" in u]
    with pytest.raises(SystemExit) as e:
        parser.parse_args(["stream-send", "--help"])
    assert e.value.code == 0


def test_cli_mock_replay_to_unreachable_host_fails_cleanly(capsys):
    from sensor_rig.cli import main

    rc = main(["stream-send", "--host", "127.0.0.1", "--port", "1", "--backend", "mock", "--duration", "0.2",
                "--connect-timeout", "1"])
    assert rc == 1 and "not reachable" in capsys.readouterr().err
