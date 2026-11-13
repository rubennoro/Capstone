"""stream pi_v1: the Pi's 3-part stream (JSON header, main JPEG, ultrawide JPEG) through process_frames to C1."""

from __future__ import annotations

import random
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import zmq

from contracts import ContractError, PiFrameStamp, PiStreamHeader, PiStreamPacket
from process_frames import FOLDER
from process_frames._dataproc.imageprep import ImagePrep
from process_frames.clock_map import OneWayClockMap
from process_frames.pi_recording import PiRecording, PiRecordingError
from process_frames.pi_stream_source import PiStreamConverter, PiStreamPacketSource
from test_pipeline import Running, check_served

RIG_640 = FOLDER / "rigs" / "rig_pi_v2_640.yaml"
EPISODE = Path("~/Downloads/episode_000045").expanduser()
PI_V1 = {"stream": "pi_v1", "decoder": "jpeg"}


def jpeg(w=640, h=480, value=100) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((h, w, 3), value, np.uint8))
    assert ok
    return buf.tobytes()


def stamp(frame_no, t_ns, exposure_us=8000) -> PiFrameStamp:
    return PiFrameStamp(frame_no, t_ns, exposure_us, t_ns - exposure_us * 500)


def message(packet_no, uw_no, reused=False, t_send=10_000_000_000, main_wh=(640, 480), uw_wh=(640, 480)):
    main, uw = stamp(packet_no, 9_000_000_000 + packet_no * 33_333_333), stamp(uw_no, 8_990_000_000 + uw_no * 33_333_333)
    h = PiStreamHeader(1, packet_no, t_send, 123, main, uw, main.t_mid_ns - uw.t_mid_ns, reused)
    return PiStreamPacket(h, jpeg(*main_wh), jpeg(*uw_wh, value=50))


# ------------------------------------------------------------------------------------------------ conversion

def test_a_new_ultrawide_frame_is_passed_on_and_paired():
    p = PiStreamConverter().convert(message(7, 6))
    h = p.header
    assert h.packet_seq == 7 and h.pair_seq == 6 and len(p.ultrawide) == 1
    assert (h.main.sensor, h.main.frame_seq, h.main.width, h.main.height) == ("imx519", 7, 640, 480)
    assert h.main.t_pi_ns == 9_000_000_000 + 7 * 33_333_333 and h.main.exposure_ns == 8_000_000
    assert (h.main.encoding, h.main.keyframe) == ("jpeg", True)
    assert (h.ultrawide[0].sensor, h.ultrawide[0].frame_seq) == ("imx219", 6)


def test_a_repeated_ultrawide_frame_is_not_decoded_twice_but_stays_the_pair():
    conv = PiStreamConverter()
    conv.convert(message(7, 6))
    p = conv.convert(message(8, 6, reused=True))
    assert p.header.ultrawide == () and p.header.pair_seq == 6 and conv.repeats == 1
    p = conv.convert(message(9, 7))
    assert [f.frame_seq for f in p.header.ultrawide] == [7] and p.header.pair_seq == 7


def test_frame_sizes_come_from_the_jpegs():
    p = PiStreamConverter().convert(message(1, 1, main_wh=(320, 240), uw_wh=(160, 120)))
    assert (p.header.main.width, p.header.main.height) == (320, 240)
    assert (p.header.ultrawide[0].width, p.header.ultrawide[0].height) == (160, 120)


# ------------------------------------------------------------------------------------------------ clock

def one_way_map(**kw) -> OneWayClockMap:
    args = dict(window_s=60.0, best_fraction=0.3, drift_min_span_s=10.0, max_drift_ppm=500.0, max_sync_age_s=10.0)
    args.update(kw)
    return OneWayClockMap(**args)


def test_one_way_offset_is_the_least_delayed_stamps():
    """Pi = PC + 1000 s; latencies 1-30 ms: the offset comes out within 1-2 ms of (true offset - 1 ms)."""
    rng = random.Random(1)
    m, offset = one_way_map(), 1_000_000_000_000
    for i in range(300):
        pc_send = 5_000_000_000 + i * 33_333_333
        latency = 1_000_000 + int(rng.random() ** 3 * 29_000_000)          # mostly short, a few long
        m.add_one_way(pc_send + offset, pc_send + latency)
    now = 5_000_000_000 + 300 * 33_333_333
    est = m.offset_ns(now)
    assert offset - 3_000_000 <= est <= offset - 1_000_000
    st = m.stats(now)
    assert st.synced and st.rtt_min_ms is None
    assert not m.is_fresh(now + 11_000_000_000)                             # packets stopped: stale


def test_one_way_drift_is_fitted():
    m, offset = one_way_map(), 500_000_000
    for i in range(30 * 30):                                                # 30 s at 30 packets/s
        pc = 1_000_000_000 + i * 33_333_333
        m.add_one_way(pc + offset + int(50e-6 * i * 33_333_333), pc + 2_000_000)     # Pi 50 ppm fast, 2 ms late
    assert abs(m.stats(1_000_000_000 + 900 * 33_333_333).drift_ppm - 50.0) < 1.0


# ------------------------------------------------------------------------------------------------ source

def test_source_receives_converts_and_feeds_the_clock():
    pub = zmq.Context.instance().socket(zmq.PUB)
    pub.setsockopt(zmq.LINGER, 0)
    pub.bind("tcp://127.0.0.1:*")
    m = one_way_map()
    src = PiStreamPacketSource(pub.getsockopt_string(zmq.LAST_ENDPOINT), 60, m)
    try:
        deadline, packet = time.monotonic() + 5, None
        msg = message(3, 2, t_send=time.monotonic_ns() + 7_000_000_000)
        while packet is None and time.monotonic() < deadline:
            pub.send_multipart(msg.to_parts())
            packet = src.next_packet(0.05)
        assert packet is not None and packet.header.packet_seq == 3
        assert m.is_fresh(time.monotonic_ns())
        assert abs(m.offset_ns(time.monotonic_ns()) - 7_000_000_000) < 1_000_000_000
        pub.send_multipart([b"{}", b"x", b"y"])
        with pytest.raises(ContractError):
            while src.next_packet(0.5) is not None:
                pass
    finally:
        src.close()
        pub.close()


# ------------------------------------------------------------------------------------------------ recording

def test_recording_is_delivered_at_the_rig_size(make_pi_episode):
    ep = make_pi_episode(n_frames=4, main_size=(320, 240), uw_size=(320, 240))
    rec = PiRecording.load(ep, "jpeg", 10.0, 30, 90, sizes={"main": (160, 120), "ultrawide": (64, 48)})
    for cam, (w, h) in (("main", (160, 120)), ("ultrawide", (64, 48))):
        f = rec.cameras[cam].frames[0]
        img = cv2.imdecode(np.frombuffer(f.data, np.uint8), cv2.IMREAD_COLOR)
        assert img.shape[:2] == (h, w) and (f.info.width, f.info.height) == (w, h)
    with pytest.raises(PiRecordingError, match="aspect ratio"):
        PiRecording.load(ep, "jpeg", 10.0, 30, 90, sizes={"main": (160, 90)})


# ------------------------------------------------------------------------------------------------ end to end

@pytest.fixture
def run_pi_v1(tmp_path):
    runs = []

    def start(episode, **overrides):
        r = Running(tmp_path / f"run{len(runs)}", episode, **{**PI_V1, **overrides})
        runs.append(r)
        return r
    yield start
    for r in runs:
        r.stop()


def test_pi_v1_ready_and_true_images_and_times(run_pi_v1, pi_episode):
    run = run_pi_v1(pi_episode)
    run.wait_ready()
    assert run.app.clock_sync is None and run.app.fake_pi.time_sync_endpoint is None
    for _ in range(3):
        reply = run.wait_frames()
        assert reply.main.shape == (2, 224, 224, 3) and reply.ultrawide.shape == (2, 224, 224, 3)
        check_served(run, reply)                       # bit-identical images, times within 2 ms of the truth
    line = run.app.stats_line(run.app.stats(run.app._clock()))
    assert "one-way" in line and "lost 0" in line


def test_pi_v1_a_reused_ultrawide_frame_keeps_serving(run_pi_v1, make_pi_episode):
    """The ultrawide captured 5 ms before the main camera: dropping one ultrawide frame makes the next packet
    repeat the previous one (uw_reused). It is not decoded again; serving goes on."""
    run = run_pi_v1(make_pi_episode(n_frames=30, uw_offset_ns=-5_000_000))
    run.wait_ready()
    run.app.fake_pi.drop_frames("ultrawide", 1)
    deadline = time.monotonic() + 10
    while run.app.fake_pi.sent["reused"] == 0 and time.monotonic() < deadline:
        time.sleep(0.02)
    sent = run.app.fake_pi.sent
    assert sent["reused"] >= 1
    reply = run.wait_frames(lambda r: run.app.receiver.counters["ultrawide"].dropped >= 1)
    check_served(run, reply)
    rx = run.app.receiver
    assert rx.counters["main"].unusable == 0 and rx.counters["ultrawide"].unusable == 0
    assert rx.packets.ignored == 0 and rx.counters["ultrawide"].pair_missing == 0


@pytest.mark.skipif(not (EPISODE / "meta.json").is_file(), reason=f"{EPISODE} not present")
def test_pi_v1_real_recording_at_640(run_pi_v1):
    """episode_000045 (1600x1200 / 1280x960) delivered at 640x480 as the Pi streams it, through to C1."""
    run = run_pi_v1(EPISODE, rig_file=str(RIG_640), mock={"max_seconds": 3.0})
    run.wait_ready(timeout_s=60)
    assert run.app.rig.rig_id == "rig_pi_v2_640"
    prep = ImagePrep(run.app.rig)
    gaps = []
    for _ in range(3):
        reply = run.wait_frames(timeout_s=20)
        assert reply.main.shape == (2, 224, 224, 3) and reply.ultrawide.shape == (2, 224, 224, 3)
        gaps += check_served(run, reply, prep)
    assert gaps
