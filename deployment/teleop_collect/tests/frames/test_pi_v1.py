# adapted from rnoro/final-deployment:deployment/policy_deploy/tests/frames/test_pi_v1.py @ 2ffe641
# changes: imports; the end-to-end part through R1 and U2 instead of C1 (previews are the bytes the fake Pi sent;
# a reused ultrawide frame is recorded once; pd's `unusable == 0` becomes "every packet accepted and written",
# nothing being decoded); conversion, clock, source and recording tests unchanged.
# adapted from rnoro/deployment:deployment/process_frames/tests/test_pi_v1.py @ 91b178b
# changes: pi_v1 is the only stream (no stream/decoder overrides, no time_sync); OneWayClockMap from contracts.clock_fit; PiRecording.load without encoding/keyframe_interval; the shared Running from tests/frames_pipeline.py; the converter counts uw_reused headers
"""pi_v1: the Pi's 3-part stream (JSON header, main JPEG, ultrawide JPEG) through frames to R1 and U2."""

from __future__ import annotations

import random
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import zmq

from deploy_contracts import ContractError
from teleop_collect.contracts import PiFrameStamp, PiStreamHeader, PiStreamPacket
from frames_helpers import FOLDER
from teleop_collect.contracts.clock_fit import OneWayClockMap
from teleop_collect.frames.pi_recording import PiRecording, PiRecordingError
from teleop_collect.frames.pi_stream_source import PiStreamConverter, PiStreamPacketSource
from frames_helpers import EPISODE, Running, read_csv, read_mkv, wait_for

RIG_640 = FOLDER / "rigs" / "rig_pi_v2_640.yaml"


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
    assert p.header.ultrawide == () and p.header.pair_seq == 6 and conv.repeats == 1 and conv.uw_reused == 1
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
    """Pi = laptop + 1000 s; latencies 1-30 ms: the offset comes out within 1-2 ms of (true offset - 1 ms)."""
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
    rec = PiRecording.load(ep, 10.0, 90, sizes={"main": (160, 120), "ultrawide": (64, 48)})
    for cam, (w, h) in (("main", (160, 120)), ("ultrawide", (64, 48))):
        f = rec.cameras[cam].frames[0]
        img = cv2.imdecode(np.frombuffer(f.data, np.uint8), cv2.IMREAD_COLOR)
        assert img.shape[:2] == (h, w) and (f.info.width, f.info.height) == (w, h)
    with pytest.raises(PiRecordingError, match="aspect ratio"):
        PiRecording.load(ep, 10.0, 90, sizes={"main": (160, 90)})


# ------------------------------------------------------------------------------------------------ end to end

@pytest.fixture
def run_pi_v1(tmp_path):
    runs = []

    def start(episode, **overrides):
        r = Running(tmp_path / f"run{len(runs)}", episode, **overrides)
        runs.append(r)
        return r
    yield start
    for r in runs:
        r.stop()


def sent_bytes(run, cam, frame_no):
    """The JPEG the fake Pi sent for that frame_no (the clip loops)."""
    frames = run.app.fake_pi.recording.cameras[cam].frames
    return frames[frame_no % len(frames)].data


def wait_preview(run, timeout_s=10.0):
    box = {}

    def ok():
        p = run.preview()
        box["p"] = p
        return p.main_jpeg is not None and p.ultrawide_jpeg is not None
    wait_for(ok, timeout_s, "a preview of both cameras")
    return box["p"]


def test_pi_v1_ready_previews_are_the_jpegs_sent(run_pi_v1, pi_episode):
    run = run_pi_v1(pi_episode)
    run.wait_ready()
    for _ in range(3):
        p = wait_preview(run)
        assert p.main_jpeg == sent_bytes(run, "main", p.main_frame_no)
        assert p.ultrawide_jpeg == sent_bytes(run, "ultrawide", p.ultrawide_frame_no)
        assert p.health.main.status == "ok" and p.health.ultrawide.status == "ok"
    line = run.app.stats_line(run.app.stats(run.app._clock()))
    assert "one-way" in line and "lost 0" in line and "recording no" in line


def test_pi_v1_a_reused_ultrawide_frame_is_recorded_once(run_pi_v1, make_pi_episode, tmp_path):
    """The ultrawide captured 5 ms before the main camera: dropping one ultrawide frame makes the next packet
    repeat the previous one (uw_reused, its second packet: accepted). It is written to ultrawide.mkv only once."""
    run = run_pi_v1(make_pi_episode(n_frames=30, uw_offset_ns=-5_000_000))
    run.wait_ready()
    ep = tmp_path / "ep"
    ep.mkdir()
    assert run.start_recording(ep).ok
    run.app.fake_pi.drop_frames("ultrawide", 1)
    wait_for(lambda: run.app.fake_pi.sent["reused"] >= 1 and run.health().frames_main >= 10, 10, "a reused frame")
    summary = run.stop_recording()
    assert summary.error == "" and summary.uw_reused >= 1 and summary.ultrawide.frame_no_gaps >= 1
    rx = run.app.receiver
    assert rx.packets.ignored == 0 and rx.counters["ultrawide"].pair_missing == 0 and rx.packets.pair_reused == 0
    assert rx.packets.uw_reused >= 1 and run.app.source.converter.uw_reused >= 1      # pd's uw_reused asserts
    # pd's `unusable == 0` (every frame decoded to a usable image): nothing is decoded now, so its equivalent is that
    # every frame of the recording was accepted and written: no packet unsynced or out of order, one main row each
    assert rx.packets.unsynced == 0 and rx.packets.out_of_order == 0
    assert summary.main.frames == summary.packets and summary.main.frame_no_gaps == 0
    _, uw_rows = read_csv(ep / "ultrawide_frames.csv")
    nos = [int(r["frame_no"]) for r in uw_rows]
    assert nos == sorted(set(nos))                                   # each ultrawide frame once
    _, pk_rows = read_csv(ep / "packets.csv")
    assert sum(r["uw_reused"] == "1" for r in pk_rows) == summary.uw_reused
    assert read_mkv(ep / "ultrawide.mkv") == [sent_bytes(run, "ultrawide", n) for n in nos]


@pytest.mark.skipif(not (EPISODE / "meta.json").is_file(), reason=f"{EPISODE} not present")
def test_pi_v1_real_recording_at_640(run_pi_v1):
    """episode_000045 (1600x1200 / 1280x960) delivered at 640x480 as the Pi streams it, through to U2."""
    run = run_pi_v1(EPISODE, rig_file=str(RIG_640), mock={"max_seconds": 3.0})
    run.wait_ready(timeout_s=60)
    assert run.app.rig.rig_id == "rig_pi_v2_640"
    p = wait_preview(run, timeout_s=20)
    for cam, jpeg in (("main", p.main_jpeg), ("ultrawide", p.ultrawide_jpeg)):
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert img.shape == (480, 640, 3), cam
