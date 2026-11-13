"""pi_collect: receives the Pi's current 3-part stream (JSON header, main JPEG, ultrawide JPEG) without replying."""

from __future__ import annotations

import json
import time

import cv2
import numpy as np
import pytest
import zmq

from process_frames.pi_collect import Collector, Stats, inspect_packet, main


def jpeg(w=640, h=480, value=128) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((h, w, 3), value, np.uint8))
    assert ok
    return buf.tobytes()


def header(packet_no=0, main_no=0, uw_no=0, reused=False) -> dict:
    frame = {"frame_no": 0, "t_sensor_ns": 1_000_000_000, "exposure_us": 8000, "t_mid_ns": 996_000_000}
    return {"version": 1, "packet_no": packet_no, "t_send_ns": 1_010_000_000, "clock_offset_ns": 12345,
            "main": {**frame, "frame_no": main_no}, "ultrawide": {**frame, "frame_no": uw_no},
            "skew_ns": 4_000_000, "uw_reused": reused}


def parts(h: dict, main=None, uw=None) -> list[bytes]:
    return [json.dumps(h).encode(), main if main is not None else jpeg(), uw if uw is not None else jpeg(value=60)]


def test_a_good_packet_has_no_problems_and_keeps_the_header_verbatim():
    h = header(packet_no=7)
    r = inspect_packet(parts(h), arrival_ns=1)
    assert r.problems == []
    assert r.header == h
    assert r.images["main"].shape == (480, 640, 3) and r.images["ultrawide"].shape == (480, 640, 3)
    assert r.record()["image_wh"] == {"main": [640, 480], "ultrawide": [640, 480]}


@pytest.mark.parametrize("msg, expect", [
    ([b"not json", jpeg(), jpeg()], "header is not JSON"),
    ([json.dumps([1, 2]).encode(), jpeg(), jpeg()], "expected an object"),
    ([json.dumps(header()).encode(), jpeg()], "2 parts, expected 3"),
    ([json.dumps(header()).encode(), b"garbage", jpeg()], "part 2 (main) is not a decodable JPEG"),
    ([json.dumps({k: v for k, v in header().items() if k != "skew_ns"}).encode(), jpeg(), jpeg()],
     "header: missing ['skew_ns']"),
    ([json.dumps({**header(), "main": {"frame_no": 1}}).encode(), jpeg(), jpeg()], "main: missing"),
    ([json.dumps({**header(), "extra": 1}).encode(), jpeg(), jpeg()], "header: unexpected ['extra']"),
])
def test_a_bad_packet_is_reported_not_raised(msg, expect):
    r = inspect_packet(msg, arrival_ns=1)
    assert any(expect in p for p in r.problems), r.problems


def test_stats_count_gaps_and_reused_ultrawide_frames():
    st = Stats()
    seq = [header(0, 0, 0), header(1, 1, 1), header(2, 2, 1, reused=True), header(5, 5, 4)]  # 2 packets lost
    for h in seq:
        st.add(inspect_packet(parts(h), 1))
    assert st.w["packet_gaps"] == 2 and st.w["main_gaps"] == 2 and st.w["uw_gaps"] == 2
    assert st.w["uw_reused"] == 1
    assert "packet_no gaps 2" in st.line(1.0)
    assert st.total == {"packets": 4, "bad": 0, "packet_gaps": 2, "uw_reused": 1}


def _send_until_received(pub, col, msg, timeout_s=5.0):
    """PUB drops messages until the SUB has joined, so resend until one arrives (bounded, no sleeps)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pub.send_multipart(msg)
        r = col.receive_one(0.05)
        if r is not None:
            return r
    raise AssertionError("nothing received")


def test_collector_receives_from_a_pi_style_pub_and_never_sends():
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.LINGER, 0)
    pub.bind("tcp://127.0.0.1:*")
    col = Collector(pub.getsockopt_string(zmq.LAST_ENDPOINT))
    try:
        r = _send_until_received(pub, col, parts(header(packet_no=3)))
        assert r.problems == [] and r.header["packet_no"] == 3
        assert col.socket.type == zmq.SUB                     # receive-only: a SUB socket cannot send
    finally:
        col.close()
        pub.close()


def test_main_writes_headers_and_jpegs(tmp_path):
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.LINGER, 0)
    pub.bind("tcp://127.0.0.1:*")
    ep = pub.getsockopt_string(zmq.LAST_ENDPOINT)
    station = tmp_path / "station.yaml"
    station.write_text(_minimal_station(tmp_path))
    import threading
    done = threading.Event()

    def send():
        n = 0
        while not done.is_set():
            pub.send_multipart(parts(header(packet_no=n, main_no=n, uw_no=n)))
            n += 1
            done.wait(1 / 30)

    t = threading.Thread(target=send, daemon=True)
    t.start()
    out = tmp_path / "out"
    try:
        rc = main(["--station", str(station), "--endpoint", ep, "--seconds", "1.5", "--out", str(out)])
    finally:
        done.set()
        t.join(2)
        pub.close()
    assert rc == 0
    lines = [json.loads(x) for x in (out / "headers.jsonl").read_text().splitlines()]
    assert len(lines) >= 10 and all(x["problems"] == [] for x in lines)
    assert lines[0]["header"]["version"] == 1
    jpgs = sorted(p.name for p in (out / "jpeg").iterdir())
    assert any(n.endswith("_main.jpg") for n in jpgs) and any(n.endswith("_ultrawide.jpg") for n in jpgs)


def _minimal_station(tmp_path) -> str:
    """The repo's station.yaml, with session_root moved under tmp_path."""
    from process_frames import FOLDER
    text = (FOLDER.parent / "station.yaml").read_text()
    lines = [f"session_root: {tmp_path / 'sessions'}" if ln.startswith("session_root:") else ln
             for ln in text.splitlines()]
    # rig_file and mock_episode are relative to the station file: make them absolute
    return "\n".join(ln.replace("rig_file: process_frames/", f"rig_file: {FOLDER}/") for ln in lines) + "\n"
