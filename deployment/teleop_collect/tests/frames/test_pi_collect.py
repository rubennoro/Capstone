# adapted from rnoro/final-deployment:deployment/policy_deploy/tests/frames/test_pi_collect.py @ 2ffe641
# changes: imports; write_station takes the pi section through sections= (this package's station_helpers)
# adapted from rnoro/deployment:deployment/process_frames/tests/test_pi_collect.py @ 91b178b
# changes: the station test uses the laptop station (pi.address/pi.port); added cases: the pair-gap report (computed from the two t_mid_ns, not skew_ns; min/median/max, histogram, longest same-pair run; pair_gap.json)
"""pi_collect: receives the Pi's current 3-part stream (JSON header, main JPEG, ultrawide JPEG) without replying."""

from __future__ import annotations

import json
import time

import cv2
import numpy as np
import pytest
import zmq

from station_helpers import write_station

from teleop_collect.frames.pi_collect import Collector, PairGapReport, Stats, inspect_packet, main


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


class _Sender:
    """A Pi-style PUB on 127.0.0.1 sending good packets at 30 per second until stopped."""

    def __init__(self):
        import threading
        self.pub = zmq.Context.instance().socket(zmq.PUB)
        self.pub.setsockopt(zmq.LINGER, 0)
        self.pub.bind("tcp://127.0.0.1:*")
        self.endpoint = self.pub.getsockopt_string(zmq.LAST_ENDPOINT)
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        n = 0
        while not self.done.is_set():
            self.pub.send_multipart(parts(header(packet_no=n, main_no=n, uw_no=n)))
            n += 1
            self.done.wait(1 / 30)

    def stop(self):
        self.done.set()
        self.thread.join(2)
        self.pub.close()


def _check_output(out):
    lines = [json.loads(x) for x in (out / "headers.jsonl").read_text().splitlines()]
    assert len(lines) >= 10 and all(x["problems"] == [] for x in lines)
    assert lines[0]["header"]["version"] == 1
    jpgs = sorted(p.name for p in (out / "jpeg").iterdir())
    assert any(n.endswith("_main.jpg") for n in jpgs) and any(n.endswith("_ultrawide.jpg") for n in jpgs)


def test_main_with_endpoint_and_out_needs_no_station(tmp_path):
    sender = _Sender()
    out = tmp_path / "out"
    try:
        rc = main(["--endpoint", sender.endpoint, "--out", str(out), "--seconds", "1.5"])
    finally:
        sender.stop()
    assert rc == 0
    _check_output(out)


def test_main_default_out_is_pi_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    sender = _Sender()
    try:
        assert main(["--endpoint", sender.endpoint, "--seconds", "1.0"]) == 0
    finally:
        sender.stop()
    (run,) = list((tmp_path / "pi_runs").iterdir())
    assert (run / "headers.jsonl").stat().st_size > 0


def test_main_station_supplies_the_endpoint(tmp_path):
    sender = _Sender()
    host, port = sender.endpoint[len("tcp://"):].rsplit(":", 1)
    station = write_station(tmp_path, sections={"pi": {"address": host, "port": int(port)}})
    out = tmp_path / "out"
    try:
        rc = main(["--station", str(station), "--out", str(out), "--seconds", "1.5"])
    finally:
        sender.stop()
    assert rc == 0
    _check_output(out)


def test_main_without_endpoint_or_station_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as e:
        main([])
    assert e.value.code == 2 and "--endpoint" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------------- pair-gap report

def _h(packet_no, main_mid, uw_no, uw_mid, skew=None):
    h = header(packet_no, packet_no, uw_no)
    h["main"]["t_mid_ns"], h["ultrawide"]["t_mid_ns"] = main_mid, uw_mid
    h["skew_ns"] = (main_mid - uw_mid) if skew is None else skew
    return h


def test_pair_gap_is_computed_from_the_two_t_mid_not_skew():
    """The live Pi sends skew_ns with the opposite sign (PI_PAYLOAD.md 0.6): the report must not care."""
    documented, live = PairGapReport(), PairGapReport()
    for i in range(10):
        main_mid, uw_mid = 1_000_000_000 + i * 33_333_333, 1_016_292_500 + i * 33_333_333   # uw 16.3 ms newer
        documented.add(_h(i, main_mid, i, uw_mid))
        live.add(_h(i, main_mid, i, uw_mid, skew=uw_mid - main_mid))                       # reversed skew_ns
    assert documented.summary() == live.summary()
    s = live.summary()
    assert s["count"] == 10 and s["min_ms"] == pytest.approx(-16.2925) and s["max_ms"] == pytest.approx(-16.2925)
    assert s["histogram"] == {"-20": 10}


def test_pair_gap_min_median_max_histogram_and_longest_run():
    r = PairGapReport()
    # pairs 0, 0, 1, 2, 2, 2, 3: the longest run of the same pair is 3 (frame 2)
    gaps_ms = [-16, 17, -15, -16, 17, 50, -14]
    pairs = [0, 0, 1, 2, 2, 2, 3]
    for i, (g, p) in enumerate(zip(gaps_ms, pairs)):
        r.add(_h(i, 2_000_000_000 + g * 1_000_000, p, 2_000_000_000))
    s = r.summary()
    assert (s["min_ms"], s["median_ms"], s["max_ms"]) == (-16.0, -14.0, 50.0)
    assert s["histogram"] == {"-20": 2, "-15": 2, "15": 2, "50": 1}     # 5 ms bins by their start
    assert s["longest_same_pair_run"] == 3 and s["longest_same_pair_run_frame_no"] == 2
    text = r.text()
    assert "min -16.0 ms, median -14.0 ms, max 50.0 ms" in text and "same pair: 3" in text


def test_pair_gap_ignores_headers_without_t_mid():
    r = PairGapReport()
    r.add(None)
    r.add({"version": 1})
    assert r.summary()["count"] == 0 and "no headers" in r.text()


def test_main_writes_the_pair_gap_report(tmp_path, capsys):
    sender = _Sender()
    out = tmp_path / "out"
    try:
        assert main(["--endpoint", sender.endpoint, "--out", str(out), "--seconds", "1.0"]) == 0
    finally:
        sender.stop()
    rep = json.loads((out / "pair_gap.json").read_text())
    assert rep["count"] >= 5 and rep["min_ms"] == rep["max_ms"] == pytest.approx(0.0)
    assert rep["longest_same_pair_run"] == 1
    assert "pair gap (main.t_mid_ns - ultrawide.t_mid_ns" in capsys.readouterr().out
