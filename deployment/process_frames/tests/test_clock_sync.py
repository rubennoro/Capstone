"""ClockSync against a FakePiSender with an offset (STRUCTURE.md §5.7)."""

from __future__ import annotations

import threading
import time

import pytest

from contracts import RpcServer, TimeSyncRequest

from conftest import FakeClock, wait_for
from process_frames.clock_map import ClockMap
from process_frames.clock_sync import ClockSync
from process_frames.errors import FatalError
from process_frames.fake_pi_sender import FakePiSender
from process_frames.pi_recording import PiRecording
from process_frames.settings import ClockSettings

OFFSET = 1000 * 1_000_000_000


def settings(**kw):
    d = dict(interval_s=1.0, timeout_s=0.3, startup_exchanges=10, window_s=60.0, best_fraction=0.3,
             drift_min_span_s=10.0, max_drift_ppm=500.0, max_sync_age_s=10.0)
    d.update(kw)
    return ClockSettings(**d)


def new_map():
    return ClockMap(60.0, 0.3, 10.0, 500.0, 10.0)


@pytest.fixture(scope="module")
def recording(pi_episode_no_exposure):
    return PiRecording.load(pi_episode_no_exposure, encoding="jpeg", max_seconds=1.0, keyframe_interval=4,
                            jpeg_quality=50)


@pytest.fixture
def fake(recording):
    f = FakePiSender(recording, "tcp://127.0.0.1:*", "tcp://127.0.0.1:*", pi_clock_offset_ns=OFFSET, drift_ppm=0.0,
                     send_hwm=10)
    f.start()
    yield f
    f.stop()


def test_offset_found_within_1_ms(fake):
    cm = new_map()
    cs = ClockSync(fake.time_sync_endpoint, cm, settings())
    try:
        for _ in range(10):
            assert cs.sync_once()
    finally:
        cs.close()
    assert cs.exchanges == 10 and cs.timeouts == 0
    now = time.monotonic_ns()
    assert abs(cm.offset_ns(now) - OFFSET) < 1_000_000
    assert cm.is_fresh(now)


def test_startup_exchanges_then_interval_rhythm(fake):
    """With a fake clock: 10 back to back, then one per interval_s of (fake) time."""
    clock = FakeClock()
    cm = new_map()
    cs = ClockSync(fake.time_sync_endpoint, cm, settings(interval_s=1.0), clock=clock)
    stop = threading.Event()
    t = threading.Thread(target=cs.run, args=(stop,))
    t.start()
    try:
        wait_for(lambda: cs.exchanges == 10, 10, "startup exchanges")
        for k in range(1, 4):
            clock.advance_s(1.0)
            wait_for(lambda: cs.exchanges == 10 + k, 10, f"exchange {10 + k}")
        assert cs.exchanges == 13
    finally:
        stop.set()
        t.join(5)
    assert not t.is_alive()


def test_a_server_that_does_not_answer_gives_counted_timeouts():
    # a REP socket nobody serves: requests are queued but never answered
    silent = RpcServer("tcp://127.0.0.1:*", {})
    try:
        cs = ClockSync(silent.endpoint, new_map(), settings(timeout_s=0.1))
        assert cs.sync_once() is False
        assert cs.sync_once() is False
        cs.close()
    finally:
        silent.close()
    assert cs.timeouts == 2 and cs.exchanges == 0


def test_a_broken_time_sync_reply_is_fatal_naming_the_field():
    """A server that answers time_sync with a reply missing t_pi_ns."""
    import zmq
    from contracts import pack_obj

    ctx = zmq.Context.instance()
    rep = ctx.socket(zmq.REP)
    rep.setsockopt(zmq.LINGER, 0)
    rep.bind("tcp://127.0.0.1:*")
    endpoint = rep.getsockopt_string(zmq.LAST_ENDPOINT)
    done = threading.Event()

    def serve():
        if rep.poll(5000):
            rep.recv()
            rep.send(pack_obj({"schema": "c0.time_sync_reply/v1"}))
        done.set()

    t = threading.Thread(target=serve)
    t.start()
    cs = ClockSync(endpoint, new_map(), settings(timeout_s=2.0))
    try:
        with pytest.raises(FatalError) as e:
            cs.sync_once()
    finally:
        cs.close()
        t.join(5)
        rep.close()
    assert "time_sync" in str(e.value) and "t_pi_ns" in str(e.value)
