"""RigFrameSource: pairing, timestamps, timeouts, caching (P2 §6.1-6.3, §7 test_frame_source::*)."""

from __future__ import annotations

import collections

import numpy as np
import pytest

from sensor_rig.deploy.frame_source import Packet, PairingConfig, RigFrameSource
from sensor_rig.drivers.base import Frame
from sensor_rig.rig_config import load_rig

MS = 1_000_000


class FakeSource:
    """Hand-fed camera backend with the CameraBackend consumer API (no thread, ``read`` never waits)."""

    def __init__(self, name: str = "fake"):
        self.name = name
        self.queue: collections.deque = collections.deque()
        self.ring: collections.deque = collections.deque(maxlen=16)
        self.running = False
        self.finished = False
        self.reads = 0

    def push(self, frame: Frame) -> None:
        self.queue.append(frame)
        self.ring.append(frame)

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def read(self, timeout_s: float = 1.0):
        self.reads += 1
        return self.queue.popleft() if self.queue else None

    def latest(self):
        return self.ring[-1] if self.ring else None

    def read_recent(self, n: int):
        return list(self.ring)[-n:] if n > 0 else []

    @property
    def exhausted(self) -> bool:
        return self.finished and not self.queue


def frame(img, t_ns, seq):
    return Frame(image=img, t_ns=int(t_ns), seq=int(seq), t_arrival_ns=int(t_ns) + 20 * MS)


@pytest.fixture(scope="module")
def rig():
    return load_rig("rig_default.yaml")


@pytest.fixture(scope="module")
def phys_imgs(rig):
    """Constant images whose value encodes the seq (value = 10 + seq % 200), one per camera resolution."""
    cache = {}

    def get(cam, seq):
        w, h = rig.cameras[cam].resolution
        key = (cam, seq % 200)
        if key not in cache:
            cache[key] = np.full((h, w, 3), 10 + seq % 200, np.uint8)
        return cache[key]

    return get


def centre_value(img) -> int:
    h, w = img.shape[:2]
    return int(img[h // 2, w // 2, 0])


def make_src(rig, main, uw, **kw):
    kw.setdefault("startup_timeout_s", 0.2)
    return RigFrameSource(rig, sources={"main": main, "ultrawide": uw}, **kw)


T0 = 5_000_000_000_000  # 5000 s of uptime


def main_t(k):
    return T0 + round(k * 1e9 / 60)


def uw_t(j, phase_ns=7 * MS):
    return T0 + phase_ns + round(j * 1e9 / 30)


def test_pairing_nearest_live(rig, phys_imgs):
    """Main 60 fps, UW 30 fps at a 7 ms phase offset; frames arrive incrementally (live). The chosen UW frame
    minimises |dt| among the frames available; both keys are always present; skew stats are right."""
    main, uw = FakeSource("main"), FakeSource("ultrawide")
    src = make_src(rig, main, uw, pairing=PairingConfig(history=4))
    j_next = 0

    def feed(k):
        nonlocal j_next
        # UW frames whose stamps are <= main stamp + 10 ms have arrived (UW pipeline latency < main's).
        while uw_t(j_next) <= main_t(k) + 10 * MS:
            uw.push(frame(phys_imgs("ultrawide", j_next), uw_t(j_next), j_next))
            j_next += 1
        main.push(frame(phys_imgs("main", k), main_t(k), k))

    feed(0)
    src.start()
    skews = [abs(src.last_packet.skew_s) * 1e3]
    for k in range(1, 40):
        feed(k)
        p = src.next_packet()
        assert set(p.images) == {"main", "ultrawide"} and not p.repeated
        assert p.seq["main"] == k
        avail = list(range(max(0, j_next - 4), j_next))
        best = min(avail, key=lambda j: (abs(uw_t(j) - main_t(k)), -uw_t(j)))
        assert p.seq["ultrawide"] == best, (k, p.seq, avail)
        assert centre_value(p.images["ultrawide"]) == 10 + best % 200
        assert centre_value(p.images["main"]) == 10 + k % 200
        assert p.skew_s == pytest.approx((main_t(k) - uw_t(best)) / 1e9, abs=1e-12)
        skews.append(abs(p.skew_s) * 1e3)
    s = src.stats()
    assert s["skew_ms"]["max"] == pytest.approx(max(skews), abs=1e-6)
    assert s["skew_ms"]["p50"] == pytest.approx(float(np.percentile(skews, 50)), abs=1e-6)
    assert s["skew_ms"]["p95"] == pytest.approx(float(np.percentile(skews, 95)), abs=1e-6)
    assert max(skews) <= 1e3 / 60 + 1e-6          # <= half a UW period (16.7 ms)
    assert s["drops"] == {"main": 0, "ultrawide": 0}
    assert s["fps"]["main"] == pytest.approx(60, rel=1e-3)


def test_pairing_lockstep_global_nearest(rig, phys_imgs):
    """Lockstep (offline replay): everything is queued up front; each packet pairs the globally nearest UW frame."""
    main, uw = FakeSource("main"), FakeSource("ultrawide")
    n_main, n_uw = 50, 26
    for j in range(n_uw):
        uw.push(frame(phys_imgs("ultrawide", j), uw_t(j, 11 * MS), j))
    for k in range(n_main):
        main.push(frame(phys_imgs("main", k), main_t(k), k))
    main.finished = uw.finished = True
    src = make_src(rig, main, uw, sync="lockstep")
    src.start()
    got = []
    while (p := src.next_packet()) is not None:
        got.append(p)
    assert [p.seq["main"] for p in got] == list(range(n_main))      # no frame skipped
    for p in got:
        k = p.seq["main"]
        best = min(range(n_uw), key=lambda j: (abs(uw_t(j, 11 * MS) - main_t(k)), -j))
        assert p.seq["ultrawide"] == best
    assert src.exhausted


def test_pairing_latest(rig, phys_imgs):
    main, uw = FakeSource("main"), FakeSource("ultrawide")
    for j in range(3):
        uw.push(frame(phys_imgs("ultrawide", j), uw_t(j), j))
    main.push(frame(phys_imgs("main", 1), main_t(1), 1))
    src = make_src(rig, main, uw, pairing={"ultrawide": "latest"})
    src.start()
    assert src.last_packet.seq["ultrawide"] == 2


def test_timestamps(rig, phys_imgs):
    """t_capture_s = (t_ns + total_offset_ns)/1e9 of the MAIN frame; strictly increasing for new frames."""
    main, uw = FakeSource("main"), FakeSource("ultrawide")
    uw.push(frame(phys_imgs("ultrawide", 0), main_t(0), 0))
    src = make_src(rig, main, uw)
    off_main = rig.cameras["main"].time.total_offset_ns
    off_uw = rig.cameras["ultrawide"].time.total_offset_ns
    assert off_main != 0                                # the default rig has a -30 ms prior: the test is meaningful
    main.push(frame(phys_imgs("main", 0), main_t(0), 0))
    src.start()
    last = src.last_packet.t_capture_s
    for k in range(1, 10):
        uw.push(frame(phys_imgs("ultrawide", k), main_t(k) + 3 * MS, k))
        main.push(frame(phys_imgs("main", k), main_t(k), k))
        p = src.next_packet()
        assert p.t_capture_s == pytest.approx((main_t(k) + off_main) / 1e9, abs=1e-9)
        assert p.t_capture_ns["main"] == main_t(k) + off_main
        assert p.t_capture_ns["ultrawide"] - p.t_capture_ns["main"] == 3 * MS + off_uw - off_main
        assert p.t_capture_s > last
        last = p.t_capture_s


def test_stale_uw_and_timeouts(rig, phys_imgs):
    # UW freezes: packets are refused (previous re-served) and RuntimeError after max_consecutive_timeouts.
    main, uw = FakeSource("main"), FakeSource("ultrawide")
    uw.push(frame(phys_imgs("ultrawide", 0), main_t(0), 0))
    main.push(frame(phys_imgs("main", 0), main_t(0), 0))
    src = make_src(rig, main, uw, max_consecutive_timeouts=3, frame_timeout_s=0.01)
    src.start()
    first = src.last_packet
    k = 1
    while main_t(k) - main_t(0) <= 0.25e9:              # still fresh enough
        main.push(frame(phys_imgs("main", k), main_t(k), k))
        assert not src.next_packet().repeated
        k += 1
    good = src.last_packet
    for i in range(3):
        main.push(frame(phys_imgs("main", k), main_t(k), k))
        k += 1
        p = src.next_packet()
        assert p.repeated and p.t_capture_s == good.t_capture_s and p.images["main"] is good.images["main"]
    main.push(frame(phys_imgs("main", k), main_t(k), k))
    with pytest.raises(RuntimeError, match="stale ultrawide"):
        src.next_packet()
    assert src.stats()["uw_stale"] == 4
    assert first.seq["main"] == 0

    # Main timeout: repeats the last packet, then raises; a new frame resets the counter.
    main, uw = FakeSource("main"), FakeSource("ultrawide")
    uw.push(frame(phys_imgs("ultrawide", 0), main_t(0), 0))
    main.push(frame(phys_imgs("main", 0), main_t(0), 0))
    src = make_src(rig, main, uw, max_consecutive_timeouts=2, frame_timeout_s=0.01)
    src.start()
    p0 = src.last_packet
    assert src.next_packet().repeated and src.next_packet().repeated
    main.push(frame(phys_imgs("main", 1), main_t(1), 1))
    assert not src.next_packet().repeated
    assert src.next_packet().repeated and src.next_packet().repeated
    with pytest.raises(RuntimeError, match="no new main frame"):
        src.next_packet()
    assert src.stats()["main_timeouts"] == 5 and p0.seq["main"] == 0


def test_startup_requires_pair(rig, phys_imgs):
    main, uw = FakeSource("main"), FakeSource("ultrawide")
    main.push(frame(phys_imgs("main", 0), main_t(0), 0))
    src = make_src(rig, main, uw, startup_timeout_s=0.05)
    with pytest.raises(RuntimeError, match="no main\\+ultrawide pair"):
        src.start()


def test_uw_remap_cached(rig, phys_imgs):
    """The UW remap runs once per UW seq, even when the same UW frame is paired with several main frames."""
    main, uw = FakeSource("main"), FakeSource("ultrawide")
    used_uw = set()
    uw.push(frame(phys_imgs("ultrawide", 0), uw_t(0, 0), 0))
    main.push(frame(phys_imgs("main", 0), main_t(0), 0))
    src = make_src(rig, main, uw, pairing={"ultrawide": "latest"})
    src.start()
    used_uw.add(src.last_packet.seq["ultrawide"])
    j = 1
    for k in range(1, 30):
        if uw_t(j, 0) <= main_t(k):
            uw.push(frame(phys_imgs("ultrawide", j), uw_t(j, 0), j))
            j += 1
        main.push(frame(phys_imgs("main", k), main_t(k), k))
        used_uw.add(src.next_packet().seq["ultrawide"])
    calls = src.stats()["remap_calls"]
    assert calls["main"] == 30
    assert calls["ultrawide"] == len(used_uw) < 30


def test_catch_up_and_drop_stats(rig, phys_imgs):
    """Live: a lagging consumer jumps to the newest main frame; seq gaps count as drops."""
    main, uw = FakeSource("main"), FakeSource("ultrawide")
    uw.push(frame(phys_imgs("ultrawide", 0), main_t(0), 0))
    main.push(frame(phys_imgs("main", 0), main_t(0), 0))
    src = make_src(rig, main, uw)
    src.start()
    for k in (1, 2, 3, 5):          # seq 4 lost before userspace
        main.push(frame(phys_imgs("main", k), main_t(k), k))
    uw.push(frame(phys_imgs("ultrawide", 1), main_t(5), 1))
    p = src.next_packet()
    assert p.seq["main"] == 5
    s = src.stats()
    assert s["catchup_skips"] == 3 and s["drops"]["main"] == 1


def test_latest_packet_fixed_rate(rig, phys_imgs):
    main, uw = FakeSource("main"), FakeSource("ultrawide")
    uw.push(frame(phys_imgs("ultrawide", 0), main_t(0), 0))
    main.push(frame(phys_imgs("main", 0), main_t(0), 0))
    src = make_src(rig, main, uw, frame_timeout_s=10.0)
    src.start()
    p0 = src.latest_packet()
    assert src.latest_packet() is p0                # no new main frame: same packet (same timestamp)
    main.push(frame(phys_imgs("main", 1), main_t(1), 1))
    main.push(frame(phys_imgs("main", 2), main_t(2), 2))
    p2 = src.latest_packet()
    assert p2.seq["main"] == 2 and p2.t_capture_s > p0.t_capture_s


def test_output_contract_and_coverage_guard(rig, phys_imgs):
    main, uw = FakeSource("main"), FakeSource("ultrawide")
    uw.push(frame(phys_imgs("ultrawide", 0), main_t(0), 0))
    main.push(frame(phys_imgs("main", 0), main_t(0), 0))
    src = make_src(rig, main, uw, output_sizes={"main": (480, 360), "ultrawide": (960, 720)})
    src.start()
    p = src.last_packet
    assert isinstance(p, Packet)
    assert p.images["main"].shape == (360, 480, 3) and p.images["ultrawide"].shape == (720, 960, 3)
    for img in p.images.values():
        assert img.dtype == np.uint8 and img.flags.c_contiguous
    with pytest.raises(ValueError, match="non-uniform"):
        make_src(rig, FakeSource(), FakeSource(), output_sizes={"main": (960, 540), "ultrawide": (960, 720)})
    with pytest.raises(ValueError, match="exactly the keys"):
        make_src(rig, FakeSource(), FakeSource(), output_sizes={"main": (960, 720)})
    with pytest.raises(ValueError, match="coverage"):
        make_src(rig, FakeSource(), FakeSource(), min_coverage=1.01)


def test_with_mock_backend(rig):
    """The real MockCamera backend through make_camera (synthetic pattern, realtime): live packets, both keys."""
    src = RigFrameSource(rig, backend="mock", backend_opts={"realtime": True, "loop": True, "restamp": True},
                         frame_timeout_s=1.0)
    with src:
        last = src.last_packet.t_capture_s
        for _ in range(5):
            p = src.next_packet()
            assert p.images["main"].shape == (720, 960, 3) and p.images["ultrawide"].shape == (720, 960, 3)
            assert p.t_capture_s > last and not p.repeated
            last = p.t_capture_s
        assert abs(src.stats()["skew_ms"]["max"]) < 34
