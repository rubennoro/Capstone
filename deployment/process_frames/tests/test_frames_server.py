"""FramesServer: C1 replies with hand-filled buffers and health (STRUCTURE.md §6.2)."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from contracts import (FramesReply, GetFramesRequest, GetHealthRequest, HealthReply, RpcClient, decode, encode)

from process_frames._dataproc.imageprep import ImagePrep
from process_frames.camera_health import CameraHealth
from process_frames.frame_buffer import BufferedFrame, FrameBuffer
from process_frames.frame_selector import FrameSelector
from process_frames.frames_server import FramesServer

MS = 1_000_000
P = 33_333_333


class Setup:
    def __init__(self, rig, clock):
        self.clock = clock
        self.buffers = {c: FrameBuffer(c, 1.0) for c in ("main", "ultrawide")}
        self.health = {c: CameraHealth(c, 0.35, 2.0, 1.0, clock) for c in ("main", "ultrawide")}
        self.prep = ImagePrep(rig)
        self.rig = rig
        self.server = FramesServer("tcp://127.0.0.1:*", self.buffers, self.health, FrameSelector(25 * MS),
                                   self.prep, 50 * MS, 100 * MS, clock)

    def fill(self, cam, n=30, offset_ns=0, drop=()):
        w, h = self.rig.cameras[cam].resolution
        rng = np.random.default_rng(len(cam))
        start = self.clock() - (n - 1) * P - 5 * MS + offset_ns
        for i in range(n):
            if i in drop:
                continue
            t = start + i * P
            rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
            self.buffers[cam].push(BufferedFrame(cam, i, t, t, t, rgb))
            self.health[cam].record_frame(t)

    def close(self):
        self.server.close()


@pytest.fixture
def setup(small_rig, fake_clock):
    s = Setup(small_rig, fake_clock)
    yield s
    s.close()


def roundtrip(reply):
    """Encode, decode with expect: the reply passes the C1 validator."""
    return decode(encode(reply), expect=type(reply))


def test_reply_with_images_passes_the_contract_and_matches_imageprep(setup):
    setup.fill("main")
    setup.fill("ultrawide", offset_ns=11 * MS)
    reply = roundtrip(setup.server.get_frames(GetFramesRequest()))
    assert isinstance(reply, FramesReply)
    assert reply.main.shape == (2, 224, 224, 3) and reply.ultrawide.shape == (2, 224, 224, 3)
    main = setup.buffers["main"].snapshot()
    anchor = main[-1].t_pc_ns
    picked_main = [f for f in main if f.t_pc_ns in reply.main_t_ns.tolist()]
    assert reply.main_t_ns[-1] == anchor and len(picked_main) == 2
    assert abs(int(reply.main_t_ns[0]) - (anchor - 50 * MS)) <= 17 * MS
    for img, f in zip(reply.main, picked_main):
        assert np.array_equal(img, setup.prep.prep_main(f.rgb))
    uw = {f.t_pc_ns: f for f in setup.buffers["ultrawide"].snapshot()}
    for img, t, target in zip(reply.ultrawide, reply.ultrawide_t_ns.tolist(), [anchor - 100 * MS, anchor]):
        assert np.array_equal(img, setup.prep.prep_ultrawide(uw[t].rgb))
        assert abs(t - target) <= 17 * MS
    assert reply.health.main.status == "ok" and reply.health.ultrawide.status == "ok"
    assert setup.server.served == 1 and setup.server.misses == 0


@pytest.mark.parametrize("cam,age_s,status", [("ultrawide", 0.5, "stale"), ("main", 2.5, "down")])
def test_both_or_nothing(setup, cam, age_s, status):
    setup.fill("main")
    setup.fill("ultrawide")
    setup.clock.advance_s(age_s)
    other = "main" if cam == "ultrawide" else "ultrawide"
    setup.health[other].record_frame(setup.clock())          # the other camera is fine
    reply = roundtrip(setup.server.get_frames(GetFramesRequest()))
    assert reply.main is None and reply.ultrawide is None and reply.main_t_ns is None and reply.ultrawide_t_ns is None
    assert getattr(reply.health, cam).status == status and getattr(reply.health, other).status == "ok"


def test_a_failed_pick_gives_no_images_and_that_camera_stale_in_that_reply(setup):
    setup.fill("main")
    setup.fill("ultrawide", drop=(25, 26))                   # nothing near anchor - 100 ms
    reply = roundtrip(setup.server.get_frames(GetFramesRequest()))
    assert reply.main is None and reply.ultrawide is None
    assert reply.health.ultrawide.status == "stale" and reply.health.main.status == "ok"
    assert setup.server.misses == 1 and setup.server.served == 0
    # health itself is unchanged
    assert setup.health["ultrawide"].status() == "ok"
    assert setup.server.get_health(GetHealthRequest()).health.ultrawide.status == "ok"


def test_get_health(setup):
    reply = roundtrip(setup.server.get_health(GetHealthRequest()))
    assert isinstance(reply, HealthReply)
    assert reply.health.main.status == "stale" and reply.health.main.fps == 0.0      # before any frame


def test_over_the_socket(setup):
    setup.fill("main")
    setup.fill("ultrawide")
    assert setup.server.endpoint.startswith("tcp://127.0.0.1:") and not setup.server.endpoint.endswith("*")
    stop = threading.Event()

    def serve():
        while not stop.is_set():
            setup.server.serve_once(0.02)

    t = threading.Thread(target=serve)
    t.start()
    client = RpcClient("process_frames", setup.server.endpoint)
    try:
        frames = client.request(GetFramesRequest(), FramesReply, 5.0)
        health = client.request(GetHealthRequest(), HealthReply, 5.0)
    finally:
        client.close()
        stop.set()
        t.join(5)
    assert frames.main.shape == (2, 224, 224, 3) and health.health.main.status == "ok"
