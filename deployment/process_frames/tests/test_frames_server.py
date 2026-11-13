"""FramesServer: C1 replies with a hand-filled packet buffer and health (STRUCTURE.md §6.2)."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from contracts import (FramesReply, GetFramesRequest, GetHealthRequest, HealthReply, RpcClient, decode, encode)

from process_frames._dataproc.imageprep import ImagePrep
from process_frames.camera_health import CameraHealth
from process_frames.frames_server import FramesServer
from process_frames.packet_buffer import BufferedPacket, PacketBuffer
from process_frames.packet_selector import PacketSelector

MS = 1_000_000
P = 33_333_333


class Setup:
    def __init__(self, rig, clock):
        self.clock = clock
        self.buffer = PacketBuffer(1.0)
        self.health = {c: CameraHealth(c, 0.35, 2.0, 1.0, clock) for c in ("main", "ultrawide")}
        self.prep = ImagePrep(rig)
        self.rig = rig
        selector = PacketSelector({c: rig.cameras[c].fps for c in self.health},
                                  {"main": 50 * MS, "ultrawide": 100 * MS}, 2, 2.0)
        self.server = FramesServer("tcp://127.0.0.1:*", self.buffer, self.health, selector, self.prep, clock)

    def image(self, cam, rng):
        w, h = self.rig.cameras[cam].resolution
        return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)

    def fill(self, n=30, drop=(), no_uw=()):
        rng = np.random.default_rng(3)
        start = self.clock() - (n - 1) * P - 5 * MS
        for i in range(n):
            if i in drop:
                continue
            t = start + i * P
            uw = None if i in no_uw else self.image("ultrawide", rng)
            self.buffer.push(BufferedPacket(packet_seq=i, t_pc_ns=t, t_pi_mid_ns=t, arrival_ns=t, main_seq=i,
                                            main_t_pi_ns=t, main_rgb=self.image("main", rng),
                                            pair_seq=-1 if uw is None else i,
                                            pair_t_pi_ns=None if uw is None else t - 8 * MS, ultrawide_rgb=uw))
            self.health["main"].record_frame(t)
            if uw is not None:
                self.health["ultrawide"].record_frame(t)

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
    setup.fill()
    reply = roundtrip(setup.server.get_frames(GetFramesRequest()))
    assert isinstance(reply, FramesReply)
    assert reply.main.shape == (2, 224, 224, 3) and reply.ultrawide.shape == (2, 224, 224, 3)
    by_t = {p.t_pc_ns: p for p in setup.buffer.snapshot()}
    anchor = setup.buffer.newest().t_pc_ns
    assert reply.main_t_ns[-1] == anchor and reply.ultrawide_t_ns[-1] == anchor      # the packets' times
    assert abs(int(reply.main_t_ns[0]) - (anchor - 50 * MS)) <= 17 * MS
    assert abs(int(reply.ultrawide_t_ns[0]) - (anchor - 100 * MS)) <= 17 * MS
    for img, t in zip(reply.main, reply.main_t_ns.tolist()):
        assert np.array_equal(img, setup.prep.prep_main(by_t[t].main_rgb))
    for img, t in zip(reply.ultrawide, reply.ultrawide_t_ns.tolist()):
        assert np.array_equal(img, setup.prep.prep_ultrawide(by_t[t].ultrawide_rgb))
    assert reply.health.main.status == "ok" and reply.health.ultrawide.status == "ok"
    assert setup.server.served == 1 and setup.server.misses == 0


@pytest.mark.parametrize("cam,age_s,status", [("ultrawide", 0.5, "stale"), ("main", 2.5, "down")])
def test_both_or_nothing(setup, cam, age_s, status):
    setup.fill()
    setup.clock.advance_s(age_s)
    other = "main" if cam == "ultrawide" else "ultrawide"
    setup.health[other].record_frame(setup.clock())          # the other camera is fine
    reply = roundtrip(setup.server.get_frames(GetFramesRequest()))
    assert reply.main is None and reply.ultrawide is None and reply.main_t_ns is None and reply.ultrawide_t_ns is None
    assert getattr(reply.health, cam).status == status and getattr(reply.health, other).status == "ok"


@pytest.mark.parametrize("kw", [dict(drop=(24, 25, 26, 27, 28)), dict(no_uw=(26,))])
def test_a_failed_pick_gives_no_images_and_that_camera_stale_in_that_reply(setup, kw):
    setup.fill(**kw)                     # nothing near anchor - 100 ms, or that packet has no paired ultrawide
    reply = roundtrip(setup.server.get_frames(GetFramesRequest()))
    assert reply.main is None and reply.ultrawide is None
    assert reply.health.ultrawide.status == "stale"
    assert setup.server.misses == 1 and setup.server.served == 0
    # health itself is unchanged
    assert setup.health["ultrawide"].status() == "ok"
    assert setup.server.get_health(GetHealthRequest()).health.ultrawide.status == "ok"


def test_get_health(setup):
    reply = roundtrip(setup.server.get_health(GetHealthRequest()))
    assert isinstance(reply, HealthReply)
    assert reply.health.main.status == "stale" and reply.health.main.fps == 0.0      # before any packet


def test_over_the_socket(setup):
    setup.fill()
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
