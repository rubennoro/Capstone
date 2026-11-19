"""FramesServers: R1 refusals, stop and health; U2 only newer frames (STRUCTURE.md 4.2, 4.3, 5.3). New.

Real REP/REQ sockets on free ports (tcp://127.0.0.1:*); the servers are served from the test's thread, as the frames
main thread does, while the client waits in a helper thread.
"""

from __future__ import annotations

import threading

import pytest
from deploy_contracts import Ack, RemoteError, RpcClient
from frames_helpers import MAIN_SIZE, UW_SIZE, FakeClock, raw_frame, read_mkv

from teleop_collect.contracts import (ClockMap, ClockSample, FramesHealth, GetFramesHealth, GetPreview, Preview,
                                      RecordingSummary, StartRecording, StopRecording)
from teleop_collect.frames.camera_health import CameraHealth
from teleop_collect.frames.preview import LatestFrames
from teleop_collect.frames.recorder import FrameRecorder, RecordItem
from teleop_collect.frames.servers import FramesServers

OFFSET = 1_000_000_000_000
P = 33_333_333
MS = 1_000_000


class Harness:
    def __init__(self):
        self.clock = FakeClock()
        self.map = ClockMap(60.0, 0.3, 10.0, 500.0, 10.0)
        self.health = {c: CameraHealth(c, 0.35, 2.0, 1.0, self.clock) for c in ("main", "ultrawide")}
        self.latest = LatestFrames(self.health, self.clock)
        self.recorder = FrameRecorder(90, 1.0, {"main": MAIN_SIZE, "ultrawide": UW_SIZE}, 30, self.map, self.clock)
        self.servers = FramesServers("tcp://127.0.0.1:*", "tcp://127.0.0.1:*", self.recorder, self.latest,
                                     self.health, self.map, self.clock)
        self.r1 = RpcClient("frames", self.servers.r1_endpoint)
        self.u2 = RpcClient("frames", self.servers.preview_endpoint)
        self.n = 0

    def ask(self, client, msg, expect):
        box = {}

        def call():
            try:
                box["reply"] = client.request(msg, expect, 5.0)
            except Exception as e:               # noqa: BLE001 (re-raised below, in the test's thread)
                box["error"] = e
        t = threading.Thread(target=call)
        t.start()
        while t.is_alive():
            self.servers.serve_once(0.02)
        if "error" in box:
            raise box["error"]
        return box["reply"]

    def sync(self):
        self.map.add_sample(ClockSample(t_send_ns=self.clock(), t_remote_ns=self.clock() + OFFSET,
                                        t_recv_ns=self.clock()))

    def frames_ok(self):
        for cam in ("main", "ultrawide"):
            self.health[cam].record_frame(self.clock())

    def packet(self, new_uw=True):
        """One RecordItem into the preview and the recorder, as the receiver hands it."""
        i = self.n
        self.n += 1
        t = self.clock() + OFFSET
        uw = raw_frame("ultrawide", i, t) if new_uw else None
        it = RecordItem(packet_no=i, t_send_pi_ns=t + 45 * MS, arrival_ns=self.clock(), main=raw_frame("main", i, t),
                        main_t_mid_pi_ns=t, main_t_mid_ns=self.clock(), ultrawide=uw,
                        ultrawide_t_mid_pi_ns=t if uw else None, ultrawide_t_mid_ns=self.clock() if uw else None,
                        ultrawide_frame_no=i if uw else i - 1, uw_reused=not new_uw, pair_used=True, pair_gap_ms=0.0)
        self.latest.update(it)
        self.recorder.enqueue(it)
        self.clock.advance(P)
        return it

    def close(self):
        self.r1.close()
        self.u2.close()
        if self.recorder.recording:
            self.recorder.stop()
        self.servers.close()


@pytest.fixture
def h():
    harness = Harness()
    yield harness
    harness.close()


def start(h, folder):
    return h.ask(h.r1, StartRecording(episode_dir=str(folder)), Ack)


def test_start_refusals_in_order(h, tmp_path):
    ep = tmp_path / "ep"
    ep.mkdir()
    assert start(h, ep).reason == "main camera stale"               # no frame yet
    h.health["main"].record_frame(h.clock())
    assert start(h, ep).reason == "ultrawide camera stale"
    h.frames_ok()
    assert start(h, ep).reason == "Pi clock not synced"
    h.sync()
    assert start(h, tmp_path / "nope").reason == f"{tmp_path / 'nope'}: not a folder"
    (ep / "main.mkv").write_bytes(b"x")
    assert start(h, ep).reason == f"{ep}: already has recording files"
    (ep / "main.mkv").unlink()
    (ep / "robot_states.csv").write_text("teleop's\n")              # teleop's own files are expected there
    reply = start(h, ep)
    assert reply.ok and reply.reason == ""
    assert start(h, tmp_path).reason == "already recording"
    h.clock.advance(2001 * MS)                                      # down while recording: refused anyway
    assert start(h, tmp_path).reason == "already recording"


def test_a_down_camera_is_named(h, tmp_path):
    h.frames_ok()
    h.sync()
    h.clock.advance(2001 * MS)
    h.health["main"].record_frame(h.clock())
    h.sync()
    assert start(h, tmp_path).reason == "ultrawide camera down"


def test_record_stop_and_health(h, tmp_path):
    ep = tmp_path / "ep"
    ep.mkdir()
    h.frames_ok()
    h.sync()
    fh = h.ask(h.r1, GetFramesHealth(), FramesHealth)
    assert (fh.recording, fh.synced, fh.frames_main, fh.recording_error) == (False, True, 0, "")
    assert start(h, ep).ok
    items = [h.packet(), h.packet(), h.packet(new_uw=False), h.packet()]
    fh = h.ask(h.r1, GetFramesHealth(), FramesHealth)
    assert fh.recording and fh.health.main.status == "ok"
    summary = h.ask(h.r1, StopRecording(), RecordingSummary)
    assert summary.packets == 4 and summary.main.frames == 4 and summary.ultrawide.frames == 3
    assert summary.uw_reused == 1 and summary.error == ""
    assert read_mkv(ep / "main.mkv") == [it.main.data for it in items]
    fh = h.ask(h.r1, GetFramesHealth(), FramesHealth)
    assert not fh.recording and fh.frames_main == 0
    with pytest.raises(RemoteError, match="not recording"):
        h.ask(h.r1, StopRecording(), RecordingSummary)


def test_health_shows_a_recording_error(h, tmp_path, monkeypatch):
    ep = tmp_path / "ep"
    ep.mkdir()
    h.frames_ok()
    h.sync()
    assert start(h, ep).ok
    session = h.recorder._session
    session.fail("disk error: test")
    fh = h.ask(h.r1, GetFramesHealth(), FramesHealth)
    assert fh.recording and fh.recording_error == "disk error: test"
    assert h.ask(h.r1, StopRecording(), RecordingSummary).error == "disk error: test"


def test_u2_sends_only_newer_frames(h):
    p = h.ask(h.u2, GetPreview(main_after=-1, ultrawide_after=-1), Preview)
    assert (p.main_jpeg, p.ultrawide_jpeg, p.main_frame_no, p.ultrawide_frame_no) == (None, None, -1, -1)
    assert p.health.main.status == "stale"
    a = h.packet()
    p = h.ask(h.u2, GetPreview(main_after=-1, ultrawide_after=-1), Preview)
    assert p.main_jpeg == a.main.data and p.ultrawide_jpeg == a.ultrawide.data
    assert (p.main_frame_no, p.ultrawide_frame_no) == (0, 0)
    p = h.ask(h.u2, GetPreview(main_after=0, ultrawide_after=0), Preview)
    assert p.main_jpeg is None and p.ultrawide_jpeg is None and p.main_frame_no == 0
    b = h.packet(new_uw=False)                                     # a reused ultrawide: only main is new
    p = h.ask(h.u2, GetPreview(main_after=0, ultrawide_after=0), Preview)
    assert p.main_jpeg == b.main.data and p.ultrawide_jpeg is None and p.ultrawide_frame_no == 0
    p = h.ask(h.u2, GetPreview(main_after=1, ultrawide_after=-1), Preview)
    assert p.main_jpeg is None and p.ultrawide_jpeg == a.ultrawide.data
    h.frames_ok()                                                   # health comes along with every reply
    assert h.ask(h.u2, GetPreview(main_after=9, ultrawide_after=9), Preview).health.main.status == "ok"


def test_each_server_answers_only_its_contract(h):
    with pytest.raises(RemoteError):
        h.ask(h.u2, GetFramesHealth(), FramesHealth)
    with pytest.raises(RemoteError):
        h.ask(h.r1, GetPreview(main_after=-1, ultrawide_after=-1), Preview)
    assert h.servers.requests["r1"] >= 1 and h.servers.requests["u2"] >= 1
