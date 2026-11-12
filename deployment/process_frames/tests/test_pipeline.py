"""The whole process inside the test: FramesApp with --mock settings on 127.0.0.1:*, the real decoders, and a C1
RpcClient (STRUCTURE.md §13.2)."""

from __future__ import annotations

import threading
import time

import av
import numpy as np
import pytest

from contracts import FramesReply, GetFramesRequest, GetHealthRequest, HealthReply, RpcClient

from conftest import fast_overrides, wait_for, write_rig, write_station
from process_frames._dataproc.imageprep import ImagePrep
from process_frames.app import FramesApp
from process_frames.errors import FatalError
from process_frames.probe import main as probe_main
from process_frames.settings import Settings

KEYINT = 10


class Running:
    """FramesApp.run() in a thread, plus a C1 client."""

    def __init__(self, tmp_path, episode, **overrides):
        self.station = write_station(tmp_path / "station.yaml",
                                     process_frames=fast_overrides(mock_episode=str(episode), **overrides))
        self.settings = Settings.load(self.station, mock=True)
        self.ready_file = tmp_path / "ready" / "process_frames.ready"
        self.app = FramesApp(self.settings, self.ready_file, tmp_path / "out")
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run)
        self.thread.start()
        self.client: RpcClient | None = None

    def _run(self):
        try:
            self.app.run()
        except BaseException as e:
            self.error = e

    def wait_ready(self, timeout_s=15.0):
        wait_for(lambda: self.ready_file.exists() or not self.thread.is_alive(), timeout_s, "ready")
        assert self.error is None, self.error
        self.client = RpcClient("process_frames", self.app.frames_endpoint)

    def frames(self) -> FramesReply:
        return self.client.request(GetFramesRequest(), FramesReply, 5.0)

    def health(self):
        return self.client.request(GetHealthRequest(), HealthReply, 5.0).health

    def wait_frames(self, condition=lambda r: True, timeout_s=10.0) -> FramesReply:
        box = {}

        def ok():
            r = self.frames()
            if r.main is not None and condition(r):
                box["r"] = r
                return True
            return False
        wait_for(ok, timeout_s, "a reply with images")
        return box["r"]

    def stop(self):
        if self.client is not None:
            self.client.close()
        self.app.request_stop()
        self.thread.join(15)
        assert not self.thread.is_alive()


@pytest.fixture
def running(tmp_path, pi_episode):
    runs = []

    def start(**overrides):
        r = Running(tmp_path, pi_episode, **overrides)
        runs.append(r)
        return r
    yield start
    for r in runs:
        r.stop()


def buffered(app, cam, t_pc_ns):
    return next(f for f in app.buffers[cam].snapshot() if f.t_pc_ns == t_pc_ns)


def offline_rgb(recording_cam, index):
    """Decode the recorded C0 stream from its start, as dataproc's read_video_frames decodes a file."""
    if recording_cam.frames[0].encoding == "jpeg":
        import cv2
        data = np.frombuffer(recording_cam.frames[index].data, np.uint8)
        return cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    ctx = av.CodecContext.create("h264", "r")
    out = None
    for f in recording_cam.frames[:index + 1]:
        for pic in ctx.decode(av.Packet(f.data)):
            out = pic.to_ndarray(format="rgb24")
    for pic in ctx.decode(None):
        out = pic.to_ndarray(format="rgb24")
    return out


def pi_to_pc(fake, t_pi_ns):
    """Invert the fake Pi's clock (a straight line)."""
    a0 = fake.pi_clock_ns(0)
    slope = (fake.pi_clock_ns(10**12) - a0) / 1e12
    return (t_pi_ns - a0) / slope


def check_served(run, reply):
    app = run.app
    prep = ImagePrep(app.rig)
    for cam, imgs, times in (("main", reply.main, reply.main_t_ns), ("ultrawide", reply.ultrawide,
                                                                      reply.ultrawide_t_ns)):
        rec = app.fake_pi.recording.cameras[cam]
        for img, t in zip(imgs, times.tolist()):
            f = buffered(app, cam, t)
            # time: within 2 ms of the fake Pi's true capture time (offset and drift removed)
            assert abs(t - pi_to_pc(app.fake_pi, f.t_pi_mid_ns)) < 2_000_000
            # image: bit-identical to the offline prep of the same recorded frame (live path = training path)
            rgb = offline_rgb(rec, f.seq % len(rec.frames))
            ref = prep.prep_main(rgb) if cam == "main" else prep.prep_ultrawide(rgb)
            assert np.array_equal(img, ref), (cam, f.seq)


@pytest.mark.parametrize("decoder", ["h264", "jpeg"])
def test_ready_valid_replies_true_times_and_bit_identical_images(running, decoder):
    run = running(decoder=decoder)
    run.wait_ready()
    assert run.app.ready.is_set()
    h = run.health()
    assert h.main.status == "ok" and h.ultrawide.status == "ok"
    # wait until both cameras have several frames, then check a few replies
    for _ in range(3):
        reply = run.wait_frames()
        assert reply.main.shape == (2, 224, 224, 3) and reply.ultrawide.shape == (2, 224, 224, 3)
        spacing = int(reply.main_t_ns[1] - reply.main_t_ns[0])
        assert abs(spacing - 50_000_000) <= 17_000_000
        check_served(run, reply)
    assert run.app.server.served >= 3


def test_dropped_h264_frame_nothing_after_it_until_the_next_keyframe(running):
    run = running()
    run.wait_ready()
    rx = run.app.receiver
    before = rx.counters["main"].dropped
    run.app.fake_pi.drop_next("main", 1)
    wait_for(lambda: rx.counters["main"].dropped > before, 10, "the drop")
    gap_seen = {}

    def recovered():
        seqs = [f.seq for f in run.app.buffers["main"].snapshot()]
        gaps = [(a, b) for a, b in zip(seqs, seqs[1:]) if b - a > 1]
        if gaps:
            gap_seen["gaps"] = gaps
        return bool(gaps) and seqs[-1] > gaps[-1][1]
    wait_for(recovered, 10, "a buffered keyframe after the drop")
    for a, b in gap_seen["gaps"]:
        assert b % KEYINT == 0, f"first buffered frame after the gap ({a} -> {b}) is not a keyframe"
    assert rx.counters["main"].unusable >= 1
    gap_end = gap_seen["gaps"][-1][1]
    t_key = next(f.t_pc_ns for f in run.app.buffers["main"].snapshot() if f.seq == gap_end)
    reply = run.wait_frames(lambda r: int(r.main_t_ns[-1]) >= t_key)        # serving resumes
    check_served(run, reply)


def test_health_ok_stale_ok_by_pause_and_ok_down_by_stop(running):
    run = running()
    run.wait_ready()
    run.app.fake_pi.pause("ultrawide")
    wait_for(lambda: run.health().ultrawide.status == "stale", 10, "ultrawide stale")
    assert run.health().main.status == "ok"
    assert run.frames().main is None                                   # no images meanwhile
    run.app.fake_pi.resume("ultrawide")
    wait_for(lambda: run.health().ultrawide.status == "ok", 10, "ultrawide ok again")
    run.wait_frames()
    run.app.fake_pi.stop()                                              # the Pi is lost
    wait_for(lambda: run.health().main.status == "down" and run.health().ultrawide.status == "down", 10, "down")
    assert run.frames().main is None


def test_stats_line_and_probe(running, tmp_path, capsys):
    run = running()
    run.wait_ready()
    wait_for(lambda: (tmp_path / "out" / "stats.jsonl").exists(), 10, "stats.jsonl")
    line = run.app.stats_line(run.app.stats(time.monotonic_ns()))
    assert line.startswith("stats main ") and "| clock offset" in line and "served" in line
    probe_station = write_station(tmp_path / "probe_station.yaml", endpoints={"frames": run.app.frames_endpoint})
    run.wait_frames()
    assert probe_main(["--station", str(probe_station), "--save", str(tmp_path / "png")]) == 0
    assert sorted(p.name for p in (tmp_path / "png").iterdir()) == ["main_0.png", "main_1.png", "ultrawide_0.png",
                                                                    "ultrawide_1.png"]
    assert "main" in capsys.readouterr().out


def test_wrong_sensor_in_the_rig_is_fatal(tmp_path, pi_episode):
    def swap(d):
        d["cameras"]["main"]["sensor"] = "imx219"
        d["cameras"]["ultrawide"]["sensor"] = "imx519"
    rig = write_rig(tmp_path, "rig_swapped", swap)
    station = write_station(tmp_path / "station.yaml",
                            process_frames=fast_overrides(mock_episode=str(pi_episode), rig_file=str(rig)))
    app = FramesApp(Settings.load(station, mock=True), None, tmp_path / "out")
    with pytest.raises(FatalError) as e:
        app.run()
    assert "stream sensor is" in str(e.value) and "rig file says" in str(e.value)
