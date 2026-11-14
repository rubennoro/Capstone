# adapted from rnoro/deployment:deployment/process_frames/tests/test_pipeline.py @ 91b178b (Running, check_served,
# assert_every_frame_decoded_once and their helpers)
# changes: importable module, so test_pipeline, test_pi_v1, test_real_episode and test_frozen_ultrawide share them;
# JPEG only; times are laptop times (t_ns); a served pair's gap is mid-exposure based and inside the PairRule window.
"""The whole frames process inside a test: FramesApp with --mock settings, the real decoders and a C1 RpcClient."""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np
from deploy_contracts import RpcClient
from frames_helpers import fast_overrides, wait_for, write_station

from policy_deploy.contracts import FramesReply, GetFramesRequest, GetHealthRequest, HealthReply
from policy_deploy.frames._dataproc.imageprep import ImagePrep
from policy_deploy.frames.app import FramesApp
from policy_deploy.frames.settings import Settings

MS = 1_000_000


class Running:
    """FramesApp.run() in a thread, plus a C1 client."""

    def __init__(self, tmp_path, episode, **overrides):
        tmp_path.mkdir(parents=True, exist_ok=True)
        self.station = write_station(tmp_path / "station.yaml",
                                     frames=fast_overrides(mock_episode=str(episode), **overrides))
        self.settings = Settings.load(self.station, mock=True)
        self.ready_file = tmp_path / "ready" / "frames.ready"
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
        # app.ready is set just after the ready file is written: waiting for it implies the file
        wait_for(lambda: self.app.ready.is_set() or not self.thread.is_alive(), timeout_s, "ready")
        assert self.error is None, self.error
        assert self.ready_file.exists()
        self.client = RpcClient("frames", self.app.frames_endpoint)

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


def snapshot_by_time(app):
    return {p.t_ns: p for p in app.buffer.snapshot()}


def offline_rgb(recording_cam, index):
    """Decode one recorded JPEG frame, as the receiver's JpegDecoder does."""
    data = np.frombuffer(recording_cam.frames[index].data, np.uint8)
    return cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def pi_to_local(fake, t_pi_ns):
    """Invert the fake Pi's clock (a straight line)."""
    a0 = fake.pi_clock_ns(0)
    slope = (fake.pi_clock_ns(10**12) - a0) / 1e12
    return (t_pi_ns - a0) / slope


def check_served(run, reply, prep=None):
    """Every served image and time against the fake Pi's truth. Returns the served pairs' mid-exposure gaps (ms)."""
    app = run.app
    by_t = snapshot_by_time(app)                        # at once, before a served packet can leave the 1 s buffer
    prep = prep or ImagePrep(app.rig)
    gaps = []
    rule = app.receiver.pair_rule
    for cam, imgs, times in (("main", reply.main, reply.main_t_ns), ("ultrawide", reply.ultrawide,
                                                                      reply.ultrawide_t_ns)):
        rec = app.fake_pi.recording.cameras[cam]
        for img, t in zip(imgs, times.tolist()):
            p = by_t[t]                                 # ultrawide times are their packets' times
            # time: within 2 ms of the fake Pi's true capture time of the main frame (offset and drift removed)
            assert abs(t - pi_to_local(app.fake_pi, p.t_pi_mid_ns)) < 2_000_000
            # image: bit-identical to the offline prep of the same recorded frame (live path = training path)
            seq = p.main_seq if cam == "main" else p.pair_seq
            rgb = offline_rgb(rec, seq % len(rec.frames))
            ref = prep.prep_main(rgb) if cam == "main" else prep.prep_ultrawide(rgb)
            assert np.array_equal(img, ref), (cam, seq)
            if cam == "ultrawide":
                gaps.append(p.pair_gap_ns / 1e6)
                assert -rule.max_lead_ns / 1e6 <= gaps[-1] <= rule.max_age_ns / 1e6   # the PairRule window
    return gaps


def assert_every_frame_decoded_once(run):
    """Stop the fake Pi, let the receiver finish, then: every packet and new ultrawide frame sent was decoded exactly
    once (no more: nothing twice; no fewer: nothing lost)."""
    run.app.fake_pi.stop()
    rx, sent = run.app.receiver, run.app.fake_pi.sent
    c = rx.counters["ultrawide"]
    last = {"n": -1, "t": time.monotonic()}

    def done():                     # everything arrived, or nothing new for 1 s (then the asserts say what is missing)
        if rx.packets.received >= sent["packets"] and c.received >= sent["ultrawide"]:
            return True
        n = rx.packets.received + c.received
        if n != last["n"]:
            last.update(n=n, t=time.monotonic())
        return time.monotonic() - last["t"] > 1.0
    wait_for(done, 30, "the receiver to finish")
    assert (rx.packets.lost, c.dropped) == (0, 0), f"lost {rx.packets.lost} packets, {c.dropped} ultrawide frames"
    assert (rx.packets.received, c.received) == (sent["packets"], sent["ultrawide"])
    assert rx.counters["main"].received == sent["packets"] and rx.counters["main"].unusable == 0
