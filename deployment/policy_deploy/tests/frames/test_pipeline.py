# adapted from rnoro/deployment:deployment/process_frames/tests/test_pipeline.py @ 91b178b
# changes: pi_v1 + JPEG only (the h264 case and the keyframe-wait cases are gone with H.264: a lost frame now costs only
# that frame); the shared helpers moved to tests/frames_pipeline.py; the v2 rig is rig_pi_v2_640; an ultrawide pause
# makes the fake Pi repeat its last frame, so the PairRule refuses it (pair_reused) and the ultrawide goes stale; the
# station schema (probe gets a station with the same ports).
"""The whole process inside the test: FramesApp with --mock settings on 127.0.0.1, the real decoders, and a C1
RpcClient."""

from __future__ import annotations

import time

import pytest
from frames_helpers import V2_RIG, fast_overrides, wait_for, write_rig, write_station
from frames_pipeline import assert_every_frame_decoded_once, check_served

from policy_deploy.frames._dataproc.imageprep import ImagePrep
from policy_deploy.frames.app import FramesApp
from policy_deploy.frames.errors import FatalError
from policy_deploy.frames.probe import main as probe_main
from policy_deploy.frames.settings import Settings


def test_ready_valid_replies_true_times_and_bit_identical_images(running):
    run = running()
    run.wait_ready()
    assert run.app.ready.is_set()
    h = run.health()
    assert h.main.status == "ok" and h.ultrawide.status == "ok"
    for _ in range(3):
        reply = run.wait_frames()
        assert reply.main.shape == (2, 224, 224, 3) and reply.ultrawide.shape == (2, 224, 224, 3)
        spacing = int(reply.main_t_ns[1] - reply.main_t_ns[0])
        assert abs(spacing - 50_000_000) <= 17_000_000
        check_served(run, reply)
    assert run.app.server.served >= 3


def test_v2_sizes_end_to_end_no_ultrawide_frame_lost_or_decoded_twice(running, v2_episode):
    """FakePiSender -> receiver -> C1 at the real rig file's sizes (both cameras 640x480)."""
    run = running(v2_episode, rig_file=str(V2_RIG))
    run.wait_ready(timeout_s=30)
    assert run.app.rig.rig_id == "rig_pi_v2_640"
    prep = ImagePrep(run.app.rig)
    gaps = []
    for _ in range(3):
        reply = run.wait_frames(timeout_s=20)
        assert reply.main.shape == (2, 224, 224, 3) and reply.ultrawide.shape == (2, 224, 224, 3)
        gaps += check_served(run, reply, prep)
    assert gaps
    assert_every_frame_decoded_once(run)


def test_a_dropped_main_frame_is_counted_and_serving_goes_on(running):
    run = running()
    run.wait_ready()
    rx = run.app.receiver
    before = rx.counters["main"].dropped
    run.app.fake_pi.drop_frames("main", 1)
    wait_for(lambda: rx.counters["main"].dropped > before, 10, "the drop")
    assert rx.packets.lost >= 1                                          # no packet for the dropped main frame
    t_after = time.monotonic_ns()
    reply = run.wait_frames(lambda r: int(r.main_t_ns[0]) > t_after)    # the next frames are served at once
    check_served(run, reply)
    assert rx.counters["main"].unusable == 0


def test_a_dropped_packet_is_counted_and_serving_goes_on(running):
    run = running()
    run.wait_ready()
    rx = run.app.receiver
    run.app.fake_pi.drop_packets(1)
    wait_for(lambda: rx.packets.lost >= 1, 10, "the lost packet")
    assert rx.counters["main"].dropped >= 1
    t_after = time.monotonic_ns()
    reply = run.wait_frames(lambda r: int(r.main_t_ns[0]) > t_after)
    check_served(run, reply)


def test_health_ok_stale_ok_by_pause_and_ok_down_by_stop(running):
    run = running()
    run.wait_ready()
    run.app.fake_pi.pause("ultrawide")           # packets keep coming, repeating the last ultrawide frame
    wait_for(lambda: run.health().ultrawide.status == "stale", 10, "ultrawide stale")
    assert run.health().main.status == "ok"
    assert run.frames().main is None                                   # no images meanwhile
    assert run.app.receiver.packets.pair_reused > 0 and run.app.receiver.packets.uw_reused > 0
    run.app.fake_pi.resume("ultrawide")
    wait_for(lambda: run.health().ultrawide.status == "ok", 10, "ultrawide ok again")
    run.wait_frames()
    run.app.fake_pi.stop()                                              # the Pi is lost
    wait_for(lambda: run.health().main.status == "down" and run.health().ultrawide.status == "down", 10, "down")
    assert run.frames().main is None


def test_stats_line_and_probe(running, tmp_path, capsys):
    run = running()
    run.wait_ready()
    wait_for(lambda: (run.app.out_dir / "stats.jsonl").exists(), 10, "stats.jsonl")
    line = run.app.stats_line(run.app.stats(time.monotonic_ns()))
    assert line.startswith("stats main ") and "| clock offset" in line and "served" in line
    assert "| packets lost " in line and "pair gap" in line and "unpaired" in line and "uw_reused" in line
    run.wait_frames()
    assert probe_main(["--station", str(run.station), "--mock", "--save", str(tmp_path / "png")]) == 0
    assert sorted(p.name for p in (tmp_path / "png").iterdir()) == ["main_0.png", "main_1.png", "ultrawide_0.png",
                                                                    "ultrawide_1.png"]
    assert "main" in capsys.readouterr().out


def test_wrong_sensor_in_the_rig_is_fatal(tmp_path, pi_episode):
    def swap(d):
        d["cameras"]["main"]["sensor"] = "imx219"
        d["cameras"]["ultrawide"]["sensor"] = "imx519"
    rig = write_rig(tmp_path, "rig_swapped", swap)
    station = write_station(tmp_path / "station.yaml",
                            frames=fast_overrides(mock_episode=str(pi_episode), rig_file=str(rig)))
    app = FramesApp(Settings.load(station, mock=True), None, tmp_path / "out")
    with pytest.raises(FatalError) as e:
        app.run()
    assert "stream sensor is" in str(e.value) and "rig file says" in str(e.value)
