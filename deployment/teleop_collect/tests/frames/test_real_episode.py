# adapted from rnoro/final-deployment:deployment/policy_deploy/tests/frames/test_real_episode.py @ ea9f7fb
# changes: imports; the end-to-end case records an episode through R1 (the .mkv frames are the JPEGs sent, none
# lost or twice) instead of serving C1; the pairing case unchanged.
# adapted from rnoro/deployment:deployment/process_frames/tests/test_real_episode.py @ 91b178b
# changes: the recording is paired by the nearest-frame rule (a main packet sent half an ultrawide period after
# capture), pairs judged against the PairRule window; JPEG at rig_pi_v3_640 (640x480 both, as the Pi streams); the
# shared Running from tests/frames_pipeline.py.
"""A real Pi recording (station_helpers.RECORDED_EPISODE; main imx519 1600x1200, ultrawide imx219 1280x960, 30 FPS):
paired by the nearest-frame rule, and played by FakePiSender through the whole process and recorded through R1 at the
deployment rig's sizes. Skipped when the recording is absent. Read only: nothing is written next to it."""

from __future__ import annotations

import collections
import csv

import numpy as np
import pytest
from frames_helpers import EPISODE, HALF_PERIOD_NS, LIVE_RIG, Running, read_csv, read_mkv, wait_for

from teleop_collect.frames.packet_pairer import PacketPairer
P_MS = 1000 / 30

pytestmark = pytest.mark.skipif(not (EPISODE / "meta.json").is_file(), reason=f"{EPISODE} not present")


def csv_times(name):
    with (EPISODE / name).open(newline="") as fh:
        return [int(row[1]) for row in list(csv.reader(fh))[1:]]


def test_the_whole_recording_pairs_by_the_nearest_frame():
    """Every recorded frame time paired as the fake Pi does: one packet per main frame, every ultrawide frame once and
    in order, pairs at most half a period apart (plus jitter)."""
    main_t, uw_t = csv_times("cam1_frames.csv"), csv_times("cam0_frames.csv")       # cam1 = imx519, cam0 = imx219
    events = sorted([(t, 0, j) for j, t in enumerate(uw_t)] + [(t + HALF_PERIOD_NS, 1, i) for i, t in enumerate(main_t)])
    pairer = PacketPairer()
    packets = []
    for _, is_main, k in events:
        if is_main:
            items, pair = pairer.take(main_t[k])
            packets.append((k, items, pair))
        else:
            pairer.add_ultrawide(k, uw_t[k], k)
    sent = [j for _, items, _ in packets for j in items]
    assert len(packets) == len(main_t) and sent == list(range(len(sent)))
    assert len(sent) >= len(uw_t) - 1                    # all but ultrawide frames after the last main frame
    inside = [(k, pair) for k, _, pair in packets if pair >= 0 and uw_t[0] <= main_t[k] <= uw_t[-1]]
    gaps = np.array([(main_t[k] - uw_t[pair]) / 1e6 for k, pair in inside])
    per_packet = collections.Counter(len(items) for _, items, _ in packets)
    unpaired = [k for k, _, pair in packets if pair < 0]
    print(f"\nreal episode: {len(packets)} packets, pair gap min/median/max {gaps.min():.3f}/{np.median(gaps):.3f}/"
          f"{gaps.max():.3f} ms, ultrawide frames per packet {dict(sorted(per_packet.items()))}, unpaired "
          f"packets {unpaired}")
    assert np.all(np.abs(gaps) <= P_MS / 2 + 5.0)        # nearest: at most half a period (+ jitter)
    assert set(per_packet) <= {0, 1, 2}
    assert all(main_t[k] + HALF_PERIOD_NS < uw_t[0] for k in unpaired)   # only before the first ultrawide frame


def test_fake_pi_to_an_episode_at_640_with_the_real_recording(tmp_path):
    """FakePiSender (the recording re-encoded as the Pi's JPEGs at 640x480) -> receiver -> R1 recording: every
    .mkv frame is the JPEG sent, every frame of the stream is written, none twice."""
    run = Running(tmp_path / "run", EPISODE, rig_file=str(LIVE_RIG), mock={"max_seconds": 3.0})
    try:
        run.wait_ready(timeout_s=60)
        assert run.app.rig.rig_id == "rig_pi_v3_640"
        ep = tmp_path / "ep"
        ep.mkdir()
        assert run.start_recording(ep).ok
        wait_for(lambda: run.health().frames_main >= 60, 20, "2 s recorded")
        s = run.stop_recording()
    finally:
        run.stop()
    rec = run.app.fake_pi.recording
    for cam in ("main", "ultrawide"):
        frames = rec.cameras[cam].frames
        _, rows = read_csv(ep / f"{cam}_frames.csv")
        video = read_mkv(ep / f"{cam}.mkv")
        assert len(video) == len(rows) == getattr(s, cam).frames
        assert all(d == frames[int(r["frame_no"]) % len(frames)].data for r, d in zip(rows, video))
        nos = [int(r["frame_no"]) for r in rows]
        assert nos == sorted(set(nos)) and getattr(s, cam).frame_no_gaps == 0
    assert s.error == "" and s.packet_no_gaps == 0 and s.packets == s.main.frames
    print(f"\nreal episode recorded: {s.packets} packets, {s.ultrawide.frames} ultrawide frames, "
          f"{s.main.bytes / s.main.frames / 1000:.0f} / {s.ultrawide.bytes / s.ultrawide.frames / 1000:.0f} kB per JPEG")
