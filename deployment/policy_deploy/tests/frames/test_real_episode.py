# adapted from rnoro/deployment:deployment/process_frames/tests/test_real_episode.py @ 91b178b
# changes: the recording is paired by the nearest-frame rule (a main packet sent half an ultrawide period after
# capture), pairs judged against the PairRule window; JPEG at rig_pi_v2_640 (640x480 both, as the Pi streams); the
# shared Running from tests/frames_pipeline.py.
"""The real Pi recording ~/Downloads/episode_000045 (main imx519 1600x1200, ultrawide imx219 1280x960, both 30 FPS):
paired by the nearest-frame rule, and played by FakePiSender through the whole process to C1 at the deployment rig's
sizes. Skipped when the recording is absent. Read only: nothing is written next to it."""

from __future__ import annotations

import collections
import csv
from pathlib import Path

import numpy as np
import pytest
from frames_helpers import HALF_PERIOD_NS, V2_RIG
from frames_pipeline import Running, assert_every_frame_decoded_once, check_served

from policy_deploy.frames._dataproc.imageprep import ImagePrep
from policy_deploy.frames.packet_pairer import PacketPairer

EPISODE = Path("~/Downloads/episode_000045").expanduser()
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


def test_fake_pi_to_c1_at_640_with_the_real_recording(tmp_path):
    """FakePiSender (the recording re-encoded as the Pi's JPEGs at 640x480) -> receiver -> C1."""
    run = Running(tmp_path, EPISODE, rig_file=str(V2_RIG), mock={"max_seconds": 3.0})
    try:
        run.wait_ready(timeout_s=60)
        assert run.app.rig.rig_id == "rig_pi_v2_640"
        prep = ImagePrep(run.app.rig)
        gaps = []
        for _ in range(4):
            reply = run.wait_frames(timeout_s=20)
            assert reply.main.shape == (2, 224, 224, 3) and reply.ultrawide.shape == (2, 224, 224, 3)
            assert reply.main.dtype == np.uint8 and reply.health.main.status == "ok"
            gaps += check_served(run, reply, prep)                       # bit-exact, true times, inside the window
        assert_every_frame_decoded_once(run)
        rx, sent = run.app.receiver, run.app.fake_pi.sent
        print(f"\nreal episode live: {sent['packets']} packets, {sent['ultrawide']} ultrawide frames sent, "
              f"{rx.counters['ultrawide'].received} decoded, served pair gaps {min(gaps):.3f}..{max(gaps):.3f} ms")
        assert rx.counters["main"].received == sent["packets"] and rx.counters["main"].unusable == 0
    finally:
        run.stop()
