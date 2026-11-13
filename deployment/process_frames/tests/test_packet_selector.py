"""PacketSelector: exactly iPhUMI's get_downsampled_camera_images over packets (STRUCTURE.md §6.2), compared with
the copied reference on the same timestamps, with jitter and dropped packets."""

from __future__ import annotations

import re

import numpy as np
import pytest

from process_frames import FOLDER
from process_frames.packet_buffer import BufferedPacket
from process_frames.packet_selector import PacketSelector
from reference_umi_utils import get_downsampled_camera_images

MS = 1_000_000
P = 33_333_333
FPS = {"main": 30.0, "ultrawide": 30.0}
SPACING = {"main": 50 * MS, "ultrawide": 100 * MS}
OBS_HZ = {"main": 20.0, "ultrawide": 10.0}
IMG = np.zeros((2, 2, 3), np.uint8)


def selector():
    return PacketSelector(FPS, SPACING, count=2, tolerance_frames=2.0)


def packets(times_ns, no_main=(), no_uw=()):
    return [BufferedPacket(packet_seq=i, t_pc_ns=int(t), t_pi_mid_ns=int(t), arrival_ns=int(t), main_seq=i,
                           main_t_pi_ns=int(t), main_rgb=None if i in no_main else IMG, pair_seq=i,
                           pair_t_pi_ns=int(t), ultrawide_rgb=None if i in no_uw else IMG)
            for i, t in enumerate(times_ns)]


class FakeCameraClient:
    """What get_downsampled_camera_images needs from iPhUMI's camera client, over the same packet times: one
    timestamp per packet shared by both cameras (as iPhUMI's packets), the packet index as the "image".

    ``window``: always hand out at least this many of the newest packets. Used to run the reference on one camera at
    a time with the window it uses when both cameras are asked for (max of their frame_num, 4 here), because the
    reference raises on the first camera that fails and so says nothing about the other one."""

    config = {"camera_configs": {"main": {"fps": 30}, "ultrawide": {"fps": 30}}}

    def __init__(self, times_ns, cameras=("main", "ultrawide"), window=0):
        self.t = np.array(times_ns, dtype=np.int64) / 1e9          # seconds, as iPhUMI's timestamps
        self.camera_names = list(cameras)
        self.window = window

    def get_latest_images_dict_THWC(self, n):
        n = max(n, self.window)
        idx = np.arange(len(self.t))[-n:]
        return {cam: idx for cam in self.camera_names}, self.t[-n:]


def reference_both(times_ns):
    """The reference exactly as iPhUMI calls it: both cameras; RuntimeError if either is beyond 2 frame periods."""
    images, _ = get_downsampled_camera_images(FakeCameraClient(times_ns), 2, 20.0, OBS_HZ)
    return {cam: [int(i) for i in images[cam]] for cam in images}


def reference_one(times_ns, cam):
    """The reference on one camera, with the both-camera window; None where it raises."""
    try:
        images, _ = get_downsampled_camera_images(FakeCameraClient(times_ns, [cam], selector().window), 2, 20.0,
                                                  OBS_HZ)
    except RuntimeError:
        return None
    return [int(i) for i in images[cam]]


def ours(times_ns, **kw):
    picks = selector().pick(packets(times_ns, **kw))
    return {cam: None if not s.ok else [p.packet_seq for p in s.packets] for cam, s in picks.items()}, picks


def test_reference_is_the_unchanged_iphumi_function():
    src = FOLDER.parents[1] / "behavior_prompting/deps/real-env/real_env/utils/umi_utils.py"
    if not src.exists():
        pytest.skip("behavior_prompting is not checked out")
    m = re.search(r"^def get_downsampled_camera_images\(.*?^    return downsampled_images, timestamps_dict$",
                  src.read_text(), re.S | re.M)
    here = (FOLDER / "tests" / "reference_umi_utils.py").read_text()
    assert m is not None and m.group(0) in here
    assert here.splitlines()[0].startswith("# from rnoro/deployment:behavior_prompting/deps/real-env/real_env/utils/"
                                           "umi_utils.py @ ")


def test_window_is_iphumis_frame_num():
    assert selector().window == 4                  # main: round(1.5) + 1 = 3; ultrawide: 3 + 1 = 4


def scenarios():
    rng = np.random.default_rng(7)
    out = []
    for k in range(300):
        n = int(rng.integers(4, 14))
        jitter = [0, 1, 3, 8, 15][k % 5] * MS
        t = 10_000 * MS + np.arange(n) * P + rng.integers(-jitter, jitter + 1, n)
        keep = rng.random(n) > [0.0, 0.1, 0.25, 0.4][k % 4]          # dropped packets
        keep[-1] = True
        t = np.sort(t[keep])
        t = t[np.concatenate([[True], np.diff(t) > 0])]
        out.append([int(x) for x in t])
    return out


SCENARIOS = scenarios()


@pytest.mark.parametrize("times_ns", SCENARIOS)
def test_picking_equals_the_reference_with_jitter_and_drops(times_ns):
    got, picks = ours(times_ns)
    try:
        both = reference_both(times_ns)
    except RuntimeError:
        both = None
    for cam in ("main", "ultrawide"):
        want = reference_one(times_ns, cam)
        if both is not None:
            assert want == both[cam]
        if want is not None and len(set(want)) < len(want):
            # the reference hands out the same packet twice; C1 needs increasing times, so ours is a miss
            assert got[cam] is None and "same packet" in picks[cam].reason
        else:
            assert got[cam] == want, (cam, picks[cam].reason)
    if both is None:
        assert got["main"] is None or got["ultrawide"] is None


def test_the_scenarios_cover_hits_misses_and_duplicates():
    kinds = set()
    for t in SCENARIOS:
        for cam in ("main", "ultrawide"):
            want = reference_one(t, cam)
            kinds.add("miss" if want is None else "duplicate" if len(set(want)) < 2 else "hit")
    assert kinds == {"hit", "miss", "duplicate"}


def test_normal_30_fps():
    t = [10_000 * MS + i * P for i in range(30)]
    got, _ = ours(t)
    assert got == {"main": [27, 29], "ultrawide": [26, 29]}        # 16.7 ms and 0 ms off


def test_dropped_packets_are_misses():
    t = [10_000 * MS + i * P for i in range(30)]
    got, picks = ours(t[:24] + t[29:])      # 5 packets lost before the newest
    # ultrawide: the nearest packet to T - 100 ms is 3 periods off; main: T itself is nearest to T - 50 ms
    assert got["ultrawide"] is None and "frame periods" in picks["ultrawide"].reason
    assert got["main"] is None and "same packet" in picks["main"].reason
    got, _ = ours(t[:23] + t[24:])          # one packet lost: both still fine (indices into the shortened list)
    assert got == {"main": [26, 28], "ultrawide": [25, 28]}       # T - 50 ms: a tie, the earlier packet


def test_no_usable_image_is_a_miss_for_that_camera_only():
    t = [10_000 * MS + i * P for i in range(30)]
    got, picks = ours(t, no_uw=(26,))
    assert got["main"] == [27, 29] and got["ultrawide"] is None
    assert "no paired ultrawide image" in picks["ultrawide"].reason
    got, picks = ours(t, no_main=(29,))
    assert got["main"] is None and got["ultrawide"] == [26, 29]
    assert "no usable main image" in picks["main"].reason


def test_no_packets():
    picks = selector().pick([])
    assert not picks["main"].ok and not picks["ultrawide"].ok


def test_targets():
    s = selector()
    assert s.targets(1_000 * MS, "main").tolist() == [950 * MS, 1_000 * MS]
    assert s.targets(1_000 * MS, "ultrawide").tolist() == [900 * MS, 1_000 * MS]
