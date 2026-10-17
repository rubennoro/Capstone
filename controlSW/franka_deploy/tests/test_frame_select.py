"""frame_select == upstream get_downsampled_camera_images whenever upstream does not raise; tolerant of a 30 fps
physical main behind a 60 Hz config."""
import numpy as np
import pytest

pytest.importorskip("real_env")
from real_env.utils.umi_utils import get_downsampled_camera_images as upstream  # noqa: E402

from franka_deploy.tasks.frame_select import get_downsampled_camera_images as ours  # noqa: E402


class Cam:
    camera_names = ["main", "ultrawide"]
    config = {"camera_configs": {"main": {"fps": 60}, "ultrawide": {"fps": 60}}}

    def __init__(self, ts):
        self.ts = np.asarray(ts, float)

    def get_latest_images_dict_THWC(self, n, **kw):
        ts = self.ts[-(n + 3):]
        imgs = np.arange(len(ts))[:, None, None, None] * np.ones((1, 1, 1, 3))
        return {"main": imgs, "ultrawide": imgs + 1000}, ts


FREQ = {"main": 20.0, "ultrawide": 10.0}


@pytest.mark.parametrize("seed", range(20))
def test_equal_to_upstream_at_60fps(seed):
    rng = np.random.default_rng(seed)
    ts = np.cumsum(np.full(40, 1 / 60) + rng.normal(0, 0.001, 40))
    cam = Cam(ts)
    a_img, a_ts = upstream(cam, 2, 20.0, FREQ)
    b_img, b_ts = ours(cam, 2, 20.0, FREQ)
    for c in a_img:
        np.testing.assert_array_equal(a_img[c], b_img[c])
        np.testing.assert_array_equal(a_ts[c], b_ts[c])


def test_30fps_behind_60hz_config():
    base = np.cumsum(np.full(20, 1 / 30))
    ts = np.repeat(base, 2)                      # fixed_rate duplicates each 30 fps frame
    ts[1::2] += 0.0004                           # tiny stamp jitter
    ts = ts + 0.001 * (np.arange(len(ts)) % 3)   # 1 ms of jitter: pushes upstream over its 33.3 ms limit sometimes
    cam = Cam(np.sort(ts))
    img, sel = ours(cam, 2, 20.0, FREQ)
    assert abs(np.diff(sel["main"])[0] - 0.05) <= 1 / 60 + 0.002
    assert abs(np.diff(sel["ultrawide"])[0] - 0.10) <= 1 / 60 + 0.002
