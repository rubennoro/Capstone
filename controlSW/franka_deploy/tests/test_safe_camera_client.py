"""SafeCameraClient pairs frames with timestamps by robotmq server time, even when a packet is being published
between the peeks (the upstream E10 race)."""
import numpy as np
import pytest

pytest.importorskip("real_env")
from robotmq import serialize  # noqa: E402

from franka_deploy.peripherals.safe_camera_client import FrameSnapshotError, SafeCameraClient  # noqa: E402

H, W = 2, 3


class FakeRMQ:
    """Packets k=0..n-1 published at server times main 10k+1, uw 10k+2, timestamp 10k+3; frame value = k."""

    def __init__(self, n_main, n_uw, n_ts):
        self.topics = {
            "main": [(np.full((H, W, 3), k, np.uint8).tobytes(), 10 * k + 1) for k in range(n_main)],
            "ultrawide": [(np.full((H, W, 3), 100 + k, np.uint8).tobytes(), 10 * k + 2) for k in range(n_uw)],
            "timestamp": [(serialize(float(k) * 0.0166), 10 * k + 3) for k in range(n_ts)],
        }

    def peek_data(self, topic, n):
        items = self.topics[topic][n:] if n < 0 else self.topics[topic][:n]
        return [d for d, _ in items], [t for _, t in items]


def client(fake):
    c = SafeCameraClient.__new__(SafeCameraClient)
    c.rmq_client = fake
    c.server_endpoint = "fake"
    c.retries = 0
    c.config = {"camera_configs": {"main": {"type": "rgb", "height": H, "width": W},
                                   "ultrawide": {"type": "rgb", "height": H, "width": W}}}
    c.camera_names = ["main", "ultrawide"]
    return c


@pytest.mark.parametrize("n_main,n_uw,n_ts,newest", [(20, 20, 20, 19), (21, 20, 20, 19), (21, 21, 20, 19), (20, 20, 21, 19)])
def test_pairs_by_server_time(n_main, n_uw, n_ts, newest):
    imgs, ts = client(FakeRMQ(n_main, n_uw, n_ts)).get_latest_images_dict_THWC(4, extra=0)
    ks = list(range(newest - 3, newest + 1))
    assert [int(x[0, 0, 0]) for x in imgs["main"]] == ks
    assert [int(x[0, 0, 0]) - 100 for x in imgs["ultrawide"]] == ks
    np.testing.assert_allclose(ts, np.array(ks) * 0.0166)


def test_not_enough_packets():
    with pytest.raises(FrameSnapshotError):
        client(FakeRMQ(2, 2, 2)).get_latest_images_dict_THWC(4)


def test_extra_packets_returned():
    imgs, ts = client(FakeRMQ(20, 20, 20)).get_latest_images_dict_THWC(4)
    assert len(ts) == 6 and [int(x[0, 0, 0]) for x in imgs["main"]] == list(range(14, 20))
