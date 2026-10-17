"""Camera client that never hangs and has no timestamp/frame race.

Upstream BaseCameraClient (a) peeks with robotmq's default automatic_resend=True (a dead camera hangs the task) and
(b) peeks the `timestamp` topic first and each camera topic afterwards; a packet published in between shifts the
frames one packet newer than their timestamps (E10; seen in the D2 harness as a 1-frame / 16.7 ms offset).
Here: NoResendClient with short timeouts, and frames paired with their timestamp by robotmq server time (see
get_latest_images_dict_THWC). Measured on the laptop: the upstream peek order mismatches ~1 call in 3 at 60 Hz.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from robotmq import deserialize

from real_env.peripherals.base_camera import BaseCameraClient

from franka_deploy.rmq_rpc import NoResendClient


class FrameSnapshotError(RuntimeError):
    pass


class SafeCameraClient(BaseCameraClient):
    def __init__(self, name: str, server_endpoint: str, timeout_s: float = 0.1, max_retries: int = 5):
        self.name = name
        self.server_endpoint = server_endpoint
        self.rmq_client = NoResendClient(f"{name}_safe", server_endpoint, default_timeout_s=timeout_s,
                                         timeouts={"config": 1.0})
        self.max_retries = max_retries
        self.retries = 0
        self.config: dict[str, Any] = self.get_config()
        self.camera_names = list(self.config["camera_configs"].keys())

    def get_latest_images_dict_THWC(self, frame_num: int, extra: int | None = None):
        """Peek every camera (frame_num + slack), then the timestamp topic, and pair by robotmq server time:
        BaseCamera.run publishes main, ultrawide, timestamp per packet, so a frame belongs to the first timestamp
        message published after it. Frames newer than the newest timestamp (packet still being published) are
        dropped; the newest `frame_num + extra` packets present in every camera (consecutive) are returned.

        `extra` (default max(2, frame_num // 2)) more packets than upstream asks for: upstream sizes the request
        by the nominal fps and then picks the frames nearest to t_latest - k/target_freq, raising if one is
        > 2/fps off. After a camera stall the server publishes a short burst, so `frame_num` packets can span
        less than the needed window (seen in the D2 harness: "ultrawide: closest frame is 50 ms from its
        target"). Extra candidates only widen the window the nearest-frame selection searches."""
        if extra is None:
            extra = max(2, frame_num // 2)
        frame_num = frame_num + extra
        slack = 3
        raws = {c: self.rmq_client.peek_data(topic=c, n=-(frame_num + slack)) for c in self.camera_names}
        ts_raw, ts_rt = self.rmq_client.peek_data(topic="timestamp", n=-(frame_num + 2 * slack))
        ts_rt = np.asarray(ts_rt, dtype=np.float64)
        packet_of = {}
        for c, (data, rt) in raws.items():
            ids = np.searchsorted(ts_rt, np.asarray(rt, dtype=np.float64), side="right")  # first ts after the frame
            packet_of[c] = {int(i): j for j, i in enumerate(ids) if i < len(ts_rt)}
        common = sorted(set.intersection(*(set(m) for m in packet_of.values())))
        if len(common) < frame_num:
            raise FrameSnapshotError(f"only {len(common)} complete packets on {self.server_endpoint} (need {frame_num})")
        sel = common[-frame_num:]
        if sel[-1] - sel[0] != frame_num - 1:
            self.retries += 1  # a packet missing in one camera inside the window: should not happen
            raise FrameSnapshotError(f"non-consecutive packets {sel}")
        self.dropped_newer = sum(len(raws[c][0]) - 1 - packet_of[c][sel[-1]] for c in raws)
        timestamps = np.array([deserialize(ts_raw[i]) for i in sel], dtype=np.float64)
        images = {}
        for c, (data, _) in raws.items():
            cc = self.config["camera_configs"][c]
            h, w = cc["height"], cc["width"]
            frames = [data[packet_of[c][i]] for i in sel]
            if cc["type"] == "rgb":
                images[c] = np.array([np.frombuffer(x, dtype=np.uint8).reshape(h, w, 3) for x in frames])
            else:
                images[c] = np.array([np.frombuffer(x, dtype=np.float32).reshape(h, w) for x in frames])
        assert np.all(np.diff(timestamps) >= 0), "timestamps not increasing"
        return images, timestamps
