"""Observation frame selection = upstream real_env.utils.umi_utils.get_downsampled_camera_images (anchor on the
newest packet, step back 1/target_freq per sample, nearest packet), with one change: the "dropped frames" tolerance
uses the camera's *effective* frame interval (median spacing of distinct packet timestamps) instead of the
configured fps. With a 30 fps physical main published at a 60 Hz config (RigCamera fixed_rate, the default rig),
packets repeat frames and upstream's 2/60 s = 33 ms tolerance is exceeded by stamp jitter (seen in the D2 harness:
"closest frame is 34.0ms from its target"), turning a healthy camera into a policy stop. The selected frames are
identical to upstream's whenever upstream does not raise.
"""

from __future__ import annotations

import numpy as np


class FrameSelectionError(RuntimeError):
    """No packet close enough to a sample time (camera stall); the task retries the observation a few times."""


def get_downsampled_camera_images(camera_client, image_history_len: int, agent_update_freq_hz: float,
                                  cam_obs_freq_hz: dict | None = None):
    if image_history_len == 1:
        images, ts = camera_client.get_latest_images_dict_THWC(1)
        return images, {c: ts for c in images}
    params = {}
    for cam in camera_client.camera_names:
        fps = camera_client.config["camera_configs"][cam]["fps"]
        target = (cam_obs_freq_hz or {}).get(cam, agent_update_freq_hz)
        params[cam] = {"fps": fps, "target": target, "n": round((image_history_len - 1) * fps / target) + 1}
    images, raw_ts = camera_client.get_latest_images_dict_THWC(max(p["n"] for p in params.values()))
    distinct = np.unique(raw_ts)
    eff_dt = float(np.median(np.diff(distinct))) if len(distinct) > 2 else 0.0
    t_latest = raw_ts[-1]
    out_img, out_ts = {}, {}
    for cam, p in params.items():
        target_ts = np.array([t_latest - (image_history_len - 1 - i) / p["target"] for i in range(image_history_len)])
        idx = np.argmin(np.abs(raw_ts[:, None] - target_ts[None, :]), axis=0)
        sel = raw_ts[idx]
        dev = float(np.max(np.abs(sel - target_ts)))
        tol = max(2.0 / p["fps"], 2.0 * eff_dt)
        if dev > tol:
            raise FrameSelectionError(f"{cam}: closest frame is {dev * 1000:.1f}ms from its target (max allowed "
                               f"{tol * 1000:.1f}ms = 2 effective frame intervals). Camera may have dropped frames.")
        out_img[cam] = images[cam][idx]
        out_ts[cam] = sel
    return out_img, out_ts
