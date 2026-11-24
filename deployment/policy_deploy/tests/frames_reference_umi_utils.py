# from rnoro/deployment:behavior_prompting/deps/real-env/real_env/utils/umi_utils.py @ d0298b9
# changes: only get_downsampled_camera_images is kept (unchanged), with the two imports it uses
"""iPhUMI's frame-picking rule, the reference PacketSelector is compared against (process_frames/STRUCTURE.md §3)."""
from typing import Any

import numpy as np


def get_downsampled_camera_images(
    camera_client: Any,
    image_history_len: int,
    agent_update_freq_hz: float,
    cam_obs_freq_hz: dict[str, float] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Fetch images from a camera client with per-camera downsampling.

    Each camera is downsampled so consecutive obs frames are spaced
    1/target_freq apart, where target_freq comes from cam_obs_freq_hz for
    that camera (falling back to agent_update_freq_hz if not specified).

    Cameras with different training obs rates (e.g. ultrawide at 10Hz vs main
    at 20Hz) should have correspondingly lower target frequencies so the number
    of action steps between obs frames matches training.

    If a camera's fps is not evenly divisible by its target frequency, the
    downsample factor is rounded to the nearest integer and a warning is printed.

    Returns:
        images_dict: cam_name → (T, H, W, 3) uint8 array
        timestamps_dict: cam_name → (T,) float array of obs-frame timestamps
    """
    if image_history_len == 1:
        images_dict, timestamps = camera_client.get_latest_images_dict_THWC(1)
        timestamps_dict = {cam_name: timestamps for cam_name in images_dict}
        return images_dict, timestamps_dict

    cam_params: dict[str, dict] = {}
    for cam_name in camera_client.camera_names:
        fps = camera_client.config["camera_configs"][cam_name]["fps"]
        target_freq = (cam_obs_freq_hz or {}).get(cam_name, agent_update_freq_hz)
        # Frames needed to cover the full obs time window at nominal fps.
        frame_num = round((image_history_len - 1) * fps / target_freq) + 1
        cam_params[cam_name] = {
            "fps": fps,
            "target_freq": target_freq,
            "frame_num": frame_num,
        }

    total_frame_num = max(p["frame_num"] for p in cam_params.values())
    images_dict, raw_timestamps = camera_client.get_latest_images_dict_THWC(total_frame_num)

    # Anchor target times to the most recent frame and step backward by
    # 1/target_freq, then select the closest actual frame for each target.
    # This handles fps drops gracefully rather than assuming uniform stride.
    t_latest = raw_timestamps[-1]
    downsampled_images = {}
    timestamps_dict = {}
    for cam_name, p in cam_params.items():
        target_ts = np.array([
            t_latest - (image_history_len - 1 - i) / p["target_freq"]
            for i in range(image_history_len)
        ])
        indices = np.argmin(np.abs(raw_timestamps[:, None] - target_ts[None, :]), axis=0)
        selected_ts = raw_timestamps[indices]
        max_deviation = np.max(np.abs(selected_ts - target_ts))
        max_allowed = 2.0 / p["fps"]
        if max_deviation > max_allowed:
            raise RuntimeError(
                f"{cam_name}: closest frame is {max_deviation*1000:.1f}ms from its target "
                f"(max allowed {max_allowed*1000:.1f}ms = 2 frames at {p['fps']}fps). "
                f"Camera may have dropped frames."
            )
        downsampled_images[cam_name] = images_dict[cam_name][indices]
        timestamps_dict[cam_name] = selected_ts

    return downsampled_images, timestamps_dict
