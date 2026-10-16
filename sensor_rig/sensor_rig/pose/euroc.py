"""Raw rig episode -> EuRoC/ASL folder for an offline VIO (ORB-SLAM3 ``mono_inertial_euroc``, OpenVINS via a bag).

Every timestamp written is on the corrected rig clock: cameras ``rig.capture_time_ns(cam, t)``, IMU
``t + rig.imu.time_offset_ns``. The trajectory the VIO produces is therefore on the clock export uses.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from sensor_rig.raw_episode import EUROC_IMU_HEADER, RawEpisode


def scaled_size(resolution, scale: float) -> tuple[int, int]:
    """Image size for a VIO downscale, snapped to the nearest *exact* uniform scale of ``resolution``.

    Camera models only accept uniform scaling (w'/w == h'/h), so independent rounding of w and h (e.g. 1640x1232 at
    0.4 -> 656x493) is not allowed. Exact sizes are multiples of (w, h) / gcd(w, h): for 1640x1232 that is 205x154,
    so 0.4 snaps to 615x462 (3/8). ``write_euroc`` reports the scale actually used.
    """
    w, h = int(resolution[0]), int(resolution[1])
    if scale == 1.0:
        return w, h
    if not (0 < scale <= 1.0):
        raise ValueError(f"scale must be in (0, 1], got {scale}")
    g = math.gcd(w, h)
    n = max(1, int(round(scale * g)))
    return (w // g) * n, (h // g) * n


def block_average_imu(t_ns: np.ndarray, gyro: np.ndarray, acc: np.ndarray, factor: int):
    """Average consecutive blocks of ``factor`` samples (drops the incomplete tail block)."""
    if factor <= 1:
        return t_ns, gyro, acc
    n = (len(t_ns) // factor) * factor
    t = t_ns[:n].reshape(-1, factor)
    t_out = t[:, 0] + (t - t[:, :1]).mean(axis=1).round().astype(np.int64)
    return t_out, gyro[:n].reshape(-1, factor, 3).mean(axis=1), acc[:n].reshape(-1, factor, 3).mean(axis=1)


def corrected_imu(ep: RawEpisode, rig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t, g, a = ep.imu()
    return t + int(rig.imu.time_offset_ns), g, a


def write_euroc(ep: RawEpisode, out_dir, camera: str = "ultrawide", scale: float = 1.0, gray: bool = True,
                imu_rate_hz: float | None = None, rig=None) -> dict:
    import cv2

    rig = rig or ep.rig()
    out = Path(out_dir)
    cam_dir = out / "mav0" / "cam0"
    (cam_dir / "data").mkdir(parents=True, exist_ok=True)
    imu_dir = out / "mav0" / "imu0"
    imu_dir.mkdir(parents=True, exist_ok=True)

    t_raw = ep.camera_times(camera)
    t_cam = np.asarray(rig.capture_time_ns(camera, t_raw), dtype=np.int64)
    w, h = scaled_size(rig.cameras[camera].resolution, scale)
    rows = []
    for i, t in enumerate(t_cam.tolist()):
        img = ep.read_image(camera, i)
        if gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if (img.shape[1], img.shape[0]) != (w, h):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        fn = f"{t}.png"
        cv2.imwrite(str(cam_dir / "data" / fn), img)
        rows.append(f"{t},{fn}")
    (cam_dir / "data.csv").write_text("#timestamp [ns],filename\n" + "".join(r + "\n" for r in rows))
    (out / "times.txt").write_text("".join(f"{t}\n" for t in t_cam.tolist()))

    ti, g, a = corrected_imu(ep, rig)
    factor = 1
    if imu_rate_hz and len(ti) > 1:
        rate = 1e9 / float(np.median(np.diff(ti)))
        factor = max(1, int(round(rate / imu_rate_hz)))
        ti, g, a = block_average_imu(ti, g, a, factor)
    lines = [f"{t},{gi[0]:.9g},{gi[1]:.9g},{gi[2]:.9g},{ai[0]:.9g},{ai[1]:.9g},{ai[2]:.9g}"
             for t, gi, ai in zip(ti.tolist(), g, a)]
    (imu_dir / "data.csv").write_text(EUROC_IMU_HEADER + "\n" + "".join(x + "\n" for x in lines))
    imu_rate = 1e9 / float(np.median(np.diff(ti))) if len(ti) > 1 else 0.0
    return {"out_dir": str(out), "camera": camera, "n_images": len(t_cam), "size": [w, h], "scale": w / int(rig.cameras[camera].resolution[0]),
            "scale_requested": scale,
            "n_imu": int(len(ti)), "imu_rate_hz": imu_rate, "imu_decimation": factor,
            "t_first_ns": int(t_cam[0]) if len(t_cam) else None, "t_last_ns": int(t_cam[-1]) if len(t_cam) else None}
