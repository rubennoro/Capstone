"""Demo library from a prompt / training zarr (env with zarr + the imagecodecs codecs, e.g. env_bp).

Per episode: time (60 Hz frames), TCP position + rotation, gripper width, positions relative to the episode-start
TCP frame, height along the dataset's "up" axis: +y for iPhUMI zarrs (ARKit world frame, y up; measured: every
firstreal demo reaches down ~20 cm in y, grasps, lifts back up, places, releases); pass `up_axis` otherwise.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from franka_deploy.eval.metrics import dtw, phases, relative_positions, resample

DEMO_HZ = 60.0


def load_demos(zarr_path: str, up_axis=(0.0, 1.0, 0.0)) -> list[dict]:
    import zarr

    try:
        from behavior_prompting.common.imagecodecs_numcodecs import register_codecs
    except ImportError:  # pragma: no cover
        from iphumi.common.imagecodecs_numcodecs import register_codecs
    register_codecs(verbose=False) if "verbose" in register_codecs.__code__.co_varnames else register_codecs()
    root = zarr.open(zarr.ZipStore(zarr_path, mode="r"), mode="r")
    pos = root["data/gripper_right_eef_pos"][:].astype(float)
    rv = root["data/gripper_right_eef_rot_axis_angle"][:].astype(float)
    w = root["data/gripper_right_gripper_width"][:, 0].astype(float)
    names = [str(x) for x in root["meta/episode_names"][:]]
    ends = root["meta/episode_ends"][:]
    up = np.asarray(up_axis, float) / np.linalg.norm(up_axis)
    out, s = [], 0
    for i, e in enumerate(ends):
        e = int(e)
        P, W = pos[s:e], w[s:e]
        t = np.arange(e - s) / DEMO_HZ
        r0 = Rotation.from_rotvec(rv[s])
        out.append({"idx": i, "name": names[i], "t": t, "pos": P, "width": W, "rel": relative_positions(P, r0),
                    "height": (P - P[0]) @ up})
        s = e
    return out


def demo_phases(demos: list[dict]) -> list[dict | None]:
    return [phases(d["width"], d["height"], d["t"]) for d in demos]


def rel_traj_10hz(t, rel):
    return resample(np.asarray(t, float), np.asarray(rel, float), 10.0)


def nearest_demo_dtw(traj10: np.ndarray, demos10: list[np.ndarray]) -> np.ndarray:
    return np.array([dtw(traj10, d) for d in demos10])


def loo_baseline(demos10: list[np.ndarray]) -> dict:
    """Leave-one-out: each demo's DTW to its nearest other demo."""
    n = len(demos10)
    D = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = dtw(demos10[i], demos10[j])
    nn = D.min(1)
    return {"nearest_dtw_m": nn.tolist(), "median_m": float(np.median(nn)), "matrix": D}
