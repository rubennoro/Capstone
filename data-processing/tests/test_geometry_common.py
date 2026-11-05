"""Shared analytic fixtures for the geometry tests (no tests of its own).

``analytic_episode`` writes a raw episode whose ultrawide frame times, IMU and ``truth.npz`` all come from one
closed-form IMU trajectory (periodic over the episode, so the mean acceleration is ~0 and gravity alignment from
the mean specific force is exact up to the IMU sampling).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from dataproc.config import parse_rig
from dataproc.sim.imu_synth import synth_imu
from fixtures import T0_NS, _write_mp4, build_episode, read_meta, write_frames_csv, write_imu_csv, write_meta

RIG_V1 = Path(__file__).resolve().parents[1] / "configs" / "rigs" / "rig_v1.yaml"
EXPOSURE_NS = 5_000_000
UW_FPS = 30
IMU_HZ = 200
DURATION_S = 4.0          # one full period of every motion component below


def rig_dict() -> dict:
    return yaml.safe_load(RIG_V1.read_text())


def make_rig(**edits):
    """rig_v1 with dotted-path edits, e.g. ``make_rig(**{"cameras.ultrawide.T_imu_cam": {...}})``."""
    d = rig_dict()
    for path, v in edits.items():
        node = d
        keys = path.split(".")
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = v
    return parse_rig(d)


def truth_poses(t_ns: np.ndarray, t0_ns: int = T0_NS) -> np.ndarray:
    """Closed-form T_W_imu (z-up world): smooth periodic motion, period DURATION_S (and DURATION_S / 2)."""
    s = (np.asarray(t_ns, np.int64) - t0_ns) * 1e-9
    w = 2 * np.pi / DURATION_S
    p = np.stack([0.10 * np.sin(w * s), 0.06 * np.sin(2 * w * s), 0.04 * (1 - np.cos(w * s))], 1) + [0.3, -0.1, 0.5]
    eul = np.stack([20 * np.sin(w * s) + 100, 15 * np.sin(2 * w * s), 30 * np.sin(w * s) + 40], 1)
    T = np.tile(np.eye(4), (len(s), 1, 1))
    T[:, :3, :3] = Rotation.from_euler("xyz", eul, degrees=True).as_matrix()
    T[:, :3, 3] = p
    return T


def analytic_episode(root: Path, rig, *, name: str = "s01_episode_000001", imu_late_frames: int = 0,
                     write_truth: bool = True, uw_frames: list | None = None, seed: int = 7) -> Path:
    """Raw episode with exact ultrawide times (end_of_exposure stamps), synthetic IMU and truth.npz.

    ``imu_late_frames``: the IMU starts that many ultrawide frames after the first one (for the EuRoC drop test).
    ``uw_frames``: RGB frames for cam1.mp4 (default: tiny generated frames from fixtures.build_episode).
    """
    n_uw = int(DURATION_S * UW_FPS) + 1
    ep = build_episode(root / name, n_cam0=4, n_cam1=n_uw if uw_frames is None else len(uw_frames))
    if uw_frames is not None:
        n_uw = len(uw_frames)
        _write_mp4(ep / "cam1.mp4", uw_frames, UW_FPS)
    t_mid = T0_NS + np.round(np.arange(n_uw) * 1e9 / UW_FPS).astype(np.int64)
    assert rig.cameras["ultrawide"].timestamp_ref == "end_of_exposure"
    write_frames_csv(ep / "cam1_frames.csv", t_mid + EXPOSURE_NS // 2 - rig.cameras["ultrawide"].time_offset_ns,
                     100 + np.arange(n_uw), exposure_ns=EXPOSURE_NS)
    # dense truth over the span (1 kHz), IMU synthesised from it
    t_truth = T0_NS + np.arange(0, int(DURATION_S * 1e9) + 1, 1_000_000, dtype=np.int64)
    T = truth_poses(t_truth)
    ti, g, a = synth_imu(t_truth, T, IMU_HZ, noise=None, seed=seed, smooth=False)
    keep = ti >= t_mid[imu_late_frames]
    ti, g, a = ti[keep], g[keep], a[keep]
    write_imu_csv(ep / "imu.csv", ti - rig.imu.time_offset_ns, np.arange(len(ti)), g, a)
    meta = read_meta(ep)
    meta["cameras"]["cam1"]["frame_count"] = n_uw
    meta["imu"]["sample_count"] = len(ti)
    meta["imu"]["rate_hz"] = IMU_HZ
    write_meta(ep, meta)
    if write_truth:
        np.savez(ep / "truth.npz", t_ns=t_truth, T_W_imu=T, width_m=np.full(len(t_truth), 0.05), rig_id=rig.rig_id,
                 seed=seed, version="test.v1")
    return ep


def yaw_residual(T_est: np.ndarray, T_true: np.ndarray) -> tuple[np.ndarray, float, float]:
    """For trajectories equal up to a world yaw + translation: (R_change (3,3) mean, max tilt of the per-frame
    change's z axis from +z in deg, max spread of the change between frames in deg)."""
    dR = np.einsum("nij,nkj->nik", T_est[:, :3, :3], T_true[:, :3, :3])     # R_est R_true^T
    z = dR[:, :, 2]
    tilt = np.degrees(np.arccos(np.clip(z[:, 2], -1, 1))).max()
    R0 = Rotation.from_matrix(dR[0])
    spread = np.degrees((Rotation.from_matrix(dR) * R0.inv()).magnitude()).max()
    return dR[0], float(tilt), float(spread)
