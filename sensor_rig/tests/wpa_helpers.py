"""Synthetic raw rig episodes with a known ground-truth trajectory (WP-A export/pose tests)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from sensor_rig.pose.trajectory import Trajectory, save_tum
from sensor_rig.raw_episode import RawEpisodeWriter, write_json_atomic
from sensor_rig.timefmt import iso_ms, make_episode_name, parse_iso

G = 9.80665
MONO_BASE = 1_234_567_890_000_000
START_ISO = "2026-10-20T15:02:11.123Z"
CALIB_NAME = "2026-10-20T14-58-40.002Z_77311_rig-test_grippercalibration_left"


def truth_pose(t_s: np.ndarray) -> np.ndarray:
    """Pose of the IMU in a z-up world at times ``t_s`` (s since episode start): smooth, never identity."""
    t = np.asarray(t_s, dtype=np.float64)
    p = np.stack([0.3 * np.sin(0.8 * t), 0.2 * np.cos(0.6 * t), 0.4 + 0.05 * np.sin(1.7 * t)], axis=1)
    rv = np.stack([0.4 * np.sin(0.9 * t) + 0.2, 0.3 * np.sin(1.3 * t), 0.7 * t + 0.1], axis=1)
    T = np.tile(np.eye(4), (len(t), 1, 1))
    T[:, :3, :3] = Rotation.from_rotvec(rv).as_matrix()
    T[:, :3, 3] = p
    return T


def synth_imu(t_s: np.ndarray, h: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Gyro (body rates) and specific force of ``truth_pose`` by central differences (no noise)."""
    Tm, T0, Tp = truth_pose(t_s - h), truth_pose(t_s), truth_pose(t_s + h)
    Rm, Rp = Rotation.from_matrix(Tm[:, :3, :3]), Rotation.from_matrix(Tp[:, :3, :3])
    gyro = (Rm.inv() * Rp).as_rotvec() / (2 * h)
    a_W = (Tp[:, :3, 3] - 2 * T0[:, :3, 3] + Tm[:, :3, 3]) / h**2
    f_W = a_W - np.array([0.0, 0.0, -G])
    acc = np.einsum("nji,nj->ni", T0[:, :3, :3], f_W)
    return gyro, acc


def _image(k: int, w: int, h: int, cam: str) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    sq = 40
    img = (((xx + 3 * k) // sq + (yy + k) // sq) % 2 * 180 + 40).astype(np.uint8)
    img = np.dstack([img, np.roll(img, 7, axis=1), 255 - img])
    cv2.putText(img, f"{cam} {k}", (40, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    return img


def make_raw_episode(raw_root: Path, rig, *, kind: str = "demonstration", main_fps: float = 60.0,
                     uw_fps: float = 30.0, duration_s: float = 2.5, traj_rate_hz: float | str = "main",
                     traj_start_s: float = 0.0, traj_gap: tuple[float, float] | None = None,
                     traj_delay_ns: int = 0, imu_rate_hz: float = 400.0, rid: str = "48213",
                     calibration_run: str = CALIB_NAME, name: str | None = None) -> Path:
    """Write a raw episode (+ trajectory for demos). All times are true capture times on the corrected clock;
    the raw stamps written are ``t_capture - offset`` exactly as a recorder would."""
    t_ref = parse_iso(START_ISO)
    if name is None:
        name = make_episode_name(t_ref, rid, "rig-test", kind, "left")
    meta = {
        "clock": {"start": {"monotonic_ns": MONO_BASE, "monotonic_raw_ns": MONO_BASE, "realtime_ns": t_ref,
                            "uncertainty_ns": 0}},
        "recording_start_time": iso_ms(t_ref),
        "gripper_calibration_run_name": calibration_run if kind == "demonstration" else "",
    }
    w = RawEpisodeWriter(raw_root, name, rig, meta)
    t0 = MONO_BASE + 20_000_000  # first frame 20 ms after the press
    streams = {}
    for cam, fps in (("main", main_fps), ("ultrawide", uw_fps)):
        n = int(duration_s * fps)
        T_cap = t0 + np.round(np.arange(n) * 1e9 / fps).astype(np.int64) + (3_000_000 if cam == "ultrawide" else 0)
        off = int(rig.capture_time_ns(cam, 0))
        cw = w.camera_writer(cam)
        res = rig.cameras[cam].resolution
        for k, tc in enumerate(T_cap.tolist()):
            t_raw = tc - off
            p = w.frame_path(cam, t_raw)
            ok, buf = cv2.imencode(".jpg", _image(k, res[0], res[1], cam), [cv2.IMWRITE_JPEG_QUALITY, 92])
            p.write_bytes(buf.tobytes())
            cw.add(k, t_raw, p.name, {"t_arrival_ns": t_raw + 25_000_000})
        streams[cam] = T_cap
    span = (int(streams["main"][0]) - 500_000_000, int(streams["main"][-1]) + 500_000_000)
    t_imu = np.arange(span[0], span[1], int(1e9 / imu_rate_hz), dtype=np.int64)
    gyro, acc = synth_imu((t_imu - t0) * 1e-9)
    w.imu_writer().add_batch(t_imu - int(rig.imu.time_offset_ns), gyro, acc)
    final = w.finalize()

    if kind == "demonstration":
        if traj_rate_hz == "main":
            tt = streams["main"].copy()
        else:
            tt = np.arange(streams["main"][0] - 50_000_000, streams["main"][-1] + 50_000_000,
                           int(1e9 / float(traj_rate_hz)), dtype=np.int64)
        keep = tt >= t0 + int(traj_start_s * 1e9)
        if traj_gap is not None:
            a, d = t0 + int(traj_gap[0] * 1e9), int(traj_gap[1] * 1e9)
            keep &= ~((tt >= a) & (tt < a + d))
        tt = tt[keep]
        traj = Trajectory.from_T(tt + traj_delay_ns, truth_pose((tt - t0) * 1e-9), frame="imu")
        save_tum(traj, final / "trajectory" / "trajectory.tum")
        write_json_atomic(final / "trajectory" / "trajectory.json", {"frame": "imu", "source_format": "test"})
    return final


def expected_arkit(rig, t_capture_ns: np.ndarray) -> np.ndarray:
    """Ground-truth exported poses at the given capture times (independent of the exporter)."""
    from sensor_rig.transforms import T_YUP_ZUP

    T = truth_pose((np.asarray(t_capture_ns) - (MONO_BASE + 20_000_000)) * 1e-9)
    return rig.export.T_world_offset @ T_YUP_ZUP @ T @ rig.T("imu", "virtual_main") @ np.diag([1.0, -1, -1, 1])
