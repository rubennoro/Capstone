"""The offline pose stage: ``prepare`` (raw episode -> VIO input) and ``import_trajectory`` (VIO output -> episode)."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path

import numpy as np

from sensor_rig.pose.euroc import corrected_imu, write_euroc
from sensor_rig.pose.frames import rig_frame
from sensor_rig.pose.trajectory import gravity_align as _gravity_align
from sensor_rig.pose.trajectory import (
    gravity_check,
    load_orbslam3_euroc,
    load_tum,
    save_tum,
    time_offset_check,
)
from sensor_rig.raw_episode import RawEpisode, write_json_atomic

TIME_BASE = "rig_monotonic_corrected_ns"


def prepare(ep: RawEpisode, out_dir, target: str = "orbslam3", camera: str | None = None, scale: float = 1.0,
            imu_rate_hz: float | None = None, rig=None) -> dict:
    """Write a EuRoC folder for ``ep`` (+ ORB-SLAM3 settings and run script, or OpenVINS/Kalibr yamls)."""
    from sensor_rig.pose import openvins, orbslam3

    rig = rig or ep.rig()
    camera = camera or rig.vio.camera
    out = Path(out_dir) / ep.name
    info = write_euroc(ep, out, camera=camera, scale=scale, gray=True, imu_rate_hz=imu_rate_hz, rig=rig)
    rate = info["imu_rate_hz"] or None
    if target == "orbslam3":
        s = orbslam3.write_orbslam3_settings(rig, camera, scale, out / "orbslam3_settings.yaml", imu_rate_hz=rate)
        info["settings"] = str(s)
        info["run_script"] = str(orbslam3.write_run_script(out / "run_orbslam3.sh", out, s, ep.name))
    elif target == "openvins":
        info["kalibr"] = openvins.write_kalibr_yamls(rig, camera, out, scale=scale, imu_rate_hz=rate)
    elif target != "euroc":
        raise ValueError(f"unknown target {target!r} (orbslam3 | openvins | euroc)")
    (out / "pose_prep.json").write_text(json.dumps(info, indent=1) + "\n")
    return info


class TrajectoryImportError(RuntimeError):
    pass


def import_trajectory(ep: RawEpisode, traj_file, fmt: str = "tum", time_unit: str = "auto", frame: str | None = None,
                      force: bool = False, rig=None, gravity_align: bool = True,
                      gravity_align_min_deg: float = 2.0) -> dict:
    """Normalise a VIO trajectory into ``<ep>/trajectory/trajectory.{tum,json}`` after sanity checks.

    By default the trajectory's world is first rotated so gravity points along -z (estimated from the episode's IMU,
    yaw untouched): ORB-SLAM3's EuRoC output lives in the first keyframe's (tilted) IMU frame. The applied correction
    is recorded in trajectory.json. The correction is only applied when the estimated tilt exceeds
    ``gravity_align_min_deg``: the mean-specific-force estimate is only good to ~0.5 deg on a moving episode (the
    mean acceleration is not exactly zero), while an inertial VIO whose world is already gravity-aligned (OpenVINS,
    the simulator) knows gravity better than that. Refuses (unless ``force``) if the trajectory overlaps < 50% of the main camera
    span or the gravity check (run after alignment) fails.
    """
    rig = rig or ep.rig()
    frame = rig_frame(frame or rig.vio.trajectory_frame)
    traj_file = Path(traj_file)
    if fmt == "tum":
        traj = load_tum(traj_file, time_unit=time_unit, frame=frame)
    elif fmt == "orbslam3-euroc":
        traj = load_orbslam3_euroc(traj_file, frame=frame)
    else:
        raise ValueError(f"unknown trajectory format {fmt!r}")
    if len(traj) < 2:
        raise TrajectoryImportError(f"{traj_file}: fewer than 2 poses")

    t_main = np.asarray(rig.capture_time_ns("main", ep.camera_times("main")), dtype=np.int64)
    span = max(int(t_main[-1] - t_main[0]), 1)
    overlap = max(0, min(int(traj.t_ns[-1]), int(t_main[-1])) - max(int(traj.t_ns[0]), int(t_main[0])))
    overlap_frac = overlap / span
    imu_t, gyro, acc = corrected_imu(ep, rig)
    align_info = {"applied": False}
    if gravity_align and overlap_frac > 0 and len(imu_t):
        try:
            aligned, align_info = _gravity_align(traj, imu_t, acc, rig)
            if align_info["angle_deg"] > gravity_align_min_deg:
                traj = aligned
            else:
                align_info = {**align_info, "applied": False,
                              "reason": f"estimated tilt below {gravity_align_min_deg} deg; world kept"}
        except ValueError as e:
            align_info = {"applied": False, "reason": str(e)}
    grav = gravity_check(traj, imu_t, acc, rig) if len(imu_t) else {"ok": False, "reason": "no IMU data"}
    toff = time_offset_check(traj, imu_t, gyro, rig) if len(imu_t) else {"ok": False, "reason": "no IMU data"}

    problems = []
    if overlap_frac < 0.5:
        problems.append(f"trajectory overlaps only {overlap_frac:.0%} of the main camera span "
                        "(wrong time unit or clock?)")
    if not grav.get("ok"):
        problems.append(f"gravity check failed: {grav}")
    if problems and not force:
        raise TrajectoryImportError("; ".join(problems) + " (use --force to import anyway)")

    out = ep.trajectory_dir()
    out.mkdir(exist_ok=True)
    save_tum(traj, out / "trajectory.tum")
    info = {
        "source_file": str(traj_file.resolve()),
        "source_sha256": hashlib.sha256(traj_file.read_bytes()).hexdigest(),
        "source_format": fmt,
        "frame": traj.frame,
        "world": "z_up_gravity",
        "time_base": TIME_BASE,
        "n_poses": len(traj),
        "t_first_ns": int(traj.t_ns[0]),
        "t_last_ns": int(traj.t_ns[-1]),
        "main_overlap_fraction": overlap_frac,
        "gravity_alignment": align_info,
        "gravity_check": grav,
        "time_offset_check": toff,
        "forced": bool(problems),
        "problems": problems,
        "imported_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    write_json_atomic(out / "trajectory.json", info)
    return info
