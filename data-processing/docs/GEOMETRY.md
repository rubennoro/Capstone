# Geometry stages: SLAM, TCP pose, gripper range, width

This covers the code in `dataproc/slam/`, `dataproc/stages/{pose,width}.py` and `dataproc/geometry/trajectory.py`.
The frame and time conventions follow [DESIGN.md](DESIGN.md): `T_a_b` maps points in frame b to frame a, and camera
frames are OpenCV optical.

## Time

Every geometry time is int64 ns on the **IMU time base**:
- ultrawide frames use `rig.camera_time_ns("ultrawide", t_ns, exposure_ns)`, which is mid-exposure;
- IMU samples use `rig.imu_time_ns(t_ns)`.

`dataproc.slam.base.ultrawide_times_ns(ep, rig)` gives the frame times. The output grid (`t_grid_ns`) must be on this
same base.

## SLAM backends (`dataproc.slam`)

```python
from dataproc.slam import get_backend
res = get_backend("oracle" | "orbslam3", **options).run(episode, rig, work_dir)   # -> SlamResult
```

`SlamResult` has these fields:

| Field | Meaning |
|---|---|
| `trajectory` | `Trajectory(frame="imu", world="z_up_gravity")`: rows are `T_W_imu`, the IMU (body) pose |
| `trajectory.t_ns` | ⊂ the ultrawide frame times. Untracked frames are absent; they are never interpolated |
| `t_frames_ns`, `tracked` | every ultrawide frame, and whether it has a pose |
| `fraction_tracked` | `tracked.mean()` |
| `n_maps`, `n_resets` | number of maps and resets (from ORB-SLAM3's log; the oracle always reports 1, 0) |
| `diagnostics` | JSON-able: `gravity_alignment`, `gravity_check`, `time_offset_check`, `motion_check`, backend details |
| `problems` | failed sanity checks, as text. They are not fatal: the validate stage decides |

**The world** is gravity-aligned with z up. Its yaw and origin are arbitrary, because they come from the backend.
`gravity_align` rotates the backend world by the minimal rotation that takes the mean specific force,
`mean_t R_W_imu(t) @ acc(t)`, to +z. This leaves yaw untouched. The `world_offset` is **not** applied here.

Accuracy of the alignment:
- On a moving episode the estimate is good to about 0.5°, because the mean acceleration is only approximately zero.
- An accelerometer bias tilts it by `|b_perp| / g`: 20 mg is about 1.2°. This is harmless for training, which uses
  relative poses.
- Because the alignment forces the mean specific force onto +z, the `gravity_check` run after it reads about 0° by
  construction. It only informs through `|g|`.

**Checks** run on every result (`finalize_trajectory`); a failure goes to `problems`, never an exception:
- gravity (`|g|`);
- time offset (|ω| cross-correlation);
- motion (gyro vs trajectory rate, translation jumps);
- **metric scale** (`scale_check`). It fits `R_W_imu·f + g_W = s·p'' + R_W_imu·b_imu + c_W` over the tracked span:
  - both sides are resampled to 100 Hz and filtered with the same Gaussian (σ 0.15 s; `p''` is the Gaussian
    second derivative);
  - `b_imu` is a constant accelerometer bias in the body frame, `c_W` a residual world offset;
  - `s` is recorded with its correlation-corrected standard error, `r2` and the excitation RMS. `s = 2` means the
    trajectory is half the metric size.

  The problem is `scale_mismatch` when `|s − 1| > SCALE_TOLERANCE` (0.1). It is `scale_unobservable` when the
  excitation RMS is below 0.05 m/s², the tracked stretch is too short, or the relative standard error is above 0.05.

- **`oracle`** reads `truth.npz` (`t_ns`, `T_W_imu`, `width_m`, `rig_id`, `seed`, `version`) and samples it at the
  ultrawide times. It then adds errors and changes the world:
  - per-frame noise: `noise_pos_m`, `noise_rot_deg`;
  - random-walk drift: `drift_pos_m_per_sqrt_s`, `drift_rot_deg_per_sqrt_s`;
  - an arbitrary world: `world_yaw_deg`, `world_tilt_deg`, `world_tilt_azimuth_deg`, `world_translation_m`.

  After that it runs the shared gravity alignment and the checks. The default `seed` is the truth's seed.
- **`orbslam3`** runs ORB-SLAM3 v1.0 `mono_inertial_euroc` in Docker on the fisheye ultrawide plus the IMU (see
  `docker/orbslam3/README.md`). Its options are `image`, `scale`, `imu_rate_hz`, `runner`, `timeout_s`, `user` and
  `n_features`. It works in four steps:
  - **EuRoC prep.** The MP4 is decoded, converted to gray and resized to the settings size. The image names are the
    frame times; frames outside `[t_imu[0], t_imu[-1])` are left out. The IMU goes to `mav0/imu0/data.csv`.
  - **Settings.** `KannalaBrandt8` with the scaled ultrawide intrinsics and k1..k4, `IMU.T_b_c1 = T_imu_cam`
    (ultrawide), noise from `imu.noise.for_vio()`, and `IMU.Frequency` set to the measured rate.
  - **Output frame.** This was checked against `System::SaveTrajectoryEuRoC` at v1.0-release. Each row of
    `f_episode.txt` is `Twb`, the IMU body pose. Its world is the **first keyframe's IMU frame of the largest map**,
    and lost frames and other maps are skipped. The time is written as `1e9·t_s` in fixed-point with 6 decimals. The
    import keeps the poses as `T_W_imu`. It snaps each time to the exact frame time written to `times.txt` (within
    1 ms; rows further away are dropped and counted). Then it gravity-aligns, which is always needed because the
    first keyframe's tilt is arbitrary.
  - **IMU initialization messages.** `diagnostics["imu_init"]` counts these upstream v1.0-release `LocalMapping.cc`
    strings: `end VIBA 1`, `end VIBA 2`, `Not enough motion for initializing. Reseting...` and `scale too small`.
    They are informational only, because the first IMU initialization prints nothing.
  - **Testing.** The docker runner can be injected (`runner(cmd, log_path, timeout_s) -> rc`), so the tests use a
    fake container.

## TCP pose (`dataproc.stages.pose`)

```
T_W'_tcp(t) = rig.T_world_offset @ T_W_imu(t) @ rig.cameras["ultrawide"].T_imu_cam @ rig.T_cam_tcp
```

`tcp_poses(traj, t_grid_ns, rig, max_gap_ns[, return_reasons])` returns `eef_pos` (N,3) and `eef_rot_axis_angle`
(N,3), both float32, plus `valid` (N,). Between poses it interpolates position linearly and rotation with SLERP.

A row is invalid for one of these reasons:
- `outside_trajectory`;
- `gap`: the bracketing poses are more than `max_gap_ns` apart;
- `rotvec_near_zero`: `|rotvec| ≤ 1e-6`. iPhUMI's `pose_6d_to_4x4` gives NaN there.

The config requires the `world_offset` yaw to be at least 1° away from any multiple of 360°. That makes
`rotvec_near_zero` practically never happen; the stage still checks for it.

## Gripper range and width (`dataproc.stages.width`)

The math is iPhUMI's.

- **`detect_tags(virtual_rgb, rig)`** runs on the virtual ultrawide image (`rig.virtual_ultrawide`):
  - it detects `DICT_4X4_50` tags with `CORNER_REFINE_SUBPIX`;
  - it runs `solvePnP` with the virtual K and zero distortion, with every tag `aruco.marker_size_m`;
  - `tvec` is in the virtual ultrawide frame.

  `detect_episode_tags(ep, rig, to_virtual)` decodes the ultrawide MP4 and remaps each frame with the callable,
  for example `UltrawidePrep(rig).to_virtual`.
- **`fit_gripper_range(dets, rig)`** works on the calibration episode:
  - it requires a both-tag detection rate of at least `tag_det_threshold` (0.2);
  - it keeps frames with both tags at `|tvec_z − nominal_z| < 0.08`, with width `right.x − left.x`;
  - `min_width` and `max_width` are the raw extremes;
  - the maps L→R and R→L are `polyfit` lines with one 3-MAD outlier pass.

  The tag ids come from `rig.aruco`, whereas iPhUMI picks the most-seen id pair. The result saves and loads as
  iPhUMI's `<side>_gripper_range.json` (`GripperRange.save` / `.load`).
- **`episode_widths(dets, t, range, rig)`** computes a width for each frame:
  - with both tags, the measured width;
  - with one tag, the other finger is predicted with the linear map;
  - the width is clipped to [min, max], then `min` is subtracted, so a closed gripper reads 0.

  It returns only the frames with a width.
- **`width_on_grid(t, w, t_grid_ns, max_gap_ns)`** returns `(N,1)` float32 by linear interpolation:
  - inside the detections, a grid time is valid when the bracketing detections are at most `max_gap_ns` apart;
  - outside them, the end value is held and is valid within `max_gap_ns` of that end.
