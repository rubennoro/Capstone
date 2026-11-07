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
  excitation RMS is below 0.05 m/s², the tracked stretch is too short, the relative standard error is above 0.05,
  or position jitter dominates (below).

  **Position jitter (errors in variables).** Noise on the trajectory positions is noise on the *regressor* `p''`,
  so plain least squares is biased towards 0 by the attenuation factor `var(p''_true) / var(p''_measured)`. On the
  full-res 15 s sim demo, 5 mm white position noise gave s ≈ 0.91 (0.88–0.92, one false `scale_mismatch`).
  `scale_check` therefore uses **corrected least squares**:
  - `position_jitter(traj)` estimates the white, frame-to-frame jitter per axis from 4th differences of the
    positions at the pose times (`var(Δ⁴p) = 70 σ²` for white noise; hand motion contributes ~1e-4 mm at 30 Hz).
    It is robust (RMS within 4σ of the MAD estimate, so a VIO jump does not inflate it) and skips windows across
    gaps or dropped frames. Its std on 450 poses is ~5 %.
  - White jitter σ at pose interval Δt adds `σ² Δt · 3 / (8 √π σ_f⁵)` per axis to the variance of `p''` (σ_f =
    `smooth_s`; checked against the exact impulse response: 0.98× at 30 Hz, 1.00× at 60 Hz). That amount is
    subtracted from the `p''` diagonal of the normal equations.
  - It scales with the trajectory, so a wrong scale is still detected: on the same demo with 5 mm jitter, ×0.85 and
    ×1.2 trajectories give s = 1.18 and 0.84 (`scale_mismatch`).
  - `attenuation` (the estimated shrink factor of the plain fit) is reported with `scale_uncorrected` and
    `jitter`. Below `SCALE_MIN_ATTENUATION` (0.5) the correction would be larger than the signal, and the result is
    `scale_unobservable`.

  Why this and not the symmetric (geometric-mean) regression: the reverse regression is biased by everything on the
  IMU side (accelerometer noise, rotation error, bias drift, model error), so the geometric mean only halves the
  jitter bias (√λ) and adds an upward bias of its own. The correction targets exactly the term that causes the bias
  and is ~0 when there is no jitter.

  Measured on the 15 s demo, 20 seeds (oracle):

  | Position error | plain LS s (mean) | corrected s: mean ± sd, max \|s−1\| | status |
  |---|---|---|---|
  | 3 mm white | 0.966 | 1.004 ± 0.011, 0.021 | 20 ok |
  | 5 mm white | 0.908 | 1.005 ± 0.021, 0.038 | 20 ok |
  | 10 mm white | 0.707 | 1.015 ± 0.062 | 20 `scale_unobservable` (std err) |
  | random walk 5 mm/√s | 0.959 | 0.960 ± 0.017, 0.075 | 20 ok |

  Combined errors, measured on the 12 s demo, 5 seeds (oracle, wave 2 review):

  | Position / rotation error | corrected s | status |
  |---|---|---|
  | 5 mm white + random walk 5 mm/√s + 0.5° white | mean 0.922 | 4 ok, 1 `scale_mismatch` |
  | random walk only | 0.923–0.976 | ok |

  Both are consistent with "drift is not corrected" (below): drift pulls s down, and stacked on white noise it can
  cross the 0.1 tolerance.

  **Remaining bias.** Position error that is *not* white is not corrected: drift and noise correlated over the
  ~0.5–3 Hz band the filter passes look like real motion to the jitter estimate (a 5 mm/√s random walk still gives
  s ≈ 0.96). Since the high-frequency estimate then under-estimates the in-band noise, the correction can only
  under-correct, never push s above its true value. Rotation noise biases s slightly up (0.5° white: s ≈ 1.015,
  through `R_W_imu·f`). The tolerance (0.1) leaves room for both.

- **`oracle`** reads `truth.npz` (`t_ns`, `T_W_imu`, `width_m`, `rig_id`, `seed`, `version`) and samples it at the
  ultrawide times. It then adds errors and changes the world:
  - per-frame noise: `noise_pos_m`, `noise_rot_deg`;
  - random-walk drift: `drift_pos_m_per_sqrt_s`, `drift_rot_deg_per_sqrt_s`;
  - an arbitrary world: `world_yaw_deg`, `world_tilt_deg`, `world_tilt_azimuth_deg`, `world_translation_m`.

  After that it runs the shared gravity alignment and the checks. The default `seed` is the truth's seed.
- **`orbslam3`** runs a native ORB-SLAM3 v1.0 `mono_inertial_euroc` build (`$ORB_SLAM3_DIR`, default `~/ORB_SLAM3`;
  see [ORBSLAM3.md](ORBSLAM3.md)) on the fisheye ultrawide plus the IMU. Its options are `orb_slam3_dir`, `scale`,
  `imu_rate_hz`, `runner`, `timeout_s` and `n_features`. It works in four steps:
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
    They are informational only, because the first IMU initialization prints nothing. The "Not enough motion" count
    is also `diagnostics["n_not_enough_motion_resets"]`.
  - **Resets.** `n_resets` counts `SYSTEM-> Reseting active map` (System.cc), printed once per executed reset
    whoever asked for it. The requesters' own lines (Tracking.cc `IMU is not or recently initialized. Reseting
    active map...`, `TRACK: Reset map because local mapper set the bad imu flag`) are not counted, so one reset
    counts once. `n_maps` counts `Creation of new map with id` (new atlas maps; a reset clears and reuses the
    active map, so it does not add one).
  - **Testing.** The runner can be injected (`runner(cmd, log_path, timeout_s, cwd) -> rc`), so the tests use a
    fake `mono_inertial_euroc`.

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
