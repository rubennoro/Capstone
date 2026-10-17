# Calibrating the sensor rig

This is the hardware procedure that replaces the placeholder numbers in `sensor_rig/configs/rig_default.yaml` with
measured ones. At the end you have a versioned `rig_v1.yaml`. The recorder snapshots it into every raw episode, and
export, the simulator and the deployment camera (`RigCamera`) all read their geometry and time offsets from it.

Conventions, used everywhere in rig.yaml:

- `T_a_b` maps points in frame b to frame a, so it is the pose of b in a. 4x4, metres. rig.yaml stores `T_imu_cam`.
  Kalibr's `T_cam_imu` is its inverse.
- Camera frames are OpenCV optical: x right, y down, z forward.
- `virtual_cameras.<cam>.rotation` is `R_phys_virt`, defined by `d_phys = R_phys_virt @ d_virt` for ray directions.
- Times are CLOCK_MONOTONIC integer ns. For each camera, `t_capture = t_stamp + offset_prior_ns + round(kalibr_timeshift_cam_imu_s * 1e9)`.
  Raw files keep the unmodified stamps.

Commands below are `sensor-rig <command>`. Without WP-A's top-level CLI, `python -m sensor_rig.calib.cli <command>`
runs the same commands.

## 0. Lenses: check coverage before you buy

The rig reproduces two *virtual* iPhone 16 Pro cameras: main pinhole 1920x1440 (68.6 x 54.2 deg) and ultrawide
pinhole 640x480 (102 x 86 deg, 114 deg diagonal). A remap cannot invent pixels the lens never saw. Coverage at
1640x1232 (`coverage()` in `camera_model.py`):

| physical lens | coverage of the virtual camera |
|---|---|
| IMX219 "77 deg" main (2.96 mm, fx = 1321 px), ideal pinhole | 0.83 (0.91 with k1 = -0.1, 0.98 with k1 = -0.2) |
| ultrawide, equidistant 120 deg diagonal | 0.83 |
| ultrawide, equidistant 140 deg diagonal | 0.995 |
| ultrawide, equidistant 160 deg diagonal | 1.0 |

So buy a main lens of at least about 90 deg DFOV and a fisheye ultrawide of about 140 to 160 deg. The default rig uses
lenses of this class, and both cameras reach coverage 1.0. Use the 4:3 full-FOV sensor mode (IMX219 1640x1232 @ 30 fps).
The 1280x720 @ 60 mode crops the field of view, which drops ultrawide coverage to 0.3 to 0.6.

## 1. Freeze the hardware

Mount both cameras and the IMU on one rigid plate, then glue the M12 focus. Put the physical main and ultrawide lens
centres where the iPhone 16 Pro's main and ultrawide lenses were: a remap can fix orientation but not position.
Calibrate in the same Argus `sensor_mode`, resolution and exposure limits that recording and deployment use
(`cameras.*.sensor_mode`, `cameras.*.argus`; on a Raspberry Pi: `cameras.*.picamera2`, `resolution`, flips and
`lens_position`). Redo steps 3 to 8 after any remount or refocus.

## 2. IMU noise (once per IMU part)

1. Record at least 3 h static, at the operating ODR and a stable temperature: `sensor-rig record --imu-only ...`.
2. Convert the recording to a ROS1 bag with the rosbags converter (pose stage) and run `allan_variance_ros` in Docker.
   Keep the resulting `imu.yaml`.
3. rig.yaml keeps the raw values. `imu.noise.vio_inflation` (default x5 noise density, x10 random walk) is applied
   only for the VIO configs (`ImuNoise.for_vio()`). Kalibr gets the raw values.

## 3. Camera intrinsics and camera-to-camera extrinsics (Kalibr)

- Target: Aprilgrid 6x6, 30 mm tags, spacing 0.3, on a rigid flat board.
- Record 60 to 90 s, moving the board slowly through the whole FOV of both cameras, including the fisheye corners.
- Build the bag at 4 Hz. Stamp each image with `t_stamp + offset_prior_ns`. The converter records that offset in the
  bag metadata. Then run:

```
kalibr_calibrate_cameras --target aprilgrid.yaml --models pinhole-radtan pinhole-equi \
    --topics /cam_main/image_raw /cam_ultrawide/image_raw --bag cams.bag --bag-freq 4
```

Accept the result if the mean reprojection error is below 0.3 px for main and below 0.5 px for the ultrawide, with no
systematic corner residuals. Only `pinhole-radtan` and `pinhole-equi` are supported. `ds`, `eucm`, `omni` and `fov`
are rejected by calib-import. If `pinhole-equi` fits badly beyond about 150 to 160 deg, use a narrower lens.

Kalibr runs in its Docker image (`ethz-asl/kalibr`, `Dockerfile_ros1_20_04`) on the laptop. The Jetson never needs ROS.

## 4. Camera-IMU extrinsics and time shift (Kalibr)

Record 60 to 90 s exciting all 6 axes, with no motion blur (lock exposure at 4 ms or less) and the IMU at full rate:

```
kalibr_calibrate_imu_camera --target aprilgrid.yaml --cam camchain.yaml --imu imu.yaml --bag camimu.bag
```

Accept the result if all of these hold:

- |timeshift_cam_imu| < 10 ms. Otherwise fix `cameras.*.time.offset_prior_ns` and rerun.
- Gyro and accelerometer residuals are white.
- Two independent runs agree to within 0.2 deg, 3 mm and 1 ms.

## 5. Import

```
sensor-rig calib-import --rig rig.yaml --camchain camchain-imucam.yaml --imu imu.yaml --out rig_v1.yaml
```

What the import does:

- Cameras are matched by `rostopic`, so `/cam_main/...` becomes `main`. Pass `--map cam0=ultrawide,cam1=main` if the
  topic names don't match.
- Intrinsics and distortion are copied. The Kalibr resolution must equal `cameras.<c>.resolution`, otherwise the
  import fails: you calibrated in the wrong sensor mode.
- `T_imu_cam = inv(T_cam_imu)`, re-orthonormalised by SVD.
- `timeshift_cam_imu` goes to `time.kalibr_timeshift_cam_imu_s`. `offset_prior_ns` is kept from the input rig.
- Kalibr's `T_cn_cnm1` is checked against the derived `T("main","ultrawide")` and warns above 1 mm or 0.05 deg.
- The imported sections lose `placeholder: true`. `calibration:` records the file names, sha256 and date.

Lines starting with `WARNING:` need attention. Add `--strict` to make warnings fail the command.
A camera-only `camchain.yaml` (no `T_cam_imu`) imports intrinsics only, and that camera stays placeholder.

## 6. Virtual camera rotation

```
sensor-rig calib-align --rig rig_v1.yaml --iphone-calib <iPhone gripper-calibration export folder> \
    --rig-calib <raw rig gripper-calibration episode> [--iphone-main-uw <iPhone demo of a scene >= 3 m away>] \
    --write rig_v1.yaml
```

**Ultrawide.** The finger tags (DICT_4X4_50, 16 mm, ids 0/1) are rigid with respect to the mount when the gripper is
closed. The tool detects them in two places, with the same detector settings as iphumi:

- the iPhone's ultrawide video, using its JSON K;
- the rig's frames remapped into the current virtual ultrawide.

From each side it takes up to 40 frames with the smallest tag separation (closed gripper) and averages the 8 tag corners.
It fits the rotation (rotation only) on the corner *bearings*, so the fingers land on the iPhone's pixels, and updates
`R_phys_virt_new = R_phys_virt_old @ R^T`.

A Kabsch fit on the PnP 3-D corners is printed as a mount diagnostic: `residual translation`. Below 3 mm is fine. Above
that, fix the mount, or accept a viewpoint shift for the iPhone-trained policy. Use this number only as a hint: with
both tags about 0.1 m away, a 1 mm lateral offset is nearly indistinguishable from a 0.6 deg rotation. Measure the lens
positions mechanically (CAD or calipers) as well.

**Main.** The main rotation is chained: `R_physmain_virtmain = R_physmain_physuw @ R_physuw_virtuw @ R_iphoneUW_iphoneMain`.

- `R_physmain_physuw` comes from Kalibr.
- `R_iphoneUW_iphoneMain` defaults to the identity: the iPhone 16 Pro wide and ultrawide axes are assumed parallel to
  well under 1 deg.
- `--iphone-main-uw` estimates it instead from SIFT matches between main frames and their ultrawide slot frames, using a
  rotation-only model. This requires a distant scene. The tool refuses it if the median residual exceeds 1 px (parallax).

## 7. Gripper calibration entry and TCP

`iPhUMI/python_package/iphumi/demonstration_processing/calibration/gripper_calibration.yaml` has a `"JetsonRig1,1"`
entry (the export's `deviceIdentifier`) copied from `iPhone17,1`. It is correct only if the physical main lens centre
sits where the iPhone main camera was and step 6 matched the orientations. Otherwise compute it from CAD:

```
sensor-rig calib-tcp --rig rig_v1.yaml --tcp-in-main-cv X Y Z [--tcp-rot-euler-xyz-deg A B C]
```

The inputs are the TCP origin (fingertip midpoint) and the TCP axes (z along the fingers, x right, y down), expressed in
the *physical* main camera's optical frame. Take the lens centre as the entrance pupil (vendor drawing, +-2 mm).
The tool prints `tcp_pose_in_arkit_frame = D @ [[R_phys_virt^T, 0], [0, 1]] @ T_physmain_TCP`, with
`D = diag(1, -1, -1, 1)`, ready to paste.

`artag_z_distance_from_ultrawide` is the median tag depth in the virtual ultrawide with the gripper closed. calib-check
prints it as `tag_z_m`. It is only a +-0.08 m outlier gate, so +-1 cm is fine.

## 8. Verify on the hardware

```
sensor-rig calib-check --rig rig_v1.yaml --live --out check/ [--reference-iphone <iPhone export>] [--watch 2]
sensor-rig calib-check --rig rig_v1.yaml --episode <raw episode> [--frame N] --out check/
sensor-rig calib-check --rig rig_v1.yaml --images main.png ultrawide.png --out check/
```

The command writes PNGs and `report.json` (`coverage`, `tag_z_m`, `tag_sep_m`, `rms_vs_reference_px`,
`rig_fingerprint`, `placeholder`). It has no GUI, so it works over ssh. For each camera:

- `<cam>_physical.png`: the raw frame, with the virtual border in green and the model's valid-radius limit in red.
- `<cam>_virtual.png`: the virtual frame. Uncovered pixels are tinted red, with a 10 % grid, the principal point and
  a coverage label.
- `<cam>_virtual_224.png`: the pad-to-square 224x224 image the policy sees.
- With a reference: `<cam>_blend.png` (50/50) and `<cam>_edges.png` (iPhone edges in magenta over the rig frame).

Acceptance:

- Coverage is at least 0.99 for both cameras.
- Both finger tags are detected in the virtual ultrawide, with the gripper both closed and open, and `tag_z_m` is within
  +-0.02 m of `artag_z_distance_from_ultrawide`.
- The blend shows no visible finger misalignment against an iPhone frame of the same gripper state. The tag-corner RMS
  should be below 2 px at 640x480.

`sensor-rig rig-validate rig_v1.yaml` prints the load errors or warnings and the coverage.
`sensor-rig bench-remap --rig rig_v1.yaml --purpose deploy --iters 600` prints remap p50/p99 per camera.
The Jetson target is p99 <= 4 ms per camera (laptop: about 2 ms on 1 thread).

## 9. Version it

Keep `rig_v1.yaml` and the Kalibr reports (PDF, camchain, imu yaml) next to the data. `rig.fingerprint()` (also in
`report.json`) identifies the geometry and excludes the `calibration:` provenance block. The recorder writes it into each
episode's `meta.json`.

## Open items that need the hardware

- Measure `offset_prior_ns` (camera stamp to mid-exposure) with an LED or QR test, for each sensor mode and exposure.
- Allan variance run; Kalibr camera and camera-IMU runs; calib-align with a real iPhone and rig gripper-calibration pair.
- CAD TCP measurement for `"JetsonRig1,1"`, plus the `artag_z_distance_from_ultrawide` reading from calib-check.
- `bench-remap` on the Orin Nano.
