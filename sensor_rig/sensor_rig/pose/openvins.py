"""OpenVINS / Kalibr-format calibration yamls for one camera + the IMU.

Time offsets are already applied to every stamp pose-prep writes, so ``timeshift_cam_imu`` is 0. OpenVINS needs a ROS
bag (``/cam0/image_raw`` + ``/imu0``) built from the EuRoC folder (e.g. with ``rosbags-convert`` or Kalibr's
``bagcreater``); record its output with ``ov_eval pose_to_file`` (TUM order) and import it with
``pose-import --format tum --frame imu``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sensor_rig.pose.orbslam3 import _camera_model_at


def _rows(M: np.ndarray, indent: str) -> str:
    return "\n".join(f"{indent}- [{', '.join(f'{v:.9f}' for v in row)}]" for row in np.asarray(M))


def write_kalibr_yamls(rig, camera: str, out_dir, scale: float = 1.0, imu_rate_hz: float | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model, w, h = _camera_model_at(rig, camera, scale)
    if model.kind == "fisheye-equidistant":
        dist_model, dist = "equidistant", [model.k1, model.k2, model.k3, model.k4]
    else:
        dist_model, dist = "radtan", [model.k1, model.k2, model.p1, model.p2]
    T_imu_cam = rig.T("imu", camera)
    imucam = f"""%YAML:1.0
---
cam0:
  T_imu_cam: # pose of the camera in the IMU frame (OpenVINS convention)
{_rows(T_imu_cam, "    ")}
  cam_overlaps: []
  camera_model: pinhole
  distortion_coeffs: [{', '.join(f'{v:.9g}' for v in dist)}]
  distortion_model: {dist_model}
  intrinsics: [{model.fx:.9f}, {model.fy:.9f}, {model.cx:.9f}, {model.cy:.9f}]
  resolution: [{w}, {h}]
  rostopic: /cam0/image_raw
  timeshift_cam_imu: 0.0  # offsets already applied by pose-prep
"""
    noise = rig.imu.noise.for_vio()
    rate = imu_rate_hz or float(rig.imu.output_rate_hz)
    imu = f"""%YAML:1.0
---
imu0:
  T_i_b:
    - [1.0, 0.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0, 0.0]
    - [0.0, 0.0, 1.0, 0.0]
    - [0.0, 0.0, 0.0, 1.0]
  accelerometer_noise_density: {noise.accelerometer_noise_density:.9g}
  accelerometer_random_walk: {noise.accelerometer_random_walk:.9g}
  gyroscope_noise_density: {noise.gyroscope_noise_density:.9g}
  gyroscope_random_walk: {noise.gyroscope_random_walk:.9g}
  model: calibrated
  rostopic: /imu0
  time_offset: 0.0
  update_rate: {rate:.1f}
"""
    estimator = """%YAML:1.0
---
# Minimal OpenVINS estimator_config.yaml template; start from ov_msckf/config/<dataset>/estimator_config.yaml
verbosity: "INFO"
use_fej: true
integration: "rk4"
use_stereo: false
max_cameras: 1
calib_cam_extrinsics: true
calib_cam_intrinsics: true
calib_cam_timeoffset: true
calib_imu_intrinsics: false
calib_imu_g_sensitivity: false
max_clones: 11
max_slam: 50
max_msckf_in_update: 40
dt_slam_delay: 1
gravity_mag: 9.80665
init_window_time: 2.0
init_imu_thresh: 1.5
relative_config_imu: "kalibr_imu_chain.yaml"
relative_config_imucam: "kalibr_imucam_chain.yaml"
"""
    paths = {
        "imucam": out / "kalibr_imucam_chain.yaml",
        "imu": out / "kalibr_imu_chain.yaml",
        "estimator": out / "estimator_config.yaml",
    }
    paths["imucam"].write_text(imucam)
    paths["imu"].write_text(imu)
    paths["estimator"].write_text(estimator)
    return {k: str(v) for k, v in paths.items()}
