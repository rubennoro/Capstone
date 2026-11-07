# Vendored/adapted from sensor_rig/sensor_rig/pose/{euroc,orbslam3,vio_hooks,stage}.py at commit 4be9982.
# Changes: EuRoC images are decoded from the ultrawide MP4 (RawEpisode.read_video_frames) instead of JPEG folders;
# times are dataproc's IMU time base (RigConfig.camera_time_ns / imu_time_ns); frames outside the IMU span are dropped
# (mono_inertial_euroc indexes past its IMU array otherwise); the runner is injectable; no mask support.
# Jetson: runs a native ORB-SLAM3 build (``$ORB_SLAM3_DIR``, default ``~/ORB_SLAM3``) instead of the Docker image.
"""``orbslam3`` backend: ORB-SLAM3 v1.0 mono-inertial (``mono_inertial_euroc``), built natively on the Jetson
(docs/ORBSLAM3.md), on the fisheye ultrawide plus the IMU.

Steps (all files in ``work_dir``, which is also the working directory of the run):

1. **EuRoC prep** (:func:`write_euroc`): ``mav0/cam0/data/<t_ns>.png`` (gray, resized to the settings size),
   ``mav0/cam0/data.csv``, ``times.txt`` (one ``t_ns`` per line; ``mono_inertial_euroc`` loads
   ``<seq>/mav0/cam0/data/<line>.png``), ``mav0/imu0/data.csv`` (``t_ns,gx,gy,gz,ax,ay,az``). Camera times are the
   ultrawide mid-exposure times, IMU times ``rig.imu_time_ns`` (one time base). Frames with ``t < t_imu[0]`` or
   ``t >= t_imu[-1]`` are left out.
2. **Settings** (:func:`write_orbslam3_settings`, format ``File.version "1.0"``): ``KannalaBrandt8`` with the
   ultrawide's ``fisheye-equidistant`` intrinsics and k1..k4 (scaled to the image size), ``IMU.T_b_c1 =
   T_imu_cam`` of the ultrawide (camera to body), IMU noise from ``rig.imu.noise.for_vio()``, ``IMU.Frequency`` =
   the measured rate of the written IMU stream.
3. **Run** (:func:`orbslam3_command`): ``<orb_dir>/Examples/Monocular-Inertial/mono_inertial_euroc
   <orb_dir>/Vocabulary/ORBvoc.txt <work>/orbslam3_settings.yaml <work> <work>/times.txt <name>`` with cwd
   ``<work>`` (where it writes ``f_<name>.txt``).
4. **Import** ``f_<name>.txt`` (:func:`load_orbslam3_euroc`).

ORB-SLAM3 output frame (``System::SaveTrajectoryEuRoC``, v1.0-release, checked against the source): for IMU sensors
every line is ``Twb = (Tbc * Tcr * Trw)^-1``, the pose of the **IMU body** ``b`` (``T_b_c1`` = our ``T_imu_cam``, so
body = our IMU frame) in a world re-based on the **first keyframe's IMU frame** ``b0`` of the **largest map only**
(``Twb0 = vpKFs[0]->GetImuPose()``). Lost frames, and frames of other maps, are skipped. The stream is ``fixed`` with
the time as ``1e9 * t_s`` at 6 decimals, then ``x y z qx qy qz qw``. So each row already is ``T_b0_imu``: the
import keeps the poses as ``T_W_imu`` with ``W = b0`` (arbitrary tilt and yaw), snaps each time back to the exact
input frame time (the ns went through a double in seconds), and :func:`dataproc.slam.base.finalize_trajectory`
gravity-aligns ``W`` from the episode's IMU. ``n_maps`` / ``n_resets`` are counted from the run's log.
"""

from __future__ import annotations

import math
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np

from dataproc.geometry.camera_model import FisheyeEquidistant, PinholeRadtan
from dataproc.geometry.trajectory import Trajectory, load_orbslam3_euroc
from dataproc.slam.base import (
    SlamBackend, SlamError, SlamResult, finalize_trajectory, imu_samples, ultrawide_camera, ultrawide_times_ns,
)

__all__ = [
    "ORB_BIN", "ORB_VOC", "EUROC_IMU_HEADER", "TRAJ_NAME", "scaled_size", "block_average_imu",
    "camera_model_at", "write_euroc", "write_orbslam3_settings", "ORB_SLAM3_DIR_ENV", "default_orb_slam3_dir",
    "orbslam3_command", "subprocess_runner",
    "count_map_events", "IMU_INIT_STRINGS", "imu_init_events", "import_trajectory", "Orbslam3Backend",
]

ORB_SLAM3_DIR_ENV = "ORB_SLAM3_DIR"                  # where ORB-SLAM3 was built (docs/ORBSLAM3.md)
ORB_BIN = "Examples/Monocular-Inertial/mono_inertial_euroc"   # relative to the ORB-SLAM3 dir
ORB_VOC = "Vocabulary/ORBvoc.txt"
SETTINGS_FILE = "orbslam3_settings.yaml"
TRAJ_NAME = "episode"           # mono_inertial_euroc writes f_episode.txt / kf_episode.txt
LOG_FILE = "orbslam3.log"
EUROC_IMU_HEADER = (
    "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
    "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]"
)

#: ``runner(cmd, log_path, timeout_s, cwd) -> returncode``; the default runs ``subprocess``. Tests inject a fake.
Runner = Callable[[list, Path, float, Path], int]


# --------------------------------------------------------------------------------------------- EuRoC prep

def scaled_size(resolution, scale: float) -> tuple[int, int]:
    """Image size for a VIO downscale, snapped to the nearest *exact* uniform scale of ``resolution``.

    Camera models only accept uniform scaling (w'/w == h'/h). Exact sizes are multiples of (w, h) / gcd(w, h): for
    820x616 that is 205x154, so 0.5 gives 410x308.
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


def camera_model_at(rig, scale: float = 1.0, role: str = "ultrawide"):
    """(model, width, height) of the rig camera at the VIO image size (vendored camera_model classes)."""
    cam = rig.cameras[role]
    w0, h0 = cam.resolution
    fx, fy, cx, cy = cam.model.intrinsics
    d = cam.model.distortion
    if cam.model.type == "fisheye-equidistant":
        model = FisheyeEquidistant(w0, h0, fx, fy, cx, cy, *d)
    elif cam.model.type == "pinhole-radtan":
        model = PinholeRadtan(w0, h0, fx, fy, cx, cy, *d)
    else:
        raise ValueError(f"unsupported camera model {cam.model.type!r} for ORB-SLAM3")
    w, h = scaled_size(cam.resolution, scale)
    if (w, h) != (w0, h0):
        model = model.scaled(w, h)
    return model, w, h


def write_euroc(episode, rig, out_dir, scale: float = 1.0, imu_rate_hz: float | None = None,
                frames=None) -> dict:
    """Write the EuRoC folder for ``episode`` into ``out_dir`` (layout: module docstring).

    ``frames`` (optional, for tests) replaces the MP4 decode: an iterable of ``(frame_index, rgb)``.
    Returns a JSON-able summary including ``t_cam_ns`` (the image times actually written).
    """
    import cv2

    out = Path(out_dir)
    cam_dir = out / "mav0" / "cam0"
    (cam_dir / "data").mkdir(parents=True, exist_ok=True)
    imu_dir = out / "mav0" / "imu0"
    imu_dir.mkdir(parents=True, exist_ok=True)

    ti, g, a = imu_samples(episode, rig)
    factor = 1
    if imu_rate_hz and len(ti) > 1:
        rate = 1e9 / float(np.median(np.diff(ti)))
        factor = max(1, int(round(rate / imu_rate_hz)))
        ti, g, a = block_average_imu(ti, g, a, factor)
    if len(ti) < 2:
        raise SlamError("orbslam3: fewer than 2 IMU samples")
    lines = [f"{t},{gi[0]:.9g},{gi[1]:.9g},{gi[2]:.9g},{ai[0]:.9g},{ai[1]:.9g},{ai[2]:.9g}"
             for t, gi, ai in zip(ti.tolist(), g, a)]
    (imu_dir / "data.csv").write_text(EUROC_IMU_HEADER + "\n" + "".join(x + "\n" for x in lines))

    t_all = ultrawide_times_ns(episode, rig)
    keep = (t_all >= ti[0]) & (t_all < ti[-1])
    idx = np.flatnonzero(keep)
    w, h = scaled_size(rig.cameras["ultrawide"].resolution, scale)
    src = frames if frames is not None else episode.read_video_frames(ultrawide_camera(episode), idx.tolist())
    want = set(idx.tolist())
    rows, written = [], []
    for i, rgb in src:
        if i not in want:
            continue
        img = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
        if (img.shape[1], img.shape[0]) != (w, h):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        t = int(t_all[i])
        fn = f"{t}.png"
        if not cv2.imwrite(str(cam_dir / "data" / fn), img):
            raise SlamError(f"orbslam3: cannot write {cam_dir / 'data' / fn}")
        rows.append(f"{t},{fn}")
        written.append(t)
    if len(written) != len(idx):
        raise SlamError(f"orbslam3: decoded {len(written)} of {len(idx)} ultrawide frames")
    (cam_dir / "data.csv").write_text("#timestamp [ns],filename\n" + "".join(r + "\n" for r in rows))
    (out / "times.txt").write_text("".join(f"{t}\n" for t in written))
    imu_rate = 1e9 / float(np.median(np.diff(ti)))
    return {"out_dir": str(out), "n_frames_total": int(len(t_all)), "n_images": len(written),
            "n_dropped_outside_imu": int(len(t_all) - len(idx)), "size": [w, h],
            "scale": w / int(rig.cameras["ultrawide"].resolution[0]), "scale_requested": scale, "n_imu": int(len(ti)),
            "imu_rate_hz": imu_rate, "imu_decimation": factor, "t_cam_ns": written}


def _matrix(name: str, M: np.ndarray) -> str:
    M = np.asarray(M, dtype=np.float64)
    data = ", ".join(f"{v:.9f}" for v in M.reshape(-1))
    return f"{name}: !!opencv-matrix\n   rows: {M.shape[0]}\n   cols: {M.shape[1]}\n   dt: f\n   data: [{data}]"


def write_orbslam3_settings(rig, path, scale: float = 1.0, imu_rate_hz: float | None = None,
                            n_features: int = 1500) -> Path:
    """ORB-SLAM3 v1.0 settings for the ultrawide + IMU (see the module docstring)."""
    model, w, h = camera_model_at(rig, scale)
    lines = ["%YAML:1.0", "---", "# Generated by dataproc (orbslam3 backend). ORB-SLAM3 settings format v1.0.",
             'File.version: "1.0"', ""]
    if model.kind == "fisheye-equidistant":
        lines.append('Camera.type: "KannalaBrandt8"')
        dist = [("k1", model.k1), ("k2", model.k2), ("k3", model.k3), ("k4", model.k4)]
    else:
        lines.append('Camera.type: "PinHole"')
        dist = [("k1", model.k1), ("k2", model.k2), ("p1", model.p1), ("p2", model.p2)]
    for k, v in [("fx", model.fx), ("fy", model.fy), ("cx", model.cx), ("cy", model.cy)] + dist:
        lines.append(f"Camera1.{k}: {float(v):.9f}")
    if not imu_rate_hz or imu_rate_hz <= 0:
        raise ValueError("imu_rate_hz is required (the measured rate of the written IMU stream)")
    rate = float(imu_rate_hz)
    noise = rig.imu.noise.for_vio()
    lines += [
        f"Camera.width: {w}",
        f"Camera.height: {h}",
        f"Camera.fps: {int(round(float(rig.cameras['ultrawide'].fps)))}",
        "Camera.RGB: 0",
        "",
        "# Transformation from camera to body-frame (IMU): T_b_c1 = T_imu_cam (ultrawide)",
        _matrix("IMU.T_b_c1", rig.cameras["ultrawide"].T_imu_cam),
        "IMU.InsertKFsWhenLost: 0",
        f"IMU.NoiseGyro: {noise.gyroscope_noise_density:.9g}",
        f"IMU.NoiseAcc: {noise.accelerometer_noise_density:.9g}",
        f"IMU.GyroWalk: {noise.gyroscope_random_walk:.9g}",
        f"IMU.AccWalk: {noise.accelerometer_random_walk:.9g}",
        f"IMU.Frequency: {float(rate):.1f}",
        "",
        # defaults from ORB-SLAM3 Examples/Monocular-Inertial/TUM-VI.yaml
        f"ORBextractor.nFeatures: {int(n_features)}",
        "ORBextractor.scaleFactor: 1.2",
        "ORBextractor.nLevels: 8",
        "ORBextractor.iniThFAST: 20",
        "ORBextractor.minThFAST: 7",
        "",
        "Viewer.KeyFrameSize: 0.05",
        "Viewer.KeyFrameLineWidth: 1.0",
        "Viewer.GraphLineWidth: 0.9",
        "Viewer.PointSize: 2.0",
        "Viewer.CameraSize: 0.08",
        "Viewer.CameraLineWidth: 3.0",
        "Viewer.ViewpointX: 0.0",
        "Viewer.ViewpointY: -0.7",
        "Viewer.ViewpointZ: -3.5",
        "Viewer.ViewpointF: 500.0",
    ]
    path = Path(path)
    path.write_text("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------------------------- run

def default_orb_slam3_dir() -> Path:
    """``$ORB_SLAM3_DIR`` if set, else ``~/ORB_SLAM3``."""
    return Path(os.environ.get(ORB_SLAM3_DIR_ENV) or Path.home() / "ORB_SLAM3").expanduser()


def orbslam3_command(orb_dir, work_dir, name: str = TRAJ_NAME) -> list[str]:
    """The native ``mono_inertial_euroc`` call for one episode; run it with cwd ``work_dir`` (where it writes
    ``f_<name>.txt``)."""
    orb_dir, work_dir = Path(orb_dir).resolve(), Path(work_dir).resolve()
    return [str(orb_dir / ORB_BIN), str(orb_dir / ORB_VOC), str(work_dir / SETTINGS_FILE), str(work_dir),
            str(work_dir / "times.txt"), name]


def subprocess_runner(cmd: list, log_path: Path, timeout_s: float, cwd: Path) -> int:
    """Run ``cmd`` in ``cwd`` with stdout+stderr to ``log_path``; the return code (SlamError if it is missing)."""
    exe = Path(cmd[0])
    if not (exe.is_file() and os.access(exe, os.X_OK)):
        raise SlamError(f"orbslam3: {exe} not found or not executable; build ORB-SLAM3 (docs/ORBSLAM3.md) or set "
                        f"${ORB_SLAM3_DIR_ENV}")
    with open(log_path, "a") as log:
        try:
            p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s, cwd=cwd)
        except subprocess.TimeoutExpired:
            raise SlamError(f"orbslam3: timed out after {timeout_s} s, see {log_path}") from None
    return p.returncode


_NEW_MAP = re.compile(r"Creation of new map with id", re.IGNORECASE)
#: ``System::TrackMonocular`` (System.cc) prints this once per executed active-map reset, whoever requested it
#: (Tracking: lost before IMU init, "IMU is not or recently initialized", bad-IMU flag, timestamp jump, wrong
#: initialisation). The requesters' own messages (e.g. Tracking.cc "IMU is not or recently initialized. Reseting
#: active map...") are *not* counted, so one reset counts once. Not anchored at the line
#: start: ORB-SLAM3's threads share stdout and can interleave on one line. "Resetting" is accepted too.
_RESET = re.compile(r"SYSTEM-> Reset(?:t)?ing active map")


def count_map_events(log_text: str) -> tuple[int, int]:
    """(n_maps, n_resets) from ORB-SLAM3's stdout: maps = "Creation of new map with id" lines (>= 1), resets =
    "SYSTEM-> Reseting active map" lines (ORB-SLAM3's spelling; one per executed reset, see ``_RESET``).

    LocalMapping's "Not enough motion for initializing. Reseting..." (a LocalMapping-side map reset during the IMU
    initialisation; it also raises the bad-IMU flag, so Tracking then requests a System reset that *is* counted
    here) is reported separately by :func:`imu_init_events` (``not_enough_motion_reset``)."""
    n_maps = max(1, len(_NEW_MAP.findall(log_text)))
    n_resets = len(_RESET.findall(log_text))
    return n_maps, n_resets


#: ``cout`` strings of ORB-SLAM3 v1.0-release ``LocalMapping.cc`` (``Run`` / ``InitializeIMU``), checked against the
#: upstream source. The first IMU initialisation prints nothing; VIBA 1 / 2 are the refinements after ~5 s / ~15 s.
IMU_INIT_STRINGS = {
    "viba1_done": "end VIBA 1",
    "viba2_done": "end VIBA 2",
    "not_enough_motion_reset": "Not enough motion for initializing. Reseting...",
    "scale_too_small": "scale too small",
}


def imu_init_events(log_text: str) -> dict:
    """Counts of the IMU-initialisation messages in ORB-SLAM3's stdout (diagnostics only: their absence does not
    mean the IMU was never initialised, the first initialisation is silent)."""
    return {k: log_text.count(v) for k, v in IMU_INIT_STRINGS.items()}


def import_trajectory(path, t_frames_ns, snap_tol_ns: int = 1_000_000) -> tuple[Trajectory, dict]:
    """Load ``f_<name>.txt`` as ``T_W_imu`` (W = ORB-SLAM3's first-keyframe IMU frame, see module docstring) and
    snap each time to the nearest of ``t_frames_ns`` (the times written to times.txt). Rows further than
    ``snap_tol_ns`` from any frame are dropped (and counted)."""
    traj = load_orbslam3_euroc(path, frame="imu")
    t_frames = np.asarray(sorted(set(int(t) for t in t_frames_ns)), dtype=np.int64)
    info = {"n_rows": len(traj), "n_unmatched": 0, "max_snap_ns": 0}
    if len(traj) == 0 or len(t_frames) == 0:
        return Trajectory(np.zeros(0, np.int64), np.zeros((0, 3)), np.zeros((0, 4)), frame="imu",
                          world="orbslam3_b0"), info
    j = np.clip(np.searchsorted(t_frames, traj.t_ns), 1, len(t_frames) - 1) if len(t_frames) > 1 else \
        np.zeros(len(traj), dtype=np.int64)
    if len(t_frames) > 1:
        left, right = t_frames[j - 1], t_frames[j]
        nearest = np.where(np.abs(traj.t_ns - left) <= np.abs(right - traj.t_ns), left, right)
    else:
        nearest = np.full(len(traj), t_frames[0])
    err = np.abs(nearest - traj.t_ns)
    ok = err <= snap_tol_ns
    T = traj.T()[ok]
    t = nearest[ok]
    # duplicates after snapping: keep the last row for each frame (as load_tum does for repeated times)
    keep = np.append(np.diff(t) != 0, True) if len(t) else np.zeros(0, bool)
    order_ok = len(t) < 2 or np.all(np.diff(t[keep]) > 0)
    if not order_ok:
        raise SlamError(f"orbslam3: {path}: times are not increasing after snapping to the frame times")
    info.update(n_unmatched=int((~ok).sum()), max_snap_ns=int(err[ok].max()) if ok.any() else 0,
                n_duplicates=int((~keep).sum()))
    return Trajectory.from_T(t[keep], T[keep], frame="imu", world="orbslam3_b0"), info


# --------------------------------------------------------------------------------------------- backend

class Orbslam3Backend(SlamBackend):
    """ORB-SLAM3 mono-inertial, native build. Options: ``orb_slam3_dir`` (default ``$ORB_SLAM3_DIR`` or
    ``~/ORB_SLAM3``), ``scale`` (VIO image downscale), ``imu_rate_hz`` (block-average the IMU down to about this
    rate), ``timeout_s``, ``runner`` (inject a fake for tests), ``n_features``, ``motion_limits``."""

    name = "orbslam3"

    def __init__(self, *, orb_slam3_dir=None, scale: float = 1.0, imu_rate_hz: float | None = None,
                 timeout_s: float = 3600.0, runner: Runner | None = None, n_features: int = 1500,
                 motion_limits: dict | None = None, frames=None):
        self.orb_dir = Path(orb_slam3_dir).expanduser() if orb_slam3_dir else default_orb_slam3_dir()
        self.scale, self.imu_rate_hz = float(scale), imu_rate_hz
        self.timeout_s, self.runner = float(timeout_s), runner
        self.n_features, self.motion_limits, self._frames = int(n_features), motion_limits, frames

    def prepare(self, episode, rig, work_dir) -> dict:
        """EuRoC folder + settings in ``work_dir``; returns the prep summary."""
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        info = write_euroc(episode, rig, work, scale=self.scale, imu_rate_hz=self.imu_rate_hz, frames=self._frames)
        write_orbslam3_settings(rig, work / SETTINGS_FILE, scale=self.scale, imu_rate_hz=info["imu_rate_hz"],
                                n_features=self.n_features)
        return info

    def run(self, episode, rig, work_dir) -> SlamResult:
        work = Path(work_dir).resolve()
        for stale in (work / f"f_{TRAJ_NAME}.txt", work / f"kf_{TRAJ_NAME}.txt", work / LOG_FILE):
            stale.unlink(missing_ok=True)
        prep = self.prepare(episode, rig, work)
        cmd = orbslam3_command(self.orb_dir, work, TRAJ_NAME)
        log = work / LOG_FILE
        log.write_text(f"$ cd {shlex.quote(str(work))} && " + " ".join(shlex.quote(c) for c in cmd) + "\n")
        rc = (self.runner or subprocess_runner)(cmd, log, self.timeout_s, work)
        out = work / f"f_{TRAJ_NAME}.txt"
        if rc != 0:
            if re.search(r"Map \d+ has 0 KFs", log.read_text(errors="replace")):
                # ORB-SLAM3 crashes while saving an empty map: it never initialised (no keyframe ever made)
                raise SlamError(f"orbslam3: never initialised (0 keyframes; mono-inertial needs the camera to "
                                f"move with parallax: a static recording cannot work), see {log}")
            raise SlamError(f"orbslam3: mono_inertial_euroc failed (rc={rc}), see {log}")
        if not out.is_file() or out.stat().st_size == 0:
            raise SlamError(f"orbslam3: no trajectory {out} (tracking never initialised?), see {log}")
        log_text = log.read_text(errors="replace")
        n_maps, n_resets = count_map_events(log_text)
        imu_init = imu_init_events(log_text)
        traj, imp = import_trajectory(out, prep["t_cam_ns"])
        if len(traj) < 2:
            raise SlamError(f"orbslam3: {out}: {len(traj)} usable poses")
        t_uw = ultrawide_times_ns(episode, rig)
        prep_d = {k: v for k, v in prep.items() if k != "t_cam_ns"}
        return finalize_trajectory(traj, episode, rig, t_uw, align=True, align_min_deg=0.0,
                                   motion_limits=self.motion_limits, n_maps=n_maps, n_resets=n_resets,
                                   diagnostics={"backend": self.name, "orb_slam3_dir": str(self.orb_dir),
                                                "command": cmd,
                                                "prep": prep_d, "import": imp, "log": str(log),
                                                "imu_init": imu_init,
                                                "n_not_enough_motion_resets": imu_init["not_enough_motion_reset"]})
