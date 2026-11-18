"""Helpers for the ``test_umiproc_*`` tests (not a test module): tiny Pi-format recordings built with the copied test
fixtures (``tests/dp/fixtures.py`` + ``umiproc.dp.pi_format.contract_to_pi``), and a digest of a folder tree."""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path

from umiproc.dp.pi_format import contract_to_pi

_FIXTURES = Path(__file__).resolve().parent / "dp" / "fixtures.py"


def _dp_fixtures():
    spec = importlib.util.spec_from_file_location("umiproc_dp_test_fixtures", _FIXTURES)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_fx = _dp_fixtures()


def pi_recording(source: Path, number: int, *, n_cam0: int = 12, n_cam1: int = 6, complete: bool = True,
                 seed: int | None = None) -> Path:
    """A valid Pi-format recording ``source/episode_<number:06d>`` (tiny videos); ``seed`` changes its IMU noise
    (another recording with the same Pi episode number)."""
    work = Path(source).parent / f".contract-{Path(source).name}-{number}-{seed}"
    contract = _fx.build_episode(work, n_cam0=n_cam0, n_cam1=n_cam1, episode_number=number if seed is None else seed)
    return contract_to_pi(contract, Path(source) / f"episode_{number:06d}", number, complete=complete)


def tree_digest(path: Path) -> dict:
    """``{relative path: sha256 or 'dir' or 'link -> target'}`` of everything under ``path``."""
    path = Path(path)
    out = {}
    if not path.exists():
        return out
    for dirpath, dirnames, filenames in os.walk(path):
        for n in dirnames + filenames:
            p = Path(dirpath) / n
            rel = str(p.relative_to(path))
            if p.is_symlink():
                out[rel] = f"link -> {os.readlink(p)}"
            elif p.is_dir():
                out[rel] = "dir"
            else:
                out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


# --------------------------------------------------------------------------------------------- umiproc map

def pi_map_recording(source: Path, number: int, *, n_frames: int = 24, seed: int | None = None) -> Path:
    """A Pi-format recording that fits rig_pi_v3's settings (IMX219 at 1280x960 and 30 fps, the IMU at 104 Hz), so
    ``umiproc map`` takes it as a mapping clip. Tiny (``n_frames`` ultrawide frames)."""
    old = dict(_fx.SIZES), _fx.IMU_HZ
    _fx.SIZES["cam1"], _fx.IMU_HZ = (1280, 960), 104
    try:
        return pi_recording(source, number, n_cam0=2 * n_frames, n_cam1=n_frames, seed=seed)
    finally:
        _fx.SIZES.clear()
        _fx.SIZES.update(old[0])
        _fx.IMU_HZ = old[1]


def fake_slam_build(base: Path, monkeypatch, patches=("0001-a.patch", "0002-b.patch"), save_points: bool = True
                    ) -> Path:
    """A native/build.sh-like build under ``base`` (executable stand-in binary, ``BUILD_INFO``, library, vocabulary),
    selected with ``$UMIPROC_SLAM_BINARY`` / ``$ORB_SLAM3_UMI_DIR``; ``ldd`` is replaced by "the binary loads the
    library". The stand-in exits 99, except ``--help``, which names ``--save_points`` unless ``save_points=False``
    (an older build: :func:`umiproc.slam.runner.supports_save_points`). Returns the binary."""
    from umiproc import slam_inputs as SI

    out, lib = base / "umiproc_build", base / "lib" / "libORB_SLAM3.so"
    out.mkdir(parents=True, exist_ok=True)
    lib.parent.mkdir(parents=True, exist_ok=True)
    (base / "Vocabulary").mkdir(exist_ok=True)
    (base / "Vocabulary" / "ORBvoc.txt").write_text("voc")
    binary = out / "umiproc_slam"
    opts = "  --save_points TEXT\\n" if save_points else ""
    binary.write_text(f'#!/bin/sh\nif [ "$1" = --help ]; then printf "Options:\\n  --save_map TEXT\\n{opts}"; exit 0; fi\n'
                      "exit 99\n")
    binary.chmod(0o755)
    lib.write_bytes(b"lib")
    lines = ["fork https://github.com/cheng-chi/ORB_SLAM3.git b741dca", f"library {lib}"]
    lines += [f"patch {hashlib.sha256(n.encode()).hexdigest()[:16]} {n}" for n in patches]
    lines += ["machine test", "built 2026-11-16T12:00:00Z"]
    (out / "BUILD_INFO").write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(SI, "loaded_orbslam3_library", lambda b: lib)
    monkeypatch.setenv("ORB_SLAM3_UMI_DIR", str(base))
    monkeypatch.setenv("UMIPROC_SLAM_BINARY", str(binary))
    return binary


class FakeSlam:
    """Stands in for umiproc_slam (``runner=`` of run_umi_slam): for every call writes the trajectory CSV (one row per
    times.txt line), its ``.t0_ns``, a log ending in ``UMIPROC_SUMMARY``, for ``--save_map`` an atlas and, for
    ``--save_points``, a small map export (points, keyframes = the CSV's keyframe rows, meta) as the front-end does.

    ``pose(k, n) -> (x, y, z, qx, qy, qz, qw)`` and ``state(k, n, mapping)`` / ``keyframe(k)`` shape the rows;
    ``maps(mapping)`` the summary's maps."""

    def __init__(self, *, pose=None, state=None, keyframe=None, maps=None, rc=0):
        self.pose = pose or (lambda k, n: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
        self.state = state or (lambda k, n, mapping: 2)
        self.keyframe = keyframe or (lambda k: k % 5 == 0)
        self.maps = maps or (lambda mapping: [{"id": 0, "keyframes": 12, "map_points": 300, "imu_initialized": True,
                                               "inertial_ba1": True, "inertial_ba2": True}])
        self.rc = rc
        self.calls = []

    def _write_points(self, pts: Path, times, t0: int, n: int) -> None:
        import json

        stem = pts.name[:-4]
        pts.write_text("id,x,y,z,observations\n" + "".join(
            f"{i},{0.1 * (i % 5) - 0.2:.6f},{0.3 + 0.01 * i:.6f},{0.5 + 0.02 * (i // 5):.6f},{2 + i % 4}\n"
            for i in range(20)))
        kfs = []
        for k, t in enumerate(times):
            if self.state(k, n, True) == 2 and self.keyframe(k):
                x, y, z, qx, qy, qz, qw = self.pose(k, n)
                kfs.append(f"{len(kfs)},{k},{(t - t0) / 1e9:.6f},{x},{y},{z},{qx},{qy},{qz},{qw}\n")
        pts.with_name(stem + "_keyframes.csv").write_text("id,frame_id,timestamp,x,y,z,q_x,q_y,q_z,q_w\n" +
                                                          "".join(kfs))
        pts.with_name(stem + "_meta.json").write_text(json.dumps(
            {"format": "umiproc.map_points.v1", "map_id": 0, "keyframes": len(kfs), "map_points": 20,
             "bad_map_points": 0, "first_keyframe_id": 0, "imu_initialized": True, "up_W": [0.0, -1.0, 0.0],
             "T_W_orbworld": None}))

    def __call__(self, cmd, cwd, log_path, timeout_s):
        import json

        args = list(cmd)
        self.calls.append(args)
        mapping = "--save_map" in args
        seq = Path(args[args.index("--sequence") + 1])
        out = Path(args[args.index("--output_trajectory_csv") + 1])
        times = [int(x) for x in (seq / "times.txt").read_text().split()]
        n, t0 = len(times), times[0]
        rows = []
        n_ok = 0
        for k, t in enumerate(times):
            st = self.state(k, n, mapping)
            lost = st not in (2, 3)
            n_ok += st == 2
            p = (0, 0, 0, 0, 0, 0, 0) if lost else self.pose(k, n)
            rows.append(f"{k},{(t - t0) / 1e9:.6f},{st},{'true' if lost else 'false'},"
                        f"{'true' if (self.keyframe(k) and not lost) else 'false'}," + ",".join(f"{v:.9f}" for v in p))
        out.write_text("frame_idx,timestamp,state,is_lost,is_keyframe,x,y,z,q_x,q_y,q_z,q_w\n" + "\n".join(rows) + "\n")
        Path(str(out) + ".t0_ns").write_text(f"{t0}\n")
        if mapping and self.rc == 0:
            Path(args[args.index("--save_map") + 1]).write_bytes(b"atlas")
        if "--save_points" in args and self.rc == 0:
            self._write_points(Path(args[args.index("--save_points") + 1]), times, t0, n)
        summary = {"format": "umiproc.slam_summary.v2", "exit_code": self.rc, "error": None, "frames_total": n,
                   "frames_fed": n, "lost_frames": n - n_ok, "not_tracking_frames": n - n_ok, "stopped_early": False,
                   "stop_reason": None, "state_counts": {"2": n_ok}, "maps": self.maps(mapping), "t0_ns": t0,
                   "map_removed": False, "options": {},
                   "pacing": {"track_ms": {"median": 20.0, "p95": 25.0, "max": 30.0}, "frames_over_period": 0,
                              "feed_rate_hz": 29.9, "recording_rate_hz": 30.0, "feed_wall_s": n / 30.0}}
        Path(log_path).write_text("frames\ninit_success=1\nNew Map created with 300 points\n"
                                  "UMIPROC_SUMMARY " + json.dumps(summary) + "\n")
        return self.rc


def render_tag(K, D, T_cam_tag, size_m, tag_id=13, res=(1280, 960), n=600):
    """An exact rendering of ArUco DICT_4X4_50 ``tag_id`` (black square ``size_m``, white around it) at
    ``T_cam_tag`` through the fisheye model (K, D): every pixel's ray is intersected with the tag plane."""
    import cv2
    import numpy as np

    w, h = res
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), tag_id, n)
    tex = np.full((2 * n, 2 * n), 255, np.uint8)          # the texture spans 2 x size_m, marker in the middle
    tex[n // 2:n // 2 + n, n // 2:n // 2 + n] = marker
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    pts = np.stack([xx.ravel(), yy.ravel()], -1).reshape(-1, 1, 2)
    und = cv2.fisheye.undistortPoints(pts, K, D).reshape(-1, 2)
    rays = np.concatenate([und, np.ones((len(und), 1))], 1)
    R, t = T_cam_tag[:3, :3], T_cam_tag[:3, 3]
    nrm = R[:, 2]
    s = (nrm @ t) / (rays @ nrm)
    u = (rays * s[:, None] - t) @ R                       # tag-frame coordinates (x right, y up)
    col = (u[:, 0] + size_m) / (2 * size_m) * (2 * n)
    row = (size_m - u[:, 1]) / (2 * size_m) * (2 * n)
    mx = col.reshape(h, w).astype(np.float32)
    my = row.reshape(h, w).astype(np.float32)
    img = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    img[(s <= 0).reshape(h, w)] = 255
    return np.repeat(img[:, :, None], 3, axis=2)


# --------------------------------------------------------------------------------------------- synthetic rig motion

GRAVITY = 9.80665


def _rot(axis: str, a):
    from scipy.spatial.transform import Rotation

    return Rotation.from_euler(axis, a).as_matrix()


def _motion(T_imu_cam, yaw_amp: float = 1.6, accel_bias=(0.0, 0.0, 0.0)):
    """The closed-form handheld motion of :func:`rig_motion` (times in seconds on the IMU clock): ``T_G_imu(t)``,
    ``imu(t) -> (gyro, acc)`` and ``T_W_G``."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    T_imu_cam = np.asarray(T_imu_cam, float)
    R_imu_cam = T_imu_cam[:3, :3]
    w = 2 * np.pi * np.array([0.4, 0.3, 0.5])                     # translation frequencies (rad/s)
    amp = np.array([0.10, 0.08, 0.08])
    ph = np.array([0.0, 1.0, 0.5])
    base = np.array([0.0, 0.0, 0.45])

    def p_imu(t):
        t = np.asarray(t, float)[..., None]
        return base + amp * np.sin(w * t + ph)

    def a_imu(t):
        t = np.asarray(t, float)[..., None]
        return -amp * w ** 2 * np.sin(w * t + ph)

    def R_G_cam(t):
        t = np.atleast_1d(np.asarray(t, float))
        yaw = yaw_amp * np.sin(2 * np.pi * 0.04 * t)
        tilt_y = 0.08 * np.sin(2 * np.pi * 0.23 * t)
        tilt_x = 0.08 * np.sin(2 * np.pi * 0.31 * t + 0.3)
        return np.stack([_rot("z", y) @ _rot("x", np.pi) @ _rot("y", a) @ _rot("x", b)
                         for y, a, b in zip(yaw, tilt_y, tilt_x)])

    def R_G_imu(t):
        return np.einsum("nij,kj->nik", R_G_cam(t), R_imu_cam)    # R_G_cam @ R_imu_cam^T

    def T_G_imu(t):
        t = np.atleast_1d(np.asarray(t, float))
        T = np.tile(np.eye(4), (len(t), 1, 1))
        T[:, :3, :3] = R_G_imu(t)
        T[:, :3, 3] = p_imu(t)
        return T

    def imu(t):
        t = np.atleast_1d(np.asarray(t, float))
        R_i = R_G_imu(t)
        acc = np.einsum("nji,nj->ni", R_i, a_imu(t) + [0.0, 0.0, GRAVITY]) + np.asarray(accel_bias, float)
        h = 1e-4
        gyro = Rotation.from_matrix(np.einsum("nji,njk->nik", R_G_imu(t - h), R_G_imu(t + h))).as_rotvec() / (2 * h)
        return gyro, acc

    T_W_G = np.eye(4)
    T_W_G[:3, :3] = Rotation.from_euler("xyz", [25.0, -40.0, 70.0], degrees=True).as_matrix()
    T_W_G[:3, 3] = [0.3, -0.2, 1.1]
    return T_G_imu, imu, T_W_G


def rig_motion(T_imu_cam, *, duration_s: float = 30.0, fps: float = 30.0, imu_hz: float = 104.0,
               yaw_amp: float = 1.6, accel_bias=(0.0, 0.0, 0.0)) -> dict:
    """A handheld clip in which every quantity agrees exactly: the camera looks down at a table (gravity frame G,
    z up, the table tag at its origin, its z axis up) from about 0.45 m, swinging its yaw by +-``yaw_amp`` rad, tilting
    a few degrees and moving 8-10 cm in every direction; the SLAM world W is G rotated and shifted arbitrarily.

    Returns ``t_cam_ns`` (frames), ``T_W_imu``/``T_W_cam`` at the frames (``T_W_cam = T_W_imu @ T_imu_cam``),
    ``T_G_cam`` (for rendering the tag), ``imu_t_ns``, ``acc_imu`` (specific force in the IMU frame,
    ``R_imu_G (a + g up)``, plus ``accel_bias``), ``gyro_imu`` (rad/s), ``T_W_G`` and ``up_W`` (the true up in W)."""
    import numpy as np

    T_imu_cam = np.asarray(T_imu_cam, float)
    T_G_imu_at, imu_at, T_W_G = _motion(T_imu_cam, yaw_amp, accel_bias)
    t_cam_ns = (np.arange(int(round(duration_s * fps))) * (1e9 / fps)).astype(np.int64) + 5_000_000_000
    t_imu_ns = np.arange(t_cam_ns[0] - 50_000_000, t_cam_ns[-1] + 50_000_000, 1e9 / imu_hz).astype(np.int64)
    ts, ti = t_cam_ns * 1e-9, t_imu_ns * 1e-9
    T_G_imu = T_G_imu_at(ts)
    T_G_cam = T_G_imu @ T_imu_cam
    gyro, acc = imu_at(ti)
    T_W_imu = T_W_G @ T_G_imu
    return {"t_cam_ns": t_cam_ns, "T_W_imu": T_W_imu, "T_W_cam": T_W_imu @ T_imu_cam, "T_G_cam": T_G_cam,
            "imu_t_ns": t_imu_ns, "acc_imu": acc, "gyro_imu": gyro, "T_W_G": T_W_G, "up_W": T_W_G[:3, 2].copy()}


def motion_T_W_cam(T_imu_cam, t_ns):
    """:func:`rig_motion`'s camera poses in its SLAM world W at any times (ns on the IMU clock): what a SLAM run of
    a recording made with :func:`moving_demo_recording` would give."""
    import numpy as np

    T_imu_cam = np.asarray(T_imu_cam, float)
    T_G_imu_at, _imu, T_W_G = _motion(T_imu_cam)
    return T_W_G @ T_G_imu_at(np.asarray(t_ns, np.int64) * 1e-9) @ T_imu_cam


def fork_trajectory(t_ns, T_W_cam, keyframe_every: int = 30):
    """A fully tracked :class:`~umiproc.slam.trajectory_csv.ForkTrajectory` (state OK) with every
    ``keyframe_every``-th frame a keyframe."""
    from pathlib import Path

    import numpy as np

    from umiproc.slam.trajectory_csv import ForkTrajectory

    n = len(t_ns)
    tracked = np.ones(n, bool)
    kf = np.zeros(n, bool)
    kf[::keyframe_every] = True
    return ForkTrajectory(path=Path("synthetic.csv"), t0_ns=int(t_ns[0]), t_ns=np.asarray(t_ns, np.int64),
                          state=np.full(n, 2), is_lost=~tracked, is_keyframe=kf, posed=tracked, tracked=tracked,
                          T_W_cam=np.asarray(T_W_cam, float), n_frames_total=n,
                          anomalies={"stale": 0, "bad_pose": 0})


# --------------------------------------------------------------------------------------------- umiproc calibrate / run

class _UniformFrames:
    """``n`` uniform RGB frames of ``size`` made on demand (the fixture's own list of full-size gradients would hold
    hundreds of MB at 1600x1200)."""

    def __init__(self, n: int, size, seed: int):
        self.n, self.size, self.seed = n, tuple(size), seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        import numpy as np

        if not 0 <= i < self.n:
            raise IndexError(i)
        w, h = self.size
        return np.full((h, w, 3), ((i * 8 + self.seed * 40) % 200) + 20, np.uint8)

    def __iter__(self):
        return (self[i] for i in range(self.n))


def pi_demo_recording(source: Path, number: int, *, n_frames: int = 40, seed: int | None = None) -> Path:
    """A Pi-format recording with rig_pi_v3's settings for both cameras (the copied check stage compares them): main
    IMX519 1600x1200 and ultrawide IMX219 1280x960, both at 30 fps, the IMU at 104 Hz; ``n_frames`` frames each
    (uniform images: no finger tags, see :func:`fake_finger_tags`)."""
    old = dict(_fx.SIZES), dict(_fx.FPS), _fx.IMU_HZ, _fx._frames, _fx._write_mp4
    _fx.SIZES.update(cam0=(1600, 1200), cam1=(1280, 960))
    _fx.FPS.update(cam0=30, cam1=30)
    _fx.IMU_HZ = 104
    _fx._frames = lambda n, size, seed: _UniformFrames(n, size, seed)
    _fx._write_mp4 = _write_mp4_fast
    try:
        return pi_recording(source, number, n_cam0=n_frames, n_cam1=n_frames, seed=seed)
    finally:
        _fx.SIZES.clear()
        _fx.SIZES.update(old[0])
        _fx.FPS.clear()
        _fx.FPS.update(old[1])
        _fx.IMU_HZ, _fx._frames, _fx._write_mp4 = old[2], old[3], old[4]


def _write_mp4_fast(path: Path, frames, fps: int) -> None:
    """The fixture's ``_write_mp4`` with libx264's ultrafast preset (large uniform frames)."""
    import av

    with av.open(str(path), "w") as c:
        st = c.add_stream("libx264", rate=fps)
        st.width, st.height, st.pix_fmt = frames[0].shape[1], frames[0].shape[0], "yuv420p"
        st.options = {"crf": "10", "preset": "ultrafast"}
        for f in frames:
            for pkt in st.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                c.mux(pkt)
        for pkt in st.encode():
            c.mux(pkt)


#: The table tag's pose in the SLAM world of :func:`session_folder`'s map (tilted and shifted, z up = tag normal).
def tx_slam_tag_example():
    import numpy as np
    from scipy.spatial.transform import Rotation

    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", [100.0, 10.0, -30.0], degrees=True).as_matrix()
    T[:3, 3] = [0.2, -0.1, 0.6]
    return T


def session_folder(root, source: Path, monkeypatch, *, task: str = "block_raise", number: int = 900,
                   tx_slam_tag=None, n_frames: int = 24, runner=None, **map_kw) -> str:
    """A task folder with a map that passed (``umiproc map`` with :class:`FakeSlam` or ``runner``, checks M4/M5 made
    to pass; ``map_kw`` e.g. ``first_map=True``) and a ``tx_slam_tag.json`` (the fake map run sees no tag, so it is
    written here). Needs :func:`fake_slam_build`. Returns the folder name."""
    import json

    from umiproc.map import build as B
    from umiproc.map import checks as C

    with monkeypatch.context() as m:
        m.setattr(C, "check_m4", lambda *a, **k: C.CheckResult("M4", True, {}, "ok (test)"))
        m.setattr(C, "check_m5", lambda *a, **k: C.CheckResult("M5", True, {}, "ok (test)"))
        pi_map_recording(source, number, n_frames=n_frames)
        o = B.build_map(root, task, f"episode_{number:06d}", source=source, rig_id="rig_pi_v3",
                        log=lambda *a, **k: None, runner=runner or FakeSlam(), **map_kw)
    assert o.passed, o.message
    T = tx_slam_tag_example() if tx_slam_tag is None else tx_slam_tag
    (o.where / B.TAG_POSE_NAME).write_text(json.dumps({"tx_slam_tag": [list(map(float, r)) for r in T]}))
    return o.folder


def fake_finger_tags(monkeypatch, *, visible=None, width=None, nominal_z: float = 0.098):
    """Replace the copied ``detect_episode_tags`` with synthetic finger tags 0 (left) and 1 (right): at frame i (of
    n) the gripper is ``width(i, n)`` metres open (tag centres at x = -+width/2, z = ``nominal_z``) and
    ``visible(i, n)`` gives the set of tag ids seen (default both). The tag-to-tag x distance is the width, so the
    default sweep (0.044-0.124 m, 3 cycles) fits rig_pi_v3's gripper section (closed 0.044 m, opening 0.080 m). The
    real fit, widths and checks run on them."""
    import numpy as np

    from umiproc.dp.stages import width as W

    visible = visible or (lambda i, n: {0, 1})
    width = width or (lambda i, n: 0.044 + 0.040 * (1 - np.cos(2 * np.pi * 3 * i / max(n, 1))))

    def detect(episode, rig, to_virtual, indices=None):
        from umiproc.dp.slam.base import ultrawide_times_ns

        t = ultrawide_times_ns(episode, rig)
        n = len(t)
        out = []
        for i in range(n):
            half = width(i, n) / 2
            f = {}
            for tag, x in ((0, -half), (1, half)):
                if tag in visible(i, n):
                    f[tag] = W.TagDetection(np.zeros(3), np.array([x, 0.045, nominal_z]), np.zeros((4, 2)))
            out.append(f)
        return t, out

    monkeypatch.setattr(W, "detect_episode_tags", detect)
    return detect


def moving_demo_recording(source: Path, number: int, T_imu_cam, *, n_frames: int = 120) -> Path:
    """A :func:`pi_demo_recording` whose IMU (``imu.csv``, the Pi's raw counts) is :func:`rig_motion`'s handheld
    motion at the recording's own sample times, so a fake SLAM run that answers :func:`motion_T_W_cam` at the
    frame times agrees with it: gravity, the metric scale and the time offset can be checked end to end."""
    import numpy as np

    from umiproc.dp.pi_format import ACCEL_G_PER_LSB, GYRO_DPS_PER_LSB, write_pi_manifest

    d = pi_demo_recording(source, number, n_frames=n_frames)
    lines = (d / "imu.csv").read_text().splitlines()
    rows = [ln.split(",") for ln in lines[1:]]
    t = np.array([int(r[1]) for r in rows], np.int64)
    _T, imu_at, _W = _motion(np.asarray(T_imu_cam, float))
    gyro, acc = imu_at(t * 1e-9)
    g = np.rint(gyro * 180.0 / np.pi / GYRO_DPS_PER_LSB).astype(np.int64)
    a = np.rint(acc / 9.80665 / ACCEL_G_PER_LSB).astype(np.int64)
    out = [lines[0]] + [f"{r[0]},{r[1]},{x[0]},{x[1]},{x[2]},{y[0]},{y[1]},{y[2]}" for r, x, y in zip(rows, g, a)]
    (d / "imu.csv").write_text("\n".join(out) + "\n")
    write_pi_manifest(d)
    return d


def session_template(base: Path, demos=(), n_frames: int = 40, moving=()) -> dict:
    """Build once (e.g. in a module fixture): a stand-in SLAM build in ``base/fork``, a data root ``base/root`` with
    a passed session folder (:func:`session_folder`) and a source ``base/src`` with the mapping clip,
    :func:`pi_demo_recording` recordings ``demos`` (Pi episode numbers) and :func:`moving_demo_recording` recordings
    ``moving`` (``{number: n_frames}``: an IMU that follows the synthetic handheld motion). Copy it per test with
    :func:`clone_session`."""
    import pytest

    from umiproc.dp.paths import DataRoot

    base = Path(base)
    src = base / "src"
    src.mkdir(parents=True)
    with pytest.MonkeyPatch.context() as mp:
        fake_slam_build(base / "fork", mp)
        folder = session_folder(DataRoot(base / "root"), src, mp)
    for k in demos:
        pi_demo_recording(src, k, n_frames=n_frames)
    if moving:
        from umiproc.dp.config import load_rig
        from umiproc.paths import repo_config

        T_imu_cam = load_rig(repo_config("rigs", "rig_pi_v3.yaml")).cameras["ultrawide"].T_imu_cam
        for k, n in dict(moving).items():
            moving_demo_recording(src, k, T_imu_cam, n_frames=n)
    return {"base": base, "folder": folder}


def clone_session(template: dict, tmp_path: Path, monkeypatch) -> dict:
    """A copy of :func:`session_template`'s root and source in ``tmp_path`` (the SLAM build stays shared: the map's
    fingerprint names it). Returns ``{root, src, folder, binary, tmp}``."""
    import shutil

    from umiproc.dp.paths import DataRoot

    base = template["base"]
    shutil.copytree(base / "root", tmp_path / "root", symlinks=True)
    shutil.copytree(base / "src", tmp_path / "src", symlinks=True)
    binary = fake_slam_build(base / "fork", monkeypatch)
    return {"root": DataRoot(tmp_path / "root"), "src": tmp_path / "src", "folder": template["folder"],
            "binary": binary, "tmp": tmp_path}
