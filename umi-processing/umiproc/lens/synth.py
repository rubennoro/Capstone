"""Synthetic lens-calibration recordings, for the tests and for trying ``umiproc lens-calibrate`` without hardware.

The board of :mod:`umiproc.lens.board`, printed on white paper (:data:`PAPER_MARGIN_MM` around it), is rendered
through a **known** fisheye model (the truth), and written as a Pi-format recording (what the Pi 5 writes into
``/data/incoming``: ``cam0.mp4`` = the IMX219, ``cam1.mp4`` = the IMX519, frame CSVs, ``imu.csv``, ``meta.json``,
``MANIFEST.sha256``, ``COMPLETE``), so it passes :func:`umiproc.dp.pi_adapter.inspect_source`.

Rendering (:class:`BoardRenderer`): every output pixel is split into ``ss x ss`` samples at
``u + (k + 0.5) / ss - 0.5`` (OpenCV pixel centres); each sample's ray is the truth model's exact ``unproject``
(computed once per renderer), intersected with the board plane, and the board texture is sampled bilinearly from a
pyramid level at least as fine as the sampling (chosen per frame at the board centre); ``INTER_AREA`` averages the
samples into the pixel. So the image geometry is the truth model up to the sampling. Then a little optical blur and
sensor noise, and a static occluder (the gripper) inside the SLAM mask.

Board poses (:func:`look_pose`): the board centre on the ray through a chosen pixel at a chosen distance, the board
facing the camera (its normal along that ray), then tilted about its own x and y axes and rolled about its normal.
:func:`calibration_poses` is a recording that covers the image outside the mask (centre, edges, top corners) with
distances 0.24-0.50 m and tilts up to 30 deg; :func:`make_recording` holds each pose for a few frames with smooth
moves in between, as DIRECTIONS.md §1 asks of a real recording.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from umiproc.lens import board as lens_board

__all__ = ["TRUTH", "PAPER_MARGIN_MM", "BoardPose", "BoardRenderer", "renderer", "truth_model", "look_pose",
           "calibration_poses", "occluder_mask", "make_recording", "write_pi_recording"]

#: Truth intrinsics near rig_pi_v3's nominal ultrawide (fx, fy, cx, cy, k1..k4) at 1280 x 960.
TRUTH = {"size": (1280, 960), "intrinsics": (574.6, 573.1, 643.2, 476.4), "distortion": (0.021, -0.0052, 0.0011, 0.0)}
PAPER_MARGIN_MM = 14.0           # white paper around the board (a whole number of texels at every pyramid level)
BACKGROUND = 92                  # grey level of everything that is not the paper
_TEXTURE_SCALES = (1, 2, 3, 4, 6)  # pyramid levels: EXACT_PX_PER_MM * scale texels per mm
_MISS = -1.0e5


def truth_model(intrinsics=None, distortion=None, size=None):
    """The truth camera as :class:`umiproc.dp.geometry.camera_model.FisheyeEquidistant` (default :data:`TRUTH`)."""
    from umiproc.dp.geometry.camera_model import FisheyeEquidistant

    w, h = size or TRUTH["size"]
    return FisheyeEquidistant(w, h, *(intrinsics or TRUTH["intrinsics"]), *(distortion or TRUTH["distortion"]))


@dataclass(frozen=True)
class BoardPose:
    """``X_cam = R @ X_board + t`` (metres; board frame as OpenCV's ChArUco frame: origin top-left, x right, y down)."""

    R: np.ndarray
    t: np.ndarray

    def lerp(self, other: "BoardPose", s: float) -> "BoardPose":
        """Pose between ``self`` (s=0) and ``other`` (s=1): rotation by slerp, translation linearly."""
        from scipy.spatial.transform import Rotation, Slerp

        rots = Rotation.from_matrix(np.stack([self.R, other.R]))
        R = Slerp([0.0, 1.0], rots)([s]).as_matrix()[0]
        return BoardPose(R, (1.0 - s) * self.t + s * other.t)


def _rot(axis: str, deg: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_euler(axis, deg, degrees=True).as_matrix()


def look_pose(model, pixel, distance_m: float, tilt_x_deg: float = 0.0, tilt_y_deg: float = 0.0,
              roll_deg: float = 0.0) -> BoardPose:
    """The board centre on the ray through ``pixel`` at ``distance_m``, facing the camera, then tilted/rolled."""
    rays, ok = model.unproject(np.asarray(pixel, np.float64)[None])
    if not ok[0]:
        raise ValueError(f"pixel {pixel} has no ray in the model")
    z = rays[0]
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1) @ _rot("x", tilt_x_deg) @ _rot("y", tilt_y_deg) @ _rot("z", roll_deg)
    bw, bh = lens_board.board_size_mm()
    centre_b = np.array([bw / 2000.0, bh / 2000.0, 0.0])
    return BoardPose(R, distance_m * z - R @ centre_b)


def calibration_poses(model, keep=None, seed: int = 0, n_extra: int = 0) -> list[BoardPose]:
    """A recording's worth of poses, as DIRECTIONS.md §1 asks a user to record: a 5 x 4 grid of directions over
    the image at distances and tilts from a fixed rotating schedule (up to 30 deg), plus the edges and corners
    with the board turned to face the lens (top-left, top-right and bottom-left corner twice each, left, right and
    top edge). Directions whose pixel is masked in ``keep`` (uint8, 0 = masked) are skipped. ``n_extra`` more
    random poses (seeded)."""
    W, H = model.width, model.height
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    grid = [(x, y) for y in (150, 350, 560, 760) for x in (170, 405, 640, 875, 1110)]
    sched = [(0.30, 0, 0, 0), (0.26, 30, 0, 10), (0.36, 0, -30, -8), (0.24, -25, 20, 5), (0.40, 20, 25, -12),
             (0.32, -30, -15, 15), (0.28, 15, -30, 0), (0.42, -10, 30, 8)]
    targets = [((u, v),) + sched[i % len(sched)] for i, (u, v) in enumerate(grid)]
    diag = math.atan2(H / 2.0, W / 2.0)
    for sx, sy in ((-1, -1), (1, -1), (-1, 1)):                     # top-left, top-right, bottom-left corners
        for rad, d, roll in ((600, 0.40, 0), (660, 0.50, 30)):
            targets.append(((cx + sx * rad * math.cos(diag), cy + sy * rad * math.sin(diag)), d, 0, 0, roll))
    targets += [((90, cy), 0.40, 0, 0, 0), ((W - 90, cy), 0.40, 0, 0, 10), ((cx, 70), 0.36, 0, 0, -10)]
    poses = []
    for (u, v), d, ax, ay, roll in targets:
        if keep is not None and keep[int(round(v)), int(round(u))] == 0:
            continue
        poses.append(look_pose(model, (u, v), d, ax, ay, roll))
    rng = np.random.default_rng(seed)
    while n_extra > 0:
        u, v = rng.uniform(0, W - 1), rng.uniform(0, H - 1)
        if keep is not None and keep[int(v), int(u)] == 0:
            continue
        poses.append(look_pose(model, (u, v), rng.uniform(0.22, 0.42), *rng.uniform(-35, 35, 2), rng.uniform(-20, 20)))
        n_extra -= 1
    return poses


def _pixel_rays(model, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """float32 unit rays ``(..., 3)`` of pixels ``(u, v)`` through the fisheye ``model``; ``(0, 0, 0)`` where the
    model has none. Same result as ``model.unproject`` (to ~1e-9 rad, tests), but the radial inverse
    ``theta_d -> theta`` is a dense monotone table instead of a per-pixel Newton solve, so 11 M samples take
    about a second."""
    xd = (u - model.cx) / model.fx
    yd = (v - model.cy) / model.fy
    td = np.hypot(xd, yd)
    theta_tab = np.linspace(0.0, model.theta_valid, 400_001)
    td_tab = model._theta_d(theta_tab)
    theta = np.interp(td, td_tab, theta_tab)
    ok = td <= td_tab[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(td > 1e-12, np.sin(theta) / td, 1.0)
    rays = np.stack([xd * s, yd * s, np.cos(theta)], axis=-1)
    rays[~ok] = 0.0                                     # no ray: never hits the board (a_z = 0)
    return rays.astype(np.float32)


class BoardRenderer:
    """Renders the paper-mounted board at a :class:`BoardPose` through ``model`` (see the module docstring)."""

    def __init__(self, model, ss: int = 3, background: int = BACKGROUND):
        self.model, self.ss, self.background = model, int(ss), int(background)
        W, H, s = model.width, model.height, self.ss
        offs = (np.arange(s) + 0.5) / s - 0.5
        u = (np.arange(W)[:, None] + offs[None, :]).ravel()
        v = (np.arange(H)[:, None] + offs[None, :]).ravel()
        uu, vv = np.meshgrid(u, v)
        self.rays = _pixel_rays(model, uu, vv)
        self.levels = [(lens_board.EXACT_PX_PER_MM * k, lens_board.board_image(k, PAPER_MARGIN_MM))
                       for k in _TEXTURE_SCALES]
        bw, bh = lens_board.board_size_mm()
        m = PAPER_MARGIN_MM
        t = np.linspace(0.0, 1.0, 64)
        edges = [np.stack([-m + t * (bw + 2 * m), np.full_like(t, -m)], 1),
                 np.stack([np.full_like(t, bw + m), -m + t * (bh + 2 * m)], 1),
                 np.stack([-m + t * (bw + 2 * m), np.full_like(t, bh + m)], 1),
                 np.stack([np.full_like(t, -m), -m + t * (bh + 2 * m)], 1)]
        e = np.concatenate(edges) / 1000.0
        self._outline = np.concatenate([e, np.zeros((len(e), 1))], axis=1)       # paper outline, board metres

    def _bbox(self, pose: BoardPose):
        """Supersampled pixel box that contains the paper (the whole image when part of it cannot be projected)."""
        W, H, s = self.model.width * self.ss, self.model.height * self.ss, self.ss
        P = self._outline @ pose.R.T + pose.t
        uv, ok = self.model.project(P)
        if not ok.all():
            return 0, 0, W, H
        x0, y0 = np.floor((uv.min(0) + 0.5) * s).astype(int) - 2 * s
        x1, y1 = np.ceil((uv.max(0) + 0.5) * s).astype(int) + 2 * s
        return max(0, x0), max(0, y0), min(W, x1), min(H, y1)

    def _level(self, pose: BoardPose):
        """The coarsest texture level at least as fine as the sampling at the board centre."""
        bw, bh = lens_board.board_size_mm()
        c = np.array([bw / 2000.0, bh / 2000.0, 0.0])
        P = np.stack([c, c + [0.001, 0, 0], c + [0, 0.001, 0]]) @ pose.R.T + pose.t
        uv, ok = self.model.project(P)
        if not ok.all():
            return self.levels[-1]
        need = self.ss * max(np.linalg.norm(uv[1] - uv[0]), np.linalg.norm(uv[2] - uv[0]))   # samples per mm
        for p, tex in self.levels:
            if p >= need:
                return p, tex
        return self.levels[-1]

    def render(self, pose: BoardPose) -> np.ndarray:
        """uint8 grey ``(H, W)`` image of the board at ``pose`` (no blur, noise or occluder)."""
        W, H, s = self.model.width, self.model.height, self.ss
        canvas = np.full((H * s, W * s), self.background, np.uint8)
        x0, y0, x1, y1 = self._bbox(pose)
        if x1 > x0 and y1 > y0:
            p, tex = self._level(pose)
            a = self.rays[y0:y1, x0:x1] @ pose.R.astype(np.float32)            # rays in board axes
            o = (-pose.R.T @ pose.t).astype(np.float32)                         # camera centre in board frame
            with np.errstate(divide="ignore", invalid="ignore"):
                lam = -o[2] / a[..., 2]
                hit = np.isfinite(lam) & (lam > 0)
                bx = (o[0] + lam * a[..., 0]) * 1000.0                          # board mm
                by = (o[1] + lam * a[..., 1]) * 1000.0
            mx = np.where(hit, (bx + PAPER_MARGIN_MM) * p - 0.5, _MISS).astype(np.float32)
            my = np.where(hit, (by + PAPER_MARGIN_MM) * p - 0.5, _MISS).astype(np.float32)
            canvas[y0:y1, x0:x1] = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                             borderValue=self.background)
        if s == 1:
            return canvas
        return cv2.resize(canvas, (W, H), interpolation=cv2.INTER_AREA)

    def true_corners(self, pose: BoardPose) -> tuple[np.ndarray, np.ndarray]:
        """``(ids, uv)`` of the board's inner corners the truth model projects into the image."""
        corners = np.asarray(lens_board.make_board().getChessboardCorners(), np.float64)
        uv, ok = self.model.project(corners @ pose.R.T + pose.t)
        ok &= self.model.in_image(uv)
        return np.nonzero(ok)[0], uv[ok]


_RENDERERS: dict = {}


def renderer(model, ss: int) -> "BoardRenderer":
    """A cached :class:`BoardRenderer` (building one takes a few seconds; tests render several clips)."""
    key = (model.width, model.height, model.fx, model.fy, model.cx, model.cy, *model.dist, int(ss))
    if key not in _RENDERERS:
        if len(_RENDERERS) >= 2:
            _RENDERERS.clear()
        _RENDERERS[key] = BoardRenderer(model, ss)
    return _RENDERERS[key]


def occluder_mask(keep: np.ndarray, inset_px: int = 12) -> np.ndarray:
    """bool ``(H, W)``: where the static occluder (the gripper) is drawn: the masked area shrunk by ``inset_px``
    (the real mask was traced with a >= 12 px margin around the gripper)."""
    masked = (keep == 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inset_px + 1, 2 * inset_px + 1))
    return cv2.erode(masked, k, borderType=cv2.BORDER_CONSTANT, borderValue=1).astype(bool)


def _noise_bank(rng, shape, sigma: float, n: int = 12) -> list:
    """``n`` (positive, negative) uint8 pairs of rounded Gaussian noise, applied with saturating cv2.add/subtract
    (as the copied dp sim does): much faster than fresh float noise per frame, and each held frame still differs."""
    bank = []
    for _ in range(n):
        z = np.rint(rng.normal(0.0, sigma, shape)).astype(np.int16)
        bank.append((np.clip(z, 0, 255).astype(np.uint8), np.clip(-z, 0, 255).astype(np.uint8)))
    return bank


def _occluder_texture(occ: np.ndarray, seed: int) -> np.ndarray:
    """Dark plastic with some structure and the two DICT_4X4_50 finger tags (ids 0 and 1) where they fit."""
    rng = np.random.default_rng(seed)
    H, W = occ.shape
    tex = cv2.GaussianBlur(rng.normal(55, 25, (H, W)).astype(np.float32), (0, 0), 3)
    tex = np.clip(tex, 0, 255).astype(np.uint8)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    side = 60
    for tag_id, cx in ((0, 520), (1, 820)):
        tag = cv2.copyMakeBorder(cv2.aruco.generateImageMarker(d, tag_id, side), 10, 10, 10, 10,
                                 cv2.BORDER_CONSTANT, value=255)
        n = tag.shape[0]
        cols = slice(cx - n // 2, cx + n - n // 2)
        for y in range(H - n + 1):                       # highest place where the whole tag is inside the occluder
            if occ[y:y + n, cols].all():
                tex[y:y + n, cols] = tag
                break
    return tex


def make_recording(dst: Path, model=None, poses=None, *, keep=None, hold: int = 3, move: int = 2, ss: int = 3,
                   blur_sigma: float = 0.7, noise_sigma: float = 1.5, seed: int = 0, episode: int = 900,
                   crf: int = 12) -> dict:
    """Render ``poses`` (default :func:`calibration_poses`) into a Pi-format recording at ``dst`` (created).

    Each pose is shown for ``hold`` identical renders (each with its own noise), and consecutive poses are joined by
    ``move`` frames along the path between them (rendered with one sample per pixel: the motion filter drops them
    anyway). ``keep`` (uint8, 0 = masked; default: the repo's rig_pi_v3 gripper mask) places the occluder.
    Returns the truth: model, poses, the frame index of each pose's first held frame, and the frame count.
    """
    from umiproc.paths import repo_config
    from umiproc.slam_config import load_slam_config, mask_keep

    model = model or truth_model()
    if keep is None:
        keep = mask_keep(load_slam_config(repo_config("slam", "rig_pi_v3.yaml")), (model.width, model.height))
    poses = calibration_poses(model, keep) if poses is None else list(poses)
    occ = occluder_mask(keep)
    occ_tex = _occluder_texture(occ, seed)
    rng = np.random.default_rng(seed)
    fine, coarse = renderer(model, ss), renderer(model, 1)
    bank = _noise_bank(rng, (model.height, model.width), noise_sigma) if noise_sigma > 0 else []
    count = [0]

    def finish(img):
        if blur_sigma > 0:
            img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
        if bank:
            pos, neg = bank[count[0] % len(bank)]
            img = cv2.subtract(cv2.add(img, pos), neg)
            count[0] += 1
        np.copyto(img, occ_tex, where=occ)
        return img

    frames, held_at = [], []
    for i, pose in enumerate(poses):
        if i > 0:
            for k in range(1, move + 1):
                frames.append(finish(coarse.render(poses[i - 1].lerp(pose, k / (move + 1)))))
        still = fine.render(pose)
        held_at.append(len(frames))
        frames.extend(finish(still.copy()) for _ in range(hold))
    write_pi_recording(dst, frames, episode=episode, crf=crf)
    return {"model": model, "poses": poses, "held_at": held_at, "hold": hold, "frames": len(frames),
            "renderer": fine}


def _write_mp4(path: Path, frames, size: tuple[int, int], fps: int, crf: int) -> int:
    import av

    n = 0
    with av.open(str(path), "w") as c:
        st = c.add_stream("libx264", rate=fps)
        st.width, st.height, st.pix_fmt = size[0], size[1], "yuv420p"
        st.options = {"crf": str(crf), "preset": "superfast"}
        for f in frames:
            for pkt in st.encode(av.VideoFrame.from_ndarray(f, format="gray")):
                c.mux(pkt)
            n += 1
        for pkt in st.encode():
            c.mux(pkt)
    return n


def write_pi_recording(dst: Path, frames, *, episode: int = 900, fps: int = 30, crf: int = 12,
                       t0_ns: int = 500_000_000_000) -> Path:
    """Write grey ``frames`` (each 1280 x 960) as the IMX219 ``cam0`` of a Pi-format recording at ``dst`` (created),
    with a short black IMX519 ``cam1``, a still IMU and the Pi's ``meta.json`` (the layout of real recordings, e.g.
    ``episode_000117``), then ``MANIFEST.sha256`` and ``COMPLETE``."""
    from umiproc.dp.pi_format import ACCEL_G_PER_LSB, GYRO_DPS_PER_LSB, write_pi_manifest

    dst = Path(dst)
    dst.mkdir(parents=True)
    frames = list(frames)
    h, w = frames[0].shape
    period = 1e9 / fps
    n0 = _write_mp4(dst / "cam0.mp4", frames, (w, h), fps, crf)
    t0 = [t0_ns + int(round(i * period)) for i in range(n0)]
    (dst / "cam0_frames.csv").write_text("frame_index,t_ns,in_episode\n" + "".join(f"{i},{t},1\n"
                                                                                  for i, t in enumerate(t0)))
    n1 = _write_mp4(dst / "cam1.mp4", [np.zeros((1200, 1600), np.uint8)] * 5, (1600, 1200), fps, 30)
    t1 = [t0_ns + 2_000_000 + int(round(i * period)) for i in range(n1)]
    (dst / "cam1_frames.csv").write_text("frame_index,t_ns,in_episode\n" + "".join(f"{i},{t},1\n"
                                                                                  for i, t in enumerate(t1)))
    imu_hz = 104
    n_imu = int(math.ceil((t0[-1] - t0[0]) / 1e9 * imu_hz)) + 2
    az = int(round(1.0 / ACCEL_G_PER_LSB))
    (dst / "imu.csv").write_text("sample_index,t_ns,gx,gy,gz,ax,ay,az\n" + "".join(
        f"{k},{t0_ns + int(round(k * 1e9 / imu_hz))},0,0,0,0,0,{az}\n" for k in range(n_imu)))
    cams = [{"file": "cam0.mp4", "picamera2_index": 1, "model": "imx219", "size": [w, h], "fps": fps,
             "bitrate": 10_000_000, "sensor_mode": "{'bit_depth': 10, 'output_size': (1640, 1232)}", "frames": n0,
             "lead_in_frames": 0, "dropped_frames": 0, "max_gap_ns": int(round(period)), "max_gap_at_s": 0.0},
            {"file": "cam1.mp4", "picamera2_index": 0, "model": "imx519", "size": [1600, 1200], "fps": fps,
             "bitrate": 15_000_000, "sensor_mode": "{'bit_depth': 10, 'output_size': (2328, 1748)}", "frames": n1,
             "lead_in_frames": 0, "dropped_frames": 0, "max_gap_ns": int(round(period)), "max_gap_at_s": 0.0}]
    meta = {"episode": int(episode), "valid": True, "problems": [], "t_start_boottime_ns": t0_ns,
            "t_end_boottime_ns": max(t0[-1], t1[-1]), "duration_s": round((t0[-1] - t0[0]) / 1e9, 3),
            "cameras": cams,
            "imu": {"address": "0x6a", "odr_hz": imu_hz, "samples": n_imu, "max_gap_ns": int(1e9 / imu_hz),
                    "max_gap_at_s": 0.0, "i2c_errors": 0, "units": "raw int16 counts",
                    "accel_g_per_lsb": ACCEL_G_PER_LSB, "gyro_dps_per_lsb": GYRO_DPS_PER_LSB},
            "system_start": {"cpu_temp_c": 40.0, "throttled": "throttled=0x0"},
            "system_end": {"cpu_temp_c": 41.0, "throttled": "throttled=0x0"}}
    (dst / "meta.json").write_text(json.dumps(meta, indent=2))
    write_pi_manifest(dst)
    (dst / "COMPLETE").write_bytes(b"")
    return dst
