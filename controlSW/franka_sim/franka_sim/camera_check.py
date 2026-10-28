"""camera-check (A7): how close are the sim renders to the real training frames? (RE: mujoco + cv2 + av + zarr)

Why: Stage 3 judges a policy that sees rendered images. Its result only means something if the renders are close to
what the policy was trained on, so this command measures the gap on the recorded firstreal demos, at matched poses:

1. Intrinsics: each episode's JSON K -> which ``sensor.training_intrinsics_variants`` entry (the deploy K is
   variant 0; by default only episodes of that variant are used, so the deploy-K render and the zarr frame share K).
2. Tags: the finger ArUco tags (iPhUMI dict + marker sizes from its aruco_config.yaml, the iPhUMI detector re-done
   here: subpixel corners + solvePnP per tag) on the raw UW frames, vs the sim tag centres projected with the
   episode's UW K at the gap of that frame's measured width (zarr width -> ``WidthMap.gap``). Per tag: pixel error
   p50/p95/max, the 3D offset (PnP tvec vs sim), the tag separation (real vs sim). A UW extrinsic correction
   ``C_main_T_C_uw`` is fitted and REPORTED (not written): translation-only (mean 3D offset) and 6-DoF (2D centres,
   rotation prior 5 deg).
3. Hand-body mask: the fraction of each stream covered by the Franka hand body + tooling (and the arm): the real
   iPhUMI frames show none of them. Overlays at open / mid / closed gaps.
4. Image similarity at matched poses: for <= N frames per demo (approach, grasp, lift) the demo TCP is mapped into
   the base frame with the Stage 1 ``base_T_world[name]`` (poses via ``stage1_replay.load_demos`` + ``prepare_demo``),
   the arm is IK'd (``Kin(sm, "tcp")``, failures counted and skipped), the eraser sits at ``grasp_xyz_base`` with the
   grasp-frame TCP yaw before ``grasp_index`` and rigidly between the pads after it, the fingers at the frame's gap.
   Paths: ``plain`` = the deploy path (960x720 at the deploy K; the iPhone wire image and also the rig's virtual
   image), ``rigb0179`` = the rig station with the rig's resampling (``Renderer(rig=...)``: B0179 fisheye UW).
   Both renders and the zarr frames go to 224 with ``get_image_transform_with_border`` (loaded by file path; never
   edited). MAE (0-255) and SSIM (grey, 11x11 Gaussian sigma 1.5, standard constants) over: ``full`` (the image area
   inside the square padding), ``roi`` (render segmentation of fingers + pads + tags) and ``scene`` (everything else
   that is not the robot: no fingers, hand, tooling or arm). Floors with the same regions: ``jpeg_wire`` (the raw
   frame through the deploy wire: 960x720 resize, JPEG q90, decode, 224) and ``next_frame`` (the next zarr frame).

Needs the finger tags: finger mode stock (and a kit whose sides place no tags) exits 2 with "camera-check needs finger
tags; finger_adapter.mode=<m> variant=<v> has none" before any data is read.

Outputs (--out, outside the repo): camera_check.json, <= 30 PNGs (< 1 MB each), renders.npz with the 224 RGB images
of every frame ``predict_demos.deploy_obs`` reads for the queries (main fi-3, fi; UW per ``DemoZarr.uw_index``):
keys ``<path>_d<demo>_f<frame>_main`` / ``<path>_d<demo>_g<uw global>_ultrawide`` for path in plain, rigb0179,
jpeg90wire, plus ``roi_...`` bool masks (plain path) and ``queries`` (JSON) for ``camera-gap``.

    MUJOCO_GL=egl python -m franka_sim camera-check --profile sim-iphone [--robot fr3] [--set k=v ...] \\
        --raw data/raw/2026-10-01 --zarr Z --stage1 results/stage1/panda_ext0.1_my_stage1.json.gz \\
        [--episodes 5 | --demos 6,15] [--frames-per-episode 40] [--rig RIG.yaml | --rig none] --out DIR
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

from franka_sim.model import CONTROLSW, PROFILES, inv_T, load_station

REPO = CONTROLSW.parent
BP_COMMON = REPO / "behavior_prompting" / "behavior_prompting" / "common"
RIG_B0179 = REPO / "sensor_rig" / "sensor_rig" / "configs" / "rig_jetson_imx219_2p1_b0179.yaml"
ARUCO_CFG = REPO / "iPhUMI" / "python_package" / "iphumi" / "demonstration_processing" / "calibration" / "aruco_config.yaml"
TAG_IDS = (0, 1)                 # left, right (fdcal width defaults; raw width = right.x - left.x)
JPEG_Q = 90
SIZE = (960, 720)
OUT_RES = (224, 224)
PNG_MAX = 30
STREAMS = ("main", "ultrawide")
ROBOT_GROUPS = ("fingers", "tags", "hand_body", "tooling", "arm")


# ------------------------------------------------------------------------------------------------ helpers
def _load_file_module(name: str):
    """A behavior_prompting.common module by file path (the package import pulls torch; never edited)."""
    sp = importlib.util.spec_from_file_location(f"_a7_bp_{name}", BP_COMMON / f"{name}.py")
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


_TF: dict = {}


def to224(img_rgb: np.ndarray) -> np.ndarray:
    """behavior_prompting cv_util.get_image_transform_with_border(in_res, (224, 224)) (pad to square, INTER_AREA)."""
    key = (img_rgb.shape[1], img_rgb.shape[0])
    if key not in _TF:
        _TF[key] = _load_file_module("cv_util").get_image_transform_with_border(key, OUT_RES)
    return _TF[key](np.ascontiguousarray(img_rgb))


def mask224(mask: np.ndarray) -> np.ndarray:
    m = np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)
    return to224(m)[..., 0] > 127


def valid224(w: int, h: int) -> np.ndarray:
    """Pixels of the 224 image that come from the image (not the square padding, not blended with it)."""
    return to224(np.full((h, w, 3), 255, np.uint8))[..., 0] == 255


def ssim_map(a_rgb: np.ndarray, b_rgb: np.ndarray) -> np.ndarray:
    """SSIM map (Wang et al. 2004): grey, 11x11 Gaussian sigma 1.5, C1 = (0.01 L)^2, C2 = (0.03 L)^2, L = 255."""
    import cv2

    a = cv2.cvtColor(a_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    b = cv2.cvtColor(b_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    g = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)  # noqa: E731
    mu_a, mu_b = g(a), g(b)
    saa, sbb, sab = g(a * a) - mu_a ** 2, g(b * b) - mu_b ** 2, g(a * b) - mu_a * mu_b
    return ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (saa + sbb + C2))


def region_metrics(a_rgb, b_rgb, regions: dict) -> dict:
    """{region: {"mae", "ssim", "frac"}} over each bool mask (None when the region is empty)."""
    diff = np.abs(a_rgb.astype(np.int16) - b_rgb.astype(np.int16)).mean(axis=2)
    sm = ssim_map(a_rgb, b_rgb)
    out = {}
    for name, m in regions.items():
        n = int(m.sum())
        out[name] = {"mae": float(diff[m].mean()) if n else None, "ssim": float(sm[m].mean()) if n else None,
                     "frac": float(m.mean())}
    return out


def jpeg_roundtrip(bgr: np.ndarray, q: int = JPEG_Q) -> np.ndarray:
    import cv2

    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def wire_resize(bgr: np.ndarray, size=SIZE) -> np.ndarray:
    """As iphone_replay_sender.resize: INTER_AREA down, INTER_LINEAR up."""
    import cv2

    w, h = size
    if bgr.shape[1] == w and bgr.shape[0] == h:
        return bgr
    return cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA if w < bgr.shape[1] else cv2.INTER_LINEAR)


def load_aruco():
    import cv2
    import yaml

    cfg = yaml.safe_load(ARUCO_CFG.read_text())
    ar = cv2.aruco
    did = getattr(ar, cfg["aruco_dict"]["predefined"])
    d = ar.getPredefinedDictionary(did) if hasattr(ar, "getPredefinedDictionary") else ar.Dictionary_get(did)
    sizes = {int(k): float(v) for k, v in cfg["marker_size_map"].items() if k != "default"}
    return d, sizes


def detect_tags(bgr: np.ndarray, K, aruco_dict, sizes) -> dict:
    """iPhUMI detect_localize_aruco_tags_iphone, re-done (subpixel corners, solvePnP per tag, zero distortion),
    with both cv2.aruco APIs as sensor_rig/record/tagcheck.py. {id: {"center_px", "tvec", "rvec", "corners"}}."""
    import cv2

    ar = cv2.aruco
    if hasattr(ar, "ArucoDetector"):
        p = ar.DetectorParameters()
        p.cornerRefinementMethod = ar.CORNER_REFINE_SUBPIX
        corners, ids, _ = ar.ArucoDetector(aruco_dict, p).detectMarkers(bgr)
    else:  # pragma: no cover - OpenCV < 4.7
        p = ar.DetectorParameters_create()
        p.cornerRefinementMethod = ar.CORNER_REFINE_SUBPIX
        corners, ids, _ = ar.detectMarkers(bgr, aruco_dict, parameters=p)
    out = {}
    if ids is None:
        return out
    for i, c in zip(ids.ravel(), corners):
        i = int(i)
        if i not in sizes:
            continue
        s = sizes[i]
        obj = np.array([[-s / 2, s / 2, 0], [s / 2, s / 2, 0], [s / 2, -s / 2, 0], [-s / 2, -s / 2, 0]], np.float32)
        img = c.reshape(-1, 2).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(obj, img, np.asarray(K, float), np.zeros((5, 1), np.float32))
        if ok:
            out[i] = {"center_px": img.mean(axis=0).astype(float), "tvec": tvec.ravel().astype(float),
                      "rvec": rvec.ravel().astype(float), "corners": img.astype(float)}
    return out


def main_uw_relative_pose(main_bgr, uw_bgr, K_main, K_uw, max_side: int = 960) -> dict | None:
    """Real-data-only main <-> UW extrinsic from one simultaneous frame pair (no sim geometry): SIFT matches, ratio
    test, essential matrix (RANSAC) on normalised points, recoverPose. Returns C_main_R_C_uw, the unit direction of
    the UW centre in the main frame and the inlier count; the baseline length is not observable."""
    import cv2

    s = min(1.0, max_side / main_bgr.shape[1])
    m_img = cv2.resize(main_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else main_bgr
    Km = np.array(K_main, float).copy()
    if s < 1:
        Km[0, 0], Km[1, 1] = Km[0, 0] * s, Km[1, 1] * s
        Km[0, 2], Km[1, 2] = s * (Km[0, 2] + 0.5) - 0.5, s * (Km[1, 2] + 0.5) - 0.5
    sift = cv2.SIFT_create(4000)
    k1, d1 = sift.detectAndCompute(cv2.cvtColor(m_img, cv2.COLOR_BGR2GRAY), None)
    k2, d2 = sift.detectAndCompute(cv2.cvtColor(uw_bgr, cv2.COLOR_BGR2GRAY), None)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return None
    good = [a for a, b in cv2.BFMatcher().knnMatch(d1, d2, k=2) if a.distance < 0.75 * b.distance]
    if len(good) < 30:
        return None
    p1 = cv2.undistortPoints(np.float64([k1[g.queryIdx].pt for g in good]).reshape(-1, 1, 2), Km, None)
    p2 = cv2.undistortPoints(np.float64([k2[g.trainIdx].pt for g in good]).reshape(-1, 1, 2), np.asarray(K_uw, float), None)
    E, inl = cv2.findEssentialMat(p1, p2, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=1.5 / float(K_uw[0][0]))
    if E is None or E.shape != (3, 3):
        return None
    n, R, t, inl2 = cv2.recoverPose(E, p1, p2, np.eye(3), mask=inl)
    R_mu = R.T                                   # x_uw = R x_main + t  ->  C_main_R_C_uw = R^T
    c_uw = -R.T @ t.ravel()                      # UW centre in the main frame (unit length)
    return {"R": R_mu, "dir": c_uw / max(1e-12, np.linalg.norm(c_uw)), "n_inliers": int(n), "n_matches": len(good)}


def pct(v, q) -> float | None:
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.percentile(v, q)) if v else None


def stats(v) -> dict:
    v = [float(x) for x in v if x is not None and np.isfinite(x)]
    if not v:
        return {"n": 0}
    return {"n": len(v), "mean": float(np.mean(v)), "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)),
            "max": float(np.max(v))}


def variant_of(K_main, variants) -> int | None:
    for i, v in enumerate(variants or []):
        if abs(float(K_main[0][0]) - float(v["main_1920_f_px"])) < 0.05:
            return i
    return None


class RawFrames:
    """The needed frames of one raw episode video (decoded once, in order), BGR."""

    def __init__(self, it, need: set[int]):
        self.frames = {}
        top = max(need) if need else -1
        for i, f in enumerate(it):
            if i > top:
                break
            if i in need:
                self.frames[i] = f

    def get(self, i):
        return self.frames.get(int(i))


def match_index(frames: RawFrames, guess: int, z224: np.ndarray, window: int = 2) -> tuple[int | None, float]:
    """The raw frame (within +-window of guess) whose 224 image is closest to the zarr frame (min MAE)."""
    best, best_m = None, np.inf
    for j in range(guess - window, guess + window + 1):
        f = frames.get(j)
        if f is None:
            continue
        m = float(np.abs(to224(f[:, :, ::-1]).astype(np.int16) - z224.astype(np.int16)).mean())
        if m < best_m:
            best, best_m = j, m
    return best, best_m


# ------------------------------------------------------------------------------------------------ zarr in RE
def open_zarr(path: str):
    """predict_demos.DemoZarr (read only), with the imagecodecs numcodecs registered by file path (RE has no torch)."""
    _load_file_module("imagecodecs_numcodecs").register_codecs(verbose=False)
    from franka_deploy.tools.predict_demos import DemoZarr

    return DemoZarr(path)


def uw_capture_frame(z, d: int, g: int) -> int:
    """The first main frame of demo d whose upsampled UW index is g (the frame at which UW frame g was captured)."""
    s, n = z.start(d), z.length(d)
    up = z._up[s:s + n]
    hit = np.nonzero(up == g)[0]
    return int(hit[0]) if hit.size else int(np.argmin(np.abs(up - g)))


def needed_frames(z, d: int, fi: int, ws: dict | None = None) -> tuple[list[int], list[int]]:
    """(main frames, global UW frames) that predict_demos.deploy_obs reads at frame fi (img_obs_horizon 2, step 3)."""
    ws = ws or {}
    himg, ds, uds = int(ws.get("img_obs_horizon", 2)), int(ws.get("obs_down_sample_steps", 3)), int(ws.get("ultrawide_down_sample_steps", 1))
    mi = [max(0, fi - ds * (himg - 1 - i)) for i in range(himg)]
    uc, us = z.uw_index(d, fi)
    nv = min(himg, (uc - us) // uds + 1)
    ix = list(range(uc - (nv - 1) * uds, uc + 1, uds))
    ix = [ix[0]] * (himg - len(ix)) + ix
    return mi, ix


def select_frames(n: int, ig: int, k: int, horizon: int = 45) -> list[int]:
    """<= k frames over approach / grasp / lift (fi >= 3, fi + horizon <= n - 1 so the GT future exists)."""
    lo, hi = 3, max(4, n - 1 - horizon)
    a, b = int(np.clip(ig - 15, lo, hi)), int(np.clip(ig + 15, lo, hi))
    na, ng = (k * 7) // 20, (k * 6) // 20
    nl = k - na - ng
    f = np.r_[np.linspace(lo, a, na, endpoint=False), np.linspace(a, b, ng, endpoint=False), np.linspace(b, hi, nl)]
    return sorted({int(round(x)) for x in f})[:k]


# ------------------------------------------------------------------------------------------------ sim side
class SimSide:
    """One station's model + renderer + IK, and the qpos of a matched frame."""

    def __init__(self, station, rig=None, colors=None):
        import mujoco

        from franka_sim.kinematics import Kin
        from franka_sim.model import build_model
        from franka_sim.render import Renderer

        self.st = station
        self.sm = build_model(station)
        tag_guard(station, self.sm)
        self.r = Renderer(self.sm, station, rig=rig, colors=colors)
        self.kin = Kin(self.sm, "tcp")
        m = self.sm.model
        nm = self.sm.names
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)  # noqa: E731
        self.arm_adr = [int(m.jnt_qposadr[jid(nm["joints"][f"j{i}"])]) for i in range(1, 8)]
        self.fing_adr = [int(m.jnt_qposadr[jid(f)]) for f in nm["fingers"].values()]
        self.obj_adr = int(m.jnt_qposadr[jid(nm["joints"]["object"])])
        self.key = np.array(m.key_qpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")], float)
        from franka_deploy.gripper.hand_logic import WidthMap

        self.wm = WidthMap.from_config(station.get("tooling.width_map"))
        es = np.asarray(station.get("scene.eraser.size_m"), float)
        self.eraser_z = float(station.get("scene.table.top_z_base_m")) + es[2] / 2

    def qpos(self, q_arm, gap: float, T_eraser) -> np.ndarray:
        from franka_sim.model import mat_to_quat

        qp = self.key.copy()
        qp[self.arm_adr] = q_arm
        qp[self.fing_adr] = gap / 2
        qp[self.obj_adr:self.obj_adr + 3] = T_eraser[:3, 3]
        qp[self.obj_adr + 3:self.obj_adr + 7] = mat_to_quat(T_eraser[:3, :3])
        return qp

    def tags_in_uw(self, gap: float) -> dict:
        """{"l"/"r": tag centre in the UW OpenCV camera frame} at this gap (rigid w.r.t. the camera: any arm q)."""
        from franka_sim.model import _id, cam_T_cv, forward_at, xform_of

        m = self.sm.model
        d = forward_at(self.sm, gap_m=gap)
        C = inv_T(cam_T_cv(m, d, "cam_uw"))
        Cm = inv_T(cam_T_cv(m, d, "cam_main"))
        out = {}
        for s in ("l", "r"):
            W = xform_of(d, "geom", _id(m, "geom", f"tag_{s}"))
            out[s] = (C @ W)[:3, 3]
            out[s + "_main"] = (Cm @ W)[:3, 3]
        out["C_main_T_C_uw"] = Cm @ inv_T(C)
        return out

    def close(self):
        self.r.close()


def eraser_poses(T_tcp: dict, Tg: np.ndarray, ig: int, xyz, z_rest: float, size) -> dict:
    """Eraser pose per frame: before ig on the table at xyz with its short side along the grasp TCP x; from ig on,
    rigidly between the pads (the grasp-frame relation carried by the TCP)."""
    cx, cy = Tg[0, 0], Tg[1, 0]                       # TCP x (closing axis) in the base xy plane
    short_axis = int(np.argmin(np.asarray(size, float)[:2]))   # 1 for [0.06, 0.02, ...]: the body y axis
    yaw = np.arctan2(cy, cx) - (np.pi / 2 if short_axis == 1 else 0.0)
    E0 = np.eye(4)
    E0[:3, :3] = [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    E0[:3, 3] = [xyz[0], xyz[1], z_rest]
    rel = inv_T(Tg) @ E0
    return {f: (E0 if f < ig else T @ rel) for f, T in T_tcp.items()}


# ------------------------------------------------------------------------------------------------ main
def _stations(args):
    def spec(profile):
        sp = {k: dict(v) for k, v in PROFILES[profile].items()}
        if args.robot:
            sp["station"]["robot"] = args.robot
        return sp

    st = load_station(spec(args.profile), sets=list(args.set))
    st_rig = None
    if args.rig and args.rig != "none":
        st_rig = load_station(spec("sim-rig"), sets=list(args.set))
    return st, st_rig


def _png(path: Path, img_bgr, budget: list) -> None:
    import cv2

    if budget[0] >= PNG_MAX:
        return
    cv2.imwrite(str(path), img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if path.stat().st_size > 1_000_000:          # keep every PNG < 1 MB: halve and rewrite
        small = cv2.resize(img_bgr, (img_bgr.shape[1] // 2, img_bgr.shape[0] // 2), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(path), small, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    budget[0] += 1


def fit_uw_extrinsic(obs: list, C0: np.ndarray, rot_prior_deg: float = 5.0) -> dict:
    """C_main_T_C_uw from tag observations [{"uv": (2,), "P_main": (3,), "K": (3,3), "P_pnp": (3,) | None}].
    translation_only: t' so that the mean 3D tag position (PnP) matches; full: 6-DoF least squares on the 2D centres
    with a Gaussian prior on the rotation change (sigma rot_prior_deg); singular values show what the data constrain."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    if not obs:
        return {"n_obs": 0}
    R0, t0 = C0[:3, :3], C0[:3, 3]

    def proj(R, t, o):
        P = R.T @ (o["P_main"] - t)
        return np.array([o["K"][0, 0] * P[0] / P[2] + o["K"][0, 2], o["K"][1, 1] * P[1] / P[2] + o["K"][1, 2]])

    def err_px(R, t):
        return np.array([np.linalg.norm(proj(R, t, o) - o["uv"]) for o in obs])

    out = {"n_obs": len(obs), "placeholder": {"px_p50": pct(err_px(R0, t0), 50), "px_p95": pct(err_px(R0, t0), 95)}}
    pn = [o for o in obs if o.get("P_pnp") is not None]
    if pn:
        dP = np.mean([o["P_pnp"] - R0.T @ (o["P_main"] - t0) for o in pn], axis=0)   # real - sim, UW frame
        t1 = t0 - R0 @ dP
        C1 = np.eye(4)
        C1[:3, :3], C1[:3, 3] = R0, t1
        out["translation_only"] = {"C_main_T_C_uw": np.round(C1, 6).tolist(), "delta_uw_mm": np.round(dP * 1e3, 2).tolist(),
                                   "px_p50": pct(err_px(R0, t1), 50), "px_p95": pct(err_px(R0, t1), 95)}
    w_rot = 1.0 / np.deg2rad(rot_prior_deg)
    scale = 1.0 / max(1, len(obs)) ** 0.5

    def res(x):
        R = R0 @ Rotation.from_rotvec(x[:3]).as_matrix()
        t = t0 + x[3:]
        r = np.concatenate([proj(R, t, o) - o["uv"] for o in obs])
        return np.concatenate([r * scale, x[:3] * w_rot])

    sol = least_squares(res, np.zeros(6), x_scale=[0.01, 0.01, 0.01, 0.005, 0.005, 0.005])
    R = R0 @ Rotation.from_rotvec(sol.x[:3]).as_matrix()
    t = t0 + sol.x[3:]
    C2 = np.eye(4)
    C2[:3, :3], C2[:3, 3] = R, t
    J = sol.jac[: 2 * len(obs)]
    sv = np.linalg.svd(J / scale, compute_uv=False)
    out["full_6dof"] = {"C_main_T_C_uw": np.round(C2, 6).tolist(), "rot_change_deg": float(np.degrees(np.linalg.norm(sol.x[:3]))),
                        "t_change_mm": np.round(sol.x[3:] * 1e3, 2).tolist(), "px_p50": pct(err_px(R, t), 50),
                        "px_p95": pct(err_px(R, t), 95), "jacobian_singular_values": np.round(sv, 3).tolist(),
                        "rot_prior_deg": rot_prior_deg}
    return out


def summarize_rel_poses(rel: list, C0: np.ndarray) -> dict:
    """Median of the per-pair main <-> UW rotations (chordal L2 mean of the inlier-weighted set) and directions."""
    from scipy.spatial.transform import Rotation

    rel = [r for r in rel if r["n_inliers"] >= 30]
    if not rel:
        return {"n_pairs": 0}
    Rs = Rotation.from_matrix(np.array([r["R"] for r in rel]))
    Rm = Rs.mean(weights=[r["n_inliers"] for r in rel])
    spread = np.degrees((Rs * Rm.inv()).magnitude())
    dirs = np.array([r["dir"] for r in rel])
    dm = np.median(dirs, axis=0)
    dm = dm / np.linalg.norm(dm)
    t0 = C0[:3, 3] / max(1e-12, np.linalg.norm(C0[:3, 3]))
    return {"n_pairs": len(rel), "C_main_R_C_uw": np.round(Rm.as_matrix(), 6).tolist(),
            "rotation_vs_used_deg": float(np.degrees((Rm * Rotation.from_matrix(C0[:3, :3]).inv()).magnitude())),
            "rotvec_deg": np.round(np.degrees(Rm.as_rotvec()), 3).tolist(), "rotation_spread_deg_p50_max": [float(np.median(spread)), float(spread.max())],
            "uw_dir_in_main": np.round(dm, 4).tolist(), "dir_vs_used_deg": float(np.degrees(np.arccos(np.clip(dm @ t0, -1, 1)))),
            "inliers_p50": float(np.median([r["n_inliers"] for r in rel])),
            "note": "real frames only (SIFT + essential matrix); the baseline length is not observable, the used one is "
                    f"{np.linalg.norm(C0[:3, 3]) * 1e3:.1f} mm"}


class NoTags(RuntimeError):
    """The built model has no finger tags (stock fingers, or a kit variant that places none)."""


def no_tags_message(station) -> str:
    fa = station.get("tooling.finger_adapter") or {}
    from franka_sim.model import finger_mode

    try:
        mode = finger_mode(fa)
    except ValueError:
        mode = fa.get("mode")
    return (f"camera-check needs finger tags; finger_adapter.mode={mode} variant={fa.get('kit_variant', 'full')} "
            f"has none")


def tag_guard(station, sm=None) -> None:
    """Raise ``NoTags`` when the station's fingers carry no ArUco tags: before building (mode stock) or, given the built
    model, when ``tag_l`` / ``tag_r`` are missing (e.g. a kit variant whose tag carriers are absent)."""
    from franka_sim.model import finger_mode

    if finger_mode(station.get("tooling.finger_adapter") or {}) == "stock":
        raise NoTags(no_tags_message(station))
    if sm is not None:
        import mujoco

        if any(mujoco.mj_name2id(sm.model, mujoco.mjtObj.mjOBJ_GEOM, f"tag_{s}") < 0 for s in ("l", "r")):
            raise NoTags(no_tags_message(station))


def tag_order(sm, proj: dict) -> tuple:
    """(order, pairing): the model side ("l"/"r") that carries ArUco id TAG_IDS[0], TAG_IDS[1].

    "by_id": ``sm.info["finger_kit"]["tag_ids"]`` ({"tag_l": id, "tag_r": id}) says which id sits on which tag, so a
    swapped holder / camera_face shows up as large errors (the sim tag is compared with the detection of ITS id).
    "by_image_order": no such info (box mode, older model): id 0 = the tag with the smaller image u."""
    ids = ((getattr(sm, "info", None) or {}).get("finger_kit") or {}).get("tag_ids")
    if ids:
        by = {int(ids[f"tag_{s}"]): s for s in ("l", "r")}
        if sorted(by) == sorted(TAG_IDS):
            return tuple(by[t] for t in TAG_IDS), "by_id"
    return tuple(sorted(("l", "r"), key=lambda s: proj[s][0])), "by_image_order"


def run(args) -> int:
    sims: dict = {}
    try:
        return _run(args, sims)
    except NoTags as e:
        print(f"[camera-check] {e}", file=sys.stderr, flush=True)
        return 2
    finally:
        for sd in sims.values():
            sd.close()


def _run(args, sims: dict) -> int:
    import cv2

    from franka_sim import stage1_replay as S1
    from sensor_rig.sim.iphone_reader import IphoneEpisode

    t_start = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    st, st_rig = _stations(args)
    for s_ in (st, st_rig):                      # fail early, before any model is built (A9 stock fingers: no tags)
        if s_ is not None:
            tag_guard(s_)
    tooling_sets = list(args.set)
    colors = json.loads(args.colors) if args.colors else None
    sims["plain"] = SimSide(st, colors=colors)
    if st_rig is not None:
        sims["rigb0179"] = SimSide(st_rig, rig=args.rig, colors=colors)
    base = sims["plain"]
    if "rigb0179" in sims and not np.allclose(sims["rigb0179"].sm.F_T_TCP, base.sm.F_T_TCP, atol=1e-9):
        print("[camera-check] note: the rig tooling F_T_TCP differs from the iPhone tooling: IK is done per station",
              file=sys.stderr)
    s1 = json.load(gzip.open(args.stage1)) if str(args.stage1).endswith(".gz") else json.loads(Path(args.stage1).read_text())
    rec = s1["recommendation"]
    s1_demo = {dm["name"]: dm for dm in s1["demos"]}
    z = open_zarr(args.zarr)
    variants = st.get("sensor.training_intrinsics_variants", None) or []
    raw = Path(args.raw)
    # ---- episode choice
    info_ep = []
    for d, name in enumerate(z.names):
        js = json.loads(next((raw / f"{name}_left").glob("*.json")).read_text()) if (raw / f"{name}_left").is_dir() else None
        info_ep.append({"demo": d, "name": name, "variant": variant_of(js["mainCameraIntrinsics"], variants) if js else None,
                        "in_stage1": name in rec["base_T_world"], "raw": js is not None})
    if args.demos:
        demos = [int(x) for x in str(args.demos).split(",")]
    else:
        ok = [e["demo"] for e in info_ep if e["raw"] and e["in_stage1"] and e["variant"] == 0]
        demos = [ok[int(round(i))] for i in np.linspace(0, len(ok) - 1, min(args.episodes, len(ok)))] if ok else []
    if not demos:
        raise SystemExit("camera-check: no episode with raw data, Stage 1 placement and the deploy-K variant")
    aruco_dict, sizes = load_aruco()
    valid = valid224(*SIZE)
    budget = [0]
    store: dict = {}
    rows: list = []
    tag_rows: list = []
    tag_pairing = None                    # "by_id" | "by_image_order" (tag_order)
    fit_obs: list = []
    rel_poses: list = []
    queries: list = []
    ep_out = []
    ws = {}
    for d in demos:
        name = z.names[d]
        t_ep = time.time()
        ep = IphoneEpisode(raw / f"{name}_left")
        dem = S1.prepare_demo(S1.load_demos(args.zarr, ids=[d])[0])
        n = len(dem["T_zup"])
        ig = int(s1_demo[name]["grasp_index"]) if name in s1_demo else int(dem["i_grasp"])
        bTw = np.array(rec["base_T_world"][name], float)
        q_seed = np.array(rec["start_q"][name] if name in rec.get("start_q", {}) else st.get("robot.home_q_rad"), float)
        fis = select_frames(n, ig, args.frames_per_episode)
        need_main, need_uw, per_q = set(), set(), {}
        for fi in fis:
            mi, ix = needed_frames(z, d, fi, ws)
            per_q[fi] = (mi, ix)
            need_main.update(mi)
            need_uw.update(ix)
        uw_cap = {g: uw_capture_frame(z, d, g) for g in need_uw}
        pose_frames = sorted(need_main | set(uw_cap.values()))
        T_tcp = {f: bTw @ dem["T_zup"][f] for f in pose_frames}
        # -- IK per station (warm-started along the sorted frames from the Stage 1 start q)
        qs = {}
        for path, sd in sims.items():
            if path != "plain" and np.allclose(sd.sm.F_T_TCP, base.sm.F_T_TCP, atol=1e-9) and \
                    sd.st.ids.get("robot") == base.st.ids.get("robot"):
                qs[path] = qs["plain"]
                continue
            qd, q_prev = {}, q_seed
            for f in pose_frames:
                q, inf = sd.kin.ik(T_tcp[f], q_prev, q_rest=q_prev, tol_p=1e-3, tol_r=np.deg2rad(0.5))
                if inf["ok"]:
                    qd[f], q_prev = q, q
            qs[path] = qd
        E = eraser_poses(T_tcp, bTw @ dem["T_zup"][ig], ig, rec["grasp_xyz_base"], base.eraser_z, st.get("scene.eraser.size_m"))
        width = z.demo(d)["w"]
        # -- raw frames (main zarr f <-> raw f-1, UW zarr local <-> raw local; both verified by MAE)
        us = z.uw_index(d, 0)[1]
        uw_slots = ep.uw_slots()
        pair_uw = sorted({per_q[fi][1][-1] - us for fi in fis})[:: max(1, len(fis) // 8)]   # ~8 main/UW pairs
        main_raw = RawFrames(ep.iter_main_frames(), {max(0, f + k) for f in need_main | {fi + 1 for fi in fis} for k in (-3, -2, -1, 0, 1)}
                             | {int(uw_slots[j]) for j in pair_uw if j < len(uw_slots)})
        uw_end = int(z._uw_ends[d])
        uw_raw = RawFrames(ep.iter_uw_frames(), {max(0, g - us + k) for g in need_uw for k in (-2, -1, 0, 1, 2)})
        K_uw = np.asarray(ep.K_uw, float)
        # -- renders of every needed frame
        rend: dict = {}
        segs: dict = {}
        for path, sd in sims.items():
            for f in pose_frames:
                if f not in qs[path]:
                    continue
                gap = sd.wm.gap(float(width[f]))
                qp = sd.qpos(qs[path][f], gap, E[f])
                rend[(path, f)] = sd.r.render(qp)
                if path == "plain":
                    segs[f] = sd.r.segmentation(qp)
        n_ik_fail = {p: len([f for f in pose_frames if f not in qs[p]]) for p in sims}
        # -- store 224 images for deploy_obs (main fi-3, fi; UW frames) and the per-frame metrics
        jw_cache: dict = {}

        def real_main(f):
            return z.main_image(d, f)

        def wire_main(f):
            key = ("m", f)
            if key not in jw_cache:
                j, mae = match_index(main_raw, max(0, f - 1), real_main(f))
                jw_cache[key] = (None, None) if j is None else (to224(jpeg_roundtrip(wire_resize(main_raw.get(j)))[:, :, ::-1]), mae)
            return jw_cache[key]

        def wire_uw(g):
            key = ("u", g)
            if key not in jw_cache:
                j, mae = match_index(uw_raw, g - us, z.uw_image(g))
                jw_cache[key] = (None, None, None) if j is None else (
                    to224(jpeg_roundtrip(wire_resize(uw_raw.get(j)))[:, :, ::-1]), mae, j)
            return jw_cache[key]

        r_groups = base.r
        for fi in fis:
            mi, ix = per_q[fi]
            frames_ok = all(("plain", f) in rend for f in mi) and all(("plain", uw_cap[g]) in rend for g in ix)
            for path in sims:
                for f in mi:
                    if (path, f) in rend:
                        store[f"{path}_d{d}_f{f}_main"] = to224(rend[(path, f)]["main"][:, :, ::-1])
                for g in ix:
                    if (path, uw_cap[g]) in rend:
                        store[f"{path}_d{d}_g{g}_ultrawide"] = to224(rend[(path, uw_cap[g])]["ultrawide"][:, :, ::-1])
            for f in mi:
                if f in segs:
                    store[f"roi_d{d}_f{f}_main"] = mask224(r_groups.mask(segs[f]["main"], ["fingers", "tags"]))
                jm = wire_main(f)
                if jm[0] is not None:
                    store[f"jpeg90wire_d{d}_f{f}_main"] = jm[0]
            for g in ix:
                fg = uw_cap[g]
                if fg in segs:
                    store[f"roi_d{d}_g{g}_ultrawide"] = mask224(r_groups.mask(segs[fg]["ultrawide"], ["fingers", "tags"]))
                ju = wire_uw(g)
                if ju[0] is not None:
                    store[f"jpeg90wire_d{d}_g{g}_ultrawide"] = ju[0]
            if frames_ok:
                queries.append({"demo": int(d), "frame": float(fi), "start_frame": 0.0})
            # metrics at fi (main) and its current UW frame
            g_cur = ix[-1]
            for stream, f_pose, real224, real_next, wire, key_sfx in (
                    ("main", fi, real_main(fi), real_main(min(fi + 1, n - 1)), wire_main(fi)[0], f"f{fi}_main"),
                    ("ultrawide", uw_cap[g_cur], z.uw_image(g_cur), z.uw_image(g_cur + 1 if g_cur + 1 < uw_end else g_cur - 1),
                     wire_uw(g_cur)[0], f"g{g_cur}_ultrawide")):
                if f_pose not in segs:
                    continue
                seg = segs[f_pose][stream]
                roi = mask224(r_groups.mask(seg, ["fingers", "tags"])) & valid
                robot = mask224(r_groups.mask(seg, ROBOT_GROUPS)) & valid
                regions = {"full": valid, "roi": roi, "scene": valid & ~robot}
                row = {"demo": d, "frame": fi, "stream": stream, "phase": "approach" if fi < ig - 15 else ("grasp" if fi <= ig + 15 else "lift"),
                       "gap_m": base.wm.gap(float(width[f_pose])),
                       "hand_frac": float(mask224(r_groups.mask(seg, ["hand_body", "tooling"]))[valid].mean()),
                       "arm_frac": float(mask224(r_groups.mask(seg, ["arm"]))[valid].mean())}
                for path in sims:
                    k = f"{path}_d{d}_{key_sfx}"
                    if k in store:
                        row[path] = region_metrics(store[k], real224, regions)
                if wire is not None:
                    row["floor_jpeg_wire"] = region_metrics(wire, real224, regions)
                row["floor_next_frame"] = region_metrics(real_next, real224, regions)
                rows.append(row)
        # -- tags on the raw UW frames of the queries
        ep_tags = {0: [], 1: []}
        seps = []
        for fi in fis:
            g = per_q[fi][1][-1]
            res = wire_uw(g)
            if res[0] is None:
                continue
            j = res[2]
            img = uw_raw.get(j)
            det = detect_tags(img, K_uw, aruco_dict, sizes)
            fg = uw_cap[g]
            gap = base.wm.gap(float(width[fg]))
            sim = base.tags_in_uw(gap)
            proj = {}
            for s in ("l", "r"):
                P = sim[s]
                proj[s] = np.array([K_uw[0, 0] * P[0] / P[2] + K_uw[0, 2], K_uw[1, 1] * P[1] / P[2] + K_uw[1, 2]])
            order, tag_pairing = tag_order(base.sm, proj)
            tr = {"demo": d, "frame": fi, "uw_raw": j, "gap_m": gap, "width_m": float(width[fg]), "detected": sorted(det)}
            for tid, s in zip(TAG_IDS, order):
                if tid not in det:
                    continue
                e = float(np.linalg.norm(det[tid]["center_px"] - proj[s]))
                ep_tags[tid].append(e)
                tr[f"err_px_{tid}"] = e
                tr[f"d3_mm_{tid}"] = (np.asarray(det[tid]["tvec"]) - sim[s]) * 1e3
                fit_obs.append({"uv": det[tid]["center_px"], "P_main": sim[s + "_main"], "K": K_uw, "P_pnp": det[tid]["tvec"]})
            if all(t in det for t in TAG_IDS):
                seps.append({"real_x_mm": (det[1]["tvec"][0] - det[0]["tvec"][0]) * 1e3,
                             "sim_x_mm": (sim[order[1]][0] - sim[order[0]][0]) * 1e3, "width_mm": float(width[fg]) * 1e3})
            tag_rows.append(tr)
            if fi in (fis[0], fis[len(fis) // 2]):           # two tag overlays per demo
                ov = img.copy()
                for tid, s in zip(TAG_IDS, order):
                    cv2.drawMarker(ov, tuple(int(round(v)) for v in proj[s]), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
                    if tid in det:
                        cv2.polylines(ov, [det[tid]["corners"].astype(np.int32)], True, (0, 255, 0), 1)
                        cv2.drawMarker(ov, tuple(int(round(v)) for v in det[tid]["center_px"]), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 10, 1)
                _png(out / f"tags_d{d}_f{fi}.png", ov, budget)
        # real main <-> UW relative pose from simultaneous raw frames (independent of the sim geometry)
        for j in pair_uw:
            if j >= len(uw_slots):
                continue
            mb, ub = main_raw.get(int(uw_slots[j])), uw_raw.get(j)
            if mb is None or ub is None:
                continue
            rp = main_uw_relative_pose(mb, ub, ep.K_main, K_uw)
            if rp is not None:
                rp.update(demo=d, uw_local=int(j))
                rel_poses.append(rp)
        # side-by-side real | plain | rig (224) for one grasp-phase frame per demo
        fi_s = min(fis, key=lambda f: abs(f - ig))
        tiles = []
        for stream, sfx, real in (("main", f"f{fi_s}_main", real_main(fi_s)),
                                  ("ultrawide", f"g{per_q[fi_s][1][-1]}_ultrawide", z.uw_image(per_q[fi_s][1][-1]))):
            row_t = [real] + [store.get(f"{p}_d{d}_{sfx}", np.zeros_like(real)) for p in sims]
            tiles.append(np.concatenate(row_t, axis=1))
        _png(out / f"compare_d{d}_f{fi_s}.png", np.concatenate(tiles, axis=0)[:, :, ::-1], budget)
        sep = {k: stats([s[k] for s in seps]) for k in ("real_x_mm", "sim_x_mm", "width_mm")} if seps else {}
        if seps:
            sep["sim_minus_real_mm"] = stats([s["sim_x_mm"] - s["real_x_mm"] for s in seps])
        ep_out.append({"demo": d, "name": name, "variant": info_ep[d]["variant"], "K_main": np.asarray(ep.K_main).tolist(),
                       "K_uw": K_uw.tolist(), "grasp_index": ig, "frames": fis, "n_pose_frames": len(pose_frames),
                       "n_ik_fail": n_ik_fail, "n_queries": sum(1 for q in queries if q["demo"] == d),
                       "tag_err_px": {str(t): stats(ep_tags[t]) for t in TAG_IDS},
                       "tag_detect_rate": {str(t): len(ep_tags[t]) / max(1, len(fis)) for t in TAG_IDS},
                       "tag_separation": sep,
                       "tag_offset_uw_mm": {str(t): np.round(np.mean([r[f"d3_mm_{t}"] for r in tag_rows if r["demo"] == d and f"d3_mm_{t}" in r], axis=0), 2).tolist()
                                            if any(f"d3_mm_{t}" in r for r in tag_rows if r["demo"] == d) else None for t in TAG_IDS},
                       "match_mae": {"main": stats([v[1] for k, v in jw_cache.items() if k[0] == "m"]),
                                     "ultrawide": stats([v[1] for k, v in jw_cache.items() if k[0] == "u"])},
                       "wall_s": round(time.time() - t_ep, 1)})
        print(f"[camera-check] demo {d} {name}: {len(fis)} frames, IK fails {n_ik_fail}, "
              f"tag px p50 {ep_out[-1]['tag_err_px']['0'].get('p50')}/{ep_out[-1]['tag_err_px']['1'].get('p50')} "
              f"({ep_out[-1]['wall_s']} s)", file=sys.stderr, flush=True)
    # ---- hand-body mask at home, open / mid / closed
    hand = {}
    q_home = np.asarray(st.get("robot.home_q_rad"), float)
    for lbl, gap in (("open", 0.08), ("mid", 0.04), ("closed", 0.0)):
        from franka_sim.model import forward_at

        qp = forward_at(base.sm, q_home, gap_m=gap).qpos.copy()
        seg = base.r.segmentation(qp)
        img = base.r.render(qp)
        hand[lbl] = {}
        for s in STREAMS:
            mk = base.r.mask(seg[s], ["hand_body", "tooling"])
            ys, xs = np.nonzero(mk)
            hand[lbl][s] = {"hand_tooling_frac": float(mk.mean()),
                            "arm_frac": float(base.r.mask(seg[s], ["arm"]).mean()),
                            "finger_tag_frac": float(base.r.mask(seg[s], ["fingers", "tags"]).mean()),
                            "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if xs.size else None}
            ov = img[s].copy()
            col = np.zeros_like(ov)
            col[mk] = (0, 0, 255)
            col[base.r.mask(seg[s], ["fingers", "tags"])] = (0, 255, 0)
            col[base.r.mask(seg[s], ["arm"])] = (255, 0, 0)
            ov = cv2.addWeighted(ov, 0.6, col, 0.4, 0)
            _png(out / f"mask_{lbl}_{s}.png", cv2.resize(ov, (480, 360), interpolation=cv2.INTER_AREA), budget)
    # ---- summary tables
    def table(rws, stream):
        t = {}
        srcs = [p for p in sims] + ["floor_jpeg_wire", "floor_next_frame"]
        for src in srcs:
            t[src] = {}
            for reg in ("full", "roi", "scene"):
                vals = [r[src][reg] for r in rws if r["stream"] == stream and src in r]
                t[src][reg] = {"mae": stats([v["mae"] for v in vals]), "ssim": stats([v["ssim"] for v in vals]),
                               "frac": pct([v["frac"] for v in vals], 50)}
        return t

    sims_info = {p: {"station_ids": dict(sd.st.ids), "station_sets": tooling_sets, "fingerprint": sd.st.fingerprint(),
                     "renderer": sd.r.info()} for p, sd in sims.items()}
    C0 = base.tags_in_uw(0.04)["C_main_T_C_uw"]
    res = {
        "schema": "camera_check/v1", "profile": args.profile, "robot": st.ids.get("robot"), "sets": tooling_sets,
        "colors": colors,
        "zarr": str(Path(args.zarr).resolve()), "stage1": str(args.stage1), "rig": None if st_rig is None else str(args.rig),
        "demos": demos, "variants_by_demo": {e["demo"]: e["variant"] for e in info_ep},
        "sims": sims_info, "episodes": ep_out,
        "image_similarity": {s: table(rows, s) for s in STREAMS},
        "image_similarity_by_phase": {ph: {s: table([r for r in rows if r["phase"] == ph], s) for s in STREAMS}
                                      for ph in ("approach", "grasp", "lift")},
        "robot_pixels_matched": {s: {"hand_tooling_frac": stats([r["hand_frac"] for r in rows if r["stream"] == s]),
                                     "arm_frac": stats([r["arm_frac"] for r in rows if r["stream"] == s])} for s in STREAMS},
        "hand_mask_home": hand,
        "tags": {"tag_pairing": tag_pairing,
                 "err_px": {str(t): stats([r[f"err_px_{t}"] for r in tag_rows if f"err_px_{t}" in r]) for t in TAG_IDS},
                 "detect_rate": {str(t): float(np.mean([t in r["detected"] for r in tag_rows])) if tag_rows else None for t in TAG_IDS},
                 "offset_uw_mm_mean": {str(t): np.round(np.mean([r[f"d3_mm_{t}"] for r in tag_rows if f"d3_mm_{t}" in r], axis=0), 2).tolist()
                                       if any(f"d3_mm_{t}" in r for r in tag_rows) else None for t in TAG_IDS},
                 "uw_extrinsic_station": "placeholder (UW_PLACEHOLDER)" if st.get("tooling.C_main_T_C_uw", None) is None else "station",
                 "C_main_T_C_uw_used": np.round(C0, 6).tolist(),
                 "fit_C_main_T_C_uw": fit_uw_extrinsic(fit_obs, C0)},
        "uw_extrinsic_from_features": summarize_rel_poses(rel_poses, C0),
        "coverage": sims["rigb0179"].r.coverage if "rigb0179" in sims else None,
        "n_queries": len(queries), "n_frames": len({(r["demo"], r["frame"]) for r in rows}),
        "regions": {"full": "the 224 image area inside the square padding", "roi": "render segmentation of fingers + pads + tags",
                    "scene": "full minus every robot pixel (fingers, tags, hand body, tooling, arm) of the render"},
        "notes": ["main zarr frame f = raw video frame f-1 and UW zarr local frame = raw UW frame (matched by MAE within +-2/3, match_mae)",
                  "floor_jpeg_wire: the raw frame resized to 960x720 (INTER_AREA main / INTER_LINEAR UW, as the replay sender), JPEG q90, 224",
                  "the real frames show no Franka hand body, tooling or arm: robot_pixels_matched is pure sim content"],
        "wall_s": round(time.time() - t_start, 1),
    }
    store["queries"] = np.array(json.dumps(queries))
    store["meta"] = np.array(json.dumps({"paths": list(sims) + ["jpeg90wire"], "demos": demos, "sets": tooling_sets,
                                         "robot": st.ids.get("robot"), "profile": args.profile}))
    np.savez_compressed(out / "renders.npz", **store)
    (out / "camera_check.json").write_text(json.dumps(res, indent=1, default=_json_default))
    print(summary_text(res), flush=True)
    print(f"[camera-check] wrote {out / 'camera_check.json'}, renders.npz ({len(store)} arrays), {budget[0]} PNGs "
          f"in {res['wall_s']} s", flush=True)
    return 0


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    raise TypeError(type(o))


def summary_text(res: dict) -> str:
    f = lambda v, k="p50", fmt="{:.2f}": ("-" if not v or v.get("n", 0) == 0 else fmt.format(v[k]))  # noqa: E731
    lines = [f"camera-check {res['robot']} sets={res['sets']} demos={res['demos']} frames={res['n_frames']} queries={res['n_queries']}"]
    for s in STREAMS:
        t = res["image_similarity"][s]
        lines.append(f"  {s:9s} {'source':16s} {'MAE full':>8s} {'roi':>6s} {'scene':>6s} | {'SSIM full':>9s} {'roi':>6s} {'scene':>6s}")
        for src, v in t.items():
            lines.append(f"  {'':9s} {src:16s} {f(v['full']['mae'], 'mean'):>8s} {f(v['roi']['mae'], 'mean'):>6s} {f(v['scene']['mae'], 'mean'):>6s} | "
                         f"{f(v['full']['ssim'], 'mean', '{:.3f}'):>9s} {f(v['roi']['ssim'], 'mean', '{:.3f}'):>6s} {f(v['scene']['ssim'], 'mean', '{:.3f}'):>6s}")
    tg = res["tags"]["err_px"]
    lines.append(f"  tag pairing: {res['tags'].get('tag_pairing')}")
    lines.append(f"  tags px id0 p50/p95/max {f(tg['0'])}/{f(tg['0'], 'p95')}/{f(tg['0'], 'max')}  id1 {f(tg['1'])}/{f(tg['1'], 'p95')}/{f(tg['1'], 'max')}")
    return "\n".join(lines)


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--profile", default="sim-iphone", choices=sorted(PROFILES))
    p.add_argument("--robot", choices=["panda", "fr3"], default=None)
    p.add_argument("--set", nargs="*", action="extend", default=[], metavar="group.path=value",
                   help="station overrides (applied to the iPhone and the rig station)")
    p.add_argument("--raw", default=str(REPO / "data" / "raw" / "2026-10-01"))
    p.add_argument("--zarr", required=True)
    p.add_argument("--stage1", required=True, help="stage1.json(.gz) whose recommendation.base_T_world places the demos")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--demos", default=None, help="explicit demo indices (default: --episodes spread over the deploy-K variant)")
    p.add_argument("--frames-per-episode", type=int, default=40)
    p.add_argument("--rig", default=str(RIG_B0179), help="rig yaml for the rig path (rig= mode), or 'none'")
    p.add_argument("--colors", default=None, help='ablation only: Renderer colors JSON, e.g. {"table": [0.6, 0.6, 0.6, 1]}')
    p.add_argument("--out", required=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_arguments(ap)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
