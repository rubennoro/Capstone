"""Virtual-camera alignment: make the rig's virtual cameras look where the iPhone's cameras looked.

A remap can only rotate a camera, so the tools estimate ``R_phys_virt`` and *report* the residual translation
(the mount must put the lens centres where the iPhone's were; P2 §5.3).

* Ultrawide: the finger ArUco tags (DICT_4X4_50, 16 mm, ids 0/1) seen with the gripper closed, by the iPhone
  (its JSON K) and by the rig (current virtual UW). The rotation ``R`` (``d_iphone = R d_virt``) is a rotation-only
  (Wahba) fit on the 8 tag-corner *bearings*, so the fingers land on the iPhone's pixels;
  ``R_phys_virt_new = R_phys_virt_old @ R.T``. Kabsch on the PnP 3-D corners (``p_iphone = R3 p_virt + t``) is
  reported as a mount diagnostic only: with all points ~0.1 m away, a lateral translation of 1 mm looks like a
  0.6 deg rotation, and PnP depth noise makes the 3-D fit's rotation unreliable (0.56 deg error on a clean synthetic
  frame where the bearing fit is exact to < 0.05 deg).
* Main: chained from Kalibr's main<->ultrawide rotation and the (assumed parallel) iPhone main/UW axes.
* TCP: ``tcp_pose_in_arkit_frame`` turns a CAD TCP pose in the physical main camera into the
  ``gripper_calibration.yaml`` matrix.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from sensor_rig.transforms import D_CV_ARKIT, make_T, orthonormalize, rotation_angle_deg

__all__ = [
    "detect_tags", "tag_corners_3d", "kabsch", "wahba", "AlignResult", "align_ultrawide", "align_ultrawide_frames",
    "align_main_from_chain", "rotation_from_image_pair", "estimate_iphone_main_uw_rotation",
    "tcp_pose_in_arkit_frame", "tcp_yaml_snippet", "load_iphone_episode", "with_virtual_rotation",
]

MARKER_SIZE_M = 0.016
TAG_IDS = (0, 1)
_DICT = cv2.aruco.DICT_4X4_50


# --------------------------------------------------------------------------------------------- tags

def detect_tags(img: np.ndarray, K: np.ndarray, marker_size: float = MARKER_SIZE_M, ids=TAG_IDS) -> dict:
    """ArUco detection + PnP exactly like iphumi's ``detect_localize_aruco_tags_iphone`` (zero distortion).

    Returns ``{id: {"corners": (4,2), "rvec": (3,), "tvec": (3,)}}`` for the wanted ids.
    """
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(_DICT), params)
    corners, found, _ = det.detectMarkers(gray)
    out = {}
    if found is None:
        return out
    h = marker_size / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)
    for c, i in zip(corners, found.ravel()):
        i = int(i)
        if i not in ids or i in out:
            continue
        c = c.reshape(4, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, c, np.asarray(K, np.float64), np.zeros(5))
        if ok:
            out[i] = {"corners": c, "rvec": rvec.ravel(), "tvec": tvec.ravel()}
    return out


def _corners_3d(tag: dict, marker_size: float) -> np.ndarray:
    h = marker_size / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)
    R, _ = cv2.Rodrigues(tag["rvec"])
    return obj @ R.T + tag["tvec"]


def _closed_frames(dets: list[dict], ids, max_frames: int, closed_margin_m: float) -> list[int]:
    """Indices of frames with all ``ids`` detected and the smallest tag separation (gripper closed)."""
    sep = []
    for k, d in enumerate(dets):
        if all(i in d for i in ids):
            sep.append((float(np.linalg.norm(d[ids[0]]["tvec"] - d[ids[1]]["tvec"])), k))
    if not sep:
        return []
    sep.sort()
    smin = sep[0][0]
    return [k for s, k in sep if s <= smin + closed_margin_m][:max_frames]


def tag_corners_3d(frames_rgb, K, marker_size: float = MARKER_SIZE_M, ids=TAG_IDS) -> np.ndarray | None:
    """Mean 3-D tag corners ``(4*len(ids), 3)`` in the camera frame over frames where all ids are detected."""
    pts = []
    for img in frames_rgb:
        d = detect_tags(img, K, marker_size, ids)
        if all(i in d for i in ids):
            pts.append(np.concatenate([_corners_3d(d[i], marker_size) for i in ids]))
    return np.mean(pts, axis=0) if pts else None


def wahba(a_src, b_dst) -> np.ndarray:
    """Rotation ``R`` minimising ``sum |b_dst - R a_src|^2`` for direction vectors (no translation)."""
    U, _, Vt = np.linalg.svd(np.asarray(a_src, np.float64).T @ np.asarray(b_dst, np.float64))
    return Vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T


def kabsch(P_src, P_dst) -> tuple[np.ndarray, np.ndarray, float]:
    """Rigid fit ``P_dst ~ R @ P_src + t`` (least squares); returns ``(R, t, rms_m)``."""
    P_src, P_dst = np.asarray(P_src, np.float64), np.asarray(P_dst, np.float64)
    cs, cd = P_src.mean(0), P_dst.mean(0)
    H = (P_src - cs).T @ (P_dst - cd)
    U, _, Vt = np.linalg.svd(H)
    S = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ S @ U.T
    t = cd - R @ cs
    rms = float(np.sqrt(np.mean(np.sum((P_src @ R.T + t - P_dst) ** 2, axis=1))))
    return R, t, rms


# --------------------------------------------------------------------------------------------- ultrawide

@dataclass(frozen=True, eq=False)
class AlignResult:
    R_phys_virt: np.ndarray          # new virtual_cameras.ultrawide.rotation
    R_correction: np.ndarray         # rotation-only fit on tag bearings: d_iphone = R d_virt
    t_residual_m: np.ndarray         # Kabsch-3D translation p_iphone = R3 p_virt + t (mount diagnostic, noisy)
    kabsch_vs_bearing_deg: float     # angle between the Kabsch-3D rotation and R_correction
    rms_px: float                    # tag-corner RMS in the iPhone image after the correction
    rms_px_before: float             # ... before it (what the current rig shows)
    n_frames: tuple[int, int]        # (iphone, rig) closed-gripper frames used

    @property
    def angle_deg(self) -> float:
        return rotation_angle_deg(self.R_correction)

    def summary(self) -> str:
        t_mm = np.linalg.norm(self.t_residual_m) * 1e3
        verdict = "OK" if t_mm < 3.0 else "TOO LARGE: fix the mount (parallax is visible) or retrain on rig data"
        return (f"ultrawide correction {self.angle_deg:.3f} deg; Kabsch-3D residual translation {t_mm:.2f} mm "
                f"({verdict}; Kabsch rotation differs by {self.kabsch_vs_bearing_deg:.2f} deg); "
                f"tag-corner RMS {self.rms_px_before:.2f} px -> {self.rms_px:.2f} px; "
                f"frames iphone/rig = {self.n_frames}")


def _project(K, P):
    p = P @ np.asarray(K, np.float64).T
    return p[:, :2] / p[:, 2:]


def align_ultrawide_frames(rig, iphone_frames, K_iphone, rig_frames_bgr, *, marker_size: float = MARKER_SIZE_M,
                           ids=TAG_IDS, max_frames: int = 40, closed_margin_m: float = 0.002) -> AlignResult:
    """Core of :func:`align_ultrawide`: iPhone UW frames (with their K) vs raw physical rig UW frames."""
    K_v = rig.virtual_camera("ultrawide", "export").K
    table = rig.remap("ultrawide", "export")
    det_i = [detect_tags(f, K_iphone, marker_size, ids) for f in iphone_frames]
    virt = [table.apply(np.ascontiguousarray(f)) for f in rig_frames_bgr]
    det_v = [detect_tags(f, K_v, marker_size, ids) for f in virt]
    sel_i = _closed_frames(det_i, ids, max_frames, closed_margin_m)
    sel_v = _closed_frames(det_v, ids, max_frames, closed_margin_m)
    if not sel_i or not sel_v:
        raise ValueError(f"finger tags {list(ids)} not detected together (iphone frames: {len(sel_i)}, "
                         f"rig frames: {len(sel_v)})")
    def mean_over(dets, sel, fn):
        return np.mean([np.concatenate([fn(dets[k][i]) for i in ids]) for k in sel], axis=0)
    uv_i = mean_over(det_i, sel_i, lambda d: d["corners"])
    uv_v = mean_over(det_v, sel_v, lambda d: d["corners"])
    b_i, b_v = _bearings(K_iphone, uv_i), _bearings(K_v, uv_v)
    R = wahba(b_v, b_i)
    rms_after = float(np.sqrt(np.mean(np.sum((_project(K_iphone, b_v @ R.T) - uv_i) ** 2, axis=1))))
    rms_before = float(np.sqrt(np.mean(np.sum((_project(K_iphone, b_v) - uv_i) ** 2, axis=1))))
    P_i = mean_over(det_i, sel_i, lambda d: _corners_3d(d, marker_size))
    P_v = mean_over(det_v, sel_v, lambda d: _corners_3d(d, marker_size))
    R3, t3, _ = kabsch(P_v, P_i)
    R_old = rig.virtual["ultrawide"].R_phys_virt
    return AlignResult(R_phys_virt=orthonormalize(R_old @ R.T), R_correction=R, t_residual_m=t3,
                       kabsch_vs_bearing_deg=rotation_angle_deg(R3.T @ R), rms_px=rms_after,
                       rms_px_before=rms_before, n_frames=(len(sel_i), len(sel_v)))


def _bearings(K, uv) -> np.ndarray:
    b = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(np.asarray(K, np.float64)).T
    return b / np.linalg.norm(b, axis=1, keepdims=True)


def _decode_video(path: Path, max_frames: int | None = None) -> list[np.ndarray]:
    frames = []
    cap = cv2.VideoCapture(str(path))
    try:
        while max_frames is None or len(frames) < max_frames:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
    finally:
        cap.release()
    if not frames:                                     # OpenCV without an ffmpeg backend: fall back to PyAV
        import av
        with av.open(str(path)) as c:
            for fr in c.decode(video=0):
                frames.append(fr.to_ndarray(format="bgr24"))
                if max_frames is not None and len(frames) >= max_frames:
                    break
    return frames


def load_iphone_episode(episode_dir) -> tuple[dict, Path]:
    """``(json_dict, folder)`` of an iPhUMI export folder ``<NAME>/<NAME>.json``."""
    d = Path(episode_dir)
    js = sorted(p for p in d.glob("*.json") if not p.name.startswith("."))
    if len(js) != 1:
        raise ValueError(f"{d}: expected exactly one <NAME>.json, found {[p.name for p in js]}")
    return json.loads(js[0].read_text()), js[0].with_suffix("")


def _rig_uw_frames(src, max_frames: int = 400) -> list[np.ndarray]:
    if isinstance(src, (str, Path)):
        from sensor_rig.raw_episode import RawEpisode
        ep = RawEpisode(Path(src))
        n = len(ep.camera_times("ultrawide"))
        idx = np.unique(np.linspace(0, n - 1, min(n, max_frames)).round().astype(int))
        return [ep.read_image("ultrawide", int(i)) for i in idx]
    return list(src)


def align_ultrawide(rig, iphone_calib_episode_dir, rig_frames_bgr, **kw) -> AlignResult:
    """Estimate the virtual UW rotation from an iPhone gripper-calibration export and rig UW frames (raw
    physical images, or a raw rig episode directory)."""
    js, stem = load_iphone_episode(iphone_calib_episode_dir)
    K = np.asarray(js["ultrawideCameraIntrinsics"], np.float64)
    frames = _decode_video(Path(str(stem) + "_ultrawidergb.mp4"))
    return align_ultrawide_frames(rig, frames, K, _rig_uw_frames(rig_frames_bgr), **kw)


# --------------------------------------------------------------------------------------------- main

def align_main_from_chain(rig, R_iuw_imain=None, R_phys_virt_uw=None) -> np.ndarray:
    """New ``R_phys_virt`` for main: ``R_physmain_physuw @ R_physuw_virtuw @ R_iphoneUW_iphoneMain``."""
    R_pm_pu = rig.T("main", "ultrawide")[:3, :3]
    R_pu_vu = rig.virtual["ultrawide"].R_phys_virt if R_phys_virt_uw is None else np.asarray(R_phys_virt_uw)
    R_iu_im = np.eye(3) if R_iuw_imain is None else np.asarray(R_iuw_imain, np.float64)
    return orthonormalize(R_pm_pu @ R_pu_vu @ R_iu_im)


def rotation_from_image_pair(img_main, img_uw, K_main, K_uw, *, max_parallax_px: float = 1.0):
    """Rotation-only fit ``d_uw = R d_main`` from feature matches of one main/UW pair of a distant scene.

    Returns ``(R, inlier_ratio, median_err_px)``; raises ``ValueError`` when the median residual (UW pixels)
    exceeds ``max_parallax_px`` (scene too close for a rotation-only model) or there are too few matches.
    """
    g1 = img_main if img_main.ndim == 2 else cv2.cvtColor(img_main, cv2.COLOR_BGR2GRAY)
    g2 = img_uw if img_uw.ndim == 2 else cv2.cvtColor(img_uw, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(4000)
    k1, d1 = sift.detectAndCompute(g1, None)
    k2, d2 = sift.detectAndCompute(g2, None)
    if d1 is None or d2 is None:
        raise ValueError("no features found")
    good = [m for m, n in cv2.BFMatcher().knnMatch(d1, d2, k=2) if m.distance < 0.75 * n.distance]
    if len(good) < 20:
        raise ValueError(f"only {len(good)} feature matches")
    p1 = np.float64([k1[m.queryIdx].pt for m in good])
    p2 = np.float64([k2[m.trainIdx].pt for m in good])
    K1i, K2 = np.linalg.inv(K_main), np.asarray(K_uw, np.float64)
    H, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 2.0)
    if H is None:
        raise ValueError("homography estimation failed")
    inl = mask.ravel().astype(bool)
    # rays of inliers -> Wahba / Kabsch on unit vectors (rotation-only model)
    r1 = np.c_[p1[inl], np.ones(inl.sum())] @ K1i.T
    r2 = np.c_[p2[inl], np.ones(inl.sum())] @ np.linalg.inv(K2).T
    r1 /= np.linalg.norm(r1, axis=1, keepdims=True)
    r2 /= np.linalg.norm(r2, axis=1, keepdims=True)
    R = wahba(r1, r2)
    err = np.linalg.norm(_project(K2, r1 @ R.T) - p2[inl], axis=1)
    med = float(np.median(err))
    if med > max_parallax_px:
        raise ValueError(f"median rotation-only residual {med:.2f} px > {max_parallax_px} px: the scene is too "
                         "close (parallax); record facing a scene >= 3 m away")
    return R, float(inl.mean()), med


def estimate_iphone_main_uw_rotation(iphone_episode_dir, max_pairs: int = 50) -> tuple[np.ndarray, float]:
    """``R_iphoneUW_iphoneMain`` from an iPhone demonstration export of a distant scene (median over pairs)."""
    js, stem = load_iphone_episode(iphone_episode_dir)
    K_m = np.asarray(js["mainCameraIntrinsics"], np.float64)
    K_u = np.asarray(js["ultrawideCameraIntrinsics"], np.float64)
    slots = [i for i, s in enumerate(js["ultrawideRGBTimes"]) if s]
    main = _decode_video(Path(str(stem) + "_rgb.mp4"), max_frames=(max(slots) + 1) if slots else 0)
    uw = _decode_video(Path(str(stem) + "_ultrawidergb.mp4"))
    pairs = [(main[i], uw[k]) for k, i in enumerate(slots) if i < len(main) and k < len(uw)]
    pairs = pairs[:: max(1, len(pairs) // max_pairs)][:max_pairs]
    Rs, ratios = [], []
    for a, b in pairs:
        try:
            R, ratio, _ = rotation_from_image_pair(a, b, K_m, K_u)
        except ValueError:
            continue
        Rs.append(R)
        ratios.append(ratio)
    if len(Rs) < max(3, len(pairs) // 4):
        raise ValueError(f"rotation-only fit succeeded on {len(Rs)}/{len(pairs)} pairs; scene too close?")
    from scipy.spatial.transform import Rotation
    R = Rotation.from_matrix(np.stack(Rs)).mean().as_matrix()
    return R, float(np.median(ratios))


# --------------------------------------------------------------------------------------------- TCP

def tcp_pose_in_arkit_frame(rig, T_physmain_tcp) -> np.ndarray:
    """``T_virtmainARKit_TCP = D @ [[R_phys_virt^T, 0], [0, 1]] @ T_physmain_TCP`` for gripper_calibration.yaml."""
    R = rig.virtual["main"].R_phys_virt
    return D_CV_ARKIT @ make_T(R.T) @ np.asarray(T_physmain_tcp, np.float64)


def tcp_yaml_snippet(T: np.ndarray, device: str = "JetsonRig1,1", artag_z: float = 0.098079) -> str:
    rows = ", ".join("[" + ", ".join(f"{v:.8g}" for v in row) + "]" for row in np.asarray(T))
    return (f'  "{device}":\n    gripper_transform:\n      tcp_pose_in_arkit_frame: [{rows}]\n'
            f"      artag_z_distance_from_ultrawide: {artag_z}\n")


def with_virtual_rotation(rig, name: str, R: np.ndarray):
    """Copy of ``rig`` with ``virtual[name].R_phys_virt = R``."""
    v = dataclasses.replace(rig.virtual[name], R_phys_virt=orthonormalize(R))
    return rig.replace(virtual={**rig.virtual, name: v})
