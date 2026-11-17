"""View selection, the Kannala-Brandt fit, held-out poses, and the model-equivalence check.

**The model.** ``cv2.fisheye`` (``K``, ``D = k1..k4``), ORB-SLAM3's ``KannalaBrandt8`` (``fx fy cx cy k1 k2 k3 k4``)
and the rig's ``fisheye-equidistant`` (:class:`umiproc.dp.geometry.camera_model.FisheyeEquidistant`, Kalibr's
``pinhole-equi``) are one model: ``theta = atan2(r, z)``, ``theta_d = theta (1 + k1 theta^2 + k2 theta^4 +
k3 theta^6 + k4 theta^8)``, ``u = fx theta_d x / r + cx``, ``v = fy theta_d y / r + cy``, no skew. The fit is done
with ``cv2.fisheye.calibrate`` and its parameters are written unchanged. :func:`model_agreement` re-checks the
equivalence numerically on every run by projecting the same rays with all three (ORB-SLAM3's formula transcribed in
:func:`kannala_brandt8_project`).

**View selection** (:func:`select_views`), greedy. Candidates are still frames with enough corners
(:mod:`umiproc.lens.detect`). Each candidate is described by (1) how many of its corners fall in each cell of an
8 x 6 grid over the image (160 x 160 px cells) and (2) its board tilt: the angle between the board normal and the
line of sight to the board centre, from a pose under the rig's nominal lens model, binned as < 10 deg (one bin)
or 10-25 / 25-40 / >= 40 deg, each split into 4 tilt directions (13 bins). Each step takes the candidate with the
largest gain::

    gain = sum over cells of [min(n_cell + k_cell, 20) - min(n_cell, 20)]      new corners in cells holding < 20
         + 15 * [view count of its tilt bin < 4]                              a tilt not yet seen 4 times

(``n_cell``: corners already selected in the cell; ``k_cell``: the candidate's). Ties, and the steps after every
cell and bin are saturated, go to the candidate with more corners, then the earlier frame. A candidate is never
taken if it nearly repeats a selected view (it shares >= half of its corners with it and they moved < 15 px:
the same pose). Selection stops at ``max_views`` or when the candidates run out. Every 5th selected view (the
5th, 10th, ...) is held out of the fit and only used to check it.

**The fit** (:func:`fit_fisheye`): ``cv2.fisheye.calibrate`` with ``CALIB_USE_INTRINSIC_GUESS`` (start from the
rig's nominal K and k1..k4), ``CALIB_RECOMPUTE_EXTRINSIC``, ``CALIB_FIX_SKEW`` and ``CALIB_CHECK_COND``. Ill-
conditioned views are handled in three ways, each removed view recorded with its reason: (1) before any fit,
:func:`screen_views` runs every view alone through OpenCV's own pose initialisation and condition check with all
intrinsics fixed at the nominal (about 2 ms a view); a view OpenCV cannot initialise (e.g. corners on a line, which
fails an assertion in ``InitExtrinsics`` that names no view) or whose pose Jacobian is ill-conditioned is dropped;
calibrate screens the candidates this way before selecting views, so held-out views are screened too; (2) when
``CALIB_CHECK_COND`` still rejects a view during the joint fit, OpenCV names it and that view is dropped and the fit
repeated; (3) after a fit, the view with the largest RMS reprojection error is dropped and the fit repeated while
that RMS exceeds both 3 x the median view RMS and 1 px (a mis-detected or blurred view), at most 20 % of the
views.

**Held-out poses** (:func:`pose_from_model`): the board pose of a held-out view from the fitted model alone:
the corners are undistorted with the model, ``solvePnP`` (IPPE, planar) gives an initial pose, refined by
Levenberg-Marquardt on the pixel reprojection error under the model.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

__all__ = ["GRID", "CELL_TARGET", "TILT_WEIGHT", "TILT_TARGET", "DUP_SHARE", "DUP_PX", "HOLD_OUT_EVERY",
           "OUTLIER_FACTOR", "OUTLIER_MIN_PX", "OUTLIER_MAX_FRACTION", "View", "FitResult", "board_points",
           "nominal_pose", "tilt_bin", "describe", "select_views", "screen_views", "fit_fisheye", "pose_from_model",
           "rms",
           "project_points",
           "kannala_brandt8_project", "model_agreement"]

GRID = (8, 6)                 # selection grid (columns, rows)
CELL_TARGET = 20              # corners per selection cell after which a cell adds no gain
TILT_WEIGHT = 15.0            # gain of a view in a tilt bin seen < TILT_TARGET times (in corners)
TILT_TARGET = 4
TILT_EDGES_DEG = (10.0, 25.0, 40.0)
DUP_SHARE = 0.5               # near-duplicate: shares >= this fraction of the candidate's corners ...
DUP_PX = 15.0                 # ... and they moved less than this (median, px)
HOLD_OUT_EVERY = 5
OUTLIER_FACTOR = 3.0
OUTLIER_MIN_PX = 1.0
OUTLIER_MAX_FRACTION = 0.2
_COND_RE = re.compile(r"input array (\d+)")


@dataclass
class View:
    """A candidate or selected view: the frame's corners, their board points and the nominal-model geometry."""

    frame: int
    ids: np.ndarray
    uv: np.ndarray                 # (N, 2) px
    obj: np.ndarray                # (N, 3) board metres
    tilt_deg: float = float("nan")
    tilt_bin: int = -1
    distance_m: float = float("nan")
    cells: np.ndarray = field(default=None, repr=False)     # (GRID[1], GRID[0]) corner counts
    role: str = "candidate"        # candidate | fit | held_out | removed
    removed_reason: str | None = None
    rms_px: float | None = None
    rvec: np.ndarray | None = field(default=None, repr=False)
    tvec: np.ndarray | None = field(default=None, repr=False)

    def summary(self) -> dict:
        d = {"frame": int(self.frame), "corners": int(len(self.ids)),
             "tilt_deg": None if math.isnan(self.tilt_deg) else round(float(self.tilt_deg), 1),
             "distance_m": None if math.isnan(self.distance_m) else round(float(self.distance_m), 3)}
        if self.rms_px is not None:
            d["rms_px"] = round(float(self.rms_px), 4)
        if self.removed_reason:
            d["removed_reason"] = self.removed_reason
        return d


def board_points(ids: np.ndarray) -> np.ndarray:
    """Board coordinates (metres, z = 0) of ChArUco corner ids."""
    from umiproc.lens.board import make_board

    corners = np.asarray(make_board().getChessboardCorners(), np.float64)
    return corners[np.asarray(ids, np.int64)]


# --------------------------------------------------------------------------------------------- poses

def _undistort(uv: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.fisheye.undistortPoints(uv.reshape(-1, 1, 2).astype(np.float64), K, D).reshape(-1, 2)


def project_points(obj, rvec, tvec, K, D) -> np.ndarray:
    """``cv2.fisheye.projectPoints`` of board points ``obj`` (N, 3) at pose (rvec, tvec): (N, 2) pixels."""
    import cv2

    uv, _ = cv2.fisheye.projectPoints(obj.reshape(-1, 1, 3).astype(np.float64), np.asarray(rvec, np.float64),
                                      np.asarray(tvec, np.float64), K, D)
    return uv.reshape(-1, 2)


def _ippe(obj, uv, K, D):
    import cv2

    und = _undistort(uv, K, D)
    ok, rvec, tvec = cv2.solvePnP(obj.astype(np.float64), und, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    return rvec.ravel(), tvec.ravel()


def nominal_pose(obj, uv, K, D):
    """``(rvec, tvec)`` of the board under a model (``None`` if PnP fails): IPPE on the undistorted corners."""
    if len(obj) < 4:
        return None
    return _ippe(obj, uv, K, D)


def pose_from_model(obj, uv, K, D):
    """``(rvec, tvec)`` minimising the pixel reprojection error under the fisheye model (K, D), or None."""
    import cv2
    from scipy.optimize import least_squares

    try:
        init = nominal_pose(obj, uv, K, D)
    except cv2.error:
        return None
    if init is None:
        return None
    x0 = np.concatenate(init)

    def resid(x):
        return (project_points(obj, x[:3], x[3:], K, D) - uv).ravel()

    r = least_squares(resid, x0, method="lm", x_scale="jac", max_nfev=200)
    if not np.all(np.isfinite(r.x)):
        return None
    return r.x[:3], r.x[3:]


def rms(err: np.ndarray) -> float:
    """sqrt(mean of squared 2-D residual lengths) of (N, 2) residuals."""
    err = np.asarray(err, np.float64).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1)))) if len(err) else float("nan")


def tilt_bin(R: np.ndarray, t_centre: np.ndarray) -> tuple[float, int]:
    """``(tilt_deg, bin)``: the angle between the board normal and the line of sight to the board centre, and its
    bin (0: < 10 deg; then 3 tilt bands x 4 directions)."""
    n = R[:, 2]
    los = t_centre / np.linalg.norm(t_centre)
    if np.dot(n, los) < 0:
        n = -n
    tilt = math.degrees(math.acos(float(np.clip(np.dot(n, los), -1.0, 1.0))))
    if tilt < TILT_EDGES_DEG[0]:
        return tilt, 0
    band = 1 + sum(tilt >= e for e in TILT_EDGES_DEG[1:])           # 1, 2, 3
    # direction of the tilt: the normal's component across the line of sight, in the camera's x/y
    a = n - np.dot(n, los) * los
    sector = int(((math.degrees(math.atan2(a[1], a[0])) + 45.0) % 360.0) // 90.0)
    return tilt, 1 + (band - 1) * 4 + sector


# --------------------------------------------------------------------------------------------- selection

def describe(view: View, size: tuple[int, int], K0: np.ndarray, D0: np.ndarray) -> View:
    """Fill a candidate's grid counts and nominal tilt/distance."""
    import cv2

    W, H = size
    gx = np.clip((view.uv[:, 0] + 0.5) * GRID[0] // W, 0, GRID[0] - 1).astype(int)
    gy = np.clip((view.uv[:, 1] + 0.5) * GRID[1] // H, 0, GRID[1] - 1).astype(int)
    cells = np.zeros((GRID[1], GRID[0]), np.int64)
    np.add.at(cells, (gy, gx), 1)
    view.cells = cells
    try:
        pose = nominal_pose(view.obj, view.uv, K0, D0)
    except cv2.error:
        pose = None
    if pose is not None:
        R, _ = cv2.Rodrigues(pose[0])
        centre = R @ view.obj.mean(axis=0) + pose[1]
        view.tilt_deg, view.tilt_bin = tilt_bin(R, centre)
        view.distance_m = float(np.linalg.norm(centre))
    return view


def _near_duplicate(c: View, s: View) -> bool:
    common, ic, is_ = np.intersect1d(c.ids, s.ids, assume_unique=True, return_indices=True)
    if len(common) < DUP_SHARE * len(c.ids):
        return False
    return float(np.median(np.linalg.norm(c.uv[ic] - s.uv[is_], axis=1))) < DUP_PX


def select_views(candidates: list[View], max_views: int) -> list[View]:
    """Greedy diverse selection (module docstring). Returns the selected views in selection order, with ``role``
    set to ``fit`` or ``held_out`` (every HOLD_OUT_EVERY-th)."""
    pool = list(candidates)
    selected: list[View] = []
    cells = np.zeros((GRID[1], GRID[0]), np.int64)
    bins: dict[int, int] = {}
    while pool and len(selected) < max_views:
        best, best_key = None, None
        for c in pool:
            gain = float(np.sum(np.minimum(cells + c.cells, CELL_TARGET) - np.minimum(cells, CELL_TARGET)))
            if c.tilt_bin >= 0 and bins.get(c.tilt_bin, 0) < TILT_TARGET:
                gain += TILT_WEIGHT
            key = (gain, len(c.ids), -c.frame)
            if best_key is None or key > best_key:
                best, best_key = c, key
        selected.append(best)
        cells += best.cells
        if best.tilt_bin >= 0:
            bins[best.tilt_bin] = bins.get(best.tilt_bin, 0) + 1
        pool = [c for c in pool if c is not best and not _near_duplicate(c, best)]
    for k, v in enumerate(selected, start=1):
        v.role = "held_out" if k % HOLD_OUT_EVERY == 0 else "fit"
    return selected


# --------------------------------------------------------------------------------------------- fit

@dataclass
class FitResult:
    K: np.ndarray
    D: np.ndarray
    cv_rms_px: float              # what cv2.fisheye.calibrate returned
    rms_px: float                 # recomputed over all fit corners
    views: list                   # the fit views (role "fit"), with rvec/tvec/rms_px
    removed: list                 # views removed by the fit (role "removed", with removed_reason)
    attempts: int


def _calibrate(views: list[View], size, K0, D0):
    import cv2

    obj = [v.obj.reshape(-1, 1, 3).astype(np.float64) for v in views]
    img = [v.uv.reshape(-1, 1, 2).astype(np.float64) for v in views]
    flags = (cv2.fisheye.CALIB_USE_INTRINSIC_GUESS | cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
             | cv2.fisheye.CALIB_FIX_SKEW | cv2.fisheye.CALIB_CHECK_COND)
    crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-12)
    ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(obj, img, tuple(int(s) for s in size), K0.copy(),
                                                    D0.reshape(4, 1).copy(), flags=flags, criteria=crit)
    return float(ret), K, D.ravel(), rvecs, tvecs


def screen_views(views: list[View], size, K0: np.ndarray, D0: np.ndarray) -> tuple[list[View], list[View]]:
    """``(kept, rejected)``: each view alone through ``cv2.fisheye.calibrate`` with every intrinsic fixed at
    (K0, D0) and ``CALIB_CHECK_COND``, i.e. OpenCV's own pose initialisation and condition check (module
    docstring). Rejected views get role ``removed`` and a reason."""
    import cv2

    f = cv2.fisheye
    flags = (f.CALIB_USE_INTRINSIC_GUESS | f.CALIB_RECOMPUTE_EXTRINSIC | f.CALIB_CHECK_COND | f.CALIB_FIX_SKEW
             | f.CALIB_FIX_FOCAL_LENGTH | f.CALIB_FIX_PRINCIPAL_POINT | f.CALIB_FIX_K1 | f.CALIB_FIX_K2
             | f.CALIB_FIX_K3 | f.CALIB_FIX_K4)
    crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 20, 1e-9)
    kept, rejected = [], []
    for v in views:
        try:
            f.calibrate([v.obj.reshape(-1, 1, 3).astype(np.float64)], [v.uv.reshape(-1, 1, 2).astype(np.float64)],
                        tuple(int(s) for s in size), K0.copy(), D0.reshape(4, 1).astype(np.float64).copy(),
                        flags=flags, criteria=crit)
            kept.append(v)
        except cv2.error as e:
            v.role = "removed"
            msg = str(e).strip().splitlines()[-1]
            v.removed_reason = ("ill-conditioned pose (CALIB_CHECK_COND)" if "CALIB_CHECK_COND" in msg
                                else "degenerate view (OpenCV: " + msg.split("error: ")[-1] + ")")
            rejected.append(v)
    return kept, rejected


def fit_fisheye(views: list[View], size, K0: np.ndarray, D0: np.ndarray, min_views: int = 3) -> FitResult:
    """Fit K, k1..k4 to ``views`` (module docstring: ill-conditioned and outlier views are removed). Raises
    RuntimeError when fewer than ``min_views`` remain or OpenCV fails for another reason."""
    import cv2

    active, removed = screen_views(views, size, K0, D0)
    max_outliers = int(math.floor(OUTLIER_MAX_FRACTION * len(views)))
    n_outliers, attempts = 0, 0
    while True:
        if len(active) < min_views:
            raise RuntimeError(f"only {len(active)} views left to fit (at least {min_views} needed)")
        attempts += 1
        try:
            cv_rms, K, D, rvecs, tvecs = _calibrate(active, size, K0, D0)
        except cv2.error as e:
            m = _COND_RE.search(str(e))
            if "CALIB_CHECK_COND" in str(e) and m and int(m.group(1)) < len(active):
                v = active.pop(int(m.group(1)))
                v.role, v.removed_reason = "removed", "ill-conditioned pose (CALIB_CHECK_COND)"
                removed.append(v)
                continue
            raise RuntimeError(f"cv2.fisheye.calibrate failed: {str(e).strip().splitlines()[-1]}") from None
        if not (np.all(np.isfinite(K)) and np.all(np.isfinite(D))):
            raise RuntimeError("cv2.fisheye.calibrate returned non-finite parameters")
        for v, r, t in zip(active, rvecs, tvecs):
            v.rvec, v.tvec = np.asarray(r, np.float64).ravel(), np.asarray(t, np.float64).ravel()
            v.rms_px = rms(project_points(v.obj, v.rvec, v.tvec, K, D) - v.uv)
        per_view = np.array([v.rms_px for v in active])
        worst = int(np.argmax(per_view))
        limit = max(OUTLIER_FACTOR * float(np.median(per_view)), OUTLIER_MIN_PX)
        if per_view[worst] > limit and n_outliers < max_outliers and len(active) > min_views:
            v = active.pop(worst)
            v.role = "removed"
            v.removed_reason = f"outlier view: RMS {per_view[worst]:.2f} px > {limit:.2f} px"
            removed.append(v)
            n_outliers += 1
            continue
        all_err = np.concatenate([project_points(v.obj, v.rvec, v.tvec, K, D) - v.uv for v in active])
        return FitResult(K=K, D=D, cv_rms_px=cv_rms, rms_px=rms(all_err), views=active, removed=removed,
                         attempts=attempts)


# --------------------------------------------------------------------------------------------- model equivalence

def kannala_brandt8_project(params, P: np.ndarray) -> np.ndarray:
    """ORB-SLAM3's ``KannalaBrandt8::project`` (src/CameraModels/KannalaBrandt8.cc), transcribed:
    ``params = [fx, fy, cx, cy, k0, k1, k2, k3]``, ``P`` (N, 3) camera points -> (N, 2) pixels."""
    fx, fy, cx, cy, k0, k1, k2, k3 = (float(p) for p in params)
    P = np.asarray(P, np.float64).reshape(-1, 3)
    x2_plus_y2 = P[:, 0] ** 2 + P[:, 1] ** 2
    theta = np.arctan2(np.sqrt(x2_plus_y2), P[:, 2])
    psi = np.arctan2(P[:, 1], P[:, 0])
    theta2 = theta * theta
    theta3 = theta * theta2
    theta5 = theta3 * theta2
    theta7 = theta5 * theta2
    theta9 = theta7 * theta2
    r = theta + k0 * theta3 + k1 * theta5 + k2 * theta7 + k3 * theta9
    return np.stack([fx * r * np.cos(psi) + cx, fy * r * np.sin(psi) + cy], axis=1)


def model_agreement(K: np.ndarray, D: np.ndarray, size: tuple[int, int], max_theta_deg: float = 85.0,
                    n: int = 4000, seed: int = 0) -> dict:
    """Project ``n`` random rays (field angle up to ``max_theta_deg``, inside the model's valid range) with
    ``cv2.fisheye.projectPoints``, the rig's ``FisheyeEquidistant`` and ORB-SLAM3's ``KannalaBrandt8`` formula, and
    report the largest pixel differences (all three should agree to rounding, < 1e-6 px)."""
    from umiproc.dp.geometry.camera_model import FisheyeEquidistant

    m = FisheyeEquidistant(int(size[0]), int(size[1]), K[0, 0], K[1, 1], K[0, 2], K[1, 2], *D)
    hi = min(math.radians(max_theta_deg), m.theta_valid)
    rng = np.random.default_rng(seed)
    th = rng.uniform(0.0, hi, n)
    ph = rng.uniform(-math.pi, math.pi, n)
    P = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], 1) * rng.uniform(0.2, 5.0, (n, 1))
    uv_cv = project_points(P, np.zeros(3), np.zeros(3), K, D)
    uv_rig, ok = m.project(P)
    uv_kb8 = kannala_brandt8_project([K[0, 0], K[1, 1], K[0, 2], K[1, 2], *D], P)
    d1 = float(np.abs(uv_cv[ok] - uv_rig[ok]).max()) if ok.any() else float("inf")
    d2 = float(np.abs(uv_cv - uv_kb8).max())
    return {"rays": n, "max_theta_deg": round(math.degrees(hi), 2), "valid_in_rig_model": int(ok.sum()),
            "max_diff_cv2_vs_rig_px": d1, "max_diff_cv2_vs_orbslam3_px": d2,
            "agree": bool(ok.all() and d1 < 1e-6 and d2 < 1e-6)}
