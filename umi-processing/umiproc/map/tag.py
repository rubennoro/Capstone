"""The table tag in the map (check M5 and the table frame step 3 uses), replicating UMI's method.

UMI (``umi_repo`` = github.com/real-stanford/universal_manipulation_interface):

1. **Detection per frame** (``umi/common/cv_util.py:161-192`` ``detect_localize_aruco_tags``, called by
   ``scripts/detect_aruco.py`` on every video frame): ``cv2.aruco.detectMarkers`` with the default
   ``DetectorParameters`` and ``CORNER_REFINE_SUBPIX``; the corners are undistorted with the **fisheye** lens model
   (``cv2.fisheye.undistortPoints(corners, K, D, P=K)``) and the pose comes from ``estimatePoseSingleMarkers(
   undistorted, size, K, zeros)``, which in UMI's OpenCV 4.7 is ``solvePnP`` (``SOLVEPNP_ITERATIVE``) on the marker's
   corners centred on the tag (``ARUCO_CCW_CENTER``: x right, y up, z out of the tag towards the camera). OpenCV 4.12
   has no ``estimatePoseSingleMarkers``, so :func:`detect_table_tag` calls that ``solvePnP`` itself. Lens: the
   effective ultrawide model (the lens file's when there is one, else the rig's nominal; ``SensorValues.rig``) at the
   native 1280x960, as the SLAM settings. Two deliberate differences: the frame is converted to grey as **RGB**
   (UMI hands an RGB frame to ``detectMarkers``, which converts it as BGR and so swaps the red and blue weights; the
   copied width stage makes the same fix), and only the table tag's id is kept (UMI keeps every id of its size map).
   UMI masks only the GoPro's mirrors before detecting (``finger=False``): our rig has none, so nothing is masked.
2. **Sightings** (``scripts/calibrate_slam_tag.py``, run by ``scripts_slam_pipeline/05_run_calibrations.py:40-46``
   with ``--keyframe_only`` on ``camera_trajectory.csv``, the mapping clip re-localized against the saved map):
   rows with ``is_lost`` false **and** ``is_keyframe`` (lines 39-41); the tag at 0.3-2 m from the camera (74-75); the
   tag centre within ``0.6 * H/2`` of the image centre (78-83, UMI's 2704x2028 there, our 1280x960 here: within
   288 px of (640, 480)); ``T_slam_tag = T_slam_cam @ T_cam_tag`` (86). UMI matches CSV rows to video frames by the
   nearest time; here row k of the trajectory is line k of times.txt (umiproc.slam.trajectory_csv), whose time is
   one ultrawide frame's exactly, so the frame is found by its time.
3. **Aggregation** (lines 92-101): the geometric median of the tag positions; keep the sightings whose distance to
   it is **below** the 0.9 quantile of those distances (strictly, as UMI); their per-axis std and mean; the table
   pose is the kept sighting whose position is nearest that mean (its rotation included). UMI writes it as
   ``tx_slam_tag.json`` ``{"tx_slam_tag": 4x4}`` and later uses its inverse as the base frame of every pose
   (``scripts_slam_pipeline/06_generate_dataset_plan.py:111-119``). Added for check M5 (not UMI's): the median
   normal of the kept sightings (the geometric median of their z axes, renormalised), which M5 judges instead of the
   one chosen sighting's.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["TagDetection", "TableTagDetector", "detect_table_tag", "marker_object_points", "geometric_median",
           "Sighting", "SightingStats", "collect_sightings", "aggregate_sightings", "MIN_DISTANCE_M",
           "MAX_DISTANCE_M", "MAX_CENTRE_FRACTION", "OUTLIER_QUANTILE"]

#: calibrate_slam_tag.py:75: sightings with the tag nearer than 0.3 m or farther than 2 m are dropped.
MIN_DISTANCE_M, MAX_DISTANCE_M = 0.3, 2.0
#: calibrate_slam_tag.py:81-83: the tag centre must be within this fraction of H/2 of the image centre.
MAX_CENTRE_FRACTION = 0.6
#: calibrate_slam_tag.py:95-96: sightings farther from the geometric median than this quantile are dropped.
OUTLIER_QUANTILE = 0.9


@dataclass(frozen=True)
class TagDetection:
    """The table tag in one frame: ``rvec``/``tvec`` (T_cam_tag, OpenCV), ``corners`` (4, 2) in the original
    (distorted) image, in ArUco order (top-left, top-right, bottom-right, bottom-left)."""

    frame_index: int
    rvec: np.ndarray
    tvec: np.ndarray
    corners: np.ndarray

    @property
    def T_cam_tag(self) -> np.ndarray:
        """``pose_to_mat(concat(tvec, rvec))`` (umi/common/pose_util.py)."""
        from scipy.spatial.transform import Rotation

        T = np.eye(4)
        T[:3, :3] = Rotation.from_rotvec(np.asarray(self.rvec, float)).as_matrix()
        T[:3, 3] = np.asarray(self.tvec, float)
        return T

    @property
    def centre_px(self) -> np.ndarray:
        return np.asarray(self.corners, float).mean(axis=0)

    def to_json(self) -> dict:
        return {"frame_index": int(self.frame_index), "rvec": [float(v) for v in self.rvec],
                "tvec": [float(v) for v in self.tvec],
                "corners": [[float(x), float(y)] for x, y in np.asarray(self.corners, float)]}

    @classmethod
    def from_json(cls, d: dict) -> "TagDetection":
        return cls(int(d["frame_index"]), np.asarray(d["rvec"], float), np.asarray(d["tvec"], float),
                   np.asarray(d["corners"], float))


def marker_object_points(size_m: float) -> np.ndarray:
    """The 4 corners of a marker of side ``size_m`` in its own frame, ArUco order, centred (OpenCV's
    ``ARUCO_CCW_CENTER``, the pattern ``estimatePoseSingleMarkers`` uses)."""
    h = float(size_m) / 2.0
    return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]], dtype=np.float64)


def _camera(model) -> tuple[np.ndarray, np.ndarray]:
    if model.type != "fisheye-equidistant":
        raise ValueError(f"the table tag needs the ultrawide's fisheye-equidistant model, got {model.type}")
    fx, fy, cx, cy = (float(v) for v in model.intrinsics)
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    D = np.asarray([float(v) for v in model.distortion], dtype=np.float64).reshape(4, 1)
    return K, D


def detect_table_tag(image, frame_index: int, detector, tag_id: int, size_m: float, K: np.ndarray,
                     D: np.ndarray) -> TagDetection | None:
    """The table tag in one frame (``image``: RGB (H, W, 3) or grey uint8), or None (method: module docstring)."""
    import cv2

    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    corners, ids, _ = detector.detectMarkers(img)
    if ids is None or len(corners) == 0:
        return None
    for this_id, this_corners in zip(ids.reshape(-1), corners):
        if int(this_id) != int(tag_id):
            continue
        c = np.asarray(this_corners, dtype=np.float64).reshape(-1, 1, 2)
        und = cv2.fisheye.undistortPoints(c, K, D, P=K)
        ok, rvec, tvec = cv2.solvePnP(marker_object_points(size_m), und.reshape(-1, 2), K, np.zeros((5, 1)),
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        return TagDetection(int(frame_index), rvec.reshape(3), tvec.reshape(3), c.reshape(4, 2))
    return None


class TableTagDetector:
    """Detects the SLAM config's table tag in frames as they stream past (``observe``), with the effective lens.

    ``detections`` maps frame index to :class:`TagDetection`; ``n_frames`` counts every frame observed."""

    def __init__(self, sensors):
        import cv2

        tag = sensors.slam.table_tag
        self.tag_id, self.size_m, self.dictionary = int(tag.id), float(tag.size_m), tag.dictionary
        self.K, self.D = _camera(sensors.ultrawide.model)
        self.resolution = tuple(int(v) for v in sensors.ultrawide.resolution)
        param = cv2.aruco.DetectorParameters()
        param.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, tag.dictionary)),
                                                 param)
        self.detections: dict[int, TagDetection] = {}
        self.n_frames = 0

    def observe(self, frame_index: int, image) -> TagDetection | None:
        self.n_frames += 1
        d = detect_table_tag(image, frame_index, self._detector, self.tag_id, self.size_m, self.K, self.D)
        if d is not None:
            self.detections[int(frame_index)] = d
        return d

    def to_json(self) -> dict:
        return {"tag": {"dictionary": self.dictionary, "id": self.tag_id, "size_m": self.size_m},
                "lens": {"K": self.K.tolist(), "D": self.D.reshape(-1).tolist()},
                "resolution": list(self.resolution), "frames_observed": self.n_frames,
                "detections": [self.detections[k].to_json() for k in sorted(self.detections)]}


# --------------------------------------------------------------------------------------------- aggregation

def geometric_median(X: np.ndarray, tol: float = 1e-8, max_iter: int = 1000) -> np.ndarray:
    """Weiszfeld's algorithm for the point minimising the sum of Euclidean distances to the rows of ``X`` (what
    ``skfda.exploratory.stats.geometric_median`` computes for UMI): start at the mean, re-weight by 1/distance until
    the estimate stops moving (relative tolerance ``tol``). A point that coincides with the estimate gets a tiny
    distance instead of a division by zero."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or len(X) == 0:
        raise ValueError("geometric_median needs a non-empty (N, d) array")
    y = X.mean(axis=0)
    for _ in range(max_iter):
        d = np.linalg.norm(X - y, axis=1)
        w = 1.0 / np.maximum(d, 1e-12)
        y_new = (w[:, None] * X).sum(axis=0) / w.sum()
        if np.allclose(y_new, y, rtol=tol, atol=1e-12):
            return y_new
        y = y_new
    return y


@dataclass(frozen=True)
class Sighting:
    """One kept sighting: trajectory row ``row`` (= times.txt line), video frame ``frame_index``, ``T_world_tag``."""

    row: int
    frame_index: int
    T_world_tag: np.ndarray
    distance_m: float
    centre_fraction: float
    state: int


@dataclass
class SightingStats:
    """What :func:`collect_sightings` saw: the counts at each filter, and the kept :class:`Sighting` s."""

    keyframes: int = 0                       # posed keyframe rows (UMI's ~is_lost & is_keyframe)
    keyframes_with_tag: int = 0              # ... whose frame has the table tag
    rejected_distance: int = 0               # tag nearer than 0.3 m or farther than 2 m
    rejected_centre: int = 0                 # tag centre beyond 0.6 * H/2 of the image centre
    sightings: list[Sighting] = field(default_factory=list)
    keyframes_not_tracked: int = 0           # kept sightings whose keyframe was not visually tracked (state != OK)

    def to_json(self) -> dict:
        return {"keyframes": self.keyframes, "keyframes_with_tag": self.keyframes_with_tag,
                "rejected_distance": self.rejected_distance, "rejected_centre": self.rejected_centre,
                "kept": len(self.sightings), "kept_not_visually_tracked": self.keyframes_not_tracked}


def collect_sightings(traj, frame_index_of_row: np.ndarray, detections: dict, resolution) -> SightingStats:
    """UMI's filters (module docstring, step 2) on a re-localized :class:`~umiproc.slam.trajectory_csv.ForkTrajectory`.

    ``frame_index_of_row[k]``: the video frame of trajectory row k; ``detections``: frame index -> TagDetection;
    ``resolution``: (W, H) of the image the corners are in."""
    w, h = (float(v) for v in resolution)
    centre = np.array([w, h]) / 2.0
    st = SightingStats()
    rows = np.flatnonzero(np.asarray(traj.posed, bool) & np.asarray(traj.is_keyframe, bool))
    st.keyframes = int(len(rows))
    for k in rows:
        det = detections.get(int(frame_index_of_row[k]))
        if det is None:
            continue
        st.keyframes_with_tag += 1
        T_cam_tag = det.T_cam_tag
        dist = float(np.linalg.norm(T_cam_tag[:3, 3]))
        if dist < MIN_DISTANCE_M or dist > MAX_DISTANCE_M:
            st.rejected_distance += 1
            continue
        frac = float(np.linalg.norm(det.centre_px - centre) / centre[1])
        if frac > MAX_CENTRE_FRACTION:
            st.rejected_centre += 1
            continue
        state = int(traj.state[k])
        if state != 2:
            st.keyframes_not_tracked += 1
        st.sightings.append(Sighting(int(k), int(det.frame_index), traj.T_W_cam[k] @ T_cam_tag, dist, frac, state))
    return st


def aggregate_sightings(T_world_tag: np.ndarray) -> dict:
    """UMI's aggregation (module docstring, step 3) of ``T_world_tag`` (N, 4, 4), N >= 1.

    Returns ``{"T_world_tag": 4x4 (the kept sighting nearest the mean), "index": its index in the input,
    "kept": bool mask, "median": geometric median, "threshold_m": the 0.9 quantile distance, "std_m": per-axis std
    (ddof 0) of the kept positions, "mean": their mean, "normal_median": the geometric median of the kept sightings'
    z axes, renormalised (unit), "normal_chosen": the chosen sighting's z axis}``. With every distance equal (e.g.
    N = 1) nothing is strictly below the quantile; then every sighting is kept (UMI would crash on the empty set)."""
    T = np.asarray(T_world_tag, dtype=np.float64)
    if T.ndim != 3 or T.shape[1:] != (4, 4) or len(T) == 0:
        raise ValueError("aggregate_sightings needs a non-empty (N, 4, 4) array")
    pos = T[:, :3, 3]
    median = geometric_median(pos)
    dists = np.linalg.norm(pos - median, axis=1)
    threshold = float(np.quantile(dists, OUTLIER_QUANTILE))
    kept = dists < threshold
    if not kept.any():
        kept = np.ones(len(T), bool)
    std = pos[kept].std(axis=0)
    mean = pos[kept].mean(axis=0)
    idx_kept = np.flatnonzero(kept)
    nn = int(idx_kept[int(np.argmin(np.linalg.norm(pos[kept] - mean, axis=1)))])
    z = T[kept, :3, 2]
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    nz = geometric_median(z)
    nz = nz / max(float(np.linalg.norm(nz)), 1e-12)
    return {"T_world_tag": T[nn].copy(), "index": nn, "kept": kept, "median": median, "threshold_m": threshold,
            "std_m": std, "mean": mean, "normal_median": nz, "normal_chosen": T[nn, :3, 2].copy()}
