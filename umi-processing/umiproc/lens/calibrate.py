"""``umiproc lens-calibrate``: fit the ultrawide (IMX219 fisheye) lens model from a board recording, check it, and
write it (DIRECTIONS.md §1). Optional: without it, the processing uses the rig's nominal model.

Steps:

1. **Input.** ``<--source>/<EPISODE_DIR>`` must pass :func:`umiproc.dp.pi_adapter.inspect_source` with hash
   verification (``COMPLETE``, ``MANIFEST.sha256`` matching every file, ``meta.json`` valid). The camera whose
   ``model`` is ``imx219`` in ``meta.json`` must be 1280 x 960; its MP4 is decoded (grey). The rig file gives the
   nominal model the fit starts from; the SLAM config gives the gripper mask. Both are the data-root copies the
   processing reads (``<root>/rigs/<rig>.yaml``, ``<root>/rigs/<rig>.slam.yaml``, checked against the repo by
   :func:`umiproc.sensors.ensure_rig_files`); before the first ``map`` copies them there, the repo's
   ``configs/rigs/<rig>.yaml`` and ``configs/slam/<rig>.yaml`` (nothing is copied here).
2. **Corners** (:mod:`umiproc.lens.detect`): ChArUco corners in every frame, minus those inside the gripper mask;
   frames with >= 12 corners; then only still frames (corners moved <= 1 px against both neighbouring frames).
3. **Views**: candidates OpenCV cannot initialise a pose for, or whose pose is ill-conditioned, are dropped
   (:func:`umiproc.lens.fit.screen_views`); then (:func:`umiproc.lens.fit.select_views`) up to ``--max-views``
   diverse views, chosen greedily for
   coverage of the image (8 x 6 grid, up to 20 corners per cell) and of board tilts (13 tilt bins under the nominal
   model), never two views of the same pose; every 5th selected view is held out.
4. **Fit** (:func:`umiproc.lens.fit.fit_fisheye`): ``cv2.fisheye.calibrate`` (Kannala-Brandt k1..k4 = ORB-SLAM3's
   KannalaBrandt8 = the rig's ``fisheye-equidistant``, re-verified numerically on every run) from the nominal K;
   ill-conditioned views (``CALIB_CHECK_COND``) and outlier views are removed and the fit repeated.
5. **Checks**, each reported with its value and pass/fail (limits are the constants below); every check but (f) is
   a hard limit:

   ====  ==============================================================================================
   (a)   >= 25 fit views
   (b)   fit RMS reprojection error <= 0.5 px
   (c)   held-out RMS <= 0.6 px (held-out board poses by PnP with the fitted model only)
   (d)   radial coverage: the 10th-farthest fit corner from the principal point is at >= 85 % of the largest
         radius any pixel outside the mask has (corners near the image's outer edge, plural)
   (e)   angular coverage: in a 3 x 3 grid, every cell that is >= 50 % outside the mask holds >= 20 fit corners
   (f)   fx and fy within +-10 % of the rig's nominal: a **warning** only (the nominal is unmeasured): reported
         and printed, but it never blocks ``lens.yaml`` or ``--apply``
   (g)   principal point within 40 px of the image centre ((W-1)/2, (H-1)/2)
   (h)   field of view: twice the field angle of each visible top image corner (outer pixel edge) is within
         140-180 deg (a ~160 deg diagonal lens)
   ====  ==============================================================================================

6. **Outputs** in ``<root>/lens/<pi_name>/`` (``<pi_name>`` = the converted name ``pi-YYYY-MM-DD_episode_NNNNNN``
   of :func:`umiproc.dp.pi_adapter.converted_name`, so a reused Pi episode number never overwrites an earlier
   result): ``report.json`` (every check, value and limit, the views used, held
   out and removed, frame statistics, settings) and ``coverage.png`` (a frame with every used corner, the mask
   outline, the 3 x 3 grid with its counts and the 85 % radius) after every run that got as far as the corners;
   ``lens.yaml`` (:mod:`umiproc.lens.lensfile`) only when every check passed (a stale one is removed otherwise).
   "Every check passed" means every hard check: a warning-only check ((f)) is printed as ``WARN`` and listed in
   ``report.json``'s ``warnings``, but never blocks.
7. ``--apply`` (only when every check passed): copy ``lens.yaml`` to ``<root>/rigs/<rig>.lens.yaml``, renaming an
   existing different one to ``.bak-<UTC timestamp>``. Without ``--apply`` the exact command that applies it is
   printed (the same command plus ``--apply``: it repeats the calibration, which is deterministic, and applies it).

Exit codes: 0 = every check passed; 1 = a check failed or the input is unusable (the message says which check and
what to record differently).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import shlex
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from umiproc.lens import detect as lens_detect
from umiproc.lens import fit as lens_fit

__all__ = ["DEFAULT_SOURCE", "DEFAULT_RIG", "DEFAULT_MAX_VIEWS", "SENSOR", "RESOLUTION", "MIN_FIT_VIEWS",
           "MAX_FIT_RMS_PX", "MAX_HELD_OUT_RMS_PX", "RADIAL_FRACTION", "RADIAL_MIN_CORNERS", "CELL_MIN_VISIBLE",
           "CELL_MIN_CORNERS", "FOCAL_TOLERANCE", "PRINCIPAL_POINT_MAX_PX", "FOV_RANGE_DEG", "REPORT_SCHEMA",
           "WARNING_CHECKS", "InputError", "Check", "Calibration", "calibrate_recording", "write_outputs",
           "add_arguments", "run"]

DEFAULT_SOURCE = Path("/data/incoming")
DEFAULT_RIG = "rig_pi_v3"
DEFAULT_MAX_VIEWS = 60
SENSOR = "imx219"
RESOLUTION = (1280, 960)
MIN_FIT_VIEWS = 25                  # (a)
MAX_FIT_RMS_PX = 0.5                # (b)
MAX_HELD_OUT_RMS_PX = 0.6           # (c)
RADIAL_FRACTION = 0.85              # (d)
RADIAL_MIN_CORNERS = 10             # (d): the radius reached by this many corners counts
CELL_GRID = (3, 3)                  # (e) columns, rows
CELL_MIN_VISIBLE = 0.5              # (e) cells at least this much outside the mask are checked
CELL_MIN_CORNERS = 20               # (e)
FOCAL_TOLERANCE = 0.10              # (f), a warning only
PRINCIPAL_POINT_MAX_PX = 40.0       # (g)
FOV_RANGE_DEG = (140.0, 180.0)      # (h)
MIN_VIEWS_TO_FIT = 5                # fewer selected fit views: no fit is attempted (check (a) fails)
REPORT_SCHEMA = "umiproc.lens_report.v1"
#: Checks that only warn: computed, reported and printed, but they never block lens.yaml or --apply. (f): the
#: nominal focal length it compares with is the rig file's unmeasured value.
WARNING_CHECKS = frozenset({"f"})
CELL_NAMES = (("top-left", "top", "top-right"), ("left", "centre", "right"),
              ("bottom-left", "bottom", "bottom-right"))

CHECK_NAMES = {
    "a": "fit views", "b": "fit RMS reprojection error", "c": "held-out RMS reprojection error",
    "d": "radial coverage", "e": "angular coverage (3 x 3 grid)", "f": "focal length vs nominal",
    "g": "principal point vs image centre", "h": "field of view at the top corners",
}
#: The check values as named in lens.yaml's ``metrics``.
METRIC_KEYS = {"a": "fit_views", "b": "fit_rms_px", "c": "held_out_rms_px", "d": "radial_coverage",
               "e": "min_cell_corners", "f": "focal_deviation", "g": "principal_point_offset_px",
               "h": "fov_top_corners_deg"}
LIGHT_TIPS = ("even, bright light with no glare on the board (matte paper, no lamp reflected in it), so the exposure "
              "stays short")
#: Each check's limit, as printed.
LIMITS = {"a": f">= {MIN_FIT_VIEWS}", "b": f"<= {MAX_FIT_RMS_PX} px", "c": f"<= {MAX_HELD_OUT_RMS_PX} px",
          "d": f">= {RADIAL_FRACTION:.0%} of the largest visible radius",
          "e": f">= {CELL_MIN_CORNERS} corners per cell >= {CELL_MIN_VISIBLE:.0%} visible",
          "f": f"within +-{FOCAL_TOLERANCE:.0%}", "g": f"<= {PRINCIPAL_POINT_MAX_PX:.0f} px",
          "h": f"{FOV_RANGE_DEG[0]:.0f}-{FOV_RANGE_DEG[1]:.0f} deg"}


class InputError(Exception):
    """The recording or the configuration cannot be used (the message says why)."""


@dataclass
class Check:
    id: str
    value: object
    limit: str
    passed: bool
    detail: str = ""
    hint: str = ""
    evaluated: bool = True

    def __post_init__(self):
        if isinstance(self.value, np.generic):           # plain numbers for JSON / YAML
            self.value = self.value.item()

    @property
    def name(self) -> str:
        return CHECK_NAMES[self.id]

    @property
    def blocking(self) -> bool:
        """False for a warning-only check (:data:`WARNING_CHECKS`): it never makes the calibration fail."""
        return self.id not in WARNING_CHECKS

    @property
    def warned(self) -> bool:
        """A warning-only check that was evaluated and is outside its limit."""
        return not self.blocking and self.evaluated and not self.passed

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "value": self.value, "limit": self.limit,
                "passed": bool(self.passed), "evaluated": bool(self.evaluated), "blocking": self.blocking,
                "detail": self.detail, "hint": "" if self.passed else self.hint}

    def line(self) -> str:
        bad = "FAIL" if self.blocking else "WARN"
        status = "PASS" if self.passed else (bad if self.evaluated else f"{bad} (not evaluated)")
        val = self.detail or str(self.value)
        return f"{status:<4} ({self.id}) {self.name}: {val} (limit {self.limit})"


@dataclass
class Calibration:
    """Everything one run computed (``report`` is what report.json holds)."""

    report: dict
    passed: bool
    size: tuple[int, int]
    keep: np.ndarray
    polygons: tuple
    backdrop: np.ndarray | None
    views: list = field(default_factory=list)        # selected views (fit / held_out / removed roles)
    K: np.ndarray | None = None
    D: np.ndarray | None = None
    checks: list = field(default_factory=list)


# --------------------------------------------------------------------------------------------- inputs

def _episode_path(episode_dir: str, source: str | os.PathLike) -> Path:
    from umiproc.dp.paths import validate_episode_dir_name

    p = Path(source).expanduser() / Path(episode_dir).expanduser()
    p = Path(os.path.normpath(p.absolute()))
    try:
        validate_episode_dir_name(p.name)
    except ValueError as e:
        raise InputError(str(e)) from None
    return p


def _load_configs(root: Path, rig_id: str, log=print):
    """The rig file and SLAM config of ``rig_id``: the data-root copies when they exist (checked against the repo,
    differences printed), else the repo's files (``lens-calibrate`` copies nothing; the first ``map`` does)."""
    from umiproc.dp.config import ConfigError, load_rig
    from umiproc.sensors import RigFilesError, ensure_rig_files
    from umiproc.slam_config import SlamConfigError, load_slam_config

    try:
        files = ensure_rig_files(root, rig_id, log=log, copy_missing=False)
    except RigFilesError as e:
        raise InputError(f"--rig {rig_id}: {e}") from None
    rig_file, slam_file = (st.path if st.path.is_file() else st.repo for st in (files.rig, files.slam))
    try:
        rig = load_rig(rig_file)
        slam = load_slam_config(slam_file)
    except (ConfigError, SlamConfigError) as e:
        raise InputError(str(e)) from None
    cam = rig.cameras["ultrawide"]
    if cam.sensor.lower() != SENSOR or tuple(cam.resolution) != RESOLUTION:
        raise InputError(f"{rig_file}: cameras.ultrawide must be the {SENSOR} at {RESOLUTION[0]}x{RESOLUTION[1]}, got "
                         f"{cam.sensor} at {cam.resolution[0]}x{cam.resolution[1]}")
    if slam.rig != rig.rig_id or tuple(slam.mask.resolution) != tuple(cam.resolution):
        raise InputError(f"{slam_file}: its rig ({slam.rig}) and mask resolution ({slam.mask.resolution}) must match "
                         f"{rig.rig_id} at {cam.resolution}")
    return rig, slam, rig_file, slam_file


def _camera_entry(src) -> dict:
    cams = (src.meta or {}).get("cameras")
    if not isinstance(cams, list):
        raise InputError(f"{src.path}/meta.json: no cameras list")
    hits = [c for c in cams if isinstance(c, dict) and str(c.get("model", "")).lower() == SENSOR]
    if len(hits) != 1:
        raise InputError(f"{src.path}/meta.json: expected exactly one camera with model {SENSOR!r}, found {len(hits)}")
    c = hits[0]
    size = c.get("size")
    if not isinstance(size, list) or [int(v) for v in size if isinstance(v, int)] != list(RESOLUTION):
        raise InputError(f"{src.path}/meta.json: the {SENSOR} records at {size}, but the lens calibration needs "
                         f"{RESOLUTION[0]}x{RESOLUTION[1]} (the camera mode the processing uses)")
    f = c.get("file")
    if not isinstance(f, str) or not f.endswith(".mp4") or "/" in f or not (src.path / f).is_file():
        raise InputError(f"{src.path}/meta.json: the {SENSOR}'s file {f!r} is not an .mp4 in the recording")
    return c


# --------------------------------------------------------------------------------------------- checks

def _check_a(n_fit: int, stats: dict) -> Check:
    ok = n_fit >= MIN_FIT_VIEWS
    hint = ("Record more distinct poses (about 60 s): move slowly and hold still about 1 s at each pose "
            "(only the pauses are used), keep most of the board in view at 20-45 cm. ")
    if stats["with_board"] == 0:
        hint += ("The board was not seen in any frame: is this the board recording, and is the board the one "
                 "`umiproc lens-board` prints (DICT_5X5_100)? ")
    elif stats["enough_corners"] < 3 * MIN_FIT_VIEWS:
        hint += (f"Only {stats['enough_corners']} of {stats['decoded']} frames had >= {lens_detect.MIN_CORNERS} board "
                 "corners outside the gripper mask: bring the board closer, turn it to face the lens, keep it out of "
                 f"the gripper's area, and check the focus and the light: {LIGHT_TIPS}. ")
    elif stats["still"] < 2 * MIN_FIT_VIEWS:
        hint += (f"Only {stats['still']} of {stats['enough_corners']} frames with the board were still: pause longer "
                 "at each pose and move more slowly. ")
    else:
        hint += ("Many pauses repeated the same pose: change position in the image, distance and tilt between "
                 "pauses. ")
    return Check("a", n_fit, f">= {MIN_FIT_VIEWS}", ok, f"{n_fit} views", hint.strip())


def _check_b(rms_px: float | None) -> Check:
    hint = ("The corners do not fit one lens model well: glue the board flat onto something rigid, move more slowly "
            "and pause longer (motion blur), lock the lens focus, and use " + LIGHT_TIPS + ".")
    if rms_px is None:
        return Check("b", None, f"<= {MAX_FIT_RMS_PX} px", False, "no fit", hint, evaluated=False)
    return Check("b", round(rms_px, 4), f"<= {MAX_FIT_RMS_PX} px", rms_px <= MAX_FIT_RMS_PX, f"{rms_px:.3f} px", hint)


def _check_c(rms_px: float | None, n: int) -> Check:
    hint = ("The model does not predict views it was not fitted to: record more varied poses (all parts of the "
            "image, tilts up to 45 deg, near and far), with the same care as for (b).")
    if rms_px is None or not math.isfinite(rms_px):
        return Check("c", None, f"<= {MAX_HELD_OUT_RMS_PX} px", False,
                     "no held-out view" if n == 0 else "no held-out pose could be solved", hint, evaluated=False)
    return Check("c", round(rms_px, 4), f"<= {MAX_HELD_OUT_RMS_PX} px", rms_px <= MAX_HELD_OUT_RMS_PX,
                 f"{rms_px:.3f} px over {n} held-out views", hint)


def _max_visible_radius(keep: np.ndarray, cx: float, cy: float) -> float:
    ys, xs = np.nonzero(keep)
    return float(np.sqrt(np.max((xs - cx) ** 2 + (ys - cy) ** 2))) if len(xs) else float("nan")


def _check_d(uv_fit: np.ndarray, keep: np.ndarray, cx: float, cy: float) -> tuple[Check, dict]:
    r_max = _max_visible_radius(keep, cx, cy)
    radii = np.sort(np.hypot(uv_fit[:, 0] - cx, uv_fit[:, 1] - cy))[::-1] if len(uv_fit) else np.zeros(0)
    r_k = float(radii[RADIAL_MIN_CORNERS - 1]) if len(radii) >= RADIAL_MIN_CORNERS else 0.0
    frac = r_k / r_max if r_max > 0 else 0.0
    info = {"max_visible_radius_px": round(r_max, 1), "radius_reached_px": round(r_k, 1),
            "largest_corner_radius_px": round(float(radii[0]), 1) if len(radii) else None,
            "corners_counted": RADIAL_MIN_CORNERS}
    hint = ("Too few corners near the image's outer edge: bring the board into the top-left and top-right corners "
            "of the image and to the far left and right, turned to face the lens so it is still detected there.")
    return Check("d", round(frac, 4), f">= {RADIAL_FRACTION:.0%} of {r_max:.0f} px", frac >= RADIAL_FRACTION,
                 f"{frac:.1%} ({RADIAL_MIN_CORNERS} corners reach {r_k:.0f} px of {r_max:.0f} px)", hint), info


def _cells(size, keep: np.ndarray, uv_fit: np.ndarray) -> list[dict]:
    W, H = size
    cols, rows = CELL_GRID
    out = []
    for r in range(rows):
        for c in range(cols):
            x0, x1 = int(round(c * W / cols)), int(round((c + 1) * W / cols))
            y0, y1 = int(round(r * H / rows)), int(round((r + 1) * H / rows))
            visible = float(np.mean(keep[y0:y1, x0:x1] != 0))
            n = int(np.sum((uv_fit[:, 0] + 0.5 >= x0) & (uv_fit[:, 0] + 0.5 < x1)
                           & (uv_fit[:, 1] + 0.5 >= y0) & (uv_fit[:, 1] + 0.5 < y1))) if len(uv_fit) else 0
            checked = visible >= CELL_MIN_VISIBLE
            out.append({"cell": CELL_NAMES[r][c], "box": [x0, y0, x1, y1], "visible_fraction": round(visible, 3),
                        "checked": checked, "corners": n, "passed": (not checked) or n >= CELL_MIN_CORNERS})
    return out


def _check_e(cells: list[dict]) -> Check:
    checked = [c for c in cells if c["checked"]]
    bad = [c for c in checked if not c["passed"]]
    worst = min((c["corners"] for c in checked), default=0)
    detail = (f"{len(checked) - len(bad)} of {len(checked)} cells hold >= {CELL_MIN_CORNERS} corners"
              + (": too few in " + ", ".join(f"{c['cell']} ({c['corners']})" for c in bad) if bad else ""))
    hint = ("Move the board through " + ", ".join(c["cell"] for c in bad) + " of the image and pause there "
            "(turned to face the lens near the edges).") if bad else ""
    return Check("e", worst, f">= {CELL_MIN_CORNERS} corners per cell >= {CELL_MIN_VISIBLE:.0%} visible", not bad,
                 detail, hint)


def _check_f(K, nominal) -> Check:
    fx0, fy0 = nominal[0], nominal[1]
    dx, dy = K[0, 0] / fx0 - 1.0, K[1, 1] / fy0 - 1.0
    dev = max(abs(dx), abs(dy))
    hint = ("The focal length is far from the nominal: check that this is the 160 deg IMX219 lens recording at "
            "1280x960 (the camera mode); otherwise the fit is not reliable (see the other checks).")
    return Check("f", round(dev, 4), f"within +-{FOCAL_TOLERANCE:.0%}", dev <= FOCAL_TOLERANCE,
                 f"fx {K[0, 0]:.1f} ({dx:+.1%}), fy {K[1, 1]:.1f} ({dy:+.1%}) vs nominal {fx0:.1f}, {fy0:.1f}", hint)


def _check_g(K, size) -> Check:
    W, H = size
    cx0, cy0 = (W - 1) / 2.0, (H - 1) / 2.0
    d = float(math.hypot(K[0, 2] - cx0, K[1, 2] - cy0))
    hint = ("The principal point is far from the image centre: usually too little coverage (see (d), (e)): record "
            "again with the board across the whole image; if it persists, check that the lens is screwed in "
            "straight.")
    return Check("g", round(d, 2), f"<= {PRINCIPAL_POINT_MAX_PX:.0f} px", d <= PRINCIPAL_POINT_MAX_PX,
                 f"({K[0, 2]:.1f}, {K[1, 2]:.1f}) is {d:.1f} px from ({cx0:.1f}, {cy0:.1f})", hint)


def _check_h(K, D, size, keep) -> tuple[Check, dict]:
    from umiproc.dp.geometry.camera_model import FisheyeEquidistant

    W, H = size
    m = FisheyeEquidistant(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2], *D)
    corners = {"top-left": ((-0.5, -0.5), (0, 0)), "top-right": ((W - 0.5, -0.5), (W - 1, 0))}
    fovs, info = {}, {}
    for name, (edge, px) in corners.items():
        if keep[px[1], px[0]] == 0:
            info[name] = "masked"
            continue
        rays, ok = m.unproject(np.array([edge], np.float64))
        if not ok[0]:
            info[name] = "beyond the model's valid range"
            fovs[name] = float("nan")
            continue
        theta = math.degrees(math.acos(float(np.clip(rays[0, 2], -1.0, 1.0))))
        fovs[name] = 2.0 * theta
        info[name] = round(theta, 2)
    lo, hi = FOV_RANGE_DEG
    vals = [v for v in fovs.values()]
    ok = bool(vals) and all(math.isfinite(v) and lo <= v <= hi for v in vals)
    mean = float(np.mean(vals)) if vals and all(math.isfinite(v) for v in vals) else None
    hint = ("The field of view does not match a ~160 deg lens: check the lens; otherwise the model is poorly "
            "constrained near the edges: record more corners near the top image corners (see (d)).")
    detail = (", ".join(f"{k} {v:.1f} deg" for k, v in fovs.items() if math.isfinite(v))
              or "no visible top corner the model can reach")
    return Check("h", None if mean is None else round(mean, 2), f"{lo:.0f}-{hi:.0f} deg", ok,
                 f"2 x field angle: {detail}", hint), {"field_angle_deg": info}


# --------------------------------------------------------------------------------------------- the run

def _say(log, msg: str) -> None:
    if log is not None:
        log(msg)


def calibrate_recording(episode: Path, rig, slam, max_views: int = DEFAULT_MAX_VIEWS, log=print,
                        src=None) -> Calibration:
    """Run steps 1-5 on one recording (``src``: its :class:`PiSource`, inspected by the caller)."""
    from umiproc.dp.pi_adapter import converted_name, inspect_source
    from umiproc.lens import board as lens_board
    from umiproc.slam_config import mask_keep

    t_start = time.monotonic()
    if src is None:
        src = inspect_source(episode, verify_hashes=True)
        if not src.ok:
            raise InputError(f"{episode}: {src.problem}")
    cam = _camera_entry(src)
    size = RESOLUTION
    keep = np.array(mask_keep(slam, size))
    nominal = rig.cameras["ultrawide"].model
    K0, D0 = nominal.K.astype(np.float64), np.asarray(nominal.distortion, np.float64)
    video = src.path / cam["file"]

    _say(log, f"decoding {video} and detecting the board in every frame ...")
    try:
        vc = lens_detect.detect_video(video, keep, progress=lambda n: _say(log, f"  {n} frames"))
    except (ValueError, OSError) as e:
        raise InputError(f"{video}: {e}") from None
    except Exception as e:                                # PyAV decoding errors
        if type(e).__module__.startswith("av"):
            raise InputError(f"{video}: cannot decode ({e})") from None
        raise
    if vc.n_decoded == 0:
        raise InputError(f"{video}: no frames decoded")
    still, motion = lens_detect.still_flags(vc.frames)
    with_board = sum(1 for f in vc.frames if f is not None and len(f) > 0)
    enough = sum(1 for f in vc.frames if f is not None and len(f) >= lens_detect.MIN_CORNERS)
    stats = {"decoded": vc.n_decoded, "meta_frames": cam.get("frames"), "with_board": with_board,
             "enough_corners": enough, "still": int(still.sum()), "corners_dropped_by_mask": vc.n_masked_corners}

    candidates = [lens_fit.View(f.frame, f.ids, f.uv, lens_fit.board_points(f.ids))
                  for f, s in zip(vc.frames, still) if s]
    candidates, screened_out = lens_fit.screen_views(candidates, size, K0, D0)
    candidates = [lens_fit.describe(v, size, K0, D0) for v in candidates]
    stats["screened_out"] = len(screened_out)
    selected = lens_fit.select_views(candidates, max_views)
    fit_views = [v for v in selected if v.role == "fit"]
    held = [v for v in selected if v.role == "held_out"]
    _say(log, f"  frames: {vc.n_decoded} decoded, {enough} with >= {lens_detect.MIN_CORNERS} corners outside the "
              f"mask, {stats['still']} still; views: {len(selected)} selected ({len(fit_views)} fit, "
              f"{len(held)} held out)")

    checks, result, K, D, extra = [], None, None, None, {}
    fit_error = None
    if len(fit_views) >= MIN_VIEWS_TO_FIT:
        try:
            result = lens_fit.fit_fisheye(fit_views, size, K0, D0)
        except RuntimeError as e:
            fit_error = str(e)
    else:
        fit_error = f"only {len(fit_views)} fit views (at least {MIN_VIEWS_TO_FIT} needed to attempt a fit)"

    n_fit = len(result.views) if result is not None else len(fit_views)
    checks.append(_check_a(n_fit, stats))
    agreement = None
    if result is None:
        checks.append(_check_b(None))
        checks.append(_check_c(None, len(held)))
        for cid in "defgh":
            checks.append(Check(cid, None, LIMITS[cid], False, "no fit", "", evaluated=False))
    else:
        K, D = result.K, result.D
        checks.append(_check_b(result.rms_px))
        held_err = []
        for v in held:
            pose = lens_fit.pose_from_model(v.obj, v.uv, K, D)
            if pose is None:
                v.removed_reason = "held-out pose could not be solved"
                continue
            v.rvec, v.tvec = pose
            e = lens_fit.project_points(v.obj, v.rvec, v.tvec, K, D) - v.uv
            v.rms_px = lens_fit.rms(e)
            held_err.append(e)
        held_rms = lens_fit.rms(np.concatenate(held_err)) if held_err else None
        checks.append(_check_c(held_rms, len(held)))
        uv_fit = np.concatenate([v.uv for v in result.views]) if result.views else np.zeros((0, 2))
        chk_d, extra["radial"] = _check_d(uv_fit, keep, K[0, 2], K[1, 2])
        checks.append(chk_d)
        extra["cells"] = _cells(size, keep, uv_fit)
        checks.append(_check_e(extra["cells"]))
        checks.append(_check_f(K, nominal.intrinsics))
        checks.append(_check_g(K, size))
        chk_h, extra["fov"] = _check_h(K, D, size, keep)
        checks.append(chk_h)
        agreement = lens_fit.model_agreement(K, D, size)
    passed = all(c.passed for c in checks if c.blocking) and bool(agreement and agreement["agree"])

    board = lens_board.board_info()
    report = {
        "schema": REPORT_SCHEMA, "passed": passed,
        "warnings": [c.id for c in checks if c.warned],     # warning-only checks outside their limit (never block)
        "date": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool": "umiproc lens-calibrate", "version": _version(),
        "source": {"episode_dir": str(src.path), "pi_name": converted_name(src),
                   "manifest_sha256": src.manifest_sha256, "camera_file": cam["file"], "sensor": SENSOR,
                   "resolution": list(size)},
        "rig": {"rig_id": rig.rig_id, "rig_file": str(rig.source_path), "slam_config": str(slam.source_path),
                "nominal": {"intrinsics": list(nominal.intrinsics), "distortion": list(nominal.distortion)}},
        "board": board,
        "settings": {"min_corners": lens_detect.MIN_CORNERS, "still_px": lens_detect.STILL_PX,
                     "min_common": lens_detect.MIN_COMMON, "edge_px": lens_detect.EDGE_PX,
                     "max_views": max_views, "hold_out_every": lens_fit.HOLD_OUT_EVERY,
                     "selection_grid": list(lens_fit.GRID), "cell_target": lens_fit.CELL_TARGET,
                     "tilt_weight": lens_fit.TILT_WEIGHT, "tilt_target": lens_fit.TILT_TARGET,
                     "duplicate_share": lens_fit.DUP_SHARE, "duplicate_px": lens_fit.DUP_PX,
                     "outlier_factor": lens_fit.OUTLIER_FACTOR, "outlier_min_px": lens_fit.OUTLIER_MIN_PX,
                     "outlier_max_fraction": lens_fit.OUTLIER_MAX_FRACTION,
                     "flags": "CALIB_USE_INTRINSIC_GUESS | CALIB_RECOMPUTE_EXTRINSIC | CALIB_FIX_SKEW | "
                              "CALIB_CHECK_COND"},
        "detector": {"corner_shift_corrected_px": vc.offset_px,
                     "note": "OpenCV detectBoard's ChArUco corners are shifted by this from the pixel-centre "
                             "convention; it is subtracted (umiproc/lens/detect.py)"},
        "frames": stats,
        "selection": {"candidates": len(candidates), "selected": len(selected), "fit": n_fit,
                      "held_out": len(held), "removed_by_fit": len(result.removed) if result else 0},
        "fit": None if result is None else {
            "intrinsics": [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])],
            "distortion": [float(v) for v in D], "rms_px": result.rms_px, "cv2_rms_px": result.cv_rms_px,
            "attempts": result.attempts},
        "fit_error": fit_error,
        "model_agreement": agreement,
        "checks": [c.to_dict() for c in checks],
        "coverage": extra,
        "views": {"fit": [v.summary() for v in (result.views if result else fit_views)],
                  "held_out": [v.summary() for v in held],
                  "removed": [v.summary() for v in (result.removed if result else [])]},
        "runtime_s": round(time.monotonic() - t_start, 2),
    }
    return Calibration(report=report, passed=passed, size=size, keep=keep, polygons=slam.mask.polygons,
                       backdrop=vc.backdrop, views=selected, K=K, D=D, checks=checks)


def _version() -> str:
    from umiproc import __version__

    return __version__


# --------------------------------------------------------------------------------------------- outputs

def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def coverage_image(cal: Calibration) -> np.ndarray:
    """BGR image: the backdrop frame (dimmed), the mask outline (red), the 3 x 3 grid with each cell's fit-corner
    count (green = enough, red = too few, grey = not checked), the radial-coverage circle (magenta), fit corners
    (green) and held-out corners (cyan)."""
    import cv2

    W, H = cal.size
    base = cal.backdrop if cal.backdrop is not None else np.full((H, W), 64, np.uint8)
    img = cv2.cvtColor((base.astype(np.float32) * 0.55).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for poly in cal.polygons:
        pts = np.round((np.asarray(poly, np.float64) - 0.5) * 16).astype(np.int32)
        cv2.polylines(img, [pts], True, (40, 40, 230), 2, cv2.LINE_AA, shift=4)
    cells = cal.report.get("coverage", {}).get("cells") or _cells(cal.size, cal.keep, np.zeros((0, 2)))
    for c in cells:
        x0, y0, x1, y1 = c["box"]
        colour = (150, 150, 150) if not c["checked"] else ((60, 200, 60) if c["passed"] else (40, 40, 230))
        cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (0, 200, 230), 1)
        cv2.putText(img, f"{c['cell']}: {c['corners']}", (x0 + 8, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2,
                    cv2.LINE_AA)
    if cal.K is not None:
        r = cal.report.get("coverage", {}).get("radial", {})
        if r.get("max_visible_radius_px"):
            cv2.circle(img, (int(round(cal.K[0, 2])), int(round(cal.K[1, 2]))),
                       int(round(RADIAL_FRACTION * r["max_visible_radius_px"])), (230, 60, 230), 1, cv2.LINE_AA)
    for v in cal.views:
        if v.role == "removed":
            continue
        colour = (60, 220, 60) if v.role == "fit" else (230, 220, 40)
        for u, w in v.uv:
            cv2.circle(img, (int(round(u * 4)), int(round(w * 4))), 8, colour, -1, cv2.LINE_AA, shift=2)
    status = ("ALL CHECKS PASSED" if cal.passed else
              "FAILED: " + ", ".join(c.id for c in cal.checks if c.blocking and not c.passed))
    warned = [c.id for c in cal.checks if c.warned]
    if warned:
        status += "  warning: " + ", ".join(warned)
    cv2.putText(img, f"{cal.report['source']['pi_name']}  fit corners green, held-out cyan, mask red, "
                     f"{RADIAL_FRACTION:.0%} radius magenta   {status}", (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def write_outputs(cal: Calibration, out_dir: Path) -> dict:
    """Write report.json, coverage.png and (only when passed) lens.yaml into ``out_dir``. Returns their paths."""
    import cv2

    from umiproc.lens.lensfile import dump_lens, lens_document

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"report": out_dir / "report.json", "coverage": out_dir / "coverage.png", "lens": out_dir / "lens.yaml"}
    ok, png = cv2.imencode(".png", coverage_image(cal))
    if not ok:
        raise RuntimeError("cannot encode coverage.png")
    _write_atomic(paths["coverage"], png.tobytes())
    rep = cal.report
    if cal.passed:
        metrics = {METRIC_KEYS[c.id]: c.value for c in cal.checks}
        metrics["held_out_views"] = rep["selection"]["held_out"]
        doc = lens_document(rig_id=rep["rig"]["rig_id"], sensor=SENSOR, resolution=cal.size,
                            intrinsics=rep["fit"]["intrinsics"], distortion=rep["fit"]["distortion"], date=rep["date"],
                            source={k: rep["source"][k] for k in ("episode_dir", "pi_name", "manifest_sha256",
                                                                  "camera_file")},
                            board=rep["board"], metrics=metrics)
        _write_atomic(paths["lens"], dump_lens(doc).encode("utf-8"))
    elif paths["lens"].exists():
        paths["lens"].unlink()                            # stale: from an earlier run of different code/settings
    rep["outputs"] = {k: str(p) for k, p in paths.items() if k != "lens" or cal.passed}
    _write_atomic(paths["report"], (json.dumps(rep, indent=2) + "\n").encode("utf-8"))
    return paths


# --------------------------------------------------------------------------------------------- command

def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("episode_dir", metavar="EPISODE_DIR",
                   help="the board recording's directory name in --source (e.g. episode_000123), or a path")
    p.add_argument("--source", default=str(DEFAULT_SOURCE), metavar="DIR",
                   help=f"where the Pi's recordings are (default {DEFAULT_SOURCE}; only read)")
    p.add_argument("--root", default=None, metavar="DIR", help="data root (default $UMIPROC_ROOT or ~/data/umi)")
    p.add_argument("--rig", default=DEFAULT_RIG, help=f"rig id (default {DEFAULT_RIG})")
    p.add_argument("--max-views", type=int, default=DEFAULT_MAX_VIEWS, metavar="N",
                   help=f"most views to select, every 5th held out (default {DEFAULT_MAX_VIEWS})")
    p.add_argument("--apply", action="store_true",
                   help="when every check passes, install the result as <root>/rigs/<rig>.lens.yaml (backs up an "
                        "existing one)")


def _apply_command(args, root: Path) -> str:
    from umiproc.paths import DEFAULT_ROOT

    parts = ["umiproc", "lens-calibrate", args.episode_dir]
    if os.path.normpath(str(Path(args.source).expanduser().absolute())) != str(DEFAULT_SOURCE):
        parts += ["--source", args.source]
    if root != DEFAULT_ROOT:
        parts += ["--root", str(root)]
    if args.rig != DEFAULT_RIG:
        parts += ["--rig", args.rig]
    if args.max_views != DEFAULT_MAX_VIEWS:
        parts += ["--max-views", str(args.max_views)]
    return shlex.join(parts + ["--apply"])


def _err(msg: str) -> int:
    print(f"umiproc lens-calibrate: {msg}", file=sys.stderr)
    return 1


def run(args) -> int:
    from umiproc.dp.pi_adapter import converted_name, inspect_source
    from umiproc.lens.lensfile import apply_lens
    from umiproc.paths import data_root, lens_dir, rig_lens_file

    if args.max_views < MIN_VIEWS_TO_FIT:
        return _err(f"--max-views must be at least {MIN_VIEWS_TO_FIT}")
    try:
        root = data_root(args.root)
        episode = _episode_path(args.episode_dir, args.source)
        rig, slam, _, _ = _load_configs(root, args.rig, log=lambda m: print(m, flush=True))
        print(f"checking {episode} (COMPLETE, manifest hashes, meta.json) ...", flush=True)
        src = inspect_source(episode, verify_hashes=True)
        if not src.ok:
            raise InputError(f"{episode}: {src.problem}")
        out_dir = lens_dir(root, converted_name(src))      # pi-YYYY-MM-DD_episode_NNNNNN: a reused Pi number is
                                                          # never written over an earlier result
        cal = calibrate_recording(episode, rig, slam, args.max_views, log=lambda m: print(m, flush=True), src=src)
    except (InputError, ValueError) as e:
        return _err(str(e))
    paths = write_outputs(cal, out_dir)

    rep = cal.report
    if rep["fit"]:
        fx, fy, cx, cy = rep["fit"]["intrinsics"]
        k = rep["fit"]["distortion"]
        print(f"fit: fx {fx:.2f}  fy {fy:.2f}  cx {cx:.2f}  cy {cy:.2f}  k1..k4 "
              + "  ".join(f"{v:.5f}" for v in k) + f"  ({rep['selection']['fit']} views, "
              f"{rep['selection']['removed_by_fit']} removed by the fit)")
    else:
        print(f"no fit: {rep['fit_error']}")
    print("checks:")
    for c in cal.checks:
        print("  " + c.line())
    if rep["model_agreement"] is not None and not rep["model_agreement"]["agree"]:
        print("  FAIL internal: cv2.fisheye, the rig's fisheye-equidistant and ORB-SLAM3's KannalaBrandt8 disagree "
              f"({rep['model_agreement']}); do not use this result and report it")
    written = [paths["report"], paths["coverage"]] + ([paths["lens"]] if cal.passed else [])
    print("wrote " + ", ".join(str(p) for p in written))
    for c in cal.checks:
        if c.warned:
            print(f"WARNING ({c.id}) {c.name}: {c.detail}. {c.hint}")
            print(f"  ({c.id}) is a warning only: the nominal value it compares with is the rig file's, unmeasured; "
                  "it does not block lens.yaml or --apply.")
    if not cal.passed:
        failed = [c for c in cal.checks if c.blocking and not c.passed and c.evaluated]
        skipped = [c for c in cal.checks if c.blocking and not c.evaluated]
        print("FAILED: " + ", ".join(f"({c.id}) {c.name}" for c in failed)
              + (f"; not evaluated: {', '.join(f'({c.id})' for c in skipped)}" if skipped else ""))
        for c in failed:
            if c.hint:
                print(f"  ({c.id}) {c.hint}")
        print(f"see {paths['coverage']} for where the corners were. Record the board again (DIRECTIONS.md section 1); "
              "the processing keeps using the lens model it had.")
        return 1
    print("all checks passed." if not rep["warnings"] else
          "all checks passed (with the warning above).")
    dest = rig_lens_file(root, rig.rig_id)
    if not args.apply:
        print(f"to use it, install it as {dest} with:\n  {_apply_command(args, root)}")
        return 0
    dest, backup, changed = apply_lens(paths["lens"], dest)
    if not changed:
        print(f"{dest} already holds exactly this lens file; nothing changed")
    else:
        print(f"applied: {dest}" + (f" (the previous one is now {backup})" if backup else ""))
    return 0
