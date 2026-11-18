"""``umiproc slam-3d FOLDER [--run mapping|mapping_full] [--root DIR] [--points CSV] [--out HTML] [--min-obs N]
[--near F|Dm] [--box F] [--no-images] [--thumb-width PX]``: a 3-D view of one ``umiproc_slam`` run as one HTML file.

``FOLDER`` is a task folder's name (its map's run ``--run``, default ``mapping``) or an existing run folder (e.g.
``maps_failed/<pi_name>/<UTC>/mapping``; :mod:`umiproc.viz.target`). Written: ``analysis/slam_3d.html`` in the task
folder (``analysis/slam_3d_<run>.html`` for another run; ``<RUN_DIR>/../analysis/`` for a run folder). The axes are
metres only as far as the map's scale is right (``map``'s check M4).

Shows, in the trajectory CSV's world W (the camera frame of the saved map's first keyframe), turned so that up is up:

* the saved map's points (``map_points.csv``), in their real colour (from the keyframe images,
  :mod:`umiproc.viz.slam_3d_images`), or coloured by height or by their number of observations (a menu);
* its keyframes (``map_points_keyframes.csv``) as small camera frustums (x right, y down, z = the optical axis);
* the run's camera path (``trajectory.csv``: every posed frame), coloured by time, broken where nothing was posed;
* a slider (and a play button) that moves a camera frustum along the path, with the frame, time and state; beside the
  plot, the photo of the last keyframe at or before that frame (its frustum in orange). Hovering a keyframe shows its
  photo, clicking one moves the slider there.

**Declutter** (the CLI sets the defaults, the page has the same controls and shows how many points each hides):
``--min-obs N`` (default :data:`MIN_OBS`: points seen by fewer keyframes are hidden); ``--near F`` (default
:data:`NEAR`: points farther than F times the path's size from the nearest camera position (path or keyframe) are
hidden; the size is the largest side of the path's bounding box, so this works whatever the map's scale; ``--near
0.5m`` gives a distance in map units instead, ``--near 0`` turns it off); ``--box F`` (off by default: only points in
the path's bounding box, up-aligned, grown by F times its size); and the far outliers (farther than
``OUTLIER_FACTOR`` times the 2-98 % spread outside it; on by default). Hidden points stay in the file.

The map files are written by the front-end's ``--save_points RUN_DIR/map_points.csv`` (``native/umi_slam``; ``umiproc
map`` passes it to its mapping runs when the build has it): next to the CSV, ``<stem>_keyframes.csv`` and
``<stem>_meta.json`` (``up_W``: gravity up in W once the IMU is initialised).
Without ``up_W``, up is the first keyframe's -y (the camera's up). A saved atlas can be exported without re-running
SLAM: ``umiproc_slam ... --load_map MAP.osa --max_frames 0 --save_points OUT.csv``; its world is then the mapping
run's (``--points OUT.csv`` with the mapping run folder). With no map files the view shows the path only.

The images: the recording (the map's ``raw/``, else the report's ``source``), the map's ``orbslam3_settings.yaml``
(the lens) and ``slam_mask.png``; without them (or with ``--no-images``) the page has no photos and the points are
coloured by height. The page loads plotly.js from a CDN (it needs internet once to show); the data and the photos
(JPEG, ``--thumb-width`` px wide) are embedded. Frame times: ``--times``, else the recording's frames CSV (the map's
``raw/``), else estimated from the CSV's timestamp column. Exit codes: 0 written; 2 the inputs cannot be read.
"""

from __future__ import annotations

import csv
import html
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from umiproc.slam.log import STATE_NAMES, read_log
from umiproc.slam.trajectory_csv import TrajectoryCsvError, read_fork_csv, read_times
from umiproc.viz.slam_3d_images import N_VIEWS
from umiproc.viz.slam_run import PlotRunError, count_rows, read_t0_ns, times_from_csv_column

__all__ = ["HTML_NAME", "POINTS_NAME", "page_name", "MapExport", "Slam3dError", "Filters", "parse_near",
           "read_map_export", "scene_rotation", "point_filters", "keyframe_images", "build_payload", "write_html",
           "add_arguments", "run"]

HTML_NAME = "slam_3d.html"
POINTS_NAME = "map_points.csv"
POINTS_HEADER = ("id", "x", "y", "z", "observations")
KEYFRAMES_HEADER = ("id", "frame_id", "timestamp", "x", "y", "z", "q_x", "q_y", "q_z", "q_w")
OUTLIER_FACTOR = 1.0
MIN_OBS = 3
NEAR = 1.0                 # x the path's size
THUMB_WIDTH = 320
SAMPLE_WIDTH = 640         # the keyframe images are decoded at this width for the colours
PLOTLY_URL = "https://cdn.plot.ly/plotly-2.35.2.min.js"


class Slam3dError(Exception):
    """The run's files cannot be read."""


@dataclass(frozen=True)
class MapExport:
    """What ``--save_points`` wrote: points and keyframes in W, and the meta JSON."""

    point_ids: np.ndarray        # (P,) int64
    points: np.ndarray           # (P, 3) float64, in W
    observations: np.ndarray     # (P,) int64
    keyframe_ids: np.ndarray     # (K,) int64
    keyframe_t_s: np.ndarray     # (K,) float64, the keyframe's time (the run's seconds)
    T_W_kf: np.ndarray           # (K, 4, 4) float64
    meta: dict
    keyframe_frames: np.ndarray | None = None   # (K,) int64, the fed frame index (frame_id)


@dataclass(frozen=True)
class Filters:
    """The declutter defaults (module docstring). ``near``: a factor of the path's size, ``near_m``: map units (wins
    when set); None: off. ``box``: a margin factor, None: off."""

    min_obs: int = MIN_OBS
    near: float | None = NEAR
    near_m: float | None = None
    box: float | None = None
    outliers: bool = True


def parse_near(v: str) -> tuple[float | None, float | None]:
    """``--near``: ``"1.5"`` -> (1.5, None) (x the path's size), ``"0.4m"`` -> (None, 0.4), ``"0"`` -> off."""
    s = str(v).strip().lower()
    absolute = s.endswith("m")
    x = float(s[:-1] if absolute else s)
    if not np.isfinite(x) or x < 0:
        raise ValueError(f"--near {v}: want a factor >= 0 or a distance like 0.5m")
    if x == 0:
        return None, None
    return (None, x) if absolute else (x, None)


def _sibling(csv_path: Path, suffix: str) -> Path:
    name = csv_path.name
    stem = name[:-4] if name.endswith(".csv") and len(name) > 4 else name
    return csv_path.with_name(stem + suffix)


def _read_rows(path: Path, header: tuple[str, ...]) -> list[list[str]]:
    try:
        with open(path, newline="") as f:
            rows = [r for r in csv.reader(f) if r]
    except OSError as e:
        raise Slam3dError(f"cannot read {path}: {e}") from e
    if not rows or tuple(c.strip() for c in rows[0]) != header:
        raise Slam3dError(f"{path}: header is not {','.join(header)}")
    for k, r in enumerate(rows[1:], 2):
        if len(r) != len(header):
            raise Slam3dError(f"{path}: line {k} has {len(r)} values, want {len(header)}")
    return rows[1:]


def read_map_export(points_csv: str | Path) -> MapExport:
    """Read ``points_csv`` and its ``_keyframes.csv`` and ``_meta.json`` (module docstring)."""
    p = Path(points_csv)
    try:
        meta = json.loads(_sibling(p, "_meta.json").read_text())
    except (OSError, ValueError) as e:
        raise Slam3dError(f"cannot read {_sibling(p, '_meta.json')}: {e}") from e
    if not isinstance(meta, dict) or meta.get("format") != "umiproc.map_points.v1":
        raise Slam3dError(f"{_sibling(p, '_meta.json')}: not a umiproc.map_points.v1 file")
    try:
        pr = _read_rows(p, POINTS_HEADER)
        kr = _read_rows(_sibling(p, "_keyframes.csv"), KEYFRAMES_HEADER)
        pts = np.array([[float(v) for v in r[1:4]] for r in pr], dtype=np.float64).reshape(-1, 3)
        kv = np.array([[float(v) for v in r] for r in kr], dtype=np.float64).reshape(-1, len(KEYFRAMES_HEADER))
        out = MapExport(point_ids=np.array([int(r[0]) for r in pr], dtype=np.int64), points=pts,
                        observations=np.array([int(r[4]) for r in pr], dtype=np.int64),
                        keyframe_ids=kv[:, 0].astype(np.int64), keyframe_t_s=kv[:, 2],
                        T_W_kf=_poses(kv[:, 3:6], kv[:, 6:10]), meta=meta,
                        keyframe_frames=kv[:, 1].astype(np.int64))
    except ValueError as e:
        raise Slam3dError(f"{p}: malformed value: {e}") from e
    if not (np.isfinite(out.points).all() and np.isfinite(out.T_W_kf).all()):
        raise Slam3dError(f"{p}: non-finite values")
    return out


def _poses(t: np.ndarray, q_xyzw: np.ndarray) -> np.ndarray:
    T = np.tile(np.eye(4), (len(t), 1, 1))
    if len(t):
        T[:, :3, :3] = Rotation.from_quat(q_xyzw).as_matrix()
        T[:, :3, 3] = t
    return T


def scene_rotation(up_W) -> np.ndarray:
    """The rotation R (3x3) that turns W so that ``up_W`` becomes +z (plotly's up); ``None``: the camera's up (-y)."""
    up = np.array([0.0, -1.0, 0.0]) if up_W is None else np.asarray(up_W, dtype=np.float64)
    up = up / np.linalg.norm(up)
    z = np.array([0.0, 0.0, 1.0])
    axis, c = np.cross(up, z), float(np.dot(up, z))
    s = float(np.linalg.norm(axis))
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    return Rotation.from_rotvec(axis / s * np.arctan2(s, c)).as_matrix()


def _frustums(T: np.ndarray, size: float) -> np.ndarray:
    """Line vertices (with NaN breaks) of a small camera frustum at each pose: apex to 4 corners, then the image
    rectangle (x right, y down, z forward; 4:3)."""
    c = np.array([[-0.67, -0.5, 1.0], [0.67, -0.5, 1.0], [0.67, 0.5, 1.0], [-0.67, 0.5, 1.0]]) * size
    out = []
    for M in T:
        o = M[:3, 3]
        w = c @ M[:3, :3].T + o
        out += [o, w[0], o, w[1], o, w[2], o, w[3], w[0], w[1], w[2], w[3], w[0], [np.nan] * 3]
    return np.asarray(out, dtype=np.float64).reshape(-1, 3)


def _robust_keep(xyz: np.ndarray, factor: float = OUTLIER_FACTOR) -> np.ndarray:
    if len(xyz) < 10:
        return np.ones(len(xyz), bool)
    lo, hi = np.percentile(xyz, 2, axis=0), np.percentile(xyz, 98, axis=0)
    pad = factor * np.maximum(hi - lo, 1e-6)
    return ((xyz >= lo - pad) & (xyz <= hi + pad)).all(axis=1)


def _r(a: np.ndarray, nd: int = 4) -> list:
    """JSON-able, rounded, NaN -> null."""
    a = np.round(np.asarray(a, dtype=np.float64), nd)
    return [None if not np.isfinite(v) else float(v) for v in a.ravel()] if a.ndim == 1 else \
        [_r(row, nd) for row in a]


def _path_distance(pts: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Each point's distance to the nearest camera position (0 everywhere when there is none)."""
    if not len(centers) or not len(pts):
        return np.zeros(len(pts))
    from scipy.spatial import cKDTree

    return cKDTree(centers).query(pts)[0]


def point_filters(pts_scene: np.ndarray, obs: np.ndarray, dist: np.ndarray, outlier: np.ndarray, extent: float,
                  box_lo: np.ndarray, box_hi: np.ndarray, f: Filters) -> dict[str, np.ndarray]:
    """Per filter, the points it hides (the page's same rules), and ``shown``."""
    near = f.near_m if f.near_m is not None else (f.near * extent if f.near is not None else None)
    hide = {"min_obs": obs < f.min_obs,
            "near": dist > near if near is not None else np.zeros(len(obs), bool),
            "outliers": outlier if f.outliers else np.zeros(len(obs), bool)}
    if f.box is not None and len(pts_scene):
        m = f.box * extent
        hide["box"] = ~((pts_scene >= box_lo - m) & (pts_scene <= box_hi + m)).all(axis=1)
    else:
        hide["box"] = np.zeros(len(obs), bool)
    hide["shown"] = ~(hide["min_obs"] | hide["near"] | hide["outliers"] | hide["box"])
    return hide


def build_payload(traj, export: MapExport | None, *, title: str, filters: Filters | None = None,
                  colours: np.ndarray | None = None, thumbs: list | None = None,
                  kf_frames: np.ndarray | None = None) -> dict:
    """The page's data: everything rotated so that up is +z (:func:`scene_rotation`). All points are in it, with what
    the filters need (the page applies them; ``filters`` are its defaults). ``colours``: (P, 3) 0-255 per point, NaN
    rows for points no image sees. ``thumbs``: per keyframe a data URI or None. ``kf_frames``: per keyframe its fed
    frame index (default the export's ``frame_id``)."""
    f = filters or Filters()
    up_W = export.meta.get("up_W") if export is not None else None
    R = scene_rotation(up_W)
    rot = lambda p: np.asarray(p, dtype=np.float64).reshape(-1, 3) @ R.T  # noqa: E731

    path_xyz = np.full((0, 3), np.nan)
    path_t = np.zeros(0)
    frames: list[dict] = []
    if traj is not None and traj.n_frames:
        t_s = (traj.t_ns - traj.t_ns[0]) / 1e9
        posed = traj.posed
        P = traj.T_W_cam[:, :3, 3]
        path_xyz = np.where(posed[:, None], P, np.nan)
        path_t = t_s
        idx = np.flatnonzero(posed)
        frames = [{"k": int(k), "t": round(float(t_s[k]), 3), "state": STATE_NAMES.get(int(traj.state[k]),
                                                                                    str(int(traj.state[k]))),
                   "tracked": bool(traj.tracked[k])} for k in idx]
        T_posed = traj.T_W_cam[idx]
    else:
        T_posed = np.zeros((0, 4, 4))

    pts = np.zeros((0, 3))
    obs = np.zeros(0, np.int64)
    kf_T = np.zeros((0, 4, 4))
    kf_ids, kf_t = np.zeros(0, np.int64), np.zeros(0)
    if export is not None:
        pts, obs = export.points, export.observations
        kf_T, kf_ids, kf_t = export.T_W_kf, export.keyframe_ids, export.keyframe_t_s
        if kf_frames is None:
            kf_frames = export.keyframe_frames
    outlier = ~_robust_keep(pts)

    # the path's size: the largest side of the bounding box of the path and keyframes (frustums are 4 % of it)
    centers = np.concatenate([path_xyz[np.isfinite(path_xyz).all(axis=1)], kf_T[:, :3, 3]])
    extent = float(np.max(np.ptp(centers, axis=0))) if len(centers) > 1 else 0.25
    extent = max(extent, 1e-3)
    size = max(0.01, 0.04 * extent)
    dist = _path_distance(pts, centers)

    Rh = np.eye(4)
    Rh[:3, :3] = R
    pts_r = rot(pts)
    c_r = rot(centers)
    box_lo = c_r.min(axis=0) if len(c_r) else np.zeros(3)
    box_hi = c_r.max(axis=0) if len(c_r) else np.zeros(3)
    hide = point_filters(pts_r, obs, dist, outlier, extent, box_lo, box_hi, f)
    # the initial view: the path and keyframes and the shown points (1-99 %), true to scale; "shown points" lets
    # plotly fit everything shown
    lo, hi = box_lo.copy(), box_hi.copy()
    sp = pts_r[hide["shown"]]
    if len(sp) >= 10:
        lo = np.minimum(lo, np.percentile(sp, 1, axis=0))
        hi = np.maximum(hi, np.percentile(sp, 99, axis=0))
    pad = 0.05 * max(float(np.max(hi - lo)), 0.1)
    lo, hi = lo - pad, hi + pad
    near = [[round(float(a), 3), round(float(b), 3)] for a, b in zip(lo, hi)]
    aspect = (hi - lo) / float(np.max(hi - lo))
    rgb = None
    if colours is not None and len(pts):
        c = np.asarray(colours, dtype=np.float64)
        rgb = [None if not np.isfinite(v).all() else "#%02x%02x%02x" % tuple(int(x) for x in v) for v in c]
    near_factor = (f.near_m / extent) if f.near_m is not None else f.near
    return {
        "near": near,
        "aspect": {"x": round(float(aspect[0]), 3), "y": round(float(aspect[1]), 3), "z": round(float(aspect[2]), 3)},
        "title": title,
        "up": "gravity (IMU initialised)" if up_W is not None else "the first keyframe's -y (no gravity)",
        "meta": export.meta if export is not None else None,
        "hidden_points": int((~hide["shown"]).sum()),
        "hidden_by": {k: int(v.sum()) for k, v in hide.items() if k != "shown"},
        "extent": round(extent, 4),
        "path_box": [_r(box_lo), _r(box_hi)],
        "filters": {"min_obs": int(f.min_obs), "near": None if near_factor is None else round(float(near_factor), 3),
                    "box": f.box, "outliers": bool(f.outliers)},
        "frustum_size": size,
        "n_views": N_VIEWS,
        "points": {"xyz": _r(pts_r), "height": _r(pts_r[:, 2] if len(pts_r) else np.zeros(0)),
                   "obs": [int(v) for v in obs], "dist": _r(dist, 3), "outlier": [int(v) for v in outlier],
                   "rgb": rgb},
        "keyframes": {"ids": [int(v) for v in kf_ids], "t": _r(kf_t, 3), "xyz": _r(rot(kf_T[:, :3, 3])),
                      "lines": _r(rot(_frustums(kf_T, size))),
                      "T": [_r((Rh @ M)[:3, :].ravel(), 5) for M in kf_T],
                      "frame": [int(v) for v in kf_frames] if kf_frames is not None else [-1] * len(kf_ids),
                      "thumb": thumbs if thumbs is not None else [None] * len(kf_ids)},
        "path": {"xyz": _r(rot(path_xyz)), "t": _r(path_t, 3)},
        "frames": frames,
        "frame_T": [_r((Rh @ M)[:3, :].ravel(), 5) for M in T_posed],
    }


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<script src="__PLOTLY__"></script>
<style>
 body { margin: 0; font: 13px system-ui, sans-serif; background: #fff; color: #222; }
 #ctl { padding: 6px 16px; display: flex; gap: 14px; align-items: center; flex-wrap: wrap;
   border-bottom: 1px solid #ddd; }
 #ctl input[type=number] { width: 4.5em; }
 #count { color: #444; font-family: monospace; }
 #main { display: flex; height: calc(100vh - 150px); min-height: 360px; }
 #plot { flex: 1; min-width: 0; }
 #side { width: 340px; padding: 8px; box-sizing: border-box; border-left: 1px solid #ddd; overflow: auto; }
 #side img { width: 100%; image-rendering: auto; background: #eee; min-height: 40px; display: block; }
 #cap { font-family: monospace; margin-top: 4px; white-space: pre-wrap; }
 #bar { padding: 6px 16px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
 #slider { flex: 1; min-width: 200px; }
 #info { font-family: monospace; min-width: 22em; }
 #note { padding: 0 16px 6px; color: #666; font-size: 12px; }
 @media (max-width: 800px) { #main { flex-direction: column; height: auto; } #plot { height: 70vh; }
   #side { width: 100%; border-left: 0; } }
</style></head><body>
<div id="ctl">
 <label>colour <select id="colour"><option value="image">image</option><option value="height">height</option>
  <option value="obs">observations</option></select></label>
 <label>min observations <input id="minobs" type="number" min="0" step="1"></label>
 <label><input id="nearon" type="checkbox"> near the path: within <input id="near" type="number" min="0" step="0.1">
  x path size <span id="nearm"></span></label>
 <label><input id="boxon" type="checkbox"> path box + <input id="box" type="number" min="0" step="0.1"> x size</label>
 <label><input id="outl" type="checkbox"> hide far outliers</label>
 <span id="count"></span>
</div>
<div id="main"><div id="plot"></div>
<div id="side"><img id="photo" alt=""><div id="cap"></div></div></div>
<div id="bar"><button id="play">play</button><input id="slider" type="range" min="0" max="0" value="0">
<span id="info"></span></div>
<div id="note"></div>
<script>
const D = __DATA__;
const X = (a, i) => a.map(p => p[i]);
const P = D.points, KF = D.keyframes, F = D.filters, NP = P.xyz.length;
const hasRgb = !!(P.rgb && P.rgb.some(c => c));
const $ = id => document.getElementById(id);
if (!hasRgb) { $("colour").querySelector("option[value=image]").disabled = true; $("colour").value = "height"; }
$("minobs").value = F.min_obs;
$("nearon").checked = F.near !== null; $("near").value = F.near !== null ? F.near : 1;
$("boxon").checked = F.box !== null; $("box").value = F.box !== null ? F.box : 0.5;
$("outl").checked = F.outliers;

const traces = [];
traces.push({type: "scatter3d", mode: "markers", name: "map points", x: [], y: [], z: [],
  marker: {size: 3}, text: [], hoverinfo: "text"});
traces.push({type: "scatter3d", mode: "lines", name: "keyframes (" + KF.ids.length + ")",
  x: X(KF.lines, 0), y: X(KF.lines, 1), z: X(KF.lines, 2),
  line: {color: "#1f5fbf", width: 1}, hoverinfo: "skip"});
traces.push({type: "scatter3d", mode: "markers", name: "keyframe centres", showlegend: false,
  x: X(KF.xyz, 0), y: X(KF.xyz, 1), z: X(KF.xyz, 2), marker: {size: 3, color: "#1f5fbf"},
  text: KF.ids.map((id, i) => "KF " + id + ", t " + KF.t[i] + " s (click: go there)"), hoverinfo: "text"});
traces.push({type: "scatter3d", mode: "lines", name: "camera path (colour = time)",
  x: X(D.path.xyz, 0), y: X(D.path.xyz, 1), z: X(D.path.xyz, 2),
  line: {color: D.path.t, colorscale: "Plasma", width: 4},
  text: D.path.t.map((t, k) => "frame " + k + ", " + t + " s"), hoverinfo: "text"});
traces.push({type: "scatter3d", mode: "lines", name: "current pose", x: [], y: [], z: [],
  line: {color: "#e0201b", width: 6}, hoverinfo: "skip"});
traces.push({type: "scatter3d", mode: "lines", name: "keyframe in the photo", x: [], y: [], z: [],
  line: {color: "#ff8c00", width: 6}, hoverinfo: "skip"});
const CUR = 4, KCUR = 5;
const layout = {title: {text: D.title, font: {size: 15}}, margin: {l: 0, r: 0, t: 36, b: 0},
  legend: {x: 0, y: 1}, scene: {aspectmode: "manual", aspectratio: D.aspect,
  camera: {eye: {x: 0.95, y: 0.95, z: 0.75}}, xaxis: {title: "x (m)", range: D.near[0]},
  yaxis: {title: "y (m)", range: D.near[1]}, zaxis: {title: "up (m)", range: D.near[2]}},
  updatemenus: [{type: "buttons", direction: "right", x: 0.99, xanchor: "right", y: 1.0, yanchor: "top",
    buttons: [
      {label: "view: near the path", method: "relayout", args: [{"scene.aspectmode": "manual",
        "scene.aspectratio": D.aspect,
        "scene.xaxis.range": D.near[0], "scene.yaxis.range": D.near[1], "scene.zaxis.range": D.near[2]}]},
      {label: "view: shown points", method: "relayout", args: [{"scene.aspectmode": "data",
        "scene.xaxis.autorange": true, "scene.yaxis.autorange": true, "scene.zaxis.autorange": true}]}]}]};
Plotly.newPlot("plot", traces, layout, {responsive: true});

function pct(a, q) { if (!a.length) return 0; const s = a.slice().sort((u, v) => u - v);
  return s[Math.min(s.length - 1, Math.max(0, Math.round(q * (s.length - 1))))]; }
function applyPoints() {
  const minobs = +$("minobs").value || 0, nearOn = $("nearon").checked, nearD = (+$("near").value) * D.extent;
  const boxOn = $("boxon").checked, bm = (+$("box").value) * D.extent, outl = $("outl").checked;
  const lo = D.path_box[0], hi = D.path_box[1];
  const by = {obs: 0, near: 0, box: 0, outl: 0}, idx = [];
  for (let i = 0; i < NP; i++) {
    const p = P.xyz[i];
    const a = P.obs[i] < minobs, b = nearOn && P.dist[i] > nearD, o = outl && P.outlier[i] === 1;
    const c = boxOn && [0, 1, 2].some(j => p[j] < lo[j] - bm || p[j] > hi[j] + bm);
    by.obs += a; by.near += b; by.box += c; by.outl += o;
    if (!(a || b || c || o)) idx.push(i);
  }
  const mode = $("colour").value;
  let marker = {size: 3};
  if (mode === "image") {
    marker.color = idx.map(i => P.rgb[i] || "#9a9a9a"); marker.showscale = false;
  } else {
    const v = idx.map(i => mode === "height" ? P.height[i] : P.obs[i]);
    marker.color = v; marker.colorscale = "Viridis"; marker.showscale = true;
    marker.cmin = mode === "height" ? pct(v, 0.02) : 2;
    marker.cmax = mode === "height" ? pct(v, 0.98) : Math.max(3, pct(v, 0.95));
    marker.colorbar = {title: {text: mode === "height" ? "height (m)" : "observations"}, len: 0.6};
  }
  Plotly.restyle("plot", {x: [idx.map(i => P.xyz[i][0])], y: [idx.map(i => P.xyz[i][1])],
    z: [idx.map(i => P.xyz[i][2])], marker: [marker], name: ["map points (" + idx.length + ")"],
    text: [idx.map(i => "obs " + P.obs[i] + ", " + P.dist[i] + " from the path")]}, [0]);
  $("nearm").textContent = "(" + nearD.toFixed(2) + " m; size " + D.extent.toFixed(2) + " m)";
  $("count").textContent = "shown " + idx.length + " of " + NP + "; hidden by: min obs " + by.obs + ", near " +
    by.near + ", box " + by.box + ", outliers " + by.outl;
}
["colour", "minobs", "nearon", "near", "boxon", "box", "outl"].forEach(id =>
  $(id).addEventListener("input", applyPoints));
["nearon", "boxon", "outl"].forEach(id => $(id).addEventListener("change", applyPoints));
applyPoints();

function frustumOf(T, s) {   // T: 3x4 row-major, camera -> scene
  const C = [[-0.67, -0.5, 1], [0.67, -0.5, 1], [0.67, 0.5, 1], [-0.67, 0.5, 1]].map(c => c.map(v => v * s));
  const ap = v => [0, 1, 2].map(r => T[4 * r] * v[0] + T[4 * r + 1] * v[1] + T[4 * r + 2] * v[2] + T[4 * r + 3]);
  const o = ap([0, 0, 0]), w = C.map(ap);
  return [o, w[0], o, w[1], o, w[2], o, w[3], w[0], w[1], w[2], w[3], w[0]];
}
function setLine(trace, L) {
  Plotly.restyle("plot", {x: [L.map(p => p[0])], y: [L.map(p => p[1])], z: [L.map(p => p[2])]}, [trace]);
}
let shownKf = -2;
function showKf(j, why) {
  if (j === shownKf) return;
  shownKf = j;
  if (j < 0) { $("photo").removeAttribute("src"); $("cap").textContent = "no keyframe yet"; setLine(KCUR, []); return; }
  if (KF.thumb[j]) $("photo").src = KF.thumb[j]; else $("photo").removeAttribute("src");
  $("cap").textContent = "KF " + KF.ids[j] + "  frame " + KF.frame[j] + "  t " + KF.t[j] + " s\\n" + why +
    (KF.thumb[j] ? "" : "\\n(no photo: the recording or lens file was not found)");
  setLine(KCUR, frustumOf(KF.T[j], D.frustum_size * 1.3));
}
const kfOrder = KF.frame.map((f, j) => [f, j]).filter(a => a[0] >= 0).sort((a, b) => a[0] - b[0]);
function kfAt(k) { let j = -1; for (const [f, i] of kfOrder) { if (f <= k) j = i; else break; } return j; }

const slider = $("slider"), info = $("info");
slider.max = Math.max(0, D.frames.length - 1);
function show(i) {
  if (!D.frames.length) { info.textContent = "no posed frame in this run"; return; }
  const f = D.frames[i];
  setLine(CUR, frustumOf(D.frame_T[i], D.frustum_size * 1.6));
  info.textContent = "frame " + f.k + "  t " + f.t.toFixed(2) + " s  " + f.state + (f.tracked ? "" : " (not tracked)");
  showKf(kfAt(f.k), "the last keyframe at or before frame " + f.k);
}
slider.addEventListener("input", () => show(+slider.value));
let timer = null;
$("play").addEventListener("click", e => {
  if (timer) { clearInterval(timer); timer = null; e.target.textContent = "play"; return; }
  e.target.textContent = "pause";
  timer = setInterval(() => { slider.value = (+slider.value + 3) % (D.frames.length || 1); show(+slider.value); }, 100);
});
const plotDiv = $("plot");
plotDiv.on("plotly_hover", ev => { const p = ev.points[0]; if (p.curveNumber === 2) { shownKf = -2;
  showKf(p.pointNumber, "hovered"); } });
plotDiv.on("plotly_unhover", () => { shownKf = -2; show(+slider.value); });
plotDiv.on("plotly_click", ev => { const p = ev.points[0]; if (p.curveNumber !== 2) return;
  const fk = KF.frame[p.pointNumber]; let best = 0;
  D.frames.forEach((f, i) => { if (Math.abs(f.k - fk) < Math.abs(D.frames[best].k - fk)) best = i; });
  slider.value = best; show(best); });
show(0);
if (!D.frames.length && KF.ids.length) showKf(0, "");
const m = D.meta;
$("note").textContent = "W = the camera frame of the saved map's first keyframe, turned so that up is " + D.up +
  ". " + (m ? "Map " + m.map_id + ": " + m.map_points + " points, " + m.keyframes + " keyframes. " :
  "No map points (run umiproc_slam with --save_points). ") + "Path: " + D.frames.length + " posed frames. " +
  (hasRgb ? "Image colours: the median of the " + D.n_views + " nearest keyframes that see the point " +
  "(gripper mask skipped, no occlusion test; grey: no keyframe sees it). " : "") +
  "Path size = the largest side of its bounding box (metres only if the map's scale is right). " +
  "Drag to rotate, scroll to zoom, click legend entries to hide; hover a keyframe for its photo, click to go there.";
</script></body></html>
"""


def write_html(path: str | Path, payload: dict) -> Path:
    out = Path(path)
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    out.write_text(PAGE.replace("__TITLE__", html.escape(payload["title"])).replace("__PLOTLY__", PLOTLY_URL)
                   .replace("__DATA__", data))
    return out


def _read_trajectory(csv_path: Path, log_path: Path | None, times_txt=None):
    """The run's trajectory, or None when it fed no frame. Frame times: ``times_txt`` (exact: a times.txt path or the
    fed frames' times in ns), else estimated from the CSV's timestamp column
    (:func:`umiproc.viz.slam_run.times_from_csv_column`)."""
    try:
        summary = read_log(log_path).summary if log_path is not None else None
    except ValueError as e:                  # SlamLogError
        raise Slam3dError(f"cannot read {log_path}: {e}") from e
    t0 = read_t0_ns(csv_path)
    n = summary.frames_fed if summary is not None else count_rows(csv_path)
    if n == 0:
        return None
    try:
        if isinstance(times_txt, (str, Path)):
            times = read_times(times_txt)
        elif times_txt is not None:
            times = np.asarray(times_txt, dtype=np.int64)[:n]
        else:
            rate = summary.pacing.recording_rate_hz if summary is not None and summary.pacing is not None else 30.0
            times = times_from_csv_column(csv_path, t0, n, 1e9 / (rate or 30.0))
        return read_fork_csv(csv_path, times, t0_ns=t0, frames_fed=n)
    except TrajectoryCsvError as e:
        raise Slam3dError(str(e)) from e


def keyframe_images(export: MapExport, recording: Path, settings: Path, mask_png: Path | None, t0_ns: int,
                    n_fed: int, *, thumb_width: int = THUMB_WIDTH) -> tuple[np.ndarray, list, dict]:
    """``(colours (P, 3) float with NaN rows, thumbnails per keyframe, stats)`` from the recording's ultrawide video
    (:mod:`umiproc.viz.slam_3d_images`). Raises PlotRunError / ValueError / OSError when the inputs do not fit."""
    import cv2

    from umiproc.viz import slam_3d_images as I
    from umiproc.viz.slam_run import times_from_frames_csv
    from umiproc.viz.target import ultrawide_video

    cam = I.read_camera(settings)
    video, frames_csv = ultrawide_video(recording)
    _times, i0 = times_from_frames_csv(frames_csv, t0_ns, n_fed)
    mask = None
    if mask_png is not None and mask_png.is_file():
        mask = cv2.imread(str(mask_png), cv2.IMREAD_GRAYSCALE)
        if mask is not None and mask.shape != (cam.height, cam.width):
            mask = cv2.resize(mask, (cam.width, cam.height), interpolation=cv2.INTER_NEAREST)
    fr = export.keyframe_frames if export.keyframe_frames is not None else np.zeros(0, np.int64)
    want = {j: i0 + int(k) for j, k in enumerate(fr) if 0 <= k < n_fed}
    imgs = I.read_keyframe_images(video, want, SAMPLE_WIDTH)
    rgb, dist = I.sample_colours(cam, export.T_W_kf, export.points, imgs, mask)
    col, views = I.point_colours(rgb, dist)
    colours = col.astype(np.float64)
    colours[views == 0] = np.nan
    thumbs = [I.thumbnail_uri(imgs[j], thumb_width, mask=mask) if j in imgs and thumb_width > 0 else None
              for j in range(len(export.keyframe_ids))]
    return colours, thumbs, {"images": len(imgs), "coloured": int((views > 0).sum()),
                             "median_views": float(np.median(views)) if len(views) else 0.0}


def make_page(run_dir: str | Path, *, points: str | Path | None = None, out: str | Path | None = None,
              title: str | None = None, times=None, filters: Filters | None = None,
              recording: str | Path | None = None, settings: str | Path | None = None,
              mask: str | Path | None = None, thumb_width: int = THUMB_WIDTH) -> tuple[Path, list[str]]:
    """Write the page for ``run_dir``; returns (path, the lines to print). ``times``: a times.txt path or an array of
    the fed frames' times in ns (default: estimated from the CSV). ``recording`` + ``settings`` (+ ``mask``): the
    keyframe photos and the points' colours (none without them)."""
    run_dir = Path(run_dir)
    filters = filters or Filters()
    csv_path = run_dir / "trajectory.csv"
    if not csv_path.is_file():
        raise Slam3dError(f"no trajectory.csv in {run_dir}")
    log_path = run_dir / "umiproc_slam.log"
    try:
        traj = _read_trajectory(csv_path, log_path if log_path.is_file() else None,
                                Path(times) if isinstance(times, str) else times)
    except PlotRunError as e:
        raise Slam3dError(str(e)) from e
    pts_path = Path(points) if points is not None else run_dir / POINTS_NAME
    lines = []
    export = None
    if pts_path.is_file():
        export = read_map_export(pts_path)
    elif points is not None:
        raise Slam3dError(f"no map points file {pts_path}")
    else:
        lines.append(f"no {POINTS_NAME} in {run_dir}: the path only (run umiproc_slam with --save_points, or pass "
                     "--points)")
    colours = thumbs = None
    if export is not None and recording is not None and settings is not None:
        if not Path(settings).is_file():
            lines.append(f"no photos: no lens/settings file {settings}")
        else:
            try:
                n_fed = traj.n_frames if traj is not None else 0
                colours, thumbs, st = keyframe_images(export, Path(recording), Path(settings),
                                                      Path(mask) if mask is not None else None,
                                                      read_t0_ns(csv_path), n_fed, thumb_width=thumb_width)
                lines.append(f"photos: {st['images']} of {len(export.keyframe_ids)} keyframes from {recording}; "
                             f"colours for {st['coloured']} of {len(export.points)} points (median of the nearest "
                             f"views, gripper mask skipped)")
            except (PlotRunError, ValueError, OSError) as e:
                lines.append(f"no photos: {e}")
    title = title or f"SLAM run {run_dir.parent.name}/{run_dir.name}"
    payload = build_payload(traj, export, title=title, filters=filters, colours=colours, thumbs=thumbs)
    if export is not None:
        m, hb, fl = export.meta, payload["hidden_by"], payload["filters"]
        near = (f"farther than {fl['near']:g} x the path's size ({fl['near'] * payload['extent']:.2f} m) from the "
                f"path {hb['near']}" if fl["near"] is not None else "near-path off")
        box = f", outside the path box + {fl['box']:g} x size {hb['box']}" if fl["box"] is not None else ""
        lines.append(f"map {m.get('map_id')}: {len(export.points)} points, {len(export.keyframe_ids)} keyframes; up = "
                     f"{payload['up']}; path size {payload['extent']:.2f} m")
        lines.append(f"points shown {len(export.points) - payload['hidden_points']} of {len(export.points)}; hidden "
                     f"by: fewer than {fl['min_obs']} observations {hb['min_obs']}, {near}{box}, far outliers "
                     f"{hb['outliers']} (the page can change these)")
    lines.append(f"path: {len(payload['frames'])} posed frames of {traj.n_frames if traj is not None else 0}")
    return write_html(Path(out) if out is not None else run_dir / HTML_NAME, payload), lines


def page_name(run: str) -> str:
    return HTML_NAME if run == "mapping" else f"slam_3d_{run}.html"


def add_arguments(p) -> None:
    p.add_argument("target", metavar="FOLDER", help="a task folder's name (under <root>/tasks/: its map's run), or a "
                                                    "run folder (trajectory.csv + .t0_ns, umiproc_slam.log, "
                                                    "map_points.csv)")
    p.add_argument("--run", choices=("mapping", "mapping_full"), default="mapping",
                   help="which run of the folder's map (default mapping)")
    p.add_argument("--root", default=None, metavar="DIR", help="data root (default $UMIPROC_ROOT or ~/data/umi)")
    p.add_argument("--points", help=f"the map export to show (default RUN_DIR/{POINTS_NAME}; its _keyframes.csv and "
                                    "_meta.json next to it)")
    p.add_argument("--times", help="the run's times.txt (exact frame times; default: from the map's raw/ recording, "
                                   "else estimated from the CSV)")
    p.add_argument("--recording", help="the recording folder for the photos (default the map's raw/, else the "
                                       "report's source)")
    p.add_argument("--min-obs", type=int, default=MIN_OBS, metavar="N",
                   help=f"hide points seen by fewer than N keyframes (default {MIN_OBS})")
    p.add_argument("--near", default=str(NEAR), metavar="F|Dm",
                   help=f"hide points farther than F x the path's size from the path (default {NEAR:g}); '0.5m': a "
                        "distance in map units; 0: off")
    p.add_argument("--box", type=float, default=None, metavar="F",
                   help="hide points outside the path's bounding box grown by F x its size (default off)")
    p.add_argument("--show-outliers", action="store_true", help="do not hide the far outliers")
    p.add_argument("--no-images", action="store_true", help="no photos and no image colours (faster)")
    p.add_argument("--thumb-width", type=int, default=THUMB_WIDTH, metavar="PX",
                   help=f"width of the embedded keyframe photos (default {THUMB_WIDTH})")
    p.add_argument("--title", help="the page's title")
    p.add_argument("--out", help=f"the HTML to write (default <folder>/analysis/{HTML_NAME})")


def run(args) -> int:
    from umiproc.viz.target import find_recording, resolve, run_times

    try:
        near, near_m = parse_near(args.near)
        filters = Filters(min_obs=args.min_obs, near=near, near_m=near_m, box=args.box,
                          outliers=not args.show_outliers)
        t = resolve(args.target, args.run, root=args.root)
        times = args.times
        if times is None:
            csv_path = t.run_dir / "trajectory.csv"
            log_path = t.run_dir / "umiproc_slam.log"
            summary = read_log(log_path).summary if log_path.is_file() else None
            n = summary.frames_fed if summary is not None else count_rows(csv_path)
            times = run_times(t, n, read_t0_ns(csv_path), args.recording) if n else None
        recording = None
        if not args.no_images:
            try:
                recording = find_recording(t, args.recording)
            except PlotRunError as e:
                print(f"no photos: {e}")
        out = Path(args.out).expanduser() if args.out else t.analysis / page_name(t.run)
        out.parent.mkdir(parents=True, exist_ok=True)
        title = args.title or (f"{t.folder}: {t.run} run" if t.folder else None)
        out, lines = make_page(t.run_dir, points=args.points, out=out, title=title, times=times, filters=filters,
                               recording=recording, settings=t.map_dir / "orbslam3_settings.yaml",
                               mask=t.map_dir / "slam_mask.png", thumb_width=args.thumb_width)
    except (Slam3dError, PlotRunError, OSError, ValueError) as e:
        print(f"umiproc slam-3d: {e}", file=sys.stderr)
        return 2
    print("\n".join(lines))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    from umiproc.next_step import block, carried, command
    from umiproc.viz.target import report_of

    rep = report_of(t)
    folder = t.folder or (rep.get("folder") if rep.get("passed") else None)      # a passed map's slot, by path
    if folder:
        print(block("remove the table tag, record 2-3 test demos and run them against the map", "5",
                    command("umiproc", "run", folder, "<FIRST_TEST_DEMO>", "<LAST_TEST_DEMO>",
                            *carried(args, source=False))))
    else:
        task = rep.get("task") or "<TASK>"
        print(block("with the table tag still in place, record a new mapping clip and map it", "4.7",
                    command("umiproc", "map", task, "<MAP_EPISODE>", *carried(args, source=False))))
    return 0


if __name__ == "__main__":
    import argparse

    _p = argparse.ArgumentParser(prog="python -m umiproc.viz.slam_3d", description=__doc__.splitlines()[0])
    add_arguments(_p)
    sys.exit(run(_p.parse_args()))
