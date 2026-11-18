"""``umiproc slam-3d RUN_DIR [--points CSV] [--out HTML]``: a 3-D view of one ``umiproc_slam`` run as one HTML file.

Shows, in the trajectory CSV's world W (the camera frame of the saved map's first keyframe), turned so that up is up:

* the saved map's points (``map_points.csv``), coloured by height or by their number of observations (a menu);
* its keyframes (``map_points_keyframes.csv``) as small camera frustums (x right, y down, z = the optical axis);
* the run's camera path (``trajectory.csv``: every posed frame), coloured by time, broken where nothing was posed;
* a slider (and a play button) that moves a camera frustum along the path, with the frame, time and state.

The map files are written by the front-end's ``--save_points RUN_DIR/map_points.csv`` (``native/umi_slam``): next
to the CSV, ``<stem>_keyframes.csv`` and ``<stem>_meta.json`` (``up_W``: gravity up in W once the IMU is initialised).
Without ``up_W``, up is the first keyframe's -y (the camera's up). A saved atlas can be exported without re-running
SLAM: ``umiproc_slam ... --load_map MAP.osa --max_frames 0 --save_points OUT.csv``; its world is then the mapping
run's (``--points OUT.csv`` with the mapping run folder). With no map files the view shows the path only.

The page loads plotly.js from a CDN (it needs internet once to show); the data are embedded. Points farther than
``OUTLIER_FACTOR`` times the 2-98 % spread outside it are left out of the view (the count is printed and shown).
Default output: ``RUN_DIR/slam_3d.html``. Exit codes: 0 written; 2 the inputs cannot be read.
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
from umiproc.viz.slam_run import PlotRunError, count_rows, read_t0_ns, times_from_csv_column

__all__ = ["HTML_NAME", "POINTS_NAME", "MapExport", "Slam3dError", "read_map_export", "scene_rotation",
           "build_payload", "write_html", "add_arguments", "run"]

HTML_NAME = "slam_3d.html"
POINTS_NAME = "map_points.csv"
POINTS_HEADER = ("id", "x", "y", "z", "observations")
KEYFRAMES_HEADER = ("id", "frame_id", "timestamp", "x", "y", "z", "q_x", "q_y", "q_z", "q_w")
OUTLIER_FACTOR = 1.0
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
                        T_W_kf=_poses(kv[:, 3:6], kv[:, 6:10]), meta=meta)
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


def build_payload(traj, export: MapExport | None, *, title: str) -> dict:
    """The page's data: everything rotated so that up is +z (:func:`scene_rotation`)."""
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
    hidden = 0
    kf_T = np.zeros((0, 4, 4))
    kf_ids, kf_t = np.zeros(0, np.int64), np.zeros(0)
    if export is not None:
        keep = _robust_keep(export.points)
        hidden = int((~keep).sum())
        pts, obs = export.points[keep], export.observations[keep]
        kf_T, kf_ids, kf_t = export.T_W_kf, export.keyframe_ids, export.keyframe_t_s

    # frustum size: 4 % of the extent of the path and keyframes (at least 1 cm)
    centers = np.concatenate([path_xyz[np.isfinite(path_xyz).all(axis=1)], kf_T[:, :3, 3]])
    extent = float(np.max(np.ptp(centers, axis=0))) if len(centers) > 1 else 0.25
    size = max(0.01, 0.04 * extent)

    Rh = np.eye(4)
    Rh[:3, :3] = R
    pts_r = rot(pts)
    # the initial view: a cube around the path and keyframes (far points would make them tiny); "all" zooms out
    c_r = rot(centers)
    mid = (c_r.min(axis=0) + c_r.max(axis=0)) / 2 if len(c_r) else np.zeros(3)
    half = max(0.3, 1.5 * extent)
    near = [[round(float(m - half), 3), round(float(m + half), 3)] for m in mid]
    h_rng = [round(float(v), 3) for v in np.percentile(pts_r[:, 2], [2, 98])] if len(pts_r) else [0.0, 1.0]
    return {
        "near": near,
        "height_range": h_rng,
        "title": title,
        "up": "gravity (IMU initialised)" if up_W is not None else "the first keyframe's -y (no gravity)",
        "meta": export.meta if export is not None else None,
        "hidden_points": hidden,
        "frustum_size": size,
        "points": {"xyz": _r(pts_r), "height": _r(pts_r[:, 2] if len(pts_r) else np.zeros(0)),
                   "obs": [int(v) for v in obs]},
        "keyframes": {"ids": [int(v) for v in kf_ids], "t": _r(kf_t, 3), "xyz": _r(rot(kf_T[:, :3, 3])),
                      "lines": _r(rot(_frustums(kf_T, size)))},
        "path": {"xyz": _r(rot(path_xyz)), "t": _r(path_t, 3)},
        "frames": frames,
        "frame_T": [_r((Rh @ M)[:3, :].ravel(), 5) for M in T_posed],
    }


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<script src="__PLOTLY__"></script>
<style>
 body { margin: 0; font: 14px system-ui, sans-serif; background: #fff; color: #222; }
 #plot { width: 100vw; height: calc(100vh - 92px); }
 #bar { padding: 6px 16px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
 #slider { flex: 1; min-width: 200px; }
 #info { font-family: monospace; min-width: 22em; }
 #note { padding: 0 16px 6px; color: #666; font-size: 12px; }
</style></head><body>
<div id="plot"></div>
<div id="bar"><button id="play">play</button><input id="slider" type="range" min="0" max="0" value="0">
<span id="info"></span></div>
<div id="note"></div>
<script>
const D = __DATA__;
const X = (a, i) => a.map(p => p[i]);
const traces = [];
traces.push({type: "scatter3d", mode: "markers", name: "map points (" + D.points.xyz.length + ")",
  x: X(D.points.xyz, 0), y: X(D.points.xyz, 1), z: X(D.points.xyz, 2),
  marker: {size: 2, color: D.points.height, colorscale: "Viridis", cmin: D.height_range[0],
    cmax: D.height_range[1], colorbar: {title: "height (m)", len: 0.6}},
  text: D.points.obs.map(o => "observations " + o), hoverinfo: "text"});
traces.push({type: "scatter3d", mode: "lines", name: "keyframes (" + D.keyframes.ids.length + ")",
  x: X(D.keyframes.lines, 0), y: X(D.keyframes.lines, 1), z: X(D.keyframes.lines, 2),
  line: {color: "#1f5fbf", width: 2}, hoverinfo: "skip"});
traces.push({type: "scatter3d", mode: "markers", name: "keyframe centres", showlegend: false,
  x: X(D.keyframes.xyz, 0), y: X(D.keyframes.xyz, 1), z: X(D.keyframes.xyz, 2),
  marker: {size: 3, color: "#1f5fbf"},
  text: D.keyframes.ids.map((id, i) => "KF " + id + ", t " + D.keyframes.t[i] + " s"), hoverinfo: "text"});
traces.push({type: "scatter3d", mode: "lines", name: "camera path (colour = time)",
  x: X(D.path.xyz, 0), y: X(D.path.xyz, 1), z: X(D.path.xyz, 2),
  line: {color: D.path.t, colorscale: "Plasma", width: 4},
  text: D.path.t.map((t, k) => "frame " + k + ", " + t + " s"), hoverinfo: "text"});
traces.push({type: "scatter3d", mode: "lines", name: "current pose", x: [], y: [], z: [],
  line: {color: "#e0201b", width: 6}, hoverinfo: "skip"});
const CUR = traces.length - 1;
const layout = {title: {text: D.title, font: {size: 15}}, margin: {l: 0, r: 0, t: 36, b: 0},
  legend: {x: 0, y: 1}, scene: {aspectmode: "cube", xaxis: {title: "x (m)", range: D.near[0]},
  yaxis: {title: "y (m)", range: D.near[1]}, zaxis: {title: "up (m)", range: D.near[2]}},
  updatemenus: [{type: "buttons", direction: "right", x: 0.75, xanchor: "center", y: 1.0, yanchor: "top",
    buttons: [
      {label: "view: near the path", method: "relayout", args: [{"scene.aspectmode": "cube",
        "scene.xaxis.range": D.near[0], "scene.yaxis.range": D.near[1], "scene.zaxis.range": D.near[2]}]},
      {label: "view: all points", method: "relayout", args: [{"scene.aspectmode": "data",
        "scene.xaxis.autorange": true, "scene.yaxis.autorange": true, "scene.zaxis.autorange": true}]}]},
    {type: "buttons", direction: "right", x: 0.35, xanchor: "center", y: 1.0, yanchor: "top",
    buttons: [
      {label: "points: height", method: "restyle", args: [{"marker.color": [D.points.height],
        "marker.cmin": D.height_range[0], "marker.cmax": D.height_range[1],
        "marker.colorbar.title.text": "height (m)"}, [0]]},
      {label: "points: observations", method: "restyle", args: [{"marker.color": [D.points.obs],
        "marker.cmin": 2, "marker.cmax": Math.max(3, ...D.points.obs.slice(0, 1e5)),
        "marker.colorbar.title.text": "observations"}, [0]]}]}]};
Plotly.newPlot("plot", traces, layout, {responsive: true});

const s = D.frustum_size * 1.6;
const C = [[-0.67, -0.5, 1], [0.67, -0.5, 1], [0.67, 0.5, 1], [-0.67, 0.5, 1]].map(c => c.map(v => v * s));
function frustum(T) {   // T: 3x4 row-major, camera -> scene
  const ap = v => [0, 1, 2].map(r => T[4 * r] * v[0] + T[4 * r + 1] * v[1] + T[4 * r + 2] * v[2] + T[4 * r + 3]);
  const o = ap([0, 0, 0]), w = C.map(ap);
  return [o, w[0], o, w[1], o, w[2], o, w[3], w[0], w[1], w[2], w[3], w[0]];
}
const slider = document.getElementById("slider"), info = document.getElementById("info");
slider.max = Math.max(0, D.frames.length - 1);
function show(i) {
  if (!D.frames.length) { info.textContent = "no posed frame in this run"; return; }
  const f = D.frames[i], L = frustum(D.frame_T[i]);
  Plotly.restyle("plot", {x: [L.map(p => p[0])], y: [L.map(p => p[1])], z: [L.map(p => p[2])]}, [CUR]);
  info.textContent = "frame " + f.k + "  t " + f.t.toFixed(2) + " s  " + f.state + (f.tracked ? "" : " (not tracked)");
}
slider.addEventListener("input", () => show(+slider.value));
let timer = null;
document.getElementById("play").addEventListener("click", e => {
  if (timer) { clearInterval(timer); timer = null; e.target.textContent = "play"; return; }
  e.target.textContent = "pause";
  timer = setInterval(() => { slider.value = (+slider.value + 3) % (D.frames.length || 1); show(+slider.value); }, 100);
});
show(0);
const m = D.meta;
document.getElementById("note").textContent = "W = the camera frame of the saved map's first keyframe, turned so " +
  "that up is " + D.up + ". " + (m ? "Map " + m.map_id + ": " + m.map_points + " points, " + m.keyframes +
  " keyframes" + (D.hidden_points ? ", " + D.hidden_points + " far outliers not shown" : "") + ". "
  : "No map points (run umiproc_slam with --save_points). ") +
  "Path: " + D.frames.length + " posed frames. Drag to rotate, scroll to zoom, click legend entries to hide.";
</script></body></html>
"""


def write_html(path: str | Path, payload: dict) -> Path:
    out = Path(path)
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    out.write_text(PAGE.replace("__TITLE__", html.escape(payload["title"])).replace("__PLOTLY__", PLOTLY_URL)
                   .replace("__DATA__", data))
    return out


def _read_trajectory(csv_path: Path, log_path: Path | None, times_txt: Path | None = None):
    """The run's trajectory, or None when it fed no frame. Frame times: ``times_txt`` (exact), else estimated from
    the CSV's timestamp column (:func:`umiproc.viz.slam_run.times_from_csv_column`), frame 0 pinned to t0."""
    try:
        summary = read_log(log_path).summary if log_path is not None else None
    except ValueError as e:                  # SlamLogError
        raise Slam3dError(f"cannot read {log_path}: {e}") from e
    t0 = read_t0_ns(csv_path)
    n = summary.frames_fed if summary is not None else count_rows(csv_path)
    if n == 0:
        return None
    try:
        if times_txt is not None:
            times = read_times(times_txt)
        else:
            rate = summary.pacing.recording_rate_hz if summary is not None and summary.pacing is not None else 30.0
            times = times_from_csv_column(csv_path, t0, n, 1e9 / (rate or 30.0))
            times[0] = t0                    # the estimate is off by a few ms when frame 0 was not posed
        return read_fork_csv(csv_path, times, t0_ns=t0, frames_fed=n)
    except TrajectoryCsvError as e:
        raise Slam3dError(str(e)) from e


def make_page(run_dir: str | Path, *, points: str | Path | None = None, out: str | Path | None = None,
              title: str | None = None, times: str | Path | None = None) -> tuple[Path, list[str]]:
    """Write the page for ``run_dir``; returns (path, the lines to print)."""
    run_dir = Path(run_dir)
    csv_path = run_dir / "trajectory.csv"
    if not csv_path.is_file():
        raise Slam3dError(f"no trajectory.csv in {run_dir}")
    log_path = run_dir / "umiproc_slam.log"
    try:
        traj = _read_trajectory(csv_path, log_path if log_path.is_file() else None,
                                Path(times) if times is not None else None)
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
    title = title or f"SLAM run {run_dir.parent.name}/{run_dir.name}"
    payload = build_payload(traj, export, title=title)
    if export is not None:
        m = export.meta
        lines.append(f"map {m.get('map_id')}: {len(export.points)} points ({payload['hidden_points']} far outliers "
                     f"not shown), {len(export.keyframe_ids)} keyframes; up = {payload['up']}")
    lines.append(f"path: {len(payload['frames'])} posed frames of {traj.n_frames if traj is not None else 0}")
    return write_html(Path(out) if out is not None else run_dir / HTML_NAME, payload), lines


def add_arguments(p) -> None:
    p.add_argument("run_dir", help="the run folder (trajectory.csv + .t0_ns, umiproc_slam.log, map_points.csv)")
    p.add_argument("--points", help=f"the map export to show (default RUN_DIR/{POINTS_NAME}; its _keyframes.csv and "
                                    "_meta.json next to it)")
    p.add_argument("--times", help="the run's times.txt (exact frame times; default: estimated from the CSV)")
    p.add_argument("--title", help="the page's title")
    p.add_argument("--out", help=f"the HTML to write (default RUN_DIR/{HTML_NAME})")


def run(args) -> int:
    try:
        out, lines = make_page(args.run_dir, points=args.points, out=args.out, title=args.title, times=args.times)
    except (Slam3dError, PlotRunError, OSError) as e:
        print(f"umiproc slam-3d: {e}", file=sys.stderr)
        return 2
    print("\n".join(lines))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    import argparse

    _p = argparse.ArgumentParser(prog="python -m umiproc.viz.slam_3d", description=__doc__.splitlines()[0])
    add_arguments(_p)
    sys.exit(run(_p.parse_args()))
