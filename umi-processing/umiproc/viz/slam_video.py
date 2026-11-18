"""``umiproc slam-video RUN_DIR [--recording DIR] [--out MP4] [--speed X]``: a video of one ``umiproc_slam`` run.

``RUN_DIR`` is a run folder of a map (``mapping/``, ``relocalize/``, ``mapping_full/`` in ``map/<pi_name>/`` or
``maps_failed/<pi_name>/<UTC>/``) or any folder with the run's ``trajectory.csv`` (+ ``.t0_ns``) and
``umiproc_slam.log``. Every fed frame becomes one video frame:

* left: the ultrawide frame fed to SLAM, the gripper mask tinted red (``slam_mask.png`` next to the run folder), in a
  border of the tracking colour (:mod:`umiproc.viz.slam_run`'s: green tracked in the saved map, amber tracked in a
  later map, red not tracking), with frame number, time, state, maps started so far and keyframes;
* right: the saved map's camera path seen from above (x right, z forward; W = the camera frame of its first keyframe),
  drawn as far as the current frame, the whole path in grey behind it;
* bottom: the whole run's timeline with a cursor; white ticks where SLAM started a new map, and for a
  ``mapping_full/`` run of ``map --first-map`` a magenta tick where the clip was cut.

The recording: ``--recording`` (the Pi recording folder: ``meta.json``, the IMX219's MP4 and its ``<stem>_frames.csv``),
else the ``source`` in the map's ``report.json`` (the folder in ``/data/incoming`` the map was made from). Written
with OpenCV's ``mp4v`` codec (VLC and most players; a browser may not play it). Default output:
``RUN_DIR/slam_run.mp4``.

Exit codes: 0 written; 2 the inputs cannot be read or do not fit together.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from umiproc.viz.slam_run import (CAT_COLORS, CAT_LATER_MAP, CAT_NOT_TRACKING, CAT_SAVED_MAP, PlotRunError, classify,
                                  read_t0_ns, times_from_frames_csv)

__all__ = ["VIDEO_NAME", "find_recording", "ultrawide_video", "render", "add_arguments", "run"]

VIDEO_NAME = "slam_run.mp4"
W_IMG, H_IMG, W_MAP, H_TL, H_HDR = 800, 600, 600, 60, 40
CUT_COLOR = (255, 0, 255)


def _bgr(hexcol: str) -> tuple[int, int, int]:
    h = hexcol.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


COL = {c: _bgr(v) for c, v in CAT_COLORS.items()}


def _report(run_dir: Path) -> dict:
    p = run_dir.parent / "report.json"
    try:
        rep = json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
    return rep if isinstance(rep, dict) else {}


def find_recording(run_dir: Path, recording: str | Path | None) -> Path:
    """``recording``, else the map report's ``source``."""
    if recording is not None:
        return Path(recording).expanduser()
    src = _report(run_dir).get("source")
    if not src:
        raise PlotRunError(f"no report.json with a source next to {run_dir}: pass --recording (the Pi recording "
                           "folder)")
    if not Path(src).is_dir():
        raise PlotRunError(f"the map's recording {src} is not there any more: pass --recording")
    return Path(src)


def ultrawide_video(recording: Path) -> tuple[Path, Path]:
    """``(mp4, frames csv)`` of the IMX219 (the ultrawide) in a Pi recording, from its ``meta.json``."""
    from umiproc.dp.pi_adapter import PiEpisodeError, _camera_roles

    try:
        meta = json.loads((recording / "meta.json").read_text())
        cam = _camera_roles(meta)["ultrawide"]
    except (OSError, ValueError, KeyError, PiEpisodeError) as e:
        raise PlotRunError(f"cannot find the ultrawide camera in {recording / 'meta.json'}: {e}") from e
    video = recording / cam["file"]
    frames = recording / f"{video.stem}_frames.csv"
    for p in (video, frames):
        if not p.is_file():
            raise PlotRunError(f"{p} is missing")
    return video, frames


def _cut_frame(run_dir: Path) -> int | None:
    """For ``mapping_full/`` of a ``--first-map`` map: the last fed frame kept."""
    from umiproc.map.build import MAPPING_FULL_DIR

    fm = _report(run_dir).get("first_map") or {}
    if run_dir.name == MAPPING_FULL_DIR and fm.get("applied"):
        return int(fm["frames"]) - 1
    return None


def render(run_dir: Path, recording: Path, out: Path, *, title: str | None = None, speed: float = 1.0,
           mask: Path | None = None, log=print) -> dict:
    """Write the video (module docstring). Returns the counts it shows."""
    import av
    import cv2

    from umiproc.slam.log import STATE_NAMES, read_log
    from umiproc.slam.trajectory_csv import TrajectoryCsvError, read_fork_csv

    csv_path, log_path = run_dir / "trajectory.csv", run_dir / "umiproc_slam.log"
    for p in (csv_path, log_path):
        if not p.is_file():
            raise PlotRunError(f"{p} is missing (is {run_dir} a SLAM run folder?)")
    summary = read_log(log_path).summary
    if summary is None:
        raise PlotRunError(f"{log_path} has no UMIPROC_SUMMARY line (the run did not finish)")
    t0 = read_t0_ns(csv_path)
    n = summary.frames_fed
    video, frames_csv = ultrawide_video(recording)
    times, i0 = times_from_frames_csv(frames_csv, t0, n)
    try:
        traj = read_fork_csv(csv_path, times, t0_ns=t0, frames_fed=n)
    except TrajectoryCsvError as e:
        raise PlotRunError(str(e)) from e
    cat = classify(traj)
    t_s = (traj.t_ns - traj.t_ns[0]) / 1e9
    new_map = np.zeros(n, bool)                       # a frame back in a no-map state after one with a map
    new_map[1:] = (traj.state[1:] <= 1) & (traj.state[:-1] > 1)
    maps_so_far = 1 + np.cumsum(new_map)
    ok = traj.tracked
    kf_so_far = np.cumsum(traj.is_keyframe & ok)
    cut = _cut_frame(run_dir)

    pos = traj.T_W_cam[:, :3, 3]
    if ok.any():
        xz = pos[ok][:, [0, 2]]
        lo, hi = xz.min(axis=0), xz.max(axis=0)
        span = max(float((hi - lo).max()), 1e-3) * 1.15
        mid = (lo + hi) / 2
    else:
        span, mid = 1.0, np.zeros(2)

    def to_px(k):
        u = (pos[k, 0] - mid[0]) / span * (W_MAP - 40) + W_MAP / 2
        v = H_IMG / 2 - (pos[k, 2] - mid[1]) / span * (H_IMG - 80)
        return int(round(u)), int(round(v))

    m = None
    if mask is not None and mask.is_file():
        img = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            m = cv2.resize(img, (W_IMG, H_IMG), interpolation=cv2.INTER_NEAREST) > 0

    W = W_IMG + W_MAP
    font = cv2.FONT_HERSHEY_SIMPLEX
    tl = np.full((H_TL, W, 3), 30, np.uint8)          # the timeline, drawn once; the cursor per frame
    xk = (np.arange(n) * (W - 20) / max(n - 1, 1) + 10).astype(int)
    for k in range(n):
        cv2.line(tl, (int(xk[k]), 10), (int(xk[k]), 34), COL[int(cat[k])], 2)
    for k in np.flatnonzero(new_map):
        cv2.line(tl, (int(xk[k]), 6), (int(xk[k]), 38), (255, 255, 255), 1)
    legend = ("green: tracked in the saved map   amber: tracked in a later map   red: not tracking   "
              "white: a new map starts")
    if cut is not None:
        cv2.line(tl, (int(xk[cut]), 2), (int(xk[cut]), 42), CUT_COLOR, 3)
        legend += "   magenta: --first-map cut"
    cv2.putText(tl, legend, (10, 54), font, 0.42, (200, 200, 200), 1)

    bg = np.full((H_IMG, W_MAP, 3), 20, np.uint8)     # the whole saved-map path in grey
    prev = None
    for k in range(n):
        p = to_px(k) if ok[k] else None
        if p is not None and prev is not None:
            cv2.line(bg, prev, p, (70, 70, 70), 1, cv2.LINE_AA)
        prev = p
    cv2.putText(bg, f"saved map from above (x right, z forward), {span:.2f} m across", (10, 22), font, 0.5,
                (200, 200, 200), 1)
    hdr = np.zeros((H_HDR, W, 3), np.uint8)
    cv2.putText(hdr, title or f"{run_dir.parent.name}/{run_dir.name}", (10, 28), font, 0.75, (255, 255, 255), 2)

    fps = 30.0 * speed
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H_HDR + H_IMG + H_TL))
    if not writer.isOpened():
        raise PlotRunError(f"cannot write {out} (OpenCV has no mp4v encoder?)")
    path_img, prev, written = bg.copy(), None, 0
    try:
        with av.open(str(video)) as c:
            stream = c.streams.video[0]
            stream.thread_type = "AUTO"
            for i, fr in enumerate(c.decode(stream)):
                k = i - i0
                if k < 0:
                    continue
                if k >= n:
                    break
                img = cv2.resize(fr.to_ndarray(format="bgr24"), (W_IMG, H_IMG), interpolation=cv2.INTER_AREA)
                if m is not None:
                    img[m] = (0.5 * img[m] + np.array([0, 0, 80])).astype(np.uint8)
                c_k = int(cat[k])
                cv2.rectangle(img, (0, 0), (W_IMG - 1, H_IMG - 1), COL[c_k], 10)
                if c_k == CAT_SAVED_MAP:
                    state = "TRACKED (saved map)"
                elif c_k == CAT_LATER_MAP:
                    state = "TRACKED (a later map)"
                else:
                    state = f"NOT TRACKING ({STATE_NAMES.get(int(traj.state[k]), int(traj.state[k]))})"
                cv2.rectangle(img, (10, 10), (540, 92), (0, 0, 0), -1)
                cv2.putText(img, state, (20, 40), font, 0.8, COL[c_k], 2)
                cv2.putText(img, f"fed frame {k} (video {i})  t = {t_s[k]:5.1f} s", (20, 64), font, 0.55,
                            (230, 230, 230), 1)
                cv2.putText(img, f"maps started so far: {int(maps_so_far[k])}   keyframes (saved map): "
                                 f"{int(kf_so_far[k])}", (20, 84), font, 0.5, (230, 230, 230), 1)
                if traj.is_keyframe[k] and ok[k]:
                    cv2.putText(img, "KEYFRAME", (W_IMG - 150, 40), font, 0.8, (255, 255, 0), 2)
                if cut is not None and k > cut:
                    cv2.putText(img, "after the --first-map cut", (20, H_IMG - 20), font, 0.7, CUT_COLOR, 2)

                if ok[k]:
                    p = to_px(k)
                    if prev is not None:
                        cv2.line(path_img, prev, p, COL[CAT_SAVED_MAP], 2, cv2.LINE_AA)
                    if traj.is_keyframe[k]:
                        cv2.circle(path_img, p, 3, (255, 255, 0), 1, cv2.LINE_AA)
                    prev = p
                else:
                    prev = None
                view = path_img.copy()
                if ok[k]:
                    cv2.circle(view, to_px(k), 7, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(view, "no pose in the saved map", (10, H_IMG - 16), font, 0.6,
                                COL[CAT_NOT_TRACKING if c_k == CAT_NOT_TRACKING else CAT_LATER_MAP], 2)
                t = tl.copy()
                cv2.line(t, (int(xk[k]), 2), (int(xk[k]), 42), (255, 255, 255), 3)
                writer.write(np.vstack([hdr, np.hstack([img, view]), t]))
                written += 1
    except av.error.FFmpegError as e:
        raise PlotRunError(f"cannot decode {video}: {e}") from e
    finally:
        writer.release()
    if written != n:
        raise PlotRunError(f"{video} ended after {written} of the run's {n} frames (from video frame {i0})")
    return {"frames": n, "fps": fps, "saved_map": int(ok.sum()), "later_map": int((cat == CAT_LATER_MAP).sum()),
            "not_tracking": int((cat == CAT_NOT_TRACKING).sum()), "new_maps": int(new_map.sum()), "cut": cut}


def add_arguments(p) -> None:
    p.add_argument("run_dir", metavar="RUN_DIR", help="a SLAM run folder: mapping/, relocalize/ or mapping_full/ of a "
                                                      "map (trajectory.csv + umiproc_slam.log)")
    p.add_argument("--recording", metavar="DIR", help="the Pi recording folder the run was made from (default: the "
                                                      "map report's source)")
    p.add_argument("--out", help=f"the MP4 to write (default RUN_DIR/{VIDEO_NAME})")
    p.add_argument("--title", help="the title line (default: the run folder)")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed (default 1.0: real time at 30 fps)")


def run(args) -> int:
    run_dir = Path(args.run_dir).expanduser()
    if not run_dir.is_dir():
        print(f"umiproc slam-video: {run_dir} is not a folder", file=sys.stderr)
        return 2
    if not args.speed > 0:
        print("umiproc slam-video: --speed must be > 0", file=sys.stderr)
        return 2
    out = Path(args.out).expanduser() if args.out else run_dir / VIDEO_NAME
    try:
        recording = find_recording(run_dir, args.recording)
        print(f"slam-video: {run_dir} with the ultrawide video of {recording} ...", flush=True)
        st = render(run_dir, recording, out, title=args.title, speed=args.speed, mask=run_dir.parent / "slam_mask.png")
    except (PlotRunError, OSError) as e:
        print(f"umiproc slam-video: {e}", file=sys.stderr)
        return 2
    print(f"slam-video: {st['frames']} frames at {st['fps']:g} fps: tracked in the saved map {st['saved_map']}, in a "
          f"later map {st['later_map']}, not tracking {st['not_tracking']}; new maps started {st['new_maps']}"
          + (f"; --first-map cut after fed frame {st['cut']}" if st["cut"] is not None else ""))
    print(f"wrote {out}")
    return 0
