"""``umiproc plot-run``: one figure of one ``umiproc_slam`` run, from its trajectory CSV (and log).

* A timeline of the fed frames: **tracked in the saved map** (state OK and not ``is_lost``; the fork's CSV holds the
  poses of one map only, the biggest by keyframes), **tracked in a later map** (state OK with ``is_lost``: tracked in
  another map, or in a map reset later; no pose in the CSV), or **not tracking**; thin black lines at the keyframes.
* Three projections of the camera positions tracked in the saved map (W = the camera frame of its first keyframe:
  x right, y down, z forward), coloured by time, keyframes circled, first and last tracked frame marked.
* A row of ultrawide video frames (``--video``, ``--frames``) titled with frame number, time and tracking state.
* A text block with the numbers (:func:`compute_stats`, :func:`stats_lines`).

Frame times: the CSV is read with :func:`umiproc.slam.read_fork_csv`, which needs the fed frames' times. With
``--video``, from the Pi's ``<stem>_frames.csv`` next to the MP4, aligned on the CSV's ``.t0_ns`` (fed frame k is video
frame i0 + k, :func:`times_from_frames_csv`); without it, from the CSV's own timestamp column on the posed rows,
interpolated in between (:func:`times_from_csv_column`).

matplotlib is optional (``pip install -e umi-processing[plot]``) and only imported to draw.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from umiproc.slam.log import STATE_NAMES, STATE_OK, MapInfo, SlamSummary, read_log
from umiproc.slam.trajectory_csv import ForkTrajectory, TrajectoryCsvError, read_fork_csv

PLOT_HINT = "matplotlib is missing: pip install -e umi-processing[plot]"
FRAME_THUMB_WIDTH = 320          # px of each video frame in the figure (keeps the PNG small)
DEFAULT_PERIOD_NS = 1e9 / 30

#: Per-frame categories (:func:`classify`) and their colours (good / warning / critical)
CAT_NOT_TRACKING, CAT_LATER_MAP, CAT_SAVED_MAP = 0, 1, 2
CAT_COLORS = {CAT_SAVED_MAP: "#0ca30c", CAT_LATER_MAP: "#fab219", CAT_NOT_TRACKING: "#d03b3b"}


class PlotRunError(Exception):
    """The inputs cannot be read or do not fit together (exit 2)."""


# --------------------------------------------------------------------------------------------- frame times

def read_t0_ns(csv_path: str | Path) -> int:
    p = Path(str(csv_path) + ".t0_ns")
    try:
        return int(p.read_text().split()[0])
    except (OSError, IndexError, ValueError) as e:
        raise PlotRunError(f"cannot read the time origin {p}: {e}") from e


def count_rows(csv_path: str | Path) -> int:
    with open(csv_path, newline="") as f:
        return max(0, sum(1 for r in csv.reader(f) if r) - 1)


def times_from_frames_csv(frames_csv: str | Path, t0_ns: int, n: int) -> tuple[np.ndarray, int]:
    """``(times of fed frames 0..n-1 in ns, i0)``: fed frame k is video frame ``i0 + k``.

    The fed frames are a contiguous run of the video's frames (those inside the IMU span) whose times are the Pi's
    stamps plus a constant (mid-exposure), so i0 is the frame stamped closest to ``t0_ns`` (within half a period).
    """
    try:
        with open(frames_csv, newline="") as f:
            t = np.array([int(r["t_ns"]) for r in csv.DictReader(f)], dtype=np.int64)
    except (OSError, KeyError, ValueError) as e:
        raise PlotRunError(f"cannot read the frame times {frames_csv}: {e}") from e
    if len(t) < 2:
        raise PlotRunError(f"{frames_csv}: fewer than 2 frames")
    i0 = int(np.argmin(np.abs(t - t0_ns)))
    if abs(int(t[i0]) - t0_ns) > float(np.median(np.diff(t))) / 2:
        raise PlotRunError(f"no frame of {frames_csv} is within half a frame period of the run's first frame time "
                           f"{t0_ns}: another recording?")
    if i0 + n > len(t):
        raise PlotRunError(f"the CSV has {n} frames from video frame {i0}, but {frames_csv} has {len(t)}")
    return t[i0:i0 + n] - t[i0] + t0_ns, i0


def times_from_csv_column(csv_path: str | Path, t0_ns: int, n: int, period_ns: float = DEFAULT_PERIOD_NS
                          ) -> np.ndarray:
    """Frame times estimated from the CSV's timestamp column: exact (1 us) on the rows with ``is_lost`` false and a
    new timestamp, interpolated by frame index in between, extrapolated at ``period_ns`` outside."""
    idx, ts, prev = [], [], None
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for r in reader:
            if len(r) < 4 or r[3].strip().lower() != "false" or r[1].strip() == prev:
                continue
            prev = r[1].strip()
            idx.append(int(r[0]))
            ts.append(float(prev) * 1e9)
    k = np.arange(n, dtype=np.float64)
    if not idx:
        return (t0_ns + np.rint(k * period_ns)).astype(np.int64)
    ia, ta = np.asarray(idx, float), np.asarray(ts, float)
    t = np.interp(k, ia, ta)
    t[k < ia[0]] = ta[0] - (ia[0] - k[k < ia[0]]) * period_ns
    t[k > ia[-1]] = ta[-1] + (k[k > ia[-1]] - ia[-1]) * period_ns
    return (t0_ns + np.rint(t)).astype(np.int64)


# --------------------------------------------------------------------------------------------- numbers

def classify(traj: ForkTrajectory) -> np.ndarray:
    """Per fed frame: saved map (``traj.tracked``), later map (state OK and ``is_lost``), else not tracking."""
    cat = np.full(traj.n_frames, CAT_NOT_TRACKING, np.int8)
    cat[(traj.state == STATE_OK) & traj.is_lost] = CAT_LATER_MAP
    cat[traj.tracked] = CAT_SAVED_MAP
    return cat


def stretches(cat: np.ndarray) -> list[tuple[int, int, int]]:
    """Maximal runs of equal category: ``(first frame, one past the last, category)``."""
    if len(cat) == 0:
        return []
    cuts = np.flatnonzero(np.diff(cat)) + 1
    starts, stops = np.concatenate([[0], cuts]), np.concatenate([cuts, [len(cat)]])
    return [(int(a), int(b), int(cat[a])) for a, b in zip(starts, stops)]


def saved_map(summary: SlamSummary | None) -> MapInfo | None:
    """The map whose poses the CSV holds: the first with the most keyframes (System::SaveTrajectoryCSV)."""
    best = None
    for m in (summary.maps or ()) if summary is not None else ():
        if m.keyframes > (best.keyframes if best else 0):
            best = m
    return best


@dataclass(frozen=True)
class RunStats:
    frames: int
    duration_s: float                            # first to last fed frame
    n_saved: int
    n_later: int
    n_not_tracking: int
    saved_span_s: tuple[float, float] | None     # first and last frame tracked in the saved map
    longest_s: tuple[int, int, float, float] | None   # longest saved-map stretch: first, last frame, their times
    keyframes_marked: int                        # tracked rows with is_keyframe
    extent_m: tuple[float, float, float] | None  # max - min of x, y, z over the tracked positions
    path_length_m: float                         # sum of the steps between consecutive tracked positions

    def pct(self, n: int) -> float:
        return 100.0 * n / self.frames if self.frames else 0.0


def compute_stats(traj: ForkTrajectory) -> RunStats:
    cat = classify(traj)
    t_s = (traj.t_ns - traj.t_ns[0]) / 1e9 if traj.n_frames else np.zeros(0)
    tr = np.flatnonzero(cat == CAT_SAVED_MAP)
    runs = [r for r in stretches(cat) if r[2] == CAT_SAVED_MAP]
    longest = None
    if runs:
        a, b, _ = max(runs, key=lambda r: r[1] - r[0])
        longest = (a, b - 1, float(t_s[a]), float(t_s[b - 1]))
    pos = traj.T_W_cam[tr, :3, 3]
    return RunStats(
        frames=traj.n_frames, duration_s=float(t_s[-1]) if len(t_s) else 0.0,
        n_saved=len(tr), n_later=int((cat == CAT_LATER_MAP).sum()), n_not_tracking=int((cat == CAT_NOT_TRACKING).sum()),
        saved_span_s=(float(t_s[tr[0]]), float(t_s[tr[-1]])) if len(tr) else None, longest_s=longest,
        keyframes_marked=int((traj.is_keyframe & (cat == CAT_SAVED_MAP)).sum()),
        extent_m=tuple(float(v) for v in pos.max(0) - pos.min(0)) if len(pos) else None,
        path_length_m=float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()) if len(pos) > 1 else 0.0)


def _yn(b: bool) -> str:
    return "yes" if b else "no"


def stats_lines(st: RunStats, summary: SlamSummary | None) -> list[str]:
    """The figure's text block."""
    m = saved_map(summary)
    name = f"map {m.id}" if m is not None else "the saved map"
    L = [f"Frames fed       {st.frames} ({st.duration_s:.1f} s)",
         f"Tracked in the saved map ({name}):",
         f"                 {st.n_saved} ({st.pct(st.n_saved):.1f}%)"
         + (f", {st.saved_span_s[0]:.1f}-{st.saved_span_s[1]:.1f} s" if st.saved_span_s else "")]
    if st.longest_s is not None:
        a, b, ta, tb = st.longest_s
        L.append(f"                 longest stretch {tb - ta:.1f} s")
        L.append(f"                 (frames {a}-{b})")
    L += ["Tracked in a later map:",
          f"                 {st.n_later} ({st.pct(st.n_later):.1f}%)",
          f"Not tracking     {st.n_not_tracking} ({st.pct(st.n_not_tracking):.1f}%)", ""]
    if summary is not None and summary.maps:
        L.append(f"Maps at the end  {len(summary.maps)}")
        L.append("   id   KFs  points  IMU init  VIBA1  VIBA2")
        for mi in summary.maps:
            L.append(f"  {mi.id:3d} {mi.keyframes:5d} {mi.map_points:7d}  {_yn(mi.imu_initialized):8s}  "
                     f"{_yn(mi.inertial_ba1):5s}  {_yn(mi.inertial_ba2):5s}")
        L.append("")
    L.append(f"Keyframes of {name}: " + (f"{m.keyframes} (summary), " if m is not None else ""))
    L.append(f"                 {st.keyframes_marked} marked in the CSV")
    if st.extent_m is not None:
        x, y, z = st.extent_m
        L += [f"Path of {name}:  extent x {x:.2f}, y {y:.2f}, z {z:.2f} m",
              f"                 length {st.path_length_m:.2f} m"]
    p = summary.pacing if summary is not None else None
    if p is not None:
        L += ["", f"Pacing           fed {p.feed_rate_hz:.1f} Hz of {p.recording_rate_hz:.1f} Hz",
              f"                 tracking median {p.track_ms_median:.0f} ms, p95 {p.track_ms_p95:.0f} ms"]
    return L


def frame_state(k: int, traj: ForkTrajectory, cat: np.ndarray, summary: SlamSummary | None) -> str:
    c = int(cat[k])
    if c == CAT_SAVED_MAP:
        m = saved_map(summary)
        return f"tracked in map {m.id}" if m is not None else "tracked in the saved map"
    if c == CAT_LATER_MAP:
        return "tracked in a later map"
    return f"not tracking ({STATE_NAMES.get(int(traj.state[k]), int(traj.state[k]))})"


# --------------------------------------------------------------------------------------------- drawing

def read_video_frames(video: str | Path, indices, width: int = FRAME_THUMB_WIDTH) -> dict[int, np.ndarray]:
    """``{index: rgb}`` of the wanted frames, scaled to ``width`` px."""
    import av
    import cv2

    want, out = set(int(i) for i in indices), {}
    try:
        with av.open(str(video)) as c:
            stream = c.streams.video[0]
            stream.thread_type = "AUTO"
            for i, frame in enumerate(c.decode(stream)):
                if i in want:
                    rgb = frame.to_ndarray(format="rgb24")
                    out[i] = cv2.resize(rgb, (width, round(rgb.shape[0] * width / rgb.shape[1])),
                                        interpolation=cv2.INTER_AREA)
                    if len(out) == len(want):
                        break
    except (av.error.FFmpegError, OSError) as e:
        raise PlotRunError(f"cannot decode {video}: {e}") from e
    if len(out) != len(want):
        raise PlotRunError(f"{video} ends before frame {min(want - set(out))}")
    return out


def draw(out: str | Path, traj: ForkTrajectory, lines: list[str], summary: SlamSummary | None, *, title: str,
         frames: list[tuple[int, np.ndarray, str]], dpi: int = 100) -> Path:
    """Write the figure. ``frames``: ``(fed frame, rgb, caption)``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    cat = classify(traj)
    t_s = (traj.t_ns - traj.t_ns[0]) / 1e9
    m = saved_map(summary)
    mname = f"map {m.id}" if m is not None else "the saved map"
    rows = 3 if frames else 2
    fig = plt.figure(figsize=(20, 11.6 if frames else 8.6))
    gs = fig.add_gridspec(rows, 4, height_ratios=[0.8, 3.3, 2.1][:rows], width_ratios=[1, 1, 1, 1.12], hspace=0.42,
                          wspace=0.28, left=0.035, right=0.99, top=0.9, bottom=0.03 if frames else 0.07)
    fig.suptitle(title, fontsize=14)

    # timeline
    ax = fig.add_subplot(gs[0, :3])
    period = float(np.median(np.diff(t_s))) if len(t_s) > 1 else DEFAULT_PERIOD_NS / 1e9
    edges = np.append(t_s, t_s[-1] + period)
    for a, b, c in stretches(cat):
        ax.broken_barh([(edges[a], edges[b] - edges[a])], (0, 1), color=CAT_COLORS[c], lw=0)
    ax.vlines(t_s[traj.is_keyframe & (cat == CAT_SAVED_MAP)], 0, 1, color="black", lw=0.5, alpha=0.6)
    last_x, lift = None, 0
    for k, _img, _cap in frames:
        ax.plot([t_s[k]], [1.0], marker="v", color="black", ms=6, clip_on=False)
        lift = 9 - lift if last_x is not None and t_s[k] - last_x < 0.03 * edges[-1] else 0    # close labels: stagger
        ax.annotate(str(k), (t_s[k], 1.0), xytext=(0, 7 + lift), textcoords="offset points", ha="center", fontsize=8)
        last_x = t_s[k]
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel(f"time in clip (s); thin black lines: keyframes of {mname}; triangles: the frames shown below",
                  fontsize=9)
    ax.legend(handles=[Patch(color=CAT_COLORS[CAT_SAVED_MAP], label=f"tracked in the saved map ({mname})"),
                       Patch(color=CAT_COLORS[CAT_LATER_MAP], label="tracked in a later map"),
                       Patch(color=CAT_COLORS[CAT_NOT_TRACKING], label="not tracking")],
              loc="lower center", bbox_to_anchor=(0.5, 1.26), ncol=3, frameon=False, fontsize=9)

    # projections
    tr = np.flatnonzero(cat == CAT_SAVED_MAP)
    P, tt, kf = traj.T_W_cam[tr, :3, 3], t_s[tr], traj.is_keyframe[tr]
    views = [(0, 2, "from above: x (right) vs z (forward)", "x (m)", "z (m)", False),
             (0, 1, "from the front: x (right) vs y (down)", "x (m)", "y (m), down", True),
             (2, 1, "from the side: z (forward) vs y (down)", "z (m)", "y (m), down", True)]
    for j, (a, b, name, xl, yl, inv) in enumerate(views):
        axp = fig.add_subplot(gs[1, j])
        axp.set_title(name, fontsize=10)
        axp.set_xlabel(xl, fontsize=9)
        axp.set_ylabel(yl, fontsize=9)
        axp.grid(alpha=0.3, lw=0.5)
        axp.tick_params(labelsize=8)
        if not len(P):
            axp.text(0.5, 0.5, "no frame tracked in the saved map", ha="center", va="center", transform=axp.transAxes)
            continue
        sc = axp.scatter(P[:, a], P[:, b], c=tt, cmap="viridis", s=4, lw=0, zorder=2)
        axp.scatter(P[kf, a], P[kf, b], s=26, facecolors="none", edgecolors="black", lw=0.7, zorder=3,
                    label="keyframes")
        axp.plot(P[0, a], P[0, b], marker="^", ms=10, color=CAT_COLORS[CAT_SAVED_MAP], mec="black", ls="", zorder=4,
                 label=f"first tracked ({tt[0]:.1f} s)")
        axp.plot(P[-1, a], P[-1, b], marker="s", ms=9, color=CAT_COLORS[CAT_NOT_TRACKING], mec="black", ls="",
                 zorder=4, label=f"last tracked ({tt[-1]:.1f} s)")
        axp.set_aspect("equal", adjustable="datalim")
        if inv:
            axp.invert_yaxis()
        if j == 0:
            axp.legend(fontsize=8, loc="best", framealpha=0.85)
        if j == 2:
            cb = fig.colorbar(sc, ax=axp, fraction=0.05, pad=0.02)
            cb.set_label("time (s)", fontsize=9)
            cb.ax.tick_params(labelsize=8)

    # numbers
    axt = fig.add_subplot(gs[0:2, 3])
    axt.axis("off")
    axt.text(0.0, 1.0, "\n".join(lines), family="monospace", fontsize=9.5, va="top", ha="left",
             transform=axt.transAxes)

    # video frames
    if frames:
        sub = gs[2, :].subgridspec(1, max(len(frames), 6), wspace=0.04)
        for j, (_k, img, cap) in enumerate(frames):
            axf = fig.add_subplot(sub[0, j])
            axf.axis("off")
            axf.imshow(img)
            axf.set_title(cap, fontsize=8.5)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------------------------- the command

def add_arguments(p) -> None:
    p.add_argument("--csv", required=True, help="the run's trajectory CSV (its .t0_ns next to it)")
    p.add_argument("--log", help="the run's log with the UMIPROC_SUMMARY line (maps, keyframes, pacing)")
    p.add_argument("--video", help="the recording's ultrawide MP4 (cam0.mp4; its cam0_frames.csv next to it)")
    p.add_argument("--frames", help="fed frame numbers to show under the plots, comma-separated (needs --video)")
    p.add_argument("--title", default="SLAM run", help="the figure's title")
    p.add_argument("--out", default="slam_run.png", help="the PNG to write (default ./slam_run.png)")


def make_figure(args, times: np.ndarray | None = None) -> tuple[Path, list[str]]:
    """Draw ``args.out``. ``times``: the fed frames' exact times in ns (``umiproc map`` has them from times.txt);
    without, from the video's frames CSV or the trajectory CSV's own column (module docstring)."""
    summary = read_log(args.log).summary if args.log else None
    t0 = read_t0_ns(args.csv)
    n = summary.frames_fed if summary is not None else count_rows(args.csv)
    if bool(args.video) != bool(args.frames):
        raise PlotRunError("--video and --frames go together")
    i0 = None
    if times is not None and args.video:
        raise PlotRunError("pass either times or --video")
    if times is not None:
        times = np.asarray(times, dtype=np.int64)[:n]
    elif args.video:
        video = Path(args.video)
        times, i0 = times_from_frames_csv(video.with_name(f"{video.stem}_frames.csv"), t0, n)
    else:
        rate = summary.pacing.recording_rate_hz if summary is not None and summary.pacing is not None else 30.0
        times = times_from_csv_column(args.csv, t0, n, 1e9 / rate)
    try:
        traj = read_fork_csv(args.csv, times, t0_ns=t0, frames_fed=n)
    except TrajectoryCsvError as e:
        raise PlotRunError(str(e)) from e
    lines = stats_lines(compute_stats(traj), summary)
    frames = []
    if args.frames:
        try:
            ks = sorted(set(int(v) for v in args.frames.split(",") if v.strip()))
        except ValueError:
            raise PlotRunError(f"--frames: not a list of frame numbers: {args.frames!r}") from None
        if any(not 0 <= k < traj.n_frames for k in ks):
            raise PlotRunError(f"--frames: outside the fed frames 0..{traj.n_frames - 1}")
        imgs = read_video_frames(args.video, [i0 + k for k in ks])
        cat = classify(traj)
        t_s = (traj.t_ns - traj.t_ns[0]) / 1e9
        frames = [(k, imgs[i0 + k], f"frame {k} (video {i0 + k}), {t_s[k]:.1f} s\n{frame_state(k, traj, cat, summary)}")
                  for k in ks]
    return draw(args.out, traj, lines, summary, title=args.title, frames=frames), lines


def run(args) -> int:
    try:
        import matplotlib  # noqa: F401
    except ModuleNotFoundError:
        print(f"umiproc plot-run: {PLOT_HINT}", file=sys.stderr)
        return 2
    try:
        out, lines = make_figure(args)
    except (PlotRunError, OSError) as e:
        print(f"umiproc plot-run: {e}", file=sys.stderr)
        return 2
    print("\n".join(lines))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    import argparse

    _p = argparse.ArgumentParser(prog="python -m umiproc.viz.slam_run", description=__doc__.splitlines()[0])
    add_arguments(_p)
    sys.exit(run(_p.parse_args()))
