"""W1 / S6: shared pieces for the real-arm trials of 2026-11-20 (``real_trials_*.py``).

Read-only inputs: the trial folders ``~/Capstone/data/<TASK>/policy/<run>/deploy/2026-11-20_*_probe_live/`` (NOTES.md,
the probe's 224x224 PNG sets, written ~10 Hz by ``python -m policy_deploy.frames.probe --live 10 --save``) and the
real sessions ``~/policy_deploy_sessions/2026-11-20_HH-MM-SS/``. Nothing here writes outside ``OUT``.

Not run on its own; imported by ``real_trials_inventory.py``, ``real_trials_images.py`` and ``real_trials_replay.py``.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

HOME = Path.home()
REPO = HOME / "Capstone-main-sim"
DATA = HOME / "Capstone/data"
SESSIONS = HOME / "policy_deploy_sessions"
OUT = REPO / "sim_evals/_pipeline/W1/real_trials"
SCRATCH = Path("/tmp/claude-1000/-home-ruben-Capstone/20deb2cf-0c82-44ce-870d-39667f0b1130/scratchpad/w1/s6")
B1_REF = REPO / "sim_evals/_camera_study/B/reference"
DAY = "2026-11-20"


def copy_version() -> dict:
    """Which deployment copy ran (BRIEF_common: diff stat + whether policy/gripper_map.py exists)."""
    dep = REPO / "deployment"
    try:
        stat = subprocess.run(["git", "-C", str(REPO), "diff", "--stat", "--", "deployment"], capture_output=True,
                              text=True, timeout=30).stdout.strip().splitlines()
        stat = stat[-1] if stat else ""
        st = subprocess.run(["git", "-C", str(REPO), "status", "--short", "--", "deployment"], capture_output=True,
                            text=True, timeout=30).stdout.strip().splitlines()
    except Exception as exc:  # noqa: BLE001
        stat, st = f"git failed: {exc}", []
    return {"git_diff_stat_tail": stat, "git_status_deployment": st[:3],
            "gripper_map_py": (dep / "policy_deploy/policy_deploy/policy/gripper_map.py").exists()}


def trials() -> list[dict]:
    """Every recorded trial folder: task, run, trial number, folder, session named in NOTES.md, probe set count."""
    out = []
    for d in sorted(DATA.glob(f"*/policy/*/deploy/{DAY}_*_trial*_probe_live")):
        notes = (d / "NOTES.md").read_text()
        m = re.search(r"policy_deploy_sessions/(\S+)", notes)
        session = m.group(1).rstrip(")").rstrip(".") if m else None
        n = int(re.search(r"_trial(\d+)_", d.name).group(1))
        sets = sorted(int(p.name[:6]) for p in d.glob("*_main_0.png"))
        out.append({"task": d.parts[-5], "run": d.parts[-3], "trial": n, "folder": str(d), "session": session,
                    "n_sets": len(sets), "label": f"{d.parts[-5]}/{d.parts[-3][:7]}/trial{n}"})
    out.sort(key=lambda t: t["folder"].split("/deploy/")[1])
    return out


def set_paths(folder: str | Path, i: int) -> dict:
    d = Path(folder)
    return {k: d / f"{i:06d}_{k}.png" for k in ("main_0", "main_1", "ultrawide_0", "ultrawide_1")}


def set_times(folder: str | Path) -> np.ndarray:
    """Wall time (s since epoch, laptop) of every probe set = the mtime of its last PNG (ultrawide_1)."""
    d = Path(folder)
    idx = sorted(int(p.name[:6]) for p in d.glob("*_main_0.png"))
    assert idx == list(range(len(idx))), f"gap in the probe sets of {d}"
    return np.array([os.stat(d / f"{i:06d}_ultrawide_1.png").st_mtime for i in idx])


def wall(s: str) -> float:
    """'2026-11-20T17:42:19.437' (laptop local wall clock) -> epoch seconds."""
    return datetime.fromisoformat(s).timestamp()


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def session(name: str) -> dict:
    s = SESSIONS / name
    st = yaml.safe_load((s / "station.yaml").read_text())
    startup = json.loads((s / "policy/startup.json").read_text()) if (s / "policy/startup.json").exists() else None
    sess = yaml.safe_load((s / "manager/session.yaml").read_text()) if (s / "manager/session.yaml").exists() else None
    return {"name": name, "dir": s, "station": st, "startup": startup, "session": sess,
            "steps": load_jsonl(s / "manager/steps.jsonl"), "episodes": load_jsonl(s / "manager/episodes.jsonl"),
            "events": load_jsonl(s / "manager/events.jsonl")}


CLAMP_RE = re.compile(r"^(\S+ \S+) .*chunk (\d+): clamped into the workspace box: (.*)$")
WP_RE = re.compile(r"waypoint (\d+) ([xyz]) ([+-]?[\d.]+) -> ([+-]?[\d.]+)")


def clamps(name: str) -> list[dict]:
    """manager.log clamp lines -> [{wall, chunk_id, waypoints: [(k, axis, value, clamped)]}]."""
    out = []
    p = SESSIONS / name / "logs/manager.log"
    for line in p.read_text().splitlines() if p.exists() else []:
        m = CLAMP_RE.match(line)
        if not m:
            continue
        t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f").timestamp()
        wps = [(int(k), ax, float(v), float(c)) for k, ax, v, c in WP_RE.findall(m.group(3))]
        out.append({"wall": t, "chunk_id": int(m.group(2)), "waypoints": wps})
    return out


def pose_summary(T) -> dict:
    """TCP xyz in the base, tilt of the TCP z (toward the fingertips) from straight down, headings (deg)."""
    T = np.asarray(T, dtype=np.float64)
    z = T[:3, 2]
    x = T[:3, 0]
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, -z[2]))))
    return {"xyz_m": T[:3, 3].round(4).tolist(), "tilt_deg": round(tilt, 1),
            "heading_tcp_z_deg": round(math.degrees(math.atan2(z[1], z[0])), 1),
            "heading_tcp_x_deg": round(math.degrees(math.atan2(x[1], x[0])), 1),
            "tcp_z_axis": z.round(4).tolist(), "tcp_x_axis": x.round(4).tolist()}


def pct(v, q) -> float | None:
    v = [x for x in v if x is not None]
    return round(float(np.percentile(v, q)), 2) if v else None


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    return x


def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(jsonable(obj), indent=1))


def episode_for_trial(t: dict, sess: dict, times: np.ndarray) -> tuple[int, dict]:
    """The session episode whose [wall_start, wall_end] overlaps the trial's probe sets most."""
    best, best_n = None, -1
    for i, e in enumerate(sess["episodes"]):
        a, b = wall(e["wall_start"]), wall(e["wall_end"])
        n = int(((times >= a) & (times <= b)).sum())
        if n > best_n:
            best, best_n = i, n
    return best, sess["episodes"][best]


def episode_t0_ns(sess: dict, episode_id: int) -> int | None:
    """Laptop-clock t_ns of the 'episode_start' event of episode_id (steps.jsonl t_anchor_ns uses the same clock)."""
    for ev in sess["events"]:
        if ev["kind"] == "episode_start" and ev["detail"] == f"episode {episode_id}":
            return int(ev["t_ns"])
    return None


# ------------------------------------------------------------------ image measurements (cv2 imported on use)

CONTENT = slice(28, 196)     # the 224 images are letterboxed 4:3 -> rows 28..195 carry the picture (checked)


class TagDetector:
    """The same detector as sim_deploy/sim_pi/handheld_check.py TagDetector (DICT_4X4_50, x4 INTER_CUBIC upscale,
    corners mapped back to 224 px pixel centres); re-written here because that module imports mujoco/EGL."""

    def __init__(self, scale: int = 4):
        import cv2
        self.cv2 = cv2
        self.s = scale
        self.det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                           cv2.aruco.DetectorParameters())

    def __call__(self, rgb: np.ndarray) -> dict[int, np.ndarray]:
        cv2 = self.cv2
        big = cv2.resize(rgb, None, fx=self.s, fy=self.s, interpolation=cv2.INTER_CUBIC)
        c, ids, _ = self.det.detectMarkers(cv2.cvtColor(big, cv2.COLOR_RGB2GRAY))
        if ids is None:
            return {}
        return {int(i): (cc.reshape(4, 2) + 0.5) / self.s - 0.5 for i, cc in zip(ids.ravel(), c)}


def tag_summary(corners: np.ndarray) -> dict:
    c = np.asarray(corners, dtype=np.float64)
    side = float(np.mean([np.linalg.norm(c[a] - c[(a + 1) % 4]) for a in range(4)]))
    return {"centre": c.mean(0).round(2).tolist(), "side_px": round(side, 2), "corners": c.round(2).tolist()}


def tags_of(det: TagDetector, rgb: np.ndarray) -> dict:
    return {str(k): tag_summary(v) for k, v in sorted(det(rgb).items()) if k in (0, 1)}


def eraser(rgb: np.ndarray) -> dict | None:
    """The yellow / lime eraser, largest blob in the picture rows of: G >= R, G - B > 60, G > 110 (training's pale lime
    eraser ~(163, 219, 146), the real one ~(195, 211, 27); rejects the wood table (G < R) and the white/pink real
    table). Hue thresholds were tried first and took the warm-tinted wood in the IMX219 images."""
    import cv2
    a = np.asarray(rgb).astype(np.int32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    m = ((g >= r) & (g - b > 60) & (g > 110)).astype(np.uint8)
    m[:CONTENT.start] = 0
    m[CONTENT.stop:] = 0
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    j = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    area = int(st[j, cv2.CC_STAT_AREA])
    if area < 60:      # run_022 trial 4: a dark object, a few yellowish pixels elsewhere
        return None
    sel = lab == j
    return {"centroid": cen[j].round(2).tolist(), "area_px": area,
            "bbox_xywh": [int(st[j, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH,
                                                  cv2.CC_STAT_HEIGHT)],
            "mean_rgb": rgb[sel].mean(0).round(1).tolist()}


def appearance(rgb: np.ndarray) -> dict:
    """Mean colour per channel, brightness (Rec.601 luma mean), contrast (luma std), saturation, over the picture."""
    import cv2
    a = np.ascontiguousarray(rgb[CONTENT]).astype(np.float64)
    y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb[CONTENT]), cv2.COLOR_RGB2HSV)
    return {"mean_rgb": a.reshape(-1, 3).mean(0).round(1).tolist(), "luma_mean": round(float(y.mean()), 1),
            "luma_std": round(float(y.std()), 1), "sat_mean": round(float(hsv[..., 1].mean()), 1)}
